import time

import numpy as np
from numpy.linalg import pinv
from scipy.spatial.transform import Rotation as R

from openteach.components.operators.operator import Operator
from openteach.constants import (
    ARM_TELEOP_STOP,
    FRANKA_CART_STEP_LIMITS,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    ROBOT_WORKSPACE,
    VR_FREQ,
    H_R_V_star,
    H_R_V,
)
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from openteach.robot.franka_stick import FrankaArm
from openteach.utils.timer import FrequencyTimer


def get_relative_affine(init_affine, current_affine):
    """Returns the relative affine from the initial affine to the current affine.
    Args:
        init_affine: Initial affine
        current_affine: Current affine
    Returns:
        Relative affine from init_affine to current_affine
    """
    # Relative affine from init_affine to current_affine in the VR controller frame.
    H_V_des = pinv(init_affine) @ current_affine

    # Transform to robot frame.
    # Flips axes
    relative_affine_rot = (pinv(H_R_V) @ H_V_des @ H_R_V)[:3, :3]
    # Translations flips are mirrored.
    relative_affine_trans = (pinv(H_R_V_star) @ H_V_des @ H_R_V_star)[:3, 3]

    # Homogeneous coordinates
    relative_affine = np.block(
        [[relative_affine_rot, relative_affine_trans.reshape(3, 1)], [0, 0, 0, 1]]
    )

    return relative_affine


class FrankaOperator(Operator):
    def __init__(
        self,
        host,
        controller_state_port,
        gripper_port=None,
        cartesian_publisher_port=None,
        joint_publisher_port=None,
        cartesian_command_publisher_port=None,
    ) -> None:
        self.notify_component_start("Franka stick operator")

        # Subscribe controller state
        self._controller_state_subscriber = ZMQKeypointSubscriber(
            host=host, port=controller_state_port, topic="controller_state"
        )

        # # Subscribers for the transformed hand keypoints
        self._transformed_arm_keypoint_subscriber = None
        self._transformed_hand_keypoint_subscriber = None

        self._robot = FrankaArm()
        self.robot.reset()

        # Gripper and cartesian publisher
        self.gripper_publisher = ZMQKeypointPublisher(host=host, port=gripper_port)

        self.cartesian_publisher = ZMQKeypointPublisher(
            host=host, port=cartesian_publisher_port
        )

        self.joint_publisher = ZMQKeypointPublisher(
            host=host, port=joint_publisher_port
        )

        self.cartesian_command_publisher = ZMQKeypointPublisher(
            host=host, port=cartesian_command_publisher_port
        )

        self.robot_init_H = self._robot.get_pose()
        self._timer = FrequencyTimer(VR_FREQ)

        # Class variables
        self.resolution_scale = 1
        self.arm_teleop_state = ARM_TELEOP_STOP
        self.is_first_frame = True
        self.prev_gripper_flag = 0
        self.prev_pause_flag = 0
        self.pause_cnt = 0
        self.gripper_correct_state = GRIPPER_OPEN
        self.gripper_flag = 1
        self.pause_flag = 1
        self.gripper_cnt = 0

        self.start_teleop = False
        self.init_affine = None

    def affine_to_robot_pose_aa(self, affine: np.ndarray) -> np.ndarray:
        """Converts an affine matrix to a robot pose in axis-angle format.
        Args:
            affine (np.ndarray): 4x4 affine matrix [[R, t],[0, 1]]
        Returns:
            list: [x, y, z, ax, ay, az] where (x, y, z) is the position and (ax, ay, az) is the axis-angle rotation.
            x, y, z are in mm and ax, ay, az are in radians.
        """
        translation = affine[:3, 3]  # TODO: check if we need SCALE_FACTOR
        rotation = R.from_matrix(affine[:3, :3]).as_rotvec()
        return np.concatenate([translation, rotation])

    def _apply_retargeted_angles(self) -> None:
        self.controller_state = self._controller_state_subscriber.recv_keypoints()

        if self.is_first_frame:
            self._robot.home()
            time.sleep(2)
            self.home_pose = self._robot._controller.robot.get_position_aa()
            self.home_affine = self._robot.get_pose()
            self.is_first_frame = False
        if self.controller_state.right_a:
            self.start_teleop = True
            self.init_affine = self.controller_state.right_affine
        if self.controller_state.right_b:
            self.start_teleop = False
            self.init_affine = None
            self.home_pose = self._robot._controller.robot.get_position_aa()
            self.home_affine = self._robot.get_pose()

        if self.start_teleop:
            relative_affine = get_relative_affine(
                self.init_affine, self.controller_state.right_affine
            )
        else:
            relative_affine = np.zeros((4, 4))
            relative_affine[3, 3] = 1

        gripper_state = None

        if self.controller_state.right_index_trigger > 0.5:
            gripper_state = GRIPPER_CLOSE
        elif self.controller_state.right_hand_trigger > 0.5:
            gripper_state = GRIPPER_OPEN
        if gripper_state is not None and gripper_state != self.gripper_correct_state:
            self._robot.set_gripper_state(gripper_state * 800)
            self.gripper_correct_state = gripper_state

        if self.start_teleop:
            home_translation = self.home_affine[:3, 3]
            home_rotation = self.home_affine[:3, :3]

            # Target
            target_translation = home_translation + relative_affine[:3, 3]
            target_rotation = home_rotation @ relative_affine[:3, :3]

            target_affine = np.block(
                [[target_rotation, target_translation.reshape(-1, 1)], [0, 0, 0, 1]]
            )

            target_pose = self.affine_to_robot_pose_aa(target_affine).tolist()
            current_pose = self._robot._controller.robot.get_position_aa()

            delta_translation = np.array(target_pose[:3] - np.array(current_pose[:3]))

            delta_translation = np.clip(
                delta_translation,
                a_min=FRANKA_CART_STEP_LIMITS[0],
                a_max=FRANKA_CART_STEP_LIMITS[1],
            )

            des_translation = delta_translation + np.array(current_pose[:3])
            des_translation = np.clip(
                des_translation, a_min=ROBOT_WORKSPACE[0], a_max=ROBOT_WORKSPACE[1]
            ).tolist()

            des_rotation = target_pose[3:]
            des_pose = des_translation + des_rotation
        else:
            des_pose = self.home_pose

        # Save the states here
        # TODO:

        if self.start_teleop:
            self.robot.arm_control(des_pose)

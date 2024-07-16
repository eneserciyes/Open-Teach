import time
from pathlib import Path
import os

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

from deoxys.utils import YamlConfig
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils

CONFIG_ROOT = Path(__file__).parent


class Robot(FrankaInterface):
    def __init__(self, cfg):
        super(Robot, self).__init__(
            general_cfg_file=os.path.join(CONFIG_ROOT, cfg),
            use_visualizer=False,
        )
        self.controller_cfg = YamlConfig(
            os.path.join(CONFIG_ROOT, "osc-pose-controller.yml")
        ).as_easydict()

    def arm_control(self, action):
        """
        Action: nd.array  -- [x, y, z, roll, yaw, pitch, gripper]
        """
        print(f"{action=}")
        self.control(
            controller_type="OSC_POSE",
            action=action,
            controller_cfg=self.controller_cfg,
        )

    def move_to_target_pose(
        self,
        controller_type,
        target_delta_pose,
        num_steps,
        num_additional_steps,
    ):
        # while robot_interface.state_buffer_size == 0:
        #     logger.warn("Robot state not received")
        #     time.sleep(0.5)

        target_delta_pos, target_delta_axis_angle = (
            target_delta_pose[:3],
            target_delta_pose[3:],
        )
        current_ee_pose = self.last_eef_pose
        current_pos = current_ee_pose[:3, 3:]
        current_rot = current_ee_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos

        target_axis_angle = np.array(target_delta_axis_angle) + current_axis_angle

        # logger.info(f"Before conversion {target_axis_angle}")
        target_quat = transform_utils.axisangle2quat(target_axis_angle)
        # target_pose = target_pos.flatten().tolist() + target_quat.flatten().tolist()

        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        # logger.info(f"After conversion {target_axis_angle}")
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        # start_pose = current_pos.flatten().tolist() + current_quat.flatten().tolist()

        self.osc_move(
            controller_type,
            (target_pos, target_quat),
            num_steps,
        )
        self.osc_move(
            controller_type,
            (target_pos, target_quat),
            num_additional_steps,
        )

    def osc_move(self, controller_type, target_pose, num_steps):
        target_pos, target_quat = target_pose
        # target_axis_angle = transform_utils.quat2axisangle(target_quat)
        current_rot, current_pos = self.last_eef_rot_and_pos

        for _ in range(num_steps):
            current_pose = self.last_eef_pose
            current_pos = current_pose[:3, 3:]
            current_rot = current_pose[:3, :3]
            current_quat = transform_utils.mat2quat(current_rot)
            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat
            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            current_axis_angle = transform_utils.quat2axisangle(current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
            action_pos = (target_pos - current_pos).flatten() * 10
            action_axis_angle = axis_angle_diff.flatten() * 1
            action_pos = np.clip(action_pos, -1.0, 1.0)
            action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

            action = action_pos.tolist() + action_axis_angle.tolist() + [-1.0]
            # logger.info(f"Action {action}")
            print(
                "Current pos:",
                np.round(current_pos, 2).tolist()
                + np.round(current_axis_angle, 2).tolist(),
            )
            print("Action:", np.round(action, 2))
            self.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
            )
        return action

    def home(self):
        # TODO: reset_robot_joints.py
        pass


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

        self._robot = Robot()
        self._robot.reset()

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
        translation = affine[:3, 3]
        rotation = R.from_matrix(affine[:3, :3]).as_rotvec()
        return np.concatenate([translation, rotation])

    def return_real(self):
        return True

    def _apply_retargeted_angles(self) -> None:
        self.controller_state = self._controller_state_subscriber.recv_keypoints()

        if self.is_first_frame:
            print("Starting control first frame")
            self._robot.home()
            time.sleep(2)
            self.home_affine = self._robot.last_eef_pose
            self.home_pose = self.affine_to_robot_pose_aa(self.home_affine)
            self.is_first_frame = False
        if self.controller_state.right_a:
            print("Start teleop")
            self.start_teleop = True
            self.init_affine = self.controller_state.right_affine
        if self.controller_state.right_b:
            self.start_teleop = False
            self.init_affine = None
            self.home_affine = self._robot.last_eef_pose
            self.home_pose = self.affine_to_robot_pose_aa(self.home_affine)

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
            self._robot.gripper_control(gripper_state)
            self.gripper_correct_state = gripper_state

        if self.start_teleop:
            delta_translation = relative_affine[:3, 3]
            delta_rotation_aa = R.from_matrix(relative_affine[:3, :3]).as_rotvec()

            delta_translation = np.clip(
                delta_translation,
                a_min=FRANKA_CART_STEP_LIMITS[0],
                a_max=FRANKA_CART_STEP_LIMITS[1],
            )

            delta_pose = np.concatenate((delta_translation, delta_rotation_aa))
        else:
            delta_pose = np.zeros((6, 1))

        print("Delta pose:", delta_pose)

        # Save the states here
        # TODO:

        if self.start_teleop:
            return
            # self._robot.move_to_target_pose(
            #     "OSC_POSE", delta_pose, num_steps=40, num_additional_steps=10
            # )

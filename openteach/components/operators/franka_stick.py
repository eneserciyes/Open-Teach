import time
from pathlib import Path
import os

import numpy as np
from numpy.linalg import pinv
from scipy.spatial.transform import Rotation as R

from openteach.components.operators.operator import Operator
from openteach.constants import (
    ARM_TELEOP_STOP,
    FRANKA_STEP_LIMITS,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    ROBOT_WORKSPACE_MAX,
    ROBOT_WORKSPACE_MIN,
    VR_FREQ,
    H_R_V_star,
    H_R_V,
)
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
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
            self.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
            )
        return action

    def get_joint_position(self):
        return self.last_q

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


def mat2posaa(mat):
    pos = mat[:3, 3]
    rot = mat[:3, :3]
    quat = transform_utils.mat2quat(rot)
    aa = transform_utils.quat2axisangle(quat)
    return pos, aa


def mat2posrot(mat):
    return mat[:3, 3:], mat[:3, :3]


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

        self._robot = Robot("deoxys.yml")
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
            self.home_rot, self.home_pos = self._robot.last_eef_rot_and_pos
            self.is_first_frame = False
        if self.controller_state.right_a:
            print("Start teleop")
            self.start_teleop = True
            self.init_affine = self.controller_state.right_affine
        if self.controller_state.right_b:
            self.start_teleop = False
            self.init_affine = None
            self.home_rot, self.home_pos = self._robot.last_eef_rot_and_pos

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
            relative_pos, relative_rot = mat2posrot(relative_affine)
            print(
                "Relative axis-angle:",
                transform_utils.quat2axisangle(transform_utils.mat2quat(relative_rot)),
            )

            target_pos = self.home_pos + relative_pos
            target_rot = self.home_rot @ relative_rot
            target_quat = transform_utils.mat2quat(target_rot)

            # clip with step limits and workspace limits
            _, current_pos = self._robot.last_eef_rot_and_pos
            target_pos = np.clip(
                np.clip(
                    target_pos,
                    a_min=current_pos + FRANKA_STEP_LIMITS[0],
                    a_max=current_pos + FRANKA_STEP_LIMITS[1],
                ),
                a_min=ROBOT_WORKSPACE_MIN,
                a_max=ROBOT_WORKSPACE_MAX,
            )

        else:
            target_pos, target_quat = (
                self.home_pos,
                transform_utils.mat2quat(self.home_rot),
            )

        print("Target axis-angle:", transform_utils.quat2axisangle(target_quat))

        # Save the states here
        # TODO:

        # if self.start_teleop:
        self._robot.osc_move("OSC_POSE", (target_pos, target_quat), num_steps=1)

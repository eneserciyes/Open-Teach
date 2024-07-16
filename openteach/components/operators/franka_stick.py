import time
from pathlib import Path
import os
from typing import Union

import numpy as np
from numpy.linalg import pinv

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
    FRANKA_HOME_JOINTS,
)
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from openteach.utils.timer import FrequencyTimer

from deoxys.utils import YamlConfig
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import (
    get_default_controller_config,
    verify_controller_config,
)


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

    def osc_move(self, controller_type, target_pose, gripper_state, num_steps):
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
            # current_axis_angle = transform_utils.quat2axisangle(current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
            action_pos = (target_pos - current_pos).flatten() * 10
            action_axis_angle = axis_angle_diff.flatten() * 1
            action_pos = np.clip(action_pos, -1.0, 1.0)
            action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

            action = action_pos.tolist() + action_axis_angle.tolist() + [gripper_state]
            # logger.info(f"Action {action}")
            self.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
            )
        return action

    def reset_joints_to(
        self,
        start_joint_pos: Union[list, np.ndarray],
        controller_cfg: dict = None,
        timeout=7,
        gripper_open=False,
    ):
        assert type(start_joint_pos) is list or type(start_joint_pos) is np.ndarray
        if controller_cfg is None:
            controller_cfg = get_default_controller_config(
                controller_type="JOINT_POSITION"
            )
        else:
            assert controller_cfg["controller_type"] == "JOINT_POSITION", (
                "This function is only for JOINT POSITION mode. You specified "
                + controller_cfg["controller_type"]
            )
            controller_cfg = verify_controller_config(controller_cfg)

        if gripper_open:
            gripper_action = -1
        else:
            gripper_action = 1
        if type(start_joint_pos) is list:
            action = start_joint_pos + [gripper_action]
        else:
            action = start_joint_pos.tolist() + [gripper_action]
        start_time = time.time()
        while True:
            if self.received_states and self.check_nonzero_configuration():
                if (
                    np.max(np.abs(np.array(self.last_q) - np.array(start_joint_pos)))
                    < 1e-3
                ):
                    break
            self.control(
                controller_type="JOINT_POSITION",
                action=action,
                controller_cfg=controller_cfg,
            )
            end_time = time.time()

            # Add timeout
            if end_time - start_time > timeout:
                break
        return True

    def get_joint_position(self):
        return self.last_q


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
        self.gripper_state = GRIPPER_OPEN
        self.gripper_flag = 1
        self.pause_flag = 1
        self.gripper_cnt = 0

        self.start_teleop = False
        self.init_affine = None

    def return_real(self):
        return True

    def _apply_retargeted_angles(self) -> None:
        self.controller_state = self._controller_state_subscriber.recv_keypoints()

        if self.is_first_frame:
            print("Starting control first frame")
            self._robot.reset_joints_to(FRANKA_HOME_JOINTS)
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

        gripper_action = None
        if self.controller_state.right_index_trigger > 0.5:
            gripper_action = GRIPPER_CLOSE
        elif self.controller_state.right_hand_trigger > 0.5:
            gripper_action = GRIPPER_OPEN

        if gripper_action is not None and gripper_action != self.gripper_state:
            print("Gripper controlling: ", gripper_action)
            self.gripper_state = gripper_action

        if self.start_teleop:
            relative_pos, relative_rot = (
                relative_affine[:3, 3:],
                relative_affine[:3, :3],
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

            # TODO: remove this
            target_quat = transform_utils.mat2quat(self.home_rot)
        else:
            target_pos, target_quat = (
                self.home_pos,
                transform_utils.mat2quat(self.home_rot),
            )

        # print("Target axis-angle:", transform_utils.quat2axisangle(target_quat))

        # Save the states here
        # TODO:

        self._robot.osc_move(
            "OSC_POSE",
            (target_pos, target_quat),
            self.gripper_state,
            num_steps=1,
        )

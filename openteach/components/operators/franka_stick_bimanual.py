import pickle
import time
from pathlib import Path
import os
from typing import Union

import numpy as np
from numpy.linalg import pinv

from openteach.components.component import Component
from openteach.constants import (
    VR_FREQ,
    FRANKA_HOME_JOINTS,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    ROBOT_WORKSPACE_MAX,
    ROBOT_WORKSPACE_MIN,
    ROTATION_VELOCITY_LIMIT,
    ROTATIONAL_POSE_VELOCITY_SCALE,
    TRANSLATION_VELOCITY_LIMIT,
    TRANSLATIONAL_POSE_VELOCITY_SCALE,
    H_R_V_left,
    H_R_V_star,
    H_R_V,
    H_R_V_star_left,
)
from openteach.utils.network import ZMQKeypointSubscriber
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
        self.velocity_controller_cfg = YamlConfig(
            os.path.join(CONFIG_ROOT, "osc-pose-controller-velocity.yml")
        ).as_easydict()

        self.velocity_controller_cfg = verify_controller_config(
            self.velocity_controller_cfg
        )

    def osc_move(self, controller_type, target_pose, gripper_state):
        target_pos, target_quat = target_pose
        target_mat = transform_utils.pose2mat(pose=(target_pos, target_quat))

        current_quat, current_pos = self.last_eef_quat_and_pos
        current_mat = transform_utils.pose2mat(
            pose=(current_pos.flatten(), current_quat.flatten())
        )

        pose_error = transform_utils.get_pose_error(
            target_pose=target_mat, current_pose=current_mat
        )

        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat

        quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        axis_angle_diff = transform_utils.quat2axisangle(quat_diff)

        action_pos = pose_error[:3] * TRANSLATIONAL_POSE_VELOCITY_SCALE
        action_axis_angle = axis_angle_diff.flatten() * ROTATIONAL_POSE_VELOCITY_SCALE

        action_pos, _ = transform_utils.clip_translation(
            action_pos, TRANSLATION_VELOCITY_LIMIT
        )
        action_axis_angle = np.clip(
            action_axis_angle, -ROTATION_VELOCITY_LIMIT, ROTATION_VELOCITY_LIMIT
        )

        action = action_pos.tolist() + action_axis_angle.tolist() + [gripper_state]

        print("Action:", action)
        self.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=self.velocity_controller_cfg,
        )

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

        # This is for varying initialization of joints a little bit to
        # increase data variation.
        start_joint_pos = [
            e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
            for e in start_joint_pos
        ]
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


def get_relative_affine(init_affine, current_affine, is_right_arm):
    H_V_des = pinv(init_affine) @ current_affine

    # Transform to robot frame.
    if is_right_arm:
        relative_affine_rot = (pinv(H_R_V) @ H_V_des @ H_R_V)[:3, :3]
        relative_affine_trans = (pinv(H_R_V_star) @ H_V_des @ H_R_V_star)[:3, 3]
    else:
        relative_affine_rot = (pinv(H_R_V_left) @ H_V_des @ H_R_V_left)[:3, :3]
        relative_affine_trans = (pinv(H_R_V_star_left) @ H_V_des @ H_R_V_star_left)[
            :3, 3
        ]

    # Homogeneous coordinates
    relative_affine = np.block(
        [[relative_affine_rot, relative_affine_trans.reshape(3, 1)], [0, 0, 0, 1]]
    )

    return relative_affine


class FrankaOperator(Component):
    def __init__(
        self,
        host,
        controller_state_port,
    ) -> None:
        # Subscribe controller state
        self._controller_state_subscriber = ZMQKeypointSubscriber(
            host=host, port=controller_state_port, topic="controller_state"
        )

        self.left_robot = Robot("deoxys_left.yml")
        self.right_robot = Robot("deoxys_right.yml")

        self.left_robot.reset()
        self.right_robot.reset()

        self.left_gripper_state = GRIPPER_OPEN
        self.right_gripper_state = GRIPPER_OPEN

        self.timer = FrequencyTimer(VR_FREQ)

        # Class variables
        self.is_first_frame = True

        self.left_start_teleop = False
        self.right_start_teleop = False

        self.left_init_affine = None
        self.right_init_affine = None

        self._states = []

    def return_real(self):
        return True

    def _apply_retargeted_angles(self) -> None:
        self.controller_state = self._controller_state_subscriber.recv_keypoints()

        if self.is_first_frame:
            print("Starting control first frame")
            self.left_robot.reset_joints_to(FRANKA_HOME_JOINTS)
            self.right_robot.reset_joints_to(FRANKA_HOME_JOINTS)
            time.sleep(2)
            self.left_home_rot, self.left_home_pos = (
                self.left_robot.last_eef_rot_and_pos
            )
            self.right_home_rot, self.right_home_pos = (
                self.right_robot.last_eef_rot_and_pos
            )
            self.is_first_frame = False

        # Check teleop state
        # Left robot
        if self.controller_state.left_x:
            print("Left Start teleop")
            self.left_start_teleop = True
            self.left_init_affine = self.controller_state.left_affine
        if self.controller_state.left_y:
            print("Left Stop teleop")
            self.left_start_teleop = False
            self.left_init_affine = None
            self.left_home_rot, self.left_home_pos = (
                self.left_robot.last_eef_rot_and_pos
            )

        # Right robot
        if self.controller_state.right_a:
            print("Right Start teleop")
            self.right_start_teleop = True
            self.right_init_affine = self.controller_state.right_affine
        if self.controller_state.right_b:
            print("Right Stop teleop")
            self.right_start_teleop = False
            self.right_init_affine = None
            self.right_home_rot, self.right_home_pos = (
                self.right_robot.last_eef_rot_and_pos
            )

        # Get relative affine
        # left robot
        if self.left_start_teleop:
            left_relative_affine = get_relative_affine(
                self.left_init_affine,
                self.controller_state.left_affine,
                is_right_arm=False,
            )
        else:
            left_relative_affine = np.zeros((4, 4))
            left_relative_affine[3, 3] = 1

        # Right robot
        if self.right_start_teleop:
            right_relative_affine = get_relative_affine(
                self.right_init_affine,
                self.controller_state.right_affine,
                is_right_arm=True,
            )
        else:
            right_relative_affine = np.zeros((4, 4))
            right_relative_affine[3, 3] = 1

        # Get gripper action
        left_gripper_action, right_gripper_action = None, None
        if self.controller_state.left_index_trigger > 0.5:
            left_gripper_action = GRIPPER_CLOSE
        elif self.controller_state.left_hand_trigger > 0.5:
            left_gripper_action = GRIPPER_OPEN

        if self.controller_state.right_index_trigger > 0.5:
            right_gripper_action = GRIPPER_CLOSE
        elif self.controller_state.right_hand_trigger > 0.5:
            right_gripper_action = GRIPPER_OPEN

        # Change gripper state
        if (
            left_gripper_action is not None
            and left_gripper_action != self.left_gripper_state
        ):
            print("Left Gripper controlling: ", left_gripper_action)
            self.left_gripper_state = left_gripper_action

        if (
            right_gripper_action is not None
            and right_gripper_action != self.right_gripper_state
        ):
            print("Right Gripper controlling: ", right_gripper_action)
            self.right_gripper_state = right_gripper_action

        # Obtain target pos and quat
        if self.left_start_teleop:
            left_target_pos = self.left_home_pos + left_relative_affine[:3, 3:]
            left_target_rot = self.left_home_rot @ left_relative_affine[:3, :3]
            left_target_quat = transform_utils.mat2quat(left_target_rot)
            left_target_pos = np.clip(
                left_target_pos,
                a_min=ROBOT_WORKSPACE_MIN,
                a_max=ROBOT_WORKSPACE_MAX,
            )
        else:
            left_target_pos, left_target_quat = (
                self.left_home_pos,
                transform_utils.mat2quat(self.left_home_rot),
            )

        if self.right_start_teleop:
            right_target_pos = self.right_home_pos + right_relative_affine[:3, 3:]
            right_target_rot = self.right_home_rot @ right_relative_affine[:3, :3]
            right_target_quat = transform_utils.mat2quat(right_target_rot)
            right_target_pos = np.clip(
                right_target_pos,
                a_min=ROBOT_WORKSPACE_MIN,
                a_max=ROBOT_WORKSPACE_MAX,
            )
        else:
            right_target_pos, right_target_quat = (
                self.right_home_pos,
                transform_utils.mat2quat(self.right_home_rot),
            )

        # Save the states here
        state = {
            "left_pose": self.left_robot.last_eef_quat_and_pos,
            "left_commanded_pose": np.concatenate(
                (left_target_pos.flatten(), left_target_quat)
            ),
            "right_pose": self.right_robot.last_eef_quat_and_pos,
            "right_commanded_pose": np.concatenate(
                (right_target_pos.flatten(), right_target_quat)
            ),
            "timestamp": time.time(),
        }
        self._states.append(state)

        # Control the robot
        self.left_robot.osc_move(
            "OSC_POSE",
            (left_target_pos.flatten(), left_target_quat.flatten()),
            self.left_gripper_state,
        )
        self.right_robot.osc_move(
            "OSC_POSE",
            (right_target_pos.flatten(), right_target_quat.flatten()),
            self.right_gripper_state,
        )

    def save_states(self):
        with open("extracted_data/test.pkl", "wb") as f:
            pickle.dump(self._states, f)

    def stream(self):
        self.notify_component_start("Bimanual Franka control")
        print("Start controlling the robot hand using the Oculus Headset.\n")

        try:
            while True:
                if (
                    self.left_robot.get_joint_position() is not None
                    and self.right_robot.get_joint_position() is not None
                ):
                    self.timer.start_loop()

                    # Retargeting function
                    self._apply_retargeted_angles()

                    self.timer.end_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.save_states()
            time.sleep(2)

        print("Stopping the teleoperator!")

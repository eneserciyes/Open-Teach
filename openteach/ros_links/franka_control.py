import numpy as np
import time
from pathlib import Path
import os

from deoxys.utils import YamlConfig
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import (
    get_default_controller_config,
    verify_controller_config,
)

from openteach.constants import (
    FRANKA_HOME_JOINTS,
    ROTATION_VELOCITY_LIMIT,
    ROTATIONAL_POSE_VELOCITY_SCALE,
    TRANSLATION_VELOCITY_LIMIT,
    TRANSLATIONAL_POSE_VELOCITY_SCALE,
)

CONFIG_ROOT = Path(__file__).parent


class Robot(FrankaInterface):
    def __init__(self, cfg):
        print("Running Franka interface:", cfg)
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

    def osc_move(self, target_pose, gripper_state):
        """
        target_pose: absolute pose as a tuple of target_pos and target_quat
         -  target_pos: (3, 1) array
         -  target_quat: (4, 1) array
        gripper_state: -1 for open, 1 for closed
        """
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
            controller_type="OSC_POSE",
            action=action,
            controller_cfg=self.velocity_controller_cfg,
        )

    def reset_joints_to(
        self,
        controller_cfg: dict = None,
        timeout=7,
        gripper_open=False,
    ):
        start_joint_pos = FRANKA_HOME_JOINTS
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


class DexArmControl:
    def __init__(self, cfg="deoxys.yml"):
        self.robot = Robot(cfg)

    def _init_franka_arm_control(self):
        self.robot.reset()

        while self.robot.state_buffer_size == 0:
            print("Warning: robot state buffer size 0")
            time.sleep(0.1)  # wait until buffer fills

    def get_arm_pose(self):
        return self.robot.last_eef_pose

    def get_arm_position(self):
        return self.robot.last_q

    def get_arm_velocity(self):
        raise ValueError(
            "get_arm_velocity() is being called - Arm Velocity cannot be collected in Franka arms, this method should not be called"
        )

    def get_arm_torque(self):
        raise ValueError(
            "get_arm_torque() is being called - Arm Torques cannot be collected in Franka arms, this method should not be called"
        )

    def get_arm_cartesian_coords(self):
        current_quat, current_pos = self.robot.last_eef_quat_and_pos

        current_pos = np.array(current_pos, dtype=np.float32).flatten()
        current_quat = np.array(current_quat, dtype=np.float32).flatten()

        current_aa = transform_utils.quat2axisangle(current_quat)

        cartesian_coord = np.concatenate([current_pos, current_aa], axis=0)

        return cartesian_coord

    def get_gripper_state(self):
        gripper_position = self.robot.last_gripper_q
        gripper_pose = dict(
            position=np.array(gripper_position, dtype=np.float32).flatten(),
            timestamp=time.time(),
        )
        return gripper_pose

    # Movement functions

    def arm_control(self, target_pose, gripper_status):
        self.robot.osc_move(target_pose, gripper_status)

    def home_arm(self):
        self.robot.reset_joints_to()

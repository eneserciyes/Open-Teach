from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import get_default_controller_config
import numpy as np
import time
import os
from copy import deepcopy as copy
from pathlib import Path

from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils

# from franka_arm.constants import *
# from franka_arm.controller import FrankaController
# from franka_arm.utils import generate_cartesian_space_min_jerk

FRANKA_HOME = [
    -1.5208185,
    1.5375434,
    1.4714179,
    -1.8101345,
    0.01227421,
    1.8809032,
    0.67484516,
]
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
        controller_cfg,
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
            controller_cfg,
            (target_pos, target_quat),
            num_steps,
        )
        self.osc_move(
            controller_type,
            controller_cfg,
            (target_pos, target_quat),
            num_additional_steps,
        )

    def osc_move(self, controller_type, controller_cfg, target_pose, num_steps):
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
                controller_cfg=controller_cfg,
            )
        return action


class DexArmControl:
    def __init__(self):
        self.robot = Robot("deoxys.yml")
        self.desired_cartesian_pose = None

    def init_franka_arm_control(self):
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
        current_quat, current_pos = self.robot.get_cartesian_position()

        current_pos = np.array(current_pos, dtype=np.float32).flatten()
        current_quat = np.array(current_quat, dtype=np.float32).flatten()

        # TODO: convert quat to axis angles in radians
        cartesian_coord = np.concatenate([current_pos, current_quat], axis=0)

        return cartesian_coord

    def get_arm_osc_position(self):
        current_pos, current_axis_angle = copy(self.franka.get_osc_position())
        current_pos = np.array(current_pos, dtype=np.float32).flatten()
        current_axis_angle = np.array(current_axis_angle, dtype=np.float32).flatten()

        osc_position = np.concatenate([current_pos, current_axis_angle], axis=0)

        return osc_position

    def get_arm_cartesian_state(self):
        current_pos, current_quat = copy(self.franka.get_cartesian_position())

        cartesian_state = dict(
            position=np.array(current_pos, dtype=np.float32).flatten(),
            orientation=np.array(current_quat, dtype=np.float32).flatten(),
            timestamp=time.time(),
        )

        return cartesian_state

    def get_arm_joint_state(self):
        joint_positions = self.robot.get_joint_position()
        return joint_positions

    # Movement functions

    def move_arm_joint(self, joint_angles):
        self.franka.joint_movement(joint_angles)

    def arm_control(self, action):
        # TODO: reactivate
        # pass
        self.robot.arm_control(action)

    def home_arm(self):
        # TODO: add move_arm_cartesian or somethign like this.
        # take this from Deoxys reset_robot_joints.py
        pass

import numpy as np
import time
import os
from copy import deepcopy as copy
from pathlib import Path

from deoxys.franka_interface import FrankaInterface

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

    def get_position_aa(self):
        # TODO: implement this
        pass


class DexArmControl:
    def __init__(self):
        self.robot = Robot("deoxys.yml")
        self.desired_cartesian_pose = None

    def init_franka_arm_control(self):
        self.robot.reset()

        # TODO: uncommment this back, or maybe we don't
        # need this because the control doesn't start
        # until the joint pose is not None.

        # while self.robot.state_buffer_size == 0:
        #     time.sleep(0.1)  # wait until buffer fills
        #     print("Warning: robot state buffer size 0")

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

    def arm_control(self, cartesian_pose):
        self.franka.cartesian_control(cartesian_pose=cartesian_pose)

    def home_arm(self):
        # TODO: add move_arm_cartesian or somethign like this.
        # take this from Deoxys reset_robot_joints.py
        pass

import time

import numpy as np
from numpy.linalg import pinv

from openteach.components.operators.operator import Operator
from openteach.constants import (
    VR_FREQ,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    ROBOT_WORKSPACE_MAX,
    ROBOT_WORKSPACE_MIN,
    H_R_V_star,
    H_R_V,
)
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from openteach.robot.franka_stick import FrankaArm
from openteach.utils.timer import FrequencyTimer

from deoxys.utils import transform_utils


def get_relative_affine(init_affine, current_affine):
    H_V_des = pinv(init_affine) @ current_affine

    # Transform to robot frame.
    relative_affine_rot = (pinv(H_R_V) @ H_V_des @ H_R_V)[:3, :3]
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

        self._robot = FrankaArm("deoxys_right.yml")
        self._robot.reset()
        self._timer = FrequencyTimer(VR_FREQ)

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

        # Class variables
        self.is_first_frame = True
        self.gripper_state = GRIPPER_OPEN
        self.start_teleop = False
        self.init_affine = None

    @property
    def timer(self):
        return self._timer

    @property
    def robot(self):
        return self._robot

    def return_real(self):
        return True

    @property
    def controller_state_subscriber(self):
        return self._controller_state_subscriber

    def _apply_retargeted_angles(self) -> None:
        self.controller_state = self._controller_state_subscriber.recv_keypoints()

        if self.is_first_frame:
            self.robot.home()
            time.sleep(2)
            self.home_rot, self.home_pos = (
                self.robot._controller.robot.last_eef_rot_and_pos
            )
            self.is_first_frame = False
        if self.controller_state.right_a:
            print("Start teleop")
            self.start_teleop = True
            self.init_affine = self.controller_state.right_affine
        if self.controller_state.right_b:
            print("Stop teleop")
            self.start_teleop = False
            self.init_affine = None
            self.home_rot, self.home_pos = (
                self.robot._controller.robot.last_eef_rot_and_pos
            )

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

            target_pos = np.clip(
                target_pos,
                a_min=ROBOT_WORKSPACE_MIN,
                a_max=ROBOT_WORKSPACE_MAX,
            )

        else:
            target_pos, target_quat = (
                self.home_pos,
                transform_utils.mat2quat(self.home_rot),
            )

        # Save the states here
        self.gripper_publisher.pub_keypoints(self.gripper_state, "gripper")
        position = self.robot.get_cartesian_position()
        joint_position = self.robot.get_joint_position()
        self.cartesian_publisher.pub_keypoints(position, "cartesian")
        self.joint_publisher.pub_keypoints(joint_position, "joint")
        self.cartesian_command_publisher.pub_keypoints(
            np.concatenate((target_pos, target_quat), "cartesian")
        )

        self._robot.arm_control(
            (target_pos.flatten(), target_quat.flatten()),
            self.gripper_state,
        )

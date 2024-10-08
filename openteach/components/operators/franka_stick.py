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
    H_R_V_star,
    H_R_V,
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
    def __init__(self, cfg, control_freq=20):
        super(Robot, self).__init__(
            general_cfg_file=os.path.join(CONFIG_ROOT, cfg),
            use_visualizer=False,
            control_freq=control_freq,
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


class FrankaOperator(Component):
    def __init__(self, host, controller_state_port, storage_path, demo_num) -> None:
        # Subscribe controller state
        self._controller_state_subscriber = ZMQKeypointSubscriber(
            host=host, port=controller_state_port, topic="controller_state"
        )
        self._storage_path = storage_path
        self._demo_num = demo_num

        self._robot = Robot("deoxys_right.yml", control_freq=90)
        self._robot.reset()
        self.timer = FrequencyTimer(VR_FREQ)

        # states lists
        self._poses = []
        self._commanded_poses = []
        self._gripper_states = []
        self._timestamps = []

        # Class variables
        self.is_first_frame = True
        self.gripper_state = GRIPPER_OPEN
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
            print("Stop teleop")
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
        quat, pos = self._robot.last_eef_quat_and_pos
        self._poses.append(np.concatenate((pos.flatten(), quat.flatten())))
        self._commanded_poses.append(
            np.concatenate((target_pos.flatten(), target_quat.flatten()))
        )
        self._gripper_states.append(self.gripper_state)
        self._timestamps.append(time.time())

        self._robot.osc_move(
            "OSC_POSE",
            (target_pos.flatten(), target_quat.flatten()),
            self.gripper_state,
        )

    def save_states(self):
        teleop_time = self._timestamps[-1] - self._timestamps[0]
        print(f"Took {teleop_time} seconds")
        print(f"Saved {len(self._timestamps)} datapoints..")
        print(f"Action save frequency : {len(self._timestamps) / teleop_time} Hz")

        save_path = Path(self._storage_path) / f"demonstration_{self._demo_num}"
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "states.pkl", "wb") as f:
            pickle.dump(
                {
                    "poses": self._poses,
                    "commanded_poses": self._commanded_poses,
                    "gripper_states": self._gripper_states,
                    "timestamps": self._timestamps,
                },
                f,
            )

    def stream(self):
        self.notify_component_start("Bimanual Franka control")
        print("Start controlling the robot hand using the Oculus Headset.\n")

        try:
            while True:
                if self._robot.get_joint_position() is not None:
                    self.timer.start_loop()

                    # Retargeting function
                    self._apply_retargeted_angles()

                    self.timer.end_loop()
        except KeyboardInterrupt:
            pass
        finally:
            if len(self._poses) != 0:
                self.save_states()
                time.sleep(2)

        print("Stopping the teleoperator!")

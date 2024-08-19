from abc import ABC
from openteach.components import Component


class Operator(Component, ABC):
    @property
    def timer(self):
        return self._timer

    # This function is used to create the robot
    @property
    def robot(self):
        return self._robot

    # This function is the subscriber for the hand keypoints
    @property
    def transformed_hand_keypoint_subscriber(self):
        return self._transformed_hand_keypoint_subscriber

    # This function is the subscriber for the arm keypoints
    @property
    def transformed_arm_keypoint_subscriber(self):
        return self._transformed_arm_keypoint_subscriber

    # This function has the majority of retargeting code happening
    def _apply_retargeted_angles(self):
        pass

    # This function applies the retargeted angles to the robot
    def stream(self):
        self.notify_component_start("{} control".format(self.robot))
        print("Start controlling the robot hand using the Oculus Headset.\n")

        while True:
            try:
                if self.return_real() is True:
                    if self.robot.get_joint_position() is not None:
                        self.timer.start_loop()

                        # Retargeting function
                        self._apply_retargeted_angles()

                        self.timer.end_loop()
                else:
                    self.timer.start_loop()

                    # Retargeting function
                    self._apply_retargeted_angles()

                    self.timer.end_loop()

            except KeyboardInterrupt:
                break

            finally:
                self._save_states()

        print("Stopping the teleoperator!")

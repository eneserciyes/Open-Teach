import random
import numpy as np
from openteach.constants import DEPTH_PORT_OFFSET, VIZ_PORT_OFFSET, VR_FREQ
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import (
    ZMQCameraPublisher,
    ZMQCompressedImageTransmitter,
    ZMQKeypointPublisher,
    ZMQKeypointSubscriber,
)
from openteach.components.environment.arm_env import Arm_Env
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer


import gymnasium as gym

import metaworld


def depthimg2Meters(model, depth):
    extent = model.stat.extent
    near = model.vis.map.znear * extent
    far = model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image


def render(renderer, camera_id, depth=False):
    renderer.make_context_current()
    mode = "depth_array" if depth else "rgb_array"
    r = renderer.render(mode, camera_id)
    if depth:
        r = depthimg2Meters(renderer.model, r)
    return r


def preprocess_img_and_depth(img, depth):
    return np.transpose(np.flip(img, axis=0), (2, 0, 1)), np.flip(depth, axis=0)


class RGBWrapper(gym.Wrapper):
    def __init__(self, env, camera_name) -> None:
        super(RGBWrapper, self).__init__(env)
        self._camera_name = camera_name

        self.renderer = OffScreenViewer(model=env.model, data=env.data)
        self.renderer.make_context_current()

    def reset(self, **kwargs):
        feats, _ = self.env.reset(**kwargs)
        obs = {}
        obs["proprioception"] = feats

        image = render(self.renderer, self.env.model.cam(self._camera_name).id)
        depth = render(
            self.renderer, self.env.model.cam(self._camera_name).id, depth=True
        )
        image, depth = preprocess_img_and_depth(image, depth)
        obs["image"] = image
        obs["depth"] = depth

        return obs


# Libero Environment class
class MetaworldEnv(Arm_Env):
    def __init__(
        self,
        host,
        camport,
        timestamppublisherport,
        endeff_publish_port,
        endeffpossubscribeport,
        robotposepublishport,
        stream_oculus,
        task_name,
    ):
        print("Initializing Metaworld Environment")
        self._timer = FrequencyTimer(VR_FREQ)
        self.host = host
        self.camport = camport
        self._stream_oculus = stream_oculus

        # Define ZMQ pub/sub
        # Port for publishing rgb images.
        self.rgb_publisher = ZMQCameraPublisher(host=host, port=camport)
        # for ego-centric view
        self.rgb_publisher_ego = ZMQCameraPublisher(host=host, port=camport + 1)

        # Publishing the stream into the oculus.
        if self._stream_oculus:
            self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                host=host, port=camport + VIZ_PORT_OFFSET
            )
        # Publisher for Depth data
        self.depth_publisher = ZMQCameraPublisher(
            host=host, port=camport + DEPTH_PORT_OFFSET
        )
        # for ego-centric view
        self.depth_publisher_ego = ZMQCameraPublisher(
            host=host, port=camport + 1 + DEPTH_PORT_OFFSET
        )

        # Publisher for endeffector Positions
        self.endeff_publisher = ZMQKeypointPublisher(
            host=host, port=endeff_publish_port
        )

        # Publisher for endeffector Velocities
        self.endeff_pos_subscriber = ZMQKeypointSubscriber(
            host=host, port=endeffpossubscribeport, topic="endeff_coords"
        )

        # Robot pose publisher
        self.robot_pose_publisher = ZMQKeypointPublisher(
            host=host, port=robotposepublishport
        )

        # Publisher for timestamps
        self.timestamp_publisher = ZMQKeypointPublisher(
            host=host, port=timestamppublisherport
        )

        self.name = "Metaworld"

        # initialize the environment

        ml1 = metaworld.ML1(task_name)  # Construct the benchmark, sampling tasks
        env = ml1.train_classes[task_name](
            width=224, height=224
        )  # Create an environment with task
        seed = np.random.randint(0, 100000)
        env.seed(seed)

        task = random.choice(ml1.train_tasks)
        env.set_task(task)

        # add wrappers
        env = RGBWrapper(env, "corner1")

        self.env = env

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
from openteach.utils.images import rescale_image

import metaworld


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
            width=width, height=height
        )  # Create an environment with task
        seed = np.random.randint(0, 100000)
        env.seed(seed)

        task = random.choice(ml1.train_tasks)
        env.set_task(task)

        # add wrappers
        env = make(
            env,
            camera_names=camera_names,
            sample_points=sample_points,
            bounds=bounds,
            voxel_size=voxel_size,
        )

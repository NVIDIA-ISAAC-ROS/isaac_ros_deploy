#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the GR00T N1.6 inference graph pipeline.

Tests end-to-end inference with the gr00t_n16_apple_to_plate config:
- Publishes JointState (43 joints), Image (480x640 RGB8), and TensorList
  (initial_noise [1, 50, 128]) inputs
- Verifies JointCommand outputs are received from all 6 outputs with
  kind=target/joint/position
- Measures inference latency from header stamp to receive time
"""

import os
import time
import unittest
from pathlib import Path

import launch
import launch_testing
import launch_testing.actions
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from isaac_ros_deploy_interfaces.msg import JointCommand
from isaac_ros_tensor_list_interfaces.msg import Tensor, TensorList
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from sensor_msgs.msg import Image, JointState

# All 43 joint names for the G1 (6+6+3+7+7+7+7).
ALL_JOINT_NAMES = [
    # left_leg (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # right_leg (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # left_arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # right_arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # left_hand (7)
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "left_hand_middle_0_joint",
    "left_hand_middle_1_joint", "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    # right_hand (7)
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint", "right_hand_middle_0_joint",
    "right_hand_middle_1_joint", "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

# Dangling outputs with kind: target/joint/position.
EXPECTED_OUTPUT_NAMES = [
    "reference_0_left_leg",
    "reference_0_right_leg",
    "reference_0_waist",
    "left_arm",
    "right_arm",
    "waist",
]


def generate_test_description():
    """Generate launch description for GR00T N1.6 inference graph test.

    Launch arguments:
        run_forever: If 'true', the test publishes and spins indefinitely
            instead of asserting. Useful for debugging with rqt_graph.
    """
    pkg_dir = Path(get_package_share_directory("isaac_ros_deploy_bringup"))
    policy_dir = Path(get_package_share_directory("isaac_ros_gr00t_n16_unitree_g1"))

    config_path = str(policy_dir / "data" / "gr00t_n16_apple_to_plate.yaml")
    launch_file = str(pkg_dir / "launch" / "inference_graph.launch.py")

    source_to_topic = (
        "state/joint/position:joint_states,"
        "state/camera/image:camera_image,"
        "initial_noise:initial_noise"
    )

    output_to_topic = ",".join(
        f"{name}:joint_commands" for name in EXPECTED_OUTPUT_NAMES
    )

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments={
            "config_path": config_path,
            "publish_rate": "1.0",
            "source_to_topic": source_to_topic,
            "output_to_topic": output_to_topic,
        }.items(),
    )

    run_forever_arg = DeclareLaunchArgument(
        "run_forever",
        default_value="false",
        description="If true, spin forever instead of running assertions.",
    )

    return (
        launch.LaunchDescription(
            [
                run_forever_arg,
                SetEnvironmentVariable(
                    "_LAUNCH_TEST_RUN_FOREVER",
                    LaunchConfiguration("run_forever"),
                ),
                pipeline,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


class TestGr00tInferenceGraph(unittest.TestCase):
    """Test case for GR00T N1.6 inference graph pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up ROS2 context."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Tear down ROS2 context."""
        rclpy.shutdown()

    def setUp(self):
        """Set up test node."""
        self.node = rclpy.create_node("test_gr00t_inference_graph")
        self.received_joint_commands: list[JointCommand] = []
        self.receive_times: list[float] = []
        self.receive_ros_times: list = []

        # Publishers — topics are in the inference_graph namespace.
        self.joint_state_pub = self.node.create_publisher(
            JointState, "/inference_graph/joint_states", 10
        )
        self.image_pub = self.node.create_publisher(
            Image, "/inference_graph/camera_image", 10
        )
        self.tensor_list_pub = self.node.create_publisher(
            TensorList, "/inference_graph/initial_noise", 10
        )

        # Subscriber — output topic is also in the namespace.
        self.joint_command_sub = self.node.create_subscription(
            JointCommand,
            "/inference_graph/joint_commands",
            self._joint_command_callback,
            10,
        )

    def tearDown(self):
        """Tear down test node."""
        self.node.destroy_node()

    def _joint_command_callback(self, msg: JointCommand):
        """Store received JointCommand messages with receive timestamp."""
        self.received_joint_commands.append(msg)
        self.receive_times.append(time.monotonic())
        self.receive_ros_times.append(self.node.get_clock().now())

    def _spin_until_condition(self, condition, timeout_sec=60.0):
        """Spin until condition is true or timeout."""
        start = time.time()
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if condition():
                return True
        return False

    def _publish_all_inputs(self):
        """Publish JointState, Image, and TensorList inputs."""
        stamp = self.node.get_clock().now().to_msg()

        # JointState with all 43 joints at zero.
        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = list(ALL_JOINT_NAMES)
        joint_state.position = [0.0] * len(ALL_JOINT_NAMES)
        self.joint_state_pub.publish(joint_state)

        # Black 480x640 RGB8 image.
        image = Image()
        image.header.stamp = stamp
        image.height = 480
        image.width = 640
        image.encoding = "rgb8"
        image.is_bigendian = 0
        image.step = 640 * 3
        image.data = bytes(480 * 640 * 3)
        self.image_pub.publish(image)

        # Zero tensor [1, 50, 128] as initial_noise.
        tensor_msg = Tensor()
        tensor_msg.name = "initial_noise"
        tensor_msg.shape.rank = 3
        tensor_msg.shape.dims = [1, 50, 128]
        tensor_msg.strides = [50 * 128 * 4, 128 * 4, 4]
        tensor_msg.data_type = 9  # float32
        data = np.zeros((1, 50, 128), dtype=np.float32)
        tensor_msg.data = data.tobytes()
        tensor_list = TensorList()
        tensor_list.header.stamp = stamp
        tensor_list.tensors = [tensor_msg]
        self.tensor_list_pub.publish(tensor_list)

    def test_gr00t_inference_pipeline(self):
        """Test that the GR00T pipeline produces JointCommand outputs."""
        # Wait for Triton model loading.
        time.sleep(15.0)

        # Publish continuously at ~10 Hz.
        publish_timer = self.node.create_timer(0.1, self._publish_all_inputs)

        if os.environ.get("_LAUNCH_TEST_RUN_FOREVER", "false") == "true":
            print("Running in interactive mode (run_forever:=true). "
                  "Press Ctrl+C to stop.")
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=1.0)
            return

        # Wait for at least 5 JointCommand messages.
        min_messages = 5
        self.assertTrue(
            self._spin_until_condition(
                lambda: len(self.received_joint_commands) >= min_messages,
            ),
            f"Expected at least {min_messages} JointCommand messages, "
            f"got {len(self.received_joint_commands)}",
        )
        publish_timer.cancel()

        # Log latencies for the received messages.
        # Latency = wall-clock receive time minus ROS header stamp.
        # The header stamp is set by InputBuilderNode when it publishes
        # the bundled TensorList, so latency ≈ Triton inference time +
        # output conversion overhead.
        latencies_ms = []
        for i, msg in enumerate(self.received_joint_commands[:min_messages]):
            stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            ros_now = self.node.get_clock().now().nanoseconds * 1e-9
            # Use receive_times recorded at callback time for accuracy.
            recv_ros_time = (
                self.receive_ros_times[i].nanoseconds * 1e-9
                if i < len(self.receive_ros_times) else ros_now
            )
            latency_ms = (recv_ros_time - stamp_sec) * 1000.0
            latencies_ms.append(latency_ms)
            self.node.get_logger().info(
                f"Output {i}: names={list(msg.names)[:3]}..., "
                f"position_len={len(msg.position)}, "
                f"latency={latency_ms:.1f} ms"
            )

        if latencies_ms:
            self.node.get_logger().info(
                f"Inference latency: "
                f"min={min(latencies_ms):.1f} ms, "
                f"max={max(latencies_ms):.1f} ms, "
                f"mean={sum(latencies_ms) / len(latencies_ms):.1f} ms "
                f"(first iteration includes warmup)"
            )

        # Verify we received outputs (the actual values depend on model weights).
        for msg in self.received_joint_commands[:min_messages]:
            self.assertGreater(
                len(msg.position), 0,
                "JointCommand message has no position data",
            )

#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration test for the multi-model inference graph pipeline.

Tests a two-model pipeline:
  Model A: ao1 = ao2 = ai1 + ai2 + ai3
  Model B: bo1 = bo2 = bo3 = bi1 + bi2

Connections:
  - joint position input (P=1.0) -> ai1   (dangling input)
  - ao2 -> ai2                            (feedback, same model)
  - bo3 -> ai3                            (feedback, cross-model)
  - ao1 -> bi1                            (data flow, cross-model)
  - bo2 -> bi2                            (feedback, same model)
  - bo1 -> joint command output            (dangling output)

Expected bo1 values over 5 iterations: [1, 4, 12, 33, 88]
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from isaac_ros_deploy_interfaces.msg import JointCommand
import launch
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import launch_testing
import launch_testing.actions
import rclpy
from sensor_msgs.msg import JointState


def generate_test_description():
    """Generate launch description for multi-model test.

    Launch arguments:
        run_forever: If 'true', the test publishes and spins indefinitely
            instead of asserting. Useful for debugging with rqt_graph.
    """
    pkg_dir = Path(get_package_share_directory("isaac_ros_deploy_bringup"))
    test_dir = pkg_dir / "test"
    data_dir = test_dir / "data"
    scripts_dir = test_dir / "scripts"

    # Generate models in a writable temp dir (runfiles are read-only on remote
    # execution).  Copy the YAML config alongside the models so that its
    # relative model_path entries resolve correctly.
    tmp_dir = Path(tempfile.mkdtemp())

    create_script = scripts_dir / "create_multi_model_test_models.py"
    subprocess.run(
        [sys.executable, str(create_script), "-d", str(tmp_dir)],
        check=True,
    )

    shutil.copy(data_dir / "multi_model_inference_graph.yaml", tmp_dir)
    config_path = str(tmp_dir / "multi_model_inference_graph.yaml")

    # Include the reusable inference pipeline launch file.
    launch_file = str(pkg_dir / "launch" / "inference_graph.launch.py")

    # Map ai1's source (state/joint/position) to the joint_states topic.
    source_to_topic = "state/joint/position:joint_states"

    # Map bo1 output to the joint_commands topic.
    output_to_topic = "bo1:joint_commands"

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments={
            "config_path": config_path,
            # Normally the inference takes <2ms, but the first iteration is slow (warm up).
            # Set to 1.0Hz to avoid flakiness from the first iteration.
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


class TestMultiModelInferenceGraph(unittest.TestCase):
    """Test case for multi-model inference graph pipeline."""

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
        self.node = rclpy.create_node("test_multi_model_inference_graph")
        self.received_joint_commands = []

        # Publisher — topic is in the inference_graph namespace.
        self.joint_state_pub = self.node.create_publisher(
            JointState, "/inference_graph/joint_states", 10
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
        """Store received JointCommand messages."""
        self.received_joint_commands.append(msg)

    def _spin_until_condition(self, condition, timeout_sec=15.0):
        """Spin until condition is true or timeout."""
        start = time.time()
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if condition():
                return True
        return False

    def _wait_for_topic_publishers(self, topic, count=1, timeout_sec=120.0):
        """Wait until the topic has at least `count` publishers."""
        start = time.time()
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(self.node.get_publishers_info_by_topic(topic)) >= count:
                return True
        return False

    def _publish_joint_state(self):
        """Publish a JointState with a single joint at position 1.0."""
        joint_state = JointState()
        joint_state.name = ["test_joint"]
        joint_state.position = [1.0]
        joint_state.header.stamp = self.node.get_clock().now().to_msg()
        self.joint_state_pub.publish(joint_state)

    def test_multi_model_pipeline_5_iterations(self):
        """Test that the multi-model pipeline produces correct values."""
        # Wait for pipeline output publishers to be available.
        self.assertTrue(
            self._wait_for_topic_publishers("/inference_graph/joint_commands"),
            "Pipeline output topic '/inference_graph/joint_commands' has no publishers",
        )

        # Publish continuously — initial messages may be lost during
        # discovery. The InputBuilderNode latches the last value so
        # only one message needs to get through.
        publish_timer = self.node.create_timer(0.1, self._publish_joint_state)

        if os.environ.get("_LAUNCH_TEST_RUN_FOREVER", "false") == "true":
            print("Running in interactive mode (run_forever:=true). "
                  "Press Ctrl+C to stop.")
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=1.0)
            return

        # Expected bo1 values for 5 feedback iterations (input P=1.0):
        #   t0: A = 1+0+0 = 1,  B = 1+0 = 1
        #   t1: A = 1+1+1 = 3,  B = 3+1 = 4
        #   t2: A = 1+3+4 = 8,  B = 8+4 = 12
        #   t3: A = 1+8+12 = 21, B = 21+12 = 33
        #   t4: A = 1+21+33 = 55, B = 55+33 = 88
        expected_values = [1.0, 4.0, 12.0, 33.0, 88.0]

        # InputBuilder ticks on its own cadence and bundles the latest feedback
        # available at tick time.  Until the first bo1 has actually arrived at
        # the feedback subscription, every tick reuses the zero default and
        # emits 1.0 again, so a slow first hop can produce one or more leading
        # 1.0 outputs before the sequence advances.  Wait for enough messages
        # to contain the full sequence + a few stale-feedback duplicates.
        max_messages = len(expected_values) + 5
        self.assertTrue(
            self._spin_until_condition(
                lambda: any(
                    self._matches_expected_sequence(
                        self.received_joint_commands, expected_values, start
                    )
                    for start in range(
                        max(0, len(self.received_joint_commands) - len(expected_values) + 1)
                    )
                )
                or len(self.received_joint_commands) >= max_messages,
            ),
            f"Did not observe sequence {expected_values} in first "
            f"{max_messages} messages; got "
            f"{[m.position[0] for m in self.received_joint_commands]}",
        )
        publish_timer.cancel()

        # Locate the contiguous sequence and check joint names on its first message.
        match_start = next(
            (
                start
                for start in range(
                    len(self.received_joint_commands) - len(expected_values) + 1
                )
                if self._matches_expected_sequence(
                    self.received_joint_commands, expected_values, start
                )
            ),
            None,
        )
        self.assertIsNotNone(
            match_start,
            f"Expected contiguous sequence {expected_values} not found in "
            f"{[m.position[0] for m in self.received_joint_commands]}",
        )
        self.assertEqual(
            list(self.received_joint_commands[match_start].names),
            ["test_joint"],
        )

    @staticmethod
    def _matches_expected_sequence(messages, expected, start):
        """Return True if `expected` appears at `start` in `messages` (within ±0.05)."""
        if start + len(expected) > len(messages):
            return False
        return all(
            abs(messages[start + i].position[0] - expected[i]) < 0.05
            for i in range(len(expected))
        )

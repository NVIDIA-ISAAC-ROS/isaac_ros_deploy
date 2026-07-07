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

"""Integration test for the inference graph pipeline.

This test verifies the full inference pipeline with multiple message types:
- JointState input with joint reordering -> JointCommand output
- Imu input (body rotation + angular velocity) -> BodyCommand output

All inputs pass through an identity ONNX model so output values should match inputs.
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
from isaac_ros_deploy_interfaces.msg import BodyCommand, JointCommand
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
from sensor_msgs.msg import Imu, JointState


def generate_test_description():
    """Generate launch description for test.

    Launch arguments:
        run_forever: If 'true', the test publishes and spins indefinitely
            instead of asserting. Useful for debugging with rqt_graph.
    """
    # Get paths
    pkg_dir = Path(get_package_share_directory("isaac_ros_deploy_bringup"))
    test_dir = pkg_dir / "test"
    data_dir = test_dir / "data"
    scripts_dir = test_dir / "scripts"

    # Generate model in a writable temp dir (runfiles are read-only on remote
    # execution).  Copy the YAML config alongside the model so that its
    # relative model_path resolves correctly.
    tmp_dir = Path(tempfile.mkdtemp())
    model_path = str(tmp_dir / "passthrough_model.onnx")

    create_script = scripts_dir / "create_passthrough_model.py"
    subprocess.run(
        [sys.executable, str(create_script), "-o", model_path],
        check=True,
    )

    shutil.copy(data_dir / "single_model_inference_graph.yaml", tmp_dir)
    config_path = str(tmp_dir / "single_model_inference_graph.yaml")

    # Include the reusable inference pipeline launch file.
    launch_file = str(pkg_dir / "launch" / "inference_graph.launch.py")
    # Map sources (kind values) to topics: joint position/velocity come from
    # joint_states, body rotation/angular_velocity come from imu.
    source_to_topic = (
        "state/joint/position:joint_states,"
        "state/joint/velocity:joint_states,"
        "state/body/rotation:imu,"
        "state/body/angular_velocity:imu"
    )
    output_to_topic = (
        "joint_pos_targets:joint_commands,"
        "joint_vel_targets:joint_commands,"
        "body_rot_target:body_commands,"
        "body_ang_vel_target:body_commands"
    )

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments={
            "config_path": config_path,
            "publish_rate": "50.0",
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


class TestInferenceGraph(unittest.TestCase):
    """Test case for inference graph pipeline."""

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
        self.node = rclpy.create_node("test_inference_graph")
        self.received_joint_commands = []
        self.received_body_commands = []

        # Publishers — topics are in the inference_graph namespace.
        self.joint_state_pub = self.node.create_publisher(
            JointState, "/inference_graph/joint_states", 10
        )
        self.imu_pub = self.node.create_publisher(
            Imu, "/inference_graph/imu", 10
        )

        # Subscribers — output topics are also in the namespace.
        self.joint_command_sub = self.node.create_subscription(
            JointCommand,
            "/inference_graph/joint_commands",
            self._joint_command_callback,
            10,
        )
        self.body_command_sub = self.node.create_subscription(
            BodyCommand,
            "/inference_graph/body_commands",
            self._body_command_callback,
            10,
        )

    def tearDown(self):
        """Tear down test node."""
        self.node.destroy_node()

    def _joint_command_callback(self, msg: JointCommand):
        """Store received JointCommand messages."""
        self.received_joint_commands.append(msg)

    def _body_command_callback(self, msg: BodyCommand):
        """Store received BodyCommand messages."""
        self.received_body_commands.append(msg)

    def _spin_until_condition(self, condition, timeout_sec=10.0):
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

    def _publish_inputs(self):
        """Publish JointState and Imu test inputs."""
        joint_state = JointState()
        joint_state.name = ["joint_3", "joint_2", "joint_1"]
        joint_state.position = [1.0, 2.0, 3.0]
        joint_state.velocity = [4.0, 5.0, 6.0]

        imu = Imu()
        imu.orientation.x = 0.1
        imu.orientation.y = 0.2
        imu.orientation.z = 0.3
        imu.orientation.w = 0.4
        imu.angular_velocity.x = 1.0
        imu.angular_velocity.y = 2.0
        imu.angular_velocity.z = 3.0

        stamp = self.node.get_clock().now().to_msg()
        joint_state.header.stamp = stamp
        imu.header.stamp = stamp
        self.joint_state_pub.publish(joint_state)
        self.imu_pub.publish(imu)

    def test_inference_graph_passthrough(self):
        """Test that all inputs pass through correctly to outputs."""
        # Wait for pipeline output publishers to be available.
        self.assertTrue(
            self._wait_for_topic_publishers("/inference_graph/joint_commands"),
            "Pipeline output topic '/inference_graph/joint_commands' has no publishers",
        )

        # Publish continuously — initial messages may be lost during
        # discovery. The InputBuilderNode latches the last value so
        # only one message needs to get through.
        publish_timer = self.node.create_timer(0.1, self._publish_inputs)

        if os.environ.get("_LAUNCH_TEST_RUN_FOREVER", "false") == "true":
            print("Running in interactive mode (run_forever:=true). "
                  "Press Ctrl+C to stop.")
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=1.0)
            return

        # --- Verify JointCommand output ---

        self.assertTrue(
            self._spin_until_condition(
                lambda: len(self.received_joint_commands) > 0
            ),
            "Did not receive any JointCommand messages",
        )

        cmd = self.received_joint_commands[-1]

        # Output should be in NN order: [joint_1, joint_2, joint_3]
        self.assertEqual(
            list(cmd.names),
            ["joint_1", "joint_2", "joint_3"],
            f"Expected names [joint_1, joint_2, joint_3], got {cmd.names}",
        )

        # After reordering to NN order and passthrough:
        # joint_1=3, joint_2=2, joint_3=1 -> position [3, 2, 1]
        expected_pos = [3.0, 2.0, 1.0]
        for i, (actual, expected) in enumerate(
            zip(cmd.position, expected_pos)
        ):
            self.assertAlmostEqual(
                actual,
                expected,
                places=5,
                msg=f"Position[{i}]: expected {expected}, got {actual}",
            )

        # velocity: joint_1=6, joint_2=5, joint_3=4 -> [6, 5, 4]
        expected_vel = [6.0, 5.0, 4.0]
        for i, (actual, expected) in enumerate(
            zip(cmd.velocity, expected_vel)
        ):
            self.assertAlmostEqual(
                actual,
                expected,
                places=5,
                msg=f"Velocity[{i}]: expected {expected}, got {actual}",
            )

        # --- Verify BodyCommand output ---

        self.assertTrue(
            self._spin_until_condition(
                lambda: len(self.received_body_commands) > 0
            ),
            "Did not receive any BodyCommand messages",
        )

        body_cmd = self.received_body_commands[-1]

        # Body rotation should pass through unchanged
        self.assertEqual(len(body_cmd.pose), 1)
        orient = body_cmd.pose[0].orientation
        self.assertAlmostEqual(orient.x, 0.1, places=5, msg="orient.x")
        self.assertAlmostEqual(orient.y, 0.2, places=5, msg="orient.y")
        self.assertAlmostEqual(orient.z, 0.3, places=5, msg="orient.z")
        self.assertAlmostEqual(orient.w, 0.4, places=5, msg="orient.w")

        # Body angular velocity should pass through unchanged
        self.assertEqual(len(body_cmd.twist), 1)
        ang_vel = body_cmd.twist[0].angular
        self.assertAlmostEqual(ang_vel.x, 1.0, places=5, msg="ang_vel.x")
        self.assertAlmostEqual(ang_vel.y, 2.0, places=5, msg="ang_vel.y")
        self.assertAlmostEqual(ang_vel.z, 3.0, places=5, msg="ang_vel.z")

        publish_timer.cancel()

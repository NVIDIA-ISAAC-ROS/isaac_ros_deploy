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

"""
Shared test helpers and base test cases for MuJoCo integration tests.

Provides reusable functions for common test operations (controller readiness
polling, simulation reset, stabilization waits, height monitoring) and two
base TestCase classes that robot-specific test files subclass with a small
set of configuration class variables.
"""

import math
import os
import time
import unittest

from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Twist
from isaac_ros_deploy_interfaces.msg import JointCommand
from mujoco_ros2_control_msgs.srv import ResetWorld
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
import rclpy.parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import SetBool as SetGantryEnabled
import tf2_ros

# Minimum root-body height to consider the robot standing. Use 0.3m as a
# conservative threshold that works for both G1 (~0.75m) and T1 (~0.7m).
MIN_ROOT_HEIGHT_M = 0.3

GANTRY_SERVICE = "/virtual_gantry/set_gantry_enabled"

# Multi-reset configuration.
RESET_COUNT = 4
RESET_SPACING_S = 0.15


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def has_display() -> bool:
    """Return whether a DISPLAY environment variable is set (for enabling the viewer)."""
    return "DISPLAY" in os.environ


def wait_for_controllers(
    test_case: unittest.TestCase,
    node: Node,
    controller_names: list[str],
    timeout_s: float,
) -> None:
    """
    Poll ListControllers until all named controllers are active.

    Fails the test if the controllers do not become active within timeout_s.
    """
    client = node.create_client(ListControllers, "/controller_manager/list_controllers")

    node.get_logger().info("Waiting for controllers to be active...")
    start = time.time()

    while time.time() - start < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)

        if not client.service_is_ready():
            continue

        future = client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)

        if future.result() is None:
            continue

        states = {c.name: c.state for c in future.result().controller}
        if all(states.get(name) == "active" for name in controller_names):
            node.get_logger().info(f"Controllers active: {', '.join(controller_names)}")
            return

    test_case.fail(
        f"Controllers did not become active within {timeout_s}s: {controller_names}"
    )


def wait_for_is_active(
    test_case: unittest.TestCase,
    node: Node,
    controller_names: list[str],
    timeout_s: float,
) -> None:
    """
    Wait for each listed controller to latch `True` on its `<name>/is_active`
    topic.

    Controllers with deferred internal activation (notably `InferenceController`,
    which defers its core until the first `/cmd_vel` arrives) report lifecycle
    `active` immediately but don't start writing commands until the core is
    ready.  This helper subscribes to the controller's latched
    `<controller>/is_active` topic (`std_msgs/Bool`) and blocks until every
    listed controller has published `data=True`.  Use this *after*
    `wait_for_controllers`, before gantry disable, so the gantry is released at
    a moment when the policy is genuinely producing commands.
    """
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    active: dict[str, bool] = {name: False for name in controller_names}
    subs = []
    for name in controller_names:
        def _callback(msg: Bool, _name: str = name) -> None:
            if msg.data:
                active[_name] = True
        subs.append(node.create_subscription(Bool, f"/{name}/is_active", _callback, qos))

    node.get_logger().info(
        f"Waiting for controllers to report is_active=True: {controller_names}"
    )
    start = time.time()
    while time.time() - start < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
        if all(active.values()):
            node.get_logger().info(
                f"Controllers is_active=True: {', '.join(controller_names)}"
            )
            for sub in subs:
                node.destroy_subscription(sub)
            return

    pending = [name for name, is_active in active.items() if not is_active]
    for sub in subs:
        node.destroy_subscription(sub)
    test_case.fail(
        f"Controllers did not report is_active=True within {timeout_s}s: {pending}"
    )


def reset_simulation(
    test_case: unittest.TestCase,
    node: Node,
    *,
    assert_on_failure: bool = True,
) -> None:
    """
    Reset the MuJoCo simulation multiple times to flush stale policy state.

    When assert_on_failure is True, test assertions are used and failures abort
    the test. When False, failures are logged as warnings (useful for pipeline
    tests where the reset service may not be critical).
    """
    service_name = "/mujoco_ros2_control_node/reset_world"
    client = node.create_client(ResetWorld, service_name)

    node.get_logger().info("Resetting simulation...")
    service_available = client.wait_for_service(timeout_sec=5.0)

    if assert_on_failure:
        test_case.assertTrue(service_available, f"{service_name} service not available")
    elif not service_available:
        node.get_logger().warning(f"{service_name} service not available")
        return

    for i in range(RESET_COUNT):
        future = client.call_async(ResetWorld.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

        if assert_on_failure:
            test_case.assertIsNotNone(future.result(), "Reset service call failed")
            test_case.assertTrue(future.result().success, "Simulation reset failed")
        elif future.result() is None:
            node.get_logger().warning("Reset service call failed")
        else:
            node.get_logger().info(f"Reset result: {future.result().message}")

        if i < RESET_COUNT - 1:
            _spin_for(node, RESET_SPACING_S)

    node.get_logger().info("Simulation reset successfully")


def spin_for(node: Node, duration_s: float, message: str = "") -> None:
    """
    Spin the node for a fixed duration while processing callbacks.

    Optionally logs a message before waiting.
    """
    if message:
        node.get_logger().info(message)
    _spin_for(node, duration_s)


def wait_for_first_message(
    test_case: unittest.TestCase,
    node: Node,
    msg_type: type,
    topic: str,
    timeout_s: float = 60.0,
) -> None:
    """
    Wait until the first message is received on a topic.

    Subscribes to the topic, spins until a message arrives, then destroys the
    subscription. Fails the test if no message arrives within timeout_s
    (wall-clock, since this may run before sim time is available).
    """
    received = []

    sub = node.create_subscription(msg_type, topic, lambda msg: received.append(msg), 10)

    node.get_logger().info(f"Waiting for first message on '{topic}'...")
    start = time.time()
    while time.time() - start < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
        if received:
            node.get_logger().info(
                f"Received first message on '{topic}' after {time.time() - start:.1f}s"
            )
            node.destroy_subscription(sub)
            return

    node.destroy_subscription(sub)
    test_case.fail(f"No message received on '{topic}' within {timeout_s}s")


def monitor_height(
    test_case: unittest.TestCase,
    node: Node,
    duration_s: float,
    *,
    root_frame: str,
    fail_early: bool = False,
) -> None:
    """
    Monitor root-body height via TF for duration_s and assert the robot did not fall.

    Args:
    ----
    test_case : unittest.TestCase
        Test case instance for assertions.
    node : Node
        ROS2 node for spinning and TF lookups.
    duration_s : float
        Duration in seconds to monitor height.
    root_frame : str
        TF frame of the robot's root body (e.g. "pelvis" for G1, "Trunk" for T1).
    fail_early : bool
        When True, the test fails immediately when the height drops
        below the threshold. Otherwise, the minimum height is checked only at the end.

    """
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)  # noqa: F841

    min_height = float("inf")
    start = node.get_clock().now()
    test_duration = Duration(seconds=duration_s)

    node.get_logger().info(f"Starting height monitoring for {duration_s}s...")

    while (node.get_clock().now() - start) < test_duration:
        rclpy.spin_once(node, timeout_sec=0.1)

        try:
            transform = tf_buffer.lookup_transform(
                "world",
                root_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
            height = transform.transform.translation.z
            min_height = min(min_height, height)

            if fail_early and height < MIN_ROOT_HEIGHT_M:
                test_case.fail(
                    f"Robot fell! Height {height:.3f}m < {MIN_ROOT_HEIGHT_M}m threshold"
                )

        except tf2_ros.TransformException:
            continue

    node.get_logger().info(f"Height monitoring complete. Min height: {min_height:.3f}m")

    test_case.assertTrue(
        math.isfinite(min_height),
        "No valid height readings received - TF lookup failed throughout the test",
    )
    test_case.assertGreater(
        min_height,
        MIN_ROOT_HEIGHT_M,
        f"Robot fell during test! Minimum height observed: {min_height:.3f}m",
    )

    node.get_logger().info(f"Test passed! Minimum height observed: {min_height:.3f}m")


def enable_gantry(node: Node, timeout_s: float = 10.0) -> bool:
    """Enable the virtual gantry. Returns True if the service succeeded."""
    client = node.create_client(SetGantryEnabled, GANTRY_SERVICE)
    if not client.wait_for_service(timeout_sec=timeout_s):
        node.get_logger().error(f"{GANTRY_SERVICE} not available")
        return False
    req = SetGantryEnabled.Request()
    req.data = True
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    result = future.result()
    if result is None or not result.success:
        node.get_logger().warning("enable_gantry: service call failed or timed out")
        return False
    node.get_logger().info("Gantry enabled")
    return True


def disable_gantry(node: Node) -> bool:
    """Disable the virtual gantry. Returns True if the service succeeded."""
    client = node.create_client(SetGantryEnabled, GANTRY_SERVICE)
    if not client.wait_for_service(timeout_sec=5.0):
        return False
    req = SetGantryEnabled.Request()
    req.data = False
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    result = future.result()
    if result is None or not result.success:
        node.get_logger().warning("disable_gantry: service call failed or timed out")
        return False
    node.get_logger().info("Gantry disabled")
    return True


def _spin_for(node: Node, duration_s: float) -> None:
    """Spin the node for a fixed duration of sim time while processing callbacks.

    Uses the node's clock (sim time when ``use_sim_time=True``) so stabilization
    waits scale with MuJoCo's actual progress rather than wall-clock; under CI
    compute load the simulator runs slower than real-time and a wall-clock wait
    expires before the robot has settled.
    """
    start = node.get_clock().now()
    duration = Duration(seconds=duration_s)
    while (node.get_clock().now() - start) < duration:
        rclpy.spin_once(node, timeout_sec=0.05)


# ---------------------------------------------------------------------------
# Base test cases
# ---------------------------------------------------------------------------

class MujocoControllerManagerTestBase(unittest.TestCase):
    """
    Base test case for controller-manager-based MuJoCo integration tests.

    Subclasses must set:
        CONTROLLERS: list of controller names to wait for before testing.
        ROOT_FRAME:  TF frame of the robot's root body (e.g. "pelvis", "Trunk").

    Subclasses may override:
        PUBLISH_CMD_VEL:           True to publish zero /cmd_vel at a fixed rate.
        CONTROLLER_STARTUP_WAIT_S: Seconds to wait for controllers to become active.
        READY_CONTROLLERS:         Controllers that additionally publish a latched
                                   `<name>/is_active` std_msgs/Bool once their
                                   deferred core is producing commands.  Gantry
                                   disable is deferred until every listed controller
                                   has reported True.
        STABILIZATION_WAIT_S:      Seconds to spin while the gantry is still holding
                                   the robot, giving the policy time to warm up before
                                   the gantry is released.
        TEST_DURATION_S:           Seconds to monitor height.
        CMD_VEL_PUBLISH_PERIOD_S:  Timer period for /cmd_vel publishing.
    """

    CONTROLLERS: list[str] = []
    READY_CONTROLLERS: list[str] = []
    PUBLISH_CMD_VEL: bool = False
    ROOT_FRAME: str

    CONTROLLER_STARTUP_WAIT_S: float = 60.0
    TEST_DURATION_S: float = 5.0
    STABILIZATION_WAIT_S: float = 1.0
    CMD_VEL_PUBLISH_PERIOD_S: float = 0.05

    @classmethod
    def setUpClass(cls):
        if not hasattr(cls, "ROOT_FRAME"):
            return  # Abstract base class — no setup needed.
        rclpy.init()
        cls.node = rclpy.create_node(
            "test_controller_manager_node",
            parameter_overrides=[
                rclpy.parameter.Parameter("use_sim_time", value=True),
            ],
        )
        if cls.PUBLISH_CMD_VEL:
            cls.cmd_vel_pub = cls.node.create_publisher(Twist, "/cmd_vel", 10)
            cls.publish_timer = cls.node.create_timer(
                cls.CMD_VEL_PUBLISH_PERIOD_S,
                lambda: cls.cmd_vel_pub.publish(Twist()),
            )

    @classmethod
    def tearDownClass(cls):
        if not hasattr(cls, "ROOT_FRAME"):
            return  # Abstract base class — nothing to tear down.
        if cls.PUBLISH_CMD_VEL:
            cls.publish_timer.cancel()
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_robot_does_not_fall(self):
        """Verify robot stays above height threshold."""
        if not hasattr(type(self), "ROOT_FRAME"):
            self.skipTest("Abstract base class — ROOT_FRAME not configured")
        wait_for_controllers(
            self, self.node, self.CONTROLLERS, self.CONTROLLER_STARTUP_WAIT_S
        )
        if self.READY_CONTROLLERS:
            wait_for_is_active(
                self, self.node, self.READY_CONTROLLERS, self.CONTROLLER_STARTUP_WAIT_S
            )
        spin_for(
            self.node,
            self.STABILIZATION_WAIT_S,
            f"Waiting {self.STABILIZATION_WAIT_S}s before releasing gantry...",
        )
        self.assertTrue(disable_gantry(self.node))
        monitor_height(self, self.node, self.TEST_DURATION_S, root_frame=self.ROOT_FRAME)


class MujocoInferenceGraphTestBase(unittest.TestCase):
    """
    Base test case for inference-graph-based MuJoCo integration tests.

    Subclasses must set:
        ROOT_FRAME: TF frame of the robot's root body (e.g. "pelvis", "Trunk").

    Subclasses may override:
        PUBLISH_CMD_VEL:           True to publish zero /cmd_vel at a fixed rate.
        CONTROLLER_STARTUP_WAIT_S: Seconds to wait for controllers to become active.
        READY_CONTROLLERS:         Controllers that additionally publish a latched
                                   `<name>/is_active` std_msgs/Bool once their
                                   deferred core is producing commands.  Gantry
                                   disable is deferred until every listed controller
                                   has reported True.
        PIPELINE_OUTPUT_TOPIC:     Topic to wait for first message on (pipeline ready signal).
        PIPELINE_OUTPUT_MSG_TYPE:  Message type for the pipeline output topic.
        STABILIZATION_WAIT_S:      Seconds to spin while the gantry is still holding
                                   the robot, giving the policy time to warm up before
                                   the gantry is released.
        TEST_DURATION_S:           Seconds to monitor height.
        CMD_VEL_PUBLISH_PERIOD_S:  Timer period for /cmd_vel publishing.
    """

    CONTROLLERS: list[str] = ["forward_joint_command_controller"]
    READY_CONTROLLERS: list[str] = []
    PUBLISH_CMD_VEL: bool = False
    ROOT_FRAME: str

    CONTROLLER_STARTUP_WAIT_S: float = 60.0
    PIPELINE_OUTPUT_TOPIC: str = "/joint_commands"
    PIPELINE_OUTPUT_MSG_TYPE: type = JointCommand
    TEST_DURATION_S: float = 5.0
    STABILIZATION_WAIT_S: float = 1.0
    CMD_VEL_PUBLISH_PERIOD_S: float = 0.05

    @classmethod
    def setUpClass(cls):
        if not hasattr(cls, "ROOT_FRAME"):
            return  # Abstract base class — no setup needed.
        rclpy.init()
        cls.node = rclpy.create_node(
            "test_inference_graph_node",
            parameter_overrides=[
                rclpy.parameter.Parameter("use_sim_time", value=True),
            ],
        )
        if cls.PUBLISH_CMD_VEL:
            cls.cmd_vel_pub = cls.node.create_publisher(Twist, "/cmd_vel", 10)
            cls.publish_timer = cls.node.create_timer(
                cls.CMD_VEL_PUBLISH_PERIOD_S,
                lambda: cls.cmd_vel_pub.publish(Twist()),
            )

    @classmethod
    def tearDownClass(cls):
        if not hasattr(cls, "ROOT_FRAME"):
            return  # Abstract base class — nothing to tear down.
        if cls.PUBLISH_CMD_VEL:
            cls.publish_timer.cancel()
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_robot_does_not_fall(self):
        """Verify robot stays above height threshold via the inference graph."""
        if not hasattr(type(self), "ROOT_FRAME"):
            self.skipTest("Abstract base class — ROOT_FRAME not configured")
        wait_for_controllers(
            self, self.node, self.CONTROLLERS, self.CONTROLLER_STARTUP_WAIT_S
        )
        if self.READY_CONTROLLERS:
            wait_for_is_active(
                self, self.node, self.READY_CONTROLLERS, self.CONTROLLER_STARTUP_WAIT_S
            )
        wait_for_first_message(
            self,
            self.node,
            self.PIPELINE_OUTPUT_MSG_TYPE,
            self.PIPELINE_OUTPUT_TOPIC,
        )
        spin_for(
            self.node,
            self.STABILIZATION_WAIT_S,
            f"Waiting {self.STABILIZATION_WAIT_S}s before releasing gantry...",
        )
        self.assertTrue(disable_gantry(self.node))
        monitor_height(
            self, self.node, self.TEST_DURATION_S, root_frame=self.ROOT_FRAME, fail_early=True
        )

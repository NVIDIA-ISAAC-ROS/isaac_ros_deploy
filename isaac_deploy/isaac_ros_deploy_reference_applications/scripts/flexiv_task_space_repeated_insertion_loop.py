#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeated DisplayPort task-space insertion loop for the Flexiv Cartesian driver.

This script assumes the Flexiv Cartesian driver and the task-space Deploy policy
launch are already running. It only orchestrates one repeated trial loop:

* disable policy passthrough,
* command a Cartesian TCP reset/hold pose,
* publish the socket pose,
* enable policy passthrough for a fixed duration,
* disable passthrough and hold the current TCP pose,
* optionally record one rosbag per trial.

It intentionally does not switch Flexiv drivers or controllers.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import select
import signal
import subprocess
import time
from typing import List, Optional, Sequence, Tuple

try:
    from flexiv_msgs.msg import CartesianMotionForceCommand
except ImportError:  # pragma: no cover - optional outside the Flexiv workspace
    CartesianMotionForceCommand = None
from geometry_msgs.msg import Pose, PoseStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


def _stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _double_array_parameter(node: Node, name: str) -> List[float]:
    value = node.get_parameter(name).get_parameter_value()
    values = list(value.double_array_value)
    if not values:
        values = [float(v) for v in value.integer_array_value]
    return values


def _string_array_parameter(node: Node, name: str) -> List[str]:
    return list(node.get_parameter(name).get_parameter_value().string_array_value)


def _pose_to_text(pose: Pose) -> str:
    return (
        "pos=["
        f"{pose.position.x:.6f}, {pose.position.y:.6f}, {pose.position.z:.6f}] "
        "quat_xyzw=["
        f"{pose.orientation.x:.6f}, {pose.orientation.y:.6f}, "
        f"{pose.orientation.z:.6f}, {pose.orientation.w:.6f}]"
    )


def _copy_pose(pose: Pose) -> Pose:
    copied = Pose()
    copied.position.x = float(pose.position.x)
    copied.position.y = float(pose.position.y)
    copied.position.z = float(pose.position.z)
    copied.orientation.x = float(pose.orientation.x)
    copied.orientation.y = float(pose.orientation.y)
    copied.orientation.z = float(pose.orientation.z)
    copied.orientation.w = float(pose.orientation.w)
    return copied


def _smoothstep(t: float) -> float:
    clamped = min(1.0, max(0.0, float(t)))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _quat_values(pose: Pose) -> Tuple[float, float, float, float]:
    return (
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )


def _normalize_quat(quat: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, z, w = (float(value) for value in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def _interpolate_pose(start: Pose, target: Pose, fraction: float) -> Pose:
    alpha = _smoothstep(fraction)
    pose = Pose()
    pose.position.x = float(start.position.x) + alpha * (
        float(target.position.x) - float(start.position.x)
    )
    pose.position.y = float(start.position.y) + alpha * (
        float(target.position.y) - float(start.position.y)
    )
    pose.position.z = float(start.position.z) + alpha * (
        float(target.position.z) - float(start.position.z)
    )

    start_quat = _quat_values(start)
    target_quat = _quat_values(target)
    dot = sum(a * b for a, b in zip(start_quat, target_quat))
    if dot < 0.0:
        target_quat = tuple(-value for value in target_quat)
    quat = _normalize_quat(
        (1.0 - alpha) * a + alpha * b
        for a, b in zip(start_quat, target_quat)
    )
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose


def _pose_from_arrays(position: Sequence[float], quat_xyzw: Sequence[float]) -> Pose:
    if len(position) != 3:
        raise ValueError(f"position must contain 3 values, got {position!r}")
    if len(quat_xyzw) != 4:
        raise ValueError(f"quat_xyzw must contain 4 values, got {quat_xyzw!r}")
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


def _position_distance(a: Pose, b: Pose) -> float:
    dx = float(a.position.x) - float(b.position.x)
    dy = float(a.position.y) - float(b.position.y)
    dz = float(a.position.z) - float(b.position.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class FlexivTaskSpaceRepeatedInsertionLoop(Node):
    """Run repeated task-space policy trials with a single Cartesian driver session."""

    def __init__(self) -> None:
        super().__init__("flexiv_task_space_repeated_insertion_loop")

        self.declare_parameter("execute", False)
        self.declare_parameter("cycles", 20)
        # First cycle to actually run. Cycles below this are skipped, but their
        # socket randomisation draws are still replayed so that a resumed run sees
        # the same socket positions it would have in an uninterrupted run.
        self.declare_parameter("start_cycle", 1)
        self.declare_parameter("feedback_topic", "/cartesian_motion_controller/tcp_pose")
        self.declare_parameter("command_topic", "/cartesian_motion_controller/command")
        self.declare_parameter(
            "enable_topic", "/displayport_task_space_policy/enable_passthrough"
        )
        self.declare_parameter("socket_pose_topic", "/displayport_socket_pose")
        self.declare_parameter("socket_pose_frame", "world")
        self.declare_parameter("socket_position", [0.473, 0.126, 0.070])
        self.declare_parameter("socket_orientation_xyzw", [-0.5, -0.5, -0.5, 0.5])
        self.declare_parameter("randomize_socket_xy", False)
        self.declare_parameter("socket_xy_random_range_m", 0.0)
        self.declare_parameter("socket_random_seed", 0)
        self.declare_parameter("socket_publish_rate_hz", 10.0)
        self.declare_parameter("insertion_duration_sec", 5.0)
        self.declare_parameter("reset_publish_rate_hz", 30.0)
        self.declare_parameter("reset_duration_sec", 5.0)
        self.declare_parameter("reset_hold_sec", 1.0)
        self.declare_parameter("feedback_timeout_sec", 5.0)
        self.declare_parameter("record_reset_pose", False)
        self.declare_parameter(
            "reset_position",
            [0.4758855998516083, 0.1288170963525772, 0.16309799253940582],
        )
        self.declare_parameter(
            "reset_orientation_xyzw",
            [
                0.7081848978996277,
                0.7060220241546631,
                0.0025250522885471582,
                -0.0007912407163530588,
            ],
        )
        self.declare_parameter("wait_for_user_after_reset", False)
        self.declare_parameter("hold_reset_during_user_wait", True)
        self.declare_parameter("hold_current_after_trial_sec", 0.5)
        self.declare_parameter("disable_passthrough_on_exit", True)
        self.declare_parameter("record_bags", False)
        self.declare_parameter(
            "bag_output_dir", "policy_debug_logs/taskspace_repeated_insertion"
        )
        self.declare_parameter("bag_pre_roll_sec", 1.0)
        self.declare_parameter("bag_storage", "mcap")
        self.declare_parameter("robot_sn", "")
        self.declare_parameter(
            "bag_topics",
            [
                "/displayport_socket_pose",
                "/displayport_task_space_policy/target_pose",
                "/displayport_task_space_policy/debug_policy_io",
                "/displayport_task_space_policy/enable_passthrough",
                "/cartesian_motion_controller/command",
                "/cartesian_motion_controller/tcp_pose",
                "/joint_states",
                "/flexiv_arm/joint_states",
            ],
        )

        self.execute = bool(self.get_parameter("execute").value)
        self.cycles = int(self.get_parameter("cycles").value)
        self.start_cycle = int(self.get_parameter("start_cycle").value)
        self.feedback_topic = str(self.get_parameter("feedback_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.enable_topic = str(self.get_parameter("enable_topic").value)
        self.socket_pose_topic = str(self.get_parameter("socket_pose_topic").value)
        self.socket_pose_frame = str(self.get_parameter("socket_pose_frame").value)
        self.socket_position = _double_array_parameter(self, "socket_position")
        self.socket_orientation_xyzw = _double_array_parameter(
            self, "socket_orientation_xyzw"
        )
        self.randomize_socket_xy = bool(self.get_parameter("randomize_socket_xy").value)
        self.socket_xy_random_range_m = float(
            self.get_parameter("socket_xy_random_range_m").value
        )
        self.socket_random_seed = int(self.get_parameter("socket_random_seed").value)
        self.socket_publish_rate_hz = float(
            self.get_parameter("socket_publish_rate_hz").value
        )
        self.insertion_duration_sec = float(
            self.get_parameter("insertion_duration_sec").value
        )
        self.reset_publish_rate_hz = float(
            self.get_parameter("reset_publish_rate_hz").value
        )
        self.reset_duration_sec = float(self.get_parameter("reset_duration_sec").value)
        self.reset_hold_sec = float(self.get_parameter("reset_hold_sec").value)
        self.feedback_timeout_sec = float(
            self.get_parameter("feedback_timeout_sec").value
        )
        self.record_reset_pose = bool(self.get_parameter("record_reset_pose").value)
        self.reset_position = _double_array_parameter(self, "reset_position")
        self.reset_orientation_xyzw = _double_array_parameter(
            self, "reset_orientation_xyzw"
        )
        self.wait_for_user_after_reset = bool(
            self.get_parameter("wait_for_user_after_reset").value
        )
        self.hold_reset_during_user_wait = bool(
            self.get_parameter("hold_reset_during_user_wait").value
        )
        self.hold_current_after_trial_sec = float(
            self.get_parameter("hold_current_after_trial_sec").value
        )
        self.disable_passthrough_on_exit = bool(
            self.get_parameter("disable_passthrough_on_exit").value
        )
        self.record_bags = bool(self.get_parameter("record_bags").value)
        self.bag_output_dir = str(self.get_parameter("bag_output_dir").value)
        self.bag_pre_roll_sec = float(self.get_parameter("bag_pre_roll_sec").value)
        self.bag_storage = str(self.get_parameter("bag_storage").value)
        self.bag_topics = _string_array_parameter(self, "bag_topics")
        robot_sn = str(self.get_parameter("robot_sn").value).strip().replace("-", "_")
        if robot_sn:
            self.bag_topics += [
                f"/{robot_sn}/flange_pose",
                f"/{robot_sn}/flexiv_robot_states",
            ]

        self._validate_parameters()

        self.current_socket_position = list(self.socket_position)
        self.current_socket_xy_offset = [0.0, 0.0]
        self._rng = random.Random(self.socket_random_seed)
        self._advance_rng_for_skipped_cycles()

        self.latest_feedback: Optional[PoseStamped] = None
        self.latest_joint_state: Optional[JointState] = None

        self.create_subscription(PoseStamped, self.feedback_topic, self._feedback_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.command_pub = self.create_publisher(
            CartesianMotionForceCommand, self.command_topic, 10
        )
        self.enable_pub = self.create_publisher(Bool, self.enable_topic, 10)
        self.socket_pub = self.create_publisher(PoseStamped, self.socket_pose_topic, 10)

        mode = "EXECUTE" if self.execute else "DRY-RUN"
        self.get_logger().warning(
            f"{mode} task-space repeated insertion loop. "
            f"running cycles {self.start_cycle}-{self.cycles}. "
            f"Reset/hold via {self.command_topic!r}; socket pose via "
            f"{self.socket_pose_topic!r}; passthrough enable via {self.enable_topic!r}."
        )
        if not self.execute:
            self.get_logger().warning(
                "execute:=false, so no robot command, socket pose, or enable messages "
                "will be published. Set execute:=true for real robot motion."
            )

    def _validate_parameters(self) -> None:
        if self.start_cycle < 1:
            raise ValueError("start_cycle must be >= 1")
        if self.start_cycle > self.cycles:
            raise ValueError(
                f"start_cycle ({self.start_cycle}) must be <= cycles ({self.cycles}); "
                "cycles is the last cycle number, not a count of cycles to run"
            )
        if self.cycles <= 0:
            raise ValueError("cycles must be positive")
        if len(self.socket_position) != 3:
            raise ValueError("socket_position must contain 3 values")
        if len(self.socket_orientation_xyzw) != 4:
            raise ValueError("socket_orientation_xyzw must contain 4 values")
        if self.socket_xy_random_range_m < 0.0:
            raise ValueError("socket_xy_random_range_m must be non-negative")
        if self.socket_publish_rate_hz <= 0.0:
            raise ValueError("socket_publish_rate_hz must be positive")
        if self.insertion_duration_sec <= 0.0:
            raise ValueError("insertion_duration_sec must be positive")
        if self.reset_publish_rate_hz <= 0.0:
            raise ValueError("reset_publish_rate_hz must be positive")
        if self.reset_duration_sec < 0.0:
            raise ValueError("reset_duration_sec must be non-negative")
        if self.reset_hold_sec < 0.0:
            raise ValueError("reset_hold_sec must be non-negative")
        if self.feedback_timeout_sec <= 0.0:
            raise ValueError("feedback_timeout_sec must be positive")
        if self.hold_current_after_trial_sec < 0.0:
            raise ValueError("hold_current_after_trial_sec must be non-negative")
        if self.bag_pre_roll_sec < 0.0:
            raise ValueError("bag_pre_roll_sec must be non-negative")
        if self.reset_position and len(self.reset_position) != 3:
            raise ValueError("reset_position must be empty or contain 3 values")
        if self.reset_orientation_xyzw and len(self.reset_orientation_xyzw) != 4:
            raise ValueError("reset_orientation_xyzw must be empty or contain 4 values")
        if bool(self.reset_position) != bool(self.reset_orientation_xyzw):
            raise ValueError(
                "reset_position and reset_orientation_xyzw must both be set or both be empty"
            )

    def _feedback_cb(self, msg: PoseStamped) -> None:
        self.latest_feedback = msg

    def _joint_state_cb(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def _wait_for_feedback(self, reason: str) -> bool:
        if not self.execute:
            # A dry run prints the trial plan without a driver, so waiting for TCP
            # feedback would only block for the timeout and log an error per reset.
            self.get_logger().info(f"DRY-RUN: not waiting for TCP feedback while {reason}.")
            return True
        deadline = time.time() + self.feedback_timeout_sec
        while rclpy.ok() and self.latest_feedback is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.latest_feedback is None:
            self.get_logger().error(
                f"No TCP feedback received on {self.feedback_topic!r} while {reason}."
            )
            return False
        return True

    def _publish_enable(self, enabled: bool) -> None:
        if not self.execute:
            self.get_logger().info(f"DRY-RUN: would publish enable={enabled}.")
            return
        msg = Bool()
        msg.data = bool(enabled)
        # Publish a few times so a late subscription match does not miss a one-shot gate.
        for _ in range(3):
            self.enable_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.05)
        self.get_logger().info(f"Published passthrough enable={enabled}.")

    def _make_socket_pose(self) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = self.socket_pose_frame
        msg.pose = _pose_from_arrays(
            self.current_socket_position,
            self.socket_orientation_xyzw,
        )
        return msg

    def _advance_rng_for_skipped_cycles(self) -> None:
        """Replay the RNG draws the skipped cycles would have consumed.

        _sample_socket_position_for_cycle() draws exactly two values per cycle, and
        only when randomisation is active, so burning 2 * skipped draws here leaves
        the generator in the state cycle `start_cycle` would have reached in an
        uninterrupted run from cycle 1.
        """
        skipped = self.start_cycle - 1
        if skipped <= 0:
            return
        if not (self.randomize_socket_xy and self.socket_xy_random_range_m > 0.0):
            self.get_logger().warning(
                f"Skipping cycles 1-{skipped}; socket randomisation is off so every "
                "cycle uses the same socket position."
            )
            return
        for _ in range(2 * skipped):
            self._rng.uniform(
                -self.socket_xy_random_range_m,
                self.socket_xy_random_range_m,
            )
        self.get_logger().warning(
            f"Skipping cycles 1-{skipped}; advanced the socket RNG by "
            f"{2 * skipped} draws (seed={self.socket_random_seed}) so cycle "
            f"{self.start_cycle} matches an uninterrupted run."
        )

    def _sample_socket_position_for_cycle(self, cycle: int) -> None:
        self.current_socket_position = list(self.socket_position)
        dx = 0.0
        dy = 0.0
        if self.randomize_socket_xy and self.socket_xy_random_range_m > 0.0:
            dx = self._rng.uniform(
                -self.socket_xy_random_range_m,
                self.socket_xy_random_range_m,
            )
            dy = self._rng.uniform(
                -self.socket_xy_random_range_m,
                self.socket_xy_random_range_m,
            )
            self.current_socket_position[0] += dx
            self.current_socket_position[1] += dy
        self.current_socket_xy_offset = [dx, dy]
        self.get_logger().warning(
            f"Cycle {cycle}: socket goal position={self.current_socket_position}; "
            f"xy_offset=[{dx * 1000.0:+.3f}, {dy * 1000.0:+.3f}] mm; "
            f"randomize_socket_xy={self.randomize_socket_xy}; "
            f"seed={self.socket_random_seed}"
        )

    def _print_socket_goal_before_user_input(self, cycle: int) -> None:
        dx, dy = self.current_socket_xy_offset
        print("", flush=True)
        print(f"--- Cycle {cycle} socket goal that will be published ---", flush=True)
        print(f"Topic: {self.socket_pose_topic}", flush=True)
        print(f"Frame: {self.socket_pose_frame}", flush=True)
        print(
            "Base socket position: "
            f"[{self.socket_position[0]:.6f}, "
            f"{self.socket_position[1]:.6f}, "
            f"{self.socket_position[2]:.6f}]",
            flush=True,
        )
        print(
            "Random XY offset: "
            f"dx={dx:+.6f} m ({dx * 1000.0:+.3f} mm), "
            f"dy={dy:+.6f} m ({dy * 1000.0:+.3f} mm)",
            flush=True,
        )
        print(
            "Published socket position: "
            f"[{self.current_socket_position[0]:.6f}, "
            f"{self.current_socket_position[1]:.6f}, "
            f"{self.current_socket_position[2]:.6f}]",
            flush=True,
        )
        print(
            "Published socket quat xyzw: "
            f"[{self.socket_orientation_xyzw[0]:.6f}, "
            f"{self.socket_orientation_xyzw[1]:.6f}, "
            f"{self.socket_orientation_xyzw[2]:.6f}, "
            f"{self.socket_orientation_xyzw[3]:.6f}]",
            flush=True,
        )
        print(
            f"Publish rate: {self.socket_publish_rate_hz:.3f} Hz for "
            f"{self.insertion_duration_sec:.3f} s",
            flush=True,
        )

    def _publish_socket_pose_once(self) -> None:
        if not self.execute:
            return
        msg = self._make_socket_pose()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.socket_pub.publish(msg)

    def _make_cartesian_command(self, pose: Pose) -> CartesianMotionForceCommand:
        msg = CartesianMotionForceCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.target_pose = _copy_pose(pose)
        return msg

    def _publish_cartesian_pose_for(self, pose: Pose, duration_sec: float, label: str) -> None:
        if duration_sec <= 0.0:
            return

        if not self.execute:
            self.get_logger().info(f"DRY-RUN: would publish {label}: {_pose_to_text(pose)}.")
            end = time.time() + duration_sec
            while rclpy.ok() and time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.1)
            return

        period = 1.0 / self.reset_publish_rate_hz
        end = time.time() + duration_sec
        count = 0
        while rclpy.ok() and time.time() < end:
            self.command_pub.publish(self._make_cartesian_command(pose))
            count += 1
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)
        self.get_logger().info(f"Published {count} {label} commands.")

    def _publish_interpolated_cartesian_reset_for(
        self, target_pose: Pose, duration_sec: float, label: str
    ) -> None:
        if duration_sec <= 0.0:
            return

        if not self._wait_for_feedback(f"starting {label}"):
            self.get_logger().warning(
                f"No current TCP feedback for {label}; publishing fixed reset target."
            )
            self._publish_cartesian_pose_for(target_pose, duration_sec, label)
            return

        start_pose = self._current_feedback_pose()
        assert start_pose is not None
        self.get_logger().warning(
            f"{label}: interpolating TCP reset over {duration_sec:.2f} s from "
            f"{_pose_to_text(start_pose)} to {_pose_to_text(target_pose)}"
        )

        if not self.execute:
            end = time.time() + duration_sec
            while rclpy.ok() and time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.1)
            return

        period = 1.0 / self.reset_publish_rate_hz
        start_time = time.time()
        end_time = start_time + duration_sec
        count = 0
        while rclpy.ok() and time.time() < end_time:
            fraction = (time.time() - start_time) / duration_sec
            pose = _interpolate_pose(start_pose, target_pose, fraction)
            self.command_pub.publish(self._make_cartesian_command(pose))
            count += 1
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        self.command_pub.publish(self._make_cartesian_command(target_pose))
        self.get_logger().info(f"Published {count + 1} interpolated {label} commands.")

    def _wait_for_enter(self, pose_to_hold: Pose, cycle: int) -> None:
        self._print_socket_goal_before_user_input(cycle)
        prompt = (
            f"\n>>> Cycle {cycle}: reset complete. Press ENTER to start insertion, "
            "or Ctrl+C to stop <<<\n"
        )
        if not self.hold_reset_during_user_wait:
            self._read_enter(prompt)
            return

        print(prompt, end="", flush=True)
        period = 1.0 / self.reset_publish_rate_hz
        try:
            with open("/dev/tty", "r", encoding="utf-8") as tty:
                while rclpy.ok():
                    ready, _, _ = select.select([tty], [], [], period)
                    if ready:
                        tty.readline()
                        return
                    if self.execute:
                        self.command_pub.publish(self._make_cartesian_command(pose_to_hold))
                    rclpy.spin_once(self, timeout_sec=0.0)
        except OSError:
            self._read_enter()

    @staticmethod
    def _read_enter(prompt: str = "") -> None:
        """Block until the operator presses Enter.

        The next thing the caller does is start the insertion, so failing to read
        the prompt must stop the run rather than fall through: with stdin closed,
        as in a detached container, returning here would move the robot without
        the confirmation the operator asked for.
        """
        try:
            input(prompt)
        except EOFError as exc:
            raise RuntimeError(
                "wait_for_user_after_reset is set but neither /dev/tty nor stdin can be "
                "read, so the insertion cannot be confirmed. Re-run with "
                "wait_for_user_after_reset:=false to run unattended."
            ) from exc

    def _start_bag(self, cycle: int) -> Optional[subprocess.Popen]:
        if not self.record_bags:
            return None
        output_root = Path(os.path.expanduser(os.path.expandvars(self.bag_output_dir)))
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_root / f"trial_{cycle:04d}_{stamp}"
        command = [
            "ros2",
            "bag",
            "record",
            "-s",
            self.bag_storage,
            "-o",
            str(output_path),
            *self.bag_topics,
        ]
        self.get_logger().warning(f"Starting rosbag for cycle {cycle}: {output_path}")
        return subprocess.Popen(command)

    def _stop_bag(self, process: Optional[subprocess.Popen], cycle: int) -> None:
        if process is None:
            return
        if process.poll() is not None:
            self.get_logger().warning(
                f"rosbag process for cycle {cycle} already exited with "
                f"code {process.returncode}."
            )
            return
        self.get_logger().warning(f"Stopping rosbag for cycle {cycle}.")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.get_logger().error(
                f"rosbag process for cycle {cycle} did not stop after SIGINT; terminating."
            )
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()

    def _run_insertion_window(self, cycle: int) -> None:
        period = 1.0 / self.socket_publish_rate_hz
        end = time.time() + self.insertion_duration_sec
        count = 0
        while rclpy.ok() and time.time() < end:
            self._publish_socket_pose_once()
            count += 1
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)
        self.get_logger().warning(
            f"Cycle {cycle}: insertion window ended after "
            f"{self.insertion_duration_sec:.2f} s; published {count} socket poses."
        )

    def _current_feedback_pose(self) -> Optional[Pose]:
        if self.latest_feedback is None:
            return None
        return _copy_pose(self.latest_feedback.pose)

    def _resolve_reset_pose(self) -> Optional[Pose]:
        if self.reset_position:
            return _pose_from_arrays(self.reset_position, self.reset_orientation_xyzw)
        if not self.record_reset_pose:
            self.get_logger().error(
                "No reset_position/reset_orientation_xyzw provided and record_reset_pose=false."
            )
            return None
        if not self._wait_for_feedback("recording reset pose"):
            return None
        assert self.latest_feedback is not None
        return _copy_pose(self.latest_feedback.pose)

    def run(self) -> int:
        if not self._wait_for_feedback("starting repeated loop"):
            return 1

        reset_pose = self._resolve_reset_pose()
        if reset_pose is None:
            return 1

        self.get_logger().warning(f"Reset TCP pose: {_pose_to_text(reset_pose)}")
        self.get_logger().warning(
            "Socket pose: "
            f"pos={self.socket_position}, quat_xyzw={self.socket_orientation_xyzw}, "
            f"frame={self.socket_pose_frame!r}, "
            f"randomize_xy={self.randomize_socket_xy}, "
            f"range=+/-{self.socket_xy_random_range_m * 1000.0:.3f} mm, "
            f"seed={self.socket_random_seed}"
        )

        exit_code = 0
        try:
            self._publish_enable(False)
            for cycle in range(self.start_cycle, self.cycles + 1):
                self.get_logger().warning(f"===== Starting cycle {cycle}/{self.cycles} =====")
                self._sample_socket_position_for_cycle(cycle)

                self._publish_enable(False)
                self._publish_interpolated_cartesian_reset_for(
                    reset_pose,
                    self.reset_duration_sec,
                    f"cycle {cycle} reset",
                )
                self._publish_cartesian_pose_for(
                    reset_pose,
                    self.reset_hold_sec,
                    f"cycle {cycle} reset-hold",
                )

                current_pose = self._current_feedback_pose()
                if current_pose is not None:
                    self.get_logger().info(
                        f"Cycle {cycle}: reset feedback error="
                        f"{_position_distance(reset_pose, current_pose):.6f} m; "
                        f"feedback={_pose_to_text(current_pose)}"
                    )

                if self.wait_for_user_after_reset:
                    self._wait_for_enter(reset_pose, cycle)

                bag_process = self._start_bag(cycle)
                try:
                    if self.bag_pre_roll_sec > 0.0:
                        self.get_logger().info(
                            f"Cycle {cycle}: bag pre-roll {self.bag_pre_roll_sec:.2f} s."
                        )
                        pre_roll_end = time.time() + self.bag_pre_roll_sec
                        while rclpy.ok() and time.time() < pre_roll_end:
                            self._publish_socket_pose_once()
                            rclpy.spin_once(self, timeout_sec=0.05)

                    self._publish_socket_pose_once()
                    self._publish_enable(True)
                    self._run_insertion_window(cycle)
                    self._publish_enable(False)

                    hold_pose = self._current_feedback_pose()
                    if hold_pose is not None:
                        self.get_logger().info(
                            f"Cycle {cycle}: holding current TCP pose after trial: "
                            f"{_pose_to_text(hold_pose)}"
                        )
                        self._publish_cartesian_pose_for(
                            hold_pose,
                            self.hold_current_after_trial_sec,
                            f"cycle {cycle} post-trial hold",
                        )
                finally:
                    self._stop_bag(bag_process, cycle)

            self.get_logger().warning("Repeated task-space insertion loop completed.")
        except KeyboardInterrupt:
            self.get_logger().warning("Interrupted by user.")
            exit_code = 130
        finally:
            if self.disable_passthrough_on_exit:
                self._publish_enable(False)
        return exit_code


FLEXIV_MSGS_HINT = (
    "flexiv_msgs is unavailable. It ships with the flexiv_ros2 source repository "
    "rather than as a Debian package; clone and build it into the workspace "
    "to use the Flexiv task-space workflow."
)


def main() -> None:
    if CartesianMotionForceCommand is None:
        raise SystemExit(FLEXIV_MSGS_HINT)
    rclpy.init()
    node = FlexivTaskSpaceRepeatedInsertionLoop()
    try:
        raise SystemExit(node.run())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

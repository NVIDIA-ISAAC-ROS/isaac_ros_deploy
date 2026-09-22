#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass DisplayPort task-space policy Cartesian targets to a Flexiv command sink.

This node is intentionally thin. It does not add insertion success thresholds,
workspace clipping, action clipping, or target rescaling. Its only hardware-facing
checks are a finite-pose check, an explicit enable gate, and stale-command
handling.
"""

from __future__ import annotations

from collections import deque
import csv
import math
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence, Tuple

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_space_pose_math import (  # noqa: E402  (needs the sys.path entry above)
    as_float_list,
    pose_to_arrays,
    quat_rotate_xyzw,
    TcpOffsetEstimator,
)

try:
    from isaac_ros_deploy_interfaces.msg import BodyCommand, CartesianPoseDeltaCommand
except ImportError:  # pragma: no cover - optional outside the Deploy workspace
    BodyCommand = None
    CartesianPoseDeltaCommand = None

try:
    from flexiv_msgs.msg import CartesianMotionForceCommand, RobotStates
except ImportError:  # pragma: no cover - optional outside the Flexiv workspace
    CartesianMotionForceCommand = None
    RobotStates = None


XYZ_NAMES = ("x", "y", "z")
QUAT_NAMES = ("qx", "qy", "qz", "qw")

# Pose-delta commands are consumed one per target pose. The bound only has to
# absorb commands that arrive while a target is in flight; it is not a step
# cache, so a small value is enough and keeps a disabled confirmation prompt
# from retaining messages indefinitely.
MAX_PENDING_POSE_DELTA_COMMANDS = 16


def _stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.9f}"


def _pose_values(msg: PoseStamped) -> Tuple[float, ...]:
    return (
        float(msg.pose.position.x),
        float(msg.pose.position.y),
        float(msg.pose.position.z),
        float(msg.pose.orientation.x),
        float(msg.pose.orientation.y),
        float(msg.pose.orientation.z),
        float(msg.pose.orientation.w),
    )


def _pose_is_finite(msg: PoseStamped) -> bool:
    return all(math.isfinite(value) for value in _pose_values(msg))


def _pose_quaternion_norm(msg: PoseStamped) -> float:
    quat = _pose_values(msg)[3:]
    return math.sqrt(sum(value * value for value in quat))


def _copy_pose_stamped(msg: PoseStamped) -> PoseStamped:
    copied = PoseStamped()
    copied.header.stamp = msg.header.stamp
    copied.header.frame_id = msg.header.frame_id
    copied.pose.position.x = float(msg.pose.position.x)
    copied.pose.position.y = float(msg.pose.position.y)
    copied.pose.position.z = float(msg.pose.position.z)
    copied.pose.orientation.x = float(msg.pose.orientation.x)
    copied.pose.orientation.y = float(msg.pose.orientation.y)
    copied.pose.orientation.z = float(msg.pose.orientation.z)
    copied.pose.orientation.w = float(msg.pose.orientation.w)
    return copied


def _command_age_s(msg: PoseStamped, now_s: float) -> float:
    stamp_s = _stamp_to_seconds(msg.header.stamp)
    if stamp_s <= 0.0:
        return float("nan")
    return now_s - stamp_s


def _evaluate_command(
    msg: PoseStamped,
    *,
    enabled: bool,
    now_s: float,
    max_command_age_s: float,
) -> Tuple[bool, str, float]:
    age_s = _command_age_s(msg, now_s)
    if not enabled:
        return False, "disabled", age_s
    if not _pose_is_finite(msg):
        return False, "nonfinite_pose", age_s
    if _pose_quaternion_norm(msg) <= 1.0e-12:
        return False, "invalid_quaternion", age_s
    if max_command_age_s > 0.0 and math.isfinite(age_s) and age_s > max_command_age_s:
        return False, "stale_command", age_s
    return True, "", age_s


def _make_body_command(msg: PoseStamped, body_name: str):
    if BodyCommand is None:
        raise RuntimeError("BodyCommand message type is unavailable")
    body_command = BodyCommand()
    body_command.header.stamp = msg.header.stamp
    body_command.header.frame_id = msg.header.frame_id
    body_command.names = [body_name]
    body_command.pose = [msg.pose]
    return body_command


def _make_cartesian_motion_force_command(
    msg: PoseStamped,
    *,
    input_pose_reference: str,
    tcp_offset: Sequence[float],
):
    if CartesianMotionForceCommand is None:
        raise RuntimeError("CartesianMotionForceCommand message type is unavailable")
    command = CartesianMotionForceCommand()
    command.header.stamp = msg.header.stamp
    command.header.frame_id = msg.header.frame_id
    command.target_pose.orientation.x = float(msg.pose.orientation.x)
    command.target_pose.orientation.y = float(msg.pose.orientation.y)
    command.target_pose.orientation.z = float(msg.pose.orientation.z)
    command.target_pose.orientation.w = float(msg.pose.orientation.w)

    target_x = float(msg.pose.position.x)
    target_y = float(msg.pose.position.y)
    target_z = float(msg.pose.position.z)
    if input_pose_reference == "flange":
        offset_x, offset_y, offset_z = quat_rotate_xyzw(
            (
                command.target_pose.orientation.x,
                command.target_pose.orientation.y,
                command.target_pose.orientation.z,
                command.target_pose.orientation.w,
            ),
            tcp_offset,
        )
        target_x += offset_x
        target_y += offset_y
        target_z += offset_z

    command.target_pose.position.x = target_x
    command.target_pose.position.y = target_y
    command.target_pose.position.z = target_z
    return command


class DisplayPortFlexivTaskSpacePassthrough(Node):
    """Forward raw task-space policy targets to robot-side command topics."""

    def __init__(self):
        super().__init__("displayport_flexiv_task_space_passthrough")

        self.declare_parameter(
            "input_target_pose_topic",
            "/displayport_task_space_policy/target_pose",
        )
        self.declare_parameter("output_target_pose_topic", "/flexiv/target_pose")
        self.declare_parameter("enable_topic", "~/enable")
        self.declare_parameter("feedback_pose_topic", "")
        self.declare_parameter("end_effector_feedback_pose_topic", "")
        self.declare_parameter("enabled_on_start", False)
        self.declare_parameter("publish_pose_stamped", True)
        self.declare_parameter("publish_body_command", False)
        self.declare_parameter("body_command_topic", "/body_commands")
        self.declare_parameter("body_command_name", "flange")
        self.declare_parameter("publish_flexiv_cartesian_command", False)
        self.declare_parameter(
            "flexiv_cartesian_command_topic", "/cartesian_motion_controller/command"
        )
        self.declare_parameter("input_pose_reference", "flange")
        self.declare_parameter("pose_delta_command_topic", "")
        # Declared as a double array so the same YAML/CLI syntax works here and in
        # displayport_task_space_pose_source. Note this is the *adapter* offset --
        # the flange->Flexiv-command translation the driver applies for the active
        # tool -- which is a different frame from the policy control-frame offset
        # that the pose source declares under the same name.
        self.declare_parameter("tcp_offset", [0.0, 0.0, 0.1925])
        self.declare_parameter("auto_tcp_offset_from_robot_states", False)
        self.declare_parameter("robot_states_topic", "")
        self.declare_parameter("auto_tcp_offset_sample_count", 20)
        self.declare_parameter("auto_tcp_offset_max_translation_std_m", 0.002)
        self.declare_parameter("max_command_age_s", 0.25)
        self.declare_parameter("stale_command_timeout_s", 0.25)
        self.declare_parameter("stale_behavior", "stop_forwarding")
        self.declare_parameter("require_user_confirmation", False)
        self.declare_parameter("confirmation_requires_feedback", True)
        self.declare_parameter("require_end_effector_feedback_for_flexiv_safety", True)
        self.declare_parameter("max_flexiv_target_translation_delta_m", 0.05)
        self.declare_parameter("max_flexiv_target_downward_delta_m", 0.03)
        self.declare_parameter("csv_output_path", "")

        self.input_target_pose_topic = str(
            self.get_parameter("input_target_pose_topic").value
        ).strip()
        self.output_target_pose_topic = str(
            self.get_parameter("output_target_pose_topic").value
        ).strip()
        self.enable_topic = str(self.get_parameter("enable_topic").value).strip()
        self.feedback_pose_topic = str(
            self.get_parameter("feedback_pose_topic").value
        ).strip()
        self.end_effector_feedback_pose_topic = str(
            self.get_parameter("end_effector_feedback_pose_topic").value
        ).strip()
        self.enabled = bool(self.get_parameter("enabled_on_start").value)
        self.publish_pose_stamped = bool(self.get_parameter("publish_pose_stamped").value)
        self.publish_body_command = bool(self.get_parameter("publish_body_command").value)
        self.body_command_topic = str(self.get_parameter("body_command_topic").value).strip()
        self.body_command_name = str(self.get_parameter("body_command_name").value).strip()
        self.publish_flexiv_cartesian_command = bool(
            self.get_parameter("publish_flexiv_cartesian_command").value
        )
        self.flexiv_cartesian_command_topic = str(
            self.get_parameter("flexiv_cartesian_command_topic").value
        ).strip()
        self.input_pose_reference = str(
            self.get_parameter("input_pose_reference").value
        ).strip().lower()
        self.tcp_offset = tuple(
            as_float_list(self.get_parameter("tcp_offset").value, 3, "tcp_offset"))
        self.auto_tcp_offset_from_robot_states = bool(
            self.get_parameter("auto_tcp_offset_from_robot_states").value
        )
        self.robot_states_topic = str(
            self.get_parameter("robot_states_topic").value
        ).strip()
        self.tcp_offset_estimator = (
            TcpOffsetEstimator(
                sample_count=max(
                    1, int(self.get_parameter("auto_tcp_offset_sample_count").value)),
                max_translation_std_m=float(
                    self.get_parameter("auto_tcp_offset_max_translation_std_m").value),
            )
            if self.auto_tcp_offset_from_robot_states
            else None
        )
        self.pose_delta_command_topic = str(
            self.get_parameter("pose_delta_command_topic").value
        ).strip()
        self.max_command_age_s = float(self.get_parameter("max_command_age_s").value)
        self.stale_command_timeout_s = float(
            self.get_parameter("stale_command_timeout_s").value
        )
        self.stale_behavior = str(self.get_parameter("stale_behavior").value).strip().lower()
        self.require_user_confirmation = bool(
            self.get_parameter("require_user_confirmation").value
        )
        self.confirmation_requires_feedback = bool(
            self.get_parameter("confirmation_requires_feedback").value
        )
        self.require_end_effector_feedback_for_flexiv_safety = bool(
            self.get_parameter("require_end_effector_feedback_for_flexiv_safety").value
        )
        self.max_flexiv_target_translation_delta_m = float(
            self.get_parameter("max_flexiv_target_translation_delta_m").value
        )
        self.max_flexiv_target_downward_delta_m = float(
            self.get_parameter("max_flexiv_target_downward_delta_m").value
        )
        self.csv_output_path = str(self.get_parameter("csv_output_path").value).strip()

        if self.stale_behavior not in {"stop_forwarding", "hold_feedback_pose"}:
            raise RuntimeError(
                "stale_behavior must be one of: stop_forwarding, hold_feedback_pose"
            )
        if self.input_pose_reference not in {"flange", "tcp"}:
            raise RuntimeError("input_pose_reference must be one of: flange, tcp")
        if (
            not self.publish_pose_stamped
            and not self.publish_body_command
            and not self.publish_flexiv_cartesian_command
        ):
            raise RuntimeError("At least one output must be enabled")
        if self.publish_body_command and BodyCommand is None:
            raise RuntimeError("publish_body_command=true but BodyCommand is unavailable")
        if self.publish_flexiv_cartesian_command and CartesianMotionForceCommand is None:
            raise RuntimeError(
                "publish_flexiv_cartesian_command=true but CartesianMotionForceCommand is "
                "unavailable"
            )
        if self.auto_tcp_offset_from_robot_states:
            if RobotStates is None:
                raise RuntimeError(
                    "auto_tcp_offset_from_robot_states=true but RobotStates is unavailable"
                )
            if not self.robot_states_topic:
                raise RuntimeError(
                    "auto_tcp_offset_from_robot_states=true requires robot_states_topic"
                )
        if self.pose_delta_command_topic and CartesianPoseDeltaCommand is None:
            raise RuntimeError(
                "pose_delta_command_topic is set but CartesianPoseDeltaCommand is unavailable"
            )
        if self.stale_behavior == "hold_feedback_pose":
            # The hold pose is republished as a policy target, so it must come from
            # the topic that carries poses in input_pose_reference's frame.
            if self.input_pose_reference == "tcp":
                if not self.end_effector_feedback_pose_topic:
                    raise RuntimeError(
                        "hold_feedback_pose with input_pose_reference=tcp requires "
                        "end_effector_feedback_pose_topic"
                    )
            elif not self.feedback_pose_topic:
                raise RuntimeError("hold_feedback_pose requires feedback_pose_topic")
        if (
            self.require_user_confirmation
            and self.confirmation_requires_feedback
            and not self.feedback_pose_topic
        ):
            raise RuntimeError(
                "require_user_confirmation with confirmation_requires_feedback=true "
                "requires feedback_pose_topic"
            )

        self.command_seq = -1
        self.forwarded_seq = -1
        self.last_command_wall_time_s: Optional[float] = None
        self.latest_feedback_pose: Optional[PoseStamped] = None
        self.latest_end_effector_feedback_pose: Optional[PoseStamped] = None
        self.pending_pose_delta_commands: deque = deque(
            maxlen=MAX_PENDING_POSE_DELTA_COMMANDS)
        self.hold_sent_for_current_stale_event = False
        self.csv_file = None
        self.csv_writer = None

        if self.publish_pose_stamped:
            self.pose_pub = self.create_publisher(
                PoseStamped,
                self.output_target_pose_topic,
                10,
            )
        else:
            self.pose_pub = None

        if self.publish_body_command:
            self.body_command_pub = self.create_publisher(
                BodyCommand,
                self.body_command_topic,
                10,
            )
        else:
            self.body_command_pub = None

        if self.publish_flexiv_cartesian_command:
            self.flexiv_cartesian_command_pub = self.create_publisher(
                CartesianMotionForceCommand,
                self.flexiv_cartesian_command_topic,
                10,
            )
        else:
            self.flexiv_cartesian_command_pub = None

        target_subscription_depth = 1 if self.require_user_confirmation else 10
        self.create_subscription(
            PoseStamped,
            self.input_target_pose_topic,
            self._target_cb,
            target_subscription_depth,
        )
        if self.enable_topic:
            self.create_subscription(Bool, self.enable_topic, self._enable_cb, 10)
        if self.feedback_pose_topic:
            self.create_subscription(PoseStamped, self.feedback_pose_topic, self._feedback_cb, 10)
        if self.end_effector_feedback_pose_topic:
            self.create_subscription(
                PoseStamped,
                self.end_effector_feedback_pose_topic,
                self._end_effector_feedback_cb,
                10,
            )
        if self.tcp_offset_estimator is not None:
            self.create_subscription(
                RobotStates,
                self.robot_states_topic,
                self._robot_states_cb,
                10,
            )
        if self.pose_delta_command_topic:
            self.create_subscription(
                CartesianPoseDeltaCommand,
                self.pose_delta_command_topic,
                self._pose_delta_command_cb,
                10,
            )

        if self.csv_output_path:
            csv_path = Path(os.path.expanduser(os.path.expandvars(self.csv_output_path)))
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = csv_path.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=[
                    "command_seq",
                    "event",
                    "wall_time_s",
                    "received_stamp_s",
                    "command_age_s",
                    "enabled",
                    "forwarded",
                    "forwarded_pose_stamped",
                    "forwarded_body_command",
                    "forwarded_flexiv_cartesian_command",
                    "reject_reason",
                    "input_frame",
                    "output_pose_topic",
                    "body_command_topic",
                    "body_command_name",
                    "flexiv_cartesian_command_topic",
                    "input_pose_reference",
                    "stale_behavior",
                    "max_command_age_s",
                    "stale_command_timeout_s",
                    *[f"target_pos_{axis}" for axis in XYZ_NAMES],
                    *[f"target_{name}" for name in QUAT_NAMES],
                    *[f"feedback_pos_{axis}" for axis in XYZ_NAMES],
                    *[f"feedback_{name}" for name in QUAT_NAMES],
                ],
            )
            self.csv_writer.writeheader()
            self.csv_file.flush()

        self.stale_timer = self.create_timer(0.02, self._stale_timer_cb)
        flexiv_cartesian_output = (
            self.flexiv_cartesian_command_topic if self.flexiv_cartesian_command_pub
            else "<disabled>"
        )
        self.get_logger().info(
            "DisplayPort Flexiv task-space passthrough ready: "
            f"input={self.input_target_pose_topic}, "
            f"pose_output={self.output_target_pose_topic if self.pose_pub else '<disabled>'}, "
            f"body_output={self.body_command_topic if self.body_command_pub else '<disabled>'}, "
            f"flexiv_cartesian_output={flexiv_cartesian_output}, "
            f"input_pose_reference={self.input_pose_reference}, "
            f"tcp_offset={self.tcp_offset}, "
            "auto_tcp_offset_from_robot_states="
            f"{self.auto_tcp_offset_from_robot_states}, "
            f"robot_states_topic={self.robot_states_topic or '<disabled>'}, "
            f"eef_feedback={self.end_effector_feedback_pose_topic or '<disabled>'}, "
            f"enabled={self.enabled}, stale_behavior={self.stale_behavior}, "
            f"require_user_confirmation={self.require_user_confirmation}, "
            "max_flexiv_target_translation_delta_m="
            f"{self.max_flexiv_target_translation_delta_m}, "
            "max_flexiv_target_downward_delta_m="
            f"{self.max_flexiv_target_downward_delta_m}"
        )

    def _now_s(self) -> float:
        return _stamp_to_seconds(self.get_clock().now().to_msg())

    def _enable_cb(self, msg: Bool) -> None:
        was_enabled = bool(self.enabled)
        self.enabled = bool(msg.data)
        self.hold_sent_for_current_stale_event = False
        if self.enabled != was_enabled:
            self.last_command_wall_time_s = None
        self.get_logger().info(f"Policy target passthrough enabled={self.enabled}")

    def _feedback_cb(self, msg: PoseStamped) -> None:
        self.latest_feedback_pose = _copy_pose_stamped(msg)

    def _end_effector_feedback_cb(self, msg: PoseStamped) -> None:
        self.latest_end_effector_feedback_pose = _copy_pose_stamped(msg)

    def _robot_states_cb(self, msg: RobotStates) -> None:
        estimator = self.tcp_offset_estimator
        if estimator is None or estimator.ready:
            return
        flange_pos, flange_quat = pose_to_arrays(msg.flange_pose)
        tcp_pos, tcp_quat = pose_to_arrays(msg.tcp_pose)
        try:
            became_ready = estimator.add_sample(
                flange_pos, flange_quat, tcp_pos, tcp_quat)
        except ValueError as exc:
            self.get_logger().warning(
                f"Ignoring RobotStates TCP offset sample: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        collected = estimator.collected
        if collected == 1 or collected % 5 == 0 or became_ready:
            latest_offset = estimator.offset_samples[-1]
            self.get_logger().info(
                "TCP offset sample from RobotStates: "
                f"sample={collected}/{estimator.sample_count}, "
                f"latest_tcp_offset_in_flange=[{latest_offset[0]:.6f}, "
                f"{latest_offset[1]:.6f}, {latest_offset[2]:.6f}] m"
            )
        if not became_ready:
            return

        self.tcp_offset = estimator.offset
        std = estimator.translation_std
        quat_offset = estimator.quat_offset
        self.get_logger().info(
            "Measured TCP offset from RobotStates without publishing TF: "
            f"topic={self.robot_states_topic}, samples={collected}, "
            f"tcp_offset_in_flange=[{self.tcp_offset[0]:.6f}, "
            f"{self.tcp_offset[1]:.6f}, {self.tcp_offset[2]:.6f}] m, "
            f"sample_std=[{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}] m, "
            f"flange_to_tcp_quat_xyzw=[{quat_offset[0]:.6f}, {quat_offset[1]:.6f}, "
            f"{quat_offset[2]:.6f}, {quat_offset[3]:.6f}]"
        )
        if estimator.translation_std_exceeded:
            self.get_logger().warning(
                "Measured TCP offset samples varied more than expected: "
                f"max_std={max(std):.6f} m > "
                f"{estimator.max_translation_std_m:.6f} m. "
                "The mean offset was still applied; make sure the robot-state topic is sane."
            )

    def _pose_delta_command_cb(self, msg) -> None:
        """Queue the policy's raw action for the interactive confirmation summary.

        Commands are consumed one per target pose. A single slot would let a
        second command overwrite the first, so the summary would describe a
        different action than the target being approved.
        """
        self.pending_pose_delta_commands.append(msg)

    def _target_cb(self, msg: PoseStamped) -> None:
        self.command_seq += 1
        now_s = self._now_s()
        self.last_command_wall_time_s = now_s
        self.hold_sent_for_current_stale_event = False
        should_forward, reject_reason, age_s = _evaluate_command(
            msg,
            enabled=self.enabled,
            now_s=now_s,
            max_command_age_s=self.max_command_age_s,
        )
        forwarded_pose = False
        forwarded_body = False
        forwarded_flexiv = False

        if (
            should_forward
            and self.tcp_offset_estimator is not None
            and not self.tcp_offset_estimator.ready
            and self.publish_flexiv_cartesian_command
            and self.input_pose_reference == "flange"
        ):
            should_forward = False
            reject_reason = "waiting_for_robot_states_tcp_offset"

        flexiv_command = None
        if should_forward and self.flexiv_cartesian_command_pub is not None:
            flexiv_command = _make_cartesian_motion_force_command(
                msg,
                input_pose_reference=self.input_pose_reference,
                tcp_offset=self.tcp_offset,
            )
            should_forward, reject_reason = self._evaluate_flexiv_command_safety(
                flexiv_command
            )

        if should_forward:
            confirmed, confirmation_reject_reason = self._confirm_command(
                msg,
                flexiv_command,
            )
            if confirmed:
                forwarded_pose, forwarded_body, forwarded_flexiv = self._publish_command(
                    msg,
                    flexiv_command,
                )
                self.forwarded_seq = self.command_seq
            else:
                reject_reason = confirmation_reject_reason
        else:
            self.get_logger().warning(
                f"Dropping policy target seq={self.command_seq}: {reject_reason}",
                throttle_duration_sec=1.0,
            )

        self._write_csv_row(
            event="target",
            command_seq=self.command_seq,
            msg=msg,
            now_s=now_s,
            age_s=age_s,
            forwarded_pose=forwarded_pose,
            forwarded_body=forwarded_body,
            forwarded_flexiv=forwarded_flexiv,
            reject_reason=reject_reason,
        )

    def _evaluate_flexiv_command_safety(
        self,
        flexiv_command: CartesianMotionForceCommand,
    ) -> Tuple[bool, str]:
        max_translation = float(self.max_flexiv_target_translation_delta_m)
        max_downward = float(self.max_flexiv_target_downward_delta_m)
        if max_translation <= 0.0 and max_downward <= 0.0:
            return True, ""

        if self.latest_end_effector_feedback_pose is None:
            if self.require_end_effector_feedback_for_flexiv_safety:
                return False, "missing_end_effector_feedback_pose_for_flexiv_safety"
            return True, ""

        target = flexiv_command.target_pose.position
        source = self.latest_end_effector_feedback_pose.pose.position
        dx = float(target.x) - float(source.x)
        dy = float(target.y) - float(source.y)
        dz = float(target.z) - float(source.z)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if max_downward > 0.0 and dz < -max_downward:
            self.get_logger().warning(
                "Rejecting Flexiv Cartesian command: target TCP drops "
                f"{-dz:.6f} m from feedback, limit={max_downward:.6f} m",
                throttle_duration_sec=1.0,
            )
            return False, "flexiv_target_downward_delta_exceeded"
        if max_translation > 0.0 and distance > max_translation:
            self.get_logger().warning(
                "Rejecting Flexiv Cartesian command: target TCP translation jump "
                f"{distance:.6f} m from feedback, limit={max_translation:.6f} m",
                throttle_duration_sec=1.0,
            )
            return False, "flexiv_target_translation_delta_exceeded"
        return True, ""

    def _confirm_command(
        self,
        msg: PoseStamped,
        flexiv_command: Optional[CartesianMotionForceCommand] = None,
    ) -> Tuple[bool, str]:
        if not self.require_user_confirmation:
            return True, ""
        if self.confirmation_requires_feedback and self.latest_feedback_pose is None:
            self.get_logger().warning(
                "Skipping policy target because user confirmation requires feedback "
                f"but no message has been received on {self.feedback_pose_topic!r}.",
                throttle_duration_sec=1.0,
            )
            return False, "missing_feedback_pose"

        if self.flexiv_cartesian_command_pub is not None:
            if flexiv_command is None:
                flexiv_command = _make_cartesian_motion_force_command(
                    msg,
                    input_pose_reference=self.input_pose_reference,
                    tcp_offset=self.tcp_offset,
                )
            if (self.confirmation_requires_feedback
                    and self.latest_end_effector_feedback_pose is None):
                self.get_logger().warning(
                    "Skipping policy target because user confirmation requires TCP feedback "
                    f"but no message has been received on "
                    f"{self.end_effector_feedback_pose_topic!r}.",
                    throttle_duration_sec=1.0,
                )
                return False, "missing_end_effector_feedback_pose"

        self._print_confirmation_summary(msg, flexiv_command)
        answer = self._read_confirmation()
        if answer is None:
            return False, "confirmation_io_error"
        if answer == "":
            return True, ""
        if answer in {"q", "quit", "disable"}:
            self.enabled = False
            self.get_logger().warning(
                "User declined command and disabled passthrough. Re-enable with the enable topic."
            )
            return False, "user_disabled_passthrough"
        if answer in {"n", "no", "skip"}:
            return False, "user_rejected"
        return False, "user_rejected"

    def _print_confirmation_summary(
        self,
        msg: PoseStamped,
        flexiv_command: Optional[CartesianMotionForceCommand],
    ) -> None:
        print("", flush=True)
        print("--- Candidate DisplayPort task-space command ---", flush=True)
        if flexiv_command is not None:
            print(
                f"Flexiv Cartesian command topic: {self.flexiv_cartesian_command_topic}",
                flush=True,
            )
            if self.latest_end_effector_feedback_pose is None:
                print(
                    "Source TCP feedback "
                    f"({self.end_effector_feedback_pose_topic or '<none>'}): unavailable",
                    flush=True,
                )
            else:
                self._print_pose(
                    "Source TCP feedback "
                    f"({self.end_effector_feedback_pose_topic})",
                    self.latest_end_effector_feedback_pose,
                )
                self._print_position_delta(
                    "Goal TCP minus source TCP",
                    flexiv_command.target_pose,
                    self.latest_end_effector_feedback_pose.pose,
                )
            self._print_raw_pose("Goal TCP command pose", flexiv_command.target_pose)
        else:
            self._print_pose("Goal pose", msg)
            if self.latest_feedback_pose is not None:
                self._print_pose(
                    f"Source feedback ({self.feedback_pose_topic})",
                    self.latest_feedback_pose,
                )
                self._print_position_delta(
                    "Goal minus source feedback",
                    msg.pose,
                    self.latest_feedback_pose.pose,
                )
        self._print_blend_metadata(msg)
        print("Press Enter to publish, type 'n' to skip, or 'q' to disable.", flush=True)

    def _print_blend_metadata(self, msg: PoseStamped) -> None:
        """Report the policy's raw action next to the motion actually commanded.

        The decoder scales the policy action before turning it into a target, so
        the ratio between the commanded motion and the raw action is the blend
        ratio in effect. Reading the raw action from the pose-delta command topic
        keeps that check independent of how the decoder was configured.
        """
        command = (
            self.pending_pose_delta_commands.popleft()
            if self.pending_pose_delta_commands
            else None
        )
        if command is None:
            print("Policy pose-delta command: unavailable", flush=True)
            return

        raw_delta = (
            float(command.delta_position.x),
            float(command.delta_position.y),
            float(command.delta_position.z),
        )
        raw_rot = (
            float(command.delta_axis_angle.x),
            float(command.delta_axis_angle.y),
            float(command.delta_axis_angle.z),
        )
        observation = command.observation_pose
        commanded_delta = (
            float(msg.pose.position.x) - float(observation.position.x),
            float(msg.pose.position.y) - float(observation.position.y),
            float(msg.pose.position.z) - float(observation.position.z),
        )

        print("Policy pose-delta command:", flush=True)
        raw_norm = math.sqrt(sum(value * value for value in raw_delta))
        commanded_norm = math.sqrt(sum(value * value for value in commanded_delta))
        print(
            "  raw policy translation delta = "
            f"[{raw_delta[0]:+.6f}, {raw_delta[1]:+.6f}, {raw_delta[2]:+.6f}] m, "
            f"norm={raw_norm:.6f} m",
            flush=True,
        )
        print(
            "  commanded translation delta (target minus observation) = "
            f"[{commanded_delta[0]:+.6f}, {commanded_delta[1]:+.6f}, "
            f"{commanded_delta[2]:+.6f}] m, norm={commanded_norm:.6f} m",
            flush=True,
        )
        if raw_norm > 1.0e-12:
            print(
                f"  implied translation blend ratio = {commanded_norm / raw_norm:.9f}",
                flush=True,
            )
        raw_rot_norm = math.sqrt(sum(value * value for value in raw_rot))
        print(
            "  raw policy angular delta = "
            f"[{raw_rot[0]:+.6f}, {raw_rot[1]:+.6f}, {raw_rot[2]:+.6f}] rad, "
            f"norm={raw_rot_norm:.6f} rad",
            flush=True,
        )
        print(
            "  observation pose = "
            f"[{observation.position.x:+.6f}, {observation.position.y:+.6f}, "
            f"{observation.position.z:+.6f}] m in {command.header.frame_id!r}",
            flush=True,
        )

    def _print_position_delta(self, label: str, target_pose, source_pose) -> None:
        dx = float(target_pose.position.x) - float(source_pose.position.x)
        dy = float(target_pose.position.y) - float(source_pose.position.y)
        dz = float(target_pose.position.z) - float(source_pose.position.z)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        print(
            f"{label} = [{dx:+.6f}, {dy:+.6f}, {dz:+.6f}] m, "
            f"norm={dist:.6f} m",
            flush=True,
        )

    def _print_pose(self, label: str, msg: PoseStamped) -> None:
        self._print_raw_pose(label, msg.pose)

    def _print_raw_pose(self, label: str, pose) -> None:
        print(f"{label}:", flush=True)
        print(
            "  pos      = "
            f"[{pose.position.x:.6f}, {pose.position.y:.6f}, {pose.position.z:.6f}]",
            flush=True,
        )
        print(
            "  quat xyzw= "
            f"[{pose.orientation.x:.6f}, {pose.orientation.y:.6f}, "
            f"{pose.orientation.z:.6f}, {pose.orientation.w:.6f}]",
            flush=True,
        )

    def _read_confirmation(self) -> Optional[str]:
        """Return the operator's answer, or None if the prompt could not be read.

        An empty string means the operator pressed Enter to approve, so a failed
        read must not return one: with no tty the command would be approved while
        the log says it was skipped.
        """
        prompt = "Press Enter to publish this command: "
        try:
            with open("/dev/tty", "r", encoding="utf-8") as tty:
                print(prompt, end="", flush=True)
                return tty.readline().strip().lower()
        except OSError as exc:
            self.get_logger().error(
                f"Unable to read confirmation from /dev/tty: {exc}. Command skipped."
            )
            return None

    def _publish_command(
        self,
        msg: PoseStamped,
        flexiv_command: Optional[CartesianMotionForceCommand] = None,
    ) -> Tuple[bool, bool, bool]:
        output_msg = _copy_pose_stamped(msg)
        forwarded_pose = False
        forwarded_body = False
        forwarded_flexiv = False
        if self.pose_pub is not None:
            self.pose_pub.publish(output_msg)
            forwarded_pose = True
        if self.body_command_pub is not None:
            self.body_command_pub.publish(_make_body_command(output_msg, self.body_command_name))
            forwarded_body = True
        if self.flexiv_cartesian_command_pub is not None:
            if flexiv_command is None:
                flexiv_command = _make_cartesian_motion_force_command(
                    output_msg,
                    input_pose_reference=self.input_pose_reference,
                    tcp_offset=self.tcp_offset,
                )
            self.flexiv_cartesian_command_pub.publish(flexiv_command)
            forwarded_flexiv = True
        return forwarded_pose, forwarded_body, forwarded_flexiv

    def _stale_timer_cb(self) -> None:
        if (
            not self.enabled
            or self.stale_command_timeout_s <= 0.0
            or self.last_command_wall_time_s is None
        ):
            return
        now_s = self._now_s()
        if now_s - self.last_command_wall_time_s <= self.stale_command_timeout_s:
            return
        if self.hold_sent_for_current_stale_event:
            return

        self.hold_sent_for_current_stale_event = True
        if self.stale_behavior == "stop_forwarding":
            self._write_csv_row(
                event="stale_stop_forwarding",
                command_seq=self.command_seq,
                msg=None,
                now_s=now_s,
                age_s=now_s - self.last_command_wall_time_s,
                forwarded_pose=False,
                forwarded_body=False,
                forwarded_flexiv=False,
                reject_reason="stale_timeout",
            )
            return

        # The hold pose is republished as a policy target, so it has to be in the
        # frame _publish_command() expects. With input_pose_reference="tcp" the
        # flange feedback pose would be forwarded as a TCP target and the hold
        # would be displaced by the whole tool offset.
        if self.input_pose_reference == "tcp":
            hold_source = self.latest_end_effector_feedback_pose
            missing_reason = "missing_end_effector_feedback_pose"
        else:
            hold_source = self.latest_feedback_pose
            missing_reason = "missing_feedback_pose"

        if hold_source is None:
            self._write_csv_row(
                event="stale_hold_missing_feedback",
                command_seq=self.command_seq,
                msg=None,
                now_s=now_s,
                age_s=now_s - self.last_command_wall_time_s,
                forwarded_pose=False,
                forwarded_body=False,
                forwarded_flexiv=False,
                reject_reason=missing_reason,
            )
            return

        hold_msg = _copy_pose_stamped(hold_source)
        hold_msg.header.stamp = self.get_clock().now().to_msg()
        forwarded_pose, forwarded_body, forwarded_flexiv = self._publish_command(hold_msg)
        self._write_csv_row(
            event="stale_hold_feedback_pose",
            command_seq=self.command_seq,
            msg=hold_msg,
            now_s=now_s,
            age_s=now_s - self.last_command_wall_time_s,
            forwarded_pose=forwarded_pose,
            forwarded_body=forwarded_body,
            forwarded_flexiv=forwarded_flexiv,
            reject_reason="stale_timeout",
        )

    def _feedback_values(self) -> Dict[str, str]:
        if self.latest_feedback_pose is None:
            return {
                **{f"feedback_pos_{axis}": "nan" for axis in XYZ_NAMES},
                **{f"feedback_{name}": "nan" for name in QUAT_NAMES},
            }
        values = _pose_values(self.latest_feedback_pose)
        return {
            **{
                f"feedback_pos_{axis}": _format_float(value)
                for axis, value in zip(XYZ_NAMES, values[:3])
            },
            **{
                f"feedback_{name}": _format_float(value)
                for name, value in zip(QUAT_NAMES, values[3:])
            },
        }

    def _write_csv_row(
        self,
        *,
        event: str,
        command_seq: int,
        msg: Optional[PoseStamped],
        now_s: float,
        age_s: float,
        forwarded_pose: bool,
        forwarded_body: bool,
        forwarded_flexiv: bool,
        reject_reason: str,
    ) -> None:
        if self.csv_writer is None:
            return
        if msg is None:
            target_values = [float("nan")] * 7
            received_stamp_s = float("nan")
            input_frame = ""
        else:
            target_values = list(_pose_values(msg))
            received_stamp_s = _stamp_to_seconds(msg.header.stamp)
            input_frame = msg.header.frame_id
        row = {
            "command_seq": str(command_seq),
            "event": event,
            "wall_time_s": _format_float(now_s),
            "received_stamp_s": _format_float(received_stamp_s),
            "command_age_s": _format_float(age_s),
            "enabled": "1" if self.enabled else "0",
            "forwarded": "1" if forwarded_pose or forwarded_body or forwarded_flexiv else "0",
            "forwarded_pose_stamped": "1" if forwarded_pose else "0",
            "forwarded_body_command": "1" if forwarded_body else "0",
            "forwarded_flexiv_cartesian_command": "1" if forwarded_flexiv else "0",
            "reject_reason": reject_reason,
            "input_frame": input_frame,
            "output_pose_topic": self.output_target_pose_topic,
            "body_command_topic": self.body_command_topic,
            "body_command_name": self.body_command_name,
            "flexiv_cartesian_command_topic": self.flexiv_cartesian_command_topic,
            "input_pose_reference": self.input_pose_reference,
            "stale_behavior": self.stale_behavior,
            "max_command_age_s": _format_float(self.max_command_age_s),
            "stale_command_timeout_s": _format_float(self.stale_command_timeout_s),
            **{
                f"target_pos_{axis}": _format_float(value)
                for axis, value in zip(XYZ_NAMES, target_values[:3])
            },
            **{
                f"target_{name}": _format_float(value)
                for name, value in zip(QUAT_NAMES, target_values[3:])
            },
            **self._feedback_values(),
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = DisplayPortFlexivTaskSpacePassthrough()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

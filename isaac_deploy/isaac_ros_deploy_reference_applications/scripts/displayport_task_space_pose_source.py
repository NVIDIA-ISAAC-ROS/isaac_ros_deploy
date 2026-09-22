#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish DisplayPort task-space pose sources for InputBuilderNode.

The DisplayPort policy observes the TCP/eef pose and socket keypoint pose. Isaac
Sim and the Flexiv driver both report the live flange/control-frame pose, so this
node applies the TCP offset and publishes the two PoseStamped ROS topics consumed
by InputBuilderNode. The launch file maps the four logical InputBuilder sources
to these topics:

* ``eef_pose_pos`` and ``eef_pose_rot6d`` read from the ``eef_pose`` topic.
* ``socket_kp_pose_pos`` and ``socket_kp_pose_rot6d`` read from the
  ``socket_kp_pose`` topic.

It intentionally does not build TensorList messages. Generic ROS-to-tensor
conversion, recurrent feedback tensors, and model input ordering are handled by
InputBuilderNode from the LEAPP export metadata.

Beyond the Isaac Sim path it also owns the observation-side pieces a real robot
needs, all of which are off by default:

* ``socket_pose_is_keypoint`` / ``socket_root_to_keypoint_offset`` advance a
  socket root pose to the insertion keypoint the policy was trained against.
* ``socket_pose_timeout_s`` stops publishing observations on a stale socket pose
  rather than driving the policy from an old target.
* ``enable_topic`` gates publishing, so an orchestrator can hold the policy
  closed until the robot is ready for it.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

from geometry_msgs.msg import Pose, PoseStamped, Vector3Stamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_space_pose_math import (  # noqa: E402  (needs the sys.path entry above)
    as_float_list,
    fill_pose,
    pose_to_arrays,
    quat_normalize_xyzw,
    quat_rotate_xyzw,
)


Vector3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

SOURCE_POSE_REFERENCES = ("flange", "tcp", "eef")


class DisplayPortTaskSpacePoseSource(Node):
    """Publish task-space PoseStamped sources from robot flange/socket poses."""

    def __init__(self):
        super().__init__("displayport_task_space_pose_source")

        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("eef_pose_topic", "eef_pose")
        self.declare_parameter("socket_kp_pose_topic", "socket_kp_pose")
        self.declare_parameter("flange_pose_topic", "")
        self.declare_parameter("socket_pose_topic", "/displayport_insertion/goal_pose")
        self.declare_parameter("base_frame", "world")
        self.declare_parameter("source_pose_reference", "flange")
        self.declare_parameter("tcp_offset", [0.0, 0.0, 0.15])
        self.declare_parameter("tcp_offset_topic", "tcp_offset")
        self.declare_parameter("socket_kp_position", [0.475, 0.125, 0.060])
        self.declare_parameter("socket_kp_quaternion_xyzw", [-0.5, -0.5, -0.5, 0.5])
        self.declare_parameter("socket_pose_is_keypoint", True)
        self.declare_parameter("socket_root_to_keypoint_offset", [0.0375, 0.0, 0.0])
        self.declare_parameter("require_socket_pose", False)
        self.declare_parameter("socket_pose_timeout_s", 0.0)
        self.declare_parameter("enable_topic", "")
        self.declare_parameter("enabled_on_start", True)

        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.eef_pose_topic = str(self.get_parameter("eef_pose_topic").value)
        self.socket_kp_pose_topic = str(self.get_parameter("socket_kp_pose_topic").value)
        self.flange_pose_topic = str(self.get_parameter("flange_pose_topic").value).strip()
        self.socket_pose_topic = str(self.get_parameter("socket_pose_topic").value).strip()
        self.base_frame = str(self.get_parameter("base_frame").value).strip()
        self.source_pose_reference = (
            str(self.get_parameter("source_pose_reference").value).strip().lower()
        )
        self.require_socket_pose = bool(self.get_parameter("require_socket_pose").value)
        self.socket_pose_timeout_s = float(self.get_parameter("socket_pose_timeout_s").value)
        self.socket_pose_is_keypoint = bool(
            self.get_parameter("socket_pose_is_keypoint").value)
        self.enable_topic = str(self.get_parameter("enable_topic").value).strip()
        self.enabled = bool(self.get_parameter("enabled_on_start").value)
        if not self.flange_pose_topic:
            raise RuntimeError(
                "flange_pose_topic is required for task-space observations; "
                "reading flange observations from /tf is not supported"
            )
        if self.publish_rate <= 0.0:
            raise RuntimeError("publish_rate must be positive")
        if self.source_pose_reference not in SOURCE_POSE_REFERENCES:
            raise RuntimeError(
                "source_pose_reference must be one of: "
                + ", ".join(SOURCE_POSE_REFERENCES)
            )

        self.tcp_offset = tuple(
            as_float_list(self.get_parameter("tcp_offset").value, 3, "tcp_offset")
        )
        self.socket_root_to_keypoint_offset = tuple(
            as_float_list(
                self.get_parameter("socket_root_to_keypoint_offset").value,
                3,
                "socket_root_to_keypoint_offset",
            )
        )
        self.default_socket_pos = tuple(
            as_float_list(
                self.get_parameter("socket_kp_position").value, 3, "socket_kp_position"
            )
        )
        self.default_socket_quat = quat_normalize_xyzw(
            as_float_list(
                self.get_parameter("socket_kp_quaternion_xyzw").value,
                4,
                "socket_kp_quaternion_xyzw",
            )
        )

        self.latest_flange_pose: Optional[Tuple[Vector3, Quat]] = None
        self.latest_socket_pose: Tuple[Vector3, Quat] = (
            self.default_socket_pos,
            self.default_socket_quat,
        )
        self.socket_pose_seen = False
        self.latest_socket_pose_receive_s: Optional[float] = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # The offset is written once and read by a node that may start later, so
        # it has to be latched rather than best effort.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.eef_pose_pub = self.create_publisher(PoseStamped, self.eef_pose_topic, 10)
        self.socket_kp_pose_pub = self.create_publisher(
            PoseStamped,
            self.socket_kp_pose_topic,
            10,
        )
        self.tcp_offset_pub = None
        tcp_offset_topic = str(self.get_parameter("tcp_offset_topic").value).strip()
        if tcp_offset_topic:
            self.tcp_offset_pub = self.create_publisher(
                Vector3Stamped, tcp_offset_topic, latched_qos)
        self.create_subscription(
            PoseStamped,
            self.flange_pose_topic,
            self._flange_pose_cb,
            qos,
        )
        if self.socket_pose_topic:
            self.create_subscription(
                PoseStamped,
                self.socket_pose_topic,
                self._socket_pose_cb,
                qos,
            )
        if self.enable_topic:
            self.create_subscription(Bool, self.enable_topic, self._enable_cb, 10)

        self._publish_tcp_offset()

        self.timer = self.create_timer(1.0 / self.publish_rate, self._timer_cb)
        self.get_logger().info(
            "DisplayPort task-space pose source ready: "
            f"eef_pose_topic={self.eef_pose_topic}, "
            f"socket_kp_pose_topic={self.socket_kp_pose_topic}, "
            f"flange_pose_topic={self.flange_pose_topic}, "
            f"source_pose_reference={self.source_pose_reference}, "
            f"tcp_offset={list(self.tcp_offset)}, "
            f"socket_pose_is_keypoint={self.socket_pose_is_keypoint}, "
            f"socket_pose_timeout_s={self.socket_pose_timeout_s}, "
            f"enable_topic={self.enable_topic or '<always enabled>'}, "
            f"enabled={self.enabled}"
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    def _enable_cb(self, msg: Bool) -> None:
        enabled = bool(msg.data)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        # Tracked for observability only: observation output is no longer gated, so
        # the downstream input builder can activate before the gate opens. See
        # _timer_cb.
        self.get_logger().info(
            f"Task-space pose source gate {self.enable_topic!r} reports "
            f"{'enabled' if enabled else 'disabled'}; observation output continues"
        )

    def _flange_pose_cb(self, msg: PoseStamped) -> None:
        self.latest_flange_pose = pose_to_arrays(msg)

    def _socket_pose_cb(self, msg: PoseStamped) -> None:
        pos, quat = pose_to_arrays(msg)
        if not self.socket_pose_is_keypoint:
            # The tracked socket pose is the connector root; the policy observes
            # the insertion keypoint a fixed distance into the connector.
            rotated = quat_rotate_xyzw(quat, self.socket_root_to_keypoint_offset)
            pos = (pos[0] + rotated[0], pos[1] + rotated[1], pos[2] + rotated[2])
        self.latest_socket_pose = (pos, quat)
        self.socket_pose_seen = True
        self.latest_socket_pose_receive_s = self._now_s()

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _now_s(self) -> float:
        now = self.get_clock().now().to_msg()
        return float(now.sec) + float(now.nanosec) * 1e-9

    def _publish_tcp_offset(self) -> None:
        if self.tcp_offset_pub is None:
            return
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "flange"
        msg.vector.x = float(self.tcp_offset[0])
        msg.vector.y = float(self.tcp_offset[1])
        msg.vector.z = float(self.tcp_offset[2])
        self.tcp_offset_pub.publish(msg)

    def _socket_pose_is_fresh(self) -> bool:
        if self.socket_pose_timeout_s <= 0.0 or not self.socket_pose_seen:
            return True
        if self.latest_socket_pose_receive_s is None:
            return False
        age_s = self._now_s() - self.latest_socket_pose_receive_s
        if age_s <= self.socket_pose_timeout_s:
            return True
        self.get_logger().warning(
            f"Socket pose on {self.socket_pose_topic!r} is stale: "
            f"age={age_s:.3f}s > timeout={self.socket_pose_timeout_s:.3f}s; "
            "pausing task-space observation output",
            throttle_duration_sec=1.0,
        )
        return False

    def _build_eef_pose(self) -> Optional[Pose]:
        flange_pose = self.latest_flange_pose
        if flange_pose is None:
            self.get_logger().warning(
                f"Waiting for flange pose on {self.flange_pose_topic!r}",
                throttle_duration_sec=2.0,
            )
            return None
        if self.require_socket_pose and not self.socket_pose_seen:
            self.get_logger().warning(
                f"Waiting for socket pose on {self.socket_pose_topic!r}",
                throttle_duration_sec=2.0,
            )
            return None
        if not self._socket_pose_is_fresh():
            return None

        source_pos, source_quat = flange_pose
        if self.source_pose_reference == "flange":
            rotated_offset = quat_rotate_xyzw(source_quat, self.tcp_offset)
            eef_pos = (
                source_pos[0] + rotated_offset[0],
                source_pos[1] + rotated_offset[1],
                source_pos[2] + rotated_offset[2],
            )
        else:
            # The source topic already carries the control-frame pose.
            eef_pos = source_pos
        return fill_pose(Pose(), eef_pos, source_quat)

    def _timer_cb(self) -> None:
        # Observations are published regardless of the enable gate. They only feed
        # the input builder, whose own publication stays gated, so nothing here can
        # run the model or command the robot. Publishing while disabled lets the
        # input builder activate before a trial opens the gate, and keeps eef_pose
        # current so a new trial cannot start on the previous trial's final pose.
        # _build_eef_pose() still withholds output while the socket pose is stale.

        eef_pose = self._build_eef_pose()
        if eef_pose is None:
            return

        stamp = self.get_clock().now().to_msg()

        eef_msg = PoseStamped()
        eef_msg.header.stamp = stamp
        eef_msg.header.frame_id = self.base_frame
        eef_msg.pose = eef_pose
        self.eef_pose_pub.publish(eef_msg)

        socket_pos, socket_quat = self.latest_socket_pose
        socket_msg = PoseStamped()
        socket_msg.header.stamp = stamp
        socket_msg.header.frame_id = self.base_frame
        fill_pose(socket_msg.pose, socket_pos, socket_quat)
        self.socket_kp_pose_pub.publish(socket_msg)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = DisplayPortTaskSpacePoseSource()
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

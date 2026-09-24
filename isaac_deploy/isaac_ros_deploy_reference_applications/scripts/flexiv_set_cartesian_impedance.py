#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Set Flexiv Cartesian impedance once and exit."""

import re
from typing import List

try:
    from flexiv_msgs.srv import SetCartesianImpedance
except ImportError:  # pragma: no cover - optional outside the Flexiv workspace
    SetCartesianImpedance = None
import rclpy
from rclpy.node import Node


def _parse_float_list(value, *, name: str, expected_len: int) -> List[float]:
    if isinstance(value, (list, tuple)):
        values = [float(v) for v in value]
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        values = [
            float(token)
            for token in re.split(r"[\s,]+", text)
            if token
        ]

    if len(values) != expected_len:
        raise ValueError(
            f"{name} must contain {expected_len} values; got {len(values)}: {values}"
        )
    return values


class FlexivSetCartesianImpedance(Node):
    """Call the Flexiv set_cartesian_impedance service once."""

    def __init__(self) -> None:
        super().__init__("flexiv_set_cartesian_impedance")

        self.declare_parameter("service_name", "")
        self.declare_parameter("stiffness_mode", "nominal_scaled")
        self.declare_parameter(
            "stiffness",
            "2000.0,2000.0,2000.0,300.0,300.0,300.0",
        )
        self.declare_parameter(
            "nominal_stiffness",
            "10000.0,10000.0,10000.0,1500.0,1500.0,1500.0",
        )
        self.declare_parameter("stiffness_scale", 0.3)
        self.declare_parameter(
            "damping_ratio",
            "0.7,0.7,0.7,0.7,0.7,0.7",
        )
        self.declare_parameter("service_timeout_sec", 10.0)

        self.service_name = str(self.get_parameter("service_name").value).strip()
        if not self.service_name:
            raise ValueError(
                "service_name is required, for example "
                "'/<robot_sn>_flexiv_hardware_services/set_cartesian_impedance'"
            )
        self.stiffness_mode = str(
            self.get_parameter("stiffness_mode").value
        ).strip().lower()
        self.explicit_stiffness = _parse_float_list(
            self.get_parameter("stiffness").value,
            name="stiffness",
            expected_len=6,
        )
        self.nominal_stiffness = _parse_float_list(
            self.get_parameter("nominal_stiffness").value,
            name="nominal_stiffness",
            expected_len=6,
        )
        self.stiffness_scale = float(self.get_parameter("stiffness_scale").value)
        if self.stiffness_mode not in {"explicit", "nominal_scaled"}:
            raise ValueError("stiffness_mode must be one of: explicit, nominal_scaled")
        if self.stiffness_scale < 0.0:
            raise ValueError("stiffness_scale must be non-negative")
        self.stiffness = self._resolve_stiffness()
        self.damping_ratio = _parse_float_list(
            self.get_parameter("damping_ratio").value,
            name="damping_ratio",
            expected_len=6,
        )
        self.service_timeout_sec = float(
            self.get_parameter("service_timeout_sec").value
        )

        self.client = self.create_client(SetCartesianImpedance, self.service_name)

    def _resolve_stiffness(self) -> List[float]:
        if self.stiffness_mode == "explicit":
            return self.explicit_stiffness
        return [self.stiffness_scale * value for value in self.nominal_stiffness]

    def run(self) -> int:
        if self.stiffness_mode == "nominal_scaled":
            self.get_logger().info(
                f"Cartesian stiffness resolved from nominal scale: "
                f"nominal_stiffness={self.nominal_stiffness}, "
                f"stiffness_scale={self.stiffness_scale}, stiffness={self.stiffness}"
            )
        self.get_logger().info(
            f"Waiting for {self.service_name!r} to set Cartesian impedance: "
            f"stiffness={self.stiffness}, damping_ratio={self.damping_ratio}"
        )
        if not self.client.wait_for_service(timeout_sec=self.service_timeout_sec):
            self.get_logger().error(
                f"Timed out after {self.service_timeout_sec:.1f} s waiting for "
                f"{self.service_name!r}"
            )
            return 1

        request = SetCartesianImpedance.Request()
        request.stiffness = self.stiffness
        request.damping_ratio = self.damping_ratio

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=self.service_timeout_sec
        )
        if not future.done():
            self.get_logger().error(
                f"Timed out after {self.service_timeout_sec:.1f} s calling "
                f"{self.service_name!r}"
            )
            return 1

        response = future.result()
        if response is None:
            self.get_logger().error("Service call returned no response")
            return 1
        if not response.success:
            self.get_logger().error(
                f"Failed to set Cartesian impedance: {response.message}"
            )
            return 1

        self.get_logger().info(f"Cartesian impedance set: {response.message}")
        return 0


FLEXIV_MSGS_HINT = (
    "flexiv_msgs is unavailable. It ships with the flexiv_ros2 source repository "
    "rather than as a Debian package; clone and build it into the workspace "
    "to use the Flexiv task-space workflow."
)


def main() -> None:
    if SetCartesianImpedance is None:
        raise SystemExit(FLEXIV_MSGS_HINT)
    rclpy.init()
    node = FlexivSetCartesianImpedance()
    try:
        raise SystemExit(node.run())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

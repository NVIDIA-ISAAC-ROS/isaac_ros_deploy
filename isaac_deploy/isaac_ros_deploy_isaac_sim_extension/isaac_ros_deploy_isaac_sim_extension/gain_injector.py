# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""
Live kp/kd injection into the Newton actuator pipeline.

The ros2_control topic-based plugin publishes per-joint kp/kd on the
configured gains topic (default ``/isaac_sim_drive_gains``) every cycle
that the policy stack authors at least one non-NaN gain. This module
subscribes to that topic and copies the values into the Newton
``ControllerPD``'s ``kp`` / ``kd`` warp arrays, which the actuator
extension reads at physics rate. NaN slots in the message are skipped
("keep previous value") so a controller that updates only a subset of
joints does not overwrite the others.

Message convention (sensor_msgs/JointState carrying gains, not state):
    name[i]     -> joint name
    position[i] -> kp for joint i  (NaN = no update)
    velocity[i] -> kd for joint i  (NaN = no update)

The kp/kd writes never reallocate -- a 1-element warp scratch array is
created once per joint at startup and re-used for every ``wp.copy`` --
so the per-message cost is one warp kernel launch per joint that gets
a new value.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Tuple

import warp as wp


class GainInjector:
    """
    Subscribe to /isaac_sim_drive_gains and stream kp/kd into Newton arrays.

    Hold a reference to each per-joint Newton actuator (or any object that
    exposes a ``controller`` with ``kp`` / ``kd`` warp arrays), plus a
    1-element warp scratch array for each. On every JointState message the
    callback walks the joints, decodes (kp, kd) from (position, velocity),
    skips NaN entries, and ``wp.copy``-es the new value into the controller's
    warp array.

    A joint named in the message that does not match any known controller
    (e.g. a hand finger when only the leg / arm configs are attached) is
    silently ignored -- this keeps the subscriber robust to the controller
    stack publishing a superset.

    Designed to be safe to construct without an rclpy node (``rclpy_node=None``)
    for unit-testing of the warp-copy hot path. The subscription is then not
    created and the caller drives the callback directly via :meth:`on_msg`.
    """

    def __init__(
        self,
        rclpy_node,
        configs_with_names: Iterable[Tuple[Any, str]],
        *,
        topic: str = '/isaac_sim_drive_gains',
    ):
        self._wp = wp

        # Per-joint cache of (controller, scratch_kp, scratch_kd). Indexed by
        # joint name. The scratch arrays match the controller's kp/kd shape +
        # device exactly (typically shape (n_robots,) on the same device as
        # the actuator pipeline) so wp.copy is a same-device same-shape blit
        # with no per-message reallocation.
        self._joints: dict[str, tuple[Any, Any, Any]] = {}
        for cfg, dof_name in configs_with_names:
            controller = cfg.controller
            if controller is None:  # pragma: no cover - defensive
                continue
            scratch_kp = wp.empty_like(controller.kp)
            scratch_kd = wp.empty_like(controller.kd)
            self._joints[dof_name] = (controller, scratch_kp, scratch_kd)

        self._topic = topic
        self._sub = None
        self._message_count = 0
        self._update_count = 0

        if rclpy_node is not None:
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState
            self._sub = rclpy_node.create_subscription(
                JointState, topic, self.on_msg, qos_profile_sensor_data,
            )

    @property
    def topic(self) -> str:
        """Topic name this injector is subscribed to."""
        return self._topic

    @property
    def message_count(self) -> int:
        """Total messages received (whether or not anything was updated)."""
        return self._message_count

    @property
    def update_count(self) -> int:
        """Total (joint, gain) updates applied since construction."""
        return self._update_count

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Names of the joints this injector can drive."""
        return tuple(self._joints.keys())

    def on_msg(self, msg) -> None:
        """
        Process one JointState message; copy kp/kd into the warp arrays.

        ``msg.position[i]`` is kp for ``msg.name[i]``; ``msg.velocity[i]`` is
        kd. Either array may be empty or shorter than ``msg.name`` -- the
        callback only updates the gain whose array entry exists and is not
        NaN. Joints in ``msg.name`` that are not in this injector are
        silently ignored.
        """
        self._message_count += 1
        names = msg.name
        kp_arr = msg.position
        kd_arr = msg.velocity
        for i, name in enumerate(names):
            entry = self._joints.get(name)
            if entry is None:
                continue
            controller, scratch_kp, scratch_kd = entry
            if i < len(kp_arr):
                kp = float(kp_arr[i])
                if not math.isnan(kp):
                    # fill_ writes a scalar to every element of the per-robot
                    # warp array (shape n_robots), then wp.copy blits it into
                    # the controller's array; both same-device, same-shape.
                    scratch_kp.fill_(kp)
                    self._wp.copy(controller.kp, scratch_kp)
                    self._update_count += 1
            if i < len(kd_arr):
                kd = float(kd_arr[i])
                if not math.isnan(kd):
                    scratch_kd.fill_(kd)
                    self._wp.copy(controller.kd, scratch_kd)
                    self._update_count += 1

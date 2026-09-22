#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record one CSV row per step of the topic-based task-space Deploy graph.

The graph's four nodes each publish part of a policy step: InputBuilderNode
publishes the model inputs, Triton the model outputs, the command builder the raw
action anchored to the observation it came from, and the decoder the target pose
that was actually commanded. This node joins them back together so a run can be
compared against the simulation it was trained in.

It is a passive observer: it subscribes only, publishes only a debug array, and
never touches a command topic, so it is safe to leave enabled on hardware. The
CSV columns are derived from the tensor names in the first complete step rather
than hard-coded, so a re-exported policy with different inputs still logs
correctly.
"""

from __future__ import annotations

import array
from collections import deque
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from geometry_msgs.msg import PoseStamped
from isaac_ros_deploy_interfaces.msg import CartesianPoseDeltaCommand
from isaac_ros_tensor_msgs.msg import TensorList
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray

# DLPack dtype code/bits -> struct format character.
_DLPACK_INT = 0
_DLPACK_FLOAT = 2
_DTYPE_FORMATS = {
    (_DLPACK_INT, 8): 'b',
    (_DLPACK_INT, 16): 'h',
    (_DLPACK_INT, 32): 'i',
    (_DLPACK_INT, 64): 'q',
    (_DLPACK_FLOAT, 32): 'f',
    (_DLPACK_FLOAT, 64): 'd',
}


def _stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _tensor_to_floats(tensor) -> List[float]:
    """Decode an ExperimentalTensor payload into a flat list of floats."""
    key = (int(tensor.dtype_code), int(tensor.dtype_bits))
    fmt = _DTYPE_FORMATS.get(key)
    if fmt is None or int(tensor.dtype_lanes) != 1:
        raise ValueError(f'unsupported tensor dtype {key} with {tensor.dtype_lanes} lanes')
    count = 1
    for dim in tensor.shape:
        count *= int(dim)
    offset = int(tensor.byte_offset)
    itemsize = int(tensor.dtype_bits) // 8
    payload = bytes(tensor.data[offset:offset + count * itemsize])
    if len(payload) != count * itemsize:
        raise ValueError(
            f'tensor has {len(payload)} usable bytes, expected {count * itemsize}')
    values = array.array(fmt)
    values.frombytes(payload)
    return [float(value) for value in values]


def _tensor_list_to_dict(msg: TensorList) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {}
    for index in range(min(len(msg.names), len(msg.tensors))):
        try:
            values[msg.names[index]] = _tensor_to_floats(msg.tensors[index])
        except ValueError:
            continue
    return values


def _flatten(values: Dict[str, List[float]]) -> Tuple[List[str], List[float]]:
    columns: List[str] = []
    flat: List[float] = []
    for name in sorted(values):
        entries = values[name]
        columns.extend(f'{name}_{index}' for index in range(len(entries)))
        flat.extend(entries)
    return columns, flat


class TaskSpacePolicyIoLogger(Node):
    """Join the task-space Deploy graph's per-step topics into one CSV row."""

    def __init__(self):
        super().__init__('task_space_policy_io_logger')

        self.declare_parameter('csv_output_path', '')
        self.declare_parameter('input_tensor_topic', 'input_tensors')
        self.declare_parameter('output_tensor_topic', 'output_tensors')
        self.declare_parameter('pose_delta_command_topic', 'pose_delta_command')
        self.declare_parameter('target_pose_topic', 'target_pose')
        self.declare_parameter('debug_policy_io_topic', 'debug_policy_io')
        self.declare_parameter('action_tensor_name', 'arm_action')
        self.declare_parameter('stamp_tolerance_s', 1.0e-3)
        self.declare_parameter('max_cached_steps', 64)

        self.csv_output_path = str(self.get_parameter('csv_output_path').value).strip()
        self.action_tensor_name = str(self.get_parameter('action_tensor_name').value)
        self.stamp_tolerance_s = max(0.0, float(self.get_parameter('stamp_tolerance_s').value))
        max_cached_steps = max(1, int(self.get_parameter('max_cached_steps').value))

        self.inputs: deque = deque(maxlen=max_cached_steps)
        self.outputs: deque = deque(maxlen=max_cached_steps)
        # A FIFO, not a single slot: if two commands arrive before the decoder
        # publishes the first target, a single slot would drop the older command
        # and pair the next target with the wrong one.
        self.pending_commands: deque = deque(maxlen=max_cached_steps)
        # Command/target pairs waiting for their model tensors. Bounded so a step
        # whose tensors never arrive cannot retain memory indefinitely.
        self.pending_pairs: deque = deque(maxlen=max_cached_steps)
        self.step = 0
        self.csv_file = None
        self.csv_writer = None
        self.csv_fieldnames: List[str] = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            TensorList, str(self.get_parameter('input_tensor_topic').value),
            self._input_cb, qos)
        self.create_subscription(
            TensorList, str(self.get_parameter('output_tensor_topic').value),
            self._output_cb, qos)
        self.create_subscription(
            CartesianPoseDeltaCommand,
            str(self.get_parameter('pose_delta_command_topic').value),
            self._command_cb, 10)
        self.create_subscription(
            PoseStamped, str(self.get_parameter('target_pose_topic').value),
            self._target_cb, 10)

        debug_topic = str(self.get_parameter('debug_policy_io_topic').value).strip()
        self.debug_pub = (
            self.create_publisher(Float64MultiArray, debug_topic, 10) if debug_topic else None)

        self.get_logger().info(
            'Task-space policy I/O logger ready: '
            f"csv_output_path={self.csv_output_path or '<disabled>'}, "
            f"debug_policy_io_topic={debug_topic or '<disabled>'}"
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    def _input_cb(self, msg: TensorList) -> None:
        self.inputs.append((_stamp_to_seconds(msg.header.stamp), _tensor_list_to_dict(msg)))
        self._drain_pending_pairs()

    def _output_cb(self, msg: TensorList) -> None:
        self.outputs.append((_stamp_to_seconds(msg.header.stamp), _tensor_list_to_dict(msg)))
        self._drain_pending_pairs()

    def _command_cb(self, msg: CartesianPoseDeltaCommand) -> None:
        self.pending_commands.append(msg)

    def _target_cb(self, msg: PoseStamped) -> None:
        """The decoder publishes a target from a command, so pair them here.

        The target's stamp is the publish time rather than the policy tick, so it
        cannot be matched by stamp. The decoder publishes one target per command
        in order, so the target belongs to the oldest command not yet paired.

        The pair is queued rather than written immediately: the model tensors for
        this step may still be in flight. Writing here and giving up when they are
        absent would drop the step permanently even though the tensors arrive a
        moment later.
        """
        if not self.pending_commands:
            return
        self.pending_pairs.append((self.pending_commands.popleft(), msg))
        self._drain_pending_pairs()

    def _drain_pending_pairs(self) -> None:
        """Write every queued pair whose model tensors have arrived.

        Pairs are written in order, so the loop stops at the first pair that is
        still waiting. A pair whose tensors have already been evicted from the
        caches can never match, so it is dropped rather than blocking the ones
        behind it.
        """
        while self.pending_pairs:
            command, target = self.pending_pairs[0]
            policy_stamp_s = _stamp_to_seconds(command.header.stamp)
            input_index = self._find_matching(self.inputs, policy_stamp_s)
            output_index = self._find_matching(self.outputs, policy_stamp_s)

            if input_index is None or output_index is None:
                if self._tensors_unreachable(policy_stamp_s):
                    self.pending_pairs.popleft()
                    self.get_logger().warning(
                        'Skipping a policy step: no cached tensors matched the command '
                        f'stamp {policy_stamp_s:.9f}',
                        throttle_duration_sec=2.0,
                    )
                    continue
                return

            model_inputs = self._consume_matching(self.inputs, input_index)
            model_outputs = self._consume_matching(self.outputs, output_index)
            self.pending_pairs.popleft()

            self._publish_debug(model_inputs, model_outputs)
            self._write_row(command, target, policy_stamp_s, model_inputs, model_outputs)
            self.step += 1

    def _tensors_unreachable(self, stamp_s: float) -> bool:
        """Report whether this step's tensors can no longer arrive.

        Both caches are ordered by arrival. Once the oldest cached entry is newer
        than the step being waited on, the step's own tensors have been evicted
        and no future message can match it.
        """
        for cache in (self.inputs, self.outputs):
            if cache and cache[0][0] - stamp_s > self.stamp_tolerance_s:
                return True
        return False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _find_matching(self, cache: deque, stamp_s: float) -> Optional[int]:
        """Return the index of the closest entry within tolerance, without consuming.

        Lookup is separate from consumption so a step is only removed from the
        caches once both the input and the output tensors have been found. A
        destructive lookup would discard one side's entries when the other side
        had not arrived yet.
        """
        best_index = None
        best_delta = float('inf')
        for index, (cached_stamp_s, _values) in enumerate(cache):
            delta = abs(cached_stamp_s - stamp_s)
            if delta < best_delta:
                best_index = index
                best_delta = delta
        if best_index is None or best_delta > self.stamp_tolerance_s:
            return None
        return best_index

    def _consume_matching(self, cache: deque, index: int) -> Dict[str, List[float]]:
        values = cache[index][1]
        # Everything at or before the match is now unreachable.
        for _ in range(index + 1):
            cache.popleft()
        return values

    def _publish_debug(
        self,
        model_inputs: Dict[str, List[float]],
        model_outputs: Dict[str, List[float]],
    ) -> None:
        if self.debug_pub is None:
            return
        _, flat_inputs = _flatten(model_inputs)
        _, flat_outputs = _flatten(model_outputs)
        msg = Float64MultiArray()
        msg.data = [
            float(self.step),
            float(len(flat_inputs)),
            float(len(flat_outputs)),
            *flat_inputs,
            *flat_outputs,
        ]
        self.debug_pub.publish(msg)

    def _write_row(
        self,
        command: CartesianPoseDeltaCommand,
        target: PoseStamped,
        policy_stamp_s: float,
        model_inputs: Dict[str, List[float]],
        model_outputs: Dict[str, List[float]],
    ) -> None:
        if not self.csv_output_path:
            return

        input_columns, flat_inputs = _flatten(model_inputs)
        output_columns, flat_outputs = _flatten(model_outputs)
        row = {
            'step': self.step,
            'policy_stamp_s': f'{policy_stamp_s:.9f}',
            'target_stamp_s': f'{_stamp_to_seconds(target.header.stamp):.9f}',
            'command_frame_id': command.header.frame_id,
            'target_frame_id': target.header.frame_id,
            'observation_pos_x': command.observation_pose.position.x,
            'observation_pos_y': command.observation_pose.position.y,
            'observation_pos_z': command.observation_pose.position.z,
            'observation_qx': command.observation_pose.orientation.x,
            'observation_qy': command.observation_pose.orientation.y,
            'observation_qz': command.observation_pose.orientation.z,
            'observation_qw': command.observation_pose.orientation.w,
            'raw_delta_x': command.delta_position.x,
            'raw_delta_y': command.delta_position.y,
            'raw_delta_z': command.delta_position.z,
            'raw_delta_rx': command.delta_axis_angle.x,
            'raw_delta_ry': command.delta_axis_angle.y,
            'raw_delta_rz': command.delta_axis_angle.z,
            'target_pos_x': target.pose.position.x,
            'target_pos_y': target.pose.position.y,
            'target_pos_z': target.pose.position.z,
            'target_qx': target.pose.orientation.x,
            'target_qy': target.pose.orientation.y,
            'target_qz': target.pose.orientation.z,
            'target_qw': target.pose.orientation.w,
        }
        row.update(zip(input_columns, flat_inputs))
        row.update(zip(output_columns, flat_outputs))

        if self.csv_writer is None:
            self._open_csv(list(row))
        if self.csv_writer is None:
            return
        # A later step cannot add columns to an open file, so drop anything the
        # header did not anticipate rather than raising mid-run.
        self.csv_writer.writerow(
            {key: value for key, value in row.items() if key in self.csv_fieldnames})
        self.csv_file.flush()

    def _open_csv(self, fieldnames: Sequence[str]) -> None:
        path = Path(os.path.expanduser(os.path.expandvars(self.csv_output_path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_fieldnames = list(fieldnames)
        self.csv_file = path.open('w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
        self.csv_writer.writeheader()
        self.get_logger().info(f'Writing task-space policy I/O to {path}')

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
        super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = TaskSpacePolicyIoLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

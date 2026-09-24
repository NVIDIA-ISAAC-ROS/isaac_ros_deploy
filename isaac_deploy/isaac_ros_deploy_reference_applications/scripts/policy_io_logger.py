#!/usr/bin/env python3

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
Policy I/O logger for the Isaac ROS Deploy inference controller.

The Deploy ``InferenceController`` keeps observations and actions as internal
LibTorch tensors, but when launched with ``publish_debug_topics:=true`` it
exposes them on:

* ``<inference_controller>/debug_observation`` (``std_msgs/Float64MultiArray``)
* ``<inference_controller>/debug_action``      (``std_msgs/Float64MultiArray``)

Both are published back-to-back in the same control step. This node subscribes
to those topics plus, when available:

* the tracked ``goal_pose`` (``geometry_msgs/PoseStamped``),
* the robot ``joint_states`` (``sensor_msgs/JointState``),
* the measured end-effector / flange pose in the world frame, published by the
  Flexiv robot-states broadcaster on ``/<robot_sn>/flange_pose``
  (``geometry_msgs/PoseStamped``), and
* the pre-blend safety command (``isaac_ros_deploy_interfaces/JointCommand``)
  via ``safety_command_topic`` — point it at the ``SafetyController``'s
  ``scaled_joint_delta`` (blend-scaled per-joint delta) or a
  ``JointCommandBroadcaster``'s ``final_joint_command`` (absolute pre-blend
  target), and
* the blended / safety-limited ABSOLUTE command actually written to hardware
  (``isaac_ros_deploy_interfaces/JointCommand``) via ``blended_command_topic``
  — the ``SafetyController``'s ``blended_command`` (position = clamped safe
  target; vel/eff/kp/kd = post-blend values), and
* the recurrent (LSTM/GRU) hidden state via ``recurrent_state_topic`` — the
  inference controller's ``debug_recurrent_state`` (``std_msgs/Float64MultiArray``),
  the concatenated ``_out`` state tensors produced this step (== the ``_in`` fed
  to the next step). Useful for diagnosing RNN state divergence vs simulation.

On every action, it pairs the action with the most recent observation / goal /
joint state / flange pose / safety command to produce:

* a throttled, human-readable console line for live debugging, and
* an analysis-ready CSV row for offline data collection.

It is a passive, read-only observer (publishes nothing, never touches command
interfaces), so it is safe to leave enabled in production. The observation and
action debug topics are best-effort and carry no per-step id, so obs/action
pairing is "latest observation seen at the time the action arrived" -- exact in
practice because the controller publishes them together each step, but not a
hard guarantee under dropped messages.

Standalone use (controller already running with publish_debug_topics:=true):

    ros2 run isaac_ros_deploy_reference_applications policy_io_logger.py --ros-args \
        -p obs_topic:=/deploy_inference_controller/debug_observation \
        -p action_topic:=/deploy_inference_controller/debug_action \
        -p goal_pose_topic:=/displayport_insertion/goal_pose \
        -p joint_states_topic:=/flexiv_arm/joint_states \
        -p csv_output_path:=/tmp/policy_io.csv

Usually it is launched automatically via ``log_policy_io:=true`` on the
DisplayPort insertion workflow, which also force-enables the controller's
debug topics.
"""

import csv
import os
import time
from typing import List, Optional

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

try:
    from isaac_ros_deploy_interfaces.msg import JointCommand
except ImportError:  # pragma: no cover - interfaces pkg always present at runtime
    JointCommand = None


class PolicyIoLogger(Node):
    """Console + CSV logger for Deploy inference controller observation/action pairs."""

    def __init__(self):
        super().__init__('policy_io_logger')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter(
            'obs_topic', '/deploy_inference_controller/debug_observation')
        self.declare_parameter(
            'action_topic', '/deploy_inference_controller/debug_action')
        self.declare_parameter('goal_pose_topic', '/displayport_insertion/goal_pose')
        self.declare_parameter('joint_states_topic', '/flexiv_arm/joint_states')
        # Measured end-effector (flange) pose in the world frame. Empty disables.
        self.declare_parameter('eef_pose_topic', '')
        # Blended / safety-gated joint command (SafetyController scaled_joint_delta).
        # Empty disables. Requires the safety controller to run with
        # publish_scaled_joint_delta:=true.
        self.declare_parameter('safety_command_topic', '')
        # Blended / safety-limited ABSOLUTE command actually written to hardware
        # (SafetyController publish_blended_command). Empty disables.
        self.declare_parameter('blended_command_topic', '')
        # Recurrent (LSTM/GRU) hidden-state "_out" tensors from the inference
        # controller (~/debug_recurrent_state). Empty disables.
        self.declare_parameter('recurrent_state_topic', '')
        # Console logging every Nth step (>=1). CSV always records every step.
        self.declare_parameter('log_every_n_steps', 1)
        self.declare_parameter('enable_console_log', True)
        # Absolute path for the CSV output. Empty disables CSV logging.
        # Supports ~ and environment variables (e.g. $ISAAC_ROS_WS).
        self.declare_parameter('csv_output_path', '')
        self.declare_parameter('log_goal_pose', True)
        self.declare_parameter('log_joint_states', True)
        self.declare_parameter('log_eef_pose', True)
        self.declare_parameter('log_safety_command', True)
        self.declare_parameter('log_blended_command', True)
        self.declare_parameter('log_recurrent_state', True)
        self.declare_parameter('float_precision', 4)
        # Seconds to wait for every enabled variable-width source (joints / eef /
        # safety / blended command) before giving up on it: after this long with
        # actions flowing, any still-silent source is dropped (with a warning) so
        # the CSV starts recording instead of deferring its header forever.
        self.declare_parameter('startup_grace_sec', 5.0)

        obs_topic = str(self.get_parameter('obs_topic').value)
        action_topic = str(self.get_parameter('action_topic').value)
        goal_pose_topic = str(self.get_parameter('goal_pose_topic').value)
        joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        eef_pose_topic = str(self.get_parameter('eef_pose_topic').value).strip()
        safety_command_topic = str(self.get_parameter('safety_command_topic').value).strip()
        blended_command_topic = str(self.get_parameter('blended_command_topic').value).strip()
        recurrent_state_topic = str(self.get_parameter('recurrent_state_topic').value).strip()
        self._every_n = max(1, int(self.get_parameter('log_every_n_steps').value))
        self._console = bool(self.get_parameter('enable_console_log').value)
        self._log_goal = bool(self.get_parameter('log_goal_pose').value)
        self._log_joints = bool(self.get_parameter('log_joint_states').value)
        # Effective flags: a source is logged only if enabled AND its topic is set
        # (and, for JointCommand sources, the message type is importable).
        self._log_eef = bool(self.get_parameter('log_eef_pose').value) and bool(eef_pose_topic)
        self._log_safety = (
            bool(self.get_parameter('log_safety_command').value)
            and bool(safety_command_topic)
            and JointCommand is not None)
        self._log_blended = (
            bool(self.get_parameter('log_blended_command').value)
            and bool(blended_command_topic)
            and JointCommand is not None)
        self._log_recurrent = (
            bool(self.get_parameter('log_recurrent_state').value)
            and bool(recurrent_state_topic))
        if JointCommand is None and (
                (bool(self.get_parameter('log_safety_command').value) and safety_command_topic)
                or (bool(self.get_parameter('log_blended_command').value)
                    and blended_command_topic)):
            self.get_logger().warning(
                'A JointCommand topic is set but isaac_ros_deploy_interfaces/JointCommand '
                'could not be imported; command logging disabled.')
        self._precision = int(self.get_parameter('float_precision').value)
        self._grace_sec = float(self.get_parameter('startup_grace_sec').value)

        # Retained for diagnostics when a source is dropped after the grace period.
        self._joints_topic = joint_states_topic
        self._eef_topic = eef_pose_topic
        self._safety_topic = safety_command_topic
        self._blended_topic = blended_command_topic
        self._recurrent_topic = recurrent_state_topic

        raw_csv_path = str(self.get_parameter('csv_output_path').value).strip()
        self._csv_path = (
            os.path.expanduser(os.path.expandvars(raw_csv_path)) if raw_csv_path else '')
        self._csv_file = None
        self._csv_writer = None
        self._csv_header_written = False

        # ── State ─────────────────────────────────────────────────────
        self._step = 0
        self._latest_obs: Optional[List[float]] = None
        self._latest_goal: Optional[PoseStamped] = None
        self._latest_joints: Optional[JointState] = None
        self._latest_eef: Optional[PoseStamped] = None
        self._latest_safety = None   # pre-blend target/delta (JointCommand)
        self._latest_blended = None  # blended/limited final command (JointCommand)
        self._latest_recurrent: Optional[List[float]] = None  # LSTM hidden "_out"
        # Grace-period bookkeeping for finalizing the CSV column schema.
        self._first_pending_time: Optional[float] = None

        # ── QoS: best-effort to match the controller's debug publishers.
        #     A best-effort subscription is also compatible with the reliable
        #     goal_pose / joint_states publishers.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(Float64MultiArray, obs_topic, self._observation_cb, qos)
        self.create_subscription(Float64MultiArray, action_topic, self._action_cb, qos)
        if self._log_goal:
            self.create_subscription(PoseStamped, goal_pose_topic, self._goal_pose_cb, qos)
        if self._log_joints:
            self.create_subscription(JointState, joint_states_topic, self._joint_states_cb, qos)
        if self._log_eef:
            self.create_subscription(PoseStamped, eef_pose_topic, self._eef_pose_cb, qos)
        if self._log_safety:
            self.create_subscription(
                JointCommand, safety_command_topic, self._safety_command_cb, qos)
        if self._log_blended:
            self.create_subscription(
                JointCommand, blended_command_topic, self._blended_command_cb, qos)
        if self._log_recurrent:
            self.create_subscription(
                Float64MultiArray, recurrent_state_topic, self._recurrent_state_cb, qos)

        if self._csv_path:
            self._open_csv()

        self.get_logger().info(
            'Policy I/O logger started\n'
            f'  obs topic     : {obs_topic}\n'
            f'  action topic  : {action_topic}\n'
            f'  goal pose     : {goal_pose_topic if self._log_goal else "(disabled)"}\n'
            f'  joint states  : {joint_states_topic if self._log_joints else "(disabled)"}\n'
            f'  eef/flange    : {eef_pose_topic if self._log_eef else "(disabled)"}\n'
            f'  safety cmd    : {safety_command_topic if self._log_safety else "(disabled)"}\n'
            f'  blended cmd   : {blended_command_topic if self._log_blended else "(disabled)"}\n'
            f'  recurrent st  : {recurrent_state_topic if self._log_recurrent else "(disabled)"}\n'
            f'  console       : {self._console} (every {self._every_n} step(s))\n'
            f'  csv output    : {self._csv_path or "(disabled)"}\n'
            f'  startup grace : {self._grace_sec:.1f}s\n'
            '  NOTE: requires the inference controller with publish_debug_topics:=true, '
            'and (for the command columns) the safety controller with '
            'publish_scaled_joint_delta / publish_blended_command enabled.')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _observation_cb(self, msg: Float64MultiArray):
        self._latest_obs = list(msg.data)

    def _goal_pose_cb(self, msg: PoseStamped):
        self._latest_goal = msg

    def _joint_states_cb(self, msg: JointState):
        self._latest_joints = msg

    def _eef_pose_cb(self, msg: PoseStamped):
        self._latest_eef = msg

    def _safety_command_cb(self, msg):
        self._latest_safety = msg

    def _blended_command_cb(self, msg):
        self._latest_blended = msg

    def _recurrent_state_cb(self, msg: Float64MultiArray):
        self._latest_recurrent = list(msg.data)

    def _action_cb(self, msg: Float64MultiArray):
        # Deploy publishes debug_observation then debug_action within the same
        # control step, so the latest observation is the one that produced this
        # action.
        self._step += 1
        action = list(msg.data)
        obs = self._latest_obs

        if self._csv_writer is not None:
            self._write_csv_row(obs, action)

        if self._console and (self._step % self._every_n == 0):
            self._console_log(obs, action)

    # ── Console formatting ─────────────────────────────────────────────

    def _fmt(self, values: Optional[List[float]]) -> str:
        if values is None:
            return 'N/A'
        p = self._precision
        return '[' + ', '.join(f'{v:.{p}f}' for v in values) + ']'

    def _console_log(self, obs, action):
        p = self._precision
        obs_len = 0 if obs is None else len(obs)
        parts = [
            f'[policy step={self._step}]',
            f'obs({obs_len})={self._fmt(obs)}',
            f'action({len(action)})={self._fmt(action)}',
        ]
        if self._log_recurrent and self._latest_recurrent is not None:
            parts.append(
                f'rnn({len(self._latest_recurrent)})={self._fmt(self._latest_recurrent)}')
        if self._log_joints and self._latest_joints is not None:
            joints = ', '.join(
                f'{n}={v:.{p}f}'
                for n, v in zip(self._latest_joints.name, self._latest_joints.position))
            parts.append(f'joints=[{joints}]')
        if self._log_goal and self._latest_goal is not None:
            gp = self._latest_goal.pose
            parts.append(
                f'goal_pos=[{gp.position.x:.{p}f}, {gp.position.y:.{p}f}, '
                f'{gp.position.z:.{p}f}] '
                f'goal_quat=[{gp.orientation.x:.{p}f}, {gp.orientation.y:.{p}f}, '
                f'{gp.orientation.z:.{p}f}, {gp.orientation.w:.{p}f}]')
        if self._log_eef and self._latest_eef is not None:
            ep = self._latest_eef.pose
            parts.append(
                f'eef_pos=[{ep.position.x:.{p}f}, {ep.position.y:.{p}f}, '
                f'{ep.position.z:.{p}f}] '
                f'eef_quat=[{ep.orientation.x:.{p}f}, {ep.orientation.y:.{p}f}, '
                f'{ep.orientation.z:.{p}f}, {ep.orientation.w:.{p}f}]')
        if self._log_safety and self._latest_safety is not None:
            sc = self._latest_safety
            safe_cmd = ', '.join(
                f'{n}={v:.{p}f}' for n, v in zip(sc.names, sc.position))
            parts.append(f'safety_cmd=[{safe_cmd}]')
        if self._log_blended and self._latest_blended is not None:
            bc = self._latest_blended
            blend_cmd = ', '.join(
                f'{n}={v:.{p}f}' for n, v in zip(bc.names, bc.position))
            parts.append(f'blend_cmd=[{blend_cmd}]')
        self.get_logger().info('  '.join(parts))

    # ── CSV output ─────────────────────────────────────────────────────

    def _open_csv(self):
        try:
            directory = os.path.dirname(self._csv_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._csv_file = open(self._csv_path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self.get_logger().info(f'Writing policy I/O CSV to {self._csv_path}')
        except OSError as exc:
            self._csv_file = None
            self._csv_writer = None
            self.get_logger().error(
                f'Failed to open csv_output_path "{self._csv_path}": {exc}. '
                'CSV logging disabled.')

    def _write_csv_header(self, joint_names, n_obs, n_action,
                          safe_cmd_names, blend_cmd_names, n_recurrent):
        header = ['wall_time', 'ros_time', 'step']
        if self._log_joints:
            header += [f'joint_{n}_pos' for n in joint_names]
        header += [f'obs_{i}' for i in range(n_obs)]
        header += [f'action_{i}' for i in range(n_action)]
        if self._log_recurrent:
            header += [f'rnn_{i}' for i in range(n_recurrent)]
        if self._log_goal:
            header += ['goal_px', 'goal_py', 'goal_pz',
                       'goal_qx', 'goal_qy', 'goal_qz', 'goal_qw']
        if self._log_eef:
            header += ['eef_px', 'eef_py', 'eef_pz',
                       'eef_qx', 'eef_qy', 'eef_qz', 'eef_qw']
        if self._log_safety:
            header += [f'safety_cmd_{n}' for n in safe_cmd_names]
        if self._log_blended:
            header += [f'blend_cmd_{n}' for n in blend_cmd_names]
        self._csv_writer.writerow(header)
        self._csv_header_written = True

    def _missing_sources(self):
        """Enabled variable-width sources that have not produced a message yet."""
        missing = []
        if self._log_joints and self._latest_joints is None:
            missing.append(('joint_states', self._joints_topic))
        if self._log_eef and self._latest_eef is None:
            missing.append(('eef_pose', self._eef_topic))
        if self._log_safety and self._latest_safety is None:
            missing.append(('safety_command', self._safety_topic))
        if self._log_blended and self._latest_blended is None:
            missing.append(('blended_command', self._blended_topic))
        if self._log_recurrent and self._latest_recurrent is None:
            missing.append(('recurrent_state', self._recurrent_topic))
        return missing

    def _disable_source(self, label):
        if label == 'joint_states':
            self._log_joints = False
        elif label == 'eef_pose':
            self._log_eef = False
        elif label == 'safety_command':
            self._log_safety = False
        elif label == 'blended_command':
            self._log_blended = False
        elif label == 'recurrent_state':
            self._log_recurrent = False

    def _write_csv_row(self, obs, action):
        if obs is None:
            return

        # Finalize the column schema exactly once. Defer until every enabled
        # variable-width source has produced a message so columns align with the
        # header; but after startup_grace_sec, drop any still-silent source (with
        # a warning) so the CSV starts recording instead of staying empty.
        if not self._csv_header_written:
            missing = self._missing_sources()
            if missing:
                now = time.monotonic()
                if self._first_pending_time is None:
                    self._first_pending_time = now
                if (now - self._first_pending_time) < self._grace_sec:
                    return
                for label, topic in missing:
                    self._disable_source(label)
                    self.get_logger().warning(
                        f'No message on {label} topic {topic!r} after '
                        f'{self._grace_sec:.1f}s; dropping it from the CSV so '
                        'recording can start.')
            joint_names = list(self._latest_joints.name) if self._log_joints else []
            safe_cmd_names = list(self._latest_safety.names) if self._log_safety else []
            blend_cmd_names = list(self._latest_blended.names) if self._log_blended else []
            n_recurrent = len(self._latest_recurrent) if self._log_recurrent else 0
            self._write_csv_header(
                joint_names, len(obs), len(action), safe_cmd_names, blend_cmd_names,
                n_recurrent)

        ros_time = self.get_clock().now().nanoseconds * 1e-9
        row = [f'{time.time():.6f}', f'{ros_time:.6f}', self._step]
        if self._log_joints:
            row += [f'{v:.6f}' for v in self._latest_joints.position]
        row += [f'{v:.6f}' for v in obs]
        row += [f'{v:.6f}' for v in action]
        if self._log_recurrent:
            row += [f'{v:.6f}' for v in self._latest_recurrent]
        if self._log_goal:
            if self._latest_goal is not None:
                gp = self._latest_goal.pose
                row += [f'{gp.position.x:.6f}', f'{gp.position.y:.6f}',
                        f'{gp.position.z:.6f}', f'{gp.orientation.x:.6f}',
                        f'{gp.orientation.y:.6f}', f'{gp.orientation.z:.6f}',
                        f'{gp.orientation.w:.6f}']
            else:
                row += ['', '', '', '', '', '', '']
        if self._log_eef:
            ep = self._latest_eef.pose
            row += [f'{ep.position.x:.6f}', f'{ep.position.y:.6f}',
                    f'{ep.position.z:.6f}', f'{ep.orientation.x:.6f}',
                    f'{ep.orientation.y:.6f}', f'{ep.orientation.z:.6f}',
                    f'{ep.orientation.w:.6f}']
        if self._log_safety:
            row += [f'{v:.6f}' for v in self._latest_safety.position]
        if self._log_blended:
            row += [f'{v:.6f}' for v in self._latest_blended.position]
        self._csv_writer.writerow(row)
        # Flush every row so a Ctrl-C / crash mid-run still leaves valid data.
        self._csv_file.flush()

    def close(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self.get_logger().info(
                f'Closed policy I/O CSV ({self._step} steps logged)')


def main():
    rclpy.init()
    node = PolicyIoLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

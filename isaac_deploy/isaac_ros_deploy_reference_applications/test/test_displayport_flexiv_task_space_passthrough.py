#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
from pathlib import Path
import sys
import types


def _is_importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _pose_stamped_stub():
    """Minimal stand-in supporting the nested field access the module uses."""

    class _Stamp:
        def __init__(self):
            self.sec = 0
            self.nanosec = 0

    class _Header:
        def __init__(self):
            self.frame_id = ""
            self.stamp = _Stamp()

    class _Point:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class _Quaternion(_Point):
        def __init__(self):
            super().__init__()
            self.w = 1.0

    class _Pose:
        def __init__(self):
            self.position = _Point()
            self.orientation = _Quaternion()

    class PoseStampedStub:
        def __init__(self):
            self.header = _Header()
            self.pose = _Pose()

    return PoseStampedStub


def _simple_msg_stub(name, **defaults):
    """Message stand-in accepting the keyword fields the tests set."""

    def __init__(self, **kwargs):
        for key, value in defaults.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    return type(name, (), {"__init__": __init__})


def _install_ros_stubs():
    """Stub only what the passthrough imports at module scope.

    Importing the real geometry_msgs pulls in numpy, which is not usable in a
    hermetic test sandbox, so prefer stubs when the packages are unavailable.
    Where the real packages are present, as under colcon, they are used instead.
    """
    for name, attrs in (
        ("geometry_msgs", {}),
        ("geometry_msgs.msg", {"PoseStamped": _pose_stamped_stub()}),
        ("std_msgs", {}),
        ("std_msgs.msg", {"Bool": _simple_msg_stub("Bool", data=False)}),
        ("rclpy", {"init": lambda args=None: None, "ok": lambda: False,
                   "shutdown": lambda: None, "spin": lambda node: None}),
        ("rclpy.executors", {"ExternalShutdownException": RuntimeError}),
        ("rclpy.node", {"Node": object}),
    ):
        if name in sys.modules or _is_importable(name.split(".")[0]):
            continue
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_install_ros_stubs()

from geometry_msgs.msg import PoseStamped  # noqa: E402  (needs the stubs above)


def _load_passthrough_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "displayport_flexiv_task_space_passthrough.py"
    )
    spec = importlib.util.spec_from_file_location(
        "displayport_flexiv_task_space_passthrough",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


passthrough = _load_passthrough_module()


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message)


class _FakePassthrough:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.hold_sent_for_current_stale_event = True
        self.last_command_wall_time_s = 123.0
        self.logger = _FakeLogger()

    def get_logger(self):
        return self.logger


def _pose_msg(stamp_s=0.0):
    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.header.stamp.sec = int(stamp_s)
    msg.header.stamp.nanosec = int(round((stamp_s - int(stamp_s)) * 1.0e9))
    msg.pose.position.x = 0.47
    msg.pose.position.y = 0.12
    msg.pose.position.z = 0.33
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0
    return msg


def test_valid_enabled_command_forwards_without_task_gating():
    should_forward, reason, age_s = passthrough._evaluate_command(
        _pose_msg(stamp_s=9.9),
        enabled=True,
        now_s=10.0,
        max_command_age_s=0.25,
    )

    assert should_forward
    assert reason == ""
    assert math.isclose(age_s, 0.1)


def test_disabled_adapter_rejects_without_modifying_policy_command():
    msg = _pose_msg(stamp_s=9.9)

    should_forward, reason, age_s = passthrough._evaluate_command(
        msg,
        enabled=False,
        now_s=10.0,
        max_command_age_s=0.25,
    )

    assert not should_forward
    assert reason == "disabled"
    assert math.isclose(age_s, 0.1)


def test_nonfinite_pose_is_rejected():
    msg = _pose_msg(stamp_s=9.9)
    msg.pose.position.z = float("nan")

    should_forward, reason, _ = passthrough._evaluate_command(
        msg,
        enabled=True,
        now_s=10.0,
        max_command_age_s=0.25,
    )

    assert not should_forward
    assert reason == "nonfinite_pose"


def test_stale_pose_is_rejected_when_age_limit_is_enabled():
    should_forward, reason, age_s = passthrough._evaluate_command(
        _pose_msg(stamp_s=9.0),
        enabled=True,
        now_s=10.0,
        max_command_age_s=0.25,
    )

    assert not should_forward
    assert reason == "stale_command"
    assert math.isclose(age_s, 1.0)


def test_zero_stamp_is_allowed_for_topic_only_sources():
    should_forward, reason, age_s = passthrough._evaluate_command(
        _pose_msg(stamp_s=0.0),
        enabled=True,
        now_s=10.0,
        max_command_age_s=0.25,
    )

    assert should_forward
    assert reason == ""
    assert math.isnan(age_s)


def test_copy_pose_stamped_preserves_raw_target_values():
    msg = _pose_msg(stamp_s=1.25)
    msg.pose.position.x = 1.0
    msg.pose.orientation.y = 0.707
    copied = passthrough._copy_pose_stamped(msg)

    assert copied is not msg
    assert copied.header.frame_id == msg.header.frame_id
    assert copied.header.stamp.sec == msg.header.stamp.sec
    assert copied.header.stamp.nanosec == msg.header.stamp.nanosec
    assert copied.pose.position.x == msg.pose.position.x
    assert copied.pose.orientation.y == msg.pose.orientation.y


def test_enable_transition_resets_stale_watchdog_timestamp():
    fake = _FakePassthrough(enabled=False)

    passthrough.DisplayPortFlexivTaskSpacePassthrough._enable_cb(
        fake, passthrough.Bool(data=True)
    )

    assert fake.enabled
    assert fake.last_command_wall_time_s is None
    assert not fake.hold_sent_for_current_stale_event


def test_repeated_enable_keeps_current_stale_watchdog_timestamp():
    fake = _FakePassthrough(enabled=True)

    passthrough.DisplayPortFlexivTaskSpacePassthrough._enable_cb(
        fake, passthrough.Bool(data=True)
    )

    assert fake.enabled
    assert fake.last_command_wall_time_s == 123.0
    assert not fake.hold_sent_for_current_stale_event


class _FakeFlexivCommand:
    def __init__(self, pose):
        self.target_pose = pose


def _fake_safety_node(feedback=None):
    fake = _FakePassthrough(enabled=True)
    fake.latest_end_effector_feedback_pose = feedback
    fake.require_end_effector_feedback_for_flexiv_safety = True
    fake.max_flexiv_target_translation_delta_m = 0.05
    fake.max_flexiv_target_downward_delta_m = 0.03
    return fake


def test_flexiv_safety_requires_tcp_feedback_by_default():
    target = _pose_msg()
    fake = _fake_safety_node(feedback=None)

    should_forward, reason = (
        passthrough.DisplayPortFlexivTaskSpacePassthrough._evaluate_flexiv_command_safety(
            fake,
            _FakeFlexivCommand(target.pose),
        )
    )

    assert not should_forward
    assert reason == "missing_end_effector_feedback_pose_for_flexiv_safety"


def test_flexiv_safety_rejects_large_downward_tcp_jump():
    feedback = _pose_msg()
    feedback.pose.position.z = 0.317
    target = _pose_msg()
    target.pose.position.z = 0.130
    fake = _fake_safety_node(feedback=feedback)

    should_forward, reason = (
        passthrough.DisplayPortFlexivTaskSpacePassthrough._evaluate_flexiv_command_safety(
            fake,
            _FakeFlexivCommand(target.pose),
        )
    )

    assert not should_forward
    assert reason == "flexiv_target_downward_delta_exceeded"


def test_flexiv_safety_allows_small_tcp_target_delta():
    feedback = _pose_msg()
    feedback.pose.position.z = 0.317
    target = _pose_msg()
    target.pose.position.x += 0.005
    target.pose.position.z = 0.310
    fake = _fake_safety_node(feedback=feedback)

    should_forward, reason = (
        passthrough.DisplayPortFlexivTaskSpacePassthrough._evaluate_flexiv_command_safety(
            fake,
            _FakeFlexivCommand(target.pose),
        )
    )

    assert should_forward
    assert reason == ""

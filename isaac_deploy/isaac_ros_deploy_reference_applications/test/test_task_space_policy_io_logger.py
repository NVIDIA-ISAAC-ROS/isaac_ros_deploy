#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the task-space policy I/O logger's decoding and step matching."""

from collections import deque
import importlib.util
from pathlib import Path
import struct
import sys
import types

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def _is_importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _install_ros_stubs():
    """Stub only what the logger imports at module scope."""
    for name, attrs in (
        ('geometry_msgs', {}),
        ('geometry_msgs.msg', {'PoseStamped': type('PoseStamped', (), {})}),
        ('isaac_ros_deploy_interfaces', {}),
        ('isaac_ros_deploy_interfaces.msg',
         {'CartesianPoseDeltaCommand': type('CartesianPoseDeltaCommand', (), {})}),
        ('isaac_ros_tensor_msgs', {}),
        ('isaac_ros_tensor_msgs.msg', {'TensorList': type('TensorList', (), {})}),
        ('std_msgs', {}),
        ('std_msgs.msg', {'Float64MultiArray': type('Float64MultiArray', (), {})}),
        ('rclpy', {'init': lambda args=None: None, 'ok': lambda: False,
                   'shutdown': lambda: None, 'spin': lambda node: None}),
        ('rclpy.executors', {'ExternalShutdownException': RuntimeError}),
        ('rclpy.node', {'Node': object}),
        ('rclpy.qos', {
            'DurabilityPolicy': types.SimpleNamespace(VOLATILE=object()),
            'HistoryPolicy': types.SimpleNamespace(KEEP_LAST=object()),
            'ReliabilityPolicy': types.SimpleNamespace(BEST_EFFORT=object()),
            'QoSProfile': object,
        }),
    ):
        if name in sys.modules or _is_importable(name.split('.')[0]):
            continue
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


def _load_logger():
    _install_ros_stubs()
    spec = importlib.util.spec_from_file_location(
        'task_space_policy_io_logger', _SCRIPTS / 'task_space_policy_io_logger.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


logger = _load_logger()


class _Tensor:
    def __init__(self, shape, dtype_code, dtype_bits, data, byte_offset=0, dtype_lanes=1):
        self.shape = shape
        self.dtype_code = dtype_code
        self.dtype_bits = dtype_bits
        self.dtype_lanes = dtype_lanes
        self.byte_offset = byte_offset
        self.data = data


def test_decodes_float32_tensor():
    values = [1.5, -2.25, 3.0]
    tensor = _Tensor([1, 3], 2, 32, struct.pack('<3f', *values))

    assert logger._tensor_to_floats(tensor) == pytest.approx(values)


def test_decodes_int64_tensor():
    tensor = _Tensor([2], 0, 64, struct.pack('<2q', 7, -9))

    assert logger._tensor_to_floats(tensor) == [7.0, -9.0]


def test_honours_byte_offset():
    payload = b'\x00\x00\x00\x00' + struct.pack('<2f', 4.0, 5.0)
    tensor = _Tensor([2], 2, 32, payload, byte_offset=4)

    assert logger._tensor_to_floats(tensor) == pytest.approx([4.0, 5.0])


def test_rejects_unsupported_dtype_and_short_payload():
    with pytest.raises(ValueError):
        logger._tensor_to_floats(_Tensor([1], 2, 16, b'\x00\x00'))
    with pytest.raises(ValueError):
        logger._tensor_to_floats(_Tensor([4], 2, 32, struct.pack('<2f', 1.0, 2.0)))


def test_flatten_is_ordered_by_tensor_name():
    columns, flat = logger._flatten({'b': [2.0], 'a': [0.0, 1.0]})

    assert columns == ['a_0', 'a_1', 'b_0']
    assert flat == [0.0, 1.0, 2.0]


class _MatcherOnly:
    """Exercise the stamp matching without constructing a real rclpy Node."""

    def __init__(self, tolerance_s):
        self.stamp_tolerance_s = tolerance_s

    _find_matching = logger.TaskSpacePolicyIoLogger._find_matching
    _consume_matching = logger.TaskSpacePolicyIoLogger._consume_matching


def test_find_matching_returns_the_closest_step_within_tolerance():
    matcher = _MatcherOnly(1.0e-3)
    cache = deque([(1.0, {'a': [1.0]}), (2.0, {'a': [2.0]}), (3.0, {'a': [3.0]})])

    index = matcher._find_matching(cache, 2.0)
    assert index == 1
    # Lookup alone must not consume: the other tensor may not have arrived yet.
    assert len(cache) == 3

    assert matcher._consume_matching(cache, index) == {'a': [2.0]}
    # The matched step and everything older than it are consumed.
    assert [stamp for stamp, _ in cache] == [3.0]


def test_find_matching_rejects_a_step_outside_tolerance():
    matcher = _MatcherOnly(1.0e-3)
    cache = deque([(1.0, {'a': [1.0]})])

    assert matcher._find_matching(cache, 5.0) is None
    # A non-match must not consume the cache.
    assert len(cache) == 1


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message)


class _StubStamp:
    def __init__(self, sec, nanosec):
        self.sec = sec
        self.nanosec = nanosec


class _StubHeader:
    def __init__(self, stamp_s):
        self.stamp = _StubStamp(int(stamp_s), int(round((stamp_s % 1) * 1e9)))


class _StubCommand:
    def __init__(self, stamp_s):
        self.header = _StubHeader(stamp_s)


class _DrainOnly:
    """Exercise _drain_pending_pairs without constructing a real rclpy Node."""

    def __init__(self, tolerance_s):
        self.stamp_tolerance_s = tolerance_s
        self.inputs = deque()
        self.outputs = deque()
        self.pending_pairs = deque()
        self.step = 0
        self.rows = []
        self._logger = _FakeLogger()

    def get_logger(self):
        return self._logger

    def _publish_debug(self, model_inputs, model_outputs):
        pass

    def _write_row(self, command, target, policy_stamp_s, model_inputs, model_outputs):
        self.rows.append((policy_stamp_s, model_inputs, model_outputs))

    _find_matching = logger.TaskSpacePolicyIoLogger._find_matching
    _consume_matching = logger.TaskSpacePolicyIoLogger._consume_matching
    _tensors_unreachable = logger.TaskSpacePolicyIoLogger._tensors_unreachable
    _drain_pending_pairs = logger.TaskSpacePolicyIoLogger._drain_pending_pairs


def test_a_step_is_written_when_its_tensors_arrive_after_the_target():
    """A late tensor must not cost the step: the pair waits until both arrive."""
    drain = _DrainOnly(1.0e-3)
    drain.pending_pairs.append((_StubCommand(2.0), object()))

    # Only the input has arrived, so nothing may be written or consumed yet.
    drain.inputs.append((2.0, {'a': [1.0]}))
    drain._drain_pending_pairs()
    assert drain.rows == []
    assert drain.step == 0
    assert len(drain.pending_pairs) == 1
    assert len(drain.inputs) == 1

    # The output lands a moment later and the step is recorded.
    drain.outputs.append((2.0, {'b': [2.0]}))
    drain._drain_pending_pairs()
    assert [row[0] for row in drain.rows] == [2.0]
    assert drain.step == 1
    assert not drain.pending_pairs
    assert drain._logger.warnings == []


def test_a_step_is_dropped_once_its_tensors_are_evicted():
    """A pair that can never match must not block the pairs behind it."""
    drain = _DrainOnly(1.0e-3)
    drain.pending_pairs.append((_StubCommand(1.0), object()))
    drain.pending_pairs.append((_StubCommand(2.0), object()))

    # Only the newer step's tensors are cached, so the older one is unreachable.
    drain.inputs.append((2.0, {'a': [1.0]}))
    drain.outputs.append((2.0, {'b': [2.0]}))
    drain._drain_pending_pairs()

    assert [row[0] for row in drain.rows] == [2.0]
    assert drain.step == 1
    assert not drain.pending_pairs
    assert len(drain._logger.warnings) == 1

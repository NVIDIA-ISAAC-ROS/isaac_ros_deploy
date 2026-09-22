#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pose math shared by the task-space DisplayPort nodes."""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pose_math = _load('task_space_pose_math')


def test_tcp_offset_applies_in_flange_frame():
    flange_pos = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    flange_quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    tcp_offset = np.asarray([0.0, 0.0, 0.15], dtype=np.float64)

    tcp_pos = flange_pos + pose_math.quat_rotate_xyzw(flange_quat, tcp_offset)

    np.testing.assert_allclose(tcp_pos, [1.0, 2.0, 3.15], atol=1e-12)


def test_tcp_offset_rotates_in_flange_frame():
    flange_quat = np.asarray([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float64)
    tcp_offset = np.asarray([0.1, 0.0, 0.0], dtype=np.float64)

    rotated = pose_math.quat_rotate_xyzw(flange_quat, tcp_offset)

    np.testing.assert_allclose(rotated, [0.0, 0.1, 0.0], atol=1e-12)


def test_zero_quaternion_normalizes_to_identity():
    assert pose_math.quat_normalize_xyzw([0.0, 0.0, 0.0, 0.0]) == (0.0, 0.0, 0.0, 1.0)


def test_as_float_list_accepts_csv_and_sequences():
    assert pose_math.as_float_list('0.1, 0.2,0.3', 3, 'tcp_offset') == [0.1, 0.2, 0.3]
    assert pose_math.as_float_list([1, 2, 3], 3, 'tcp_offset') == [1.0, 2.0, 3.0]


def test_as_float_list_rejects_wrong_length_and_non_finite():
    with pytest.raises(ValueError):
        pose_math.as_float_list([1.0, 2.0], 3, 'tcp_offset')
    with pytest.raises(ValueError):
        pose_math.as_float_list([1.0, 2.0, float('nan')], 3, 'tcp_offset')


def _homogeneous(pos, quat_xyzw):
    matrix = np.eye(4)
    rows = pose_math.quat_to_matrix_xyzw(quat_xyzw)
    matrix[:3, :3] = np.asarray(rows, dtype=np.float64)
    matrix[:3, 3] = np.asarray(pos, dtype=np.float64)
    return matrix


def test_relative_pose_matches_homogeneous_transform_inverse():
    """The quaternion form must agree with inverse(parent) @ child."""
    parent_pos = [0.3, -0.2, 0.7]
    parent_quat = pose_math.quat_normalize_xyzw([0.1, 0.3, -0.2, 0.9])
    child_pos = [0.35, -0.1, 0.9]
    child_quat = pose_math.quat_normalize_xyzw([-0.2, 0.1, 0.4, 0.8])

    offset, quat_offset = pose_math.relative_pose_from_world_poses(
        parent_pos, parent_quat, child_pos, child_quat)

    expected = np.linalg.inv(_homogeneous(parent_pos, parent_quat)) @ _homogeneous(
        child_pos, child_quat)
    np.testing.assert_allclose(offset, expected[:3, 3], atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(pose_math.quat_to_matrix_xyzw(quat_offset)), expected[:3, :3], atol=1e-12)


def test_relative_pose_recovers_a_pure_translation_offset():
    flange_pos = [0.4, 0.1, 0.5]
    flange_quat = pose_math.quat_normalize_xyzw([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    tcp_offset = [0.0, 0.0, 0.1925]
    tcp_pos = np.asarray(flange_pos) + pose_math.quat_rotate_xyzw(flange_quat, tcp_offset)

    offset, quat_offset = pose_math.relative_pose_from_world_poses(
        flange_pos, flange_quat, tcp_pos, flange_quat)

    np.testing.assert_allclose(offset, tcp_offset, atol=1e-12)
    np.testing.assert_allclose(quat_offset, [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_tcp_offset_estimator_averages_until_sample_count():
    estimator = pose_math.TcpOffsetEstimator(sample_count=3, max_translation_std_m=0.002)
    identity = [0.0, 0.0, 0.0, 1.0]

    assert not estimator.add_sample([0.0] * 3, identity, [0.0, 0.0, 0.10], identity)
    assert not estimator.add_sample([0.0] * 3, identity, [0.0, 0.0, 0.11], identity)
    assert not estimator.ready
    assert estimator.add_sample([0.0] * 3, identity, [0.0, 0.0, 0.12], identity)

    assert estimator.ready
    np.testing.assert_allclose(estimator.offset, [0.0, 0.0, 0.11], atol=1e-12)
    # Further samples are ignored once the offset is fixed.
    assert not estimator.add_sample([0.0] * 3, identity, [0.0, 0.0, 9.0], identity)
    np.testing.assert_allclose(estimator.offset, [0.0, 0.0, 0.11], atol=1e-12)


def test_tcp_offset_estimator_flags_inconsistent_samples():
    estimator = pose_math.TcpOffsetEstimator(sample_count=2, max_translation_std_m=0.002)
    identity = [0.0, 0.0, 0.0, 1.0]

    estimator.add_sample([0.0] * 3, identity, [0.0, 0.0, 0.10], identity)
    estimator.add_sample([0.0] * 3, identity, [0.0, 0.0, 0.20], identity)

    assert estimator.ready
    assert estimator.translation_std_exceeded


def test_tcp_offset_estimator_rejects_non_finite_samples():
    estimator = pose_math.TcpOffsetEstimator(sample_count=1, max_translation_std_m=0.0)
    identity = [0.0, 0.0, 0.0, 1.0]

    with pytest.raises(ValueError):
        estimator.add_sample([0.0] * 3, identity, [0.0, 0.0, float('inf')], identity)
    assert not estimator.ready


def _declared_parameters(script_name):
    tree = ast.parse((_SCRIPTS / f'{script_name}.py').read_text())
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, 'attr', '') == 'declare_parameter'
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            names.add(node.args[0].value)
    return names


def _declared_parameter_default(script_name, parameter_name):
    tree = ast.parse((_SCRIPTS / f'{script_name}.py').read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, 'attr', '') == 'declare_parameter'
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == parameter_name):
            return ast.literal_eval(node.args[1])
    raise AssertionError(f'{script_name} does not declare {parameter_name}')


@pytest.mark.parametrize(
    'script_name',
    ['displayport_task_space_pose_source', 'displayport_flexiv_task_space_passthrough'],
)
def test_tcp_offset_is_declared_as_a_double_array(script_name):
    """Both nodes must take the same YAML/CLI syntax for their tcp_offset parameter.

    A string default silently changes the accepted parameter type, so an operator
    parameter file written for one node would be rejected by the other.
    """
    default = _declared_parameter_default(script_name, 'tcp_offset')

    assert isinstance(default, list), f'{script_name} declares tcp_offset as {type(default)}'
    assert len(default) == 3
    assert all(isinstance(item, float) for item in default)


def test_pose_source_declares_the_real_robot_parameters():
    """The launch file writes these by name; a rename here breaks it silently."""
    declared = _declared_parameters('displayport_task_space_pose_source')

    assert {
        'enable_topic',
        'enabled_on_start',
        'socket_pose_is_keypoint',
        'socket_pose_timeout_s',
        'socket_root_to_keypoint_offset',
        'source_pose_reference',
        'tcp_offset',
        'tcp_offset_topic',
    } <= declared

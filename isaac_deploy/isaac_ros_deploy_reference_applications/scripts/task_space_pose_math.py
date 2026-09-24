#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose and quaternion helpers shared by the task-space DisplayPort nodes.

The pose source and the Flexiv passthrough both need the same small set of
xyzw-quaternion operations and the same "measure a rigid offset from a stream of
parent/child pose pairs" logic. Keeping one copy here means the observation path
and the command path cannot drift apart in how they interpret a TCP offset.

This module deliberately depends only on ``math`` so it can be imported by the
scripts' unit tests without a ROS workspace.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Vector3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def as_float_list(value: object, expected_len: int, name: str) -> List[float]:
    """Coerce a ROS parameter to a fixed-length float list.

    Accepts either a real sequence or the comma-separated string form that
    launch arguments produce.
    """
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
        values = [float(part) for part in parts]
    else:
        values = [float(item) for item in value]  # type: ignore[arg-type]
    if len(values) != expected_len:
        raise ValueError(f'{name} must have {expected_len} values, got {values!r}')
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f'{name} must contain only finite values, got {values!r}')
    return values


def quat_normalize_xyzw(quat: Sequence[float]) -> Quat:
    x, y, z, w = (float(value) for value in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0 or not math.isfinite(norm):
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def quat_conjugate_xyzw(quat: Sequence[float]) -> Quat:
    x, y, z, w = quat_normalize_xyzw(quat)
    return (-x, -y, -z, w)


def quat_multiply_xyzw(lhs: Sequence[float], rhs: Sequence[float]) -> Quat:
    ax, ay, az, aw = quat_normalize_xyzw(lhs)
    bx, by, bz, bw = quat_normalize_xyzw(rhs)
    return quat_normalize_xyzw(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def quat_to_matrix_xyzw(quat: Sequence[float]) -> Tuple[Vector3, Vector3, Vector3]:
    x, y, z, w = quat_normalize_xyzw(quat)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def quat_rotate_xyzw(quat: Sequence[float], vector: Sequence[float]) -> Vector3:
    matrix = quat_to_matrix_xyzw(quat)
    vx, vy, vz = (float(value) for value in vector)
    return (
        matrix[0][0] * vx + matrix[0][1] * vy + matrix[0][2] * vz,
        matrix[1][0] * vx + matrix[1][1] * vy + matrix[1][2] * vz,
        matrix[2][0] * vx + matrix[2][1] * vy + matrix[2][2] * vz,
    )


def relative_pose_from_world_poses(
    parent_pos: Sequence[float],
    parent_quat_xyzw: Sequence[float],
    child_pos: Sequence[float],
    child_quat_xyzw: Sequence[float],
) -> Tuple[Vector3, Quat]:
    """Return the child pose expressed in the parent frame.

    Equivalent to ``inverse(parent) @ child`` on homogeneous transforms, written
    directly in quaternion form so the nodes do not need tf_transformations.
    """
    parent_quat = quat_normalize_xyzw(parent_quat_xyzw)
    inverse_parent_quat = quat_conjugate_xyzw(parent_quat)
    delta = (
        float(child_pos[0]) - float(parent_pos[0]),
        float(child_pos[1]) - float(parent_pos[1]),
        float(child_pos[2]) - float(parent_pos[2]),
    )
    offset = quat_rotate_xyzw(inverse_parent_quat, delta)
    quat_offset = quat_multiply_xyzw(inverse_parent_quat, child_quat_xyzw)
    return offset, quat_offset


def pose_to_arrays(pose) -> Tuple[Vector3, Quat]:
    """Split a geometry_msgs Pose or PoseStamped into position and xyzw quaternion."""
    inner = getattr(pose, 'pose', pose)
    return (
        (
            float(inner.position.x),
            float(inner.position.y),
            float(inner.position.z),
        ),
        quat_normalize_xyzw(
            (
                inner.orientation.x,
                inner.orientation.y,
                inner.orientation.z,
                inner.orientation.w,
            )
        ),
    )


def fill_pose(pose, pos: Sequence[float], quat_xyzw: Sequence[float]):
    """Write position and orientation into an existing geometry_msgs Pose."""
    pose.position.x = float(pos[0])
    pose.position.y = float(pos[1])
    pose.position.z = float(pos[2])
    quat = quat_normalize_xyzw(quat_xyzw)
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose


def mean_and_std(samples: Sequence[Sequence[float]]) -> Tuple[Vector3, Vector3]:
    """Per-axis mean and population standard deviation of 3-vectors."""
    count = len(samples)
    if count == 0:
        raise ValueError('mean_and_std requires at least one sample')
    mean = tuple(
        sum(float(sample[axis]) for sample in samples) / count for axis in range(3))
    std = tuple(
        math.sqrt(
            sum((float(sample[axis]) - mean[axis]) ** 2 for sample in samples) / count)
        for axis in range(3)
    )
    return mean, std  # type: ignore[return-value]


class TcpOffsetEstimator:
    """Measure a constant child-in-parent offset from a stream of pose pairs.

    The Flexiv driver reports both the flange pose and the active-tool TCP pose
    in the robot's own frame, which is a more trustworthy source for the control
    frame offset than a hand-entered ``tcp_offset``. Averaging a handful of
    samples rejects sensor noise; the reported translation spread tells the
    caller whether the samples were consistent enough to trust.
    """

    def __init__(self, sample_count: int, max_translation_std_m: float):
        if sample_count < 1:
            raise ValueError('sample_count must be at least 1')
        self.sample_count = int(sample_count)
        self.max_translation_std_m = float(max_translation_std_m)
        self.offset_samples: List[Vector3] = []
        self.quat_offset_samples: List[Quat] = []
        self.ready = False
        self.offset: Optional[Vector3] = None
        self.quat_offset: Optional[Quat] = None
        self.translation_std: Optional[Vector3] = None

    @property
    def collected(self) -> int:
        return len(self.offset_samples)

    def add_sample(
        self,
        parent_pos: Sequence[float],
        parent_quat_xyzw: Sequence[float],
        child_pos: Sequence[float],
        child_quat_xyzw: Sequence[float],
    ) -> bool:
        """Add one pose pair. Returns True once enough samples produced an offset."""
        if self.ready:
            return False
        offset, quat_offset = relative_pose_from_world_poses(
            parent_pos, parent_quat_xyzw, child_pos, child_quat_xyzw)
        if not all(math.isfinite(value) for value in (*offset, *quat_offset)):
            raise ValueError('non-finite TCP offset sample')
        self.offset_samples.append(offset)
        self.quat_offset_samples.append(quat_offset)
        if len(self.offset_samples) < self.sample_count:
            return False

        mean, std = mean_and_std(self.offset_samples)
        self.offset = mean
        self.translation_std = std
        self.quat_offset = self.quat_offset_samples[-1]
        self.ready = True
        return True

    @property
    def translation_std_exceeded(self) -> bool:
        if self.translation_std is None or self.max_translation_std_m <= 0.0:
            return False
        return max(self.translation_std) > self.max_translation_std_m

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
Unit tests for the virtual gantry.

The real Isaac Sim physics-tensors articulation view only exists inside a
running kit; these tests use a tiny duck-typed stand-in that records every
``apply_forces_and_torques_at_position`` invocation so we can assert
magnitude, direction and one-sided-damping behaviour against the same
math the MuJoCo plugin runs.
"""

import logging
import math
import sys
import types

from isaac_ros_deploy_isaac_sim_extension import gantry as gantry_mod
from isaac_ros_deploy_isaac_sim_extension.gantry import (
    _quat_rotate_wxyz,
    VirtualGantry,
    VirtualGantryConfig,
)
import pytest


@pytest.fixture
def gantry_logs(caplog):
    """
    Capture gantry log records regardless of logger propagation.

    Under colcon/ament the gantry logger comes pre-configured with
    ``propagate=False``, so caplog's root handler never sees its records.
    Attach the capture handler straight to the gantry logger instead.
    """
    logger = gantry_mod._LOGGER
    logger.addHandler(caplog.handler)
    prev_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(prev_level)


# --------------------------------------------------------------------------
# Fake articulation view: just enough surface for VirtualGantry to drive it.
# --------------------------------------------------------------------------
class _FakeView:
    """Minimal stand-in for the physics-tensors articulation view."""

    def __init__(self, link_transforms):
        # link_transforms: list of lists of 7-tuples (x,y,z,qx,qy,qz,qw).
        # Outer index = articulation, inner index = link slot.
        self._link_transforms = link_transforms
        self.count = len(link_transforms)
        self.max_links = max(len(row) for row in link_transforms)
        self.applied_calls = []

    def get_link_transforms(self):
        return self._link_transforms

    def apply_forces_and_torques_at_position(
        self, force_data, torque_data, position_data, indices, is_global,
    ):
        self.applied_calls.append({
            'force': [list(map(float, row)) for row in force_data[0]],
            'torque': (
                [list(map(float, row)) for row in torque_data[0]]
                if torque_data is not None else None
            ),
            'position': [list(map(float, row)) for row in position_data[0]],
            'indices': list(indices),
            'is_global': is_global,
        })


class _FakeArticulation:
    """Mimics enough of isaacsim.core.experimental.prims.Articulation."""

    def __init__(self, link_names, link_transforms):
        self.link_names = link_names
        self._physics_articulation_view = _FakeView(link_transforms)


def _make_articulation(attach_xyz, body_offset=(0.0, 0.0, 0.3)):
    """
    Build a one-link articulation positioned so the attachment matches.

    The link world transform is set so that ``link_pos + rotated body
    offset`` equals the requested ``attach_xyz``. The link is unrotated
    (identity quaternion) so the body offset is applied directly.
    """
    lx = attach_xyz[0] - body_offset[0]
    ly = attach_xyz[1] - body_offset[1]
    lz = attach_xyz[2] - body_offset[2]
    link_transforms = [[(lx, ly, lz, 0.0, 0.0, 0.0, 1.0)]]
    return _FakeArticulation(['torso_link'], link_transforms)


def _move_articulation(art, attach_xyz, body_offset=(0.0, 0.0, 0.3)):
    """Update the fake articulation's link transform to match ``attach_xyz``."""
    lx = attach_xyz[0] - body_offset[0]
    ly = attach_xyz[1] - body_offset[1]
    lz = attach_xyz[2] - body_offset[2]
    art._physics_articulation_view._link_transforms = [
        [(lx, ly, lz, 0.0, 0.0, 0.0, 1.0)]
    ]


# --------------------------------------------------------------------------
# Test cases
# --------------------------------------------------------------------------
def test_quat_rotate_identity():
    """A unit (identity) quaternion should leave the vector unchanged."""
    v = _quat_rotate_wxyz((1.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
    assert math.isclose(v[0], 0.1, abs_tol=1e-9)
    assert math.isclose(v[1], 0.2, abs_tol=1e-9)
    assert math.isclose(v[2], 0.3, abs_tol=1e-9)


def test_rope_force_zero_when_slack():
    """rope_dist < rope_length must produce no force application."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        body_offset=(0.0, 0.0, 0.3),
        kp_pos=5000.0,
        kd_pos=800.0,
        anchor_z=1.5,
        rope_length=1.0,  # explicit so we don't auto-tune to 'just taut'.
    )
    # Place attachment well inside the rope length: anchor is at z=1.5,
    # attach at z=1.0 -> rope_dist = 0.5 < 1.0, so slack.
    art = _make_articulation(attach_xyz=(0.0, 0.0, 1.0))
    g = VirtualGantry(art, cfg)
    cmd = g.step(dt=1.0 / 60.0)
    assert cmd.applied is False
    # And no force should have been pushed to the view.
    assert art._physics_articulation_view.applied_calls == []


def test_rope_force_direction():
    """Force vector should point from attachment toward anchor."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        body_offset=(0.0, 0.0, 0.3),
        kp_pos=5000.0,
        kd_pos=800.0,
        anchor_z=1.5,
        rope_length=0.5,  # taut from the first tick.
    )
    # Attach below+to-the-side of the anchor. The first step also captures
    # the anchor at attach_xy + (anchor_z); the rope is then taut.
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))
    g = VirtualGantry(art, cfg)
    # First step: capture anchor at (0, 0, 1.5), rope_length pinned to 0.5.
    # rope_dist = ||(0,0,0.5) - (0,0,1.5)|| = 1.0, > 0.5 -> taut.
    cmd = g.step(dt=1.0 / 60.0)
    assert cmd.applied is True

    # Force must point from attach (0,0,0.5) toward anchor (0,0,1.5):
    # purely +Z, and the spring should be 5000 * (1.0 - 0.5) = 2500 N.
    fx, fy, fz = cmd.force_world
    assert math.isclose(fx, 0.0, abs_tol=1e-6)
    assert math.isclose(fy, 0.0, abs_tol=1e-6)
    assert fz > 0.0
    # No damping on the first taut step (prev_rope_dist sentinel).
    assert math.isclose(fz, 2500.0, abs_tol=1e-3)


def test_one_sided_damping():
    """Damping must contribute only when the rope is extending (rope_dist_dot > 0)."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        body_offset=(0.0, 0.0, 0.3),
        kp_pos=5000.0,
        kd_pos=800.0,
        anchor_z=1.5,
        rope_length=0.5,
        ema_alpha=1.0,  # no smoothing -> rope_dist_dot == raw FD.
    )
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))  # rope_dist = 1.0
    g = VirtualGantry(art, cfg)
    dt = 1.0 / 60.0
    # Tick 1: capture + first taut step, no FD yet.
    g.step(dt)

    # Tick 2: move attach DOWN by 0.06 m -> rope_dist grows from 1.0 to 1.06,
    # rope_dist_dot = +0.06 / dt > 0, so damp = kd * rate.
    _move_articulation(art, attach_xyz=(0.0, 0.0, 0.44))
    cmd_extending = g.step(dt)
    assert cmd_extending.applied is True
    rate = 0.06 / dt
    expected_damp = 800.0 * rate
    expected_spring = 5000.0 * (1.06 - 0.5)
    fz_expected = expected_spring + expected_damp
    assert math.isclose(
        cmd_extending.force_world[2], fz_expected, rel_tol=1e-5,
    )

    # Tick 3: move attach back UP by 0.04 m so rope_dist shrinks from 1.06
    # to 1.02 -> rope_dist_dot < 0, damping is ONE-SIDED and contributes 0.
    _move_articulation(art, attach_xyz=(0.0, 0.0, 0.48))
    cmd_contracting = g.step(dt)
    assert cmd_contracting.applied is True
    fz_no_damp = 5000.0 * (1.02 - 0.5)
    assert math.isclose(
        cmd_contracting.force_world[2], fz_no_damp, rel_tol=1e-5,
    )


def test_force_matches_mujoco_plugin():
    """Replicate the plugin's formula (virtual_gantry_plugin.cpp:187-201)."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        body_offset=(0.0, 0.0, 0.3),
        kp_pos=5000.0,
        kd_pos=800.0,
        anchor_z=1.5,
        rope_length=0.5,
        ema_alpha=1.0,  # match the "single-tick raw FD" we'll compute by hand.
    )
    art = _make_articulation(attach_xyz=(0.1, -0.2, 0.4))
    g = VirtualGantry(art, cfg)
    dt = 1.0 / 240.0  # MuJoCo-like step

    # Tick 1 captures anchor and writes rope_length, no FD yet.
    g.step(dt)

    # Tick 2: shift attach to a known new pose; compute reference force.
    new_attach = (0.12, -0.18, 0.38)
    _move_articulation(art, attach_xyz=new_attach)
    cmd = g.step(dt)
    assert cmd.applied is True

    # Anchor was captured on tick 1 at (0.1, -0.2, 1.5); rope_length = 0.5.
    anchor = (0.1, -0.2, 1.5)
    dx = new_attach[0] - anchor[0]
    dy = new_attach[1] - anchor[1]
    dz = new_attach[2] - anchor[2]
    rope_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    # First taut tick's rope_dist (used as previous reference).
    prev_dx = 0.1 - anchor[0]
    prev_dy = -0.2 - anchor[1]
    prev_dz = 0.4 - anchor[2]
    prev_rope_dist = math.sqrt(
        prev_dx * prev_dx + prev_dy * prev_dy + prev_dz * prev_dz,
    )
    rope_dist_dot = (rope_dist - prev_rope_dist) / dt
    damp = 800.0 * rope_dist_dot if rope_dist_dot > 0 else 0.0
    tension = 5000.0 * (rope_dist - 0.5) + damp
    fx_ref = -tension * dx / rope_dist
    fy_ref = -tension * dy / rope_dist
    fz_ref = -tension * dz / rope_dist
    assert math.isclose(cmd.force_world[0], fx_ref, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(cmd.force_world[1], fy_ref, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(cmd.force_world[2], fz_ref, rel_tol=1e-6, abs_tol=1e-6)


def test_apply_pushed_to_articulation_view():
    """The wrench should reach the underlying tensor view at the right slot."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        rope_length=0.5,
        ema_alpha=1.0,
    )
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))
    g = VirtualGantry(art, cfg)
    g.step(1.0 / 60.0)
    calls = art._physics_articulation_view.applied_calls
    assert len(calls) == 1
    call = calls[0]
    assert call['is_global'] is True
    # Forces non-zero only on link 0 (torso_link).
    assert call['force'][0][2] > 0.0
    # Position recorded should equal the attachment point in world frame.
    assert math.isclose(call['position'][0][2], 0.5, abs_tol=1e-6)


def test_unknown_body_disables_gantry():
    """If body_name is not in link_names, gantry must refuse to enable."""
    cfg = VirtualGantryConfig(body_name='nonexistent_link', enabled=True)
    art = _FakeArticulation(
        ['torso_link'], [[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]],
    )
    g = VirtualGantry(art, cfg)
    assert g.is_enabled() is False
    cmd = g.step(1.0 / 60.0)
    assert cmd.applied is False


class _FakeWarpArray:
    """
    Tiny stand-in for ``wp.array`` that records shape/dtype/device.

    The real warp array is a thin wrapper over a device buffer; for the
    regression test we only need to capture the inputs the view receives
    so we can assert types end up correct (the original bug was passing
    raw numpy, which the PhysX warp frontend rejects with the
    ``issubclass`` traceback the gantry self-disabled on).
    """

    def __init__(self, data, dtype, device, shape=None):
        self.data = data
        self.dtype = dtype
        self.device = device
        self.shape = shape if shape is not None else _infer_shape(data)

    def __getitem__(self, key):
        # Mimic the slice semantics used by VirtualGantry._scatter_*:
        # forces[0, idx:idx + 1] is treated as a writable subview. We
        # return a sentinel object that wp.copy() can write into.
        return _FakeWarpSlice(self, key)


class _FakeWarpSlice:
    """Sentinel returned by ``_FakeWarpArray.__getitem__`` for wp.copy."""

    def __init__(self, parent, key):
        self.parent = parent
        self.key = key


class _FakeWarp(types.ModuleType):
    """Minimal warp module surface for the regression test."""

    def __init__(self):
        super().__init__('warp_fake')
        self.float32 = 'float32'
        self.int32 = 'int32'
        self.copies = []
        self.zeros_calls = []
        self.array_calls = []

    def zeros(self, shape, dtype=None, device=None):
        self.zeros_calls.append((shape, dtype, device))
        return _FakeWarpArray(None, dtype, device, shape=shape)

    def array(self, data, dtype=None, device=None):
        self.array_calls.append((data, dtype, device))
        return _FakeWarpArray(data, dtype, device)

    def copy(self, dst, src):
        self.copies.append((dst, src))

    def get_device(self):
        return 'cuda:0'


def _infer_shape(data):
    """Best-effort shape probe for nested Python lists (test helper)."""
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    return (len(data),) + _infer_shape(data[0])


def test_warp_path_builds_correct_tensors(monkeypatch):
    """
    Check the view call gets warp arrays of the right dtype/shape.

    This is the regression test for the original bug ('issubclass() arg
    1 must be a class'): the gantry was handing numpy buffers to the
    view, which the PhysX warp frontend rejected. Inject a fake warp
    module, run a single taut tick, and assert the view's
    apply_forces_and_torques_at_position got float32 buffers of shape
    (count, max_links, 3) plus an int32 indices array.
    """
    fake_wp = _FakeWarp()
    # Force _import_warp() to hand back our fake module.
    monkeypatch.setattr(gantry_mod._import_warp, '_module', fake_wp, raising=False)
    monkeypatch.setitem(sys.modules, 'warp', fake_wp)

    cfg = VirtualGantryConfig(
        body_name='torso_link',
        rope_length=0.5,
        ema_alpha=1.0,
    )
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))

    # Capture the args the fake view receives.
    captured = {}
    orig = art._physics_articulation_view.apply_forces_and_torques_at_position

    def _capturing(force_data, torque_data, position_data, indices, is_global):
        captured['force'] = force_data
        captured['torque'] = torque_data
        captured['position'] = position_data
        captured['indices'] = indices
        captured['is_global'] = is_global

    art._physics_articulation_view.apply_forces_and_torques_at_position = _capturing

    g = VirtualGantry(art, cfg)
    g.step(1.0 / 60.0)
    # Reset the bound method so other tests aren't affected (the fixture
    # tears the article down at function exit, but be explicit).
    art._physics_articulation_view.apply_forces_and_torques_at_position = orig

    # The forces / positions tensors should be _FakeWarpArray (i.e. our
    # warp build path ran, not the numpy fallback).
    assert isinstance(captured['force'], _FakeWarpArray)
    assert isinstance(captured['position'], _FakeWarpArray)
    assert isinstance(captured['indices'], _FakeWarpArray)

    # dtypes: float32 buffers + int32 indices.
    assert captured['force'].dtype == 'float32'
    assert captured['position'].dtype == 'float32'
    assert captured['indices'].dtype == 'int32'

    # shape (count=1, max_links=1, 3) for force/pos; (1,) for indices.
    assert captured['force'].shape == (1, 1, 3)
    assert captured['position'].shape == (1, 1, 3)
    assert captured['indices'].shape == (1,)

    # is_global must be True.
    assert captured['is_global'] is True

    # The scatter step ran twice (force + position).
    assert len(fake_wp.copies) == 2


def test_rope_length_override_wins(gantry_logs):
    """
    cfg.rope_length must win over auto-compute, regardless of physics state.

    Regression for VG3: at cold launch the robot can fall during the
    pre-actuator warmup window; without the override the auto-computed
    rope length would catch the robot at floor level. The override lets
    the operator pin the rope to a useful value (e.g. 0.3 m for G1).
    """
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        body_offset=(0.0, 0.0, 0.3),
        anchor_z=1.5,
        rope_length=0.3,
    )
    # Place the robot wherever — the override should ignore the world Z.
    art = _make_articulation(attach_xyz=(0.0, 0.0, -0.5))
    g = VirtualGantry(art, cfg)
    g.step(1.0 / 60.0)
    assert math.isclose(g.current_rope_length(), 0.3, abs_tol=1e-9)
    assert 'rope_length override' in gantry_logs.text


def test_rope_length_warns_on_fallen_robot(gantry_logs):
    """
    Auto-compute path must warn when the torso is sitting on the floor.

    Without the warning, an operator reading the log won't realise the
    captured rope length latched onto an already-collapsed robot.
    """
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        body_offset=(0.0, 0.0, 0.3),
        anchor_z=1.5,
        rope_length=None,  # force auto-compute path
    )
    # Attach z = -0.5 m: robot is below the ground plane.
    art = _make_articulation(attach_xyz=(0.0, 0.0, -0.5))
    g = VirtualGantry(art, cfg)
    g.step(1.0 / 60.0)
    assert 'looks like the robot is already on the floor' in gantry_logs.text
    assert 'set rope_length explicitly' in gantry_logs.text


def test_rope_length_clamped_to_max(gantry_logs):
    """
    Absurdly-far attach points must clamp the rope to 1.5 * |anchor_z|.

    Without the clamp the spring would latch the robot at whatever depth
    physics dragged it to, which defeats the gantry's purpose entirely.
    """
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        body_offset=(0.0, 0.0, 0.3),
        anchor_z=1.5,
        rope_length=None,  # force auto-compute path
    )
    # attach_z = -10 m -> raw rope_length would be ~11.5 m; clamp at 2.25 m.
    art = _make_articulation(attach_xyz=(0.0, 0.0, -10.0))
    g = VirtualGantry(art, cfg)
    g.step(1.0 / 60.0)
    expected_clamp = 1.5 * abs(1.5)
    assert math.isclose(
        g.current_rope_length(), expected_clamp, abs_tol=1e-9,
    )
    assert 'exceeds clamp' in gantry_logs.text


def test_toggle_disables_then_reenables():
    """``toggle()`` must flip the enabled flag in either direction."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        rope_length=0.5,
        ema_alpha=1.0,
    )
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))
    g = VirtualGantry(art, cfg)
    # Constructor enables by default; one toggle should disable.
    assert g.is_enabled() is True
    g.toggle()
    assert g.is_enabled() is False
    # Second toggle re-enables.
    g.toggle()
    assert g.is_enabled() is True


def test_adjust_rope_length_increments():
    """A small positive delta must increase ``current_rope_length`` by that delta."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        rope_length=0.5,
        ema_alpha=1.0,
    )
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))
    g = VirtualGantry(art, cfg)
    # First step captures the rope length at the configured override (0.5).
    g.step(1.0 / 60.0)
    initial = g.current_rope_length()
    assert math.isclose(initial, 0.5, abs_tol=1e-9)

    g.adjust_rope_length(+0.01)
    assert math.isclose(
        g.current_rope_length(), initial + 0.01, abs_tol=1e-9,
    )


def test_adjust_rope_length_clamps_min():
    """Deltas that would drop rope length below ``min_rope_length`` must clamp."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        rope_length=0.15,  # close enough to the 0.1 floor that one big step trips it
        min_rope_length=0.1,
        ema_alpha=1.0,
    )
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))
    g = VirtualGantry(art, cfg)
    g.step(1.0 / 60.0)

    g.adjust_rope_length(-1.0)  # would go to -0.85; must clamp to 0.1
    assert math.isclose(g.current_rope_length(), 0.1, abs_tol=1e-9)


def test_adjust_rope_length_clamps_max():
    """Deltas above ``1.5 * |anchor_z|`` must clamp to that ceiling."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        anchor_z=1.5,
        rope_length=0.5,
        ema_alpha=1.0,
    )
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))
    g = VirtualGantry(art, cfg)
    g.step(1.0 / 60.0)

    g.adjust_rope_length(+10.0)  # would go to 10.5; ceiling is 1.5 * 1.5 = 2.25
    assert math.isclose(g.current_rope_length(), 2.25, abs_tol=1e-9)


def test_adjust_rope_length_noop_when_disabled():
    """When the gantry is disabled, ``adjust_rope_length`` must not mutate state."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        rope_length=0.5,
        ema_alpha=1.0,
    )
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))
    g = VirtualGantry(art, cfg)
    g.step(1.0 / 60.0)

    g.disable()
    # Snapshot after disable: disable() now clears rope_length to nan so
    # the next enable() recaptures it cleanly. adjust_rope_length must be
    # an exact no-op against that post-disable state, regardless of value.
    before = g._rope_length
    g.adjust_rope_length(+0.5)
    after = g._rope_length
    # Direct equality fails for nan == nan; compare via math.isnan + value.
    if math.isnan(before):
        assert math.isnan(after)
    else:
        assert after == before


def test_disable_then_enable_re_anchors():
    """Disable + re-enable should drop the captured anchor."""
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        rope_length=0.5,
        ema_alpha=1.0,
    )
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))
    g = VirtualGantry(art, cfg)
    g.step(1.0 / 60.0)
    assert g.is_enabled() is True
    g.disable()
    assert g.is_enabled() is False
    # Move the robot before re-enabling; the anchor must follow.
    _move_articulation(art, attach_xyz=(0.7, 0.7, 0.5))
    g.enable()
    # Calls so far: 1 from the pre-disable step. After enable + step,
    # the new anchor should be (0.7, 0.7, anchor_z).
    g.step(1.0 / 60.0)
    assert g._anchor[0] == 0.7
    assert g._anchor[1] == 0.7


def test_enable_disable_enable_re_resolves_body_index():
    """
    Re-enable must re-resolve the body index against current link_names.

    Timeline Stop can tear down and rebuild the underlying articulation
    view, so the body_index cached from a prior enable() may point at the
    wrong link slot in the rebuilt view. enable() must always re-resolve
    rather than trusting a stale cache.
    """
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        rope_length=0.5,
        ema_alpha=1.0,
    )
    # Initial articulation has torso_link at index 0.
    art = _FakeArticulation(
        ['torso_link', 'pelvis_link'],
        [[
            (-0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),  # torso_link
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),   # pelvis_link
        ]],
    )
    g = VirtualGantry(art, cfg)
    assert g.is_enabled() is True
    assert g._body_index == 0

    g.disable()
    assert g._body_index is None

    # Simulate the articulation view being rebuilt with link_names in a
    # DIFFERENT order — now torso_link is at index 1.
    art.link_names = ['pelvis_link', 'torso_link']
    art._physics_articulation_view._link_transforms = [[
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),    # pelvis_link
        (-0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),   # torso_link
    ]]

    g.enable()
    # Without the fix, body_index would remain 0 (pelvis_link) — wrong.
    assert g._body_index == 1
    assert g.is_enabled() is True


def test_transient_read_failure_recovers_on_reenable():
    """
    Read failure must clear body_index so future enable() re-resolves.

    On IndexError from _read_attach_world, _enabled flips False AND
    _body_index is cleared. Without the clear, a future enable() would
    skip the body-name resolve step and the gantry would stay broken.
    """
    cfg = VirtualGantryConfig(
        body_name='torso_link',
        rope_length=0.5,
        ema_alpha=1.0,
    )
    art = _make_articulation(attach_xyz=(0.0, 0.0, 0.5))
    g = VirtualGantry(art, cfg)
    g.step(1.0 / 60.0)
    assert g.is_enabled() is True
    captured_index = g._body_index
    assert captured_index is not None

    # Induce a transient read failure: hand back a view with no link rows
    # so arr[0][body_index] raises IndexError.
    art._physics_articulation_view._link_transforms = [[]]
    cmd = g.step(1.0 / 60.0)
    assert cmd.applied is False
    assert g.is_enabled() is False
    assert g._body_index is None  # cleared, so future enable() re-resolves

    # Repair the view and re-enable; the gantry should recover.
    _move_articulation(art, attach_xyz=(0.0, 0.0, 0.5))
    g.enable()
    assert g.is_enabled() is True
    assert g._body_index == 0

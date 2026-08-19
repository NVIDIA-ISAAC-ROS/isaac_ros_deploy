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
Virtual gantry: a per-step external-wrench "rope" suspending the robot.

Mirrors ``mujoco_ros2_control_plugins/virtual_gantry_plugin.cpp`` so the
behaviour matches the MuJoCo path one-for-one. The constraint is purely
runtime — no USD joint, no kinematic anchor — implemented as a
pre-physics callback that applies an external wrench to a single body of
the articulation. All 6 base DOFs stay free; the rope only catches the
robot when extended beyond ``rope_length``.

Force law (matches ``virtual_gantry_plugin.cpp:187-201``):

* ``rope_vec = anchor - attach``, ``rope_dist = ||rope_vec||``
* Finite-difference + EMA-smoothed ``rope_dist_dot`` (alpha = 0.2).
* If ``rope_dist > rope_length``:
    ``extension = rope_dist - rope_length``
    ``damp = kd_pos * rope_dist_dot`` if ``rope_dist_dot > 0`` else 0
    ``tension = kp_pos * extension + damp``
    ``force_world = -tension * rope_vec / rope_dist``  (toward anchor)
* Else: zero force.

The force is applied via the underlying PhysX-tensors articulation view's
``apply_forces_and_torques_at_position`` because the experimental
``Articulation`` class itself does not expose per-link external-force
application. The view-level API is identical between PhysX and Newton
backends, so the gantry physics matches across both. Inputs to that
view call are built as ``warp.array`` tensors directly (float32 for
forces/positions, int32 for indices) — the PhysX warp frontend probes
``tensor.dtype`` via ``wp.types.type_ctype`` and the Newton path
rebuilds ``wp.array`` from arbitrary inputs, so a warp-typed buffer
is the one shape both accept without raising.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional, Tuple

# Plain stdlib logging so this module stays importable outside Kit (the test
# suite runs it without carb); set the level via the root logger as usual.
_LOGGER = logging.getLogger(__name__)


@dataclass
class VirtualGantryConfig:
    """
    Configuration for :class:`VirtualGantry`.

    The defaults mirror the canonical ``mujoco_plugins.virtual_gantry``
    block in ``unitree_g1_bringup/config/mujoco_pid.yaml``.
    """

    body_name: str = 'torso_link'
    body_offset: Tuple[float, float, float] = (0.0, 0.0, 0.3)
    kp_pos: float = 5000.0
    kd_pos: float = 800.0
    anchor_z: float = 1.5
    enabled: bool = True
    rope_length: Optional[float] = None
    ema_alpha: float = 0.2
    min_rope_length: float = 0.1
    visualize: bool = True


def _quat_rotate_wxyz(q: Tuple[float, float, float, float],
                      v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Rotate vector ``v`` by quaternion ``q`` given in ``(w, x, y, z)`` order.

    This is the standard ``q * v * q^-1`` formula expanded into scalar
    arithmetic, kept dependency-free so tests can run outside an Isaac
    Sim kit.
    """
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * (q.xyz x v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + (q.xyz x t)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


@dataclass
class _StepCommand:
    """
    Wrench produced by one :meth:`VirtualGantry.step` invocation.

    Exposed to keep the math pure-Python and easy to unit-test without
    pulling in any Isaac Sim runtime. ``applied`` is False when the rope
    is slack (or the gantry disabled), in which case ``force_world``,
    ``torque_world`` and ``position_world`` are all zero-tuples and the
    caller should skip the API call entirely.
    """

    applied: bool = False
    force_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class VirtualGantry:
    """
    One-sided spring-damper rope suspending an articulation body.

    The class is a thin coordinator: on every physics tick it reads the
    target body's world pose, evaluates the rope force, and pushes an
    external wrench into the underlying physics-tensors articulation
    view. State is held in plain Python floats; warp/numpy is only used
    to ferry numbers across the runtime boundary.

    The ``articulation`` argument is a live
    :class:`isaacsim.core.experimental.prims.Articulation` (or a
    duck-typed stand-in exposing ``link_names`` and
    ``_physics_articulation_view`` with the standard PhysX-tensors
    surface); it is stored by reference and not owned. The ``cfg``
    argument is the static configuration; the instance keeps a private
    copy of the relevant fields so live mutation of ``cfg`` after
    construction has no effect.
    """

    def __init__(self, articulation: Any, cfg: VirtualGantryConfig) -> None:
        self._articulation = articulation
        self._cfg = cfg

        # Mirrors of cfg, kept mutable for runtime tweaks (e.g. rope_length
        # adjustment in a future revision).
        self._body_name: str = cfg.body_name
        self._body_offset: Tuple[float, float, float] = tuple(cfg.body_offset)
        self._kp_pos: float = float(cfg.kp_pos)
        self._kd_pos: float = float(cfg.kd_pos)
        self._anchor_z: float = float(cfg.anchor_z)
        self._ema_alpha: float = float(cfg.ema_alpha)
        self._min_rope_length: float = float(cfg.min_rope_length)
        self._configured_rope_length: Optional[float] = cfg.rope_length
        self._visualize: bool = bool(cfg.visualize)

        # Runtime state.
        self._enabled: bool = False
        self._body_index: Optional[int] = None
        self._anchor: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._rope_length: float = float('nan')
        self._prev_rope_dist: float = -1.0  # -1 = not yet captured
        self._rope_dist_dot: float = 0.0
        self._spawn_pos_captured: bool = False

        # Rope + anchor visualization prims, authored lazily on first draw;
        # _UNSET until then, None when no USD stage is available (headless
        # without a stage / unit tests).
        self._viz: Any = _UNSET

        if cfg.enabled:
            self.enable()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enable(self) -> None:
        """
        Enable the gantry, re-anchoring above the current body position.

        Resolves the body index from ``articulation.link_names`` on every
        call: timeline Stop can tear down and rebuild the underlying
        articulation view, so a body_index cached from a previous enable()
        may point at the wrong link slot in the new view. Idempotent over
        the articulation view's lifecycle — failure paths clear
        ``_body_index`` so a subsequent enable() re-validates.

        If the body is not in the list, logs a clear error and leaves the
        gantry disabled. The first :meth:`step` call after enable
        captures the anchor (and ``rope_length`` if not explicitly set).
        """
        try:
            link_names = list(self._articulation.link_names)
        except Exception as exc:  # noqa: BLE001 — defensive boundary
            _LOGGER.warning(
                f'[virtual-gantry] cannot read link_names: {exc!r}; '
                f'staying disabled',
            )
            self._enabled = False
            self._body_index = None
            return
        try:
            self._body_index = link_names.index(self._body_name)
        except ValueError:
            _LOGGER.warning(
                f"[virtual-gantry] body '{self._body_name}' not found "
                f'in articulation links {link_names!r}; staying disabled',
            )
            self._enabled = False
            self._body_index = None
            return

        self._enabled = True
        # Force a fresh anchor on the next step; reset finite-difference
        # state so the first taut step doesn't see a stale derivative.
        self._spawn_pos_captured = False
        self._prev_rope_dist = -1.0
        self._rope_dist_dot = 0.0
        _LOGGER.info(
            f"[virtual-gantry] enabled on body '{self._body_name}' "
            f'(link_index={self._body_index}, anchor_z={self._anchor_z:.3f}, '
            f'kp={self._kp_pos:.1f}, kd={self._kd_pos:.1f})',
        )

    def current_rope_length(self) -> float:
        """Return the active rope length, or nan before the first step captures it."""
        if not self._enabled or not self._spawn_pos_captured:
            return float('nan')
        return self._rope_length

    def disable(self) -> None:
        """Turn the gantry off; no force is applied on subsequent steps."""
        if self._enabled:
            _LOGGER.info('[virtual-gantry] disabled')
        self._enabled = False
        # Clear cached articulation-view state so a future enable() must
        # re-resolve the body index against the (possibly rebuilt) view.
        # Timeline Stop tears down the underlying view; without this reset
        # the cached index could point at the wrong link on the next Play.
        self._body_index = None
        # Reset FD state so a future re-enable starts fresh.
        self._prev_rope_dist = -1.0
        self._rope_dist_dot = 0.0
        self._spawn_pos_captured = False
        self._anchor = (0.0, 0.0, 0.0)
        self._rope_length = float('nan')
        self._clear_gantry_drawing()

    def is_enabled(self) -> bool:
        """Return True if the gantry is currently applying forces."""
        return self._enabled

    def toggle(self) -> None:
        """
        Flip the enabled state.

        Re-anchors on enable so the next play uses the current pose; mirrors
        the MuJoCo virtual_gantry_plugin's ``G`` hotkey behaviour.
        """
        if self._enabled:
            self.disable()
        else:
            self.enable()

    def adjust_rope_length(self, delta_meters: float) -> None:
        """
        Lengthen (delta>0) or shorten (delta<0) the rope at runtime.

        Clamps to ``[min_rope_length, 1.5 * |anchor_z|]`` to match the bounds
        the auto-compute path uses on first capture. No-op when the gantry
        is disabled or before the first step has captured the anchor. Logs
        only on actual change so holding the key at a clamp limit doesn't
        spam the terminal.
        """
        if not self._enabled:
            return
        if not self._spawn_pos_captured:
            return  # rope length not yet captured; nothing meaningful to adjust
        new_len = self._rope_length + float(delta_meters)
        lower = self._min_rope_length
        upper = 1.5 * abs(self._anchor_z)
        new_len = max(lower, min(upper, new_len))
        if new_len != self._rope_length:
            self._rope_length = new_len
            _LOGGER.info(
                f'[virtual-gantry] rope_length adjusted: '
                f'{self._rope_length:.3f} m',
            )

    def setup_hotkeys(self):
        """
        Bind G / [ / ] keyboard hotkeys that drive this gantry.

        Mirrors the MuJoCo ``virtual_gantry_plugin`` hotkeys: ``G``
        (press) toggles enable/disable, ``[`` (press + repeat) shortens
        the rope by 5 mm, ``]`` (press + repeat) lengthens it by 5 mm.

        The Isaac Sim window must have keyboard focus for events to
        fire; carb's input bus only delivers events to the active app
        window. Returns the carb subscription handle so the caller can
        keep it alive for the duration of the run and unsubscribe in
        their cleanup. Returns ``None`` when the carb input interface
        is unavailable (e.g. headless runs where ``omni.appwindow`` has
        no default window) -- the gantry continues to work, just
        without the keyboard affordance.
        """
        try:
            import carb.input  # type: ignore[import-not-found]
            import omni.appwindow  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                f'[virtual-gantry] cannot bind hotkeys '
                f'(carb.input/omni.appwindow unavailable: {exc!r})',
            )
            return None

        try:
            input_iface = carb.input.acquire_input_interface()
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                f'[virtual-gantry] cannot acquire keyboard for hotkeys '
                f'({exc!r})',
            )
            return None

        KEY = carb.input.KeyboardInput
        toggle_key = KEY.G
        shorter_key = KEY.LEFT_BRACKET
        longer_key = KEY.RIGHT_BRACKET
        step = 0.005  # 5 mm rope-length step per [/] press

        press_type = carb.input.KeyboardEventType.KEY_PRESS
        repeat_type = carb.input.KeyboardEventType.KEY_REPEAT

        def _on_keyboard(event, *_a, **_kw):
            et = event.type
            press = et == press_type
            repeat = et == repeat_type
            # Only press/repeat matter; release events would double-fire toggles.
            if not (press or repeat):
                return True
            key = event.input
            if key == toggle_key:
                # Toggle on press only; KEY_REPEAT would flip back and forth
                # while the operator holds G, which is never what they want.
                if press:
                    self.toggle()
                    state = 'ENABLED' if self.is_enabled() else 'DISABLED'
                    _LOGGER.info(
                        f'[virtual-gantry] toggle via hotkey: now {state}',
                    )
            elif key == shorter_key:
                self.adjust_rope_length(-step)
            elif key == longer_key:
                self.adjust_rope_length(+step)
            return True

        try:
            sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                f'[virtual-gantry] subscribe_to_keyboard_events failed '
                f'({exc!r}); hotkeys disabled',
            )
            return None

        _LOGGER.info(
            f'[virtual-gantry] hotkeys: G=toggle, '
            f'[/-{step:.3f}m, ]/+{step:.3f}m',
        )
        return sub

    def step(self, dt: float) -> _StepCommand:
        """
        Run one tick of the rope force law and push the wrench to the body.

        ``dt`` is the physics timestep in seconds; it must be positive
        when the rope is taut for the EMA to update, and values <= 1e-9
        are treated as "no time elapsed" so the previous EMA state is
        reused. The return value is the wrench computed for this step;
        its ``applied`` flag is ``False`` when the rope is slack or the
        gantry is disabled.
        """
        if not self._enabled or self._body_index is None:
            return _StepCommand()

        attach_world = self._read_attach_world()
        if attach_world is None:
            return _StepCommand()

        # First-step anchor capture (mirrors the MuJoCo plugin).
        if not self._spawn_pos_captured:
            # Anchor x/y track the attachment; z is fixed to anchor height.
            ax, ay, _az = attach_world
            self._anchor = (ax, ay, self._anchor_z)

            # rope_length: an explicit override wins; otherwise derive it
            # from the current attach height, clamped, warning if the robot
            # already looks collapsed.
            if self._configured_rope_length is not None:
                self._rope_length = max(
                    self._min_rope_length, float(self._configured_rope_length),
                )
                _LOGGER.info(
                    f'[virtual-gantry] rope_length override: '
                    f'{self._rope_length:.3f} m',
                )
            else:
                attach_z = float(attach_world[2])
                if attach_z < 0.3:
                    _LOGGER.warning(
                        f'[virtual-gantry] attach z={attach_z:.3f} m '
                        f'looks like the robot is already on the floor; set '
                        f'rope_length explicitly if the auto value is wrong.',
                    )
                rope_length = abs(self._anchor_z - attach_z)
                max_rope_length = 1.5 * abs(self._anchor_z)
                if rope_length > max_rope_length:
                    _LOGGER.warning(
                        f'[virtual-gantry] auto rope_length {rope_length:.3f} m '
                        f'exceeds clamp {max_rope_length:.3f} m; clamping.',
                    )
                    rope_length = max_rope_length
                self._rope_length = max(self._min_rope_length, rope_length)

            self._spawn_pos_captured = True
            self._prev_rope_dist = -1.0
            self._rope_dist_dot = 0.0
            _LOGGER.info(
                f'[virtual-gantry] anchor=({self._anchor[0]:.3f}, '
                f'{self._anchor[1]:.3f}, {self._anchor[2]:.3f}), '
                f'rope_length={self._rope_length:.3f} m',
            )

        # Rope vector points FROM anchor TO attachment. The force toward
        # the anchor is therefore -tension * rope_vec / rope_dist, which
        # matches the plugin's sign convention.
        dx = attach_world[0] - self._anchor[0]
        dy = attach_world[1] - self._anchor[1]
        dz = attach_world[2] - self._anchor[2]
        rope_dist = (dx * dx + dy * dy + dz * dz) ** 0.5

        if rope_dist < 1e-6 or rope_dist <= self._rope_length:
            # Slack or degenerate; reset FD state so the first taut step
            # has no stale derivative spike.
            self._prev_rope_dist = -1.0
            self._rope_dist_dot = 0.0
            self._draw_gantry(attach_world, taut=False)
            return _StepCommand()

        # EMA-smoothed rope-extension rate. Skip the update when no time
        # has elapsed or when this is the first taut tick.
        if self._prev_rope_dist >= 0.0 and dt > 1e-9:
            raw_dot = (rope_dist - self._prev_rope_dist) / dt
            self._rope_dist_dot = (
                self._ema_alpha * raw_dot
                + (1.0 - self._ema_alpha) * self._rope_dist_dot
            )
        self._prev_rope_dist = rope_dist

        extension = rope_dist - self._rope_length
        damp = self._kd_pos * self._rope_dist_dot if self._rope_dist_dot > 0.0 else 0.0
        tension = self._kp_pos * extension + damp

        inv = 1.0 / rope_dist
        fx = -tension * dx * inv
        fy = -tension * dy * inv
        fz = -tension * dz * inv

        cmd = _StepCommand(
            applied=True,
            force_world=(fx, fy, fz),
            torque_world=(0.0, 0.0, 0.0),
            position_world=attach_world,
        )
        self._apply_to_articulation(cmd)
        self._draw_gantry(attach_world, taut=True)
        return cmd

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _read_attach_world(
        self,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Read the attachment-point world position via the tensor view.

        Returns ``None`` if the view is not yet available (e.g. called
        before the simulation has ticked) or the read fails.
        """
        view = getattr(self._articulation, '_physics_articulation_view', None)
        if view is None:
            return None
        try:
            link_transforms = view.get_link_transforms()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                f'[virtual-gantry] get_link_transforms failed: {exc!r}; '
                f'disabling',
            )
            self._enabled = False
            return None

        # link_transforms shape: (count, max_links, 7); quaternion is xyzw.
        # Single articulation, so robot index 0.
        arr = _to_host_array(link_transforms)
        if arr is None:
            return None
        try:
            row = arr[0][self._body_index]
        except (IndexError, KeyError, TypeError):
            _LOGGER.warning(
                f'[virtual-gantry] link index {self._body_index} out of '
                f'range for link_transforms; disabling',
            )
            self._enabled = False
            # Clear the cached index so a future enable() re-resolves it.
            self._body_index = None
            return None

        px, py, pz = float(row[0]), float(row[1]), float(row[2])
        qx, qy, qz, qw = (
            float(row[3]), float(row[4]), float(row[5]), float(row[6]),
        )
        # Body-local offset to world: rot * offset + pos.
        ox, oy, oz = _quat_rotate_wxyz((qw, qx, qy, qz), self._body_offset)
        return (px + ox, py + oy, pz + oz)

    def _apply_to_articulation(self, cmd: _StepCommand) -> None:
        """
        Push the wrench into the underlying tensor articulation view.

        The tensor view's ``apply_forces_and_torques_at_position`` is
        backend-sensitive about input types. The PhysX-tensors warp
        frontend probes ``tensor.dtype`` via ``wp.types.type_ctype`` —
        which only works on warp dtypes, not numpy dtypes — and the
        Newton backend's ``_wrap_input_tensor`` ultimately rebuilds
        ``wp.array`` from whatever it gets. Either way the safest input
        is a pre-built ``wp.array`` of the exact dtype/shape the
        underlying kernel reads. We mirror the conversion pattern from
        ``isaacsim.core.experimental.prims.RigidPrim`` here: build a
        scatter array of zeros, drop our single non-zero row into the
        target link slot, then call the low-level API.
        """
        view = getattr(self._articulation, '_physics_articulation_view', None)
        if view is None or self._body_index is None:
            return

        try:
            count = int(view.count)
            max_links = int(view.max_links)
        except AttributeError:
            count, max_links = 1, max(self._body_index + 1, 1)

        wp = _import_warp()
        if wp is None:
            # Test-only path: feed plain Python lists; the fake view in
            # test_gantry.py records the call without invoking warp.
            forces = [[[0.0, 0.0, 0.0] for _ in range(max_links)] for _ in range(count)]
            positions = [[[0.0, 0.0, 0.0] for _ in range(max_links)] for _ in range(count)]
            forces[0][self._body_index] = list(cmd.force_world)
            positions[0][self._body_index] = list(cmd.position_world)
            indices = [0]
            try:
                view.apply_forces_and_torques_at_position(
                    forces, None, positions, indices, True,
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    f'[virtual-gantry] apply_forces_and_torques_at_position '
                    f'failed: {exc!r}; disabling',
                )
                self._enabled = False
                self._body_index = None
            return

        device = self._resolve_device(view, wp)

        # Build the scatter buffers as warp arrays of the exact shape
        # and dtype the underlying kernel expects:
        #   forces / positions: float32, shape (count, max_links, 3)
        #   indices:            int32,   shape (1,)
        # int32 satisfies both the PhysX uint32/int32 check and the
        # Newton int32/int64 overload set.
        try:
            forces = wp.zeros((count, max_links, 3), dtype=wp.float32, device=device)
            positions = wp.zeros((count, max_links, 3), dtype=wp.float32, device=device)
            indices = wp.array([0], dtype=wp.int32, device=device)
            # Scatter the single non-zero slot via host->device numpy round-trip.
            # warp doesn't let us assign a Python tuple directly into an
            # indexed slice, but it does accept wp.from_numpy through wp.copy.
            self._scatter_force_and_position(
                wp, forces, positions, cmd.force_world, cmd.position_world, device,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                f'[virtual-gantry] failed to build warp tensors: {exc!r}; '
                f'disabling',
            )
            self._enabled = False
            self._body_index = None
            return

        try:
            view.apply_forces_and_torques_at_position(
                forces, None, positions, indices, True,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                f'[virtual-gantry] apply_forces_and_torques_at_position '
                f'failed: {exc!r}; disabling',
            )
            self._enabled = False
            self._body_index = None

    def _scatter_force_and_position(
        self,
        wp,
        forces,
        positions,
        force_world: Tuple[float, float, float],
        position_world: Tuple[float, float, float],
        device,
    ) -> None:
        """
        Drop ``cmd.force_world`` / ``cmd.position_world`` into slot [0, body_index, :].

        Build a tiny (1, 3) warp array per channel on the same device and
        use ``wp.copy`` into the (count, max_links, 3) buffer's slice.
        Kept as its own method so tests can patch the warp-side scatter
        without faking out the whole pipeline.
        """
        idx = self._body_index
        force_row = wp.array(
            [[float(force_world[0]), float(force_world[1]), float(force_world[2])]],
            dtype=wp.float32, device=device,
        )
        pos_row = wp.array(
            [[float(position_world[0]), float(position_world[1]), float(position_world[2])]],
            dtype=wp.float32, device=device,
        )
        wp.copy(forces[0, idx:idx + 1], force_row)
        wp.copy(positions[0, idx:idx + 1], pos_row)

    @staticmethod
    def _resolve_device(view: Any, wp: Any) -> Any:
        """
        Pick the warp device the view's tensors live on.

        Falls back to the default warp device when the view doesn't
        expose one — both PhysX and Newton views keep it on
        ``_frontend.device``, but we don't want a hard dependency on
        that private attribute.
        """
        frontend = getattr(view, '_frontend', None)
        device = getattr(frontend, 'device', None)
        if device is not None:
            return device
        try:
            return wp.get_device()
        except Exception:  # noqa: BLE001
            return 'cpu'

    # ------------------------------------------------------------------
    # Visualization (self-contained USD decor; no-op without a stage)
    # ------------------------------------------------------------------
    def _viz_prims(self) -> Any:
        """
        Return the rope + anchor visualization prims, authoring them once.

        Gives back a ``(group, rope, anchor)`` tuple of UsdGeom schemas, or
        ``None`` when no stage is available (headless without a stage, unit
        tests). The prims are pure graphics — no physics — under a top-level
        ``/Gantry`` scope, so the articulation never discovers them.
        """
        if self._viz is not _UNSET:
            return self._viz
        self._viz = None
        try:
            import omni.usd
            from pxr import Gf, UsdGeom
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return None
            group = UsdGeom.Xform.Define(stage, '/Gantry')
            rope = UsdGeom.BasisCurves.Define(stage, '/Gantry/rope')
            rope.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
            rope.CreateCurveVertexCountsAttr().Set([2])
            rope.CreatePointsAttr().Set(
                [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0)],
            )
            rope.CreateWidthsAttr().Set([0.02, 0.02])
            rope.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
            rope.CreateDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.95, 0.2)])
            anchor = UsdGeom.Sphere.Define(stage, '/Gantry/anchor')
            anchor.CreateRadiusAttr().Set(0.03)
            anchor.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            self._viz = (group, rope, anchor)
        except Exception:  # noqa: BLE001 — visualization is best-effort only.
            self._viz = None
        return self._viz

    def _draw_gantry(self, attach_world: Tuple[float, float, float],
                     taut: bool) -> None:
        """
        Update the rope line + anchor marker for the current step.

        Mirrors the MuJoCo plugin's decor: a line from anchor to attach and
        a marker at the anchor. The rope is green when carrying load, grey
        when slack, so the operator can tell at a glance whether the robot
        is hanging.
        """
        if not self._visualize:
            return
        prims = self._viz_prims()
        if prims is None:
            return
        group, rope, anchor = prims
        from pxr import Gf, UsdGeom
        ax, ay, az = (float(c) for c in self._anchor)
        bx, by, bz = (float(c) for c in attach_world)
        rope.GetPointsAttr().Set([Gf.Vec3f(ax, ay, az), Gf.Vec3f(bx, by, bz)])
        color = (0.1, 0.95, 0.2) if taut else (0.6, 0.6, 0.6)
        rope.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        UsdGeom.XformCommonAPI(anchor.GetPrim()).SetTranslate(
            Gf.Vec3d(ax, ay, az),
        )
        UsdGeom.Imageable(group.GetPrim()).MakeVisible()

    def _clear_gantry_drawing(self) -> None:
        """Hide the rope/anchor decor; a no-op if it was never authored."""
        if not isinstance(self._viz, tuple):
            return
        try:
            from pxr import UsdGeom
            group = self._viz[0]
            UsdGeom.Imageable(group.GetPrim()).MakeInvisible()
        except Exception:  # noqa: BLE001
            pass


_UNSET = object()


def _import_warp():
    """
    Lazy-import ``warp`` so the unit tests can run without Isaac Sim.

    Returns the module on success, or ``None`` when warp isn't
    importable. The result is cached on the function object so we only
    pay the import cost once per process.
    """
    cached = getattr(_import_warp, '_module', _UNSET)
    if cached is not _UNSET:
        return cached
    try:
        import warp as wp  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — warp is optional at unit-test time.
        wp = None
    _import_warp._module = wp  # type: ignore[attr-defined]
    return wp


def _to_host_array(arr: Any):
    """
    Convert a ``warp.array`` / torch tensor / ndarray to a host NumPy array.

    Returns ``None`` on failure. Kept outside the class so tests can
    swap in a plain ``[[7-tuple, ...]]`` Python list without going
    through warp.
    """
    if arr is None:
        return None
    if hasattr(arr, 'numpy'):
        try:
            return arr.numpy()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(arr, 'cpu'):
        try:
            return arr.cpu().numpy()
        except Exception:  # noqa: BLE001
            pass
    # Already a Python sequence / numpy array.
    return arr

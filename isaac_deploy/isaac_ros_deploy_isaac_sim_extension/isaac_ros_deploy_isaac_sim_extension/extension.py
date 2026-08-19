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
Kit extension entry point for the actuator bridge.

On stage open, builds an ``ArticulationActuators`` view from the
``NewtonActuator`` prims baked into the robot USD, instantiates the
virtual gantry, and wires the rclpy node (gain injection +
gantry-toggle service), all driven by Kit lifecycle hooks. Kit is
already running by the time ``on_startup`` fires, so this module skips
``SimulationApp`` creation and the main loop, registering a Kit update
event for the rclpy spin instead.

Configuration is read from Kit settings under
``/exts/isaac_ros_deploy_isaac_sim_extension/*`` (see ``config/extension.toml``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Base class is ``omni.ext.IExt`` inside Kit (a C-extension type that must
# be set at class-definition time) and ``object`` outside Kit, so the
# package stays importable for tests and linting without the Kit runtime.
try:
    import omni.ext as _omni_ext  # type: ignore[import-not-found]
    _IExtBase: type = _omni_ext.IExt
except ImportError:
    _IExtBase = object


class IsaacsimActuatorBridgeExtension(_IExtBase):
    """
    ``omni.ext.IExt`` extension wrapper around the bridge.

    Inherits from ``omni.ext.IExt`` inside Kit and from ``object`` when
    imported elsewhere, so test environments can import the module
    without needing the Kit runtime.
    """

    def __init__(self) -> None:
        """Initialise the extension wrapper with no active session."""
        super().__init__()
        self._ext_id: Optional[str] = None
        self._state: Dict[str, Any] = {}

    def on_startup(self, ext_id: str) -> None:
        """
        Bootstrap the actuator bridge when Kit enables the extension.

        Defers the heavy setup until ``isaacsim.app.setup`` has finished
        composing its default stage - it loads later in the boot order
        than us and will otherwise blow away whatever stage we open.
        Strategy: subscribe to the stage event stream and wait for the
        first OPENED event (the default empty stage), then take over.
        """
        import carb
        import omni.usd
        self._ext_id = ext_id

        carb.log_info(
            '[isaac_ros_deploy_isaac_sim_extension] on_startup; '
            'waiting for default stage before running setup',
        )

        stage_event_type_opened = int(omni.usd.StageEventType.OPENED)

        def _on_stage_event(event: Any) -> None:
            if int(event.type) != stage_event_type_opened:
                return
            sub = self._state.pop('stage_event_sub', None)
            if sub is not None:
                sub.unsubscribe()
            try:
                self._do_setup()
            except Exception as exc:  # noqa: BLE001
                carb.log_error(
                    f'[isaac_ros_deploy_isaac_sim_extension] deferred setup failed: {exc!r}',
                )

        self._state['stage_event_sub'] = (
            omni.usd.get_context()
            .get_stage_event_stream()
            .create_subscription_to_pop(_on_stage_event)
        )

    def _do_setup(self) -> None:
        """Run the actual bridge setup after Kit's app-init has settled."""
        import carb
        settings = carb.settings.get_settings()
        ns = '/exts/isaac_ros_deploy_isaac_sim_extension'
        usd_path = settings.get(f'{ns}/usd') or ''
        robot_path = settings.get(f'{ns}/robot_path') or ''
        gantry_enabled = bool(settings.get(f'{ns}/gantry_enabled'))
        physics_hz = float(settings.get(f'{ns}/physics_hz') or 0.0)

        if not usd_path:
            carb.log_error(
                '[isaac_ros_deploy_isaac_sim_extension] missing required Kit '
                f'setting ({ns}/usd); aborting.',
            )
            return

        # Open the stage. omni.usd.open_stage is synchronous w.r.t. USD
        # composition; OmniGraph node activation happens on subsequent
        # update ticks, but the articulation prim is queryable as soon
        # as the stage is composed.
        import omni.usd
        ctx = omni.usd.get_context()
        ctx.open_stage(usd_path)
        stage = ctx.get_stage()
        if stage is None:
            carb.log_error(
                f'[isaac_ros_deploy_isaac_sim_extension] failed to open USD {usd_path!r}; '
                'check the path and that the file exists. Aborting.',
            )
            return

        # Default to the articulation-root prim discovered in the stage so
        # the extension is not tied to one robot; robot_path overrides it.
        if not robot_path:
            from pxr import UsdPhysics
            root_prim = next(
                (p for p in stage.Traverse()
                 if p.HasAPI(UsdPhysics.ArticulationRootAPI)),
                None,
            )
            if root_prim is None:
                carb.log_error(
                    '[isaac_ros_deploy_isaac_sim_extension] no UsdPhysicsArticulationRootAPI '
                    f'prim in {usd_path!r} and no {ns}/robot_path set; aborting.',
                )
                return
            robot_path = root_prim.GetPath().pathString

        carb.log_info(
            f'[isaac_ros_deploy_isaac_sim_extension] deferred setup '
            f'usd={usd_path!r} robot_path={robot_path!r} '
            f'gantry_enabled={gantry_enabled}',
        )

        # A bare robot USD has no floor or lights; add sensible defaults so the
        # deploy renders and the robot has something to stand on. A USD that
        # already ships its own scene (ground/lights) is left untouched.
        self._ensure_ground_and_light(stage)

        # The USD authors the robot with its pelvis at the origin and the legs
        # hanging below z=0; raise it so its feet rest on the ground (z=0).
        self._place_robot_on_ground(stage, robot_path)

        # Materialise the physics view. Newton's articulation view is
        # lazy and only initialises on a physics tick or on an explicit
        # initialize_physics() call; calling it unconditionally is a
        # no-op on PhysX.
        from isaacsim.core.simulation_manager import (
            SimulationEvent,
            SimulationManager,
        )

        if physics_hz > 0.0:
            try:
                SimulationManager.set_physics_dt(1.0 / physics_hz)
                carb.log_info(
                    f'[isaac_ros_deploy_isaac_sim_extension] physics rate set to '
                    f'{physics_hz:.0f} Hz (dt={1.0 / physics_hz:.5f}s)',
                )
            except Exception as exc:  # noqa: BLE001
                carb.log_warn(
                    f'[isaac_ros_deploy_isaac_sim_extension] could not set physics '
                    f'rate to {physics_hz} Hz: {exc!r}',
                )
        SimulationManager.initialize_physics()

        from isaacsim.core.experimental.prims import Articulation
        articulation = Articulation(robot_path)
        joint_names = list(articulation.dof_names)
        carb.log_info(
            f'[isaac_ros_deploy_isaac_sim_extension] discovered {len(joint_names)} '
            f'DOFs on {robot_path}',
        )
        if not joint_names:
            carb.log_error(
                '[isaac_ros_deploy_isaac_sim_extension] articulation has zero DOFs; '
                f'check {ns}/robot_path and the opened USD',
            )
            return

        # Build the actuator chain from the NewtonActuator prims in the USD.
        from isaacsim.core.experimental.actuators import ArticulationActuators
        actuated = ArticulationActuators(robot_path)
        n_actuators = len(actuated.actuators)
        carb.log_info(
            '[isaac_ros_deploy_isaac_sim_extension] ArticulationActuators attached: '
            f'{n_actuators} NewtonActuator prims discovered under {robot_path!r}',
        )
        if n_actuators == 0:
            carb.log_error(
                '[isaac_ros_deploy_isaac_sim_extension] no NewtonActuator prims found - '
                'the robot will receive no actuator torque and collapse. ',
            )

        # Virtual gantry, created disabled; enabled below per the
        # gantry_enabled setting, or later via the service / G hotkey.
        from isaac_ros_deploy_isaac_sim_extension.gantry import (
            VirtualGantry,
            VirtualGantryConfig,
        )
        gantry = VirtualGantry(articulation, VirtualGantryConfig(enabled=False))

        # rclpy node for the GainInjector subscription + gantry-enable service.
        # Created before the pre-step callback below so the callback's closure
        # over rclpy / rclpy_node is always bound, even if a physics tick fires
        # during setup.
        import rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        rclpy_node = rclpy.create_node('isaacsim_launcher')

        # Register the rope step on the SimulationManager pre-physics bus.
        def _gantry_pre_step(step_dt: float, _context: Any = None) -> None:
            try:
                rclpy.spin_once(rclpy_node, timeout_sec=0.0)
            except Exception:  # noqa: BLE001
                pass
            try:
                gantry.step(float(step_dt))
            except Exception as exc:  # noqa: BLE001
                carb.log_warn(
                    f'[isaac_ros_deploy_isaac_sim_extension] gantry.step error: {exc!r}',
                )

        gantry_cb_id = SimulationManager.register_callback(
            _gantry_pre_step, event=SimulationEvent.PHYSICS_PRE_STEP,
        )

        # Pair each actuator with its joint name for the GainInjector.
        dof_names = list(articulation.dof_names)
        actuator_name_pairs = []
        for act in actuated.actuators:
            indices_cpu = act.indices.numpy()
            for idx in indices_cpu:
                actuator_name_pairs.append((act, dof_names[int(idx)]))

        from isaac_ros_deploy_isaac_sim_extension.gain_injector import GainInjector
        gain_injector = GainInjector(rclpy_node, actuator_name_pairs)
        carb.log_info(
            f'[isaac_ros_deploy_isaac_sim_extension] gain injector subscribed to '
            f'{gain_injector.topic}',
        )

        from std_srvs.srv import SetBool

        def _on_set_gantry(req: Any, resp: Any) -> Any:
            if req.data:
                gantry.enable()
            else:
                gantry.disable()
            resp.success = True
            resp.message = 'enabled' if req.data else 'disabled'
            return resp

        rclpy_node.create_service(
            SetBool, '/launcher/gantry/set_enabled', _on_set_gantry,
        )

        # Timeline STOP - re-enable the gantry so the next Play starts ready.
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        _timeline_event_sub = None
        try:
            stop_type = int(omni.timeline.TimelineEventType.STOP)

            def _on_timeline_event(event: Any) -> None:
                if int(event.type) != stop_type:
                    return
                try:
                    t_now = float(timeline.get_current_time())
                except Exception:  # noqa: BLE001
                    t_now = 0.0
                if t_now > 1e-6:
                    return
                try:
                    gantry.enable()
                except Exception as exc:  # noqa: BLE001
                    carb.log_warn(
                        f'[virtual-gantry] re-enable on Stop failed: {exc!r}',
                    )

            _timeline_event_sub = (
                timeline.get_timeline_event_stream()
                .create_subscription_to_pop(_on_timeline_event)
            )
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(
                f'[isaac_ros_deploy_isaac_sim_extension] timeline-event sub failed '
                f'({exc!r}); Stop will not auto-re-enable gantry',
            )

        # Carb keyboard hotkeys (G / [ / ]).
        _gantry_hotkey_sub = gantry.setup_hotkeys()

        # Enable gantry if requested.
        if gantry_enabled:
            gantry.enable()

        # Kit update event - replaces the launcher's main loop. Drives
        # the rclpy spin once per Kit tick (~60 Hz).
        import omni.kit.app

        def _on_kit_update(_event: Any) -> None:
            if rclpy.ok():
                try:
                    rclpy.spin_once(rclpy_node, timeout_sec=0.0)
                except Exception as exc:  # noqa: BLE001
                    carb.log_warn(
                        f'[isaac_ros_deploy_isaac_sim_extension] rclpy.spin_once '
                        f'error: {exc!r}',
                    )

        update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(_on_kit_update)
        )

        # Stash everything for on_shutdown teardown.
        self._state = {
            'gantry': gantry,
            'gantry_cb_id': gantry_cb_id,
            'timeline_event_sub': _timeline_event_sub,
            'hotkey_sub': _gantry_hotkey_sub,
            'rclpy_node': rclpy_node,
            'update_sub': update_sub,
            'actuated_keepalive': actuated,
        }
        carb.log_info(
            '[isaac_ros_deploy_isaac_sim_extension] setup complete',
        )

    def _ensure_ground_and_light(self, stage: Any) -> None:
        """Add a ground plane and a dome light when the opened USD lacks them."""
        import carb
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

        prims = list(stage.Traverse())
        has_ground = any(
            prim.GetTypeName() == 'Plane'
            or 'ground' in prim.GetName().lower()
            or 'floor' in prim.GetName().lower()
            for prim in prims
        )
        if not has_ground:
            # Large thin box collider with its top face at world z=0 (robust on
            # both PhysX and Newton, unlike an infinite plane).
            half_extents = (50.0, 50.0, 0.05)
            cube = UsdGeom.Cube.Define(stage, Sdf.Path('/DeployGroundPlane'))
            cube.GetSizeAttr().Set(2.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -half_extents[2]))
            xform.AddScaleOp().Set(Gf.Vec3f(*half_extents))
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.3, 0.3)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

            ground_mat_path = Sdf.Path('/DeployGroundPlane/PhysicsMaterial')
            ground_mat = UsdShade.Material.Define(stage, ground_mat_path)
            mat_api = UsdPhysics.MaterialAPI.Apply(ground_mat.GetPrim())
            mat_api.CreateStaticFrictionAttr().Set(1.0)
            mat_api.CreateDynamicFrictionAttr().Set(1.0)
            mat_api.CreateRestitutionAttr().Set(0.0)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                ground_mat,
                bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                materialPurpose='physics',
            )
            carb.log_info(
                '[isaac_ros_deploy_isaac_sim_extension] added a default ground '
                'plane (top face at world z=0)',
            )

        # Always author our own dome light. The host GUI app injects a default
        # light prim that does not actually illuminate the opened deploy stage,
        # so keying off an existing light leaves the viewport black.
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path('/DeployDomeLight'))
        dome.CreateIntensityAttr(1000.0)
        carb.log_info(
            '[isaac_ros_deploy_isaac_sim_extension] added a default dome light',
        )

    def _place_robot_on_ground(self, stage: Any, robot_path: str) -> None:
        """
        Raise the robot so its lowest geometry rests on the ground (z=0).

        The USD authors the G1 standing but with its pelvis at the origin, so
        the feet hang to about z=-0.79 and the robot would spawn sunk into the
        floor. Shift the robot's top-level prim up by its foot depth; a robot
        already on the ground (foot depth ~0) is left untouched.
        """
        import carb
        from pxr import Gf, Sdf, Usd, UsdGeom

        prim = stage.GetPrimAtPath(robot_path)
        if not prim.IsValid():
            return
        top = prim
        while top.GetParent().GetPath() != Sdf.Path.absoluteRootPath:
            top = top.GetParent()

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        lowest_z = None
        for mesh_prim in Usd.PrimRange(
            top, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        ):
            if not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            extent = UsdGeom.Mesh(mesh_prim).GetExtentAttr().Get()
            if not extent:
                continue
            to_world = cache.GetLocalToWorldTransform(mesh_prim)
            lo, hi = extent[0], extent[1]
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    for z in (lo[2], hi[2]):
                        world_z = to_world.Transform(Gf.Vec3d(x, y, z))[2]
                        if lowest_z is None or world_z < lowest_z:
                            lowest_z = world_z
        if lowest_z is None or abs(lowest_z) < 1e-3:
            return

        xform = UsdGeom.Xformable(top)
        translate_op = next(
            (op for op in xform.GetOrderedXformOps()
             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        ) or xform.AddTranslateOp()
        current = translate_op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
        translate_op.Set(type(current)(current[0], current[1], current[2] - lowest_z))
        carb.log_info(
            f'[isaac_ros_deploy_isaac_sim_extension] raised {top.GetName()} by '
            f'{-lowest_z:+.3f} m so its feet rest on the ground',
        )

    def on_shutdown(self) -> None:
        """
        Tear the bridge down when Kit disables the extension.

        Mirrors the launcher's ``finally`` block (callback deregister,
        rclpy node teardown). Does NOT close Kit - the extension being
        disabled is independent of the host app lifecycle.
        """
        import carb
        carb.log_info(
            f'[isaac_ros_deploy_isaac_sim_extension] on_shutdown ext_id={self._ext_id}',
        )

        state = self._state
        self._state = {}

        try:
            cb_id = state.get('gantry_cb_id')
            if cb_id is not None:
                from isaacsim.core.simulation_manager import SimulationManager
                SimulationManager.deregister_callback(cb_id)
        except Exception:  # noqa: BLE001
            pass

        try:
            tl_sub = state.get('timeline_event_sub')
            if tl_sub is not None:
                tl_sub.unsubscribe()
        except Exception:  # noqa: BLE001
            pass

        try:
            hk_sub = state.get('hotkey_sub')
            if hk_sub is not None:
                import carb.input  # type: ignore[import-not-found]
                import omni.appwindow  # type: ignore[import-not-found]
                input_iface = carb.input.acquire_input_interface()
                keyboard = (
                    omni.appwindow.get_default_app_window().get_keyboard()
                )
                input_iface.unsubscribe_to_keyboard_events(keyboard, hk_sub)
        except Exception:  # noqa: BLE001
            pass

        # Dropping the stashed reference releases the Kit update subscription.
        state.pop('update_sub', None)

        node = state.get('rclpy_node')
        if node is not None:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
            try:
                import rclpy
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass

        self._ext_id = None

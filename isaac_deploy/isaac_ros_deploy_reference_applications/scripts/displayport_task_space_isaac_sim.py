#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open and play the DisplayPort task-space Isaac Sim tutorial scene.

Run this script with Isaac Sim's ``python.sh``. It opens the configured USD,
enables the ROS 2 bridge and ScriptNode extension, applies the Isaac Sim-side
defaults used by the tutorial, initializes physics, selects a viewport camera,
and starts timeline playback.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
from typing import Dict, Optional


DEFAULT_DISPLAYPORT_USD_URI = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "Isaac/6.1/Isaac/Samples/ROS2/Scenario/flexiv_displayport_insertion.usd"
)

DEFAULT_SIM_ENV: Dict[str, str] = {
    "ROS_DISTRO": "jazzy",
    "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4",
    "DISPLAYPORT_PLAY_STEP_MODE": "play",
    "DISPLAYPORT_PLAY_GRIPPER_TARGET_MODE": "finger",
    "DISPLAYPORT_PLAY_GRIPPER_EFFORT_MODE": "finger",
    "DISPLAYPORT_PLAY_GRIPPER_COMMAND_PATH": "direct",
    "DISPLAYPORT_PLAY_ARM_COMMAND_PATH": "direct",
    "DISPLAYPORT_PLAY_DISABLE_ROBOT_GRAVITY": "1",
    "DISPLAYPORT_PLAY_JOINT_RESYNC_TOLERANCE_RAD": "0.05",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a DisplayPort tutorial USD in Isaac Sim and press Play."
    )
    parser.add_argument(
        "--usd",
        default=os.environ.get("DISPLAYPORT_PLAY_USD", DEFAULT_DISPLAYPORT_USD_URI),
        help="DisplayPort open-play USD path or URI.",
    )
    parser.add_argument(
        "--camera",
        default=os.environ.get("DISPLAYPORT_PLAY_CAMERA", "wide_overview"),
        help=(
            "Viewport camera name under /World/Cameras, an absolute camera prim "
            "path, or 'perspective'."
        ),
    )
    parser.add_argument(
        "--ros-domain-id",
        default=os.environ.get("ROS_DOMAIN_ID", "231"),
        help="ROS_DOMAIN_ID shared with the Isaac ROS workflow terminal.",
    )
    parser.add_argument(
        "--renderer",
        default=os.environ.get("DISPLAYPORT_PLAY_RENDERER", "RayTracedLighting"),
        help="Isaac Sim renderer.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim without opening a GUI window.",
    )
    return parser.parse_args()


def _apply_default_environment(ros_domain_id: str) -> None:
    os.environ.setdefault("ROS_DOMAIN_ID", str(ros_domain_id))
    for name, value in DEFAULT_SIM_ENV.items():
        os.environ.setdefault(name, value)


def _set_active_camera(stage, camera: str) -> Optional[object]:
    from omni.kit.viewport.utility import get_active_viewport

    camera_name = camera.strip()
    camera_mode = camera_name.lower()
    if camera_mode in {"perspective", "persp"}:
        camera_path = "/OmniverseKit_Persp"
    elif camera_name.startswith("/"):
        camera_path = camera_name
    else:
        camera_path = f"/World/Cameras/{camera_name}"

    if not stage.GetPrimAtPath(camera_path).IsValid() and camera_path != "/OmniverseKit_Persp":
        raise RuntimeError(f"Camera prim does not exist in USD: {camera_path}")

    viewport = get_active_viewport()
    if viewport is not None:
        viewport.camera_path = camera_path
    print(f"DISPLAYPORT_ISAAC_SIM_CAMERA={camera_path}", flush=True)
    return viewport


def _initialize_core_physics(simulation_app) -> bool:
    try:
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.simulation_manager import SimulationManager
    except Exception as exc:
        print(f"DISPLAYPORT_ISAAC_SIM_CORE_PHYSICS_IMPORT_FAILED error={exc}", flush=True)
        return False

    try:
        context = SimulationContext.instance()
        if context is None:
            context = SimulationContext(set_defaults=False)
        ready_before = SimulationManager.get_physics_sim_view() is not None
        if not ready_before:
            context.initialize_physics()
            for _ in range(4):
                simulation_app.update()
        ready_after = SimulationManager.get_physics_sim_view() is not None
        print(
            "DISPLAYPORT_ISAAC_SIM_CORE_PHYSICS_READY "
            f"before={ready_before} after={ready_after}",
            flush=True,
        )
        return bool(ready_after)
    except Exception as exc:
        print(f"DISPLAYPORT_ISAAC_SIM_CORE_PHYSICS_INIT_FAILED error={exc}", flush=True)
        return False


def main() -> None:
    args = _parse_args()
    if not args.usd:
        raise RuntimeError("Pass --usd or set DISPLAYPORT_PLAY_USD")

    usd = str(args.usd)
    if "://" not in usd:
        usd_path = Path(usd).expanduser()
        if not usd_path.is_file():
            raise FileNotFoundError(f"DisplayPort USD does not exist: {usd_path}")
        usd = str(usd_path)

    _apply_default_environment(args.ros_domain_id)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "renderer": args.renderer,
        }
    )

    import carb
    import omni.kit.app
    import omni.timeline
    import omni.usd

    stop_requested = False

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    settings = carb.settings.get_settings()
    manager = omni.kit.app.get_app().get_extension_manager()
    settings.set("/app/omni.graph.scriptnode/opt_in", True)
    manager.set_extension_enabled_immediate("omni.graph.scriptnode", True)
    manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

    for _ in range(20):
        simulation_app.update()

    context = omni.usd.get_context()
    print(f"DISPLAYPORT_ISAAC_SIM_OPEN_STAGE_BEGIN usd={usd}", flush=True)
    if not context.open_stage(usd):
        raise RuntimeError(f"Failed to open stage: {usd}")

    for _ in range(5):
        simulation_app.update()

    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"Stage failed to load: {usd}")

    if not _initialize_core_physics(simulation_app):
        raise RuntimeError("Failed to initialize Isaac Core physics before Play")

    for _ in range(120):
        simulation_app.update()

    _set_active_camera(stage, args.camera)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print(
        "DISPLAYPORT_ISAAC_SIM_PLAYING "
        f"usd={usd} ros_domain_id={os.environ['ROS_DOMAIN_ID']}",
        flush=True,
    )

    try:
        while simulation_app.is_running() and not stop_requested:
            simulation_app.update()
    finally:
        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()

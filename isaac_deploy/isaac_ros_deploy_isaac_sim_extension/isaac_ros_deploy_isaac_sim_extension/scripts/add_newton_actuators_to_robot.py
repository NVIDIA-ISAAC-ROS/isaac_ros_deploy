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
r"""
Author a PD actuator with a max-effort clamp into a robot USD.

Adds one NewtonActuator prim per movable joint: a PD controller plus a
flat effort clamp that ``ArticulationActuators(robot)`` discovers and runs
at the physics rate. Every parameter comes from the joint's own USD drive,
so the script is robot-agnostic and needs no configuration:

* kp / kd     -> placeholders, overwritten by the gain injector at runtime
* max_effort  -> the joint drive's ``physics:maxForce``

The clamp bounds the torque the PD law can command (without it a stiff
gain commands unbounded effort and the robot launches). The joint's own
velocity limit (``physxJoint:maxJointVelocity``) is enforced by the engine,
so it is not duplicated on the actuator.

``--usd`` takes a local USD or an Isaac assets-root URL for the G1, so the
model never has to be committed to this repo:

    <isaac-sim>/python.sh \\
        -m isaac_ros_deploy_isaac_sim_extension.scripts.add_newton_actuators_to_robot \\
        --usd <path-or-url-to.usd> [--out <path-to-output.usd>]
"""

import argparse
import functools
import os
import sys


# Placeholder gains; the gain injector overwrites them at runtime.
_DEFAULT_KP = 1.0
_DEFAULT_KD = 0.1


def _read_max_effort(joint_prim):
    """Return the joint drive's authored max force (the actuator effort limit)."""
    from pxr import UsdPhysics
    is_linear = joint_prim.IsA(UsdPhysics.PrismaticJoint)
    drive = UsdPhysics.DriveAPI.Get(
        joint_prim, 'linear' if is_linear else 'angular',
    )
    attr = drive.GetMaxForceAttr() if drive else None
    if attr and attr.IsValid() and attr.HasAuthoredValue():
        value = attr.Get()
        if value is not None and value > 0.0:
            return float(value)
    raise RuntimeError(
        f'joint {joint_prim.GetPath()} has no positive drive maxForce; '
        'the actuator effort limit must be authored in the USD',
    )


def _disable_native_drive(joint_prim):
    """
    Zero the joint's native PhysX drive so only the NewtonActuator controls it.

    Robots from the Isaac assets root ship with native drive stiffness/damping
    (their training gains) set. Left active, those fight the baked
    NewtonActuator's PD - the joints get pulled toward target=0 and jitter - so
    zero them here. ``maxForce`` is preserved for the effort clamp.
    """
    from pxr import UsdPhysics
    is_linear = joint_prim.IsA(UsdPhysics.PrismaticJoint)
    drive = UsdPhysics.DriveAPI.Get(
        joint_prim, 'linear' if is_linear else 'angular',
    )
    if drive:
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)


def _resolve_newton_experience():
    """Resolve the Newton-backed Kit experience path from ``$ISAAC_PATH``."""
    root = os.environ.get('ISAAC_PATH')
    if not root:
        raise RuntimeError(
            '$ISAAC_PATH is not set. Run via the Isaac Sim python.sh '
            '(which exports it), or set the variable to the install root.'
        )
    return os.path.join(root, 'apps/isaacsim.exp.full.newton.kit')


def _get_prim_path_spec_layer(stage, prim_path):
    """
    Return the layer that carries ``prim_path``'s spec.

    ``add_actuator`` writes through the stage's edit target, which is the root
    layer for a local source but the session layer when the root layer is a
    read-only remote (an ``omniverse://`` or ``https://`` asset). Copying from
    a layer that does not hold the spec takes the process down inside
    ``Sdf.CopySpec`` with no Python exception, so resolve the owner first.
    """
    for layer in stage.GetLayerStack():
        if layer.GetPrimAtPath(prim_path) is not None:
            return layer
    raise RuntimeError(f'no layer in the stage holds a spec for {prim_path}')


def _export_stage(stage, usd_in, usd_out):
    """
    Write the baked stage to ``usd_out``.

    A remote source is flattened so its (often relative) references and
    payloads resolve into a self-contained file; a local source keeps its
    root-layer structure and is exported as-is.
    """
    if '://' in usd_in:
        stage.Export(usd_out)
    else:
        stage.GetRootLayer().Export(usd_out)


def _parse_args(argv):
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog='add_newton_actuators_to_robot',
        description=(
            'Author a PD + max-effort NewtonActuator prim per joint into a '
            'robot USD, with the effort limit taken from each joint drive.'
        ),
    )
    parser.add_argument(
        '--usd', required=True,
        help='Path or Isaac assets-root URL of the USD to read.',
    )
    parser.add_argument(
        '--out', default=None,
        help='Where to save the modified USD. Defaults to overwriting --usd.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Create a PD + max-effort NewtonActuator for every joint."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    from isaacsim import SimulationApp
    sim_app = SimulationApp(
        {'headless': True},
        experience=_resolve_newton_experience(),
    )

    try:
        import omni.usd
        from isaacsim.core.experimental.actuators import (
            MaxEffortClampingConfig,
            PDControlConfig,
            add_actuator,
        )
        from pxr import Sdf, UsdPhysics

        # A bare path is made absolute; an assets-root URL is passed through.
        usd_in = args.usd if '://' in args.usd else os.path.abspath(args.usd)
        usd_out = os.path.abspath(args.out) if args.out else usd_in

        ctx = omni.usd.get_context()
        ctx.open_stage(usd_in)
        stage = ctx.get_stage()

        # Let Kit finish composing the stage before querying it.
        for _ in range(60):
            sim_app.update()

        joint_paths = [
            prim.GetPath()
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint)
        ]
        if not joint_paths:
            raise RuntimeError(f'no movable joints found in {usd_in!r}')
        print(f'[add-newton-actuators] discovered {len(joint_paths)} joints')

        # add_actuator validates every target name against the subtree of
        # this prim, so it must be an ancestor of all joints.
        authoring_root = functools.reduce(
            lambda a, b: a.GetCommonPrefix(b), joint_paths,
        )
        print(f'[add-newton-actuators] authoring root: {authoring_root}')

        # ArticulationActuators only discovers NewtonActuator prims under
        # the ArticulationRootAPI prim, which is often not an ancestor of
        # the joints (the G1's is `.../pelvis`), so we author against
        # authoring_root and reparent the result here afterwards.
        root_api_prim = next(
            (p for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)),
            None,
        )
        if root_api_prim is None:
            raise RuntimeError(
                f'no prim with UsdPhysicsArticulationRootAPI found in {usd_in!r}',
            )
        print(
            f'[add-newton-actuators] articulation-root prim: {root_api_prim.GetPath()}',
        )

        for joint_path in joint_paths:
            joint_name = joint_path.name
            joint_prim = stage.GetPrimAtPath(joint_path)
            _disable_native_drive(joint_prim)
            add_actuator(
                str(authoring_root),
                target_names=joint_name,
                name=f'{joint_name}_actuator',
                controller=PDControlConfig(kp=_DEFAULT_KP, kd=_DEFAULT_KD),
                clamping=[
                    MaxEffortClampingConfig(
                        max_effort=_read_max_effort(
                            stage.GetPrimAtPath(joint_path),
                        ),
                    ),
                ],
                overwrite_existing=True,
            )
        print(f'[add-newton-actuators] authored {len(joint_paths)} NewtonActuator prims')

        # Reparent the authored scope under the discovery root. Absolute
        # newton:targets survive the move.
        authored_scope = authoring_root.AppendPath('Actuators')
        target_scope = root_api_prim.GetPath().AppendPath('Actuators')
        if authored_scope != target_scope:
            layer = _get_prim_path_spec_layer(stage, authored_scope)
            Sdf.CreatePrimInLayer(layer, target_scope)
            if not Sdf.CopySpec(layer, authored_scope, layer, target_scope):
                raise RuntimeError(
                    f'failed to copy {authored_scope} to {target_scope} in '
                    f'layer {layer.identifier!r}',
                )
            stage.RemovePrim(authored_scope)
            print(
                f'[add-newton-actuators] moved Actuators scope '
                f'{authored_scope} -> {target_scope}',
            )

        _export_stage(stage, usd_in, usd_out)
        print(f'[add-newton-actuators] saved to {usd_out}')

    finally:
        sim_app.close()


if __name__ == '__main__':
    main()

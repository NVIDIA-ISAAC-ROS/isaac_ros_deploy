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
Author the ROS2 bridge into a robot USD.

Turns a plain articulated-robot USD into one that talks to ROS2: adds the
OmniGraphs that publish joint states + the clock + the base IMU and
subscribe joint commands (driving the articulation), all bound to the
discovered ``UsdPhysicsArticulationRootAPI`` prim. A base-link
``IsaacImuSensor`` is created if the robot has none. Robot-agnostic;
nothing here is specific to a given robot.

    /ROS2_JointStates   : OnPhysicsStep -> Context -> ReadSimTime
                          -> ROS2PublishJointState
    /ROS2_JointCommands : OnPhysicsStep -> Context -> ROS2SubscribeJointState
                          -> IsaacArticulationController
    /ROS2_Clock         : OnPhysicsStep -> Context -> ReadSimTime
                          -> ROS2PublishClock
    /ROS2_IMU           : OnPhysicsStep -> Context -> ReadSimTime
                          -> IsaacReadIMU -> ROS2PublishImu

This authors ROS2 OmniGraph nodes, so the Isaac ROS environment must be
sourced first (sets ``LD_LIBRARY_PATH`` to the bundled ROS libraries):

    source <isaac-sim>/setup_ros_env.sh
    <isaac-sim>/python.sh \\
        -m isaac_ros_deploy_isaac_sim_extension.scripts.add_ros2_bridge_to_robot \\
        --usd <path> [--out <path>]
"""

import argparse
import os
import sys


# Topic names are fixed by the deploy contract - the topic-based hardware
# interface and the isaacsim URDF read these exact topics - so they are
# baked here rather than exposed as arguments.
_JOINT_STATES_TOPIC = '/isaac_sim_joint_states'
_JOINT_COMMANDS_TOPIC = '/isaac_sim_joint_commands'
_CLOCK_TOPIC = '/clock'
_IMU_TOPIC = '/isaac_sim_imu'
_IMU_FRAME_ID = 'imu'


def _resolve_newton_experience():
    """Resolve the Newton-backed Kit experience path from ``$ISAAC_PATH``."""
    root = os.environ.get('ISAAC_PATH')
    if not root:
        raise RuntimeError(
            '$ISAAC_PATH is not set. Run via the Isaac Sim python.sh '
            '(which exports it), or set the variable to the install root.'
        )
    return os.path.join(root, 'apps/isaacsim.exp.full.newton.kit')


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
        prog='add_ros2_bridge_to_robot',
        description='Author the joint-state/command + clock ROS2 bridge into a robot USD.',
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


def _find_articulation_root(stage):
    """Return the prim that carries ``UsdPhysicsArticulationRootAPI``."""
    from pxr import UsdPhysics
    root = next(
        (p for p in stage.Traverse()
         if p.HasAPI(UsdPhysics.ArticulationRootAPI)),
        None,
    )
    if root is None:
        raise RuntimeError('no UsdPhysicsArticulationRootAPI prim found')
    return root.GetPath().pathString


def _ensure_imu_sensor(stage, root_path):
    """Return an IsaacImuSensor on the base link, creating one if absent."""
    from pxr import Sdf, Usd
    root_prim = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() == 'IsaacImuSensor':
            return prim.GetPath().pathString
    imu_path = root_path + '/imu_sensor'
    stage.DefinePrim(imu_path, 'IsaacImuSensor').CreateAttribute(
        'enabled', Sdf.ValueTypeNames.Bool,
    ).Set(True)
    return imu_path


def _author_graph(og, stage, graph_path, create_nodes, connect, set_values):
    """
    Create one OmniGraph, replacing any graph already at ``graph_path``.

    The graph is forced on-demand so the OnPhysicsStep trigger fires on
    every physics step, matching the other /ROS2_* graphs.
    """
    from pxr import Sdf
    if stage.GetPrimAtPath(graph_path).IsValid():
        stage.RemovePrim(graph_path)
    keys = og.Controller.Keys
    og.Controller.edit(
        {'graph_path': graph_path, 'evaluator_name': 'execution'},
        {
            keys.CREATE_NODES: create_nodes,
            keys.CONNECT: connect,
            keys.SET_VALUES: set_values,
        },
    )
    graph_prim = stage.GetPrimAtPath(graph_path)
    graph_prim.CreateAttribute(
        'pipelineStage', Sdf.ValueTypeNames.Token,
    ).Set('pipelineStageOnDemand')
    graph_prim.CreateAttribute(
        'evaluationMode', Sdf.ValueTypeNames.Token,
    ).Set('Automatic')
    graph_prim.CreateAttribute(
        'fabricCacheBacking', Sdf.ValueTypeNames.Token,
    ).Set('StageWithoutHistory')


def main(argv=None):
    """Add the ROS2 joint + clock bridge graphs to the robot USD."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    from isaacsim import SimulationApp
    sim_app = SimulationApp(
        {'headless': True},
        experience=_resolve_newton_experience(),
    )

    try:
        import omni.usd
        import isaacsim.core.experimental.utils.app as app_utils
        # Registers the ROS2/Isaac OmniGraph node types used below.
        app_utils.enable_extension('isaacsim.ros2.bridge')
        import omni.graph.core as og
        import usdrt

        # A bare path is made absolute; an assets-root URL is passed through.
        usd_in = args.usd if '://' in args.usd else os.path.abspath(args.usd)
        usd_out = os.path.abspath(args.out) if args.out else usd_in

        ctx = omni.usd.get_context()
        ctx.open_stage(usd_in)
        stage = ctx.get_stage()
        for _ in range(60):
            sim_app.update()

        root = _find_articulation_root(stage)
        print(f'[add-ros2-bridge] articulation root: {root}')

        _author_graph(
            og, stage, '/ROS2_JointStates',
            create_nodes=[
                ('OnPhysicsStep', 'isaacsim.core.nodes.OnPhysicsStep'),
                ('Context', 'isaacsim.ros2.bridge.ROS2Context'),
                ('ReadSimTime', 'isaacsim.core.nodes.IsaacReadSimulationTime'),
                ('PublishJointState', 'isaacsim.ros2.bridge.ROS2PublishJointState'),
            ],
            connect=[
                ('OnPhysicsStep.outputs:step', 'PublishJointState.inputs:execIn'),
                ('Context.outputs:context', 'PublishJointState.inputs:context'),
                ('ReadSimTime.outputs:simulationTime',
                 'PublishJointState.inputs:timeStamp'),
            ],
            set_values=[
                ('PublishJointState.inputs:targetPrim', [usdrt.Sdf.Path(root)]),
                ('PublishJointState.inputs:topicName', _JOINT_STATES_TOPIC),
            ],
        )

        _author_graph(
            og, stage, '/ROS2_JointCommands',
            create_nodes=[
                ('OnPhysicsStep', 'isaacsim.core.nodes.OnPhysicsStep'),
                ('Context', 'isaacsim.ros2.bridge.ROS2Context'),
                ('SubscribeJointCommand', 'isaacsim.ros2.bridge.ROS2SubscribeJointState'),
                ('ArticulationController', 'isaacsim.core.nodes.IsaacArticulationController'),
            ],
            connect=[
                ('OnPhysicsStep.outputs:step', 'SubscribeJointCommand.inputs:execIn'),
                ('Context.outputs:context', 'SubscribeJointCommand.inputs:context'),
                ('SubscribeJointCommand.outputs:execOut',
                 'ArticulationController.inputs:execIn'),
                ('SubscribeJointCommand.outputs:jointNames',
                 'ArticulationController.inputs:jointNames'),
                ('SubscribeJointCommand.outputs:positionCommand',
                 'ArticulationController.inputs:positionCommand'),
            ],
            set_values=[
                ('SubscribeJointCommand.inputs:topicName', _JOINT_COMMANDS_TOPIC),
                ('ArticulationController.inputs:targetPrim', [usdrt.Sdf.Path(root)]),
                ('ArticulationController.inputs:robotPath', root),
            ],
        )

        _author_graph(
            og, stage, '/ROS2_Clock',
            create_nodes=[
                ('OnPhysicsStep', 'isaacsim.core.nodes.OnPhysicsStep'),
                ('Context', 'isaacsim.ros2.bridge.ROS2Context'),
                ('ReadSimTime', 'isaacsim.core.nodes.IsaacReadSimulationTime'),
                ('PublishClock', 'isaacsim.ros2.bridge.ROS2PublishClock'),
            ],
            connect=[
                ('OnPhysicsStep.outputs:step', 'PublishClock.inputs:execIn'),
                ('Context.outputs:context', 'PublishClock.inputs:context'),
                ('ReadSimTime.outputs:simulationTime', 'PublishClock.inputs:timeStamp'),
            ],
            set_values=[
                ('PublishClock.inputs:topicName', _CLOCK_TOPIC),
            ],
        )

        imu = _ensure_imu_sensor(stage, root)
        _author_graph(
            og, stage, '/ROS2_IMU',
            create_nodes=[
                ('OnPhysicsStep', 'isaacsim.core.nodes.OnPhysicsStep'),
                ('Context', 'isaacsim.ros2.bridge.ROS2Context'),
                ('ReadSimTime', 'isaacsim.core.nodes.IsaacReadSimulationTime'),
                ('ReadIMU', 'isaacsim.sensors.physics.IsaacReadIMU'),
                ('PublishIMU', 'isaacsim.ros2.bridge.ROS2PublishImu'),
            ],
            connect=[
                ('OnPhysicsStep.outputs:step', 'ReadIMU.inputs:execIn'),
                ('OnPhysicsStep.outputs:step', 'PublishIMU.inputs:execIn'),
                ('ReadIMU.outputs:angVel', 'PublishIMU.inputs:angularVelocity'),
                ('ReadIMU.outputs:linAcc', 'PublishIMU.inputs:linearAcceleration'),
                ('ReadIMU.outputs:orientation', 'PublishIMU.inputs:orientation'),
                ('Context.outputs:context', 'PublishIMU.inputs:context'),
                ('ReadSimTime.outputs:simulationTime', 'PublishIMU.inputs:timeStamp'),
            ],
            set_values=[
                ('ReadIMU.inputs:imuPrim', [usdrt.Sdf.Path(imu)]),
                ('ReadIMU.inputs:readGravity', True),
                ('PublishIMU.inputs:topicName', _IMU_TOPIC),
                ('PublishIMU.inputs:frameId', _IMU_FRAME_ID),
            ],
        )

        print(
            '[add-ros2-bridge] authored /ROS2_JointStates '
            f'({_JOINT_STATES_TOPIC}), /ROS2_JointCommands '
            f'({_JOINT_COMMANDS_TOPIC}), /ROS2_Clock ({_CLOCK_TOPIC}), '
            f'/ROS2_IMU ({_IMU_TOPIC}, sensor {imu})',
        )
        _export_stage(stage, usd_in, usd_out)
        print(f'[add-ros2-bridge] saved to {usd_out}')

    finally:
        sim_app.close()


if __name__ == '__main__':
    main()

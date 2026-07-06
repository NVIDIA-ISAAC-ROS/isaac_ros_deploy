#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reusable launch file for the inference graph pipeline.

This launch file creates the core inference pipeline nodes:
- InputBuilderNode: Converts ROS messages to a single bundled TensorList
- TritonNode: Runs neural network inference via a Triton ensemble model
- OutputBuilderNode: Converts output TensorList to typed ROS messages

All nodes are placed in a configurable namespace (default: 'inference_graph').
InputBuilderNode publishes a single TensorList with all model inputs.
TritonNode receives the bundled TensorList and publishes inference outputs.
OutputBuilderNode subscribes to the output TensorList.

The InputBuilderNode uses source_to_topic parameters to map input sources to
ROS topics. Pass these via the source_to_topic launch arguments:
  source_to_topic.state/joint/position:=joint_states
  source_to_topic.state/joint/velocity:=joint_states
  source_to_topic.state/body/rotation:=imu

Usage:
    Include this launch file from your application-specific launch file
    and apply remappings using GroupAction with SetRemap.
"""

import os
from pathlib import Path
import tempfile

from isaac_ros_deploy_converters.create_triton_model_repo import create_triton_model_repo
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for inference pipeline nodes."""
    declared_arguments = [
        DeclareLaunchArgument(
            'config_path',
            description='Path to the inference pipeline YAML config file.',
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='50.0',
            description='Rate at which InputBuilderNode publishes (Hz).',
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='inference_graph',
            description='Namespace for all pipeline nodes and topics.',
        ),
        DeclareLaunchArgument(
            'container_name',
            default_value='',
            description='Name of an existing container to load '
                        'nodes into. If empty, a new container '
                        'is created.',
        ),
        DeclareLaunchArgument(
            'source_to_topic',
            default_value='',
            description='Comma-separated source:topic mappings '
                        'for InputBuilderNode. E.g., '
                        '"state/joint/position:joint_states,'
                        'state/joint/velocity:joint_states,'
                        'state/body/rotation:imu". '
                        'Sources not listed default to their '
                        'own name as topic.',
        ),
        DeclareLaunchArgument(
            'source_message_type',
            default_value='',
            description='Comma-separated source:message_type overrides. '
                        'E.g., "velocity_command:geometry_msgs/msg/TwistStamped". '
                        'Sources not listed use the converter default.',
        ),
        DeclareLaunchArgument(
            'output_to_topic',
            default_value='',
            description='Comma-separated output:topic overrides '
                        'for OutputBuilderNode. E.g., '
                        '"bo1:joint_commands". Outputs not listed '
                        'fall back to the converter message-type default '
                        '(JointCommand -> joint_commands, '
                        'JointCommandTrajectory -> joint_commands_trajectory, '
                        'BodyCommand -> body_commands, Twist -> cmd_vel).',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Run the pipeline nodes on sim time so their clock '
                        'matches MuJoCo / controllers when replaying '
                        'recorded data through the graph in tests.',
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=_launch)]
    )


def csv_string_to_dict(mapping_str: str, prefix: str) -> dict[str, str]:
    """Parse a comma-separated 'key:value' string into prefixed ROS params."""
    params: dict[str, str] = {}
    if not mapping_str:
        return params
    for entry in mapping_str.split(','):
        entry = entry.strip()
        if ':' in entry:
            key, value = entry.split(':', 1)
            params[f'{prefix}.{key.strip()}'] = value.strip()
    return params


def _launch(context):
    config_path = LaunchConfiguration('config_path').perform(context)

    namespace = LaunchConfiguration('namespace')
    container_name = LaunchConfiguration('container_name')
    use_existing = PythonExpression(
        ["'", container_name, "' != ''"]
    )

    source_to_topic_params = csv_string_to_dict(
        LaunchConfiguration('source_to_topic').perform(context),
        'source_to_topic',
    )
    source_message_type_params = csv_string_to_dict(
        LaunchConfiguration('source_message_type').perform(context),
        'source_message_type',
    )
    output_to_topic_params = csv_string_to_dict(
        LaunchConfiguration('output_to_topic').perform(context),
        'output_to_topic',
    )

    use_sim_time = (
        LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    )
    sim_time_params = {'use_sim_time': use_sim_time}

    input_builder_params = {
        'config_path': config_path,
        'publish_rate': float(LaunchConfiguration('publish_rate').perform(context)),
        'output_topic': 'input_tensors',
        **sim_time_params,
        **source_to_topic_params,
        **source_message_type_params,
    }

    input_builder_node = ComposableNode(
        package='isaac_ros_deploy_converters',
        plugin='isaac_ros_deploy_converters::InputBuilderNode',
        name='input_builder_node',
        namespace=namespace,
        parameters=[input_builder_params],
    )

    # Create Triton ensemble model repository.
    # In test environments ($TEST_TMPDIR set by Bazel), use the per-test temp
    # directory to avoid collisions between parallel tests on shared workers.
    # Otherwise use a deterministic path under /tmp so repeated launches reuse
    # the same directory instead of accumulating unbounded temp dirs.
    # The Triton server reads from this directory at runtime, so it must outlive
    # the launch function (ruling out TemporaryDirectory context manager).
    config_file = Path(config_path)
    test_tmpdir = os.environ.get('TEST_TMPDIR')
    if test_tmpdir:
        repo_dir = Path(test_tmpdir) / f'triton_repo_{config_file.stem}'
    else:
        repo_dir = Path(tempfile.gettempdir()) / f'triton_repo_{config_file.stem}'
    try:
        repo = create_triton_model_repo(config_file, repo_dir)
    except FileNotFoundError as e:
        raise RuntimeError(
            f'Failed to create Triton model repo: {e}. '
            f'Check that model paths in \'{config_path}\' are correct.'
        ) from e
    except Exception as e:
        raise RuntimeError(
            f'Failed to create Triton model repo from \'{config_path}\': {e}'
        ) from e
    triton_node_name = f'triton_{repo.model_name}'.replace('-', '_')

    triton_params = {
        'model_name': repo.model_name,
        'model_repository_paths': [str(repo_dir)],
        'max_batch_size': 0,
        'num_concurrent_requests': 1,
        'input_tensor_names': repo.input_tensor_names,
        'input_binding_names': repo.input_binding_names,
        'output_tensor_names': repo.output_tensor_names,
        'output_binding_names': repo.output_binding_names,
        'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
        'output_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
        'enable_triton_logging': False,
        'log_level': 0,
        **sim_time_params,
    }

    triton_node = ComposableNode(
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        name=triton_node_name,
        namespace=namespace,
        parameters=[triton_params],
        remappings=[
            ('tensor_pub', 'input_tensors'),
            ('tensor_sub', 'output_tensors'),
        ],
    )

    output_builder_params = {
        'config_path': config_path,
        'input_topic': 'output_tensors',
        **sim_time_params,
        **output_to_topic_params,
    }

    output_builder_node = ComposableNode(
        package='isaac_ros_deploy_converters',
        plugin='isaac_ros_deploy_converters::OutputBuilderNode',
        name='output_builder_node',
        namespace=namespace,
        parameters=[output_builder_params],
    )

    composable_nodes = [input_builder_node, triton_node, output_builder_node]

    # Create a new container if no existing container specified.
    new_container = ComposableNodeContainer(
        name='inference_pipeline_container',
        namespace=namespace,
        package='isaac_ros_deploy_converters',
        executable='component_container',
        output='screen',
        condition=UnlessCondition(use_existing),
    )

    # Load nodes: use the provided container_name or fall back to the new container.
    new_container_name = [namespace, '/inference_pipeline_container']
    target = PythonExpression(
        ["'", container_name, "' if '", container_name, "' != '' else '",
         *new_container_name, "'"]
    )
    load_nodes = LoadComposableNodes(
        target_container=target,
        composable_node_descriptions=composable_nodes,
    )

    return [new_container, load_nodes]

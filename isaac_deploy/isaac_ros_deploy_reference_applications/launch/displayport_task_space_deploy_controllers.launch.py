#!/usr/bin/env python3

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

import os
import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml


APP_CONFIG_KEY = 'displayport_task_space'


def _value(context, name):
    return LaunchConfiguration(name).perform(context)


def _optional_value(context, name):
    try:
        value = _value(context, name).strip()
    except Exception:
        return None
    return value if value else None


def _config_value(context, config, name, default=None):
    override = _optional_value(context, name)
    if override is not None:
        return override
    return config.get(name, default)


def _as_bool(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('true', '1', 'yes', 'on'):
            return True
        if lowered in ('false', '0', 'no', 'off'):
            return False
    raise ValueError(f'{name} must be a boolean, got {value!r}')


def _as_float(value, name):
    result = float(value)
    if not result >= 0.0:
        raise ValueError(f'{name} must be non-negative, got {result}')
    return result


def _as_int(value, name):
    result = int(value)
    if result <= 0:
        raise ValueError(f'{name} must be positive, got {result}')
    return result


def _as_string_list(value, name):
    if isinstance(value, str):
        parsed = yaml.safe_load(value)
    else:
        parsed = value
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f'{name} must be a list of strings, got {value!r}')
    return parsed


def _as_string_list_map(value, name):
    if isinstance(value, str):
        parsed = yaml.safe_load(value)
    else:
        parsed = value
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError(f'{name} must be a map of string lists, got {value!r}')
    result = {}
    for key, items in parsed.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f'{name} keys must be non-empty strings, got {key!r}')
        result[key.strip()] = _as_string_list(items, f'{name}.{key}')
    return result


def _load_app_config(path):
    if not os.path.isfile(path):
        raise ValueError(f'DisplayPort task-space config does not exist: {path}')
    with open(path, 'r', encoding='utf-8') as config_file:
        data = yaml.safe_load(config_file) or {}
    config = data.get(APP_CONFIG_KEY, data)
    if not isinstance(config, dict):
        raise ValueError(f'DisplayPort task-space config must be a map, got {config!r}')
    return config


def _write_controller_params(context):
    app_config_path = _value(context, 'app_config_path').strip()
    app_config = _load_app_config(app_config_path)

    deploy_config_path = _value(context, 'config_path').strip()
    if not deploy_config_path:
        raise ValueError('config_path is required')
    if not os.path.isfile(deploy_config_path):
        raise ValueError(f'Deploy config does not exist: {deploy_config_path}')

    inference = str(_config_value(
        context, app_config, 'inference_controller_name',
        'displayport_task_space_inference_controller')).strip()
    safety = str(_config_value(
        context, app_config, 'safety_controller_name',
        'displayport_task_space_safety_controller')).strip()
    if not inference or not safety or inference == safety:
        raise ValueError('controller names must be non-empty and distinct')

    controller_manager_name = str(_config_value(
        context, app_config, 'controller_manager_name', '/controller_manager')).strip()
    controller_manager_timeout_s = str(_config_value(
        context, app_config, 'controller_manager_timeout_s', 10.0)).strip()
    start_inactive = _as_bool(_config_value(
        context, app_config, 'start_inactive', True), 'start_inactive')
    use_sim_time = _as_bool(_config_value(
        context, app_config, 'use_sim_time', False), 'use_sim_time')

    socket_pose_topic = str(_config_value(
        context, app_config, 'socket_pose_topic', '/displayport_socket_pose')).strip()
    socket_sources = _as_string_list(
        _config_value(
            context, app_config, 'socket_sources',
            ['socket_kp_pos', 'socket_kp_rot_6d']),
        'socket_sources')
    socket_message_type = str(_config_value(
        context, app_config, 'socket_message_type',
        'geometry_msgs/msg/PoseStamped')).strip()

    arm_action_name = str(_config_value(
        context, app_config, 'arm_action_name', 'arm_action')).strip()
    current_observation_name = str(_config_value(
        context, app_config, 'current_observation_name',
        'current_observation')).strip()
    input_interfaces = _as_string_list_map(
        _config_value(context, app_config, 'input_interfaces', {}),
        'input_interfaces')
    current_observation_reference_inputs = _as_string_list_map(
        _config_value(
            context, app_config, 'current_observation_reference_inputs', {}),
        'current_observation_reference_inputs')
    if not current_observation_reference_inputs:
        raise ValueError('current_observation_reference_inputs must configure at least one source')
    action_element_names = _as_string_list(
        _config_value(
            context, app_config, 'action_element_names',
            [
                'delta_x',
                'delta_y',
                'delta_z',
                'delta_axis_angle_x',
                'delta_axis_angle_y',
                'delta_axis_angle_z',
            ]),
        'action_element_names')
    eef_state_position_interfaces = _as_string_list(
        _config_value(
            context, app_config, 'eef_state_position_interfaces', []),
        'eef_state_position_interfaces')
    eef_state_orientation_interfaces = _as_string_list(
        _config_value(
            context, app_config, 'eef_state_orientation_interfaces', []),
        'eef_state_orientation_interfaces')
    eef_command_position_interfaces = _as_string_list(
        _config_value(
            context, app_config, 'eef_command_position_interfaces', []),
        'eef_command_position_interfaces')
    eef_command_orientation_interfaces = _as_string_list(
        _config_value(
            context, app_config, 'eef_command_orientation_interfaces', []),
        'eef_command_orientation_interfaces')
    command_prefix = f'{safety}/{arm_action_name}'
    reference_input_interfaces = {
        source: [
            f'{safety}/{current_observation_name}/{component}'
            for component in components
        ]
        for source, components in current_observation_reference_inputs.items()
    }
    publish_debug_topics = _as_bool(_config_value(
        context, app_config, 'publish_debug_topics', False),
        'publish_debug_topics')

    param_data = {
        'controller_manager': {'ros__parameters': {
            safety: {'type': 'isaac_ros_deploy_ros2_control/TaskSpaceSafetyController'},
            inference: {'type': 'isaac_ros_deploy_ros2_control/InferenceController'},
        }},
        safety: {'ros__parameters': {
            'type': 'isaac_ros_deploy_ros2_control/TaskSpaceSafetyController',
            'arm_action_name': arm_action_name,
            'current_observation_name': current_observation_name,
            'action_element_names': action_element_names,
            'state_position_interfaces': eef_state_position_interfaces,
            'state_orientation_interfaces': eef_state_orientation_interfaces,
            'command_position_interfaces': eef_command_position_interfaces,
            'command_orientation_interfaces': eef_command_orientation_interfaces,
            'blend_ratio': _as_float(_config_value(
                context, app_config, 'blend_ratio', 1.0), 'blend_ratio'),
            'use_sim_time': use_sim_time,
        }},
        inference: {'ros__parameters': {
            'type': 'isaac_ros_deploy_ros2_control/InferenceController',
            'config_path': deploy_config_path,
            'decimation': _as_int(_config_value(
                context, app_config, 'decimation', 1), 'decimation'),
            # The task-space safety controller exports arm_action/delta_x_raw, etc.
            # This keeps policy output as raw LEAPP action until safety applies blend_ratio.
            'command_prefix': command_prefix,
            'command_suffix': '_raw',
            'input_interfaces': input_interfaces,
            'reference_input_sources': list(current_observation_reference_inputs.keys()),
            'reference_input_interfaces': reference_input_interfaces,
            'topic_input_sources': socket_sources,
            'topic_input_timeout_ms': _as_float(_config_value(
                context, app_config, 'topic_input_timeout_ms', 250.0),
                'topic_input_timeout_ms'),
            'source_to_topic': {source: socket_pose_topic for source in socket_sources},
            'source_message_type': {
                source: socket_message_type for source in socket_sources},
            'publish_debug_topics': publish_debug_topics,
            'log_debug_to_console': _as_bool(_config_value(
                context, app_config, 'log_debug_to_console', False),
                'log_debug_to_console'),
            'debug_action_output_name': str(_config_value(
                context, app_config, 'debug_action_output_name', 'arm_action')).strip(),
            'use_sim_time': use_sim_time,
        }},
    }

    output = tempfile.NamedTemporaryFile(
        mode='w', prefix='displayport_task_space_ros2_control_', suffix='.yaml', delete=False)
    with output:
        yaml.safe_dump(param_data, output, sort_keys=False)
    return (
        output.name,
        controller_manager_name,
        controller_manager_timeout_s,
        safety,
        inference,
        start_inactive,
    )


def _remove_file(context, path):
    del context
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    return []


def _launch(context, *args, **kwargs):
    del args, kwargs
    (
        controller_params,
        controller_manager_name,
        controller_manager_timeout_s,
        safety,
        inference,
        start_inactive,
    ) = _write_controller_params(context)

    spawner_args = [
        safety,
        inference,
        '--controller-manager', controller_manager_name,
        '--controller-manager-timeout', controller_manager_timeout_s,
        '--service-call-timeout', controller_manager_timeout_s,
        '--param-file', controller_params,
    ]
    if start_inactive:
        spawner_args.append('--inactive')

    spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=spawner_args,
        output='both',
    )
    cleanup = RegisterEventHandler(
        OnProcessExit(
            target_action=spawner,
            on_exit=[OpaqueFunction(function=_remove_file, args=[controller_params])]))
    return [cleanup, spawner]


def generate_launch_description():
    default_app_config = PathJoinSubstitution([
        FindPackageShare('isaac_ros_deploy_reference_applications'),
        'config',
        'displayport_task_space_ros2_control.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'app_config_path',
            default_value=default_app_config,
            description='Task-space DisplayPort ros2_control app config.'),
        DeclareLaunchArgument(
            'config_path',
            default_value='',
            description='Path to the task-space LEAPP Deploy YAML.'),
        DeclareLaunchArgument('controller_manager_timeout_s', default_value=''),
        DeclareLaunchArgument(
            'controller_manager_name',
            default_value='',
            description='Optional controller-manager name override.'),
        DeclareLaunchArgument('inference_controller_name', default_value=''),
        DeclareLaunchArgument('safety_controller_name', default_value=''),
        DeclareLaunchArgument('start_inactive', default_value=''),
        DeclareLaunchArgument('socket_pose_topic', default_value=''),
        DeclareLaunchArgument('socket_sources', default_value=''),
        DeclareLaunchArgument('socket_message_type', default_value=''),
        DeclareLaunchArgument('eef_state_position_interfaces', default_value=''),
        DeclareLaunchArgument('eef_state_orientation_interfaces', default_value=''),
        DeclareLaunchArgument('eef_command_position_interfaces', default_value=''),
        DeclareLaunchArgument('eef_command_orientation_interfaces', default_value=''),
        DeclareLaunchArgument('arm_action_name', default_value=''),
        DeclareLaunchArgument('action_element_names', default_value=''),
        DeclareLaunchArgument('current_observation_name', default_value=''),
        DeclareLaunchArgument('input_interfaces', default_value=''),
        DeclareLaunchArgument('current_observation_reference_inputs', default_value=''),
        DeclareLaunchArgument('decimation', default_value=''),
        DeclareLaunchArgument('topic_input_timeout_ms', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value=''),
        DeclareLaunchArgument('publish_debug_topics', default_value=''),
        DeclareLaunchArgument('log_debug_to_console', default_value=''),
        DeclareLaunchArgument('debug_action_output_name', default_value=''),
        DeclareLaunchArgument(
            'blend_ratio',
            default_value='',
            description='Optional task-space current_observation blend ratio override.'),
        OpaqueFunction(function=_launch),
    ])

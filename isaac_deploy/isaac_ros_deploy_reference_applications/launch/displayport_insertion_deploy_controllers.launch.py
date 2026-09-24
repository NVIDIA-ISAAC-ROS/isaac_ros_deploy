#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


APP_CONFIG_KEY = 'displayport_insertion'


def _value(context, name):
    return LaunchConfiguration(name).perform(context)


def _optional_value(context, name):
    value = _value(context, name).strip()
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


def _as_float(value, name, allow_zero=True):
    result = float(value)
    if result < 0.0 or (result == 0.0 and not allow_zero):
        requirement = 'non-negative' if allow_zero else 'positive'
        raise ValueError(f'{name} must be {requirement}, got {result}')
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


def _load_app_config(path):
    if not os.path.isfile(path):
        raise ValueError(f'DisplayPort insertion config does not exist: {path}')
    with open(path, 'r', encoding='utf-8') as config_file:
        data = yaml.safe_load(config_file) or {}
    config = data.get(APP_CONFIG_KEY, data)
    if not isinstance(config, dict):
        raise ValueError(f'DisplayPort insertion config must be a map, got {config!r}')
    return config


def _prefixed_joints(joints, prefix):
    if not prefix:
        return joints
    return [joint if joint.startswith(prefix) else f'{prefix}{joint}' for joint in joints]


def _write_controller_params(context):
    app_config_path = _value(context, 'app_config_path').strip()
    app_config = _load_app_config(app_config_path)

    deploy_config_path = _value(context, 'config_path').strip()
    if not deploy_config_path:
        raise ValueError('config_path is required')
    if not os.path.isfile(deploy_config_path):
        raise ValueError(f'Deploy config does not exist: {deploy_config_path}')

    inference = str(_config_value(
        context, app_config, 'inference_controller_name', 'deploy_inference_controller')).strip()
    safety = str(_config_value(
        context, app_config, 'safety_controller_name', 'deploy_safety_controller')).strip()
    if not inference or not safety or inference == safety:
        raise ValueError('controller names must be non-empty and distinct')

    robot_sn = str(_config_value(context, app_config, 'robot_sn', '')).strip()
    joint_name_prefix = str(_config_value(context, app_config, 'joint_name_prefix', '')).strip()
    if not joint_name_prefix and robot_sn:
        joint_name_prefix = f'{robot_sn}_'

    joints = _prefixed_joints(
        _as_string_list(_config_value(context, app_config, 'joints', []), 'joints'),
        joint_name_prefix)
    if not joints:
        raise ValueError('joints must not be empty')

    goal_pose_topic = str(_config_value(
        context, app_config, 'goal_pose_topic', '/goal_pose')).strip()
    if not goal_pose_topic:
        raise ValueError('goal_pose_topic must not be empty')

    controller_manager_name = str(_config_value(
        context, app_config, 'controller_manager_name', '/controller_manager')).strip()
    controller_manager_timeout_s = str(_config_value(
        context, app_config, 'controller_manager_timeout_s', 10.0)).strip()
    start_inactive = _as_bool(_config_value(
        context, app_config, 'start_inactive', True), 'start_inactive')
    use_sim_time = _as_bool(_config_value(
        context, app_config, 'use_sim_time', False), 'use_sim_time')

    # The policy I/O logger consumes the controller's debug topics, so enabling
    # it force-enables publish_debug_topics regardless of the app-config value.
    log_policy_io = _as_bool(
        _optional_value(context, 'log_policy_io') or 'false', 'log_policy_io')
    publish_debug_topics = _as_bool(_config_value(
        context, app_config, 'publish_debug_topics', False), 'publish_debug_topics')
    if log_policy_io:
        publish_debug_topics = True

    # Blend-scaled per-joint delta (pre-blend safety command). Enabling the logger
    # force-enables its publisher so it can be recorded.
    publish_scaled_joint_delta = _as_bool(_config_value(
        context, app_config, 'publish_scaled_joint_delta', False),
        'publish_scaled_joint_delta')
    scaled_joint_delta_topic = str(_config_value(
        context, app_config, 'scaled_joint_delta_topic', '~/scaled_joint_delta')).strip()
    if log_policy_io:
        publish_scaled_joint_delta = True

    # Resolve a (possibly relative / private) controller topic the way ros2_control
    # would, so a plain subscriber can find it. '~' resolves to the controller node.
    def _resolve_controller_topic(topic):
        if topic.startswith('~/'):
            return f'/{safety}/' + topic[2:]
        if topic.startswith('/'):
            return topic
        return f'/{safety}/{topic}'

    safety_command_topic = _resolve_controller_topic(scaled_joint_delta_topic)

    # Blended / safety-limited ABSOLUTE command actually written to hardware
    # (position = clamped safe target; vel/eff/kp/kd = post-blend values). This
    # is the true safety-gated command, distinct from scaled_joint_delta (a delta).
    # The logger force-enables its publisher so it can be recorded.
    publish_blended_command = _as_bool(_config_value(
        context, app_config, 'publish_blended_command', False),
        'publish_blended_command')
    blended_command_topic_param = str(_config_value(
        context, app_config, 'blended_command_topic', '~/blended_command')).strip()
    if log_policy_io:
        publish_blended_command = True
    blended_command_topic = _resolve_controller_topic(blended_command_topic_param)

    # Measured end-effector (flange) pose in the world frame, published by the
    # Flexiv robot-states broadcaster on /<robot_sn>/flange_pose ('-' -> '_').
    robot_sn_topic = robot_sn.replace('-', '_')
    eef_pose_topic = f'/{robot_sn_topic}/flange_pose' if robot_sn_topic else ''

    goal_sources = _as_string_list(
        _config_value(context, app_config, 'goal_sources', []), 'goal_sources')
    if not goal_sources:
        raise ValueError('goal_sources must not be empty')
    goal_message_type = str(_config_value(
        context, app_config, 'goal_message_type', 'geometry_msgs/msg/PoseStamped')).strip()
    if not goal_message_type:
        raise ValueError('goal_message_type must not be empty')

    source_to_topic = {source: goal_pose_topic for source in goal_sources}
    source_message_type = {source: goal_message_type for source in goal_sources}

    param_data = {
        'controller_manager': {'ros__parameters': {
            safety: {'type': 'isaac_ros_deploy_ros2_control/SafetyController'},
            inference: {'type': 'isaac_ros_deploy_ros2_control/InferenceController'},
        }},
        safety: {'ros__parameters': {
            'type': 'isaac_ros_deploy_ros2_control/SafetyController',
            'joints': joints,
            'hardware_command_interfaces': _as_string_list(
                _config_value(
                    context, app_config, 'hardware_command_interfaces', ['position']),
                'hardware_command_interfaces'),
            'blend_strategy': str(_config_value(
                context, app_config, 'blend_strategy', 'interpolate')).strip(),
            'blend_reference': str(_config_value(
                context, app_config, 'blend_reference', 'activation')).strip(),
            'blend_ratio': _as_float(_config_value(
                context, app_config, 'blend_ratio', 0.0), 'blend_ratio'),
            'max_blend_ratio_speed': _as_float(_config_value(
                context, app_config, 'max_blend_ratio_speed', 1.0), 'max_blend_ratio_speed'),
            'publish_scaled_joint_delta': publish_scaled_joint_delta,
            'scaled_joint_delta_topic': scaled_joint_delta_topic,
            'publish_blended_command': publish_blended_command,
            'blended_command_topic': blended_command_topic_param,
            'use_sim_time': use_sim_time,
        }},
        inference: {'ros__parameters': {
            'type': 'isaac_ros_deploy_ros2_control/InferenceController',
            'config_path': deploy_config_path,
            'decimation': _as_int(_config_value(
                context, app_config, 'decimation', 1), 'decimation'),
            'command_prefix': safety,
            'command_suffix': str(_config_value(
                context, app_config, 'command_suffix', '_raw')).strip(),
            'joint_name_prefix': joint_name_prefix,
            'topic_input_sources': goal_sources,
            'topic_input_timeout_ms': _as_float(_config_value(
                context, app_config, 'topic_input_timeout_ms', 250.0),
                'topic_input_timeout_ms'),
            'source_to_topic': source_to_topic,
            'source_message_type': source_message_type,
            'publish_debug_topics': publish_debug_topics,
            'log_debug_to_console': _as_bool(_config_value(
                context, app_config, 'log_debug_to_console', False), 'log_debug_to_console'),
            'debug_action_output_name': str(_config_value(
                context, app_config, 'debug_action_output_name', 'arm_action')).strip(),
            'use_sim_time': use_sim_time,
        }},
    }

    output = tempfile.NamedTemporaryFile(
        mode='w', prefix='displayport_insertion_ros2_control_', suffix='.yaml', delete=False)
    with output:
        yaml.safe_dump(param_data, output, sort_keys=False)
    return (
        output.name,
        controller_manager_name,
        controller_manager_timeout_s,
        safety,
        inference,
        start_inactive,
        goal_pose_topic,
        use_sim_time,
        eef_pose_topic,
        safety_command_topic,
        blended_command_topic,
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
        param_file,
        controller_manager_name,
        controller_manager_timeout_s,
        safety,
        inference,
        start_inactive,
        goal_pose_topic,
        use_sim_time,
        eef_pose_topic,
        safety_command_topic,
        blended_command_topic,
    ) = _write_controller_params(context)

    spawner_args = [
        safety,
        inference,
        '--controller-manager', controller_manager_name,
        '--controller-manager-timeout', controller_manager_timeout_s,
        '--service-call-timeout', controller_manager_timeout_s,
        '--param-file', param_file,
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
            on_exit=[OpaqueFunction(function=_remove_file, args=[param_file])]))
    actions = [cleanup, spawner]

    # Optional passive policy I/O logger (console + CSV) for debugging and data
    # collection. Consumes the inference controller's debug topics, which
    # _write_controller_params force-enables when log_policy_io is set.
    if _as_bool(_optional_value(context, 'log_policy_io') or 'false', 'log_policy_io'):
        joint_states_topic = _optional_value(context, 'joint_states_topic') \
            or '/flexiv_arm/joint_states'
        # Allow an explicit override of the flange (eef) pose topic; otherwise
        # use the one derived from robot_sn.
        eef_pose_topic = (_optional_value(context, 'eef_pose_topic') or '').strip() \
            or eef_pose_topic
        # Allow an explicit override of the pre-blend command topic. Defaults to
        # the SafetyController scaled_joint_delta (a per-joint delta); point it
        # at e.g. /<safety>/final_joint_command to log the absolute pre-blend
        # target if a JointCommandBroadcaster publishes it.
        safety_command_topic = (_optional_value(context, 'safety_command_topic') or '').strip() \
            or safety_command_topic
        # Blended / safety-limited absolute command actually sent to hardware
        # (SafetyController blended_command). Overridable; defaults to the
        # resolved /<safety>/blended_command topic.
        blended_command_topic = (_optional_value(context, 'blended_command_topic') or '').strip() \
            or blended_command_topic
        # Recurrent (LSTM) hidden-state topic published by the inference controller
        # when publish_debug_topics is on (force-enabled by log_policy_io).
        # Overridable; defaults to /<inference>/debug_recurrent_state.
        recurrent_state_topic = (_optional_value(context, 'recurrent_state_topic') or '').strip() \
            or f'/{inference}/debug_recurrent_state'
        csv_path = _value(context, 'policy_io_csv_path').strip()
        every_n = _as_int(
            _optional_value(context, 'log_policy_io_every_n') or '1',
            'log_policy_io_every_n')
        # Whether to log the pre-blend safety command group (safety_cmd_*).
        # Defaults to true; set log_safety_command:=false to drop it entirely
        # (e.g. when final_joint_command isn't published, to avoid the startup
        # grace wait). action_* and blend_cmd_* are unaffected.
        log_safety_command = _as_bool(
            _optional_value(context, 'log_safety_command') or 'true',
            'log_safety_command')
        actions.append(
            Node(
                package='isaac_ros_deploy_reference_applications',
                executable='policy_io_logger.py',
                name='policy_io_logger',
                parameters=[{
                    'obs_topic': f'/{inference}/debug_observation',
                    'action_topic': f'/{inference}/debug_action',
                    'goal_pose_topic': goal_pose_topic,
                    'joint_states_topic': joint_states_topic,
                    'eef_pose_topic': eef_pose_topic,
                    'safety_command_topic': safety_command_topic,
                    'blended_command_topic': blended_command_topic,
                    'recurrent_state_topic': recurrent_state_topic,
                    'log_safety_command': log_safety_command,
                    'csv_output_path': csv_path,
                    'log_every_n_steps': every_n,
                    'use_sim_time': use_sim_time,
                }],
                output='both',
            )
        )
    return actions


def generate_launch_description():
    default_app_config_path = PathJoinSubstitution([
        FindPackageShare('isaac_ros_deploy_reference_applications'),
        'config',
        'displayport_insertion_ros2_control.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_path',
            description=(
                'Absolute path to the LEAPP-exported Isaac ROS Deploy configuration YAML.'
            )),
        DeclareLaunchArgument(
            'app_config_path',
            default_value=default_app_config_path,
            description='DisplayPort insertion ros2_control application config YAML.'),
        DeclareLaunchArgument('robot_sn', default_value=''),
        DeclareLaunchArgument('joint_name_prefix', default_value=''),
        DeclareLaunchArgument('goal_pose_topic', default_value=''),
        DeclareLaunchArgument('joints', default_value=''),
        DeclareLaunchArgument('hardware_command_interfaces', default_value=''),
        DeclareLaunchArgument('goal_sources', default_value=''),
        DeclareLaunchArgument('goal_message_type', default_value=''),
        DeclareLaunchArgument('command_suffix', default_value=''),
        DeclareLaunchArgument('controller_manager_name', default_value=''),
        DeclareLaunchArgument('controller_manager_timeout_s', default_value=''),
        DeclareLaunchArgument('inference_controller_name', default_value=''),
        DeclareLaunchArgument('safety_controller_name', default_value=''),
        DeclareLaunchArgument('start_inactive', default_value=''),
        DeclareLaunchArgument('decimation', default_value=''),
        DeclareLaunchArgument('topic_input_timeout_ms', default_value=''),
        DeclareLaunchArgument('blend_strategy', default_value=''),
        DeclareLaunchArgument('blend_reference', default_value=''),
        DeclareLaunchArgument('blend_ratio', default_value=''),
        DeclareLaunchArgument('max_blend_ratio_speed', default_value=''),
        DeclareLaunchArgument('publish_debug_topics', default_value=''),
        DeclareLaunchArgument('log_debug_to_console', default_value=''),
        DeclareLaunchArgument('debug_action_output_name', default_value=''),
        DeclareLaunchArgument('publish_scaled_joint_delta', default_value=''),
        DeclareLaunchArgument('scaled_joint_delta_topic', default_value=''),
        DeclareLaunchArgument('publish_blended_command', default_value=''),
        DeclareLaunchArgument('blended_command_topic', default_value=''),
        DeclareLaunchArgument(
            'recurrent_state_topic', default_value='',
            description='Recurrent (LSTM) hidden-state topic the policy I/O logger '
                        'records (std_msgs/Float64MultiArray). Defaults to '
                        '/<inference_controller>/debug_recurrent_state.'),
        DeclareLaunchArgument('use_sim_time', default_value=''),
        DeclareLaunchArgument(
            'log_policy_io', default_value='',
            description='Launch the passive policy_io_logger node (console + CSV) '
                        'to log observations, actions, joint states, goal/eef pose, '
                        'safety/blended commands and recurrent state. Force-enables '
                        'the inference/safety controller debug topics.'),
        DeclareLaunchArgument(
            'log_policy_io_every_n', default_value='',
            description='Console-log every Nth policy step (CSV always records '
                        'every step). Defaults to 1.'),
        DeclareLaunchArgument(
            'policy_io_csv_path', default_value='',
            description='If set, append per-step policy I/O to this CSV file for '
                        'offline data collection. Supports ~ and env vars.'),
        DeclareLaunchArgument(
            'joint_states_topic', default_value='',
            description='Joint states topic the policy I/O logger records. '
                        'Defaults to /flexiv_arm/joint_states.'),
        DeclareLaunchArgument(
            'eef_pose_topic', default_value='',
            description='Measured end-effector (flange) pose topic the policy I/O '
                        'logger records. Defaults to /<robot_sn>/flange_pose.'),
        DeclareLaunchArgument(
            'safety_command_topic', default_value='',
            description='Pre-blend safety command topic the policy I/O logger '
                        'records (isaac_ros_deploy_interfaces/JointCommand). Defaults '
                        'to the SafetyController scaled_joint_delta (a per-joint '
                        'delta). Set to /<safety_controller>/final_joint_command to '
                        'log the absolute pre-blend target instead.'),
        DeclareLaunchArgument(
            'log_safety_command', default_value='',
            description='Log the pre-blend safety command group (safety_cmd_*). '
                        'Defaults to true. Set to false to drop it entirely (e.g. '
                        'when final_joint_command is not published) so the logger '
                        'does not wait on it during startup. action_* and '
                        'blend_cmd_* are unaffected.'),
        OpaqueFunction(function=_launch),
    ])

#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch DisplayPort task-space policy inference with a Flexiv passthrough sink.

This is the topic-based deploy path: the policy bridge publishes an absolute
Cartesian target pose and the passthrough node forwards it to the Flexiv
Cartesian command sink. It does not use the ros2_control controller stack.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _robot_sn_topic(context):
    return LaunchConfiguration("robot_sn").perform(context).strip().replace("-", "_")


def _topic_or_robot_sn_default(context, name, suffix):
    """Return the explicit argument value, or one derived from ``robot_sn``."""
    value = LaunchConfiguration(name).perform(context).strip()
    if value:
        return value
    robot_sn = _robot_sn_topic(context)
    return f"/{robot_sn}/{suffix}" if robot_sn else ""


def _cartesian_impedance_service_name(context):
    value = LaunchConfiguration("cartesian_impedance_service_name").perform(context).strip()
    if value:
        return value
    robot_sn = _robot_sn_topic(context)
    if robot_sn:
        return f"/{robot_sn}_flexiv_hardware_services/set_cartesian_impedance"
    enabled = LaunchConfiguration("set_cartesian_impedance_on_start").perform(context)
    if enabled.strip().lower() in ("1", "true", "yes", "on"):
        raise ValueError(
            "cartesian_impedance_service_name or robot_sn is required when "
            "set_cartesian_impedance_on_start is true"
        )
    return ""


def _adapter_tcp_offset(context):
    """Return the flange->Flexiv-command offset for the passthrough.

    This is the offset Flexiv itself applies for the active tool, which is not
    necessarily the policy's control-frame offset. When the robot has no tool
    configured its command frame is the flange, so this is zero; when a tool is
    active it is that tool's offset. Defaults to tcp_offset for the common case
    where the two are the same.

    Returned as a list of floats because the passthrough declares ``tcp_offset``
    as a double array.
    """
    value = LaunchConfiguration("adapter_tcp_offset").perform(context).strip()
    if not value:
        value = LaunchConfiguration("tcp_offset").perform(context)
    values = [float(entry) for entry in value.split(",") if entry.strip()]
    if len(values) != 3:
        raise ValueError("adapter_tcp_offset must have 3 comma-separated values")
    return values


def _launch(context):
    robot_states_topic = _topic_or_robot_sn_default(
        context, "robot_states_topic", "flexiv_robot_states")
    flange_pose_topic = _topic_or_robot_sn_default(
        context, "flange_pose_topic", "flange_pose")
    policy_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("isaac_ros_deploy_reference_applications"),
                    "launch",
                    "displayport_task_space_policy.launch.py",
                ]
            )
        ),
        launch_arguments={
            "config_path": LaunchConfiguration("config_path"),
            "namespace": LaunchConfiguration("policy_namespace"),
            "container_name": LaunchConfiguration("container_name"),
            "publish_rate": LaunchConfiguration("publish_rate"),
            "max_policy_input_output_delay_s": LaunchConfiguration(
                "max_policy_input_output_delay_s"
            ),
            "use_sim_time": "false",
            "base_frame": LaunchConfiguration("base_frame"),
            "source_pose_reference": LaunchConfiguration("source_pose_reference"),
            "tcp_offset": LaunchConfiguration("tcp_offset"),
            "flange_pose_topic": flange_pose_topic,
            "socket_pose_topic": LaunchConfiguration("socket_pose_topic"),
            "socket_kp_position": LaunchConfiguration("socket_kp_position"),
            "socket_kp_quaternion_xyzw": LaunchConfiguration("socket_kp_quaternion_xyzw"),
            "socket_pose_is_keypoint": LaunchConfiguration("socket_pose_is_keypoint"),
            "socket_root_to_keypoint_offset": LaunchConfiguration(
                "socket_root_to_keypoint_offset"
            ),
            "require_socket_pose": LaunchConfiguration("require_socket_pose"),
            "socket_pose_timeout_s": LaunchConfiguration("socket_pose_timeout_s"),
            "target_pose_topic": LaunchConfiguration("policy_target_pose_topic"),
            "pose_delta_command_topic": LaunchConfiguration(
                "policy_pose_delta_command_topic"
            ),
            "published_target_pose_reference": LaunchConfiguration(
                "published_target_pose_reference"
            ),
            "action_blend_ratio": LaunchConfiguration("action_blend_ratio"),
            "action_tensor_name": LaunchConfiguration("action_tensor_name"),
            # The policy path shares the passthrough's gate so a trial opens and
            # closes the whole chain at once, and the recurrent state restarts with it.
            "enable_topic": LaunchConfiguration("adapter_enable_topic"),
            "enabled_on_start": LaunchConfiguration("adapter_enabled_on_start"),
            "reset_recurrent_state_on_enable": LaunchConfiguration(
                "policy_reset_recurrent_state_on_enable"
            ),
            "csv_output_path": LaunchConfiguration("policy_csv_output_path"),
            "debug_policy_io_topic": LaunchConfiguration("debug_policy_io_topic"),
        }.items(),
    )

    passthrough = Node(
        package="isaac_ros_deploy_reference_applications",
        executable="displayport_flexiv_task_space_passthrough.py",
        name="displayport_flexiv_task_space_passthrough",
        output="screen",
        parameters=[
            {
                "input_target_pose_topic": LaunchConfiguration("policy_target_pose_topic"),
                "output_target_pose_topic": LaunchConfiguration(
                    "flexiv_target_pose_topic"
                ),
                "enable_topic": LaunchConfiguration("adapter_enable_topic"),
                "feedback_pose_topic": flange_pose_topic,
                "end_effector_feedback_pose_topic": LaunchConfiguration(
                    "adapter_end_effector_feedback_pose_topic"
                ),
                "enabled_on_start": LaunchConfiguration("adapter_enabled_on_start"),
                "publish_pose_stamped": LaunchConfiguration("adapter_publish_pose_stamped"),
                "publish_body_command": LaunchConfiguration("adapter_publish_body_command"),
                "body_command_topic": LaunchConfiguration("adapter_body_command_topic"),
                "body_command_name": LaunchConfiguration("adapter_body_command_name"),
                "publish_flexiv_cartesian_command": LaunchConfiguration(
                    "adapter_publish_flexiv_cartesian_command"
                ),
                "flexiv_cartesian_command_topic": LaunchConfiguration(
                    "adapter_flexiv_cartesian_command_topic"
                ),
                "input_pose_reference": LaunchConfiguration("adapter_input_pose_reference"),
                "pose_delta_command_topic": LaunchConfiguration(
                    "policy_pose_delta_command_topic"
                ),
                "tcp_offset": _adapter_tcp_offset(context),
                "auto_tcp_offset_from_robot_states": LaunchConfiguration(
                    "auto_tcp_offset_from_robot_states"
                ),
                "robot_states_topic": robot_states_topic,
                "auto_tcp_offset_sample_count": LaunchConfiguration(
                    "auto_tcp_offset_sample_count"
                ),
                "auto_tcp_offset_max_translation_std_m": LaunchConfiguration(
                    "auto_tcp_offset_max_translation_std_m"
                ),
                "max_command_age_s": LaunchConfiguration("adapter_max_command_age_s"),
                "stale_command_timeout_s": LaunchConfiguration(
                    "adapter_stale_command_timeout_s"
                ),
                "stale_behavior": LaunchConfiguration("adapter_stale_behavior"),
                "require_user_confirmation": LaunchConfiguration(
                    "adapter_require_user_confirmation"
                ),
                "confirmation_requires_feedback": LaunchConfiguration(
                    "adapter_confirmation_requires_feedback"
                ),
                "csv_output_path": LaunchConfiguration("adapter_csv_output_path"),
            }
        ],
    )

    set_cartesian_impedance = Node(
        package="isaac_ros_deploy_reference_applications",
        executable="flexiv_set_cartesian_impedance.py",
        name="flexiv_set_cartesian_impedance",
        output="screen",
        condition=IfCondition(LaunchConfiguration("set_cartesian_impedance_on_start")),
        parameters=[
            {
                "service_name": _cartesian_impedance_service_name(context),
                "stiffness_mode": LaunchConfiguration(
                    "cartesian_impedance_stiffness_mode"
                ),
                "stiffness": LaunchConfiguration("cartesian_impedance_stiffness"),
                "nominal_stiffness": LaunchConfiguration(
                    "cartesian_impedance_nominal_stiffness"
                ),
                "stiffness_scale": LaunchConfiguration(
                    "cartesian_impedance_stiffness_scale"
                ),
                "damping_ratio": LaunchConfiguration(
                    "cartesian_impedance_damping_ratio"
                ),
                "service_timeout_sec": LaunchConfiguration(
                    "cartesian_impedance_service_timeout_sec"
                ),
            }
        ],
    )

    return [set_cartesian_impedance, policy_launch, passthrough]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_sn",
                default_value="",
                description=(
                    "Flexiv robot serial number used to derive the default Flexiv topic and "
                    "service names."
                ),
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value="",
                description="Absolute path to the task-space LEAPP Deploy YAML.",
            ),
            DeclareLaunchArgument(
                "policy_namespace",
                default_value="displayport_task_space_policy",
            ),
            DeclareLaunchArgument("container_name", default_value=""),
            DeclareLaunchArgument("publish_rate", default_value="15.0"),
            DeclareLaunchArgument("max_policy_input_output_delay_s", default_value="0.25"),
            DeclareLaunchArgument("base_frame", default_value="world"),
            DeclareLaunchArgument("tcp_offset", default_value="0.0,0.0,0.1925"),
            DeclareLaunchArgument("auto_tcp_offset_from_robot_states", default_value="false"),
            DeclareLaunchArgument(
                "robot_states_topic",
                default_value="",
                description="Defaults to /<robot_sn>/flexiv_robot_states.",
            ),
            DeclareLaunchArgument("auto_tcp_offset_sample_count", default_value="20"),
            DeclareLaunchArgument(
                "auto_tcp_offset_max_translation_std_m",
                default_value="0.002",
            ),
            DeclareLaunchArgument(
                "flange_pose_topic",
                default_value="",
                description="Defaults to /<robot_sn>/flange_pose.",
            ),
            DeclareLaunchArgument("socket_pose_topic", default_value="/displayport_socket_pose"),
            DeclareLaunchArgument("socket_kp_position", default_value="0.475,0.125,0.070"),
            DeclareLaunchArgument(
                "socket_kp_quaternion_xyzw",
                default_value="-0.5,-0.5,-0.5,0.5",
            ),
            DeclareLaunchArgument("socket_pose_is_keypoint", default_value="true"),
            DeclareLaunchArgument(
                "socket_root_to_keypoint_offset",
                default_value="0.0375,0.0,0.0",
            ),
            DeclareLaunchArgument("require_socket_pose", default_value="true"),
            DeclareLaunchArgument("socket_pose_timeout_s", default_value="0.0"),
            DeclareLaunchArgument(
                "policy_target_pose_topic",
                default_value="/displayport_task_space_policy/target_pose",
            ),
            DeclareLaunchArgument(
                "policy_pose_delta_command_topic",
                default_value="/displayport_task_space_policy/pose_delta_command",
                description=(
                    "Raw policy action anchored to its observation. The passthrough reads it "
                    "for the interactive confirmation summary."
                ),
            ),
            DeclareLaunchArgument("action_blend_ratio", default_value="0.35"),
            DeclareLaunchArgument("action_tensor_name", default_value="arm_action"),
            DeclareLaunchArgument("policy_csv_output_path", default_value=""),
            DeclareLaunchArgument("debug_policy_io_topic", default_value=""),
            DeclareLaunchArgument(
                "flexiv_target_pose_topic",
                default_value="/flexiv/displayport_task_space_target_pose",
            ),
            DeclareLaunchArgument(
                "adapter_enable_topic",
                default_value="/displayport_task_space_policy/enable_passthrough",
            ),
            DeclareLaunchArgument("adapter_enabled_on_start", default_value="false"),
            DeclareLaunchArgument(
                "policy_reset_recurrent_state_on_enable",
                default_value="true",
                description=(
                    "Restore the exported initial recurrent tensors each time the gate "
                    "opens, so repeated trials do not inherit the previous trial's "
                    "hidden state."
                ),
            ),
            DeclareLaunchArgument("source_pose_reference", default_value="flange"),
            DeclareLaunchArgument("published_target_pose_reference", default_value="flange"),
            DeclareLaunchArgument("adapter_publish_pose_stamped", default_value="true"),
            DeclareLaunchArgument("adapter_publish_body_command", default_value="false"),
            DeclareLaunchArgument("adapter_body_command_topic", default_value="/body_commands"),
            DeclareLaunchArgument("adapter_body_command_name", default_value="flange"),
            DeclareLaunchArgument(
                "adapter_publish_flexiv_cartesian_command",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "adapter_flexiv_cartesian_command_topic",
                default_value="/cartesian_motion_controller/command",
            ),
            DeclareLaunchArgument("adapter_input_pose_reference", default_value="flange"),
            DeclareLaunchArgument(
                "adapter_tcp_offset",
                default_value="",
                description=(
                    "Flange-to-command-frame offset the passthrough applies when "
                    "adapter_input_pose_reference is 'flange'. This is the Flexiv active tool "
                    "offset, which may differ from the policy's tcp_offset. Defaults to "
                    "tcp_offset."
                ),
            ),
            DeclareLaunchArgument(
                "adapter_end_effector_feedback_pose_topic",
                default_value="/cartesian_motion_controller/tcp_pose",
            ),
            DeclareLaunchArgument("adapter_max_command_age_s", default_value="0.25"),
            DeclareLaunchArgument("adapter_stale_command_timeout_s", default_value="0.25"),
            DeclareLaunchArgument("adapter_stale_behavior", default_value="stop_forwarding"),
            DeclareLaunchArgument("adapter_require_user_confirmation", default_value="false"),
            DeclareLaunchArgument("adapter_confirmation_requires_feedback", default_value="true"),
            DeclareLaunchArgument("adapter_csv_output_path", default_value=""),
            DeclareLaunchArgument("set_cartesian_impedance_on_start", default_value="true"),
            DeclareLaunchArgument(
                "cartesian_impedance_service_name",
                default_value="",
                description=(
                    "Defaults to /<robot_sn>_flexiv_hardware_services/set_cartesian_impedance."
                ),
            ),
            DeclareLaunchArgument(
                "cartesian_impedance_stiffness_mode",
                default_value="nominal_scaled",
            ),
            DeclareLaunchArgument(
                "cartesian_impedance_stiffness",
                default_value="2000.0,2000.0,2000.0,300.0,300.0,300.0",
            ),
            DeclareLaunchArgument(
                "cartesian_impedance_nominal_stiffness",
                default_value="10000.0,10000.0,10000.0,1500.0,1500.0,1500.0",
            ),
            DeclareLaunchArgument(
                "cartesian_impedance_stiffness_scale",
                default_value="0.3",
            ),
            DeclareLaunchArgument(
                "cartesian_impedance_damping_ratio",
                default_value="0.7,0.7,0.7,0.7,0.7,0.7",
            ),
            DeclareLaunchArgument(
                "cartesian_impedance_service_timeout_sec",
                default_value="10.0",
            ),
            OpaqueFunction(function=_launch),
        ]
    )

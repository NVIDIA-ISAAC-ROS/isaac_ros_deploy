#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the DisplayPort task-space Deploy policy path.

This launch file intentionally starts Triton plus task-space pose/tensor converters only.
It does not start the joint-space ros2_control InferenceController or
SafetyController because the task-space policy output is a relative Cartesian
pose delta, not an absolute joint command.
"""

import os
from pathlib import Path
import tempfile

from isaac_ros_deploy_converters.create_triton_model_repo import create_triton_model_repo
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode
import yaml


# InputBuilder source names are logical converter inputs. Pairs intentionally
# share one PoseStamped topic below while selecting different tensor converters.
EXPECTED_INPUT_SOURCES = {
    "eef_pos": ("eef_pose_pos", "state/body/position"),
    "eef_rot_6d": ("eef_pose_rot6d", "state/body/rotation_6d"),
    "socket_kp_pos": ("socket_kp_pose_pos", "state/body/position"),
    "socket_kp_rot_6d": ("socket_kp_pose_rot6d", "state/body/rotation_6d"),
}


def _as_bool(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _csv_string_to_list(csv_str):
    return [entry.strip() for entry in csv_str.split(",") if entry.strip()]


def _csv_float_list(context, name, expected_len):
    values = [
        float(entry)
        for entry in _csv_string_to_list(LaunchConfiguration(name).perform(context))
    ]
    if len(values) != expected_len:
        raise ValueError(f"{name} must have {expected_len} comma-separated values")
    return values


def _resolve_config_path(context):
    config_path = LaunchConfiguration("config_path").perform(context).strip()
    if not config_path:
        raise ValueError("config_path is required")
    return config_path


def _validate_input_builder_metadata(config_path):
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    models = config.get("models", {})
    for model_name, model_config in models.items():
        inputs = {
            str(input_config.get("name", "")): input_config
            for input_config in model_config.get("inputs", [])
        }
        if not all(name in inputs for name in EXPECTED_INPUT_SOURCES):
            continue
        errors = []
        for input_name, (expected_source, expected_kind) in EXPECTED_INPUT_SOURCES.items():
            input_config = inputs[input_name]
            actual_source = str(input_config.get("source", ""))
            actual_kind = str(input_config.get("kind", ""))
            if actual_source != expected_source or actual_kind != expected_kind:
                errors.append(
                    f"{input_name}: expected source={expected_source!r}, "
                    f"kind={expected_kind!r}; got source={actual_source!r}, "
                    f"kind={actual_kind!r}"
                )
        if errors:
            raise ValueError(
                "DisplayPort task-space export is missing InputBuilder source/kind metadata "
                f"for model {model_name!r}: " + "; ".join(errors)
            )
        return
    raise ValueError(
        "DisplayPort task-space export must contain inputs "
        f"{sorted(EXPECTED_INPUT_SOURCES)} with InputBuilder source/kind metadata"
    )


def _launch(context):
    config_path = _resolve_config_path(context)
    _validate_input_builder_metadata(config_path)
    namespace = LaunchConfiguration("namespace")
    container_name = LaunchConfiguration("container_name")
    use_existing = PythonExpression(["'", container_name, "' != ''"])
    use_sim_time = _as_bool(context, "use_sim_time")
    eef_pose_topic = "eef_pose"
    socket_kp_pose_topic = "socket_kp_pose"
    pose_delta_command_topic = LaunchConfiguration(
        "pose_delta_command_topic").perform(context)
    tcp_offset_topic = "tcp_offset"
    feedback_reset_topic = "reset_recurrent_state"
    enable_topic = LaunchConfiguration("enable_topic").perform(context)
    enabled_on_start = _as_bool(context, "enabled_on_start")
    tcp_offset = _csv_float_list(context, "tcp_offset", 3)
    # The decoder subtracts this from the decoded control-frame target, so the
    # TCP offset publishes a flange target and zero publishes the TCP target.
    published_target_pose_reference = (
        LaunchConfiguration("published_target_pose_reference").perform(context).strip().lower()
    )
    if published_target_pose_reference not in ("flange", "tcp", "eef"):
        raise ValueError(
            "published_target_pose_reference must be one of: flange, tcp, eef")
    publishes_flange_target = published_target_pose_reference == "flange"

    config_file = Path(config_path)
    test_tmpdir = os.environ.get("TEST_TMPDIR")
    repo_dir = (
        Path(test_tmpdir) / f"triton_repo_{config_file.stem}"
        if test_tmpdir
        else Path(tempfile.gettempdir()) / f"triton_repo_{config_file.stem}"
    )
    try:
        repo = create_triton_model_repo(
            config_file,
            repo_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create Triton model repo from '{config_path}': {exc}"
        ) from exc

    triton_node_name = f"triton_{repo.model_name}".replace("-", "_")
    triton_node = ComposableNode(
        package="isaac_ros_triton",
        plugin="nvidia::isaac_ros::dnn_inference::TritonNode",
        name=triton_node_name,
        namespace=namespace,
        parameters=[
            {
                "model_name": repo.model_name,
                "model_repository_paths": [str(repo_dir)],
                "max_batch_size": 0,
                "num_concurrent_requests": 1,
                "input_tensor_names": repo.input_tensor_names,
                "input_binding_names": repo.input_binding_names,
                "output_tensor_names": repo.output_tensor_names,
                "output_binding_names": repo.output_binding_names,
                "input_tensor_formats": ["nitros_tensor_list_nchw_rgb_f32"],
                "output_tensor_formats": ["nitros_tensor_list_nchw_rgb_f32"],
                "enable_triton_logging": False,
                "log_level": 0,
                "use_sim_time": use_sim_time,
            }
        ],
        remappings=[
            ("tensor_pub", "input_tensors"),
            ("tensor_sub", "output_tensors"),
        ],
    )

    new_container = ComposableNodeContainer(
        name="displayport_task_space_policy_container",
        namespace=namespace,
        package="isaac_ros_deploy_converters",
        executable="component_container",
        output="screen",
        condition=UnlessCondition(use_existing),
    )

    load_nodes = LoadComposableNodes(
        target_container=PythonExpression(
            [
                "'/",
                namespace,
                "/displayport_task_space_policy_container' if '",
                container_name,
                "' == '' else '",
                container_name,
                "'",
            ]
        ),
        composable_node_descriptions=[triton_node],
    )

    pose_source_params = {
        "publish_rate": float(LaunchConfiguration("publish_rate").perform(context)),
        "use_sim_time": use_sim_time,
        "base_frame": LaunchConfiguration("base_frame").perform(context),
        "source_pose_reference": LaunchConfiguration(
            "source_pose_reference").perform(context),
        "tcp_offset": tcp_offset,
        "tcp_offset_topic": tcp_offset_topic,
        "eef_pose_topic": eef_pose_topic,
        "socket_kp_pose_topic": socket_kp_pose_topic,
        "flange_pose_topic": LaunchConfiguration("flange_pose_topic").perform(context),
        "socket_pose_topic": LaunchConfiguration("socket_pose_topic").perform(context),
        "socket_kp_position": _csv_float_list(context, "socket_kp_position", 3),
        "socket_kp_quaternion_xyzw": _csv_float_list(
            context,
            "socket_kp_quaternion_xyzw",
            4,
        ),
        "socket_pose_is_keypoint": _as_bool(context, "socket_pose_is_keypoint"),
        "socket_root_to_keypoint_offset": _csv_float_list(
            context,
            "socket_root_to_keypoint_offset",
            3,
        ),
        "require_socket_pose": _as_bool(context, "require_socket_pose"),
        "socket_pose_timeout_s": float(
            LaunchConfiguration("socket_pose_timeout_s").perform(context)
        ),
        "enable_topic": enable_topic,
        "enabled_on_start": enabled_on_start,
    }
    input_builder_params = {
        "config_path": config_path,
        "publish_rate": float(LaunchConfiguration("publish_rate").perform(context)),
        "output_topic": "input_tensors",
        "use_sim_time": use_sim_time,
        "enable_topic": enable_topic,
        "enabled_on_start": enabled_on_start,
        "reset_feedback_on_enable": _as_bool(context, "reset_recurrent_state_on_enable"),
        "feedback_reset_topic": feedback_reset_topic,
        "source_to_topic.eef_pose_pos": eef_pose_topic,
        "source_message_type.eef_pose_pos": "geometry_msgs/msg/PoseStamped",
        "source_to_topic.eef_pose_rot6d": eef_pose_topic,
        "source_message_type.eef_pose_rot6d": "geometry_msgs/msg/PoseStamped",
        "source_to_topic.socket_kp_pose_pos": socket_kp_pose_topic,
        "source_message_type.socket_kp_pose_pos": "geometry_msgs/msg/PoseStamped",
        "source_to_topic.socket_kp_pose_rot6d": socket_kp_pose_topic,
        "source_message_type.socket_kp_pose_rot6d": "geometry_msgs/msg/PoseStamped",
    }
    pose_source = Node(
        package="isaac_ros_deploy_reference_applications",
        executable="displayport_task_space_pose_source.py",
        name="task_space_pose_source",
        namespace=namespace,
        parameters=[pose_source_params],
        output="screen",
    )
    input_builder = Node(
        package="isaac_ros_deploy_converters",
        executable="input_builder_node",
        name="input_builder",
        namespace=namespace,
        parameters=[input_builder_params],
        output="screen",
    )
    pose_delta_command_builder = Node(
        package="isaac_ros_deploy_converters",
        executable="cartesian_pose_delta_command_builder_node",
        name="cartesian_pose_delta_command_builder",
        namespace=namespace,
        parameters=[
            {
                "action_tensor_topic": "output_tensors",
                "observation_tensor_topic": "input_tensors",
                "output_topic": pose_delta_command_topic,
                "action_tensor_name": LaunchConfiguration("action_tensor_name").perform(context),
                "observation_position_tensor_name": "eef_pos",
                "observation_rotation_6d_tensor_name": "eef_rot_6d",
                "observation_frame_id": LaunchConfiguration("base_frame").perform(context),
                "max_policy_delay_s": float(
                    LaunchConfiguration("max_policy_input_output_delay_s").perform(context)
                ),
                "enable_topic": enable_topic,
                "enabled_on_start": enabled_on_start,
                "feedback_reset_topic": feedback_reset_topic,
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )
    pose_delta_decoder = Node(
        package="isaac_ros_deploy_converters",
        executable="cartesian_pose_delta_decoder_node",
        name="cartesian_pose_delta_decoder",
        namespace=namespace,
        parameters=[
            {
                "input_topic": pose_delta_command_topic,
                "output_topic": LaunchConfiguration("target_pose_topic").perform(context),
                "action_scale": float(LaunchConfiguration("action_blend_ratio").perform(context)),
                "output_tcp_offset": tcp_offset if publishes_flange_target else [0.0, 0.0, 0.0],
                "output_tcp_offset_topic": tcp_offset_topic if publishes_flange_target else "",
                "output_frame_id": LaunchConfiguration("base_frame").perform(context),
                "enable_topic": enable_topic,
                "enabled_on_start": enabled_on_start,
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )
    policy_nodes = [
        pose_source,
        input_builder,
        pose_delta_command_builder,
        pose_delta_decoder,
    ]

    # Passive observer; only started when it has somewhere to put the data.
    csv_output_path = LaunchConfiguration("csv_output_path").perform(context).strip()
    debug_policy_io_topic = LaunchConfiguration("debug_policy_io_topic").perform(context).strip()
    if csv_output_path or debug_policy_io_topic:
        policy_nodes.append(
            Node(
                package="isaac_ros_deploy_reference_applications",
                executable="task_space_policy_io_logger.py",
                name="task_space_policy_io_logger",
                namespace=namespace,
                parameters=[
                    {
                        "csv_output_path": csv_output_path,
                        "debug_policy_io_topic": debug_policy_io_topic,
                        "input_tensor_topic": "input_tensors",
                        "output_tensor_topic": "output_tensors",
                        "pose_delta_command_topic": pose_delta_command_topic,
                        "target_pose_topic": LaunchConfiguration(
                            "target_pose_topic").perform(context),
                        "action_tensor_name": LaunchConfiguration(
                            "action_tensor_name").perform(context),
                        "use_sim_time": use_sim_time,
                    }
                ],
                output="screen",
            )
        )

    return [new_container, load_nodes, *policy_nodes]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value="",
                description="Absolute path to the task-space LEAPP Deploy YAML.",
            ),
            DeclareLaunchArgument("namespace", default_value="displayport_task_space_policy"),
            DeclareLaunchArgument("container_name", default_value=""),
            DeclareLaunchArgument("publish_rate", default_value="15.0"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("base_frame", default_value="world"),
            DeclareLaunchArgument("tcp_offset", default_value="0.0,0.0,0.15"),
            DeclareLaunchArgument(
                "source_pose_reference",
                default_value="flange",
                description=(
                    "Whether flange_pose_topic carries the flange pose or the "
                    "control-frame (tcp/eef) pose."
                ),
            ),
            DeclareLaunchArgument("flange_pose_topic", default_value=""),
            DeclareLaunchArgument(
                "socket_pose_topic",
                default_value="/displayport_insertion/goal_pose",
            ),
            DeclareLaunchArgument("socket_kp_position", default_value="0.475,0.125,0.060"),
            DeclareLaunchArgument(
                "socket_kp_quaternion_xyzw",
                default_value="-0.5,-0.5,-0.5,0.5",
            ),
            DeclareLaunchArgument(
                "socket_pose_is_keypoint",
                default_value="true",
                description=(
                    "False when socket_pose_topic carries the connector root pose, which "
                    "socket_root_to_keypoint_offset then advances to the insertion keypoint."
                ),
            ),
            DeclareLaunchArgument(
                "socket_root_to_keypoint_offset",
                default_value="0.0375,0.0,0.0",
            ),
            DeclareLaunchArgument("require_socket_pose", default_value="false"),
            DeclareLaunchArgument(
                "socket_pose_timeout_s",
                default_value="0.0",
                description=(
                    "Stop publishing observations when the socket pose is older than this. "
                    "0 disables the check."
                ),
            ),
            DeclareLaunchArgument(
                "enable_topic",
                default_value="",
                description=(
                    "std_msgs/Bool topic gating the whole policy path. Empty leaves it "
                    "always enabled."
                ),
            ),
            DeclareLaunchArgument("enabled_on_start", default_value="true"),
            DeclareLaunchArgument(
                "reset_recurrent_state_on_enable",
                default_value="true",
                description=(
                    "Restore the exported initial recurrent tensors on the enable gate's "
                    "rising edge, so each trial starts from the state the policy was "
                    "trained from."
                ),
            ),
            DeclareLaunchArgument("target_pose_topic", default_value="target_pose"),
            DeclareLaunchArgument(
                "pose_delta_command_topic",
                default_value="pose_delta_command",
                description=(
                    "Topic carrying the raw policy action anchored to its "
                    "observation, between the command builder and the decoder."
                ),
            ),
            DeclareLaunchArgument(
                "published_target_pose_reference",
                default_value="flange",
                description=(
                    "Frame of the published target pose. 'flange' subtracts the TCP offset "
                    "from the decoded control-frame target."
                ),
            ),
            DeclareLaunchArgument(
                "max_policy_input_output_delay_s",
                default_value="0.0",
                description=(
                    "Drop a policy action whose observation is older than this. "
                    "0 disables the check."
                ),
            ),
            DeclareLaunchArgument(
                "action_blend_ratio",
                default_value="0.25",
                description=(
                    "Scales the policy's raw Cartesian delta before it is applied to the "
                    "observation pose: delta_applied = action_blend_ratio * delta_policy. "
                    "Forwarded to the decoder node's 'action_scale' parameter."
                ),
            ),
            DeclareLaunchArgument("action_tensor_name", default_value="arm_action"),
            DeclareLaunchArgument(
                "csv_output_path",
                default_value="",
                description=(
                    "Write one CSV row per policy step here. Empty leaves the logger off."
                ),
            ),
            DeclareLaunchArgument(
                "debug_policy_io_topic",
                default_value="",
                description=(
                    "Publish each step's model inputs and outputs as a Float64MultiArray. "
                    "Empty leaves the logger off."
                ),
            ),
            OpaqueFunction(function=_launch),
        ]
    )

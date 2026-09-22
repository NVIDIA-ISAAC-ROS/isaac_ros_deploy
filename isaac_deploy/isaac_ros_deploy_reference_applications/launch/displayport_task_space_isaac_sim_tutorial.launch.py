#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch DisplayPort task-space policy plus pick/place workflow for Isaac Sim.

This launch file assumes the DisplayPort Isaac Sim launcher has already opened
and started playing the pickup fixture scene. Isaac Sim publishes:

* /clock
* /rizon_joint_states
* /displayport_insertion/sim_flange_pose

It starts the manipulation workflow/orchestrator and the task-space Deploy
policy.  The Isaac Sim side consumes the published Cartesian target pose.

The sim flange pose is the measured/control-frame boundary with Isaac Sim.
The policy observation contract remains eef/socket only; the task-space pose
source derives eef/socket PoseStamped sources, InputBuilderNode builds the model
input tensors, and the generic Cartesian pose-delta decoder converts policy
actions back into flange targets for the sim scene.
"""

import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


_CUMOTION_DESCRIPTION_PKG = "isaac_ros_cumotion_robot_description"
_SIM_ROBOT_DESCRIPTION_URDF_PATH = "/tmp/flexiv_rizon4s_grav_nominal_sim.urdf"
_SIM_INITIAL_JOINT_POSITIONS = {
    "joint1": "0.0",
    "joint2": "0.0",
    "joint3": "0.0",
    "joint4": "0.0",
    "joint5": "0.0",
    "joint6": "0.0",
    "joint7": "0.0",
}
_HELPER_FRAMES = {
    "grasp_frame": ("grasp_joint", "gripper_frame", "0 0 0.15", "0 0 0"),
    "insertion_frame": ("insertion_joint", "gripper_frame", "0 0 0.26", "0 3.14 0"),
}


def _include(package_name, relative_path, launch_arguments):
    launch_file = os.path.join(
        get_package_share_directory(package_name),
        *relative_path,
    )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments=launch_arguments.items(),
    )


def _write_sim_grav_urdf(source_path, output_path):
    tree = ET.parse(source_path)
    root = tree.getroot()
    root.set("name", "rizon_grav")

    for ros2_control in root.findall("ros2_control"):
        root.remove(ros2_control)
    root.append(_make_sim_ros2_control(root))

    for frame_name, (joint_name, parent_name, xyz, rpy) in _HELPER_FRAMES.items():
        if root.find(f"./link[@name='{frame_name}']") is None:
            root.append(_make_inertial_link(frame_name))
        if root.find(f"./joint[@name='{joint_name}']") is None:
            root.append(_make_fixed_joint(joint_name, parent_name, frame_name, xyz, rpy))

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode")


def _make_inertial_link(name):
    link = ET.Element("link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", {"value": "0.01"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.0001",
            "ixy": "0.0",
            "ixz": "0.0",
            "iyy": "0.0001",
            "iyz": "0.0",
            "izz": "0.0001",
        },
    )
    ET.SubElement(inertial, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
    return link


def _make_fixed_joint(name, parent_name, child_name, xyz, rpy):
    joint = ET.Element("joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "origin", {"rpy": rpy, "xyz": xyz})
    ET.SubElement(joint, "parent", {"link": parent_name})
    ET.SubElement(joint, "child", {"link": child_name})
    return joint


def _add_param(parent, name, value):
    ET.SubElement(parent, "param", {"name": name}).text = value


def _joint_command_limits(root, joint_name):
    joint = root.find(f"./joint[@name='{joint_name}']")
    if joint is None:
        raise RuntimeError(f"Missing joint {joint_name} in source robot description")
    limit = joint.find("limit")
    if limit is None:
        raise RuntimeError(f"Missing joint limits for {joint_name} in source robot description")

    lower = limit.get("lower")
    upper = limit.get("upper")
    velocity = limit.get("velocity")
    if lower is None or upper is None or velocity is None:
        raise RuntimeError(
            f"Joint {joint_name} must define lower, upper, and velocity limits"
        )
    return lower, upper, f"{-float(velocity):g}", velocity


def _make_sim_ros2_control(root):
    ros2_control = ET.Element(
        "ros2_control", {"name": "rizon_ros2_control_sim", "type": "system"}
    )
    hardware = ET.SubElement(ros2_control, "hardware")
    ET.SubElement(hardware, "plugin").text = (
        "isaac_ros_deploy_ros2_control/TopicBasedSystemInterface"
    )
    _add_param(hardware, "joint_commands_topic", "/rizon_arm_command")
    _add_param(hardware, "joint_states_topic", "/rizon_joint_states")

    for joint_name in _SIM_INITIAL_JOINT_POSITIONS:
        position_min, position_max, velocity_min, velocity_max = _joint_command_limits(
            root, joint_name
        )
        joint = ET.SubElement(ros2_control, "joint", {"name": joint_name})
        command = ET.SubElement(joint, "command_interface", {"name": "position"})
        _add_param(command, "min", position_min)
        _add_param(command, "max", position_max)
        command = ET.SubElement(joint, "command_interface", {"name": "velocity"})
        _add_param(command, "min", velocity_min)
        _add_param(command, "max", velocity_max)
        state = ET.SubElement(joint, "state_interface", {"name": "position"})
        _add_param(state, "initial_value", _SIM_INITIAL_JOINT_POSITIONS[joint_name])
        ET.SubElement(joint, "state_interface", {"name": "velocity"})
        ET.SubElement(joint, "state_interface", {"name": "effort"})

    return ros2_control


def _prepare_cumotion_descriptions(context, *args, **kwargs):
    del context, args, kwargs

    cumotion_desc_dir = get_package_share_directory(_CUMOTION_DESCRIPTION_PKG)
    _write_sim_grav_urdf(
        os.path.join(cumotion_desc_dir, "urdf", "flexiv_rizon4s_grav.urdf"),
        _SIM_ROBOT_DESCRIPTION_URDF_PATH,
    )

    return []


def generate_launch_description():
    workflow_config = LaunchConfiguration("workflow_config")
    default_workflow_config = os.path.join(
        get_package_share_directory("isaac_ros_manipulation_bringup"),
        "params",
        "flexiv_rizon4s_grav_displayport_task_space_isaac_sim_tutorial.yaml",
    )

    workflow_launch = _include(
        "isaac_ros_manipulation_bringup",
        ("launch", "workflows.launch.py"),
        {
            "manipulator_workflow_config": workflow_config,
        },
    )

    policy_launch = _include(
        "isaac_ros_deploy_reference_applications",
        ("launch", "displayport_task_space_policy.launch.py"),
        {
            "config_path": LaunchConfiguration("config_path"),
            "namespace": LaunchConfiguration("namespace"),
            "publish_rate": LaunchConfiguration("publish_rate"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "base_frame": LaunchConfiguration("base_frame"),
            "flange_pose_topic": LaunchConfiguration("flange_pose_topic"),
            "tcp_offset": LaunchConfiguration("tcp_offset"),
            "socket_pose_topic": LaunchConfiguration("socket_pose_topic"),
            "socket_kp_position": LaunchConfiguration("socket_kp_position"),
            "socket_kp_quaternion_xyzw": LaunchConfiguration(
                "socket_kp_quaternion_xyzw"
            ),
            "target_pose_topic": LaunchConfiguration("target_pose_topic"),
            "action_blend_ratio": LaunchConfiguration("action_blend_ratio"),
            "require_socket_pose": LaunchConfiguration("require_socket_pose"),
        },
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "workflow_config",
                default_value=default_workflow_config,
                description="Runtime manipulation workflow config.",
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value="",
                description="Task-space LEAPP YAML path visible inside the ROS container.",
            ),
            DeclareLaunchArgument("namespace", default_value="displayport_task_space_policy"),
            DeclareLaunchArgument("publish_rate", default_value="30.0"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("base_frame", default_value="world"),
            DeclareLaunchArgument(
                "flange_pose_topic",
                default_value="/displayport_insertion/sim_flange_pose",
            ),
            DeclareLaunchArgument("tcp_offset", default_value="0.0,0.0,0.15"),
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
                "target_pose_topic",
                default_value="/displayport_task_space_policy/target_pose",
            ),
            DeclareLaunchArgument("action_blend_ratio", default_value="0.5"),
            DeclareLaunchArgument("require_socket_pose", default_value="true"),
            OpaqueFunction(function=_prepare_cumotion_descriptions),
            workflow_launch,
            policy_launch,
        ]
    )

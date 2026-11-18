// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "isaac_ros_deploy_converters/nodes/joint_command_to_joint_state_node.hpp"

#include <functional>

#include "rclcpp_components/register_node_macro.hpp"

namespace isaac_ros_deploy_converters
{

JointCommandToJointStateNode::JointCommandToJointStateNode(const rclcpp::NodeOptions & options)
: Node("joint_command_to_joint_state_node", options)
{
  subscription_ = create_subscription<isaac_ros_deploy_interfaces::msg::JointCommand>(
    "~/input", 1,
    std::bind(&JointCommandToJointStateNode::callback, this, std::placeholders::_1));

  publisher_ = create_publisher<sensor_msgs::msg::JointState>("~/output", 1);

  RCLCPP_INFO(get_logger(), "JointCommandToJointStateNode initialized");
}

void JointCommandToJointStateNode::callback(
  const isaac_ros_deploy_interfaces::msg::JointCommand::SharedPtr msg)
{
  sensor_msgs::msg::JointState output;
  output.header = msg->header;
  output.name = msg->names;
  output.position = msg->position;
  output.velocity = msg->velocity;
  output.effort = msg->effort;

  publisher_->publish(output);
}

}  // namespace isaac_ros_deploy_converters

RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_deploy_converters::JointCommandToJointStateNode)

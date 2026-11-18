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

#pragma once

#include "isaac_ros_deploy_interfaces/msg/joint_command.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace isaac_ros_deploy_converters
{

/// Converts JointCommand messages to JointState messages for visualization.
///
/// This enables visualizing commanded joint positions using robot_state_publisher
/// with a TF prefix, creating a "ghost" robot showing the commanded pose.
///
/// Subscribes to: ~/input (isaac_ros_deploy_interfaces/JointCommand)
/// Publishes to: ~/output (sensor_msgs/JointState)
class JointCommandToJointStateNode : public rclcpp::Node
{
public:
  explicit JointCommandToJointStateNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void callback(const isaac_ros_deploy_interfaces::msg::JointCommand::SharedPtr msg);

  rclcpp::Subscription<isaac_ros_deploy_interfaces::msg::JointCommand>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
};

}  // namespace isaac_ros_deploy_converters

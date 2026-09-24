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

#include <atomic>
#include <array>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "isaac_ros_deploy_interfaces/msg/cartesian_pose_delta_command.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

namespace isaac_ros_deploy_converters
{

/// Converts an anchored Cartesian pose-delta command into an absolute target PoseStamped.
///
/// ``output_tcp_offset`` is subtracted from the decoded control-frame target, so
/// setting it to the policy's TCP offset publishes a flange target and leaving it
/// at zero publishes the control-frame target itself. On hardware the offset is
/// measured at runtime rather than configured, so ``tcp_offset_topic`` lets the
/// pose source hand over the value it built the observation with.
class CartesianPoseDeltaDecoderNode : public rclcpp::Node
{
public:
  explicit CartesianPoseDeltaDecoderNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using Vec3 = std::array<double, 3>;
  using Quat = std::array<double, 4>;

  void command_callback(
    const isaac_ros_deploy_interfaces::msg::CartesianPoseDeltaCommand::SharedPtr command);

  void tcp_offset_callback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);

  /// Handle a change of the enable gate.
  void enable_callback(const std_msgs::msg::Bool::SharedPtr msg);

  double action_scale_{1.0};
  Vec3 output_tcp_offset_{0.0, 0.0, 0.0};
  std::string output_frame_id_;
  size_t published_count_{0};
  std::atomic_bool enabled_{true};
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher_;
  rclcpp::Subscription<isaac_ros_deploy_interfaces::msg::CartesianPoseDeltaCommand>::SharedPtr
    subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr tcp_offset_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_sub_;
};

}  // namespace isaac_ros_deploy_converters

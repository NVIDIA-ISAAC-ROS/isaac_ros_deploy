// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_publisher.hpp>

#include "isaac_ros_deploy_interfaces/msg/joint_command.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

/// Broadcaster that publishes joint commands as isaac_ros_deploy_interfaces/JointCommand.
///
/// Reads position, velocity, effort, kp, and kd command values from a chainable
/// controller's reference interfaces (exposed as state interfaces) and publishes
/// them on a configurable topic.
///
/// Parameters:
///   - joints: list of joint names to broadcast
///   - topic_name: output topic (default: "joint_commands" → resolves to /joint_commands)
///   - command_prefix: prefix for state interface names (default: "safety_controller")
///   - command_suffix: suffix for state interface names (default: "_raw")
///
/// With defaults, reads state interfaces named:
///   safety_controller/<joint>/position_raw
class JointCommandBroadcaster : public controller_interface::ControllerInterface
{
public:
  JointCommandBroadcaster();

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  // Parameters.
  std::vector<std::string> joint_names_;
  std::string topic_name_;
  std::string command_prefix_;
  std::string command_suffix_;

  // Cached state interface indices per command type.
  std::vector<size_t> position_indices_;
  std::vector<size_t> velocity_indices_;
  std::vector<size_t> effort_indices_;
  std::vector<size_t> kp_indices_;
  std::vector<size_t> kd_indices_;

  // Real-time safe publisher.
  using MsgType = isaac_ros_deploy_interfaces::msg::JointCommand;
  using RealtimePublisher = realtime_tools::RealtimePublisher<MsgType>;
  std::shared_ptr<RealtimePublisher> realtime_publisher_;
  rclcpp::Publisher<MsgType>::SharedPtr publisher_;
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

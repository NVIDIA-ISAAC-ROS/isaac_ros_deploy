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

#include "isaac_ros_deploy_ros2_control/controllers/joint_command_broadcaster.hpp"

#include <limits>

#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

JointCommandBroadcaster::JointCommandBroadcaster()
: controller_interface::ControllerInterface()
{
}

controller_interface::CallbackReturn JointCommandBroadcaster::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>());
    auto_declare<std::string>("topic_name", "joint_commands");
    auto_declare<std::string>("command_prefix", "safety_controller");
    auto_declare<std::string>("command_suffix", "_raw");
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn JointCommandBroadcaster::on_configure(
  const rclcpp_lifecycle::State &)
{
  joint_names_ = get_node()->get_parameter("joints").as_string_array();
  topic_name_ = get_node()->get_parameter("topic_name").as_string();
  command_prefix_ = get_node()->get_parameter("command_prefix").as_string();
  command_suffix_ = get_node()->get_parameter("command_suffix").as_string();

  if (joint_names_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joints specified");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Create publisher and real-time wrapper.
  publisher_ = get_node()->create_publisher<MsgType>(
    topic_name_, rclcpp::SystemDefaultsQoS());
  realtime_publisher_ = std::make_shared<RealtimePublisher>(publisher_);

  // Pre-populate the static fields of the message.
  auto & msg = realtime_publisher_->msg_;
  msg.names = joint_names_;
  const auto n = joint_names_.size();
  msg.position.resize(n, 0.0);
  msg.velocity.resize(n, 0.0);
  msg.effort.resize(n, 0.0);
  msg.kp.resize(n, 0.0);
  msg.kd.resize(n, 0.0);

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Configured JointCommandBroadcaster with %zu joints, topic: %s, "
    "reading from: %s/<joint>/{position,velocity,effort,kp,kd}%s",
    joint_names_.size(), topic_name_.c_str(),
    command_prefix_.c_str(), command_suffix_.c_str());

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
JointCommandBroadcaster::command_interface_configuration() const
{
  // Broadcaster only reads — no command interfaces needed.
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::NONE;
  return config;
}

controller_interface::InterfaceConfiguration
JointCommandBroadcaster::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto & joint_name : joint_names_) {
    const auto prefix = command_prefix_ + "/" + joint_name + "/";
    config.names.push_back(prefix + "position" + command_suffix_);
    config.names.push_back(prefix + "velocity" + command_suffix_);
    config.names.push_back(prefix + "effort" + command_suffix_);
    config.names.push_back(prefix + "kp" + command_suffix_);
    config.names.push_back(prefix + "kd" + command_suffix_);
  }

  return config;
}

controller_interface::CallbackReturn JointCommandBroadcaster::on_activate(
  const rclcpp_lifecycle::State &)
{
  // Build expected interface names per type and find their indices.
  std::vector<std::string> pos_names, vel_names, eff_names, kp_names, kd_names;
  pos_names.reserve(joint_names_.size());
  vel_names.reserve(joint_names_.size());
  eff_names.reserve(joint_names_.size());
  kp_names.reserve(joint_names_.size());
  kd_names.reserve(joint_names_.size());

  for (const auto & joint_name : joint_names_) {
    const auto prefix = command_prefix_ + "/" + joint_name + "/";
    pos_names.push_back(prefix + "position" + command_suffix_);
    vel_names.push_back(prefix + "velocity" + command_suffix_);
    eff_names.push_back(prefix + "effort" + command_suffix_);
    kp_names.push_back(prefix + "kp" + command_suffix_);
    kd_names.push_back(prefix + "kd" + command_suffix_);
  }

  auto pos_idx = utils::find_state_interface_indices(pos_names, state_interfaces_);
  auto vel_idx = utils::find_state_interface_indices(vel_names, state_interfaces_);
  auto eff_idx = utils::find_state_interface_indices(eff_names, state_interfaces_);
  auto kp_idx = utils::find_state_interface_indices(kp_names, state_interfaces_);
  auto kd_idx = utils::find_state_interface_indices(kd_names, state_interfaces_);

  if (!pos_idx || !vel_idx || !eff_idx || !kp_idx || !kd_idx) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to find command state interfaces. "
      "Ensure the chainable controller '%s' is active and exports reference interfaces.",
      command_prefix_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  position_indices_ = std::move(pos_idx.value());
  velocity_indices_ = std::move(vel_idx.value());
  effort_indices_ = std::move(eff_idx.value());
  kp_indices_ = std::move(kp_idx.value());
  kd_indices_ = std::move(kd_idx.value());

  RCLCPP_INFO(
    get_node()->get_logger(),
    "JointCommandBroadcaster activated — publishing %zu joints on %s",
    joint_names_.size(), topic_name_.c_str());

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn JointCommandBroadcaster::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  position_indices_.clear();
  velocity_indices_.clear();
  effort_indices_.clear();
  kp_indices_.clear();
  kd_indices_.clear();

  RCLCPP_INFO(get_node()->get_logger(), "JointCommandBroadcaster deactivated");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type JointCommandBroadcaster::update(
  const rclcpp::Time & time, const rclcpp::Duration &)
{
  if (realtime_publisher_ && realtime_publisher_->trylock()) {
    auto & msg = realtime_publisher_->msg_;
    msg.header.stamp = time;

    // Use NaN (rather than 0.0) on a read failure so training data makes
    // the gap visible instead of silently encoding a spurious
    // "commanded 0 to all joints" frame.
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      msg.position[i] =
        state_interfaces_[position_indices_[i]].get_optional<double>().value_or(kNaN);
      msg.velocity[i] =
        state_interfaces_[velocity_indices_[i]].get_optional<double>().value_or(kNaN);
      msg.effort[i] =
        state_interfaces_[effort_indices_[i]].get_optional<double>().value_or(kNaN);
      msg.kp[i] =
        state_interfaces_[kp_indices_[i]].get_optional<double>().value_or(kNaN);
      msg.kd[i] =
        state_interfaces_[kd_indices_[i]].get_optional<double>().value_or(kNaN);
    }

    realtime_publisher_->unlockAndPublish();
  }

  return controller_interface::return_type::OK;
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::JointCommandBroadcaster,
  controller_interface::ControllerInterface)

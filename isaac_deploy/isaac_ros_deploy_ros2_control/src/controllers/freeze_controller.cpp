// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/controllers/freeze_controller.hpp"

#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

FreezeController::FreezeController()
: controller_interface::ControllerInterface()
{
}

controller_interface::CallbackReturn FreezeController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>());
    auto_declare<double>("kp", 1.0);  // Stiffness gain
    auto_declare<double>("kd", 0.1);   // Damping gain
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FreezeController::on_configure(
  const rclcpp_lifecycle::State &)
{
  joint_names_ = get_node()->get_parameter("joints").as_string_array();
  kp_ = get_node()->get_parameter("kp").as_double();
  kd_ = get_node()->get_parameter("kd").as_double();

  if (joint_names_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joints specified");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Pre-allocate frozen positions vector.
  frozen_positions_.resize(joint_names_.size(), 0.0);

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Configured FreezeController with %zu joints, kp=%.2f, kd=%.2f",
    joint_names_.size(), kp_, kd_);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
FreezeController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Claim all interfaces needed for impedance control:
  // effort = effort_ff + kp * (pos_cmd - pos) + kd * (vel_cmd - vel)
  for (const auto & joint_name : joint_names_) {
    config.names.push_back(joint_name + "/position");
    config.names.push_back(joint_name + "/velocity");
    config.names.push_back(joint_name + "/effort");
    config.names.push_back(joint_name + "/kp");
    config.names.push_back(joint_name + "/kd");
  }

  return config;
}

controller_interface::InterfaceConfiguration
FreezeController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto & joint_name : joint_names_) {
    config.names.push_back(joint_name + "/position");
  }

  return config;
}

controller_interface::CallbackReturn FreezeController::on_activate(
  const rclcpp_lifecycle::State &)
{
  // Build interface name lists.
  std::vector<std::string> position_names, velocity_names, effort_names, kp_names, kd_names;
  position_names.reserve(joint_names_.size());
  velocity_names.reserve(joint_names_.size());
  effort_names.reserve(joint_names_.size());
  kp_names.reserve(joint_names_.size());
  kd_names.reserve(joint_names_.size());

  for (const auto & name : joint_names_) {
    position_names.push_back(name + "/position");
    velocity_names.push_back(name + "/velocity");
    effort_names.push_back(name + "/effort");
    kp_names.push_back(name + "/kp");
    kd_names.push_back(name + "/kd");
  }

  // Find state interface indices (position only needed for reading).
  auto state_indices = utils::find_state_interface_indices(position_names, state_interfaces_);
  if (!state_indices.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find joint position state interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  position_state_indices_ = std::move(state_indices.value());

  // Find command interface indices for all impedance control interfaces.
  auto pos_cmd_indices = utils::find_command_interface_indices(position_names, command_interfaces_);
  if (!pos_cmd_indices.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find position command interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  position_command_indices_ = std::move(pos_cmd_indices.value());

  auto vel_cmd_indices = utils::find_command_interface_indices(velocity_names, command_interfaces_);
  if (!vel_cmd_indices.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find velocity command interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  velocity_command_indices_ = std::move(vel_cmd_indices.value());

  auto eff_cmd_indices = utils::find_command_interface_indices(effort_names, command_interfaces_);
  if (!eff_cmd_indices.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find effort command interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  effort_command_indices_ = std::move(eff_cmd_indices.value());

  auto kp_cmd_indices = utils::find_command_interface_indices(kp_names, command_interfaces_);
  if (!kp_cmd_indices.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find kp command interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  kp_command_indices_ = std::move(kp_cmd_indices.value());

  auto kd_cmd_indices = utils::find_command_interface_indices(kd_names, command_interfaces_);
  if (!kd_cmd_indices.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find kd command interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  kd_command_indices_ = std::move(kd_cmd_indices.value());

  // Capture current positions as frozen positions.
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    auto value_opt = state_interfaces_[position_state_indices_[i]].get_optional<double>();
    if (!value_opt.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to get position for joint '%s'",
        joint_names_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    frozen_positions_[i] = value_opt.value();
    RCLCPP_DEBUG(
      get_node()->get_logger(),
      "Freezing joint '%s' at position: %f",
      joint_names_[i].c_str(), frozen_positions_[i]);
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "FreezeController activated - holding %zu joints using impedance control (kp=%.2f, kd=%.2f)",
    joint_names_.size(), kp_, kd_);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FreezeController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  position_state_indices_.clear();
  position_command_indices_.clear();
  velocity_command_indices_.clear();
  effort_command_indices_.clear();
  kp_command_indices_.clear();
  kd_command_indices_.clear();

  RCLCPP_INFO(get_node()->get_logger(), "FreezeController deactivated");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type FreezeController::update(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  // Write impedance control commands: hold at frozen position with zero velocity
  // and no feed-forward.
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    (void)command_interfaces_[position_command_indices_[i]].set_value(frozen_positions_[i]);
    (void)command_interfaces_[velocity_command_indices_[i]].set_value(0.0);
    (void)command_interfaces_[effort_command_indices_[i]].set_value(0.0);
    (void)command_interfaces_[kp_command_indices_[i]].set_value(kp_);
    (void)command_interfaces_[kd_command_indices_[i]].set_value(kd_);
  }

  return controller_interface::return_type::OK;
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::FreezeController,
  controller_interface::ControllerInterface)

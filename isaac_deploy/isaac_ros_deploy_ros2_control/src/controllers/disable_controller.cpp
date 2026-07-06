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

#include "isaac_ros_deploy_ros2_control/controllers/disable_controller.hpp"

#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

DisableController::DisableController()
: controller_interface::ControllerInterface()
{
}

controller_interface::CallbackReturn DisableController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn DisableController::on_configure(
  const rclcpp_lifecycle::State &)
{
  joint_names_ = get_node()->get_parameter("joints").as_string_array();

  if (joint_names_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joints specified");
    return controller_interface::CallbackReturn::ERROR;
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Configured DisableController with %zu joints (kp=0, kd=0)",
    joint_names_.size());

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
DisableController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Claim all interfaces needed for impedance control (to set them to zero).
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
DisableController::state_interface_configuration() const
{
  // No state interfaces needed - we just send zeros.
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::NONE;
  return config;
}

controller_interface::CallbackReturn DisableController::on_activate(
  const rclcpp_lifecycle::State &)
{
  // Helper to find command interface indices with error logging.
  auto find_indices = [this](const std::string & suffix,
    std::vector<size_t> & out_indices) -> bool {
      std::vector<std::string> names;
      names.reserve(joint_names_.size());
      for (const auto & joint : joint_names_) {
        names.push_back(joint + "/" + suffix);
      }
      auto indices = utils::find_command_interface_indices(names, command_interfaces_);
      if (!indices.has_value()) {
        RCLCPP_ERROR(
          get_node()->get_logger(), "Failed to find %s command interfaces",
          suffix.c_str());
        return false;
      }
      out_indices = std::move(indices.value());
      return true;
    };

  // Find all command interface indices.
  if (!find_indices("position", position_command_indices_) ||
    !find_indices("velocity", velocity_command_indices_) ||
    !find_indices("effort", effort_command_indices_) ||
    !find_indices("kp", kp_command_indices_) ||
    !find_indices("kd", kd_command_indices_))
  {
    return controller_interface::CallbackReturn::ERROR;
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "DisableController activated - disabling %zu joints (kp=0, kd=0)",
    joint_names_.size());

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn DisableController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  position_command_indices_.clear();
  velocity_command_indices_.clear();
  effort_command_indices_.clear();
  kp_command_indices_.clear();
  kd_command_indices_.clear();

  RCLCPP_INFO(get_node()->get_logger(), "DisableController deactivated");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type DisableController::update(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  // Send zero commands: kp=0, kd=0, position=0, velocity=0, effort=0.
  // With kp=0 and kd=0, the robot will collapse under gravity.
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    (void)command_interfaces_[position_command_indices_[i]].set_value(0.0);
    (void)command_interfaces_[velocity_command_indices_[i]].set_value(0.0);
    (void)command_interfaces_[effort_command_indices_[i]].set_value(0.0);
    (void)command_interfaces_[kp_command_indices_[i]].set_value(0.0);
    (void)command_interfaces_[kd_command_indices_[i]].set_value(0.0);
  }

  return controller_interface::return_type::OK;
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::DisableController,
  controller_interface::ControllerInterface)

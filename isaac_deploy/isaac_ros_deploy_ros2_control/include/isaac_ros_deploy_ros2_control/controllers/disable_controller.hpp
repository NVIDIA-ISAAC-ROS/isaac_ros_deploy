// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

/// Controller that disables all joints by setting kp=0, kd=0.
///
/// This controller sends zero stiffness and zero damping commands,
/// effectively disabling the actuators and letting the robot collapse
/// under gravity. Useful for testing and safety scenarios.
class DisableController : public controller_interface::ControllerInterface
{
public:
  DisableController();

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

  // Cached interface indices for command interfaces.
  std::vector<size_t> position_command_indices_;
  std::vector<size_t> velocity_command_indices_;
  std::vector<size_t> effort_command_indices_;
  std::vector<size_t> kp_command_indices_;
  std::vector<size_t> kd_command_indices_;
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

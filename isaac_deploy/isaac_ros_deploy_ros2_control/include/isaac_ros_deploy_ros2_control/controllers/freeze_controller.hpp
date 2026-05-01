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

/// Impedance-based position-hold controller.
///
/// This controller captures the robot's joint positions at activation time
/// and continuously commands those positions to hold the robot in place.
/// Uses impedance control:
// effort = kp * (pos_cmd - pos) + kd * (0 - vel) + effort_feed_forward
/// where kp and kd are configurable parameters.
class FreezeController : public controller_interface::ControllerInterface
{
public:
  FreezeController();

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
  double kp_{100.0};  // Default stiffness
  double kd_{10.0};   // Default damping

  // Frozen positions captured at activation.
  std::vector<double> frozen_positions_;

  // Cached interface indices for state interfaces.
  std::vector<size_t> position_state_indices_;

  // Cached interface indices for command interfaces.
  std::vector<size_t> position_command_indices_;
  std::vector<size_t> velocity_command_indices_;
  std::vector<size_t> effort_command_indices_;
  std::vector<size_t> kp_command_indices_;
  std::vector<size_t> kd_command_indices_;
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

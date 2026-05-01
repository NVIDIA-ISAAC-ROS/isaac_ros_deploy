// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <controller_interface/chainable_controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>

#include "isaac_deploy_core/core/types.h"
#include "isaac_ros_deploy_ros2_control/safety_controller/safety_controller.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

/// Chainable ROS 2 controller wrapping isaac_deploy_core::SafetyController.
///
/// This controller:
/// - Receives position commands via reference interfaces (from upstream controller)
/// - Reads current positions from joint state interfaces
/// - Uses blend_ratio parameter for safety blending (dynamically adjustable via rqt_reconfigure)
/// - Applies safety constraints based on blend_strategy (interpolation or velocity clamping)
/// - Outputs safe positions to hardware command interfaces
class SafetyController : public controller_interface::ChainableControllerInterface
{
public:
  SafetyController();

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update_and_write_commands(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

  // Chainable controller interface.
  std::vector<hardware_interface::CommandInterface> on_export_reference_interfaces() override;
  controller_interface::return_type update_reference_from_subscribers(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  // Parameters.
  isaac_deploy_core::BlendStrategy blend_strategy_{
    isaac_deploy_core::BlendStrategy::kLinearInterpolate};
  std::vector<std::string> joint_names_;
  double default_kp_{0.0};  // Default gains for uncontrolled joints
  double default_kd_{0.0};
  std::atomic<double> target_blend_ratio_{0.0};
  double current_blend_ratio_{0.0};  // only accessed from RT update thread
  std::atomic<double> max_blend_ratio_speed_{1.0};  // units/second

  // Parameter callback handle
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  // Joint positions captured at activation time
  std::vector<double> activate_positions_;

  // Core controller.
  std::optional<isaac_deploy_core::SafetyController> safety_controller_;

  // Pre-allocated I/O tensors.
  // Inputs: command_positions, current_positions, blend_ratio, dt
  // Outputs: safe_positions
  std::vector<isaac_deploy_core::NamedTensor> inputs_;
  std::vector<isaac_deploy_core::NamedTensor> outputs_;
  std::vector<isaac_deploy_core::TensorSpec> input_specs_;
  std::vector<isaac_deploy_core::TensorSpec> output_specs_;

  // Cached interface indices.
  std::vector<size_t> position_state_indices_;
  std::vector<size_t> position_command_indices_;
  std::vector<size_t> velocity_command_indices_;
  std::vector<size_t> effort_command_indices_;
  std::vector<size_t> kp_command_indices_;
  std::vector<size_t> kd_command_indices_;

  // Reference interface layout indices (within reference_interfaces_).
  // Layout: [pos_0, vel_0, eff_0, kp_0, kd_0, pos_1, ...]
  static constexpr size_t kInterfacesPerJoint = 5;

  // Helper methods.
  bool create_safety_controller();
  void populate_inputs_from_interfaces(const rclcpp::Duration & period, int64_t timestamp_ns);
  void write_outputs_to_interfaces();
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

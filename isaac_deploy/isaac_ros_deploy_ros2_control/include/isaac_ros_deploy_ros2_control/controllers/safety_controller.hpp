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

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <controller_interface/chainable_controller_interface.hpp>
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_inverse_dynamics/inverse_dynamics_solver.hpp"

#include "isaac_deploy_core/core/types.hpp"
#include "isaac_ros_deploy_ros2_control/safety_controller/safety_controller.hpp"
#include "isaac_ros_deploy_ros2_control/utils/gain_utils.hpp"
#include "isaac_ros_deploy_ros2_control/utils/realtime_service_client.hpp"

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
  // Export the same reference-interface buffers as read-only state interfaces
  // so broadcasters (e.g. JointCommandBroadcaster) can observe commanded
  // values without having to claim the reference interfaces themselves.
  std::vector<hardware_interface::StateInterface> on_export_state_interfaces() override;
  controller_interface::return_type update_reference_from_subscribers(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  friend class SafetyControllerTestAccess;

  // Parameters. Pre-configure default is the pass-through strategy; on_configure always
  // overwrites this from the "blend_strategy" parameter (default "interpolate").
  isaac_deploy_core::BlendStrategy blend_strategy_{
    isaac_deploy_core::BlendStrategy::kNoPostProcessing};
  std::vector<std::string> joint_names_;
  std::vector<double> per_joint_kp_;
  std::vector<double> per_joint_kd_;
  // Optional per-joint default ("home") position; empty when not configured, in which
  // case interpolate falls back to the measured position at activation.
  std::vector<double> per_joint_default_position_;
  std::atomic<double> target_blend_ratio_{0.0};
  double current_blend_ratio_{0.0};  // only accessed from RT update thread
  std::atomic<double> max_blend_ratio_speed_{1.0};  // units/second
  bool velocity_threshold_enabled_{false};
  double max_joint_velocity_{0.0};
  double mean_joint_velocity_{0.0};
  std::vector<std::string> excluded_joint_patterns_;
  std::vector<size_t> excluded_joint_indices_;
  std::optional<size_t> joint_velocities_input_index_;
  std::string emergency_controller_;
  double emergency_switch_timeout_s_{2.0};
  std::vector<std::string> configured_emergency_deactivate_controllers_;

  using EmergencySwitchClient =
    utils::RealtimeServiceClient<controller_manager_msgs::srv::SwitchController>;
  EmergencySwitchClient emergency_switch_client_;
  mutable std::mutex emergency_reason_mutex_;
  std::string emergency_reason_;

  // kOverwrite: gravity replaces effort at any blend. kBlend: crossfade gravity (full at
  // blend_ratio 0) into the policy effort.
  enum class GravityMode { kOverwrite, kBlend };
  GravityMode gravity_mode_{GravityMode::kOverwrite};
  std::optional<isaac_ros_inverse_dynamics::InverseDynamicsSolver> id_solver_;
  Eigen::VectorXd gravity_q_;
  Eigen::VectorXd gravity_tau_;
  std::vector<bool> gravity_apply_;

  // Parameter callback handle
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  // Core controller.
  std::optional<isaac_deploy_core::SafetyController> safety_controller_;

  // Pre-allocated I/O tensors.
  // Inputs: command_positions, current_positions, blend_ratio, dt, optional joint_velocities
  // Outputs: safe_positions
  std::vector<isaac_deploy_core::NamedTensor> inputs_;
  std::vector<isaac_deploy_core::NamedTensor> outputs_;
  std::vector<isaac_deploy_core::TensorSpec> input_specs_;
  std::vector<isaac_deploy_core::TensorSpec> output_specs_;

  // Cached interface indices.
  std::vector<size_t> position_state_indices_;
  std::vector<size_t> velocity_state_indices_;
  std::vector<size_t> position_command_indices_;
  std::vector<size_t> velocity_command_indices_;
  std::vector<size_t> effort_command_indices_;
  std::vector<size_t> kp_command_indices_;
  std::vector<size_t> kd_command_indices_;

  // Reference interface layout indices (within reference_interfaces_).
  // Layout: [pos_0, vel_0, eff_0, kp_0, kd_0, pos_1, ...]
  static constexpr size_t kInterfacesPerJoint = 5;

  // Helper methods.
  void load_out_of_domain_detection_params();
  void resolve_excluded_joint_indices();
  bool create_safety_controller();
  void populate_inputs_from_interfaces(const rclcpp::Duration & period, int64_t timestamp_ns);
  void write_outputs_to_interfaces();
  void configure_emergency_switch_client();
  controller_manager_msgs::srv::SwitchController::Request build_emergency_switch_request() const;
  void latch_emergency(const std::string & reason);
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

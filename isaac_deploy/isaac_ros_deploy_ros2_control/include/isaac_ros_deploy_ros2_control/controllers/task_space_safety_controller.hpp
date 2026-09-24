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

#include <array>
#include <atomic>
#include <string>
#include <vector>

#include <controller_interface/chainable_controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

/// Decodes task-space policy deltas and writes blended absolute Cartesian commands.
///
/// The blend reference is assembled from the forwarded policy input tensors, so
/// blending uses the same pose observation that produced the action.
/// The controller intentionally does not implement insertion termination, workspace
/// clipping, max-delta clipping, or action rescaling.
class TaskSpaceSafetyController : public controller_interface::ChainableControllerInterface
{
public:
  TaskSpaceSafetyController();

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

  std::vector<hardware_interface::CommandInterface> on_export_reference_interfaces() override;
  std::vector<hardware_interface::StateInterface> on_export_state_interfaces() override;
  controller_interface::return_type update_reference_from_subscribers(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  static constexpr size_t kPositionSize = 3;
  static constexpr size_t kQuatSize = 4;
  static constexpr size_t kRotation6DSize = 6;
  static constexpr size_t kActionSize = 6;
  static constexpr size_t kObservedPoseReferenceSize = kPositionSize + kRotation6DSize;
  static constexpr size_t kReferenceSize = kActionSize + kObservedPoseReferenceSize;

  std::string arm_action_name_{"arm_action"};
  std::string current_observation_name_{"current_observation"};
  std::vector<std::string> action_element_names_{
    "delta_x", "delta_y", "delta_z",
    "delta_axis_angle_x", "delta_axis_angle_y", "delta_axis_angle_z"};
  std::vector<std::string> state_position_interfaces_;
  std::vector<std::string> state_orientation_interfaces_;
  std::vector<std::string> command_position_interfaces_;
  std::vector<std::string> command_orientation_interfaces_;

  std::vector<size_t> state_position_indices_;
  std::vector<size_t> state_orientation_indices_;
  std::vector<size_t> command_position_indices_;
  std::vector<size_t> command_orientation_indices_;

  std::atomic<double> blend_ratio_{0.0};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  bool resolve_interface_indices();
  bool read_current_pose(
    std::array<double, kPositionSize> & position,
    std::array<double, kQuatSize> & quat) const;
  bool read_action_reference(std::array<double, kActionSize> & action) const;
  bool read_observed_pose_reference(
    std::array<double, kPositionSize> & position,
    std::array<double, kQuatSize> & quat) const;
  void write_command(
    const std::array<double, kPositionSize> & position,
    const std::array<double, kQuatSize> & quat);
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

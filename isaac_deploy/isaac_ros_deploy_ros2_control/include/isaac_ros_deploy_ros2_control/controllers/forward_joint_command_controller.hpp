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

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>

#include "isaac_deploy_core/chunk_sampler.hpp"
#include "isaac_ros_deploy_interfaces/msg/joint_command.hpp"
#include "isaac_ros_deploy_interfaces/msg/joint_command_trajectory.hpp"
#include "isaac_ros_deploy_ros2_control/utils/gain_utils.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

/// Impedance controller that subscribes to JointCommand messages.
///
/// Writes position, velocity, effort, kp, and kd to hardware command interfaces,
/// enabling impedance control in the hardware interface:
///   effort = effort_ff + kp * (pos_cmd - pos) + kd * (vel_cmd - vel)
///
/// Before first message, holds activation positions with zero vel/effort and default kp/kd.
class ForwardJointCommandController : public controller_interface::ControllerInterface
{
public:
  ForwardJointCommandController();

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
  /// Operating mode — selected at configure time based on which topic parameter is set.
  enum class Mode
  {
    SingleStep,   ///< joint_command_topic non-empty
    JointState,   ///< joint_state_topic non-empty
    Trajectory,   ///< joint_command_trajectory_topic non-empty
  };

  // Parameters.
  std::vector<std::string> joint_names_;
  std::vector<double> per_joint_kp_;  // Per-joint default kp (sized to joint_names_).
  std::vector<double> per_joint_kd_;  // Per-joint default kd (sized to joint_names_).
  // Per-joint hold-posture position override (sized to joint_names_).  Empty
  // when no `default_position.*` override is configured, in which case
  // `write_hold_posture` falls back to the activation-time pose.
  std::vector<double> per_joint_default_position_;
  std::string command_prefix_;
  std::string command_suffix_;

  Mode mode_{Mode::SingleStep};

  bool received_first_message_{false};
  bool received_first_trajectory_{false};

  // Subscriptions — exactly one is active depending on config.
  rclcpp::Subscription<isaac_ros_deploy_interfaces::msg::JointCommand>::SharedPtr
    joint_command_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
  rclcpp::Subscription<isaac_ros_deploy_interfaces::msg::JointCommandTrajectory>::SharedPtr
    trajectory_sub_;
  isaac_ros_deploy_interfaces::msg::JointCommand::SharedPtr latest_msg_;
  isaac_ros_deploy_interfaces::msg::JointCommandTrajectory::SharedPtr latest_traj_;
  std::mutex msg_mutex_;

  // Cached name-to-index mapping for incoming messages.
  std::unordered_map<std::string, size_t> input_name_to_idx_;
  std::vector<std::string> last_input_names_;

  // Joint positions captured at activation, used as default
  std::vector<double> initial_positions_;

  // Cached interface indices for state and command interfaces.
  std::vector<size_t> position_state_indices_;
  std::vector<size_t> position_command_indices_;
  std::vector<size_t> velocity_command_indices_;
  std::vector<size_t> effort_command_indices_;
  std::vector<size_t> kp_command_indices_;
  std::vector<size_t> kd_command_indices_;

  std::string make_command_interface_name(
    const std::string & joint, const std::string & type) const;

  /// Interface name vectors for position, velocity, effort, kp, kd -- one per joint.
  struct InterfaceNames
  {
    std::vector<std::string> position;
    std::vector<std::string> velocity;
    std::vector<std::string> effort;
    std::vector<std::string> kp;
    std::vector<std::string> kd;
  };

  InterfaceNames build_command_interface_names() const;

  /// Single-step / joint-state dispatch path (latched JointCommand snapshot).
  controller_interface::return_type update_single_step();

  /// Trajectory dispatch path (chunked JointCommandTrajectory with linear interp).
  controller_interface::return_type update_trajectory(const rclcpp::Time & time);

  /// Write the activation-time hold posture for joint index `i`
  /// (initial position, zero velocity/effort, per-joint default gains).
  void write_hold_posture(std::size_t i);
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

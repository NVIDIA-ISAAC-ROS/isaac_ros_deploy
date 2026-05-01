// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_deploy_interfaces/msg/joint_command.hpp"
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
  // Parameters.
  std::vector<std::string> joint_names_;
  double default_kp_{0.0};
  double default_kd_{0.0};
  std::string command_prefix_;
  std::string command_suffix_;

  bool received_first_message_{false};

  // Subscriptions — exactly one is active depending on config.
  rclcpp::Subscription<isaac_ros_deploy_interfaces::msg::JointCommand>::SharedPtr joint_command_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
  isaac_ros_deploy_interfaces::msg::JointCommand::SharedPtr latest_msg_;
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
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

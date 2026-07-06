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

#include "isaac_ros_deploy_ros2_control/controllers/forward_joint_command_controller.hpp"

#include <algorithm>
#include <cmath>

#include "isaac_ros_deploy_ros2_control/utils/gain_utils.hpp"
#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

ForwardJointCommandController::ForwardJointCommandController() = default;

controller_interface::CallbackReturn ForwardJointCommandController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>());
    auto_declare<std::string>("joint_command_topic", "");
    auto_declare<std::string>("joint_state_topic", "");
    auto_declare<std::string>("joint_command_trajectory_topic", "");
    auto_declare<std::string>("command_prefix", "");
    auto_declare<std::string>("command_suffix", "");
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ForwardJointCommandController::on_configure(
  const rclcpp_lifecycle::State &)
{
  joint_names_ = get_node()->get_parameter("joints").as_string_array();
  command_prefix_ = get_node()->get_parameter("command_prefix").as_string();
  command_suffix_ = get_node()->get_parameter("command_suffix").as_string();

  if (joint_names_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joints specified");
    return controller_interface::CallbackReturn::ERROR;
  }

  try {
    per_joint_kp_ = utils::resolve_gains_from_params(*get_node(), "kp", joint_names_);
    per_joint_kd_ = utils::resolve_gains_from_params(*get_node(), "kd", joint_names_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to resolve per-joint gains: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  if (per_joint_kp_.size() != joint_names_.size() ||
    per_joint_kd_.size() != joint_names_.size())
  {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Gain vector size mismatch: expected %zu, got kp=%zu, kd=%zu",
      joint_names_.size(), per_joint_kp_.size(), per_joint_kd_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  // Optional `default_position.<pattern>` regex overrides for the hold-posture target.
  // Empty when no override is present; joints matched by no pattern come back NaN and
  // fall back to the activation-time pose (see hold_position below).
  try {
    per_joint_default_position_ = utils::resolve_default_position(*get_node(), joint_names_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to resolve per-joint default_position: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  for (size_t i = 0; i < joint_names_.size(); ++i) {
    if (per_joint_default_position_.empty()) {
      RCLCPP_INFO(
        get_node()->get_logger(), "  %s: kp=%.1f, kd=%.1f",
        joint_names_[i].c_str(), per_joint_kp_[i], per_joint_kd_[i]);
    } else {
      RCLCPP_INFO(
        get_node()->get_logger(), "  %s: kp=%.1f, kd=%.1f, default_pos=%.3f",
        joint_names_[i].c_str(), per_joint_kp_[i], per_joint_kd_[i],
        per_joint_default_position_[i]);
    }
  }

  const auto joint_command_topic = get_node()->get_parameter("joint_command_topic").as_string();
  const auto joint_state_topic = get_node()->get_parameter("joint_state_topic").as_string();
  const auto joint_command_trajectory_topic =
    get_node()->get_parameter("joint_command_trajectory_topic").as_string();

  const int n_set =
    (!joint_command_topic.empty() ? 1 : 0) +
    (!joint_state_topic.empty() ? 1 : 0) +
    (!joint_command_trajectory_topic.empty() ? 1 : 0);
  if (n_set != 1) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Exactly one of 'joint_command_topic', 'joint_state_topic', "
      "'joint_command_trajectory_topic' must be set; %d are set "
      "(joint_command_topic='%s', joint_state_topic='%s', "
      "joint_command_trajectory_topic='%s').",
      n_set,
      joint_command_topic.c_str(),
      joint_state_topic.c_str(),
      joint_command_trajectory_topic.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (!joint_command_trajectory_topic.empty()) {
    mode_ = Mode::Trajectory;
    auto qos = rclcpp::QoS(1).best_effort();
    trajectory_sub_ =
      get_node()->create_subscription<isaac_ros_deploy_interfaces::msg::JointCommandTrajectory>(
        joint_command_trajectory_topic, qos,
      [this](isaac_ros_deploy_interfaces::msg::JointCommandTrajectory::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(msg_mutex_);
        latest_traj_ = msg;
      });
  } else if (!joint_command_topic.empty()) {
    mode_ = Mode::SingleStep;
    joint_command_subscription_ =
      get_node()->create_subscription<isaac_ros_deploy_interfaces::msg::JointCommand>(
        joint_command_topic, 1,
      [this](isaac_ros_deploy_interfaces::msg::JointCommand::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(msg_mutex_);
        latest_msg_ = msg;
      });
  } else {
    mode_ = Mode::JointState;
    joint_state_subscription_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic, 1,
      [this](sensor_msgs::msg::JointState::SharedPtr msg) {
        auto cmd = std::make_shared<isaac_ros_deploy_interfaces::msg::JointCommand>();
        cmd->names = msg->name;
        cmd->position = msg->position;
        cmd->velocity = msg->velocity;
        // kp and kd are intentionally left empty —
        // update() falls back to per_joint_kp_/per_joint_kd_.
        std::lock_guard<std::mutex> lock(msg_mutex_);
        latest_msg_ = cmd;
      });
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Configured ForwardJointCommandController with %zu joints",
    joint_names_.size());

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
ForwardJointCommandController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  const auto names = build_command_interface_names();
  config.names.reserve(joint_names_.size() * 5);
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    config.names.push_back(names.position[i]);
    config.names.push_back(names.velocity[i]);
    config.names.push_back(names.effort[i]);
    config.names.push_back(names.kp[i]);
    config.names.push_back(names.kd[i]);
  }

  return config;
}

controller_interface::InterfaceConfiguration
ForwardJointCommandController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joint_names_) {
    config.names.push_back(joint + "/position");
  }
  return config;
}

controller_interface::CallbackReturn ForwardJointCommandController::on_activate(
  const rclcpp_lifecycle::State &)
{
  const auto names = build_command_interface_names();

  auto pos_cmd = utils::find_command_interface_indices(names.position, command_interfaces_);
  auto vel_cmd = utils::find_command_interface_indices(names.velocity, command_interfaces_);
  auto eff_cmd = utils::find_command_interface_indices(names.effort, command_interfaces_);
  auto kp_cmd = utils::find_command_interface_indices(names.kp, command_interfaces_);
  auto kd_cmd = utils::find_command_interface_indices(names.kd, command_interfaces_);

  if (!pos_cmd || !vel_cmd || !eff_cmd || !kp_cmd || !kd_cmd) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find command interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  position_command_indices_ = std::move(pos_cmd.value());
  velocity_command_indices_ = std::move(vel_cmd.value());
  effort_command_indices_ = std::move(eff_cmd.value());
  kp_command_indices_ = std::move(kp_cmd.value());
  kd_command_indices_ = std::move(kd_cmd.value());

  // Cache state interface indices for correct joint-to-interface mapping.
  std::vector<std::string> position_names;
  position_names.reserve(joint_names_.size());
  for (const auto & name : joint_names_) {
    position_names.push_back(name + "/position");
  }
  const auto pos_state = utils::find_state_interface_indices(position_names, state_interfaces_);
  if (!pos_state.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find joint position state interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  position_state_indices_ = std::move(pos_state.value());

  // Capture joint positions at activation using correct index mapping.
  initial_positions_.resize(joint_names_.size(), 0.0);
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto current_pos =
      state_interfaces_[position_state_indices_[i]].get_optional<double>();
    if (!current_pos.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to read initial position for joint '%s' — cannot activate safely",
        joint_names_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    initial_positions_[i] = current_pos.value();
  }

  received_first_message_ = false;

  RCLCPP_INFO(
    get_node()->get_logger(),
    "ForwardJointCommandController activated with %zu joints",
    joint_names_.size());

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ForwardJointCommandController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  position_state_indices_.clear();
  position_command_indices_.clear();
  velocity_command_indices_.clear();
  effort_command_indices_.clear();
  kp_command_indices_.clear();
  kd_command_indices_.clear();
  initial_positions_.clear();
  received_first_message_ = false;
  received_first_trajectory_ = false;
  input_name_to_idx_.clear();
  last_input_names_.clear();
  {
    std::lock_guard<std::mutex> lock(msg_mutex_);
    latest_msg_.reset();
    latest_traj_.reset();
  }

  RCLCPP_INFO(get_node()->get_logger(), "ForwardJointCommandController deactivated");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type ForwardJointCommandController::update(
  const rclcpp::Time & time, const rclcpp::Duration &)
{
  if (mode_ == Mode::Trajectory) {
    return update_trajectory(time);
  }
  return update_single_step();
}

controller_interface::return_type ForwardJointCommandController::update_single_step()
{
  // Get latest message.
  isaac_ros_deploy_interfaces::msg::JointCommand::SharedPtr msg;
  {
    std::lock_guard<std::mutex> lock(msg_mutex_);
    msg = latest_msg_;
  }

  if (!msg) {
    // No message yet: hold joint positions captured at activation.
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      write_hold_posture(i);
    }
    return controller_interface::return_type::OK;
  }

  if (!received_first_message_) {
    received_first_message_ = true;
    RCLCPP_INFO(get_node()->get_logger(), "Received first JointCommand, switching to tracking");
  }

  // Rebuild name-to-index map if names changed.
  if (msg->names != last_input_names_) {
    input_name_to_idx_.clear();
    for (size_t i = 0; i < msg->names.size(); ++i) {
      input_name_to_idx_[msg->names[i]] = i;
    }
    last_input_names_ = msg->names;
  }

  const bool has_velocity = !msg->velocity.empty();
  const bool has_effort = !msg->effort.empty();
  const bool has_kp = !msg->kp.empty();
  const bool has_kd = !msg->kd.empty();

  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto it = input_name_to_idx_.find(joint_names_[i]);
    if (it == input_name_to_idx_.end() || it->second >= msg->position.size()) {
      // Joint not in message: hold position captured at activation.
      write_hold_posture(i);
      continue;
    }

    const size_t idx = it->second;
    (void)command_interfaces_[position_command_indices_[i]].set_value(msg->position[idx]);
    (void)command_interfaces_[velocity_command_indices_[i]].set_value(
      has_velocity && idx < msg->velocity.size() ? msg->velocity[idx] : 0.0);
    (void)command_interfaces_[effort_command_indices_[i]].set_value(
      has_effort && idx < msg->effort.size() ? msg->effort[idx] : 0.0);
    (void)command_interfaces_[kp_command_indices_[i]].set_value(
      has_kp && idx < msg->kp.size() ? msg->kp[idx] : per_joint_kp_[i]);
    (void)command_interfaces_[kd_command_indices_[i]].set_value(
      has_kd && idx < msg->kd.size() ? msg->kd[idx] : per_joint_kd_[i]);
  }

  return controller_interface::return_type::OK;
}

controller_interface::return_type
ForwardJointCommandController::update_trajectory(const rclcpp::Time & time)
{
  isaac_ros_deploy_interfaces::msg::JointCommandTrajectory::SharedPtr traj;
  {
    std::lock_guard<std::mutex> lock(msg_mutex_);
    traj = latest_traj_;
  }

  if (!traj) {
    // No message yet: hold joint positions captured at activation.
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      write_hold_posture(i);
    }
    return controller_interface::return_type::OK;
  }

  if (!received_first_trajectory_) {
    received_first_trajectory_ = true;
    RCLCPP_INFO(
      get_node()->get_logger(),
      "Received first JointCommandTrajectory, switching to tracking");
  }

  // Rebuild name-to-index map if names changed.
  if (traj->names != last_input_names_) {
    input_name_to_idx_.clear();
    for (size_t i = 0; i < traj->names.size(); ++i) {
      input_name_to_idx_[traj->names[i]] = i;
    }
    last_input_names_ = traj->names;
  }

  // Match `time`'s clock type when interpreting the wire-level header stamp,
  // so `compute_chunk_position`'s internal subtraction doesn't throw on
  // mismatched time sources (e.g. sim time vs system time).
  const auto pos = isaac_deploy_core::compute_chunk_position(
    rclcpp::Time(traj->header.stamp, time.get_clock_type()),
    rclcpp::Duration(traj->step_dt),
    static_cast<std::size_t>(traj->horizon),
    time);

  if (pos.past_end) {
    auto clock = *get_node()->get_clock();
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), clock, 1000,
      "Trajectory exhausted (past step H-1); holding last step.");
  }

  const bool has_velocity = !traj->velocity.empty();
  const bool has_effort = !traj->effort.empty();
  const bool has_kp = !traj->kp.empty();
  const bool has_kd = !traj->kd.empty();
  const std::size_t n_in = traj->names.size();

  auto interp_at = [&](const std::vector<double> & field, std::size_t col) {
      return isaac_deploy_core::interpolate_field(
      std::span<const double>(field.data(), field.size()), n_in, col, pos);
    };

  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto it = input_name_to_idx_.find(joint_names_[i]);
    // Position is the authoritative field — must exist for this joint's column.
    const bool missing =
      it == input_name_to_idx_.end() ||
      it->second >= n_in ||
      traj->position.size() < ((pos.hi * n_in) + it->second + 1);
    if (missing) {
      // Joint not in message (or message has no position row for this step):
      // hold activation posture with default gains.
      write_hold_posture(i);
      continue;
    }
    const std::size_t col = it->second;

    (void)command_interfaces_[position_command_indices_[i]].set_value(
      interp_at(traj->position, col));
    (void)command_interfaces_[velocity_command_indices_[i]].set_value(
      has_velocity ? interp_at(traj->velocity, col) : 0.0);
    (void)command_interfaces_[effort_command_indices_[i]].set_value(
      has_effort ? interp_at(traj->effort, col) : 0.0);
    (void)command_interfaces_[kp_command_indices_[i]].set_value(
      has_kp ? interp_at(traj->kp, col) : per_joint_kp_[i]);
    (void)command_interfaces_[kd_command_indices_[i]].set_value(
      has_kd ? interp_at(traj->kd, col) : per_joint_kd_[i]);
  }

  return controller_interface::return_type::OK;
}

void ForwardJointCommandController::write_hold_posture(std::size_t i)
{
  // Hold at the configured default where finite; otherwise (no default, or this joint
  // matched no pattern -> NaN) hold the pose captured at activation.
  const double hold_position =
    (per_joint_default_position_.empty() || std::isnan(per_joint_default_position_[i])) ?
    initial_positions_[i] : per_joint_default_position_[i];
  (void)command_interfaces_[position_command_indices_[i]].set_value(hold_position);
  (void)command_interfaces_[velocity_command_indices_[i]].set_value(0.0);
  (void)command_interfaces_[effort_command_indices_[i]].set_value(0.0);
  (void)command_interfaces_[kp_command_indices_[i]].set_value(per_joint_kp_[i]);
  (void)command_interfaces_[kd_command_indices_[i]].set_value(per_joint_kd_[i]);
}

ForwardJointCommandController::InterfaceNames
ForwardJointCommandController::build_command_interface_names() const
{
  InterfaceNames names;
  names.position.reserve(joint_names_.size());
  names.velocity.reserve(joint_names_.size());
  names.effort.reserve(joint_names_.size());
  names.kp.reserve(joint_names_.size());
  names.kd.reserve(joint_names_.size());

  for (const auto & joint : joint_names_) {
    names.position.push_back(make_command_interface_name(joint, "position"));
    names.velocity.push_back(make_command_interface_name(joint, "velocity"));
    names.effort.push_back(make_command_interface_name(joint, "effort"));
    names.kp.push_back(make_command_interface_name(joint, "kp"));
    names.kd.push_back(make_command_interface_name(joint, "kd"));
  }

  return names;
}

std::string ForwardJointCommandController::make_command_interface_name(
  const std::string & joint, const std::string & type) const
{
  std::string name;
  if (!command_prefix_.empty()) {
    name = command_prefix_ + "/";
  }
  name += joint + "/" + type;
  if (!command_suffix_.empty()) {
    name += command_suffix_;
  }
  return name;
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::ForwardJointCommandController,
  controller_interface::ControllerInterface)

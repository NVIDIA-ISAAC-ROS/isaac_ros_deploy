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

#include "isaac_ros_deploy_ros2_control/controllers/task_space_safety_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{
namespace
{

std::array<double, 4> normalize_quat_xyzw(const std::array<double, 4> & q)
{
  const double norm = std::sqrt(
    q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm <= 0.0 || !std::isfinite(norm)) {
    return {0.0, 0.0, 0.0, 1.0};
  }
  return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
}

std::array<double, 3> invalid_vector3()
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  return {nan, nan, nan};
}

std::array<double, 4> invalid_quat_xyzw()
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  return {nan, nan, nan, nan};
}

std::array<double, 3> normalize_vector3(const std::array<double, 3> & value)
{
  const double norm = std::sqrt(
    value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
  if (norm <= 1e-12 || !std::isfinite(norm)) {
    return invalid_vector3();
  }
  return {value[0] / norm, value[1] / norm, value[2] / norm};
}

std::array<double, 3> cross_product(
  const std::array<double, 3> & lhs,
  const std::array<double, 3> & rhs)
{
  return {
    lhs[1] * rhs[2] - lhs[2] * rhs[1],
    lhs[2] * rhs[0] - lhs[0] * rhs[2],
    lhs[0] * rhs[1] - lhs[1] * rhs[0],
  };
}

std::array<double, 4> matrix_row_major_to_quat_xyzw(const std::array<double, 9> & m)
{
  const double trace = m[0] + m[4] + m[8];
  if (!std::isfinite(trace)) {
    return invalid_quat_xyzw();
  }

  std::array<double, 4> quat{};
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    if (s <= 1e-12 || !std::isfinite(s)) {
      return invalid_quat_xyzw();
    }
    quat = {
      (m[7] - m[5]) / s,
      (m[2] - m[6]) / s,
      (m[3] - m[1]) / s,
      0.25 * s,
    };
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const double s = std::sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0;
    if (s <= 1e-12 || !std::isfinite(s)) {
      return invalid_quat_xyzw();
    }
    quat = {
      0.25 * s,
      (m[1] + m[3]) / s,
      (m[2] + m[6]) / s,
      (m[7] - m[5]) / s,
    };
  } else if (m[4] > m[8]) {
    const double s = std::sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0;
    if (s <= 1e-12 || !std::isfinite(s)) {
      return invalid_quat_xyzw();
    }
    quat = {
      (m[1] + m[3]) / s,
      0.25 * s,
      (m[5] + m[7]) / s,
      (m[2] - m[6]) / s,
    };
  } else {
    const double s = std::sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0;
    if (s <= 1e-12 || !std::isfinite(s)) {
      return invalid_quat_xyzw();
    }
    quat = {
      (m[2] + m[6]) / s,
      (m[5] + m[7]) / s,
      0.25 * s,
      (m[3] - m[1]) / s,
    };
  }
  return normalize_quat_xyzw(quat);
}

std::array<double, 4> rotation_6d_to_quat_xyzw(const std::array<double, 6> & values)
{
  const auto row0 = normalize_vector3({values[0], values[1], values[2]});
  const std::array<double, 3> raw_row1{values[3], values[4], values[5]};
  const double row1_projection =
    row0[0] * raw_row1[0] + row0[1] * raw_row1[1] + row0[2] * raw_row1[2];
  const auto row1 = normalize_vector3(
        {
          raw_row1[0] - row1_projection * row0[0],
          raw_row1[1] - row1_projection * row0[1],
          raw_row1[2] - row1_projection * row0[2],
    });
  const auto row2 = cross_product(row0, row1);
  return matrix_row_major_to_quat_xyzw(
        {
          row0[0], row0[1], row0[2],
          row1[0], row1[1], row1[2],
          row2[0], row2[1], row2[2],
    });
}

std::array<double, 4> multiply_quat_xyzw(
  const std::array<double, 4> & lhs,
  const std::array<double, 4> & rhs)
{
  const auto a = normalize_quat_xyzw(lhs);
  const auto b = normalize_quat_xyzw(rhs);
  return normalize_quat_xyzw(
        {
          a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
          a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
          a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
          a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    });
}

std::array<double, 4> quat_from_axis_angle(const std::array<double, 3> & axis_angle)
{
  const double angle = std::sqrt(
    axis_angle[0] * axis_angle[0] + axis_angle[1] * axis_angle[1] +
    axis_angle[2] * axis_angle[2]);
  if (angle < 1e-12 || !std::isfinite(angle)) {
    return {0.0, 0.0, 0.0, 1.0};
  }
  const double inv_angle = 1.0 / angle;
  const double half = 0.5 * angle;
  const double sin_half = std::sin(half);
  return normalize_quat_xyzw(
        {
          axis_angle[0] * inv_angle * sin_half,
          axis_angle[1] * inv_angle * sin_half,
          axis_angle[2] * inv_angle * sin_half,
          std::cos(half),
    });
}

std::array<double, 4> slerp_quat_xyzw(
  const std::array<double, 4> & from,
  const std::array<double, 4> & to,
  double ratio)
{
  auto q0 = normalize_quat_xyzw(from);
  auto q1 = normalize_quat_xyzw(to);
  ratio = std::clamp(ratio, 0.0, 1.0);
  double dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
  if (dot < 0.0) {
    dot = -dot;
    for (auto & value : q1) {
      value = -value;
    }
  }
  if (dot > 0.9995) {
    return normalize_quat_xyzw(
          {
            q0[0] + ratio * (q1[0] - q0[0]),
            q0[1] + ratio * (q1[1] - q0[1]),
            q0[2] + ratio * (q1[2] - q0[2]),
            q0[3] + ratio * (q1[3] - q0[3]),
      });
  }
  const double theta_0 = std::acos(std::clamp(dot, -1.0, 1.0));
  const double theta = theta_0 * ratio;
  const double sin_theta = std::sin(theta);
  const double sin_theta_0 = std::sin(theta_0);
  const double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
  const double s1 = sin_theta / sin_theta_0;
  return normalize_quat_xyzw(
        {
          s0 * q0[0] + s1 * q1[0],
          s0 * q0[1] + s1 * q1[1],
          s0 * q0[2] + s1 * q1[2],
          s0 * q0[3] + s1 * q1[3],
    });
}

template<size_t N>
bool all_finite(const std::array<double, N> & values)
{
  return std::all_of(values.begin(), values.end(), [](double value) {
             return std::isfinite(value);
    });
}

std::vector<std::string> default_position_interfaces(const std::string & body)
{
  return {body + "/position.x", body + "/position.y", body + "/position.z"};
}

std::vector<std::string> default_orientation_interfaces(const std::string & body)
{
  return {
    body + "/orientation.x", body + "/orientation.y",
    body + "/orientation.z", body + "/orientation.w"};
}

std::vector<std::string> default_target_position_interfaces(const std::string & body)
{
  return {
    body + "/target_position.x", body + "/target_position.y", body + "/target_position.z"};
}

std::vector<std::string> default_target_orientation_interfaces(const std::string & body)
{
  return {
    body + "/target_orientation.x", body + "/target_orientation.y",
    body + "/target_orientation.z", body + "/target_orientation.w"};
}

}  // namespace

TaskSpaceSafetyController::TaskSpaceSafetyController()
: controller_interface::ChainableControllerInterface()
{
}

controller_interface::CallbackReturn TaskSpaceSafetyController::on_init()
{
  try {
    auto_declare<std::string>("arm_action_name", arm_action_name_);
    auto_declare<std::string>("current_observation_name", current_observation_name_);
    auto_declare<std::vector<std::string>>(
      "action_element_names",
      std::vector<std::string>{
          "delta_x", "delta_y", "delta_z",
          "delta_axis_angle_x", "delta_axis_angle_y", "delta_axis_angle_z"});
    auto_declare<std::string>("body_name", "eef");
    auto_declare<std::vector<std::string>>(
      "state_position_interfaces", std::vector<std::string>());
    auto_declare<std::vector<std::string>>(
      "state_orientation_interfaces", std::vector<std::string>());
    auto_declare<std::vector<std::string>>(
      "command_position_interfaces", std::vector<std::string>());
    auto_declare<std::vector<std::string>>(
      "command_orientation_interfaces", std::vector<std::string>());
    auto_declare<double>("blend_ratio", 0.0);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn TaskSpaceSafetyController::on_configure(
  const rclcpp_lifecycle::State &)
{
  arm_action_name_ = get_node()->get_parameter("arm_action_name").as_string();
  current_observation_name_ = get_node()->get_parameter("current_observation_name").as_string();
  action_element_names_ = get_node()->get_parameter("action_element_names").as_string_array();
  const auto body_name = get_node()->get_parameter("body_name").as_string();
  state_position_interfaces_ =
    get_node()->get_parameter("state_position_interfaces").as_string_array();
  state_orientation_interfaces_ =
    get_node()->get_parameter("state_orientation_interfaces").as_string_array();
  command_position_interfaces_ =
    get_node()->get_parameter("command_position_interfaces").as_string_array();
  command_orientation_interfaces_ =
    get_node()->get_parameter("command_orientation_interfaces").as_string_array();
  if (state_position_interfaces_.empty()) {
    state_position_interfaces_ = default_position_interfaces(body_name);
  }
  if (state_orientation_interfaces_.empty()) {
    state_orientation_interfaces_ = default_orientation_interfaces(body_name);
  }
  if (command_position_interfaces_.empty()) {
    command_position_interfaces_ = default_target_position_interfaces(body_name);
  }
  if (command_orientation_interfaces_.empty()) {
    command_orientation_interfaces_ = default_target_orientation_interfaces(body_name);
  }
  if (
    state_position_interfaces_.size() != kPositionSize ||
    command_position_interfaces_.size() != kPositionSize ||
    state_orientation_interfaces_.size() != kQuatSize ||
    command_orientation_interfaces_.size() != kQuatSize ||
    action_element_names_.size() != kActionSize)
  {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Task-space Cartesian interfaces must be position[3], orientation[4], and action[6]");
    return controller_interface::CallbackReturn::ERROR;
  }

  const double blend_ratio = get_node()->get_parameter("blend_ratio").as_double();
  if (!std::isfinite(blend_ratio) || blend_ratio < 0.0 || blend_ratio > 1.0) {
    RCLCPP_ERROR(get_node()->get_logger(), "blend_ratio must be finite and in [0, 1]");
    return controller_interface::CallbackReturn::ERROR;
  }
  blend_ratio_.store(blend_ratio);

  param_callback_handle_ = get_node()->add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & parameters) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;
      for (const auto & param : parameters) {
        if (param.get_name() == "blend_ratio") {
          const double value = param.as_double();
          if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
            result.successful = false;
            result.reason = "blend_ratio must be finite and in [0, 1]";
            return result;
          }
          blend_ratio_.store(value);
        }
      }
      return result;
    });

  reference_interfaces_.assign(kReferenceSize, std::numeric_limits<double>::quiet_NaN());
  RCLCPP_INFO(
    get_node()->get_logger(),
    "Configured TaskSpaceSafetyController with current_observation blending, blend_ratio=%.3f",
    blend_ratio_.load());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
TaskSpaceSafetyController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.insert(
    config.names.end(), state_position_interfaces_.begin(), state_position_interfaces_.end());
  config.names.insert(
    config.names.end(), state_orientation_interfaces_.begin(), state_orientation_interfaces_.end());
  return config;
}

controller_interface::InterfaceConfiguration
TaskSpaceSafetyController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.insert(
    config.names.end(), command_position_interfaces_.begin(), command_position_interfaces_.end());
  config.names.insert(
    config.names.end(), command_orientation_interfaces_.begin(),
    command_orientation_interfaces_.end());
  return config;
}

std::vector<hardware_interface::CommandInterface>
TaskSpaceSafetyController::on_export_reference_interfaces()
{
  const std::string controller_name = get_node()->get_name();
  const std::array<std::string, kObservedPoseReferenceSize> observed_components{
    "position.x", "position.y", "position.z",
    "rotation_6d.r00", "rotation_6d.r01", "rotation_6d.r02",
    "rotation_6d.r10", "rotation_6d.r11", "rotation_6d.r12"};
  std::vector<hardware_interface::CommandInterface> interfaces;
  interfaces.reserve(kReferenceSize);
  for (size_t i = 0; i < kActionSize; ++i) {
    interfaces.emplace_back(
      controller_name, arm_action_name_ + "/" + action_element_names_[i] + "_raw",
      &reference_interfaces_[i]);
  }
  for (size_t i = 0; i < kObservedPoseReferenceSize; ++i) {
    interfaces.emplace_back(
      controller_name, current_observation_name_ + "/" + observed_components[i],
      &reference_interfaces_[kActionSize + i]);
  }
  return interfaces;
}

std::vector<hardware_interface::StateInterface>
TaskSpaceSafetyController::on_export_state_interfaces()
{
  const std::string controller_name = get_node()->get_name();
  const std::array<std::string, kObservedPoseReferenceSize> observed_components{
    "position.x", "position.y", "position.z",
    "rotation_6d.r00", "rotation_6d.r01", "rotation_6d.r02",
    "rotation_6d.r10", "rotation_6d.r11", "rotation_6d.r12"};
  std::vector<hardware_interface::StateInterface> interfaces;
  interfaces.reserve(kReferenceSize);
  for (size_t i = 0; i < kActionSize; ++i) {
    interfaces.emplace_back(
      controller_name, arm_action_name_ + "/" + action_element_names_[i] + "_raw",
      &reference_interfaces_[i]);
  }
  for (size_t i = 0; i < kObservedPoseReferenceSize; ++i) {
    interfaces.emplace_back(
      controller_name, current_observation_name_ + "/" + observed_components[i],
      &reference_interfaces_[kActionSize + i]);
  }
  return interfaces;
}

controller_interface::return_type TaskSpaceSafetyController::update_reference_from_subscribers(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  return controller_interface::return_type::OK;
}

bool TaskSpaceSafetyController::resolve_interface_indices()
{
  auto resolve_state = [this](const std::vector<std::string> & names, std::vector<size_t> & out) {
      auto indices = utils::find_state_interface_indices(names, state_interfaces_);
      if (!indices.has_value()) {
        return false;
      }
      out = std::move(indices.value());
      return true;
    };
  auto resolve_command = [this](const std::vector<std::string> & names, std::vector<size_t> & out) {
      auto indices = utils::find_command_interface_indices(names, command_interfaces_);
      if (!indices.has_value()) {
        return false;
      }
      out = std::move(indices.value());
      return true;
    };
  return resolve_state(state_position_interfaces_, state_position_indices_) &&
         resolve_state(state_orientation_interfaces_, state_orientation_indices_) &&
         resolve_command(command_position_interfaces_, command_position_indices_) &&
         resolve_command(command_orientation_interfaces_, command_orientation_indices_);
}

controller_interface::CallbackReturn TaskSpaceSafetyController::on_activate(
  const rclcpp_lifecycle::State &)
{
  if (!resolve_interface_indices()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to resolve Cartesian state/command interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  std::fill(
    reference_interfaces_.begin(), reference_interfaces_.end(),
    std::numeric_limits<double>::quiet_NaN());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn TaskSpaceSafetyController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  state_position_indices_.clear();
  state_orientation_indices_.clear();
  command_position_indices_.clear();
  command_orientation_indices_.clear();
  return controller_interface::CallbackReturn::SUCCESS;
}

bool TaskSpaceSafetyController::read_current_pose(
  std::array<double, kPositionSize> & position,
  std::array<double, kQuatSize> & quat) const
{
  for (size_t i = 0; i < kPositionSize; ++i) {
    auto value = state_interfaces_[state_position_indices_[i]].get_optional<double>();
    if (!value.has_value()) {
      return false;
    }
    position[i] = value.value();
  }
  for (size_t i = 0; i < kQuatSize; ++i) {
    auto value = state_interfaces_[state_orientation_indices_[i]].get_optional<double>();
    if (!value.has_value()) {
      return false;
    }
    quat[i] = value.value();
  }
  quat = normalize_quat_xyzw(quat);
  return all_finite(position) && all_finite(quat);
}

bool TaskSpaceSafetyController::read_action_reference(std::array<double, kActionSize> & action)
const
{
  for (size_t i = 0; i < kActionSize; ++i) {
    action[i] = reference_interfaces_[i];
  }
  return all_finite(action);
}

bool TaskSpaceSafetyController::read_observed_pose_reference(
  std::array<double, kPositionSize> & position,
  std::array<double, kQuatSize> & quat) const
{
  for (size_t i = 0; i < kPositionSize; ++i) {
    position[i] = reference_interfaces_[kActionSize + i];
  }
  std::array<double, kRotation6DSize> rotation_6d{};
  for (size_t i = 0; i < kRotation6DSize; ++i) {
    rotation_6d[i] = reference_interfaces_[kActionSize + kPositionSize + i];
  }
  quat = rotation_6d_to_quat_xyzw(rotation_6d);
  return all_finite(position) && all_finite(rotation_6d) && all_finite(quat);
}

void TaskSpaceSafetyController::write_command(
  const std::array<double, kPositionSize> & position,
  const std::array<double, kQuatSize> & quat)
{
  for (size_t i = 0; i < kPositionSize; ++i) {
    (void)command_interfaces_[command_position_indices_[i]].set_value(position[i]);
  }
  const auto normalized_quat = normalize_quat_xyzw(quat);
  for (size_t i = 0; i < kQuatSize; ++i) {
    (void)command_interfaces_[command_orientation_indices_[i]].set_value(normalized_quat[i]);
  }
}

controller_interface::return_type TaskSpaceSafetyController::update_and_write_commands(
  const rclcpp::Time &,
  const rclcpp::Duration &)
{
  std::array<double, kPositionSize> measured_pos{};
  std::array<double, kQuatSize> measured_quat{};
  std::array<double, kPositionSize> observed_pos{};
  std::array<double, kQuatSize> observed_quat{};
  std::array<double, kActionSize> action{};

  const bool measured_pose_valid = read_current_pose(measured_pos, measured_quat);
  if (!measured_pose_valid) {
    return controller_interface::return_type::OK;
  }

  if (!read_observed_pose_reference(observed_pos, observed_quat)) {
    write_command(measured_pos, measured_quat);
    return controller_interface::return_type::OK;
  }
  if (!read_action_reference(action)) {
    write_command(measured_pos, measured_quat);
    return controller_interface::return_type::OK;
  }

  std::array<double, kPositionSize> raw_target_pos{
    observed_pos[0] + action[0],
    observed_pos[1] + action[1],
    observed_pos[2] + action[2],
  };
  const std::array<double, 3> delta_rot{action[3], action[4], action[5]};
  const auto raw_target_quat = multiply_quat_xyzw(
    quat_from_axis_angle(delta_rot), observed_quat);

  if (!all_finite(raw_target_pos) || !all_finite(raw_target_quat)) {
    write_command(measured_pos, measured_quat);
    return controller_interface::return_type::OK;
  }

  const double ratio = blend_ratio_.load();
  std::array<double, kPositionSize> final_pos{
    observed_pos[0] + ratio * (raw_target_pos[0] - observed_pos[0]),
    observed_pos[1] + ratio * (raw_target_pos[1] - observed_pos[1]),
    observed_pos[2] + ratio * (raw_target_pos[2] - observed_pos[2]),
  };
  const auto final_quat = slerp_quat_xyzw(observed_quat, raw_target_quat, ratio);
  write_command(final_pos, final_quat);
  return controller_interface::return_type::OK;
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::TaskSpaceSafetyController,
  controller_interface::ChainableControllerInterface)

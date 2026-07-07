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

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/parameter_value.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace utils
{

/// Apply (pattern, value) pairs to joint names via full-match regex.
/// Patterns are applied in order; later matches override earlier ones.
/// Joints that match no pattern get `unmatched_default` (0.0 by default).
std::vector<double> resolve_gains(
  const std::vector<std::pair<std::string, double>> & patterns,
  const std::vector<std::string> & joint_names,
  double unmatched_default = 0.0);

/// Read gain patterns from a ROS parameter dict, then resolve per-joint gains.
/// Uses parameter overrides to discover sub-keys without prior declaration.
/// Works with both rclcpp::Node and rclcpp_lifecycle::LifecycleNode.
///
/// YAML format:
///   kp:
///     ".*": 100.0
///     ".*_wrist_.*_joint": 20.0
///     "waist_yaw_joint": 250.0
template<typename NodeT>
std::vector<double> resolve_gains_from_params(
  NodeT & node, const std::string & param_name,
  const std::vector<std::string> & joint_names,
  double unmatched_default = 0.0)
{
  const auto & overrides = node.get_node_parameters_interface()->get_parameter_overrides();
  const std::string prefix = param_name + ".";
  std::vector<std::pair<std::string, double>> patterns;
  for (const auto & [key, value] : overrides) {
    if (key.rfind(prefix, 0) != 0) {
      continue;
    }
    const auto pattern = key.substr(prefix.size());
    if (value.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      patterns.emplace_back(
        pattern, value.template get<rclcpp::ParameterType::PARAMETER_DOUBLE>());
    } else if (value.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
      patterns.emplace_back(
        pattern,
        static_cast<double>(
          value.template get<rclcpp::ParameterType::PARAMETER_INTEGER>()));
    } else {
      throw std::runtime_error(
        "Gain '" + key + "' must be a numeric value, got type " +
        std::to_string(static_cast<int>(value.get_type())));
    }
  }
  return resolve_gains(patterns, joint_names, unmatched_default);
}

/// Resolve the `default_position` regex map. NaN for unmatched joints (caller uses
/// activation pose). Shared by safety and forward-joint-command controllers.
template<typename NodeT>
std::vector<double> resolve_default_position(
  NodeT & node, const std::vector<std::string> & joint_names)
{
  const auto & overrides = node.get_node_parameters_interface()->get_parameter_overrides();
  const bool has_default_position = std::any_of(
    overrides.begin(), overrides.end(),
    [](const auto & kv) {return kv.first.rfind("default_position.", 0) == 0;});
  if (!has_default_position) {
    return {};
  }
  return resolve_gains_from_params(
    node, "default_position", joint_names, std::numeric_limits<double>::quiet_NaN());
}

}  // namespace utils
}  // namespace isaac_ros_deploy_ros2_control

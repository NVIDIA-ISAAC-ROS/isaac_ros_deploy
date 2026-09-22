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

#include "isaac_ros_deploy_ros2_control/converters/command_interface_converter.hpp"

#include <mutex>
#include <stdexcept>


namespace isaac_ros_deploy_ros2_control
{

namespace
{

/// Build a command interface name with optional prefix and suffix.
std::string make_interface_name(
  const std::string & joint, const std::string & type,
  const std::string & prefix, const std::string & suffix)
{
  std::string name;
  if (!prefix.empty()) {
    name = prefix + "/";
  }
  name += joint + "/" + type;
  if (!suffix.empty()) {
    name += suffix;
  }
  return name;
}

/// Converter for joint command interfaces, parameterized by interface type.
class JointCommandConverter : public CommandInterfaceConverter
{
public:
  explicit JointCommandConverter(std::string interface_type)
  : interface_type_(std::move(interface_type)) {}

  std::vector<std::string> get_required_command_interfaces(
    const std::vector<std::vector<std::string>> & element_names,
    const std::string & prefix, const std::string & suffix) const override
  {
    if (element_names.empty()) {
      throw std::runtime_error(
              "element_names must not be empty for command converter '" + interface_type_ + "'");
    }
    std::vector<std::string> interfaces;
    for (const auto & name : element_names.back()) {
      interfaces.push_back(make_interface_name(name, interface_type_, prefix, suffix));
    }
    return interfaces;
  }

  isaac_deploy_core::TensorSpec get_tensor_spec(
    const std::vector<std::vector<std::string>> & element_names) const override
  {
    return {.names = element_names};
  }

private:
  std::string interface_type_;
};

/// Converter for flat relative Cartesian pose action command/reference interfaces.
class FlatBodyPoseCommandConverter : public CommandInterfaceConverter
{
public:
  std::vector<std::string> get_required_command_interfaces(
    const std::vector<std::vector<std::string>> & element_names,
    const std::string & prefix, const std::string & suffix) const override
  {
    const std::vector<std::string> default_names{
      "delta_x", "delta_y", "delta_z",
      "delta_axis_angle_x", "delta_axis_angle_y", "delta_axis_angle_z"};
    const auto & names = element_names.empty() || element_names.back().empty() ?
      default_names : element_names.back();
    std::vector<std::string> interfaces;
    interfaces.reserve(names.size());
    for (const auto & name : names) {
      std::string interface_name;
      if (!prefix.empty()) {
        interface_name = prefix + "/";
      }
      interface_name += name + suffix;
      interfaces.push_back(interface_name);
    }
    return interfaces;
  }

  isaac_deploy_core::TensorSpec get_tensor_spec(
    const std::vector<std::vector<std::string>> & element_names) const override
  {
    if (element_names.empty() || element_names.back().empty()) {
      return {.names = {{},
          {"delta_x", "delta_y", "delta_z",
            "delta_axis_angle_x", "delta_axis_angle_y", "delta_axis_angle_z"}}};
    }
    return {.names = element_names};
  }
};

}  // namespace

void initialize_command_interface_converters()
{
  static std::once_flag flag;
  std::call_once(flag, []() {
      auto & registry = CommandInterfaceConverterRegistry::instance();

      registry.register_converter(
        "target/joint/position", []() {
          return std::make_shared<JointCommandConverter>("position");
        });
      registry.register_converter(
        "target/body/pose_relative", []() {
          return std::make_shared<FlatBodyPoseCommandConverter>();
        });
      registry.register_converter(
        "target/body/pose_rel", []() {
          return std::make_shared<FlatBodyPoseCommandConverter>();
        });
      registry.register_converter(
        "target/body/pose_delta", []() {
          return std::make_shared<FlatBodyPoseCommandConverter>();
        });
      registry.register_converter(
        "target/body/relative_pose", []() {
          return std::make_shared<FlatBodyPoseCommandConverter>();
        });
      registry.register_converter(
        "command/body/pose_rel", []() {
          return std::make_shared<FlatBodyPoseCommandConverter>();
        });
      registry.register_converter(
        "command/body/pose_relative", []() {
          return std::make_shared<FlatBodyPoseCommandConverter>();
        });
      registry.register_converter(
        "command/body/pose_delta", []() {
          return std::make_shared<FlatBodyPoseCommandConverter>();
        });
      registry.register_converter(
        "command/body/relative_pose", []() {
          return std::make_shared<FlatBodyPoseCommandConverter>();
        });
      registry.register_converter(
        "kp", []() {
          return std::make_shared<JointCommandConverter>("kp");
        });
      registry.register_converter(
        "kd", []() {
          return std::make_shared<JointCommandConverter>("kd");
        });
    });
}

}  // namespace isaac_ros_deploy_ros2_control

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

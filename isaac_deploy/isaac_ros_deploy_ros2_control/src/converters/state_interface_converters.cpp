// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/converters/state_interface_converter.hpp"

#include <mutex>
#include <stdexcept>


namespace isaac_ros_deploy_ros2_control
{

namespace
{

/// Converter for joint state interfaces, parameterized by interface suffix.
class JointStateConverter : public StateInterfaceConverter
{
public:
  explicit JointStateConverter(std::string interface_suffix)
  : interface_suffix_(std::move(interface_suffix)) {}

  std::vector<std::string> get_required_state_interfaces(
    const std::vector<std::vector<std::string>> & element_names) const override
  {
    if (element_names.empty()) {
      throw std::runtime_error(
              "element_names must not be empty for state converter '" + interface_suffix_ + "'");
    }
    std::vector<std::string> interfaces;
    for (const auto & name : element_names.back()) {
      interfaces.push_back(name + "/" + interface_suffix_);
    }
    return interfaces;
  }

  isaac_deploy_core::TensorSpec get_tensor_spec(
    const std::vector<std::vector<std::string>> & element_names) const override
  {
    return {.names = element_names};
  }

private:
  std::string interface_suffix_;
};

/// Converter for fixed-layout IMU state interfaces.
/// Returns hardware-order TensorSpec (not YAML element_names) so that
/// InputTerm can reorder from hardware order to the NN order.
class ImuStateConverter : public StateInterfaceConverter
{
public:
  ImuStateConverter(
    std::string interface_prefix,
    std::vector<std::string> components)
  : interface_prefix_(std::move(interface_prefix)),
    components_(std::move(components)) {}

  std::vector<std::string> get_required_state_interfaces(
    const std::vector<std::vector<std::string>> &) const override
  {
    std::vector<std::string> interfaces;
    for (const auto & comp : components_) {
      interfaces.push_back(interface_prefix_ + "." + comp);
    }
    return interfaces;
  }

  isaac_deploy_core::TensorSpec get_tensor_spec(
    const std::vector<std::vector<std::string>> &) const override
  {
    return {.names = {{}, components_}};
  }

private:
  std::string interface_prefix_;
  std::vector<std::string> components_;
};

}  // namespace

void initialize_state_interface_converters()
{
  static std::once_flag flag;
  std::call_once(flag, []() {
      auto & registry = StateInterfaceConverterRegistry::instance();

      registry.register_converter(
        "state/joint/position", []() {
          return std::make_shared<JointStateConverter>("position");
        });
      registry.register_converter(
        "state/joint/velocity", []() {
          return std::make_shared<JointStateConverter>("velocity");
        });
      registry.register_converter(
        "state/body/rotation", []() {
          return std::make_shared<ImuStateConverter>(
            "imu/orientation", std::vector<std::string>{"x", "y", "z", "w"});
        });
      registry.register_converter(
        "state/body/angular_velocity", []() {
          return std::make_shared<ImuStateConverter>(
            "imu/angular_velocity", std::vector<std::string>{"x", "y", "z"});
        });
    });
}

}  // namespace isaac_ros_deploy_ros2_control

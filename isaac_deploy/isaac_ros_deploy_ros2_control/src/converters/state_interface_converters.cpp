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

#include "isaac_ros_deploy_ros2_control/converters/state_interface_converter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>


namespace isaac_ros_deploy_ros2_control
{

namespace
{

constexpr std::array<const char *, 3> kPositionNames{"x", "y", "z"};
constexpr std::array<const char *, 6> kRotation6DNames{
  "r00", "r01", "r02", "r10", "r11", "r12"};

std::array<double, 4> normalize_quat_xyzw(double x, double y, double z, double w)
{
  const double norm = std::sqrt(x * x + y * y + z * z + w * w);
  if (norm <= 0.0 || !std::isfinite(norm)) {
    return {0.0, 0.0, 0.0, 1.0};
  }
  return {x / norm, y / norm, z / norm, w / norm};
}

std::array<float, 6> quat_xyzw_to_rot6d_rows(double x, double y, double z, double w)
{
  const auto q = normalize_quat_xyzw(x, y, z, w);
  x = q[0];
  y = q[1];
  z = q[2];
  w = q[3];
  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;
  const double wx = w * x;
  const double wy = w * y;
  const double wz = w * z;
  return {
    static_cast<float>(1.0 - 2.0 * (yy + zz)),
    static_cast<float>(2.0 * (xy - wz)),
    static_cast<float>(2.0 * (xz + wy)),
    static_cast<float>(2.0 * (xy + wz)),
    static_cast<float>(1.0 - 2.0 * (xx + zz)),
    static_cast<float>(2.0 * (yz - wx)),
  };
}

std::vector<std::string> default_names(
  const std::array<const char *, 3> & names)
{
  return {names.begin(), names.end()};
}

std::vector<std::string> default_names(
  const std::array<const char *, 6> & names)
{
  return {names.begin(), names.end()};
}

std::vector<std::string> interface_names_or_body_default(
  const std::vector<std::vector<std::string>> & element_names,
  const std::string & default_body_name,
  const std::string & interface_prefix,
  const std::vector<std::string> & components)
{
  if (!element_names.empty() && element_names.front().size() == components.size()) {
    const auto & candidate_interfaces = element_names.front();
    const bool looks_like_interfaces = std::any_of(
      candidate_interfaces.begin(), candidate_interfaces.end(), [](const std::string & name) {
        return name.find('/') != std::string::npos;
      });
    if (looks_like_interfaces) {
      return candidate_interfaces;
    }
  }
  const std::string body_name =
    (!element_names.empty() && element_names.front().size() == 1) ?
    element_names.front().front() : default_body_name;
  std::vector<std::string> interfaces;
  interfaces.reserve(components.size());
  for (const auto & component : components) {
    interfaces.push_back(body_name + "/" + interface_prefix + "." + component);
  }
  return interfaces;
}

std::vector<std::string> tensor_names_or_default(
  const std::vector<std::vector<std::string>> & element_names,
  size_t expected_size,
  const std::vector<std::string> & default_element_names)
{
  if (!element_names.empty() && element_names.back().size() == expected_size) {
    return element_names.back();
  }
  return default_element_names;
}

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

class BodyPositionStateConverter : public StateInterfaceConverter
{
public:
  std::vector<std::string> get_required_state_interfaces(
    const std::vector<std::vector<std::string>> & element_names) const override
  {
    return interface_names_or_body_default(
      element_names, "eef", "position", default_names(kPositionNames));
  }

  isaac_deploy_core::TensorSpec get_tensor_spec(
    const std::vector<std::vector<std::string>> & element_names) const override
  {
    return {.names = {{}, tensor_names_or_default(
        element_names, 3, default_names(kPositionNames))}};
  }
};

class BodyRotation6DStateConverter : public StateInterfaceConverter
{
public:
  std::vector<std::string> get_required_state_interfaces(
    const std::vector<std::vector<std::string>> & element_names) const override
  {
    return interface_names_or_body_default(
      element_names, "eef", "orientation", {"x", "y", "z", "w"});
  }

  isaac_deploy_core::TensorSpec get_tensor_spec(
    const std::vector<std::vector<std::string>> & element_names) const override
  {
    return {.names = {{}, tensor_names_or_default(
        element_names, 6, default_names(kRotation6DNames))}};
  }

  void read(
    const std::vector<hardware_interface::LoanedStateInterface> & interfaces,
    const std::vector<size_t> & indices,
    torch::Tensor & output) const override
  {
    if (indices.size() != 4) {
      RCLCPP_WARN_ONCE(
        rclcpp::get_logger("BodyRotation6DStateConverter"),
        "Rotation 6D converter expected four quaternion state interfaces");
      return;
    }
    std::array<double, 4> quat{0.0, 0.0, 0.0, 1.0};
    for (size_t i = 0; i < indices.size(); ++i) {
      auto value_opt = interfaces[indices[i]].get_optional<double>();
      if (value_opt.has_value()) {
        quat[i] = value_opt.value();
      } else {
        RCLCPP_WARN_ONCE(
          rclcpp::get_logger("BodyRotation6DStateConverter"),
          "Quaternion state interface at index %zu returned no value, using identity",
          indices[i]);
        quat = {0.0, 0.0, 0.0, 1.0};
        break;
      }
    }
    const auto rot6d = quat_xyzw_to_rot6d_rows(quat[0], quat[1], quat[2], quat[3]);
    auto accessor = output.accessor<float, 2>();
    for (size_t i = 0; i < rot6d.size(); ++i) {
      accessor[0][i] = rot6d[i];
    }
  }
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
        "state/body/position", []() {
          return std::make_shared<BodyPositionStateConverter>();
        });
      registry.register_converter(
        "state/body/rotation", []() {
          return std::make_shared<ImuStateConverter>(
            "imu/orientation", std::vector<std::string>{"x", "y", "z", "w"});
        });
      registry.register_converter(
        "state/body/rotation_6d", []() {
          return std::make_shared<BodyRotation6DStateConverter>();
        });
      registry.register_converter(
        "state/body/rot_6d", []() {
          return std::make_shared<BodyRotation6DStateConverter>();
        });
      registry.register_converter(
        "state/body/angular_velocity", []() {
          return std::make_shared<ImuStateConverter>(
            "imu/angular_velocity", std::vector<std::string>{"x", "y", "z"});
        });
    });
}

}  // namespace isaac_ros_deploy_ros2_control

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"

#include <torch/torch.h>

namespace isaac_ros_deploy_ros2_control
{
namespace utils
{

isaac_deploy_core::TensorSpec create_joint_tensor_spec(
  const std::vector<std::string> & joint_names)
{
  isaac_deploy_core::TensorSpec spec;
  spec.names.resize(2);
  spec.names[0] = {""};  // Batch dimension.
  spec.names[1] = joint_names;
  return spec;
}

isaac_deploy_core::TensorSpec create_scalar_tensor_spec(const std::string & name)
{
  isaac_deploy_core::TensorSpec spec;
  spec.names.resize(2);
  spec.names[0] = {""};
  spec.names[1] = {name};
  return spec;
}

std::optional<std::vector<size_t>> find_state_interface_indices(
  const std::vector<std::string> & interface_names,
  const std::vector<hardware_interface::LoanedStateInterface> & state_interfaces)
{
  std::vector<size_t> indices;
  indices.reserve(interface_names.size());

  for (const auto & name : interface_names) {
    bool found = false;
    for (size_t i = 0; i < state_interfaces.size(); ++i) {
      if (state_interfaces[i].get_name() == name) {
        indices.push_back(i);
        found = true;
        break;
      }
    }
    if (!found) {
      return std::nullopt;
    }
  }

  return indices;
}

std::optional<std::vector<size_t>> find_command_interface_indices(
  const std::vector<std::string> & interface_names,
  const std::vector<hardware_interface::LoanedCommandInterface> & command_interfaces)
{
  std::vector<size_t> indices;
  indices.reserve(interface_names.size());

  for (const auto & name : interface_names) {
    bool found = false;
    for (size_t i = 0; i < command_interfaces.size(); ++i) {
      if (command_interfaces[i].get_name() == name) {
        indices.push_back(i);
        found = true;
        break;
      }
    }
    if (!found) {
      return std::nullopt;
    }
  }

  return indices;
}

bool populate_tensor_from_state_interfaces(
  isaac_deploy_core::NamedTensor & tensor,
  const std::vector<size_t> & indices,
  const std::vector<hardware_interface::LoanedStateInterface> & state_interfaces,
  int64_t timestamp_ns)
{
  tensor.timestamp_ns = timestamp_ns;
  auto accessor = tensor.tensor.accessor<float, 2>();
  bool all_ok = true;
  for (size_t i = 0; i < indices.size(); ++i) {
    const auto value_opt = state_interfaces[indices[i]].get_optional<double>();
    if (value_opt.has_value()) {
      accessor[0][i] = static_cast<float>(value_opt.value());
    } else {
      // Keep stale value in tensor; caller should log.
      all_ok = false;
    }
  }
  return all_ok;
}

void write_tensor_to_command_interfaces(
  const isaac_deploy_core::NamedTensor & tensor,
  const std::vector<size_t> & indices,
  std::vector<hardware_interface::LoanedCommandInterface> & command_interfaces)
{
  auto accessor = tensor.tensor.accessor<float, 2>();
  for (size_t i = 0; i < indices.size(); ++i) {
    (void)command_interfaces[indices[i]].set_value(static_cast<double>(accessor[0][i]));
  }
}

isaac_deploy_core::NamedTensor create_preallocated_tensor(
  const std::string & name,
  const std::vector<int64_t> & shape)
{
  isaac_deploy_core::NamedTensor tensor;
  tensor.name = name;
  tensor.timestamp_ns = 0;
  tensor.tensor = torch::zeros(shape, torch::kFloat32);
  return tensor;
}

}  // namespace utils
}  // namespace isaac_ros_deploy_ros2_control

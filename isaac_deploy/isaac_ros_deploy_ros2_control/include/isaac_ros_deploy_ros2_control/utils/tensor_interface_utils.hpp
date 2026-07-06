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

#include <optional>
#include <string>
#include <vector>

#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>

#include "isaac_deploy_core/core/types.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace utils
{

/// Create a TensorSpec for a joint tensor with shape [1, N].
/// @param joint_names Names of joints (becomes element names for dimension 1).
/// @return TensorSpec with structure {{""}, {joint_names...}}.
isaac_deploy_core::TensorSpec create_joint_tensor_spec(
  const std::vector<std::string> & joint_names);

/// Create a TensorSpec for a scalar tensor with shape [1, 1].
/// @param name Name of the scalar element.
/// @return TensorSpec with structure {{""}, {name}}.
isaac_deploy_core::TensorSpec create_scalar_tensor_spec(const std::string & name);

/// Find indices of named state interfaces.
/// @param interface_names Full interface names (e.g., "joint1/position").
/// @param state_interfaces Available state interfaces.
/// @return Vector of indices, or nullopt if any interface not found.
std::optional<std::vector<size_t>> find_state_interface_indices(
  const std::vector<std::string> & interface_names,
  const std::vector<hardware_interface::LoanedStateInterface> & state_interfaces);

/// Find indices of named command interfaces.
/// @param interface_names Full interface names.
/// @param command_interfaces Available command interfaces.
/// @return Vector of indices, or nullopt if any interface not found.
std::optional<std::vector<size_t>> find_command_interface_indices(
  const std::vector<std::string> & interface_names,
  const std::vector<hardware_interface::LoanedCommandInterface> & command_interfaces);

/// Populate a NamedTensor from state interfaces (real-time safe).
/// @param tensor NamedTensor to populate (must have pre-allocated tensor).
/// @param indices Cached indices into state_interfaces.
/// @param state_interfaces Available state interfaces.
/// @param timestamp_ns Timestamp to set on the tensor.
/// @return true if all reads succeeded, false if any interface returned nullopt
///         (stale values are kept in the tensor for failed reads).
bool populate_tensor_from_state_interfaces(
  isaac_deploy_core::NamedTensor & tensor,
  const std::vector<size_t> & indices,
  const std::vector<hardware_interface::LoanedStateInterface> & state_interfaces,
  int64_t timestamp_ns);

/// Write a NamedTensor to command interfaces (real-time safe).
/// @param tensor NamedTensor containing values to write.
/// @param indices Cached indices into command_interfaces.
/// @param command_interfaces Target command interfaces.
void write_tensor_to_command_interfaces(
  const isaac_deploy_core::NamedTensor & tensor,
  const std::vector<size_t> & indices,
  std::vector<hardware_interface::LoanedCommandInterface> & command_interfaces);

/// Pre-allocate a NamedTensor with given shape.
/// @param name Name for the tensor.
/// @param shape Tensor shape.
/// @return Pre-allocated NamedTensor with zero-initialized tensor.
isaac_deploy_core::NamedTensor create_preallocated_tensor(
  const std::string & name,
  const std::vector<int64_t> & shape);

}  // namespace utils
}  // namespace isaac_ros_deploy_ros2_control

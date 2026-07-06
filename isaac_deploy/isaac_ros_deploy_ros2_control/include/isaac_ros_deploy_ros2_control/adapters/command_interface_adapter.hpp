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

#include <memory>
#include <string>
#include <vector>

#include <hardware_interface/loaned_command_interface.hpp>

#include "isaac_deploy_core/core/types.hpp"
#include "isaac_deploy_core/inference_controller/output/output_term.hpp"
#include "isaac_ros_deploy_ros2_control/converters/command_interface_converter.hpp"

namespace isaac_ros_deploy_ros2_control
{

/// Adapter that bridges tensors to ROS2 control's scalar command interfaces.
///
/// Uses the CommandInterfaceConverter registry to look up converters by kind.
///
/// This class:
/// - Looks up converters from the registry based on output kind
/// - Aggregates required command interface names from all converters
/// - Caches interface indices for O(1) lookup during the control loop
/// - Delegates writing to converters (RT-safe, no allocations in hot path)
class CommandInterfaceAdapter
{
public:
  /// Create adapter from output configurations.
  /// Looks up converters from the registry for each config.
  /// @param configs List of output configurations (only hardware outputs).
  /// @param command_prefix Prefix for command interfaces (for chained controllers).
  /// @param command_suffix Suffix for command interfaces (e.g., "_raw").
  /// @throws std::runtime_error if no converter found for any config's kind.
  explicit CommandInterfaceAdapter(
    const std::vector<isaac_deploy_core::OutputTermConfig> & configs,
    const std::string & command_prefix = "",
    const std::string & command_suffix = "");

  /// Get list of required command interface names based on configured outputs.
  std::vector<std::string> get_required_command_interfaces() const;

  /// Set the loaned command interfaces and resolve interface indices.
  /// @throws std::runtime_error if any required interface is not found.
  void set_command_interfaces(
    std::vector<hardware_interface::LoanedCommandInterface> & command_interfaces);

  /// Write tensor values to command interfaces (RT-safe, no allocations).
  /// @param index Index of the hardware output to write.
  /// @param tensor The tensor with values to write.
  void write_tensor(size_t index, const isaac_deploy_core::NamedTensor & tensor) const;

  /// Get TensorSpec for a specific hardware output by index.
  isaac_deploy_core::TensorSpec get_tensor_spec(size_t index) const;

  /// Get output name for a specific hardware output by index.
  std::string get_output_name(size_t index) const;

  /// Get number of hardware outputs.
  size_t get_num_outputs() const;

private:
  struct Entry
  {
    isaac_deploy_core::OutputTermConfig config;
    std::shared_ptr<CommandInterfaceConverter> converter;
    std::vector<std::string> interface_names;   // Cached from converter.
    std::vector<size_t> interface_indices;       // Resolved in set_command_interfaces.
  };

  const Entry & entry_at(size_t index) const;

  std::vector<Entry> entries_;
  std::string command_prefix_;
  std::string command_suffix_;
  std::vector<hardware_interface::LoanedCommandInterface> * command_interfaces_{nullptr};
};

}  // namespace isaac_ros_deploy_ros2_control

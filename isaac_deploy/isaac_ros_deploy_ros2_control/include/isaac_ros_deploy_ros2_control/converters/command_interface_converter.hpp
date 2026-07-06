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

#include <torch/torch.h>

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include <hardware_interface/loaned_command_interface.hpp>

#include "isaac_deploy_core/core/types.hpp"
#include "isaac_ros_deploy_ros2_control/converters/converter_registry.hpp"

namespace isaac_ros_deploy_ros2_control
{

/// Abstract base class for command interface converters.
///
/// Each converter handles a single output kind (e.g., "target/joint/position") and knows:
/// - Which hardware command interfaces to claim
/// - What TensorSpec to report (for reordering by the core OutputTerm)
/// - How to write tensor values to command interfaces (RT-safe)
class CommandInterfaceConverter
{
public:
  virtual ~CommandInterfaceConverter() = default;

  /// Command interface names to claim.
  virtual std::vector<std::string> get_required_command_interfaces(
    const std::vector<std::vector<std::string>> & element_names,
    const std::string & prefix, const std::string & suffix) const = 0;

  /// TensorSpec for reordering (provides element names in hardware order).
  virtual isaac_deploy_core::TensorSpec get_tensor_spec(
    const std::vector<std::vector<std::string>> & element_names) const = 0;

  /// Write tensor values to command interfaces (RT-safe, no allocations).
  /// Default implementation writes flattened tensor[i] to each interface.
  virtual void write(
    const torch::Tensor & tensor,
    std::vector<hardware_interface::LoanedCommandInterface> & interfaces,
    const std::vector<size_t> & indices) const
  {
    auto flat = tensor.flatten();
    auto accessor = flat.accessor<float, 1>();
    if (indices.size() > static_cast<size_t>(accessor.size(0))) {
      return;  // Tensor too small for command interfaces — skip to avoid OOB.
    }
    for (size_t i = 0; i < indices.size(); ++i) {
      (void)interfaces[indices[i]].set_value(static_cast<double>(accessor[i]));
    }
  }
};

using CommandInterfaceConverterRegistry = ConverterRegistry<CommandInterfaceConverter>;

/// Initialize all built-in command interface converters.
/// Must be called before using the registry.
void initialize_command_interface_converters();

}  // namespace isaac_ros_deploy_ros2_control

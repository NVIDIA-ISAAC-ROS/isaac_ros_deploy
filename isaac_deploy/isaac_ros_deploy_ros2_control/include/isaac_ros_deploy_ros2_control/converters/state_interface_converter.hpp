// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#include <hardware_interface/loaned_state_interface.hpp>
#include <rclcpp/logging.hpp>

#include "isaac_deploy_core/core/types.h"
#include "isaac_ros_deploy_ros2_control/converters/converter_registry.hpp"

namespace isaac_ros_deploy_ros2_control
{

/// Abstract base class for state interface converters.
///
/// Each converter handles a single input kind (e.g., "state/joint/position") and knows:
/// - Which hardware state interfaces to claim
/// - What TensorSpec to report (for reordering by the core InputTerm)
/// - How to read state interface values into a pre-allocated tensor (RT-safe)
class StateInterfaceConverter
{
public:
  virtual ~StateInterfaceConverter() = default;

  /// State interface names to claim (e.g., ["joint1/position", "joint2/position"]).
  virtual std::vector<std::string> get_required_state_interfaces(
    const std::vector<std::vector<std::string>> & element_names) const = 0;

  /// TensorSpec in hardware order (for reordering by core InputTerm).
  virtual isaac_deploy_core::TensorSpec get_tensor_spec(
    const std::vector<std::vector<std::string>> & element_names) const = 0;

  /// Read state interface values into pre-allocated tensor (RT-safe, no allocations).
  /// Default implementation reads each interface value into tensor[0][i].
  virtual void read(
    const std::vector<hardware_interface::LoanedStateInterface> & interfaces,
    const std::vector<size_t> & indices,
    torch::Tensor & output) const
  {
    auto accessor = output.accessor<float, 2>();
    for (size_t i = 0; i < indices.size(); ++i) {
      auto value_opt = interfaces[indices[i]].get_optional<double>();
      if (value_opt.has_value()) {
        accessor[0][i] = static_cast<float>(value_opt.value());
      } else {
        accessor[0][i] = 0.0f;
        RCLCPP_WARN_ONCE(
          rclcpp::get_logger("StateInterfaceConverter"),
          "State interface at index %zu returned no value, using 0.0 as fallback",
          indices[i]);
      }
    }
  }
};

using StateInterfaceConverterRegistry = ConverterRegistry<StateInterfaceConverter>;

/// Initialize all built-in state interface converters.
/// Must be called before using the registry.
void initialize_state_interface_converters();

}  // namespace isaac_ros_deploy_ros2_control

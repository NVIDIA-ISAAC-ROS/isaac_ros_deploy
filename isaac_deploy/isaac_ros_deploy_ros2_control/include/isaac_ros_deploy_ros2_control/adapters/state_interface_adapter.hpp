// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#include <hardware_interface/loaned_state_interface.hpp>

#include "isaac_deploy_core/core/types.h"
#include "isaac_deploy_core/inference_controller/input/input_term.h"
#include "isaac_ros_deploy_ros2_control/converters/state_interface_converter.hpp"

namespace isaac_ros_deploy_ros2_control
{

/// Adapter that bridges ROS2 control's scalar state interfaces to tensors.
///
/// Uses the StateInterfaceConverter registry to look up converters by kind.
/// Only handles hardware-backed inputs (joint positions, IMU, etc.).
///
/// This class:
/// - Looks up converters from the registry based on input kind
/// - Aggregates required state interface names from all converters
/// - Caches interface indices for O(1) lookup during the control loop
/// - Delegates reading to converters (RT-safe, no allocations in hot path)
class StateInterfaceAdapter
{
public:
  /// Create adapter from input configurations.
  /// Looks up converters from the registry for each config.
  /// @throws std::runtime_error if no converter found for a config's kind.
  explicit StateInterfaceAdapter(
    const std::vector<isaac_deploy_core::InputTermConfig> & configs);

  /// Get list of required state interface names based on configured inputs.
  std::vector<std::string> get_required_state_interfaces() const;

  /// Set the loaned state interfaces and resolve interface indices.
  /// @throws std::runtime_error if any required interface is not found.
  void set_state_interfaces(
    const std::vector<hardware_interface::LoanedStateInterface> & state_interfaces);

  /// Read state interface values into a pre-allocated tensor (RT-safe, no allocations).
  /// @param index Index of the hardware input to read.
  /// @param output Pre-allocated tensor to write values into.
  void read_tensor(size_t index, torch::Tensor & output) const;

  /// Get TensorSpec for a specific hardware input by index.
  isaac_deploy_core::TensorSpec get_tensor_spec(size_t index) const;

  /// Get the name of a hardware input by index.
  std::string get_input_name(size_t index) const;

  /// Get the source of a hardware input by index.
  std::string get_input_source(size_t index) const;

  /// Get the shape of a hardware input by index.
  std::vector<int64_t> get_shape(size_t index) const;

  /// Get number of hardware inputs.
  size_t get_num_hardware_inputs() const;

private:
  struct Entry
  {
    isaac_deploy_core::InputTermConfig config;
    std::shared_ptr<StateInterfaceConverter> converter;
    std::vector<std::string> interface_names;   // Cached from converter.
    std::vector<size_t> interface_indices;       // Resolved in set_state_interfaces.
  };

  const Entry & entry_at(size_t index) const;

  std::vector<Entry> entries_;
  const std::vector<hardware_interface::LoanedStateInterface> * state_interfaces_{nullptr};
};

}  // namespace isaac_ros_deploy_ros2_control

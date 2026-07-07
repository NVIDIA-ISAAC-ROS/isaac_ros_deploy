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

#include "isaac_ros_deploy_ros2_control/adapters/state_interface_adapter.hpp"

#include <set>
#include <stdexcept>

#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"

namespace isaac_ros_deploy_ros2_control
{

StateInterfaceAdapter::StateInterfaceAdapter(
  const std::vector<isaac_deploy_core::InputTermConfig> & configs)
{
  auto & registry = StateInterfaceConverterRegistry::instance();

  for (const auto & config : configs) {
    auto converter = registry.create_for_kind(config.kind);
    if (!converter) {
      throw std::runtime_error("No state interface converter for kind: " + config.kind);
    }

    entries_.push_back(
      {
        .config = config,
        .converter = converter,
        .interface_names = converter->get_required_state_interfaces(config.element_names),
      });
  }
}

std::vector<std::string> StateInterfaceAdapter::get_required_state_interfaces() const
{
  std::set<std::string> interfaces_set;
  for (const auto & entry : entries_) {
    interfaces_set.insert(entry.interface_names.begin(), entry.interface_names.end());
  }
  return {interfaces_set.begin(), interfaces_set.end()};
}

void StateInterfaceAdapter::set_state_interfaces(
  const std::vector<hardware_interface::LoanedStateInterface> & state_interfaces)
{
  state_interfaces_ = &state_interfaces;

  for (auto & entry : entries_) {
    auto indices = utils::find_state_interface_indices(entry.interface_names, state_interfaces);
    if (!indices.has_value()) {
      throw std::runtime_error(
              "Failed to find state interfaces for kind: " + entry.config.kind);
    }
    entry.interface_indices = std::move(indices.value());
  }
}

const StateInterfaceAdapter::Entry & StateInterfaceAdapter::entry_at(size_t index) const
{
  if (index >= entries_.size()) {
    throw std::runtime_error("Input index out of bounds: " + std::to_string(index));
  }
  return entries_[index];
}

void StateInterfaceAdapter::read_tensor(size_t index, torch::Tensor & output) const
{
  if (state_interfaces_ == nullptr) {
    throw std::runtime_error("State interfaces not set. Call set_state_interfaces first.");
  }
  const auto & entry = entry_at(index);
  entry.converter->read(*state_interfaces_, entry.interface_indices, output);
}

isaac_deploy_core::TensorSpec StateInterfaceAdapter::get_tensor_spec(size_t index) const
{
  const auto & entry = entry_at(index);
  return entry.converter->get_tensor_spec(entry.config.element_names);
}

std::string StateInterfaceAdapter::get_input_name(size_t index) const
{
  return entry_at(index).config.name;
}

std::string StateInterfaceAdapter::get_input_source(size_t index) const
{
  return entry_at(index).config.source;
}

std::vector<int64_t> StateInterfaceAdapter::get_shape(size_t index) const
{
  return entry_at(index).config.shape;
}

size_t StateInterfaceAdapter::get_num_hardware_inputs() const
{
  return entries_.size();
}

}  // namespace isaac_ros_deploy_ros2_control

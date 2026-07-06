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

#include "isaac_ros_deploy_ros2_control/adapters/command_interface_adapter.hpp"

#include <stdexcept>

#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"

namespace isaac_ros_deploy_ros2_control
{

CommandInterfaceAdapter::CommandInterfaceAdapter(
  const std::vector<isaac_deploy_core::OutputTermConfig> & configs,
  const std::string & command_prefix,
  const std::string & command_suffix)
: command_prefix_(command_prefix),
  command_suffix_(command_suffix)
{
  auto & registry = CommandInterfaceConverterRegistry::instance();

  for (const auto & config : configs) {
    auto converter = registry.create_for_kind(config.kind);
    if (!converter) {
      throw std::runtime_error(
              "No command interface converter for kind: " + config.kind);
    }

    entries_.push_back(
      {
        .config = config,
        .converter = converter,
        .interface_names = converter->get_required_command_interfaces(
          config.element_names, command_prefix_, command_suffix_),
      });
  }
}

std::vector<std::string> CommandInterfaceAdapter::get_required_command_interfaces() const
{
  std::vector<std::string> interfaces;
  for (const auto & entry : entries_) {
    interfaces.insert(
      interfaces.end(),
      entry.interface_names.begin(), entry.interface_names.end());
  }
  return interfaces;
}

void CommandInterfaceAdapter::set_command_interfaces(
  std::vector<hardware_interface::LoanedCommandInterface> & command_interfaces)
{
  command_interfaces_ = &command_interfaces;

  for (auto & entry : entries_) {
    auto indices = utils::find_command_interface_indices(
      entry.interface_names, command_interfaces);
    if (!indices.has_value()) {
      throw std::runtime_error(
              "Failed to find command interfaces for kind: " + entry.config.kind);
    }
    entry.interface_indices = std::move(indices.value());
  }
}

const CommandInterfaceAdapter::Entry & CommandInterfaceAdapter::entry_at(size_t index) const
{
  if (index >= entries_.size()) {
    throw std::runtime_error("Output index out of bounds: " + std::to_string(index));
  }
  return entries_[index];
}

void CommandInterfaceAdapter::write_tensor(
  size_t index,
  const isaac_deploy_core::NamedTensor & tensor) const
{
  if (command_interfaces_ == nullptr) {
    throw std::runtime_error("Command interfaces not set. Call set_command_interfaces first.");
  }
  const auto & entry = entry_at(index);
  entry.converter->write(tensor.tensor, *command_interfaces_, entry.interface_indices);
}

isaac_deploy_core::TensorSpec CommandInterfaceAdapter::get_tensor_spec(size_t index) const
{
  const auto & entry = entry_at(index);
  return entry.converter->get_tensor_spec(entry.config.element_names);
}

std::string CommandInterfaceAdapter::get_output_name(size_t index) const
{
  return entry_at(index).config.name;
}

size_t CommandInterfaceAdapter::get_num_outputs() const
{
  return entries_.size();
}

}  // namespace isaac_ros_deploy_ros2_control

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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "torch/torch.h"

#include "isaac_deploy_core/core/types.hpp"

namespace isaac_ros_deploy_converters
{

/// Abstract base class for message-to-tensor converters.
///
/// Each converter handles a single kind (e.g., "joint_pos") and produces
/// a single tensor from a serialized ROS message.
class MessageToTensorConverter
{
public:
  virtual ~MessageToTensorConverter() = default;

  /// The single kind this converter produces (e.g., "joint_pos").
  virtual std::string get_kind() const = 0;

  /// The ROS message type this converter consumes (e.g., "sensor_msgs/msg/JointState").
  virtual std::string get_message_type() const = 0;

  /// Convert a serialized ROS message to a single tensor.
  /// @param msg Serialized ROS message.
  /// @return Tensor in message order (no reordering).
  virtual torch::Tensor convert(
    const std::shared_ptr<rclcpp::SerializedMessage> & msg) = 0;

  /// Get TensorSpec with element names in message order (for reordering).
  /// Called once during activation, not every tick.
  virtual isaac_deploy_core::TensorSpec get_tensor_spec() const {return {};}
};

/// Factory function type for creating message-to-tensor converters.
/// The source parameter identifies the input source (e.g., tensor name).
/// Most converters ignore it; parameterized converters like TensorListConverter use it.
using MessageToTensorConverterFactory =
  std::function<std::shared_ptr<MessageToTensorConverter>(const std::string & source)>;

/// Registry for message-to-tensor converters.
///
/// Each converter is registered under a single kind. Lookup is by kind.
// TODO(lgulich): Replace with ConverterRegistry<MessageToTensorConverter>
// from isaac_ros_deploy_ros2_control/converters/converter_registry.hpp
// (move template to shared package).
class MessageToTensorConverterRegistry
{
public:
  static MessageToTensorConverterRegistry & instance()
  {
    static MessageToTensorConverterRegistry registry;
    return registry;
  }

  /// Register a converter factory for a (kind, message_type) pair.
  /// The first message type registered for a given kind becomes the default.
  void register_converter(
    std::string kind, std::string message_type, MessageToTensorConverterFactory factory)
  {
    auto & entry = kind_to_entry_[kind];
    if (entry.default_message_type.empty()) {
      entry.default_message_type = message_type;
    }
    entry.type_to_factory[std::move(message_type)] = std::move(factory);
  }

  /// Create a converter for a specific kind, optionally specifying message_type.
  /// If message_type is empty, uses the default for that kind.
  /// @param source Input source name, forwarded to the factory (e.g., tensor name).
  /// @param message_type ROS message type override. Empty = use default.
  std::shared_ptr<MessageToTensorConverter> create_for_kind(
    const std::string & kind,
    const std::string & source = {},
    const std::string & message_type = {}) const
  {
    auto kind_it = kind_to_entry_.find(kind);
    if (kind_it == kind_to_entry_.end()) {
      return nullptr;
    }
    const auto & entry = kind_it->second;
    const auto & effective_type = message_type.empty() ?
      entry.default_message_type : message_type;
    auto factory_it = entry.type_to_factory.find(effective_type);
    if (factory_it == entry.type_to_factory.end()) {
      return nullptr;
    }
    return factory_it->second(source);
  }

private:
  MessageToTensorConverterRegistry() = default;

  struct KindEntry
  {
    std::string default_message_type;
    std::unordered_map<std::string, MessageToTensorConverterFactory> type_to_factory;
  };
  std::unordered_map<std::string, KindEntry> kind_to_entry_;
};

/// Initialize all built-in input converters.
/// Must be called before using the registry.
void initialize_input_converters();

}  // namespace isaac_ros_deploy_converters

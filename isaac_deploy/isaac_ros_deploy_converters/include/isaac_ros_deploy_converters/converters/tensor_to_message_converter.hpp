// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "torch/torch.h"

namespace isaac_ros_deploy_converters
{

/// Abstract base class for tensor-to-message output converters.
///
/// Each converter handles a single kind (e.g., "joint_pos_targets") and writes
/// its data into a shared typed message. Multiple converters can write to the
/// same message sequentially (e.g., position + stiffness into JointCommand).
///
/// This is symmetric with the input side's MessageToTensorConverter: one
/// converter per kind, stateless, grouped by topic in the node.
class TensorToMessageConverter
{
public:
  virtual ~TensorToMessageConverter() = default;

  /// The single kind this converter handles (e.g., "joint_pos_targets").
  virtual std::string get_kind() const = 0;

  /// The ROS message type (e.g., "isaac_ros_deploy_interfaces/msg/JointCommand").
  virtual std::string get_message_type() const = 0;

  /// Create a new default message instance (type-erased).
  virtual std::shared_ptr<void> create_message() const = 0;

  /// Write tensor data into the message.
  ///
  /// Each converter writes its own data field plus shared fields (header, names).
  /// Redundant writes to shared fields are harmless — values should be identical
  /// across converters in the same publication group.
  ///
  /// @param tensor The output tensor to write.
  /// @param element_names Element names for the tensor (e.g., joint names).
  /// @param msg The typed message (type-erased via shared_ptr<void>).
  /// @param timestamp_ns Timestamp in nanoseconds.
  virtual void write(
    const torch::Tensor & tensor,
    const std::vector<std::string> & element_names,
    const std::shared_ptr<void> & msg,
    int64_t timestamp_ns) const = 0;

  /// Serialize the typed message to a SerializedMessage.
  /// @param msg The typed message (type-erased via shared_ptr<void>).
  /// @return Serialized ROS message.
  virtual std::shared_ptr<rclcpp::SerializedMessage> serialize(
    const std::shared_ptr<void> & msg) const = 0;
};

/// Template base class that provides create_message() and serialize()
/// for a specific ROS message type. Per-kind converters only need to
/// implement get_kind(), get_message_type(), and write().
template<typename MsgT>
class TypedOutputConverter : public TensorToMessageConverter
{
public:
  std::shared_ptr<void> create_message() const override
  {
    return std::make_shared<MsgT>();
  }

  std::shared_ptr<rclcpp::SerializedMessage> serialize(
    const std::shared_ptr<void> & msg) const override
  {
    auto serialized = std::make_shared<rclcpp::SerializedMessage>();
    rclcpp::Serialization<MsgT> serializer;
    serializer.serialize_message(static_cast<const MsgT *>(msg.get()), serialized.get());
    return serialized;
  }
};

/// Factory function type for creating output converters.
using OutputConverterFactory = std::function<std::shared_ptr<TensorToMessageConverter>()>;

/// Registry for output converters.
class OutputConverterRegistry
{
public:
  static OutputConverterRegistry & instance()
  {
    static OutputConverterRegistry registry;
    return registry;
  }

  /// Register a converter factory for a single kind.
  void register_converter(std::string kind, OutputConverterFactory factory)
  {
    kind_to_factory_[std::move(kind)] = std::move(factory);
  }

  /// Create a converter for a specific kind. Returns nullptr if not found.
  std::shared_ptr<TensorToMessageConverter> create_for_kind(const std::string & kind) const
  {
    auto it = kind_to_factory_.find(kind);
    if (it != kind_to_factory_.end()) {
      return it->second();
    }
    return nullptr;
  }

private:
  OutputConverterRegistry() = default;
  std::unordered_map<std::string, OutputConverterFactory> kind_to_factory_;
};

/// Initialize all built-in output converters.
/// Must be called before using the registry.
void initialize_output_converters();

}  // namespace isaac_ros_deploy_converters

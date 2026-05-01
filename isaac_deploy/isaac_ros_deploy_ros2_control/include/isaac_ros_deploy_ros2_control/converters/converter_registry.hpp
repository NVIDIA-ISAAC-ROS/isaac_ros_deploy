// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace isaac_ros_deploy_ros2_control
{

/// Generic singleton registry for converter types, indexed by kind string.
///
/// @tparam ConverterT The converter base class type.
template<typename ConverterT>
class ConverterRegistry
{
public:
  using Factory = std::function<std::shared_ptr<ConverterT>()>;

  static ConverterRegistry & instance()
  {
    static ConverterRegistry registry;
    return registry;
  }

  void register_converter(const std::string & kind, Factory factory)
  {
    kind_to_factory_[kind] = std::move(factory);
  }

  /// Check if a converter exists for a specific kind.
  bool contains(const std::string & kind) const
  {
    return kind_to_factory_.contains(kind);
  }

  /// Create a converter for a specific kind. Returns nullptr if not found.
  std::shared_ptr<ConverterT> create_for_kind(const std::string & kind) const
  {
    auto it = kind_to_factory_.find(kind);
    if (it != kind_to_factory_.end()) {
      return it->second();
    }
    return nullptr;
  }

private:
  ConverterRegistry() = default;

  std::unordered_map<std::string, Factory> kind_to_factory_;
};

}  // namespace isaac_ros_deploy_ros2_control

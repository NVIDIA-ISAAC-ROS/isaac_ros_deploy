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

#include <string>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_tensor_list_interfaces/msg/tensor.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

#include "isaac_deploy_core/core/types.hpp"

namespace isaac_ros_deploy_converters
{

/// Convert a TensorList message to a TensorDict.
/// @param msg The TensorList message.
/// @return Map of tensor names to torch::Tensor.
isaac_deploy_core::TensorDict tensor_list_to_dict(
  const isaac_ros_tensor_list_interfaces::msg::TensorList & msg);

/// Convert a TensorDict to a TensorList message.
/// @param dict Map of tensor names to torch::Tensor.
/// @param stamp Timestamp for the message header.
/// @param frame_id Frame ID for the message header.
/// @return TensorList message.
isaac_ros_tensor_list_interfaces::msg::TensorList dict_to_tensor_list(
  const isaac_deploy_core::TensorDict & dict,
  const rclcpp::Time & stamp,
  const std::string & frame_id = "");

/// Convert a single Tensor message to a torch::Tensor.
/// @param msg The Tensor message.
/// @return torch::Tensor.
torch::Tensor tensor_msg_to_torch(const isaac_ros_tensor_list_interfaces::msg::Tensor & msg);

/// Convert a torch::Tensor to a Tensor message.
/// @param tensor The torch::Tensor.
/// @param name Name for the tensor.
/// @return Tensor message.
isaac_ros_tensor_list_interfaces::msg::Tensor torch_to_tensor_msg(
  const torch::Tensor & tensor,
  const std::string & name);

/// Convert NamedTensors to a TensorList message.
/// @param named_tensors Map of names to NamedTensors.
/// @param stamp Timestamp for the message header.
/// @param frame_id Frame ID for the message header.
/// @return TensorList message.
isaac_ros_tensor_list_interfaces::msg::TensorList named_tensors_to_tensor_list(
  const std::unordered_map<std::string, isaac_deploy_core::NamedTensor> & named_tensors,
  const rclcpp::Time & stamp,
  const std::string & frame_id = "");

/// Convert a TensorList message to NamedTensors.
/// @param msg The TensorList message.
/// @return Map of names to NamedTensors.
std::unordered_map<std::string, isaac_deploy_core::NamedTensor> tensor_list_to_named_tensors(
  const isaac_ros_tensor_list_interfaces::msg::TensorList & msg);

/// Wrap a single tensor into a TensorList message.
/// @param name Tensor name.
/// @param tensor The torch::Tensor.
/// @param stamp Timestamp for the message header.
/// @return TensorList message containing the single tensor.
isaac_ros_tensor_list_interfaces::msg::TensorList tensor_to_tensor_list(
  const std::string & name,
  const torch::Tensor & tensor,
  const rclcpp::Time & stamp);

/// Extract the first tensor from a TensorList message.
/// @param msg The TensorList message (must contain at least one tensor).
/// @return The first tensor.
torch::Tensor tensor_list_to_tensor(const isaac_ros_tensor_list_interfaces::msg::TensorList & msg);

}  // namespace isaac_ros_deploy_converters

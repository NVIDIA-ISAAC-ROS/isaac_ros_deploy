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

#include "isaac_ros_deploy_converters/converters/message_to_tensor_converter.hpp"

#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "isaac_ros_deploy_converters/utils/tensor_list_utils.hpp"
#include "rclcpp/serialization.hpp"

namespace isaac_ros_deploy_converters
{

/// Converter that extracts a single named tensor from a TensorList message.
///
/// Used for feedback inputs (e.g., last_actions). Parameterized by tensor name,
/// not registered in the registry -- created directly by InputBuilderNode for each
/// feedback source from InputBuilder::get_feedback_input_names().
class TensorListConverter : public MessageToTensorConverter
{
public:
  explicit TensorListConverter(std::string tensor_name)
  : tensor_name_(std::move(tensor_name)) {}

  std::string get_kind() const override {return "feedback";}
  std::string get_message_type() const override
  {
    return "isaac_ros_tensor_list_interfaces/msg/TensorList";
  }

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    isaac_ros_tensor_list_interfaces::msg::TensorList tensor_list;
    rclcpp::Serialization<isaac_ros_tensor_list_interfaces::msg::TensorList> serializer;
    serializer.deserialize_message(msg.get(), &tensor_list);

    for (const auto & tensor_msg : tensor_list.tensors) {
      if (tensor_msg.name == tensor_name_) {
        return tensor_msg_to_torch(tensor_msg);
      }
    }
    return torch::empty({});
  }

  /// Feedback tensors are already in NN order, so no reordering is needed.
  isaac_deploy_core::TensorSpec get_tensor_spec() const override {return {};}

private:
  std::string tensor_name_;
};

}  // namespace isaac_ros_deploy_converters

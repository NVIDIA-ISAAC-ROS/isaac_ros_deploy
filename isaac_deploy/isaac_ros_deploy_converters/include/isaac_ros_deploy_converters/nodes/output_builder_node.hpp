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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

#include "isaac_deploy_core/inference_controller/output/output_builder.hpp"

#include "isaac_ros_deploy_converters/converters/tensor_to_message_converter.hpp"

namespace isaac_ros_deploy_converters
{

/// A converter entry within a publication group, linking an output name to its converter.
struct OutputConverterEntry
{
  std::string output_name;
  std::shared_ptr<TensorToMessageConverter> converter;
};

/// ROS 2 node that converts neural network outputs to ROS messages.
///
/// The OutputBuilderNode:
/// 1. Subscribes to a single TensorList topic from the inference node
/// 2. Uses OutputBuilder to process tensors (reordering, etc.)
/// 3. Converts tensors to ROS messages using per-kind converters
/// 4. Publishes to configured topics (JointCommand, etc.)
///
/// Only outputs that have registered output converters (i.e., external outputs
/// that produce typed ROS messages) are processed. State/feedback outputs are
/// ignored.
///
/// Parameters:
/// - config_path: Path to YAML configuration file
/// - output_to_topic.<name>: Override output topic.  The default is derived
///   from the converter's ROS message type.
/// - input_topic: Topic name for the input TensorList (default: "output_tensors")
class OutputBuilderNode : public rclcpp::Node
{
public:
  explicit OutputBuilderNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  /// Publication group for outputs to the same topic.
  ///
  /// Multiple output kinds may share a topic (e.g., joint_pos_targets and
  /// stiffness_targets both publish to "joint_commands"). Each kind has its own
  /// converter that writes its field into a shared message.
  struct PublicationGroup
  {
    std::string topic;
    std::string message_type;
    std::vector<OutputConverterEntry> converters;
    rclcpp::GenericPublisher::SharedPtr publisher;
  };

  /// Load configuration and create publishers.
  void configure();

  /// Handle incoming TensorList from the inference node.
  void on_tensor_list(const isaac_ros_tensor_list_interfaces::msg::TensorList::SharedPtr msg);

  /// Create publication groups from OutputBuilder metadata.
  void create_publication_groups();

  /// Path to configuration file.
  std::string config_path_;

  /// OutputBuilder from core library.
  std::optional<isaac_deploy_core::OutputBuilder> output_builder_;

  /// Publication groups organized by topic.
  std::vector<std::unique_ptr<PublicationGroup>> publication_groups_;

  /// Subscription for the input TensorList.
  rclcpp::Subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr input_sub_;

  /// Output tensors for OutputBuilder.
  std::vector<isaac_deploy_core::NamedTensor> outputs_;

  /// Pre-allocated output map for converters (refreshed in-place each callback).
  std::unordered_map<std::string, isaac_deploy_core::NamedTensor> output_map_;

  /// Set of output names we expect from the TensorList.
  std::unordered_map<std::string, size_t> output_name_to_index_;
};

}  // namespace isaac_ros_deploy_converters

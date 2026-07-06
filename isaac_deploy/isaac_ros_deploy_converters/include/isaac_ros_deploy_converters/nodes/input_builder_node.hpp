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
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

#include "isaac_deploy_core/inference_controller/input/input_builder.hpp"

#include "isaac_ros_deploy_converters/converters/message_to_tensor_converter.hpp"

namespace isaac_ros_deploy_converters
{

/// A converter entry within a subscription group, linking a source to its converter.
struct ConverterEntry
{
  std::string source;
  std::shared_ptr<MessageToTensorConverter> converter;
  /// If set, use this tensor before the first message arrives (e.g., zero-init for feedback).
  std::optional<torch::Tensor> initial_value;
};

/// Subscription group for inputs from the same ROS topic.
///
/// Multiple sources may share a topic (e.g., joint_pos and joint_vel both from
/// /joint_states). Each source has its own converter that extracts a single tensor.
struct SubscriptionGroup
{
  /// Topic name.
  std::string topic;
  /// Message type (all converters in this group must agree).
  std::string message_type;
  /// Converters for each source sharing this topic.
  std::vector<ConverterEntry> converters;
  /// Generic subscription.
  rclcpp::GenericSubscription::SharedPtr subscription;
  /// Latest received message.
  std::shared_ptr<rclcpp::SerializedMessage> latest_msg;
  /// Time when latest message was received.
  rclcpp::Time receive_time{0, 0, RCL_ROS_TIME};
  /// Mutex for thread-safe access.
  std::mutex mutex;
};

/// ROS 2 node that converts ROS messages to tensors and builds neural network inputs.
///
/// The InputBuilderNode:
/// 1. Uses InputBuilder to determine required sources and kinds
/// 2. Uses source_to_topic parameter to map sources to ROS topics
/// 3. Converts incoming messages to tensors using per-kind converters
/// 4. Uses InputBuilder to process tensors (reordering, history, etc.)
/// 5. Publishes a single TensorList message containing all model inputs
///
/// Parameters:
/// - config_path: Path to YAML configuration file
/// - publish_rate: Rate at which to publish TensorList (Hz)
/// - source_to_topic.<source>: ROS topic to subscribe to for a given source
///   (defaults to the source name itself)
/// - output_topic: Topic name for the output TensorList (default: "input_tensors")
class InputBuilderNode : public rclcpp::Node
{
public:
  explicit InputBuilderNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  /// Load configuration and create subscriptions.
  void configure();

  /// Timer callback to collect inputs and publish TensorList.
  void timer_callback();

  /// Create subscription groups based on source_to_topic mapping.
  void create_subscription_groups(
    const std::unordered_map<std::string, torch::Dtype> & feedback_dtypes);

  /// Check if input timestamps are synchronized and warn if not.
  void validate_input_synchronization(const rclcpp::Time & current_time);

  /// Path to configuration file.
  std::string config_path_;

  /// Publish rate in Hz.
  double publish_rate_;

  /// InputBuilder from core library.
  std::optional<isaac_deploy_core::InputBuilder> input_builder_;

  /// Subscription groups organized by topic.
  std::vector<std::unique_ptr<SubscriptionGroup>> subscription_groups_;

  /// Timer for periodic publishing.
  rclcpp::TimerBase::SharedPtr timer_;

  /// Single output publisher for the bundled TensorList.
  rclcpp::Publisher<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr output_pub_;

  /// Pre-allocated input vector for InputBuilder (positionally aligned with activate).
  std::vector<isaac_deploy_core::NamedTensor> builder_inputs_;
  std::vector<isaac_deploy_core::TensorSpec> builder_input_specs_;

  /// Pre-allocated output TensorDict for InputBuilder.
  isaac_deploy_core::TensorDict nn_inputs_;

  /// Whether the node is activated.
  bool activated_{false};
};

}  // namespace isaac_ros_deploy_converters

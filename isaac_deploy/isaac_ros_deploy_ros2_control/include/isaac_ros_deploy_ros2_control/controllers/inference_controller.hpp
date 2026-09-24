// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "isaac_deploy_core/core/types.hpp"
#include "isaac_deploy_core/inference_controller/inference_controller.hpp"
#include "isaac_ros_deploy_converters/converters/message_to_tensor_converter.hpp"

namespace isaac_ros_deploy_ros2_control
{

// Forward declarations.
class StateInterfaceAdapter;
class CommandInterfaceAdapter;

namespace controllers
{

/// ROS 2 controller wrapping isaac_deploy_core::InferenceController.
///
/// This controller:
/// - Reads robot state from hardware interfaces via StateInterfaceAdapter
/// - Subscribes to ROS topics for non-hardware inputs (reference motion, cmd_vel, etc.)
///   using the shared MessageToTensorConverter registry
/// - Delegates all tensor processing to the core controller (history, reordering, inference)
/// - Writes outputs back to command interfaces via CommandInterfaceAdapter
class InferenceController : public controller_interface::ControllerInterface
{
public:
  InferenceController();

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  // Parameters.
  std::string config_path_;
  std::string model_path_;
  int decimation_{4};
  std::string command_prefix_;
  std::string command_suffix_;
  std::string joint_name_prefix_;
  std::vector<std::string> topic_input_sources_;
  int64_t topic_input_timeout_ns_{0};
  std::vector<std::string> reference_input_sources_;

  // Interface adapters.
  std::unique_ptr<StateInterfaceAdapter> state_adapter_;
  std::unique_ptr<CommandInterfaceAdapter> command_adapter_;

  // Core controller (handles history, reordering, inference).
  std::unique_ptr<isaac_deploy_core::InferenceController> inference_controller_;

  // I/O tensors and specs (one entry per unique source / output).
  std::vector<isaac_deploy_core::NamedTensor> inputs_;
  std::vector<isaac_deploy_core::TensorSpec> input_specs_;
  std::vector<isaac_deploy_core::NamedTensor> outputs_;
  std::vector<isaac_deploy_core::TensorSpec> output_specs_;

  // Maps state adapter index -> inputs_ index (handles duplicate sources).
  std::vector<size_t> hw_to_input_idx_;

  // Maps output name -> outputs_ index for O(1) lookup.
  std::unordered_map<std::string, size_t> output_name_to_idx_;

  // Optional chained reference interfaces carrying selected input tensors to
  // downstream chainable controllers.
  struct ReferenceInputForwarding
  {
    std::string source;
    std::vector<std::string> command_interfaces;
    std::vector<size_t> command_indices;
    size_t input_index{0};
  };
  std::vector<ReferenceInputForwarding> reference_inputs_;

  // Decimation counter.
  int update_counter_{0};

  // Deferred activation: if there are topic inputs, core controller activation
  // is deferred until all topic groups have received at least one message, so
  // that converters can provide TensorSpecs with element names for reordering.
  // If there are no topic inputs, the core is activated immediately in on_activate().
  bool core_activated_{false};

  // Latched `~/is_active` publisher that emits a single `std_msgs/Bool(true)`
  // once `core_activated_` first flips true.  Tests poll this topic to
  // distinguish "controller lifecycle active" from "policy is actually
  // producing commands" (the latter is what AGILE needs to stabilise the
  // robot before `reset_simulation`).
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr is_active_publisher_;

  // --- Topic-based input handling (reference motion, cmd_vel, etc.) ---

  // Topic source configs parsed from YAML during load_config().
  struct TopicSourceConfig
  {
    std::string source;  // Effective source name (defaults to kind).
    std::string kind;    // Converter kind string.
    std::vector<int64_t> shape;
  };
  std::vector<TopicSourceConfig> topic_source_configs_;

  // Subscription groups for topic inputs.
  struct TopicGroup
  {
    struct TopicSample
    {
      std::shared_ptr<rclcpp::SerializedMessage> message;
      int64_t receive_time_ns{0};
    };

    std::string topic;
    std::string message_type;
    struct Entry
    {
      std::string source;  // Effective source name for core matching.
      std::shared_ptr<isaac_ros_deploy_converters::MessageToTensorConverter> converter;
    };
    std::vector<Entry> entries;
    rclcpp::GenericSubscription::SharedPtr subscription;
    realtime_tools::RealtimeBuffer<std::shared_ptr<TopicSample>> rt_msg_buffer;
  };
  std::vector<std::unique_ptr<TopicGroup>> topic_groups_;

  // Maps source name -> inputs_ index for O(1) topic update.
  std::unordered_map<std::string, size_t> source_to_input_idx_;

  // All output term info (name + shape), saved during load_config() for
  // building the outputs_ vector in on_activate().
  struct OutputTermInfo
  {
    std::string name;
    std::vector<int64_t> shape;
  };
  std::vector<OutputTermInfo> all_output_terms_;

  // Debug publishers: observation inputs and the selected model output published as
  // Float64MultiArray on ~/debug_observation and ~/debug_action.
  // Enabled by the publish_debug_topics parameter (default false).
  bool publish_debug_topics_{false};
  bool log_debug_to_console_{false};  // also print to terminal via RCLCPP_INFO
  int64_t debug_step_{0};             // inference step counter for log lines
  using DebugPublisher =
    realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>;
  std::shared_ptr<DebugPublisher> obs_debug_publisher_;
  std::shared_ptr<DebugPublisher> action_debug_publisher_;
  std_msgs::msg::Float64MultiArray obs_debug_msg_;
  std_msgs::msg::Float64MultiArray action_debug_msg_;
  // Publishes the concatenated recurrent (LSTM) `_out` state tensors on
  // ~/debug_recurrent_state so divergence in the hidden state can be inspected.
  std::shared_ptr<DebugPublisher> recurrent_debug_publisher_;
  std_msgs::msg::Float64MultiArray recurrent_debug_msg_;
  // Names of inputs to publish (external inputs, excluding LSTM feedback states).
  std::vector<std::string> debug_obs_input_names_;
  // inputs_ indices matching debug_obs_input_names_ for O(1) update-loop access.
  std::vector<size_t> debug_obs_input_indices_;
  // Names of recurrent (`_out`) outputs to publish, and their outputs_ indices.
  std::vector<std::string> debug_recurrent_output_names_;
  std::vector<size_t> debug_recurrent_output_indices_;
  // Name of the action output tensor to publish.
  std::string debug_action_output_name_{"arm_action"};

  // Helper methods.
  bool load_config();
  void create_topic_subscriptions();
  bool try_activate_core();
  bool topic_inputs_are_fresh(int64_t current_time_ns) const;
  bool resolve_reference_input_indices();
  bool write_reference_inputs();
  void invalidate_reference_inputs();
  void publish_debug_topics();
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

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

#include "isaac_ros_deploy_converters/nodes/input_builder_node.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "rclcpp_components/register_node_macro.hpp"

#include "isaac_deploy_core/inference_controller/config_parser.hpp"
#include "isaac_ros_deploy_converters/converters/message_to_tensor_converter_nodes.hpp"
#include "isaac_ros_deploy_converters/utils/tensor_list_utils.hpp"

namespace isaac_ros_deploy_converters
{

namespace
{

torch::Dtype parse_torch_dtype(const std::string & dtype_str)
{
  if (dtype_str == "float32") {return torch::kFloat32;}
  if (dtype_str == "float64") {return torch::kFloat64;}
  if (dtype_str == "int32") {return torch::kInt32;}
  if (dtype_str == "int64") {return torch::kInt64;}
  if (dtype_str == "int16") {return torch::kInt16;}
  if (dtype_str == "int8") {return torch::kInt8;}
  if (dtype_str == "uint8") {return torch::kUInt8;}
  throw std::runtime_error("Unsupported dtype string: " + dtype_str);
}

}  // namespace

InputBuilderNode::InputBuilderNode(const rclcpp::NodeOptions & options)
: Node("input_builder_node", options)
{
  initialize_input_converters();
  MessageToTensorConverterRegistry::instance().register_converter(
    "tensor", "isaac_ros_tensor_list_interfaces/msg/TensorList",
    [](const std::string & source) {
      return std::make_shared<TensorListConverter>(source);
    });

  declare_parameter<std::string>("config_path", "");
  declare_parameter<double>("publish_rate", 50.0);
  declare_parameter<std::string>("output_topic", "input_tensors");

  config_path_ = get_parameter("config_path").as_string();
  publish_rate_ = get_parameter("publish_rate").as_double();

  if (config_path_.empty()) {
    throw std::runtime_error("InputBuilderNode: config_path parameter is required");
  }

  configure();
}

void InputBuilderNode::configure()
{
  YAML::Node config;
  try {
    config = YAML::LoadFile(config_path_);
  } catch (const std::exception & e) {
    throw std::runtime_error("Failed to load config file: " + std::string(e.what()));
  }

  auto graph_result = isaac_deploy_core::parse_graph_config(config);
  if (!graph_result) {
    throw std::runtime_error(
      "Failed to parse config: " + graph_result.error().message);
  }

  auto model_config_result = isaac_deploy_core::merge_graph_to_model_config(*graph_result, config);
  if (!model_config_result) {
    throw std::runtime_error(
      "Failed to merge graph config: " + model_config_result.error().message);
  }

  // Collect feedback target names and source mapping from all models in the graph.
  std::unordered_set<std::string> feedback_target_names;
  std::unordered_map<std::string, std::string> feedback_source_map;  // input_name -> output_name
  for (const auto & [model_name, model_config] : graph_result->models) {
    for (const auto & [output_name, input_names] : model_config.feedback_connections) {
      for (const auto & name : input_names) {
        feedback_target_names.insert(name);
        feedback_source_map[name] = output_name;
      }
    }
  }

  // Ensure feedback inputs and connections are present in merged config.
  // For single-model, merge preserves them; for multi-model, it strips them.
  // Re-adding unconditionally is safe: connections are map-assigned (idempotent),
  // and inputs are filtered first to avoid duplicates.
  //
  // NOTE: We must build the complete new_inputs list before assigning to
  // model_config_result->inputs because YAML::Node operator= mutates through
  // aliases — the single-model path shares the underlying node with graph_result.
  if (!feedback_target_names.empty()) {
    YAML::Node new_inputs(YAML::NodeType::Sequence);
    for (const auto & input : model_config_result->inputs) {
      if (!feedback_target_names.contains(input["name"].as<std::string>())) {
        new_inputs.push_back(input);
      }
    }

    for (const auto & [model_name, model_config] : graph_result->models) {
      for (const auto & [output_name, input_names] : model_config.feedback_connections) {
        model_config_result->feedback_connections[output_name] = input_names;
      }
      for (const auto & input : model_config.inputs) {
        if (feedback_target_names.contains(input["name"].as<std::string>())) {
          new_inputs.push_back(input);
        }
      }
    }

    model_config_result->inputs = new_inputs;
  }

  const auto builder_config_result =
    isaac_deploy_core::InputBuilder::Config::create_from_model_config(*model_config_result);
  if (!builder_config_result) {
    throw std::runtime_error(
      "Failed to parse InputBuilder config: " + builder_config_result.error().message);
  }

  auto builder_result = isaac_deploy_core::InputBuilder::create(*builder_config_result);
  if (!builder_result) {
    throw std::runtime_error(
      "Failed to create InputBuilder: " + builder_result.error().message);
  }
  input_builder_ = std::move(*builder_result);

  // Collect feedback dtype info for zero-initialization.
  // Use feedback_source_map to resolve each feedback input's source (the output
  // name it connects to), since "source" is not a raw YAML field.
  std::unordered_map<std::string, torch::Dtype> feedback_dtypes;
  for (const auto & input : model_config_result->inputs) {
    const auto name = input["name"].as<std::string>();
    if (const auto it = feedback_source_map.find(name); it != feedback_source_map.end()) {
      feedback_dtypes[it->second] = parse_torch_dtype(input["dtype"].as<std::string>("float32"));
    }
  }

  create_subscription_groups(feedback_dtypes);

  // Create single output publisher for the bundled TensorList.
  const auto output_topic = get_parameter("output_topic").as_string();
  output_pub_ =
    create_publisher<isaac_ros_tensor_list_interfaces::msg::TensorList>(output_topic, 10);
  RCLCPP_INFO(get_logger(), "Publishing bundled TensorList on '%s'", output_topic.c_str());

  const auto period = std::chrono::duration<double>(1.0 / publish_rate_);
  timer_ = create_wall_timer(
    period,
    std::bind(&InputBuilderNode::timer_callback, this));

  RCLCPP_INFO(
    get_logger(), "InputBuilderNode configured with %zu subscription groups",
    subscription_groups_.size());
}

void InputBuilderNode::create_subscription_groups(
  const std::unordered_map<std::string, torch::Dtype> & feedback_dtypes)
{
  const auto source_to_kind = input_builder_->get_source_to_kind_map();
  auto & registry = MessageToTensorConverterRegistry::instance();

  // Helper: resolve the ROS topic for a given source name via parameter.
  auto resolve_topic = [this](const std::string & source) -> std::string {
      const std::string param_name = "source_to_topic." + source;
      declare_parameter<std::string>(param_name, source);
      return get_parameter(param_name).as_string();
    };

  // Helper: resolve the ROS message type override for a given source via parameter.
  auto resolve_message_type = [this](const std::string & source) -> std::string {
      const std::string param_name = "source_message_type." + source;
      declare_parameter<std::string>(param_name, "");
      const auto value = get_parameter(param_name).as_string();
      return value;
    };

  // Helper: add a converter to the appropriate group, creating a new group if needed.
  std::unordered_map<std::string, std::unique_ptr<SubscriptionGroup>> groups;
  auto add_to_group = [&](
    const std::string & source, const std::string & topic,
    std::shared_ptr<MessageToTensorConverter> converter,
    std::optional<torch::Tensor> initial_value = std::nullopt) {
      const std::string message_type = converter->get_message_type();
      auto it = groups.find(topic);
      if (it == groups.end()) {
        auto group = std::make_unique<SubscriptionGroup>();
        group->topic = topic;
        group->message_type = message_type;
        group->converters.push_back({source, converter, initial_value});
        groups[topic] = std::move(group);
        return;
      }
      if (it->second->message_type != message_type) {
        RCLCPP_ERROR(
          get_logger(),
          "Message type conflict for topic '%s': '%s' vs '%s' (source '%s')",
          topic.c_str(), it->second->message_type.c_str(),
          message_type.c_str(), source.c_str());
        return;
      }
      it->second->converters.push_back({source, converter, initial_value});
    };

  // External inputs: create converters from registry by kind.
  // Inputs with no declared kind (e.g. dangling tensor inputs in clean leapp
  // exports such as the diffusion-policy initial_noise seed) fall back to the
  // raw 'tensor' kind, which subscribes to a TensorList topic.
  for (const auto & [source, raw_kind] : source_to_kind) {
    const std::string kind = raw_kind.empty() ? "tensor" : raw_kind;
    const auto message_type = resolve_message_type(source);
    auto converter = registry.create_for_kind(kind, source, message_type);
    if (!converter) {
      RCLCPP_WARN(
        get_logger(), "No converter found for kind '%s' (source '%s'), skipping",
        kind.c_str(), source.c_str());
      continue;
    }
    add_to_group(source, resolve_topic(source), converter);
  }

  // Feedback inputs: subscribe to inference output topic with zero-initialized tensors.
  const auto feedback_names = input_builder_->get_feedback_input_names();
  const auto feedback_shapes = input_builder_->get_feedback_input_shapes();
  const std::string feedback_topic = "output_tensors";

  for (size_t i = 0; i < feedback_names.size(); ++i) {
    auto converter = std::make_shared<TensorListConverter>(feedback_names[i]);
    const auto dtype_it = feedback_dtypes.find(feedback_names[i]);
    const auto dtype = dtype_it != feedback_dtypes.end() ? dtype_it->second : torch::kFloat32;
    auto zero_tensor = torch::zeros(feedback_shapes[i], dtype);
    add_to_group(feedback_names[i], feedback_topic, converter, zero_tensor);
  }

  // Create subscriptions for each group.
  for (auto & [topic, group] : groups) {
    auto callback = [this, grp = group.get()](std::shared_ptr<rclcpp::SerializedMessage> msg) {
        std::lock_guard<std::mutex> lock(grp->mutex);
        grp->latest_msg = msg;
        grp->receive_time = now();
      };

    group->subscription = create_generic_subscription(
      topic, group->message_type, rclcpp::QoS(10), callback);

    std::string names_str;
    for (const auto & entry : group->converters) {
      if (!names_str.empty()) {names_str += ", ";}
      names_str += entry.source;
    }

    RCLCPP_INFO(
      get_logger(), "Subscribed to '%s' [%s] for sources: %s",
      topic.c_str(), group->message_type.c_str(), names_str.c_str());

    subscription_groups_.push_back(std::move(group));
  }
}

void InputBuilderNode::timer_callback()
{
  const rclcpp::Time current_stamp = now();

  // Convert messages from all subscription groups.
  isaac_deploy_core::TensorDict all_converted;
  bool all_groups_ready = true;

  for (auto & group : subscription_groups_) {
    std::shared_ptr<rclcpp::SerializedMessage> msg;
    {
      std::lock_guard<std::mutex> lock(group->mutex);
      msg = group->latest_msg;
    }

    if (!msg) {
      const bool has_initial = std::all_of(
        group->converters.begin(), group->converters.end(),
        [](const auto & e) {return e.initial_value.has_value();});
      if (has_initial) {
        for (const auto & entry : group->converters) {
          all_converted[entry.source] = *entry.initial_value;
        }
      } else {
        all_groups_ready = false;
      }
      continue;
    }

    for (const auto & entry : group->converters) {
      all_converted[entry.source] = entry.converter->convert(msg);
    }
  }

  // Activate InputBuilder on first complete set of inputs.
  if (!activated_ && input_builder_) {
    if (!all_groups_ready) {
      for (const auto & group : subscription_groups_) {
        if (group->latest_msg) {
          continue;
        }
        // Skip groups with initial values (e.g., feedback) — they don't block activation.
        const bool has_initial = std::all_of(
          group->converters.begin(), group->converters.end(),
          [](const auto & e) {return e.initial_value.has_value();});
        if (has_initial) {
          continue;
        }
        if (group->subscription->get_publisher_count() == 0) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Waiting for publisher on topic '%s' [%s] — no publishers connected",
            group->topic.c_str(), group->message_type.c_str());
        } else {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Waiting for first message on topic '%s' [%s]",
            group->topic.c_str(), group->message_type.c_str());
        }
      }
      return;
    }

    // Build TensorSpec map from non-feedback converters (keyed by source).
    std::unordered_map<std::string, isaac_deploy_core::TensorSpec> input_specs_map;
    for (const auto & group : subscription_groups_) {
      for (const auto & entry : group->converters) {
        input_specs_map[entry.source] = entry.converter->get_tensor_spec();
      }
    }

    // Build activation vectors ordered by get_unique_source_names().
    const auto required_sources = input_builder_->get_unique_source_names();
    builder_inputs_.clear();
    builder_input_specs_.clear();
    for (const auto & source : required_sources) {
      const auto tensor_it = all_converted.find(source);
      const auto tensor = tensor_it != all_converted.end() ? tensor_it->second : torch::Tensor{};
      builder_inputs_.push_back({source, 0, tensor});

      const auto spec_it = input_specs_map.find(source);
      builder_input_specs_.push_back(
        spec_it != input_specs_map.end() ? spec_it->second : isaac_deploy_core::TensorSpec{});
    }

    auto result = input_builder_->activate(builder_inputs_, builder_input_specs_);
    if (!result) {
      RCLCPP_ERROR(
        get_logger(), "Failed to activate InputBuilder: %s",
        result.error().message.c_str());
      return;
    }

    for (const auto & key : input_builder_->get_output_names()) {
      nn_inputs_[key] = torch::empty({});
    }

    activated_ = true;
    RCLCPP_INFO(get_logger(), "InputBuilder activated");
  }

  if (!activated_) {
    return;
  }

  validate_input_synchronization(current_stamp);

  // Update builder_inputs_ with latest converted tensors (positionally aligned).
  for (size_t i = 0; i < builder_inputs_.size(); ++i) {
    auto it = all_converted.find(builder_inputs_[i].name);
    if (it != all_converted.end()) {
      builder_inputs_[i].tensor = it->second;
    }
  }

  auto result = input_builder_->advance(builder_inputs_, nn_inputs_);
  if (!result) {
    RCLCPP_ERROR(
      get_logger(), "InputBuilder advance failed: %s",
      result.error().message.c_str());
    return;
  }

  // Build a single TensorList with all model inputs.
  isaac_ros_tensor_list_interfaces::msg::TensorList msg;
  msg.header.stamp = current_stamp;

  for (const auto & [name, tensor] : nn_inputs_) {
    msg.tensors.push_back(torch_to_tensor_msg(tensor, name));
  }

  output_pub_->publish(msg);
}

void InputBuilderNode::validate_input_synchronization(const rclcpp::Time & current_time)
{
  if (subscription_groups_.empty()) {
    return;
  }

  // Collect receive times from all groups that have received messages.
  std::vector<std::pair<std::string, rclcpp::Time>> receive_times;
  for (const auto & group : subscription_groups_) {
    rclcpp::Time recv_time;
    {
      std::lock_guard<std::mutex> lock(group->mutex);
      if (!group->latest_msg) {
        continue;  // No message received yet.
      }
      recv_time = group->receive_time;
    }
    receive_times.emplace_back(group->topic, recv_time);
  }

  if (receive_times.size() < 2) {
    return;  // Need at least 2 inputs to compare.
  }

  // Find min and max receive times.
  auto [min_it, max_it] = std::minmax_element(
    receive_times.begin(), receive_times.end(),
    [](const auto & a, const auto & b) {
      return a.second < b.second;
    });

  const int64_t spread_ns = max_it->second.nanoseconds() - min_it->second.nanoseconds();

  // Also check staleness: how old is the oldest input relative to current time.
  const int64_t staleness_ns = current_time.nanoseconds() - min_it->second.nanoseconds();

  constexpr int64_t kSyncToleranceNs = 50'000'000;  // 50ms tolerance for input sync.
  constexpr int64_t kStalenessToleranceNs = 100'000'000;  // 100ms tolerance for staleness.

  if (spread_ns > kSyncToleranceNs) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Input timestamps out of sync: '%s' and '%s' differ by %.1f ms (threshold: %.1f ms)",
      min_it->first.c_str(),
      max_it->first.c_str(),
      static_cast<double>(spread_ns) / 1e6,
      static_cast<double>(kSyncToleranceNs) / 1e6);
  }

  if (staleness_ns > kStalenessToleranceNs) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Input '%s' is stale: last received %.1f ms ago (threshold: %.1f ms)",
      min_it->first.c_str(),
      static_cast<double>(staleness_ns) / 1e6,
      static_cast<double>(kStalenessToleranceNs) / 1e6);
  }
}

}  // namespace isaac_ros_deploy_converters

RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_deploy_converters::InputBuilderNode)

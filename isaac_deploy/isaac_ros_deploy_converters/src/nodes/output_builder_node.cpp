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

#include "isaac_ros_deploy_converters/nodes/output_builder_node.hpp"

#include "yaml-cpp/yaml.h"

#include "rclcpp_components/register_node_macro.hpp"

#include "isaac_deploy_core/inference_controller/config_parser.hpp"
#include "isaac_ros_deploy_converters/utils/tensor_list_utils.hpp"
#include "isaac_ros_deploy_interfaces/msg/joint_command_trajectory.hpp"

namespace isaac_ros_deploy_converters
{

namespace
{

/// Default topic for each ROS message type emitted by an output converter.
/// Callers set `output_to_topic.<name>` to override (e.g., routing two
/// models to distinct topics).
std::string default_topic_for_message_type(
  const std::string & message_type, const std::string & output_name)
{
  if (message_type == "isaac_ros_deploy_interfaces/msg/JointCommand") {
    return "joint_commands";
  }
  if (message_type == "isaac_ros_deploy_interfaces/msg/JointCommandTrajectory") {
    return "joint_commands_trajectory";
  }
  if (message_type == "isaac_ros_deploy_interfaces/msg/BodyCommand") {
    return "body_commands";
  }
  if (message_type == "geometry_msgs/msg/Twist") {
    return "cmd_vel";
  }
  // Unknown type: fall back to the output name so unrecognised converters
  // still get a unique topic.
  return output_name;
}

}  // namespace

OutputBuilderNode::OutputBuilderNode(const rclcpp::NodeOptions & options)
: Node("output_builder_node", options)
{
  initialize_output_converters();

  declare_parameter<std::string>("config_path", "");
  declare_parameter<std::string>("input_topic", "output_tensors");

  config_path_ = get_parameter("config_path").as_string();

  if (config_path_.empty()) {
    throw std::runtime_error("OutputBuilderNode: config_path parameter is required");
  }

  configure();
}

void OutputBuilderNode::configure()
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

  auto sections_result = isaac_deploy_core::merge_graph_to_model_config(*graph_result, config);
  if (!sections_result) {
    throw std::runtime_error(
      "Failed to merge graph config: " + sections_result.error().message);
  }

  // Filter outputs to only those that have registered output converters
  // (i.e., dangling outputs that produce typed ROS messages). State/feedback
  // outputs are ignored.
  auto & registry = OutputConverterRegistry::instance();
  YAML::Node filtered_outputs(YAML::NodeType::Sequence);
  for (const auto & output_yaml : sections_result->outputs) {
    if (!output_yaml["kind"].IsDefined()) {
      continue;
    }
    auto kind = output_yaml["kind"].as<std::string>();
    if (!output_yaml["shape"].IsDefined()) {
      continue;
    }
    const auto shape = output_yaml["shape"].as<std::vector<int64_t>>();
    if (registry.create_for_kind(kind, shape)) {
      filtered_outputs.push_back(output_yaml);
    } else {
      RCLCPP_WARN(
        get_logger(),
        "Output '%s' (kind: %s) has no registered converter for shape rank %zu, skipping",
        output_yaml["name"].as<std::string>().c_str(), kind.c_str(), shape.size());
    }
  }
  sections_result->outputs = filtered_outputs;

  auto builder_config_result =
    isaac_deploy_core::OutputBuilder::Config::create_from_model_config(
    *sections_result);
  if (!builder_config_result) {
    throw std::runtime_error(
      "Failed to parse OutputBuilder config: " + builder_config_result.error().message);
  }

  // Read semantic.scene.dt from the raw YAML so the builder can stamp step_dt
  // on JointCommandTrajectory messages. Defaults to 0.02 s (50 Hz) if unset.
  if (config["semantic"] && config["semantic"]["scene"] &&
    config["semantic"]["scene"]["dt"].IsDefined())
  {
    builder_config_result->policy_dt_seconds =
      config["semantic"]["scene"]["dt"].as<double>();
  }

  auto builder_result = isaac_deploy_core::OutputBuilder::create(*builder_config_result);
  if (!builder_result) {
    throw std::runtime_error(
      "Failed to create OutputBuilder: " + builder_result.error().message);
  }
  output_builder_ = std::move(*builder_result);

  // Pre-allocate output tensors and output map from OutputBuilder metadata.
  auto output_names = output_builder_->get_output_names();
  auto output_shapes = output_builder_->get_output_shapes();
  for (size_t i = 0; i < output_names.size(); ++i) {
    isaac_deploy_core::NamedTensor tensor;
    tensor.name = output_names[i];
    tensor.timestamp_ns = 0;
    tensor.tensor = torch::zeros(output_shapes[i], torch::kFloat32);
    outputs_.push_back(std::move(tensor));
    output_name_to_index_[output_names[i]] = i;
  }
  for (const auto & output : outputs_) {
    output_map_[output.name] = output;
  }

  // Activate OutputBuilder (no reordering in nodes path — empty specs).
  std::vector<isaac_deploy_core::TensorSpec> output_specs(outputs_.size());
  auto activate_result = output_builder_->activate(output_specs, outputs_);
  if (!activate_result) {
    throw std::runtime_error(
      "Failed to activate OutputBuilder: " + activate_result.error().message);
  }

  create_publication_groups();

  // Create single input subscription for the bundled TensorList.
  const auto input_topic = get_parameter("input_topic").as_string();
  input_sub_ = create_subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>(
    input_topic, 10,
    std::bind(&OutputBuilderNode::on_tensor_list, this, std::placeholders::_1));

  RCLCPP_INFO(
    get_logger(), "OutputBuilderNode configured with %zu publication groups, "
    "subscribing to '%s'",
    publication_groups_.size(), input_topic.c_str());
}

void OutputBuilderNode::create_publication_groups()
{
  auto & registry = OutputConverterRegistry::instance();
  auto output_to_kind = output_builder_->get_output_to_kind_map();
  auto output_names = output_builder_->get_output_names();
  auto output_shapes = output_builder_->get_output_shapes();

  // For each output, create a converter and resolve its topic.
  // Group by topic (validate message_type consistency within group).
  std::unordered_map<std::string, std::unique_ptr<PublicationGroup>> groups;

  for (size_t i = 0; i < output_names.size(); ++i) {
    const auto & name = output_names[i];
    const auto & kind = output_to_kind[name];
    const auto & shape = output_shapes[i];

    auto converter = registry.create_for_kind(kind, shape);
    if (!converter) {
      RCLCPP_WARN(
        get_logger(), "No converter for kind '%s', skipping output '%s'",
        kind.c_str(), name.c_str());
      continue;
    }

    // Default topic follows the converter's ROS message type; override
    // via `output_to_topic.<name>`.
    const std::string default_topic = default_topic_for_message_type(
      converter->get_message_type(), name);
    const std::string param_name = "output_to_topic." + name;
    declare_parameter<std::string>(param_name, default_topic);
    std::string topic = get_parameter(param_name).as_string();

    auto it = groups.find(topic);
    if (it == groups.end()) {
      auto group = std::make_unique<PublicationGroup>();
      group->topic = topic;
      group->message_type = converter->get_message_type();
      group->converters.push_back({name, converter});
      groups[topic] = std::move(group);
    } else {
      // Validate message type consistency.
      if (it->second->message_type != converter->get_message_type()) {
        RCLCPP_ERROR(
          get_logger(),
          "Message type conflict for topic '%s': '%s' vs '%s' (output '%s', kind '%s')",
          topic.c_str(), it->second->message_type.c_str(),
          converter->get_message_type().c_str(), name.c_str(), kind.c_str());
        continue;
      }
      it->second->converters.push_back({name, converter});
    }
  }

  // Create publishers.
  for (auto & [topic, group] : groups) {
    group->publisher = create_generic_publisher(
      group->topic, group->message_type, rclcpp::QoS(10));

    // Build output list for log message.
    std::string outputs_str;
    for (const auto & entry : group->converters) {
      if (!outputs_str.empty()) {outputs_str += ", ";}
      outputs_str += entry.output_name;
    }

    RCLCPP_INFO(
      get_logger(), "Publishing to '%s' [%s] for outputs: %s",
      topic.c_str(), group->message_type.c_str(), outputs_str.c_str());

    publication_groups_.push_back(std::move(group));
  }
}

void OutputBuilderNode::on_tensor_list(
  const isaac_ros_tensor_list_interfaces::msg::TensorList::SharedPtr msg)
{
  // Extract tensors matching our expected outputs from the TensorList.
  bool all_received = true;
  isaac_deploy_core::TensorDict nn_outputs;

  for (const auto & tensor_msg : msg->tensors) {
    auto it = output_name_to_index_.find(tensor_msg.name);
    if (it != output_name_to_index_.end()) {
      nn_outputs[tensor_msg.name] = tensor_msg_to_torch(tensor_msg);
    }
  }

  // Check if all expected outputs were present.
  for (const auto & [name, idx] : output_name_to_index_) {
    if (!nn_outputs.contains(name)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Output tensor '%s' not found in TensorList", name.c_str());
      all_received = false;
    }
  }

  if (!all_received) {
    return;
  }

  int64_t timestamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();

  // Run OutputBuilder.
  auto result = output_builder_->advance(nn_outputs, outputs_);
  if (!result) {
    RCLCPP_ERROR(
      get_logger(), "OutputBuilder advance failed: %s",
      result.error().message.c_str());
    return;
  }

  // Refresh pre-allocated output map.
  for (const auto & tensor : outputs_) {
    output_map_[tensor.name] = tensor;
  }

  // Publish to each publication group: create -> write all -> serialize -> publish.
  for (auto & group : publication_groups_) {
    auto typed_msg = group->converters.front().converter->create_message();

    // Stamp step_dt if this publication group emits a JointCommandTrajectory.
    if (group->converters.front().converter->get_message_type() ==
      "isaac_ros_deploy_interfaces/msg/JointCommandTrajectory")
    {
      auto & traj = *std::static_pointer_cast<
        isaac_ros_deploy_interfaces::msg::JointCommandTrajectory>(typed_msg);
      traj.step_dt = rclcpp::Duration::from_seconds(
        output_builder_->get_policy_dt_seconds());
    }

    bool group_ok = true;
    for (const auto & entry : group->converters) {
      auto it = output_map_.find(entry.output_name);
      if (it == output_map_.end()) {continue;}

      try {
        entry.converter->write(
          it->second.tensor,
          output_builder_->get_element_names(entry.output_name),
          typed_msg,
          timestamp_ns);
      } catch (const std::exception & e) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Converter for output '%s' threw: %s; skipping publication group",
          entry.output_name.c_str(), e.what());
        group_ok = false;
        break;
      }
    }

    if (!group_ok) {continue;}

    auto serialized = group->converters.front().converter->serialize(typed_msg);
    if (serialized) {
      group->publisher->publish(*serialized);
    } else {
      RCLCPP_ERROR(
        get_logger(), "Failed to serialize message for topic '%s'",
        group->topic.c_str());
    }
  }
}

}  // namespace isaac_ros_deploy_converters

RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_deploy_converters::OutputBuilderNode)

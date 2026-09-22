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

#include "isaac_ros_deploy_ros2_control/controllers/inference_controller.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "isaac_deploy_core/inference_controller/config_parser.hpp"
#include "isaac_deploy_core/inference_controller/safetensors_loader.hpp"
#include "isaac_ros_deploy_ros2_control/adapters/command_interface_adapter.hpp"
#include "isaac_ros_deploy_ros2_control/adapters/state_interface_adapter.hpp"
#include "isaac_ros_deploy_ros2_control/controllers/leapp_backend_mapping.hpp"
#include "isaac_ros_deploy_ros2_control/converters/command_interface_converter.hpp"
#include "isaac_ros_deploy_ros2_control/converters/state_interface_converter.hpp"
#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{
namespace
{

void set_source_interface_names(
  isaac_deploy_core::InputTermConfig & term,
  const std::vector<std::string> & interface_names)
{
  if (interface_names.empty()) {
    return;
  }
  while (term.element_names.size() < term.shape.size()) {
    term.element_names.insert(term.element_names.begin(), std::vector<std::string>{});
  }
  if (term.element_names.empty()) {
    term.element_names.push_back(interface_names);
  } else {
    term.element_names.front() = interface_names;
  }
}

template<typename NodeT>
std::vector<std::string> get_optional_string_array_parameter(
  const NodeT & node, const std::string & param_name)
{
  try {
    if (!node->has_parameter(param_name)) {
      node->template declare_parameter<std::vector<std::string>>(
        param_name, std::vector<std::string>{});
    }
    std::vector<std::string> value;
    if (node->get_parameter(param_name, value)) {
      return value;
    }
  } catch (const std::exception & e) {
    const std::string error = e.what();
    const bool is_not_set =
      error.find("got [not set]") != std::string::npos ||
      error.find("must be initialized") != std::string::npos;
    if (!is_not_set) {
      throw std::runtime_error(
              "parameter " + param_name + " must be a string array: " + error);
    }
    return {};
  }
  return {};
}

}  // namespace

InferenceController::InferenceController() = default;

controller_interface::CallbackReturn InferenceController::on_init()
{
  try {
    auto_declare<std::string>("config_path", "");
    auto_declare<std::string>("model_path", "");
    auto_declare<int>("decimation", 4);
    auto_declare<std::string>("command_prefix", "");
    auto_declare<std::string>("command_suffix", "");
    auto_declare<std::string>("joint_name_prefix", "");
    auto_declare<std::vector<std::string>>("topic_input_sources", {});
    auto_declare<double>("topic_input_timeout_ms", 0.0);
    auto_declare<bool>("publish_debug_topics", false);
    auto_declare<bool>("log_debug_to_console", false);
    auto_declare<std::string>("debug_action_output_name", "arm_action");
    auto_declare<std::vector<std::string>>("reference_input_sources", {});
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn InferenceController::on_configure(
  const rclcpp_lifecycle::State &)
{
  config_path_ = get_node()->get_parameter("config_path").as_string();
  model_path_ = get_node()->get_parameter("model_path").as_string();
  decimation_ = get_node()->get_parameter("decimation").as_int();
  command_prefix_ = get_node()->get_parameter("command_prefix").as_string();
  command_suffix_ = get_node()->get_parameter("command_suffix").as_string();
  joint_name_prefix_ = get_node()->get_parameter("joint_name_prefix").as_string();
  topic_input_sources_ =
    get_optional_string_array_parameter(get_node(), "topic_input_sources");
  reference_input_sources_ =
    get_optional_string_array_parameter(get_node(), "reference_input_sources");
  const double topic_input_timeout_ms =
    get_node()->get_parameter("topic_input_timeout_ms").as_double();

  if (config_path_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "config_path parameter is required");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (decimation_ <= 0) {
    RCLCPP_ERROR(get_node()->get_logger(), "decimation must be positive");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (!std::isfinite(topic_input_timeout_ms) || topic_input_timeout_ms < 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(), "topic_input_timeout_ms must be non-negative");
    return controller_interface::CallbackReturn::ERROR;
  }
  topic_input_timeout_ns_ =
    static_cast<int64_t>(std::llround(topic_input_timeout_ms * 1e6));

  reference_inputs_.clear();
  std::unordered_set<std::string> reference_input_source_set;
  for (const auto & source : reference_input_sources_) {
    if (source.empty()) {
      RCLCPP_ERROR(get_node()->get_logger(), "reference_input_sources cannot contain empty values");
      return controller_interface::CallbackReturn::ERROR;
    }
    if (!reference_input_source_set.insert(source).second) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Duplicate reference input source: %s", source.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    auto interface_names = get_optional_string_array_parameter(
      get_node(), "reference_input_interfaces." + source);
    if (interface_names.empty()) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "reference_input_interfaces.%s must be non-empty when source is forwarded",
        source.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    reference_inputs_.push_back(
        {
          .source = source,
          .command_interfaces = std::move(interface_names),
      });
  }

  // Initialize the converter registries needed by this controller.
  initialize_state_interface_converters();
  initialize_command_interface_converters();
  isaac_ros_deploy_converters::initialize_input_converters();

  if (!load_config()) {
    return controller_interface::CallbackReturn::ERROR;
  }

  create_topic_subscriptions();

  // Latched `~/is_active` publisher — emits once when `core_activated_` first
  // flips true.  Subscribers created after that still receive the message
  // thanks to TRANSIENT_LOCAL durability.
  is_active_publisher_ = get_node()->create_publisher<std_msgs::msg::Bool>(
    "~/is_active", rclcpp::QoS(1).transient_local().reliable());

  publish_debug_topics_ = get_node()->get_parameter("publish_debug_topics").as_bool();
  log_debug_to_console_ = get_node()->get_parameter("log_debug_to_console").as_bool();
  debug_action_output_name_ =
    get_node()->get_parameter("debug_action_output_name").as_string();

  if (publish_debug_topics_) {
    auto qos = rclcpp::QoS(10).best_effort();
    obs_debug_publisher_ = std::make_shared<DebugPublisher>(
      get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
        "~/debug_observation", qos));
    action_debug_publisher_ = std::make_shared<DebugPublisher>(
      get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
        "~/debug_action", qos));
    recurrent_debug_publisher_ = std::make_shared<DebugPublisher>(
      get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
        "~/debug_recurrent_state", qos));
    RCLCPP_INFO(
      get_node()->get_logger(),
      "Debug topic publishing enabled: ~/debug_observation, ~/debug_action, "
      "~/debug_recurrent_state");
  }

  RCLCPP_INFO(
    get_node()->get_logger(), "Configured InferenceController with config: %s",
    config_path_.c_str());

  return controller_interface::CallbackReturn::SUCCESS;
}

bool InferenceController::load_config()
{
  try {
    YAML::Node yaml = YAML::LoadFile(config_path_);

    // Extract graph config and merge into single ModelConfig.
    // The ros2_control runtime only supports single-model graphs.
    auto graph_result = isaac_deploy_core::parse_graph_config(yaml);
    if (!graph_result.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Failed to parse config: %s",
        std::string(graph_result.error().message).c_str());
      return false;
    }
    if (graph_result->models.size() != 1) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "ros2_control InferenceController only supports single-model graphs, "
        "got %zu models", graph_result->models.size());
      return false;
    }
    auto sections_result = isaac_deploy_core::merge_graph_to_model_config(*graph_result, yaml);
    if (!sections_result.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Failed to merge graph config: %s",
        std::string(sections_result.error().message).c_str());
      return false;
    }
    auto sections = std::move(*sections_result);

    // Parse InputBuilder config (for the core controller).
    auto inputs_result =
      isaac_deploy_core::InputBuilder::Config::create_from_model_config(sections);
    if (!inputs_result.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Failed to parse inputs config: %s",
        std::string(inputs_result.error().message).c_str());
      return false;
    }
    for (auto & term : inputs_result.value().terms) {
      set_source_interface_names(
        term,
        get_optional_string_array_parameter(get_node(), "input_interfaces." + term.source));
    }

    // Classify input terms using the state interface converter registry.
    const auto & state_registry = StateInterfaceConverterRegistry::instance();
    const std::unordered_set<std::string> topic_input_sources(
      topic_input_sources_.begin(), topic_input_sources_.end());
    std::vector<isaac_deploy_core::InputTermConfig> hardware_input_configs;
    topic_source_configs_.clear();

    for (const auto & term : inputs_result.value().terms) {
      if (!term.output_key.empty()) {
        // Feedback input — handled internally by the core controller.
        RCLCPP_INFO(
          get_node()->get_logger(),
          "Input '%s' (kind: %s) is feedback, handled by core library",
          term.name.c_str(), term.kind.c_str());
      } else if (!topic_input_sources.contains(term.source) && state_registry.contains(term.kind)) {
        // Hardware input — read from state interfaces.
        hardware_input_configs.push_back(term);
        RCLCPP_INFO(
          get_node()->get_logger(), "Found hardware input: %s (kind: %s, source: %s)",
          term.name.c_str(), term.kind.c_str(), term.source.c_str());
      } else {
        // Topic input — subscribe to ROS topic via converter.
        topic_source_configs_.push_back({term.source, term.kind, term.shape});
        RCLCPP_INFO(
          get_node()->get_logger(), "Found topic input: %s (kind: %s, source: %s)",
          term.name.c_str(), term.kind.c_str(), term.source.c_str());
      }
    }

    // Create state interface adapter.
    state_adapter_ = std::make_unique<StateInterfaceAdapter>(
      hardware_input_configs, joint_name_prefix_);

    auto outputs_result =
      isaac_deploy_core::OutputBuilder::Config::create_from_model_config(sections);
    if (!outputs_result.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Failed to parse outputs config: %s",
        std::string(outputs_result.error().message).c_str());
      return false;
    }

    // Build output configs. Only pass hardware outputs (those with a registered
    // converter) to the command adapter.
    const auto & cmd_registry = CommandInterfaceConverterRegistry::instance();
    std::vector<isaac_deploy_core::OutputTermConfig> output_configs;
    all_output_terms_.clear();
    for (const auto & term : outputs_result.value().terms) {
      all_output_terms_.push_back({term.name, term.shape});
      if (cmd_registry.contains(term.kind)) {
        output_configs.push_back(term);
        RCLCPP_INFO(
          get_node()->get_logger(), "Found hardware output: %s (kind: %s)",
          term.name.c_str(), term.kind.c_str());
      } else {
        RCLCPP_INFO(
          get_node()->get_logger(), "Found non-hardware output: %s (kind: %s)",
          term.name.c_str(), term.kind.c_str());
      }
    }

    // Create command interface adapter.
    command_adapter_ = std::make_unique<CommandInterfaceAdapter>(
      output_configs, command_prefix_, command_suffix_, joint_name_prefix_);

    // Parse runner config (from config sections, with parameter overrides).
    isaac_deploy_core::InferenceRunner::Config runner_config;
    runner_config.model_path = sections.model_path;
    const auto runner_type = leapp_backend_to_runner_type(sections.backend);
    if (!runner_type.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "%s",
        runner_type.error().message.c_str());
      return false;
    }
    runner_config.runner_type = *runner_type;
    // Command-line parameter overrides YAML.
    if (!model_path_.empty()) {
      runner_config.model_path = model_path_;
    }

    if (runner_config.model_path.empty()) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Model path must be specified either in YAML parameters section "
        "or as 'model_path' parameter");
      return false;
    }

    // Resolve relative model paths against the config file directory.
    std::filesystem::path model_fs_path(runner_config.model_path);
    if (!model_fs_path.is_absolute()) {
      runner_config.model_path =
        (std::filesystem::path(config_path_).parent_path() / model_fs_path).string();
    }

    RCLCPP_INFO(
      get_node()->get_logger(), "Using model: %s (runner: %s)",
      runner_config.model_path.c_str(), runner_config.runner_type.c_str());

    // Create the core inference controller.
    isaac_deploy_core::InferenceControllerConfig config;
    config.inputs = std::move(inputs_result.value());
    config.outputs = std::move(outputs_result.value());
    config.runner = runner_config;

    auto feedback_initial_values_result = isaac_deploy_core::load_feedback_initial_values(
      *graph_result, config_path_);
    if (!feedback_initial_values_result.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to load feedback initial values: %s",
        feedback_initial_values_result.error().message.c_str());
      return false;
    }
    config.feedback_initial_values = std::move(*feedback_initial_values_result);

    auto controller_result = isaac_deploy_core::InferenceController::create(std::move(config));
    if (!controller_result.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Failed to create InferenceController: %s",
        std::string(controller_result.error().message).c_str());
      return false;
    }

    inference_controller_ =
      std::make_unique<isaac_deploy_core::InferenceController>(
      std::move(controller_result.value()));

    // Trigger TensorRT JIT compilation now, in on_configure(), before the real-time
    // control loop starts.  Without this, the first runner_->run() call inside
    // update() would block for ~400 ms, freezing the entire control thread
    // (including the gantry).  on_configure() runs outside the real-time thread
    // so the stall is harmless here.
    RCLCPP_INFO(get_node()->get_logger(), "Running inference warmup (TensorRT compilation)...");
    auto warmup_result = inference_controller_->warmup();
    if (!warmup_result.has_value()) {
      RCLCPP_WARN(get_node()->get_logger(), "Inference warmup failed: %s",
        std::string(warmup_result.error().message).c_str());
    } else {
      RCLCPP_INFO(get_node()->get_logger(), "Inference warmup complete.");
    }

    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to load config: %s", e.what());
    return false;
  }
}

void InferenceController::create_topic_subscriptions()
{
  topic_groups_.clear();
  auto & registry = isaac_ros_deploy_converters::MessageToTensorConverterRegistry::instance();

  // Resolve the ROS topic for a given source name via parameter.
  // Parameters may already be declared if set via launch file (as nested dict).
  auto resolve_topic = [this](const std::string & source) -> std::string {
      const std::string param_name = "source_to_topic." + source;
      if (!get_node()->has_parameter(param_name)) {
        get_node()->declare_parameter<std::string>(param_name, source);
      }
      return get_node()->get_parameter(param_name).as_string();
    };

  // Resolve the ROS message type override for a given source via parameter.
  // Parameters come from the controller manager YAML config (e.g., controller_manager.yaml),
  // following the same pattern as source_to_topic.<source>.
  auto resolve_message_type = [this](const std::string & source) -> std::string {
      const std::string param_name = "source_message_type." + source;
      if (!get_node()->has_parameter(param_name)) {
        get_node()->declare_parameter<std::string>(param_name, "");
      }
      return get_node()->get_parameter(param_name).as_string();
    };

  // Group topic sources by resolved topic to share subscriptions.
  std::unordered_map<std::string, std::unique_ptr<TopicGroup>> groups_by_topic;

  size_t skipped_converters = 0;
  for (const auto & topic_config : topic_source_configs_) {
    const auto message_type_override = resolve_message_type(topic_config.source);
    auto converter = registry.create_for_kind(
      topic_config.kind, topic_config.source, message_type_override);
    if (!converter) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "No converter found for kind '%s' (source '%s'), skipping — "
        "this input will use zero tensors",
        topic_config.kind.c_str(), topic_config.source.c_str());
      ++skipped_converters;
      continue;
    }

    const std::string topic = resolve_topic(topic_config.source);
    const std::string message_type = converter->get_message_type();

    auto it = groups_by_topic.find(topic);
    if (it == groups_by_topic.end()) {
      auto group = std::make_unique<TopicGroup>();
      group->topic = topic;
      group->message_type = message_type;
      group->entries.push_back({topic_config.source, converter});
      groups_by_topic[topic] = std::move(group);
    } else {
      if (it->second->message_type != message_type) {
        RCLCPP_ERROR(
          get_node()->get_logger(),
          "Message type conflict for topic '%s': '%s' vs '%s' (source '%s')",
          topic.c_str(), it->second->message_type.c_str(),
          message_type.c_str(), topic_config.source.c_str());
        continue;
      }
      it->second->entries.push_back({topic_config.source, converter});
    }
  }

  if (skipped_converters > 0) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "%zu of %zu topic inputs have no converter and will use zero tensors",
      skipped_converters, topic_source_configs_.size());
  }

  // Create one generic subscription per topic group.
  for (auto & [topic, group] : groups_by_topic) {
    auto callback = [this, grp = group.get()](std::shared_ptr<rclcpp::SerializedMessage> msg) {
        auto sample = std::make_shared<TopicGroup::TopicSample>();
        sample->message = msg;
        sample->receive_time_ns = get_node()->now().nanoseconds();
        grp->rt_msg_buffer.writeFromNonRT(sample);
      };

    group->subscription = get_node()->create_generic_subscription(
      topic, group->message_type, rclcpp::QoS(10), callback);

    std::string sources_str;
    for (const auto & entry : group->entries) {
      if (!sources_str.empty()) {sources_str += ", ";}
      sources_str += entry.source;
    }

    RCLCPP_INFO(
      get_node()->get_logger(), "Subscribed to '%s' [%s] for sources: %s",
      topic.c_str(), group->message_type.c_str(), sources_str.c_str());

    topic_groups_.push_back(std::move(group));
  }
}

controller_interface::InterfaceConfiguration
InferenceController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  if (command_adapter_) {
    config.names = command_adapter_->get_required_command_interfaces();
    for (const auto & reference_input : reference_inputs_) {
      config.names.insert(
        config.names.end(),
        reference_input.command_interfaces.begin(),
        reference_input.command_interfaces.end());
    }
    RCLCPP_DEBUG(
      get_node()->get_logger(), "Requesting %zu command interfaces", config.names.size());
    for (const auto & name : config.names) {
      RCLCPP_DEBUG(get_node()->get_logger(), "  - %s", name.c_str());
    }
  }

  return config;
}

controller_interface::InterfaceConfiguration
InferenceController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  if (state_adapter_) {
    config.names = state_adapter_->get_required_state_interfaces();
    RCLCPP_DEBUG(
      get_node()->get_logger(), "Requesting %zu state interfaces", config.names.size());
    for (const auto & name : config.names) {
      RCLCPP_DEBUG(get_node()->get_logger(), "  - %s", name.c_str());
    }
  }

  return config;
}

controller_interface::CallbackReturn InferenceController::on_activate(
  const rclcpp_lifecycle::State &)
{
  if (!inference_controller_) {
    RCLCPP_ERROR(get_node()->get_logger(), "Inference controller not initialized");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Set up adapters with loaned interfaces.
  try {
    state_adapter_->set_state_interfaces(state_interfaces_);
    command_adapter_->set_command_interfaces(command_interfaces_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Failed to set up interface adapters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  // Build initial input tensors and specs (one entry per unique source).
  // Multiple state adapter entries can share a source (e.g., joint_pos for both
  // current-value and history terms). De-duplicate here.
  inputs_.clear();
  input_specs_.clear();
  hw_to_input_idx_.clear();
  source_to_input_idx_.clear();

  const int64_t timestamp_ns{0};
  std::unordered_map<std::string, size_t> source_seen;
  for (size_t i = 0; i < state_adapter_->get_num_hardware_inputs(); ++i) {
    const std::string source = state_adapter_->get_input_source(i);
    auto [it, inserted] = source_seen.try_emplace(source, inputs_.size());
    if (inserted) {
      source_to_input_idx_[source] = it->second;
      inputs_.push_back(
          {
            .name = source,
            .timestamp_ns = timestamp_ns,
            .tensor = torch::zeros(state_adapter_->get_shape(i), torch::kFloat32),
          });
      state_adapter_->read_tensor(i, inputs_.back().tensor);
      input_specs_.push_back(state_adapter_->get_tensor_spec(i));
    }
    hw_to_input_idx_.push_back(it->second);
  }

  // Add topic input placeholders (zero tensors, empty TensorSpecs).
  // TensorSpecs will be populated from converters during deferred activation.
  for (const auto & topic_config : topic_source_configs_) {
    auto [it, inserted] = source_seen.try_emplace(topic_config.source, inputs_.size());
    if (inserted) {
      source_to_input_idx_[topic_config.source] = inputs_.size();
      inputs_.push_back(
          {
            .name = topic_config.source,
            .timestamp_ns = timestamp_ns,
            .tensor = torch::zeros(topic_config.shape, torch::kFloat32),
          });
      input_specs_.push_back(isaac_deploy_core::TensorSpec{});
    }
  }

  try {
    if (!resolve_reference_input_indices()) {
      return controller_interface::CallbackReturn::ERROR;
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Failed to set up reference input forwarding: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  // Build output tensors for ALL output terms (hardware + state/feedback).
  // The core's OutputBuilder expects entries for every output, including state outputs
  // that are only used internally for the feedback loop.
  outputs_.clear();
  output_specs_.clear();
  output_name_to_idx_.clear();

  for (const auto & term : all_output_terms_) {
    output_name_to_idx_[term.name] = outputs_.size();
    outputs_.push_back(
        {
          .name = term.name,
          .timestamp_ns = timestamp_ns,
          .tensor = torch::zeros(term.shape, torch::kFloat32),
        });
    output_specs_.push_back(isaac_deploy_core::TensorSpec{});
  }

  // Set TensorSpecs for hardware outputs from command adapter.
  for (size_t i = 0; i < command_adapter_->get_num_outputs(); ++i) {
    const std::string name = command_adapter_->get_output_name(i);
    auto it = output_name_to_idx_.find(name);
    if (it != output_name_to_idx_.end()) {
      output_specs_[it->second] = command_adapter_->get_tensor_spec(i);
    }
  }

  RCLCPP_INFO(
    get_node()->get_logger(), "Activating with %zu inputs and %zu outputs",
    inputs_.size(), outputs_.size());

  // If there are no topic inputs, activate the core controller immediately.
  // Otherwise, defer activation until all topic groups have received their
  // first message (so converters can provide TensorSpecs for reordering).
  core_activated_ = false;
  if (topic_groups_.empty()) {
    if (!try_activate_core()) {
      return controller_interface::CallbackReturn::ERROR;
    }
  }

  update_counter_ = 0;
  RCLCPP_INFO(
    get_node()->get_logger(),
    "InferenceController activated (decimation: %d, inference runs every %d update cycles)",
    decimation_, decimation_);
  return controller_interface::CallbackReturn::SUCCESS;
}

bool InferenceController::try_activate_core()
{
  // Check that all topic groups have received at least one message.
  for (const auto & group : topic_groups_) {
    const auto sample = *group->rt_msg_buffer.readFromRT();
    if (!sample || !sample->message) {
      return false;
    }
  }

  // Convert first messages and populate TensorSpecs from converters.
  for (const auto & group : topic_groups_) {
    const auto sample = *group->rt_msg_buffer.readFromRT();

    for (const auto & entry : group->entries) {
      auto it = source_to_input_idx_.find(entry.source);
      if (it != source_to_input_idx_.end()) {
        inputs_[it->second].tensor = entry.converter->convert(sample->message);
        inputs_[it->second].timestamp_ns = sample->receive_time_ns;
        input_specs_[it->second] = entry.converter->get_tensor_spec();
      }
    }
  }

  // Activate the core controller with complete TensorSpecs.
  auto result =
    inference_controller_->activate(inputs_, input_specs_, output_specs_, outputs_);
  if (!result.has_value()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to activate InferenceController: %s",
      result.error().message.c_str());
    return false;
  }

  core_activated_ = true;
  if (is_active_publisher_) {
    std_msgs::msg::Bool msg;
    msg.data = true;
    is_active_publisher_->publish(msg);
  }

  // Pre-allocate debug publisher messages now that input/output sizes are known.
  if (publish_debug_topics_) {
    // Collect external input names (exclude LSTM feedback states identified by "_in" suffix).
    debug_obs_input_names_.clear();
    debug_obs_input_indices_.clear();
    size_t obs_size = 0;
    for (size_t input_idx = 0; input_idx < inputs_.size(); ++input_idx) {
      const auto & inp = inputs_[input_idx];
      const bool is_feedback = inp.name.size() >= 3 &&
        inp.name.substr(inp.name.size() - 3) == "_in";
      if (!is_feedback) {
        debug_obs_input_names_.push_back(inp.name);
        debug_obs_input_indices_.push_back(input_idx);
        obs_size += static_cast<size_t>(inp.tensor.numel());
      }
    }
    size_t action_size = 0;
    auto action_it = output_name_to_idx_.find(debug_action_output_name_);
    if (action_it != output_name_to_idx_.end()) {
      action_size = static_cast<size_t>(outputs_[action_it->second].tensor.numel());
    } else {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "debug_action_output_name '%s' not found in outputs; action debug topic will be empty",
        debug_action_output_name_.c_str());
    }
    if (obs_debug_publisher_) {
      obs_debug_msg_.data.resize(obs_size, 0.0);
      RCLCPP_INFO(
        get_node()->get_logger(),
        "Debug observation: %zu values from inputs [%s]",
        obs_size,
        [&]() {
          std::string s;
          for (const auto & n : debug_obs_input_names_) {
            s += n + " ";
          }
          return s;
        }().c_str());
    }
    if (action_debug_publisher_) {
      action_debug_msg_.data.resize(action_size, 0.0);
      RCLCPP_INFO(
        get_node()->get_logger(),
        "Debug action: %zu values from output '%s'",
        action_size, debug_action_output_name_.c_str());
    }

    // Collect recurrent hidden-state outputs (names ending in "_out").
    debug_recurrent_output_names_.clear();
    debug_recurrent_output_indices_.clear();
    size_t recurrent_size = 0;
    for (size_t out_idx = 0; out_idx < outputs_.size(); ++out_idx) {
      const auto & out = outputs_[out_idx];
      const bool is_recurrent = out.name.size() >= 4 &&
        out.name.substr(out.name.size() - 4) == "_out";
      if (is_recurrent) {
        debug_recurrent_output_names_.push_back(out.name);
        debug_recurrent_output_indices_.push_back(out_idx);
        recurrent_size += static_cast<size_t>(out.tensor.numel());
      }
    }
    if (recurrent_debug_publisher_) {
      recurrent_debug_msg_.data.resize(recurrent_size, 0.0);
      RCLCPP_INFO(
        get_node()->get_logger(),
        "Debug recurrent state: %zu values from outputs [%s]",
        recurrent_size,
        [&]() {
          std::string s;
          for (const auto & n : debug_recurrent_output_names_) {
            s += n + " ";
          }
          return s;
        }().c_str());
    }
  }

  RCLCPP_INFO(get_node()->get_logger(), "Core InferenceController activated");
  return true;
}

controller_interface::CallbackReturn InferenceController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  if (inference_controller_ && core_activated_) {
    auto result = inference_controller_->deactivate();
    if (!result.has_value()) {
      RCLCPP_WARN(
        get_node()->get_logger(), "Failed to deactivate InferenceController: %s",
        result.error().message.c_str());
    }
  }

  core_activated_ = false;
  debug_step_ = 0;
  if (is_active_publisher_) {
    std_msgs::msg::Bool msg;
    msg.data = false;
    is_active_publisher_->publish(msg);
  }
  inputs_.clear();
  input_specs_.clear();
  outputs_.clear();
  output_specs_.clear();
  hw_to_input_idx_.clear();
  source_to_input_idx_.clear();
  output_name_to_idx_.clear();
  for (auto & reference_input : reference_inputs_) {
    reference_input.command_indices.clear();
    reference_input.input_index = 0;
  }
  debug_obs_input_names_.clear();
  debug_obs_input_indices_.clear();
  debug_recurrent_output_names_.clear();
  debug_recurrent_output_indices_.clear();

  // Clear latest messages from topic groups.
  for (auto & group : topic_groups_) {
    group->rt_msg_buffer.writeFromNonRT(nullptr);
  }

  RCLCPP_INFO(get_node()->get_logger(), "InferenceController deactivated");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type InferenceController::update(
  const rclcpp::Time & time, const rclcpp::Duration &)
{
  if (!inference_controller_) {
    return controller_interface::return_type::ERROR;
  }

  // Decimation: only run inference every N cycles.
  update_counter_++;
  if (update_counter_ < decimation_) {
    return controller_interface::return_type::OK;
  }
  update_counter_ = 0;

  const int64_t timestamp_ns = time.nanoseconds();

  // Deferred activation: wait for all topic groups to have their first message.
  if (!core_activated_) {
    if (!try_activate_core()) {
      std::string pending_topics;
      for (const auto & group : topic_groups_) {
        const auto sample = *group->rt_msg_buffer.readFromRT();
        if (!sample || !sample->message) {
          if (!pending_topics.empty()) {pending_topics += ", ";}
          pending_topics += "'" + group->topic + "'";
        }
      }
      RCLCPP_INFO_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 5000,
        "Waiting for topic messages before activating core controller: %s",
        pending_topics.c_str());
      return controller_interface::return_type::OK;
    }
  }

  // Update hardware inputs (RT-safe: reads into pre-allocated tensors).
  for (size_t i = 0; i < state_adapter_->get_num_hardware_inputs(); ++i) {
    const size_t input_idx = hw_to_input_idx_[i];
    state_adapter_->read_tensor(i, inputs_[input_idx].tensor);
    inputs_[input_idx].timestamp_ns = timestamp_ns;
  }

  if (!topic_inputs_are_fresh(timestamp_ns)) {
    command_adapter_->invalidate_command_interfaces();
    invalidate_reference_inputs();
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "A topic input is stale; invalidating policy commands so SafetyController holds the "
      "measured positions captured at stale entry");
    return controller_interface::return_type::OK;
  }

  // Update topic inputs from latest messages via converters.
  for (const auto & group : topic_groups_) {
    const auto sample = *group->rt_msg_buffer.readFromRT();
    if (!sample || !sample->message) {continue;}

    for (const auto & entry : group->entries) {
      auto it = source_to_input_idx_.find(entry.source);
      if (it != source_to_input_idx_.end()) {
        inputs_[it->second].tensor = entry.converter->convert(sample->message);
        inputs_[it->second].timestamp_ns = sample->receive_time_ns;
      }
    }
  }

  // Run inference.
  auto result = inference_controller_->advance(inputs_, outputs_);
  if (!result.has_value()) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Inference failed: %s",
      result.error().message.c_str());
    return controller_interface::return_type::ERROR;
  }

  if (!write_reference_inputs()) {
    command_adapter_->invalidate_command_interfaces();
    invalidate_reference_inputs();
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "Failed to publish reference input tensors; "
      "invalidating policy command references");
    return controller_interface::return_type::OK;
  }

  // Write outputs to command interfaces.
  for (size_t i = 0; i < command_adapter_->get_num_outputs(); ++i) {
    const std::string name = command_adapter_->get_output_name(i);
    auto it = output_name_to_idx_.find(name);
    if (it != output_name_to_idx_.end()) {
      command_adapter_->write_tensor(i, outputs_[it->second]);
    }
  }

  if (publish_debug_topics_) {
    publish_debug_topics();
  }

  return controller_interface::return_type::OK;
}

bool InferenceController::resolve_reference_input_indices()
{
  for (auto & reference_input : reference_inputs_) {
    auto interface_indices = utils::find_command_interface_indices(
      reference_input.command_interfaces, command_interfaces_);
    if (!interface_indices.has_value()) {
      throw std::runtime_error(
              "Failed to find command interfaces for forwarded input source: " +
              reference_input.source);
    }
    const auto input_it = source_to_input_idx_.find(reference_input.source);
    if (input_it == source_to_input_idx_.end()) {
      throw std::runtime_error(
              "Forwarded input source not found in configured inputs: " +
              reference_input.source);
    }
    reference_input.command_indices = std::move(interface_indices.value());
    reference_input.input_index = input_it->second;
    const auto numel = inputs_[reference_input.input_index].tensor.numel();
    if (numel < 0 ||
      static_cast<size_t>(numel) != reference_input.command_indices.size())
    {
      throw std::runtime_error(
              "Forwarded input source '" + reference_input.source + "' has " +
              std::to_string(numel) + " tensor values but " +
              std::to_string(reference_input.command_indices.size()) +
              " command interfaces");
    }
  }
  return true;
}

bool InferenceController::write_reference_inputs()
{
  for (const auto & reference_input : reference_inputs_) {
    if (reference_input.input_index >= inputs_.size()) {
      return false;
    }
    const auto & tensor = inputs_[reference_input.input_index].tensor;
    if (!tensor.device().is_cpu() || tensor.scalar_type() != torch::kFloat32 ||
      !tensor.is_contiguous() ||
      static_cast<size_t>(tensor.numel()) != reference_input.command_indices.size())
    {
      return false;
    }

    const float * values = tensor.data_ptr<float>();
    for (size_t i = 0; i < reference_input.command_indices.size(); ++i) {
      const double value = static_cast<double>(values[i]);
      if (!std::isfinite(value)) {
        return false;
      }
      (void)command_interfaces_[reference_input.command_indices[i]].set_value(value);
    }
  }
  return true;
}

void InferenceController::invalidate_reference_inputs()
{
  for (const auto & reference_input : reference_inputs_) {
    for (const auto index : reference_input.command_indices) {
      (void)command_interfaces_[index].set_value(std::numeric_limits<double>::quiet_NaN());
    }
  }
}

void InferenceController::publish_debug_topics()
{
  ++debug_step_;

  // Collect observation vector (external inputs, excluding LSTM feedback states).
  std::vector<float> obs_flat;
  for (const auto input_idx : debug_obs_input_indices_) {
    const auto flat = inputs_[input_idx].tensor.flatten().to(torch::kFloat32).contiguous();
    const float * ptr = flat.data_ptr<float>();
    obs_flat.insert(obs_flat.end(), ptr, ptr + flat.numel());
  }

  // Collect action vector.
  std::vector<float> action_flat;
  auto action_it = output_name_to_idx_.find(debug_action_output_name_);
  if (action_it != output_name_to_idx_.end()) {
    const auto flat =
      outputs_[action_it->second].tensor.flatten().to(torch::kFloat32).contiguous();
    const float * ptr = flat.data_ptr<float>();
    action_flat.insert(action_flat.end(), ptr, ptr + flat.numel());
  }

  // Console log.
  if (log_debug_to_console_) {
    std::string obs_str;
    for (size_t i = 0; i < obs_flat.size(); ++i) {
      if (i) {obs_str += ", ";}
      char buf[16];
      snprintf(buf, sizeof(buf), "%.4f", obs_flat[i]);
      obs_str += buf;
    }
    std::string act_str;
    for (size_t i = 0; i < action_flat.size(); ++i) {
      if (i) {act_str += ", ";}
      char buf[16];
      snprintf(buf, sizeof(buf), "%.4f", action_flat[i]);
      act_str += buf;
    }
    RCLCPP_INFO(
      get_node()->get_logger(),
      "[STEP %ld] obs(%zu)=[%s] action(%zu)=[%s]",
      debug_step_, obs_flat.size(), obs_str.c_str(),
      action_flat.size(), act_str.c_str());
  }

  // Topic publish.
  if (obs_debug_publisher_) {
    auto & data = obs_debug_msg_.data;
    for (size_t i = 0; i < obs_flat.size() && i < data.size(); ++i) {
      data[i] = static_cast<double>(obs_flat[i]);
    }
    obs_debug_publisher_->try_publish(obs_debug_msg_);
  }

  if (action_debug_publisher_) {
    auto & data = action_debug_msg_.data;
    for (size_t i = 0; i < action_flat.size() && i < data.size(); ++i) {
      data[i] = static_cast<double>(action_flat[i]);
    }
    action_debug_publisher_->try_publish(action_debug_msg_);
  }

  // Recurrent hidden state ("_out" tensors): the state produced this step, which
  // is fed back as the next step's "_in". Concatenated in output order.
  if (recurrent_debug_publisher_) {
    auto & data = recurrent_debug_msg_.data;
    size_t offset = 0;
    for (const auto output_idx : debug_recurrent_output_indices_) {
      const auto flat =
        outputs_[output_idx].tensor.flatten().to(torch::kFloat32).contiguous();
      const float * ptr = flat.data_ptr<float>();
      for (int64_t i = 0; i < flat.numel() && offset < data.size(); ++i, ++offset) {
        data[offset] = static_cast<double>(ptr[i]);
      }
    }
    recurrent_debug_publisher_->try_publish(recurrent_debug_msg_);
  }
}

bool InferenceController::topic_inputs_are_fresh(int64_t current_time_ns) const
{
  if (topic_input_timeout_ns_ <= 0) {
    return true;
  }

  for (const auto & group : topic_groups_) {
    const auto sample = *group->rt_msg_buffer.readFromRT();
    if (!sample || !sample->message) {
      return false;
    }
    const int64_t age_ns = std::max<int64_t>(
      current_time_ns - sample->receive_time_ns, 0);
    if (age_ns > topic_input_timeout_ns_) {
      return false;
    }
  }
  return true;
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::InferenceController,
  controller_interface::ControllerInterface)

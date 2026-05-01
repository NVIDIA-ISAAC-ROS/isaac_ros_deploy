// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/controllers/inference_controller.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "isaac_deploy_core/inference_controller/config_parser.h"
#include "isaac_ros_deploy_ros2_control/adapters/command_interface_adapter.hpp"
#include "isaac_ros_deploy_ros2_control/adapters/state_interface_adapter.hpp"
#include "isaac_ros_deploy_ros2_control/converters/command_interface_converter.hpp"
#include "isaac_ros_deploy_ros2_control/converters/state_interface_converter.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

namespace
{

/// Extract a human-readable error message from an isaac_deploy_core::Error.
const std::string & error_message(const isaac_deploy_core::Error & error)
{
  return error.message;
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

  if (config_path_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "config_path parameter is required");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Initialize all converter registries before loading config.
  initialize_state_interface_converters();
  initialize_command_interface_converters();
  isaac_ros_deploy_converters::initialize_input_converters();

  if (!load_config()) {
    return controller_interface::CallbackReturn::ERROR;
  }

  create_topic_subscriptions();

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

    // Classify input terms using the state interface converter registry.
    const auto & state_registry = StateInterfaceConverterRegistry::instance();
    std::vector<isaac_deploy_core::InputTermConfig> hardware_input_configs;
    topic_source_configs_.clear();

    for (const auto & term : inputs_result.value().terms) {
      if (!term.output_key.empty()) {
        // Feedback input — handled internally by the core controller.
        RCLCPP_INFO(
          get_node()->get_logger(),
          "Input '%s' (kind: %s) is feedback, handled by core library",
          term.name.c_str(), term.kind.c_str());
      } else if (state_registry.contains(term.kind)) {
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
    state_adapter_ = std::make_unique<StateInterfaceAdapter>(hardware_input_configs);

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
      output_configs, command_prefix_, command_suffix_);

    // Parse runner config (from config sections, with parameter overrides).
    isaac_deploy_core::InferenceRunner::Config runner_config;
    runner_config.model_path = sections.model_path;
    runner_config.runner_type = sections.backend;
    // Command-line parameter overrides YAML.
    if (!model_path_.empty()) {
      runner_config.model_path = model_path_;
    }
    if (runner_config.runner_type.empty()) {
      runner_config.runner_type = "triton";
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

    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to load config: %s", e.what());
    return false;
  }
}

void InferenceController::create_topic_subscriptions()
{
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

  for (const auto & topic_config : topic_source_configs_) {
    const auto message_type_override = resolve_message_type(topic_config.source);
    auto converter = registry.create_for_kind(
      topic_config.kind, topic_config.source, message_type_override);
    if (!converter) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "No converter found for kind '%s' (source '%s'), skipping",
        topic_config.kind.c_str(), topic_config.source.c_str());
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

  // Create one generic subscription per topic group.
  for (auto & [topic, group] : groups_by_topic) {
    auto callback = [grp = group.get()](std::shared_ptr<rclcpp::SerializedMessage> msg) {
        grp->rt_msg_buffer.writeFromNonRT(msg);
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
    RCLCPP_INFO(
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
    RCLCPP_INFO(
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
    if (!try_activate_core(timestamp_ns)) {
      return controller_interface::CallbackReturn::ERROR;
    }
  }

  update_counter_ = 0;
  RCLCPP_INFO(get_node()->get_logger(), "InferenceController activated");
  return controller_interface::CallbackReturn::SUCCESS;
}

bool InferenceController::try_activate_core(int64_t timestamp_ns)
{
  // Check that all topic groups have received at least one message.
  for (const auto & group : topic_groups_) {
    if (!*group->rt_msg_buffer.readFromRT()) {
      return false;
    }
  }

  // Convert first messages and populate TensorSpecs from converters.
  for (const auto & group : topic_groups_) {
    const auto msg = *group->rt_msg_buffer.readFromRT();

    for (const auto & entry : group->entries) {
      auto it = source_to_input_idx_.find(entry.source);
      if (it != source_to_input_idx_.end()) {
        inputs_[it->second].tensor = entry.converter->convert(msg);
        inputs_[it->second].timestamp_ns = timestamp_ns;
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
      error_message(result.error()).c_str());
    return false;
  }

  core_activated_ = true;
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
        error_message(result.error()).c_str());
    }
  }

  core_activated_ = false;
  inputs_.clear();
  input_specs_.clear();
  outputs_.clear();
  output_specs_.clear();
  hw_to_input_idx_.clear();
  source_to_input_idx_.clear();
  output_name_to_idx_.clear();

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
    if (!try_activate_core(timestamp_ns)) {
      std::string pending_topics;
      for (const auto & group : topic_groups_) {
        if (!*group->rt_msg_buffer.readFromRT()) {
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

  // Update topic inputs from latest messages via converters.
  for (const auto & group : topic_groups_) {
    const auto msg = *group->rt_msg_buffer.readFromRT();
    if (!msg) {continue;}

    for (const auto & entry : group->entries) {
      auto it = source_to_input_idx_.find(entry.source);
      if (it != source_to_input_idx_.end()) {
        inputs_[it->second].tensor = entry.converter->convert(msg);
        inputs_[it->second].timestamp_ns = timestamp_ns;
      }
    }
  }

  // Run inference.
  auto result = inference_controller_->advance(inputs_, outputs_);
  if (!result.has_value()) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Inference failed: %s",
      error_message(result.error()).c_str());
    return controller_interface::return_type::ERROR;
  }

  // Write outputs to command interfaces.
  for (size_t i = 0; i < command_adapter_->get_num_outputs(); ++i) {
    const std::string name = command_adapter_->get_output_name(i);
    auto it = output_name_to_idx_.find(name);
    if (it != output_name_to_idx_.end()) {
      command_adapter_->write_tensor(i, outputs_[it->second]);
    }
  }

  return controller_interface::return_type::OK;
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::InferenceController,
  controller_interface::ControllerInterface)

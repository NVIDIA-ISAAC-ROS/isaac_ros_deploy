// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/controllers/forward_joint_command_controller.hpp"

#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

ForwardJointCommandController::ForwardJointCommandController() = default;

controller_interface::CallbackReturn ForwardJointCommandController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>());
    auto_declare<double>("default_kp", 0.0);
    auto_declare<double>("default_kd", 1.0);
    auto_declare<std::string>("joint_command_topic", "");
    auto_declare<std::string>("joint_state_topic", "");
    auto_declare<std::string>("command_prefix", "");
    auto_declare<std::string>("command_suffix", "");
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ForwardJointCommandController::on_configure(
  const rclcpp_lifecycle::State &)
{
  joint_names_ = get_node()->get_parameter("joints").as_string_array();
  default_kp_ = get_node()->get_parameter("default_kp").as_double();
  default_kd_ = get_node()->get_parameter("default_kd").as_double();
  command_prefix_ = get_node()->get_parameter("command_prefix").as_string();
  command_suffix_ = get_node()->get_parameter("command_suffix").as_string();

  if (joint_names_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joints specified");
    return controller_interface::CallbackReturn::ERROR;
  }

  const auto joint_command_topic = get_node()->get_parameter("joint_command_topic").as_string();
  const auto joint_state_topic = get_node()->get_parameter("joint_state_topic").as_string();

  if (!joint_command_topic.empty() && !joint_state_topic.empty()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Only one of 'joint_command_topic' or 'joint_state_topic' may be set");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (joint_command_topic.empty() && joint_state_topic.empty()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "One of 'joint_command_topic' or 'joint_state_topic' must be set");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (!joint_command_topic.empty()) {
    joint_command_subscription_ =
      get_node()->create_subscription<isaac_ros_deploy_interfaces::msg::JointCommand>(
        joint_command_topic, 1,
        [this](isaac_ros_deploy_interfaces::msg::JointCommand::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(msg_mutex_);
          latest_msg_ = msg;
        });
  } else {
    joint_state_subscription_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic, 1,
      [this](sensor_msgs::msg::JointState::SharedPtr msg) {
        auto cmd = std::make_shared<isaac_ros_deploy_interfaces::msg::JointCommand>();
        cmd->names = msg->name;
        cmd->position = msg->position;
        cmd->velocity = msg->velocity;
        // kp and kd are intentionally left empty — update() falls back to default_kp/default_kd.
        std::lock_guard<std::mutex> lock(msg_mutex_);
        latest_msg_ = cmd;
      });
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Configured ForwardJointCommandController with %zu joints, default_kp=%.2f, default_kd=%.2f",
    joint_names_.size(), default_kp_, default_kd_);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
ForwardJointCommandController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  const auto names = build_command_interface_names();
  config.names.reserve(joint_names_.size() * 5);
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    config.names.push_back(names.position[i]);
    config.names.push_back(names.velocity[i]);
    config.names.push_back(names.effort[i]);
    config.names.push_back(names.kp[i]);
    config.names.push_back(names.kd[i]);
  }

  return config;
}

controller_interface::InterfaceConfiguration
ForwardJointCommandController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joint_names_) {
    config.names.push_back(joint + "/position");
  }
  return config;
}

controller_interface::CallbackReturn ForwardJointCommandController::on_activate(
  const rclcpp_lifecycle::State &)
{
  const auto names = build_command_interface_names();

  auto pos_cmd = utils::find_command_interface_indices(names.position, command_interfaces_);
  auto vel_cmd = utils::find_command_interface_indices(names.velocity, command_interfaces_);
  auto eff_cmd = utils::find_command_interface_indices(names.effort, command_interfaces_);
  auto kp_cmd = utils::find_command_interface_indices(names.kp, command_interfaces_);
  auto kd_cmd = utils::find_command_interface_indices(names.kd, command_interfaces_);

  if (!pos_cmd || !vel_cmd || !eff_cmd || !kp_cmd || !kd_cmd) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find command interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  position_command_indices_ = std::move(pos_cmd.value());
  velocity_command_indices_ = std::move(vel_cmd.value());
  effort_command_indices_ = std::move(eff_cmd.value());
  kp_command_indices_ = std::move(kp_cmd.value());
  kd_command_indices_ = std::move(kd_cmd.value());

  // Cache state interface indices for correct joint-to-interface mapping.
  std::vector<std::string> position_names;
  position_names.reserve(joint_names_.size());
  for (const auto & name : joint_names_) {
    position_names.push_back(name + "/position");
  }
  const auto pos_state = utils::find_state_interface_indices(position_names, state_interfaces_);
  if (!pos_state.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find joint position state interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  position_state_indices_ = std::move(pos_state.value());

  // Capture joint positions at activation using correct index mapping.
  initial_positions_.resize(joint_names_.size(), 0.0);
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    initial_positions_[i] =
      state_interfaces_[position_state_indices_[i]].get_optional<double>().value_or(0.0);
  }

  received_first_message_ = false;

  RCLCPP_INFO(
    get_node()->get_logger(),
    "ForwardJointCommandController activated with %zu joints",
    joint_names_.size());

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ForwardJointCommandController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  position_state_indices_.clear();
  position_command_indices_.clear();
  velocity_command_indices_.clear();
  effort_command_indices_.clear();
  kp_command_indices_.clear();
  kd_command_indices_.clear();
  initial_positions_.clear();
  received_first_message_ = false;

  RCLCPP_INFO(get_node()->get_logger(), "ForwardJointCommandController deactivated");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type ForwardJointCommandController::update(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  // Get latest message.
  isaac_ros_deploy_interfaces::msg::JointCommand::SharedPtr msg;
  {
    std::lock_guard<std::mutex> lock(msg_mutex_);
    msg = latest_msg_;
  }

  if (!msg) {
    // No message yet: hold joint positions captured at activation.
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      (void)command_interfaces_[position_command_indices_[i]].set_value(initial_positions_[i]);
      (void)command_interfaces_[velocity_command_indices_[i]].set_value(0.0);
      (void)command_interfaces_[effort_command_indices_[i]].set_value(0.0);
      (void)command_interfaces_[kp_command_indices_[i]].set_value(default_kp_);
      (void)command_interfaces_[kd_command_indices_[i]].set_value(default_kd_);
    }
    return controller_interface::return_type::OK;
  }

  if (!received_first_message_) {
    received_first_message_ = true;
    RCLCPP_INFO(get_node()->get_logger(), "Received first JointCommand, switching to tracking");
  }

  // Rebuild name-to-index map if names changed.
  if (msg->names != last_input_names_) {
    input_name_to_idx_.clear();
    for (size_t i = 0; i < msg->names.size(); ++i) {
      input_name_to_idx_[msg->names[i]] = i;
    }
    last_input_names_ = msg->names;
  }

  const bool has_velocity = !msg->velocity.empty();
  const bool has_effort = !msg->effort.empty();
  const bool has_kp = !msg->kp.empty();
  const bool has_kd = !msg->kd.empty();

  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto it = input_name_to_idx_.find(joint_names_[i]);
    if (it == input_name_to_idx_.end() || it->second >= msg->position.size()) {
      // Joint not in message: hold position captured at activation.
      (void)command_interfaces_[position_command_indices_[i]].set_value(initial_positions_[i]);
      (void)command_interfaces_[velocity_command_indices_[i]].set_value(0.0);
      (void)command_interfaces_[effort_command_indices_[i]].set_value(0.0);
      (void)command_interfaces_[kp_command_indices_[i]].set_value(default_kp_);
      (void)command_interfaces_[kd_command_indices_[i]].set_value(default_kd_);
      continue;
    }

    const size_t idx = it->second;
    (void)command_interfaces_[position_command_indices_[i]].set_value(msg->position[idx]);
    (void)command_interfaces_[velocity_command_indices_[i]].set_value(
      has_velocity && idx < msg->velocity.size() ? msg->velocity[idx] : 0.0);
    (void)command_interfaces_[effort_command_indices_[i]].set_value(
      has_effort && idx < msg->effort.size() ? msg->effort[idx] : 0.0);
    (void)command_interfaces_[kp_command_indices_[i]].set_value(
      has_kp && idx < msg->kp.size() ? msg->kp[idx] : default_kp_);
    (void)command_interfaces_[kd_command_indices_[i]].set_value(
      has_kd && idx < msg->kd.size() ? msg->kd[idx] : default_kd_);
  }

  return controller_interface::return_type::OK;
}

ForwardJointCommandController::InterfaceNames
ForwardJointCommandController::build_command_interface_names() const
{
  InterfaceNames names;
  names.position.reserve(joint_names_.size());
  names.velocity.reserve(joint_names_.size());
  names.effort.reserve(joint_names_.size());
  names.kp.reserve(joint_names_.size());
  names.kd.reserve(joint_names_.size());

  for (const auto & joint : joint_names_) {
    names.position.push_back(make_command_interface_name(joint, "position"));
    names.velocity.push_back(make_command_interface_name(joint, "velocity"));
    names.effort.push_back(make_command_interface_name(joint, "effort"));
    names.kp.push_back(make_command_interface_name(joint, "kp"));
    names.kd.push_back(make_command_interface_name(joint, "kd"));
  }

  return names;
}

std::string ForwardJointCommandController::make_command_interface_name(
  const std::string & joint, const std::string & type) const
{
  std::string name;
  if (!command_prefix_.empty()) {
    name = command_prefix_ + "/";
  }
  name += joint + "/" + type;
  if (!command_suffix_.empty()) {
    name += command_suffix_;
  }
  return name;
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::ForwardJointCommandController,
  controller_interface::ControllerInterface)

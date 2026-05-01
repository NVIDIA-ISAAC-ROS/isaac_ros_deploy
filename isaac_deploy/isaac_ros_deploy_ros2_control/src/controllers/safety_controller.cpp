// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/controllers/safety_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{
namespace
{

std::optional<isaac_deploy_core::BlendStrategy> parse_blend_strategy(
  const std::string & strategy)
{
  if (strategy == "clamp_velocity") {
    return isaac_deploy_core::BlendStrategy::kClampVelocity;
  }
  if (strategy == "linear_interpolate") {
    return isaac_deploy_core::BlendStrategy::kLinearInterpolate;
  }
  if (strategy == "no_post_processing") {
    return isaac_deploy_core::BlendStrategy::kNoPostProcessing;
  }
  return std::nullopt;
}

const char * to_string(isaac_deploy_core::BlendStrategy type)
{
  switch (type) {
    case isaac_deploy_core::BlendStrategy::kClampVelocity:
      return "clamp_velocity";
    case isaac_deploy_core::BlendStrategy::kLinearInterpolate:
      return "linear_interpolate";
    case isaac_deploy_core::BlendStrategy::kNoPostProcessing:
      return "no_post_processing";
  }
  return "unknown";
}

}  // namespace

SafetyController::SafetyController()
: controller_interface::ChainableControllerInterface()
{
}

controller_interface::CallbackReturn SafetyController::on_init()
{
  try {
    auto_declare<std::string>("blend_strategy", "linear_interpolate");
    auto_declare<double>("default_kp", 0.0);
    auto_declare<double>("default_kd", 0.0);

    // Joints
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>());

    // Declare blend_ratio as a dynamic parameter with constraints for rqt_reconfigure
    rcl_interfaces::msg::ParameterDescriptor blend_ratio_desc;
    blend_ratio_desc.description =
      "Blend ratio for safety blending (0.0 = policy disabled, 1.0 = policy fully enabled). "
      "Dynamically adjustable via rqt_reconfigure.";
    blend_ratio_desc.read_only = false;
    blend_ratio_desc.dynamic_typing = false;
    rcl_interfaces::msg::FloatingPointRange range;
    range.from_value = 0.0;
    range.to_value = 1.0;
    range.step = 0.01;
    blend_ratio_desc.floating_point_range.push_back(range);
    get_node()->declare_parameter("blend_ratio", rclcpp::ParameterValue(0.0), blend_ratio_desc);

    rcl_interfaces::msg::ParameterDescriptor max_speed_desc;
    max_speed_desc.description =
      "Maximum rate of change for blend_ratio (units/second). "
      "Controls how fast blend_ratio ramps toward its target.";
    max_speed_desc.read_only = false;
    max_speed_desc.dynamic_typing = false;
    rcl_interfaces::msg::FloatingPointRange speed_range;
    speed_range.from_value = 0.0;
    speed_range.to_value = 1.0;
    speed_range.step = 0.01;
    max_speed_desc.floating_point_range.push_back(speed_range);
    get_node()->declare_parameter(
      "max_blend_ratio_speed", rclcpp::ParameterValue(1.0), max_speed_desc);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn SafetyController::on_configure(
  const rclcpp_lifecycle::State &)
{
  // Load parameters
  const auto blend_strategy_str =
    get_node()->get_parameter("blend_strategy").as_string();
  const auto blend_strategy = parse_blend_strategy(blend_strategy_str);
  if (!blend_strategy.has_value()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Unknown blend_strategy '%s'. Supported values: clamp_velocity, "
      "linear_interpolate, no_post_processing",
      blend_strategy_str.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  blend_strategy_ = blend_strategy.value();
  joint_names_ = get_node()->get_parameter("joints").as_string_array();
  target_blend_ratio_.store(get_node()->get_parameter("blend_ratio").as_double());
  default_kp_ = get_node()->get_parameter("default_kp").as_double();
  default_kd_ = get_node()->get_parameter("default_kd").as_double();
  max_blend_ratio_speed_.store(get_node()->get_parameter("max_blend_ratio_speed").as_double());

  if (joint_names_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joints specified");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Set up dynamic parameter callback for blend_ratio
  param_callback_handle_ = get_node()->add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & parameters) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;

      for (const auto & param : parameters) {
        if (param.get_name() == "blend_ratio") {
          const double value = param.as_double();
          if (value < 0.0 || value > 1.0) {
            result.successful = false;
            result.reason = "blend_ratio must be between 0.0 and 1.0";
            return result;
          }
          target_blend_ratio_.store(value);
          RCLCPP_DEBUG(
            get_node()->get_logger(), "Updated blend_ratio target to: %.3f", value);
        } else if (param.get_name() == "max_blend_ratio_speed") {
          const double value = param.as_double();
          if (value < 0.0) {
            result.successful = false;
            result.reason = "max_blend_ratio_speed must be non-negative";
            return result;
          }
          max_blend_ratio_speed_.store(value);
          RCLCPP_DEBUG(
            get_node()->get_logger(), "Updated max_blend_ratio_speed to: %.3f", value);
        }
      }
      return result;
    });

  // Create safety controller from parameters
  if (!create_safety_controller()) {
    return controller_interface::CallbackReturn::ERROR;
  }

  // Initialize reference interfaces storage with NaN
  reference_interfaces_.resize(
    joint_names_.size() * kInterfacesPerJoint,
    std::numeric_limits<double>::quiet_NaN());

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Configured SafetyController with %zu joints, blend_strategy: %s, blend_ratio: %.3f, "
    "max_blend_ratio_speed: %.3f",
    joint_names_.size(), to_string(blend_strategy_), target_blend_ratio_.load(),
    max_blend_ratio_speed_.load());

  return controller_interface::CallbackReturn::SUCCESS;
}

bool SafetyController::create_safety_controller()
{
  try {
    isaac_deploy_core::SafetyControllerConfig config;
    config.blend_ratio.type = blend_strategy_;

    auto controller_result = isaac_deploy_core::SafetyController::create(config);
    if (!controller_result.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Failed to create SafetyController: %s",
        std::string(controller_result.error().message).c_str());
      return false;
    }

    safety_controller_.emplace(std::move(controller_result.value()));

    // Pre-allocate tensors.
    // Inputs: command_positions, current_positions, blend_ratio, dt
    const int64_t num_joints = static_cast<int64_t>(joint_names_.size());

    inputs_.clear();
    input_specs_.clear();

    // command_positions [1, N]
    inputs_.push_back(utils::create_preallocated_tensor("command_positions", {1, num_joints}));
    input_specs_.push_back(utils::create_joint_tensor_spec(joint_names_));

    // current_positions [1, N]
    inputs_.push_back(utils::create_preallocated_tensor("current_positions", {1, num_joints}));
    input_specs_.push_back(utils::create_joint_tensor_spec(joint_names_));

    // blend_ratio [1, 1]
    inputs_.push_back(utils::create_preallocated_tensor("blend_ratio", {1, 1}));
    input_specs_.push_back(utils::create_scalar_tensor_spec("blend_ratio"));

    // dt [1, 1]
    inputs_.push_back(utils::create_preallocated_tensor("dt", {1, 1}));
    input_specs_.push_back(utils::create_scalar_tensor_spec("dt"));

    // Outputs: safe_positions [1, N]
    outputs_.clear();
    output_specs_.clear();
    outputs_.push_back(utils::create_preallocated_tensor("safe_positions", {1, num_joints}));
    output_specs_.push_back(utils::create_joint_tensor_spec(joint_names_));

    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to load config: %s", e.what());
    return false;
  }
}

controller_interface::InterfaceConfiguration
SafetyController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Request all command interfaces for all joints.
  for (const auto & joint_name : joint_names_) {
    config.names.push_back(joint_name + "/position");
    config.names.push_back(joint_name + "/velocity");
    config.names.push_back(joint_name + "/effort");
    config.names.push_back(joint_name + "/kp");
    config.names.push_back(joint_name + "/kd");
  }

  return config;
}

controller_interface::InterfaceConfiguration
SafetyController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Request position state interfaces for all joints.
  for (const auto & joint_name : joint_names_) {
    config.names.push_back(joint_name + "/position");
  }

  return config;
}

std::vector<hardware_interface::CommandInterface>
SafetyController::on_export_reference_interfaces()
{
  std::vector<hardware_interface::CommandInterface> reference_interfaces;

  const std::string controller_name = get_node()->get_name();

  // Export reference interfaces for all command types with _raw suffix.
  // Layout in reference_interfaces_: [pos_0, vel_0, eff_0, kp_0, kd_0, pos_1, ...]
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    reference_interfaces.emplace_back(
      controller_name, joint_names_[i] + "/position_raw",
      &reference_interfaces_[i * kInterfacesPerJoint + 0]);
    reference_interfaces.emplace_back(
      controller_name, joint_names_[i] + "/velocity_raw",
      &reference_interfaces_[i * kInterfacesPerJoint + 1]);
    reference_interfaces.emplace_back(
      controller_name, joint_names_[i] + "/effort_raw",
      &reference_interfaces_[i * kInterfacesPerJoint + 2]);
    reference_interfaces.emplace_back(
      controller_name, joint_names_[i] + "/kp_raw",
      &reference_interfaces_[i * kInterfacesPerJoint + 3]);
    reference_interfaces.emplace_back(
      controller_name, joint_names_[i] + "/kd_raw",
      &reference_interfaces_[i * kInterfacesPerJoint + 4]);
  }

  return reference_interfaces;
}

controller_interface::return_type SafetyController::update_reference_from_subscribers(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  // No subscribers - blend ratio comes from parameter.
  return controller_interface::return_type::OK;
}

controller_interface::CallbackReturn SafetyController::on_activate(
  const rclcpp_lifecycle::State &)
{
  if (!safety_controller_) {
    RCLCPP_ERROR(get_node()->get_logger(), "Safety controller not initialized");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Cache joint position state indices.
  std::vector<std::string> position_names;
  position_names.reserve(joint_names_.size());
  for (const auto & name : joint_names_) {
    position_names.push_back(name + "/position");
  }
  auto pos_indices = utils::find_state_interface_indices(position_names, state_interfaces_);
  if (!pos_indices.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find joint position state interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  position_state_indices_ = std::move(pos_indices.value());

  // Initialize reference interfaces to safe defaults (current positions + zero velocity/effort)
  // This prevents NaN from propagating if upstream controllers haven't written yet
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto current_pos =
      state_interfaces_[position_state_indices_[i]].get_optional<double>();
    if (!current_pos.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to read initial position for joint '%s' — cannot activate safely",
        joint_names_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    reference_interfaces_[i * kInterfacesPerJoint + 0] = current_pos.value();  // position
    reference_interfaces_[i * kInterfacesPerJoint + 1] = 0.0;  // velocity
    reference_interfaces_[i * kInterfacesPerJoint + 2] = 0.0;  // effort
    reference_interfaces_[i * kInterfacesPerJoint + 3] = default_kp_;  // kp
    reference_interfaces_[i * kInterfacesPerJoint + 4] = default_kd_;  // kd
  }

  // Initialize current blend ratio to target so there's no ramp on first activation
  current_blend_ratio_ = target_blend_ratio_.load();

  RCLCPP_INFO(
    get_node()->get_logger(), "Using blend_ratio parameter, initial value: %.3f",
    target_blend_ratio_.load());

  // Cache command interface indices.
  std::vector<std::string> cmd_pos_names, cmd_vel_names, cmd_eff_names, cmd_kp_names, cmd_kd_names;
  cmd_pos_names.reserve(joint_names_.size());
  cmd_vel_names.reserve(joint_names_.size());
  cmd_eff_names.reserve(joint_names_.size());
  cmd_kp_names.reserve(joint_names_.size());
  cmd_kd_names.reserve(joint_names_.size());

  for (const auto & name : joint_names_) {
    cmd_pos_names.push_back(name + "/position");
    cmd_vel_names.push_back(name + "/velocity");
    cmd_eff_names.push_back(name + "/effort");
    cmd_kp_names.push_back(name + "/kp");
    cmd_kd_names.push_back(name + "/kd");
  }

  auto maybe_cmd_pos = utils::find_command_interface_indices(cmd_pos_names, command_interfaces_);
  auto maybe_cmd_vel = utils::find_command_interface_indices(cmd_vel_names, command_interfaces_);
  auto maybe_cmd_eff = utils::find_command_interface_indices(cmd_eff_names, command_interfaces_);
  auto maybe_cmd_kp = utils::find_command_interface_indices(cmd_kp_names, command_interfaces_);
  auto maybe_cmd_kd = utils::find_command_interface_indices(cmd_kd_names, command_interfaces_);

  if (!maybe_cmd_pos || !maybe_cmd_vel || !maybe_cmd_eff || !maybe_cmd_kp || !maybe_cmd_kd) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find one or more command interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }

  position_command_indices_ = std::move(maybe_cmd_pos.value());
  velocity_command_indices_ = std::move(maybe_cmd_vel.value());
  effort_command_indices_ = std::move(maybe_cmd_eff.value());
  kp_command_indices_ = std::move(maybe_cmd_kp.value());
  kd_command_indices_ = std::move(maybe_cmd_kd.value());

  // Populate initial input values.
  int64_t timestamp_ns = 0;
  rclcpp::Duration initial_period(0, 1000000);  // 1ms
  populate_inputs_from_interfaces(initial_period, timestamp_ns);

  // Activate the core controller.
  auto result = safety_controller_->activate(input_specs_, inputs_, output_specs_, outputs_);
  if (!result.has_value()) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Failed to activate SafetyController: %s",
      std::string(result.error().message).c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  RCLCPP_INFO(get_node()->get_logger(), "SafetyController activated");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn SafetyController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  if (safety_controller_) {
    auto result = safety_controller_->deactivate();
    if (!result.has_value()) {
      RCLCPP_WARN(
        get_node()->get_logger(), "Failed to deactivate SafetyController: %s",
        std::string(result.error().message).c_str());
    }
  }

  position_state_indices_.clear();
  position_command_indices_.clear();
  velocity_command_indices_.clear();
  effort_command_indices_.clear();
  kp_command_indices_.clear();
  kd_command_indices_.clear();

  RCLCPP_INFO(get_node()->get_logger(), "SafetyController deactivated");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type SafetyController::update_and_write_commands(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  if (!safety_controller_) {
    return controller_interface::return_type::ERROR;
  }

  int64_t timestamp_ns = time.nanoseconds();

  // Populate inputs from interfaces.
  populate_inputs_from_interfaces(period, timestamp_ns);

  // Run safety controller.
  auto result = safety_controller_->advance(timestamp_ns, inputs_, outputs_);
  if (!result.has_value()) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Safety controller advance failed: %s",
      std::string(result.error().message).c_str());
    return controller_interface::return_type::ERROR;
  }

  // Write safe positions to command interfaces.
  write_outputs_to_interfaces();

  return controller_interface::return_type::OK;
}

void SafetyController::populate_inputs_from_interfaces(
  const rclcpp::Duration & period, int64_t timestamp_ns)
{
  // command_positions from reference interfaces (index 0 in layout: pos, vel, eff, kp, kd)
  auto cmd_pos_accessor = inputs_[0].tensor.accessor<float, 2>();
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    double pos_cmd = reference_interfaces_[i * kInterfacesPerJoint + 0];
    // Use current position if reference position is NaN (uninitialized)
    if (std::isnan(pos_cmd)) {
      const auto current_pos =
        state_interfaces_[position_state_indices_[i]].get_optional<double>();
      if (!current_pos.has_value()) {
        RCLCPP_ERROR_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 1000,
          "Failed to read position for joint '%s' — using last known value",
          joint_names_[i].c_str());
        // Keep whatever was previously in the tensor
        continue;
      }
      pos_cmd = current_pos.value();
    }
    cmd_pos_accessor[0][i] = static_cast<float>(pos_cmd);
  }
  inputs_[0].timestamp_ns = timestamp_ns;

  // current_positions from state interfaces
  if (!utils::populate_tensor_from_state_interfaces(
      inputs_[1], position_state_indices_, state_interfaces_, timestamp_ns))
  {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "Failed to read one or more joint position state interfaces — using stale values");
  }

  // Rate-limit blend_ratio toward target
  const double max_delta = max_blend_ratio_speed_.load() * period.seconds();
  const double error = target_blend_ratio_.load() - current_blend_ratio_;
  current_blend_ratio_ += std::clamp(error, -max_delta, max_delta);

  auto blend_ratio_accessor = inputs_[2].tensor.accessor<float, 2>();
  blend_ratio_accessor[0][0] = static_cast<float>(current_blend_ratio_);
  inputs_[2].timestamp_ns = timestamp_ns;

  // dt
  auto dt_accessor = inputs_[3].tensor.accessor<float, 2>();
  dt_accessor[0][0] = static_cast<float>(period.seconds());
  inputs_[3].timestamp_ns = timestamp_ns;
}

void SafetyController::write_outputs_to_interfaces()
{
  // Write safe_positions to position command interfaces.
  utils::write_tensor_to_command_interfaces(
    outputs_[0], position_command_indices_, command_interfaces_);

  // Write velocity, effort, kp, kd according to the active blend strategy.
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    double vel_cmd = reference_interfaces_[i * kInterfacesPerJoint + 1];
    double eff_cmd = reference_interfaces_[i * kInterfacesPerJoint + 2];
    double kp_cmd  = reference_interfaces_[i * kInterfacesPerJoint + 3];
    double kd_cmd  = reference_interfaces_[i * kInterfacesPerJoint + 4];

    if (blend_strategy_ == isaac_deploy_core::BlendStrategy::kLinearInterpolate) {
      // Scale velocity and effort by blend ratio; pass gains through unchanged from the policy.
      vel_cmd = std::isnan(vel_cmd) ? 0.0 : current_blend_ratio_ * vel_cmd;
      eff_cmd = std::isnan(eff_cmd) ? 0.0 : current_blend_ratio_ * eff_cmd;
      kp_cmd  = std::isnan(kp_cmd)  ? default_kp_ : kp_cmd;
      kd_cmd  = std::isnan(kd_cmd)  ? default_kd_ : kd_cmd;
    } else {
      // Pass through: NaN replaced with zero for vel/eff, default_kp/kd for gains.
      if (std::isnan(vel_cmd)) {vel_cmd = 0.0;}
      if (std::isnan(eff_cmd)) {eff_cmd = 0.0;}
      if (std::isnan(kp_cmd)) {kp_cmd = default_kp_;}
      if (std::isnan(kd_cmd)) {kd_cmd = default_kd_;}
    }

    (void)command_interfaces_[velocity_command_indices_[i]].set_value(vel_cmd);
    (void)command_interfaces_[effort_command_indices_[i]].set_value(eff_cmd);
    (void)command_interfaces_[kp_command_indices_[i]].set_value(kp_cmd);
    (void)command_interfaces_[kd_command_indices_[i]].set_value(kd_cmd);
  }
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::SafetyController,
  controller_interface::ChainableControllerInterface)

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

#include "isaac_ros_deploy_ros2_control/controllers/safety_controller.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <regex>
#include <string_view>

#include "isaac_ros_deploy_ros2_control/utils/gain_utils.hpp"
#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{
namespace
{

constexpr char kSwitchControllerService[] = "/controller_manager/switch_controller";
constexpr double kEmergencySwitchWatchdogGraceS = 1.0;
constexpr uint32_t kNanosecondsPerSecond = 1000000000u;

std::optional<isaac_deploy_core::BlendStrategy> parse_blend_strategy(
  const std::string & strategy)
{
  if (strategy == "interpolate") {
    return isaac_deploy_core::BlendStrategy::kInterpolate;
  }
  if (strategy == "no_post_processing") {
    return isaac_deploy_core::BlendStrategy::kNoPostProcessing;
  }
  return std::nullopt;
}

std::optional<isaac_deploy_core::BlendReference> parse_blend_reference(
  const std::string & reference)
{
  if (reference == "activation") {
    return isaac_deploy_core::BlendReference::kActivation;
  }
  if (reference == "current") {
    return isaac_deploy_core::BlendReference::kCurrent;
  }
  return std::nullopt;
}

const char * to_string(isaac_deploy_core::BlendStrategy type)
{
  switch (type) {
    case isaac_deploy_core::BlendStrategy::kInterpolate:
      return "interpolate";
    case isaac_deploy_core::BlendStrategy::kNoPostProcessing:
      return "no_post_processing";
  }
  return "unknown";
}

const char * to_string(isaac_deploy_core::BlendReference reference)
{
  switch (reference) {
    case isaac_deploy_core::BlendReference::kActivation:
      return "activation";
    case isaac_deploy_core::BlendReference::kCurrent:
      return "current";
  }
  return "unknown";
}

bool is_out_of_domain_detection_parameter(const std::string & name)
{
  constexpr char kPrefix[] = "out_of_domain_detection.";
  return name.starts_with(kPrefix);
}

bool has_command_interface(
  const std::vector<std::string> & command_interfaces,
  const std::string & interface_name)
{
  return std::find(
    command_interfaces.begin(), command_interfaces.end(), interface_name) !=
         command_interfaces.end();
}

bool is_supported_hardware_command_interface(const std::string & interface_name)
{
  constexpr std::array<std::string_view, 5> kSupportedInterfaces{
    "position", "velocity", "effort", "kp", "kd"};
  return std::find(
    kSupportedInterfaces.begin(), kSupportedInterfaces.end(), interface_name) !=
         kSupportedInterfaces.end();
}

rclcpp::Logger get_logger_or_default(const SafetyController & controller)
{
  try {
    const auto node = controller.get_node();
    if (node) {
      return node->get_logger();
    }
  } catch (const std::exception &) {
  }
  return rclcpp::get_logger("SafetyController");
}

controller_manager_msgs::srv::SwitchController::Request make_emergency_switch_request(
  const std::string & emergency_controller,
  double switch_timeout_s,
  const std::vector<std::string> & emergency_deactivate_controllers)
{
  using SwitchController = controller_manager_msgs::srv::SwitchController;

  SwitchController::Request request;
  request.activate_controllers = {emergency_controller};
  request.deactivate_controllers = emergency_deactivate_controllers;
  // TODO(lgulich): Remove this explicit deactivate workaround after moving to ROS Kilted
  // or newer, where controller_manager's FORCE_AUTO mode deactivates conflicting controllers.
  request.strictness = request.deactivate_controllers.empty() ?
    SwitchController::Request::FORCE_AUTO :
    SwitchController::Request::BEST_EFFORT;
  request.activate_asap = true;

  if (!std::isfinite(switch_timeout_s) || switch_timeout_s <= 0.0) {
    request.timeout.sec = 0;
    request.timeout.nanosec = 0;
    return request;
  }

  const auto max_seconds = static_cast<double>(std::numeric_limits<int32_t>::max());
  const double clamped_timeout_s = std::min(switch_timeout_s, max_seconds);
  const auto seconds = static_cast<int32_t>(std::floor(clamped_timeout_s));
  const double fractional_s = clamped_timeout_s - static_cast<double>(seconds);
  request.timeout.sec = seconds;
  request.timeout.nanosec = static_cast<uint32_t>(
    std::llround(fractional_s * static_cast<double>(kNanosecondsPerSecond)));
  if (request.timeout.nanosec >= kNanosecondsPerSecond) {
    request.timeout.nanosec = 0;
    if (request.timeout.sec < std::numeric_limits<int32_t>::max()) {
      ++request.timeout.sec;
    }
  }

  return request;
}

}  // namespace

SafetyController::SafetyController()
: controller_interface::ChainableControllerInterface()
{
}

controller_interface::CallbackReturn SafetyController::on_init()
{
  try {
    auto_declare<std::string>("blend_strategy", "interpolate");
    auto_declare<std::string>("blend_reference", "activation");
    // interpolate_max_velocity is a per-joint regex map (like kp/kd), resolved from
    // parameter overrides in create_safety_controller — not declared as a scalar here.

    // Joints
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>());
    auto_declare<std::vector<std::string>>(
      "hardware_command_interfaces", {"position", "velocity", "effort", "kp", "kd"});

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
    range.step = 0.001;
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
    speed_range.to_value = 10.0;
    speed_range.step = 0.01;
    max_speed_desc.floating_point_range.push_back(speed_range);
    get_node()->declare_parameter(
      "max_blend_ratio_speed", rclcpp::ParameterValue(1.0), max_speed_desc);

    auto_declare<double>("out_of_domain_detection.velocity_threshold.max_velocity", 0.0);
    auto_declare<double>("out_of_domain_detection.velocity_threshold.mean_velocity", 0.0);
    auto_declare<std::vector<std::string>>(
      "out_of_domain_detection.velocity_threshold.excluded_joints",
      std::vector<std::string>());
    auto_declare<std::string>("out_of_domain_detection.emergency_controller", "");
    auto_declare<double>("out_of_domain_detection.switch_timeout", 2.0);
    auto_declare<std::vector<std::string>>(
      "out_of_domain_detection.deactivate_controllers",
      std::vector<std::string>());

    auto_declare<bool>("publish_scaled_joint_delta", false);
    auto_declare<std::string>("scaled_joint_delta_topic", "~/scaled_joint_delta");
    auto_declare<bool>("publish_blended_command", false);
    auto_declare<std::string>("blended_command_topic", "~/blended_command");

    // Gravity compensation: path to the (fixed-base) URDF used to build the Pinocchio
    // model, and the joints the gravity torque is applied to. Both must be set to
    // enable it; the torque is computed on the clamped position and applied at full
    // magnitude regardless of blend_ratio.
    auto_declare<std::string>("gravity_compensation_urdf_path", "");
    auto_declare<std::vector<std::string>>(
      "gravity_compensation_joints", std::vector<std::string>());
    // overwrite: gravity replaces effort; blend: crossfade into policy effort by blend_ratio.
    auto_declare<std::string>("gravity_compensation_mode", "overwrite");
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
      "Unknown blend_strategy '%s'. Supported values: interpolate, "
      "no_post_processing",
      blend_strategy_str.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  blend_strategy_ = blend_strategy.value();

  const auto blend_reference_str =
    get_node()->get_parameter("blend_reference").as_string();
  const auto blend_reference = parse_blend_reference(blend_reference_str);
  if (!blend_reference.has_value()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Unknown blend_reference '%s'. Supported values: activation, current",
      blend_reference_str.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  blend_reference_ = blend_reference.value();

  joint_names_ = get_node()->get_parameter("joints").as_string_array();
  hardware_command_interfaces_ =
    get_node()->get_parameter("hardware_command_interfaces").as_string_array();
  target_blend_ratio_.store(get_node()->get_parameter("blend_ratio").as_double());
  max_blend_ratio_speed_.store(get_node()->get_parameter("max_blend_ratio_speed").as_double());
  per_joint_kp_ = utils::resolve_gains_from_params(*get_node(), "kp", joint_names_);
  per_joint_kd_ = utils::resolve_gains_from_params(*get_node(), "kd", joint_names_);

  if (joint_names_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joints specified");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Optional per-joint default ("home") position for interpolate (gr00t regex
  // convention). Empty when not configured; joints matched by no pattern come back NaN
  // and the strategy leaves them at the measured activation pose.
  try {
    per_joint_default_position_ = utils::resolve_default_position(*get_node(), joint_names_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to resolve per-joint default_position: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (hardware_command_interfaces_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No hardware_command_interfaces specified");
    return controller_interface::CallbackReturn::ERROR;
  }
  for (size_t i = 0; i < hardware_command_interfaces_.size(); ++i) {
    const auto & interface_name = hardware_command_interfaces_[i];
    if (!is_supported_hardware_command_interface(interface_name)) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Unsupported hardware command interface '%s'",
        interface_name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    const auto first = std::find(
      hardware_command_interfaces_.begin(), hardware_command_interfaces_.end(), interface_name);
    if (static_cast<size_t>(std::distance(hardware_command_interfaces_.begin(), first)) != i) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Duplicate hardware command interface '%s'",
        interface_name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }
  if (!has_command_interface(hardware_command_interfaces_, "position")) {
    RCLCPP_ERROR(get_node()->get_logger(), "hardware_command_interfaces must include position");
    return controller_interface::CallbackReturn::ERROR;
  }
  load_out_of_domain_detection_params();
  resolve_excluded_joint_indices();

  scaled_joint_delta_publisher_.reset();
  scaled_joint_delta_realtime_publisher_.reset();
  if (publish_scaled_joint_delta_) {
    scaled_joint_delta_publisher_ = get_node()->create_publisher<JointCommandMsg>(
      scaled_joint_delta_topic_, rclcpp::SystemDefaultsQoS());
    scaled_joint_delta_realtime_publisher_ =
      std::make_shared<JointCommandPublisher>(scaled_joint_delta_publisher_);

    auto & msg = scaled_joint_delta_msg_;
    msg.names = joint_names_;
    msg.position.assign(joint_names_.size(), 0.0);
    msg.velocity.assign(joint_names_.size(), 0.0);
    msg.effort.assign(joint_names_.size(), 0.0);
    msg.kp.assign(joint_names_.size(), 0.0);
    msg.kd.assign(joint_names_.size(), 0.0);

    RCLCPP_INFO(
      get_node()->get_logger(),
      "Publishing scaled SafetyController joint deltas on %s",
      scaled_joint_delta_topic_.c_str());
  }

  blended_command_publisher_.reset();
  blended_command_realtime_publisher_.reset();
  if (publish_blended_command_) {
    blended_command_publisher_ = get_node()->create_publisher<JointCommandMsg>(
      blended_command_topic_, rclcpp::SystemDefaultsQoS());
    blended_command_realtime_publisher_ =
      std::make_shared<JointCommandPublisher>(blended_command_publisher_);

    auto & msg = blended_command_msg_;
    msg.names = joint_names_;
    msg.position.assign(joint_names_.size(), 0.0);
    msg.velocity.assign(joint_names_.size(), 0.0);
    msg.effort.assign(joint_names_.size(), 0.0);
    msg.kp.assign(joint_names_.size(), 0.0);
    msg.kd.assign(joint_names_.size(), 0.0);

    RCLCPP_INFO(
      get_node()->get_logger(),
      "Publishing blended SafetyController joint commands on %s",
      blended_command_topic_.c_str());
  }

  id_solver_.reset();
  gravity_apply_.assign(joint_names_.size(), false);
  const auto gravity_urdf =
    get_node()->get_parameter("gravity_compensation_urdf_path").as_string();
  const auto gravity_joints =
    get_node()->get_parameter("gravity_compensation_joints").as_string_array();
  const auto gravity_mode_str =
    get_node()->get_parameter("gravity_compensation_mode").as_string();
  if (gravity_mode_str == "blend") {
    gravity_mode_ = GravityMode::kBlend;
  } else {
    gravity_mode_ = GravityMode::kOverwrite;
    if (gravity_mode_str != "overwrite") {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Unknown gravity_compensation_mode '%s'; using 'overwrite'. "
        "Supported values: overwrite, blend.",
        gravity_mode_str.c_str());
    }
  }
  if (!gravity_urdf.empty() && !gravity_joints.empty()) {
    try {
      id_solver_.emplace(gravity_urdf, joint_names_);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to build inverse-dynamics solver from '%s': %s",
        gravity_urdf.c_str(), e.what());
      return controller_interface::CallbackReturn::ERROR;
    }
    const auto n = joint_names_.size();
    gravity_q_ = Eigen::VectorXd::Zero(n);
    gravity_tau_ = Eigen::VectorXd::Zero(n);

    for (size_t i = 0; i < n; ++i) {
      gravity_apply_[i] = std::find(
        gravity_joints.begin(), gravity_joints.end(), joint_names_[i]) !=
        gravity_joints.end();
    }
    RCLCPP_INFO(
      get_node()->get_logger(),
      "Gravity compensation enabled (model '%s') for %zu of %zu joints",
      gravity_urdf.c_str(), gravity_joints.size(), joint_names_.size());
  } else if (!gravity_joints.empty() && gravity_urdf.empty()) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "gravity_compensation_joints is set (%zu joints) but gravity_compensation_urdf_path "
      "is empty; gravity compensation is disabled.",
      gravity_joints.size());
  }

  // Set up dynamic parameter callback for blend_ratio
  param_callback_handle_ = get_node()->add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & parameters) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;

      for (const auto & param : parameters) {
        if (is_out_of_domain_detection_parameter(param.get_name())) {
          result.successful = false;
          result.reason =
          "out_of_domain_detection parameters are static; reconfigure the controller";
          return result;
        }
        if (param.get_name() == "blend_ratio") {
          const double value = param.as_double();
          if (value < 0.0 || value > 1.0) {
            result.successful = false;
            result.reason = "blend_ratio must be between 0.0 and 1.0";
            return result;
          }
        } else if (param.get_name() == "max_blend_ratio_speed") {
          const double value = param.as_double();
          if (value < 0.0) {
            result.successful = false;
            result.reason = "max_blend_ratio_speed must be non-negative";
            return result;
          }
        }
      }

      for (const auto & param : parameters) {
        if (param.get_name() == "blend_ratio") {
          const double value = param.as_double();
          target_blend_ratio_.store(value);
          RCLCPP_DEBUG(
            get_node()->get_logger(), "Updated blend_ratio target to: %.3f", value);
        } else if (param.get_name() == "max_blend_ratio_speed") {
          const double value = param.as_double();
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

  configure_emergency_switch_client();

  // Initialize reference interfaces storage with NaN
  reference_interfaces_.resize(
    joint_names_.size() * kInterfacesPerJoint,
    std::numeric_limits<double>::quiet_NaN());

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Configured SafetyController with %zu joints, blend_strategy: %s, "
    "blend_reference: %s, blend_ratio: %.3f, max_blend_ratio_speed: %.3f",
    joint_names_.size(), to_string(blend_strategy_), to_string(blend_reference_),
    target_blend_ratio_.load(), max_blend_ratio_speed_.load());

  RCLCPP_INFO(
    get_node()->get_logger(),
    "SafetyController out_of_domain velocity check: enabled=%s, emergency_controller='%s', "
    "max_velocity=%.3f rad/s, mean_velocity=%.3f rad/s",
    velocity_threshold_enabled_ ? "true" : "false", emergency_controller_.c_str(),
    max_joint_velocity_, mean_joint_velocity_);

  return controller_interface::CallbackReturn::SUCCESS;
}

void SafetyController::load_out_of_domain_detection_params()
{
  max_joint_velocity_ = get_node()
    ->get_parameter("out_of_domain_detection.velocity_threshold.max_velocity")
    .as_double();
  mean_joint_velocity_ = get_node()
    ->get_parameter("out_of_domain_detection.velocity_threshold.mean_velocity")
    .as_double();
  excluded_joint_patterns_ = get_node()
    ->get_parameter("out_of_domain_detection.velocity_threshold.excluded_joints")
    .as_string_array();

  emergency_controller_ =
    get_node()->get_parameter("out_of_domain_detection.emergency_controller").as_string();
  emergency_switch_timeout_s_ =
    get_node()->get_parameter("out_of_domain_detection.switch_timeout").as_double();
  configured_emergency_deactivate_controllers_ =
    get_node()->get_parameter("out_of_domain_detection.deactivate_controllers").as_string_array();
  publish_scaled_joint_delta_ =
    get_node()->get_parameter("publish_scaled_joint_delta").as_bool();
  scaled_joint_delta_topic_ =
    get_node()->get_parameter("scaled_joint_delta_topic").as_string();
  if (publish_scaled_joint_delta_ && scaled_joint_delta_topic_.empty()) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "publish_scaled_joint_delta is true but scaled_joint_delta_topic is empty; "
      "using ~/scaled_joint_delta");
    scaled_joint_delta_topic_ = "~/scaled_joint_delta";
  }
  publish_blended_command_ =
    get_node()->get_parameter("publish_blended_command").as_bool();
  blended_command_topic_ =
    get_node()->get_parameter("blended_command_topic").as_string();
  if (publish_blended_command_ && blended_command_topic_.empty()) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "publish_blended_command is true but blended_command_topic is empty; "
      "using ~/blended_command");
    blended_command_topic_ = "~/blended_command";
  }
  velocity_threshold_enabled_ = !emergency_controller_.empty() &&
    (max_joint_velocity_ > 0.0 || mean_joint_velocity_ > 0.0);
}

void SafetyController::resolve_excluded_joint_indices()
{
  excluded_joint_indices_.clear();
  for (const auto & pattern : excluded_joint_patterns_) {
    try {
      const std::regex joint_regex(pattern);
      for (size_t i = 0; i < joint_names_.size(); ++i) {
        if (std::regex_match(joint_names_[i], joint_regex)) {
          excluded_joint_indices_.push_back(i);
        }
      }
    } catch (const std::regex_error & e) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Ignoring invalid excluded joint regex '%s': %s", pattern.c_str(), e.what());
    }
  }

  std::sort(excluded_joint_indices_.begin(), excluded_joint_indices_.end());
  excluded_joint_indices_.erase(
    std::unique(excluded_joint_indices_.begin(), excluded_joint_indices_.end()),
    excluded_joint_indices_.end());
}

bool SafetyController::create_safety_controller()
{
  try {
    isaac_deploy_core::SafetyControllerConfig config;
    config.blend_ratio.type = blend_strategy_;
    config.blend_ratio.reference = blend_reference_;
    if (blend_strategy_ == isaac_deploy_core::BlendStrategy::kInterpolate) {
      auto max_velocities = utils::resolve_gains_from_params(
        *get_node(), "interpolate_max_velocity", joint_names_);
      // Allowlist convention: joints left unmatched default to 0.0 and are intentionally
      // unconstrained (silent). Only warn on an explicitly negative value, which usually
      // signals a config mistake.
      for (size_t i = 0; i < max_velocities.size(); ++i) {
        if (max_velocities[i] < 0.0) {
          RCLCPP_WARN(
            get_node()->get_logger(),
            "interpolate_max_velocity for joint '%s' is negative (%.3f); its "
            "velocity will NOT be constrained.",
            joint_names_[i].c_str(), max_velocities[i]);
        }
      }
      config.blend_ratio.max_velocities = std::move(max_velocities);
      config.blend_ratio.default_position = per_joint_default_position_;
    }
    joint_velocities_input_index_.reset();

    if (velocity_threshold_enabled_) {
      config.out_of_domain_detection.velocity_threshold.input_names = {"joint_velocities"};
      config.out_of_domain_detection.velocity_threshold.max_velocity = max_joint_velocity_;
      config.out_of_domain_detection.velocity_threshold.mean_velocity = mean_joint_velocity_;
    }

    auto controller_result = isaac_deploy_core::SafetyController::create(config);
    if (!controller_result.has_value()) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Failed to create SafetyController: %s",
        std::string(controller_result.error().message).c_str());
      return false;
    }

    safety_controller_.emplace(std::move(controller_result.value()));

    // Pre-allocate tensors.
    // Inputs: command_positions, current_positions, blend_ratio, dt, optional joint_velocities
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

    if (velocity_threshold_enabled_) {
      joint_velocities_input_index_ = inputs_.size();
      inputs_.push_back(utils::create_preallocated_tensor("joint_velocities", {1, num_joints}));
      input_specs_.push_back(utils::create_joint_tensor_spec(joint_names_));
    }

    // Outputs: safe_positions [1, N]
    outputs_.clear();
    output_specs_.clear();
    outputs_.push_back(utils::create_preallocated_tensor("safe_positions", {1, num_joints}));
    output_specs_.push_back(utils::create_joint_tensor_spec(joint_names_));

    stale_position_hold_.assign(joint_names_.size(), 0.0);
    stale_position_hold_active_ = false;

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

  for (const auto & joint_name : joint_names_) {
    for (const auto & interface_name : hardware_command_interfaces_) {
      config.names.push_back(joint_name + "/" + interface_name);
    }
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
  if (velocity_threshold_enabled_) {
    for (const auto & joint_name : joint_names_) {
      config.names.push_back(joint_name + "/velocity");
    }
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

std::vector<hardware_interface::StateInterface>
SafetyController::on_export_state_interfaces()
{
  // Mirror on_export_reference_interfaces() as read-only state interfaces,
  // pointing at the same buffers.  JointCommandBroadcaster reads these.
  std::vector<hardware_interface::StateInterface> state_interfaces;

  const std::string controller_name = get_node()->get_name();

  for (size_t i = 0; i < joint_names_.size(); ++i) {
    state_interfaces.emplace_back(
      controller_name, joint_names_[i] + "/position_raw",
      &reference_interfaces_[i * kInterfacesPerJoint + 0]);
    state_interfaces.emplace_back(
      controller_name, joint_names_[i] + "/velocity_raw",
      &reference_interfaces_[i * kInterfacesPerJoint + 1]);
    state_interfaces.emplace_back(
      controller_name, joint_names_[i] + "/effort_raw",
      &reference_interfaces_[i * kInterfacesPerJoint + 2]);
    state_interfaces.emplace_back(
      controller_name, joint_names_[i] + "/kp_raw",
      &reference_interfaces_[i * kInterfacesPerJoint + 3]);
    state_interfaces.emplace_back(
      controller_name, joint_names_[i] + "/kd_raw",
      &reference_interfaces_[i * kInterfacesPerJoint + 4]);
  }

  return state_interfaces;
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

  configure_emergency_switch_client();

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
  velocity_state_indices_.clear();
  if (velocity_threshold_enabled_) {
    std::vector<std::string> velocity_names;
    velocity_names.reserve(joint_names_.size());
    for (const auto & name : joint_names_) {
      velocity_names.push_back(name + "/velocity");
    }
    auto vel_indices = utils::find_state_interface_indices(velocity_names, state_interfaces_);
    if (!vel_indices.has_value()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to find joint velocity state interfaces");
      return controller_interface::CallbackReturn::ERROR;
    }
    velocity_state_indices_ = std::move(vel_indices.value());
  }

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
    reference_interfaces_[i * kInterfacesPerJoint + 3] = per_joint_kp_[i];  // kp
    reference_interfaces_[i * kInterfacesPerJoint + 4] = per_joint_kd_[i];  // kd
  }

  // Initialize current blend ratio to target so there's no ramp on first activation
  current_blend_ratio_ = target_blend_ratio_.load();
  stale_position_hold_active_ = false;

  RCLCPP_INFO(
    get_node()->get_logger(), "Using blend_ratio parameter, initial value: %.3f",
    target_blend_ratio_.load());

  // Cache command interface indices.
  auto command_names = [this](const std::string & interface_name) {
      std::vector<std::string> names;
      names.reserve(joint_names_.size());
      for (const auto & joint_name : joint_names_) {
        names.push_back(joint_name + "/" + interface_name);
      }
      return names;
    };
  auto cache_command_interfaces = [this, &command_names](
    const std::string & interface_name, std::vector<size_t> & indices) {
      if (!has_command_interface(hardware_command_interfaces_, interface_name)) {
        indices.clear();
        return true;
      }
      auto maybe_indices = utils::find_command_interface_indices(
        command_names(interface_name), command_interfaces_);
      if (!maybe_indices) {
        RCLCPP_ERROR(
          get_node()->get_logger(), "Failed to find %s command interfaces",
          interface_name.c_str());
        return false;
      }
      indices = std::move(maybe_indices.value());
      return true;
    };

  position_command_indices_.clear();
  velocity_command_indices_.clear();
  effort_command_indices_.clear();
  kp_command_indices_.clear();
  kd_command_indices_.clear();

  const std::array<std::pair<const char *, std::vector<size_t> *>, 5> command_indices{
    {{"position", &position_command_indices_}, {"velocity", &velocity_command_indices_},
      {"effort", &effort_command_indices_}, {"kp", &kp_command_indices_},
      {"kd", &kd_command_indices_}}};

  for (const auto & [interface_name, indices] : command_indices) {
    if (!cache_command_interfaces(interface_name, *indices)) {
      return controller_interface::CallbackReturn::ERROR;
    }
  }

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
  emergency_switch_client_.reset();

  if (safety_controller_) {
    auto result = safety_controller_->deactivate();
    if (!result.has_value()) {
      RCLCPP_WARN(
        get_node()->get_logger(), "Failed to deactivate SafetyController: %s",
        std::string(result.error().message).c_str());
    }
  }

  position_state_indices_.clear();
  velocity_state_indices_.clear();
  position_command_indices_.clear();
  velocity_command_indices_.clear();
  effort_command_indices_.clear();
  kp_command_indices_.clear();
  kd_command_indices_.clear();
  stale_position_hold_active_ = false;

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
    const auto error_message = std::string(result.error().message);
    if (velocity_threshold_enabled_ &&
      result.error().code == isaac_deploy_core::Error::Code::kFailedPrecondition)
    {
      latch_emergency(error_message);
      return controller_interface::return_type::OK;
    }

    RCLCPP_ERROR(
      get_node()->get_logger(), "Safety controller advance failed: %s",
      error_message.c_str());
    return controller_interface::return_type::ERROR;
  }

  // Write safe positions to command interfaces.
  write_outputs_to_interfaces(time);

  return controller_interface::return_type::OK;
}

void SafetyController::populate_position_commands_from_references()
{
  bool has_invalid_position = false;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    if (std::isnan(reference_interfaces_[i * kInterfacesPerJoint])) {
      has_invalid_position = true;
      break;
    }
  }

  const auto current_pos_accessor = inputs_[1].tensor.accessor<float, 2>();
  if (has_invalid_position && !stale_position_hold_active_) {
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      stale_position_hold_[i] = static_cast<double>(current_pos_accessor[0][i]);
    }
    stale_position_hold_active_ = true;
    RCLCPP_WARN(
      get_node()->get_logger(),
      "Position reference became invalid; holding measured positions captured at stale entry");
  } else if (!has_invalid_position && stale_position_hold_active_) {
    stale_position_hold_active_ = false;
    RCLCPP_INFO(get_node()->get_logger(), "Position reference recovered; resuming normal blending");
  }

  auto cmd_pos_accessor = inputs_[0].tensor.accessor<float, 2>();
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const double pos_cmd = stale_position_hold_active_ ?
      stale_position_hold_[i] : reference_interfaces_[i * kInterfacesPerJoint];
    cmd_pos_accessor[0][i] = static_cast<float>(pos_cmd);
  }
}

void SafetyController::populate_inputs_from_interfaces(
  const rclcpp::Duration & period, int64_t timestamp_ns)
{
  // current_positions from state interfaces
  if (!utils::populate_tensor_from_state_interfaces(
      inputs_[1], position_state_indices_, state_interfaces_, timestamp_ns))
  {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "Failed to read one or more joint position state interfaces — using stale values");
  }

  // command_positions from reference interfaces (index 0 in layout: pos, vel, eff, kp, kd)
  populate_position_commands_from_references();
  inputs_[0].timestamp_ns = timestamp_ns;

  // Rate-limit blend_ratio toward target
  const double max_delta = max_blend_ratio_speed_.load() * period.seconds();
  const double error = target_blend_ratio_.load() - current_blend_ratio_;
  current_blend_ratio_ += std::clamp(error, -max_delta, max_delta);

  auto blend_ratio_accessor = inputs_[2].tensor.accessor<float, 2>();
  blend_ratio_accessor[0][0] =
    stale_position_hold_active_ ? 1.0F : static_cast<float>(current_blend_ratio_);
  inputs_[2].timestamp_ns = timestamp_ns;

  // dt
  auto dt_accessor = inputs_[3].tensor.accessor<float, 2>();
  dt_accessor[0][0] = static_cast<float>(period.seconds());
  inputs_[3].timestamp_ns = timestamp_ns;

  if (velocity_threshold_enabled_ && joint_velocities_input_index_.has_value()) {
    auto & velocity_input = inputs_[joint_velocities_input_index_.value()];
    if (!utils::populate_tensor_from_state_interfaces(
        velocity_input, velocity_state_indices_, state_interfaces_, timestamp_ns))
    {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "Failed to read one or more joint velocity state interfaces — using stale values");
    }
    auto velocity_accessor = velocity_input.tensor.accessor<float, 2>();
    for (const auto joint_index : excluded_joint_indices_) {
      velocity_accessor[0][joint_index] = 0.0f;
    }
  }
}

void SafetyController::write_outputs_to_interfaces(const rclcpp::Time & time)
{
  const auto safe_positions = outputs_[0].tensor.accessor<float, 2>();
  const auto command_positions = inputs_[0].tensor.accessor<float, 2>();
  const auto current_positions = inputs_[1].tensor.accessor<float, 2>();
  const auto blend_ratio = static_cast<double>(inputs_[2].tensor.accessor<float, 2>()[0][0]);

  JointCommandMsg * scaled_delta_msg =
    scaled_joint_delta_realtime_publisher_ ? &scaled_joint_delta_msg_ : nullptr;
  if (scaled_delta_msg) {
    scaled_delta_msg->header.stamp = time;
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    std::fill(scaled_delta_msg->position.begin(), scaled_delta_msg->position.end(), kNaN);
    std::fill(scaled_delta_msg->velocity.begin(), scaled_delta_msg->velocity.end(), kNaN);
    std::fill(scaled_delta_msg->effort.begin(), scaled_delta_msg->effort.end(), kNaN);
    std::fill(scaled_delta_msg->kp.begin(), scaled_delta_msg->kp.end(), kNaN);
    std::fill(scaled_delta_msg->kd.begin(), scaled_delta_msg->kd.end(), kNaN);
  }

  // Blended / safety-limited absolute command actually written to hardware this cycle.
  JointCommandMsg * blended_msg =
    blended_command_realtime_publisher_ ? &blended_command_msg_ : nullptr;
  if (blended_msg) {
    blended_msg->header.stamp = time;
  }

  // Write safe_positions to position command interfaces.
  utils::write_tensor_to_command_interfaces(
    outputs_[0], position_command_indices_, command_interfaces_);

  if (scaled_delta_msg) {
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      scaled_delta_msg->position[i] =
        blend_ratio * (static_cast<double>(command_positions[0][i]) -
        static_cast<double>(current_positions[0][i]));
    }
  }

  if (id_solver_) {
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      gravity_q_(static_cast<Eigen::Index>(i)) = static_cast<double>(safe_positions[0][i]);
    }
    id_solver_->computeGravityCompensation(gravity_q_, gravity_tau_);
  }

  // Write velocity, effort, kp, kd according to the active blend strategy.
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    double vel_cmd = reference_interfaces_[i * kInterfacesPerJoint + 1];
    double eff_cmd = reference_interfaces_[i * kInterfacesPerJoint + 2];
    double kp_cmd = reference_interfaces_[i * kInterfacesPerJoint + 3];
    double kd_cmd = reference_interfaces_[i * kInterfacesPerJoint + 4];

    if (blend_strategy_ == isaac_deploy_core::BlendStrategy::kInterpolate) {
      // Scale velocity and effort by blend ratio; pass gains through unchanged from the policy.
      vel_cmd = std::isnan(vel_cmd) ? 0.0 : current_blend_ratio_ * vel_cmd;
      eff_cmd = std::isnan(eff_cmd) ? 0.0 : current_blend_ratio_ * eff_cmd;
      kp_cmd = std::isnan(kp_cmd) ? per_joint_kp_[i] : kp_cmd;
      kd_cmd = std::isnan(kd_cmd) ? per_joint_kd_[i] : kd_cmd;
    } else {
      // Pass through: NaN replaced with zero for vel/eff, per-joint gains for kp/kd.
      if (std::isnan(vel_cmd)) {vel_cmd = 0.0;}
      if (std::isnan(eff_cmd)) {eff_cmd = 0.0;}
      if (std::isnan(kp_cmd)) {kp_cmd = per_joint_kp_[i];}
      if (std::isnan(kd_cmd)) {kd_cmd = per_joint_kd_[i];}
    }

    // eff_cmd already holds blend * policy_eff. kOverwrite replaces it with gravity;
    // kBlend adds (1 - blend) * gravity so gravity is full at blend 0 and gone at 1.
    if (id_solver_ && gravity_apply_[i]) {
      const double gravity_tau = gravity_tau_(static_cast<Eigen::Index>(i));
      eff_cmd = (gravity_mode_ == GravityMode::kOverwrite) ?
        gravity_tau :
        eff_cmd + (1.0 - current_blend_ratio_) * gravity_tau;
    }

    if (!velocity_command_indices_.empty()) {
      (void)command_interfaces_[velocity_command_indices_[i]].set_value(vel_cmd);
    }
    if (!effort_command_indices_.empty()) {
      (void)command_interfaces_[effort_command_indices_[i]].set_value(eff_cmd);
    }
    if (!kp_command_indices_.empty()) {
      (void)command_interfaces_[kp_command_indices_[i]].set_value(kp_cmd);
    }
    if (!kd_command_indices_.empty()) {
      (void)command_interfaces_[kd_command_indices_[i]].set_value(kd_cmd);
    }

    if (blended_msg) {
      blended_msg->position[i] = static_cast<double>(safe_positions[0][i]);
      blended_msg->velocity[i] = vel_cmd;
      blended_msg->effort[i] = eff_cmd;
      blended_msg->kp[i] = kp_cmd;
      blended_msg->kd[i] = kd_cmd;
    }
  }

  if (scaled_delta_msg) {
    scaled_joint_delta_realtime_publisher_->try_publish(*scaled_delta_msg);
  }

  if (blended_msg) {
    blended_command_realtime_publisher_->try_publish(*blended_msg);
  }
}

void SafetyController::configure_emergency_switch_client()
{
  using namespace std::chrono_literals;

  emergency_switch_client_.reset();
  {
    std::lock_guard<std::mutex> lock(emergency_reason_mutex_);
    emergency_reason_.clear();
  }
  if (!velocity_threshold_enabled_) {
    return;
  }

  const auto node = get_node();
  const auto logger = node->get_logger();
  const auto clock = node->get_clock();
  const auto emergency_controller = emergency_controller_;
  const auto request = build_emergency_switch_request();
  emergency_switch_client_.configure(
    node,
    kSwitchControllerService,
    20ms,
    std::max(1.0, emergency_switch_timeout_s_ + kEmergencySwitchWatchdogGraceS),
    [request]() {
      return request;
    },
    [logger, emergency_controller](
      rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedFuture future) {
      try {
        const auto response = future.get();
        if (!response || !response->ok) {
          RCLCPP_ERROR(
            logger, "Emergency controller switch to '%s' failed", emergency_controller.c_str());
          return false;
        }
      } catch (const std::exception & e) {
        RCLCPP_ERROR(
          logger, "Emergency controller switch to '%s' failed: %s",
          emergency_controller.c_str(), e.what());
        return false;
      }
      return true;
    },
    [logger, clock, emergency_controller]() {
      RCLCPP_ERROR_THROTTLE(
        logger, *clock, 1000,
        "Emergency switch to controller '%s' is pending, but service '%s' is not ready",
        emergency_controller.c_str(), kSwitchControllerService);
    },
    [logger, emergency_controller](const std::string & error) {
      RCLCPP_ERROR(
        logger, "Failed to send emergency controller switch to '%s': %s",
        emergency_controller.c_str(), error.c_str());
    },
    [logger, clock, emergency_controller](double response_timeout_s) {
      RCLCPP_ERROR_THROTTLE(
        logger, *clock, 1000,
        "Emergency switch to controller '%s' did not receive a response within %.3f seconds",
        emergency_controller.c_str(), response_timeout_s);
    });
}

controller_manager_msgs::srv::SwitchController::Request
SafetyController::build_emergency_switch_request() const
{
  return make_emergency_switch_request(
    emergency_controller_, emergency_switch_timeout_s_,
    configured_emergency_deactivate_controllers_);
}

void SafetyController::latch_emergency(const std::string & reason)
{
  if (!emergency_switch_client_.trigger_once()) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(emergency_reason_mutex_);
    emergency_reason_ = reason;
  }

  RCLCPP_ERROR(
    get_logger_or_default(*this),
    "Safety emergency latched: %s. Requesting emergency controller '%s'",
    reason.c_str(), emergency_controller_.c_str());
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::SafetyController,
  controller_interface::ChainableControllerInterface)

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

#include "isaac_ros_deploy_ros2_control/controllers/joint_parameter_controller.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <rcl_interfaces/msg/floating_point_range.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <urdf/model.hpp>

#include "isaac_ros_deploy_ros2_control/utils/gain_utils.hpp"
#include "isaac_ros_deploy_ros2_control/utils/tensor_interface_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

namespace
{
constexpr char kKpSuffix[] = "kp";
constexpr char kKdSuffix[] = "kd";
constexpr char kPositionSuffix[] = "position_target";

/// Build a ParameterDescriptor that exposes a floating-point slider in
/// rqt_reconfigure / Foxglove.
///
/// ``step`` is intentionally 0 (continuous) - rclcpp's parameter
/// validator rejects values that don't lie exactly on a non-zero step
/// grid, which would otherwise reject the seeded current-position value
/// on activation (e.g. q = 0.123456 is not a multiple of 0.001). rqt
/// falls back to auto-deriving slider granularity from the range.
rcl_interfaces::msg::ParameterDescriptor make_slider_descriptor(
  const std::string & description, double from, double to)
{
  rcl_interfaces::msg::ParameterDescriptor desc;
  desc.description = description;
  desc.read_only = false;
  desc.dynamic_typing = false;
  rcl_interfaces::msg::FloatingPointRange range;
  range.from_value = from;
  range.to_value = to;
  range.step = 0.0;
  desc.floating_point_range.push_back(range);
  return desc;
}

}  // namespace

JointParameterController::JointParameterController()
: controller_interface::ControllerInterface()
{
}

std::string JointParameterController::parameter_name(
  const std::string & joint_name, const std::string & suffix)
{
  return "joint." + joint_name + "." + suffix;
}

bool JointParameterController::load_joint_limits_from_robot_description()
{
  joint_limits_.assign(joint_names_.size(), JointLimits{});
  const std::string robot_description = get_robot_description();
  if (robot_description.empty()) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "Controller cached robot description is empty; position_target sliders will use fallback "
      "ranges");
    return false;
  }

  urdf::Model urdf_model;
  if (!urdf_model.initString(robot_description)) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "Failed to parse cached robot description; position_target sliders will use fallback "
      "ranges");
    return false;
  }

  size_t parsed = 0;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto joint = urdf_model.getJoint(joint_names_[i]);
    if (!joint || !joint->limits || joint->type == urdf::Joint::CONTINUOUS) {
      continue;
    }
    joint_limits_[i].has_limits = true;
    joint_limits_[i].lower = joint->limits->lower;
    joint_limits_[i].upper = joint->limits->upper;
    ++parsed;
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Loaded position limits for %zu / %zu joints from cached robot description",
    parsed, joint_names_.size());

  if (parsed < joint_names_.size()) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "%zu joint(s) have no position limits; position_target sliders will use fallback range",
      joint_names_.size() - parsed);
  }
  return parsed > 0;
}

controller_interface::CallbackReturn JointParameterController::on_init()
{
  try {
    // Top-level params are read_only: each is consumed once in on_configure,
    // so editing them at runtime would be a false affordance. Only the
    // per-joint params (joint.<j>.{kp,kd,position_target}) are runtime-editable.

    rcl_interfaces::msg::ParameterDescriptor joints_desc;
    joints_desc.description =
      "Ordered list of joint names this controller drives. Configure-time "
      "only - the joint set determines which command/state interfaces are "
      "claimed.";
    joints_desc.read_only = true;
    get_node()->declare_parameter(
      "joints", rclcpp::ParameterValue(std::vector<std::string>{}), joints_desc);

    rcl_interfaces::msg::ParameterDescriptor default_kp_desc;
    default_kp_desc.description =
      "Initial value for each per-joint 'joint.<j>.kp' parameter when no "
      "per-joint YAML override is provided. Configure-time only; runtime "
      "tuning happens via the per-joint sliders.";
    default_kp_desc.read_only = true;
    get_node()->declare_parameter(
      "default_kp", rclcpp::ParameterValue(0.0), default_kp_desc);

    rcl_interfaces::msg::ParameterDescriptor default_kd_desc;
    default_kd_desc.description =
      "Initial value for each per-joint 'joint.<j>.kd' parameter when no "
      "per-joint YAML override is provided. Configure-time only; runtime "
      "tuning happens via the per-joint sliders.";
    default_kd_desc.read_only = true;
    get_node()->declare_parameter(
      "default_kd", rclcpp::ParameterValue(0.0), default_kd_desc);

    // Slider upper bounds for the per-joint kp / kd sliders, baked into each
    // FloatingPointRange at on_configure time. The descriptor is immutable
    // once declared, so these are read_only (see descriptions below).
    rcl_interfaces::msg::ParameterDescriptor kp_max_desc;
    kp_max_desc.description =
      "Upper bound of the per-joint 'joint.<j>.kp' slider. Configure-time "
      "only: rclcpp parameter descriptors are immutable, so changing this "
      "at runtime has no effect on existing sliders. Set via YAML and "
      "reload the controller to widen.";
    kp_max_desc.read_only = true;
    get_node()->declare_parameter("kp_max", rclcpp::ParameterValue(kp_max_), kp_max_desc);

    rcl_interfaces::msg::ParameterDescriptor kd_max_desc;
    kd_max_desc.description =
      "Upper bound of the per-joint 'joint.<j>.kd' slider. Configure-time "
      "only: rclcpp parameter descriptors are immutable, so changing this "
      "at runtime has no effect on existing sliders. Set via YAML and "
      "reload the controller to widen.";
    kd_max_desc.read_only = true;
    get_node()->declare_parameter("kd_max", rclcpp::ParameterValue(kd_max_), kd_max_desc);

    // Override the fallback position-target range used when the URDF
    // does not provide limits for a joint (e.g. continuous joints).
    // Same configure-time-only contract as kp_max / kd_max.
    rcl_interfaces::msg::ParameterDescriptor pos_fallback_min_desc;
    pos_fallback_min_desc.description =
      "Lower bound of the per-joint 'joint.<j>.position_target' slider when "
      "the URDF has no <limit> block (continuous joint). Configure-time only; "
      "set via YAML and reload the controller to widen.";
    pos_fallback_min_desc.read_only = true;
    get_node()->declare_parameter(
      "position_target_fallback_min",
      rclcpp::ParameterValue(position_target_fallback_min_),
      pos_fallback_min_desc);

    rcl_interfaces::msg::ParameterDescriptor pos_fallback_max_desc;
    pos_fallback_max_desc.description =
      "Upper bound of the per-joint 'joint.<j>.position_target' slider when "
      "the URDF has no <limit> block (continuous joint). Configure-time only; "
      "set via YAML and reload the controller to widen.";
    pos_fallback_max_desc.read_only = true;
    get_node()->declare_parameter(
      "position_target_fallback_max",
      rclcpp::ParameterValue(position_target_fallback_max_),
      pos_fallback_max_desc);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn JointParameterController::on_configure(
  const rclcpp_lifecycle::State &)
{
  try {
    joint_names_ = get_node()->get_parameter("joints").as_string_array();
    default_kp_ = get_node()->get_parameter("default_kp").as_double();
    default_kd_ = get_node()->get_parameter("default_kd").as_double();
    kp_max_ = get_node()->get_parameter("kp_max").as_double();
    kd_max_ = get_node()->get_parameter("kd_max").as_double();
    position_target_fallback_min_ =
      get_node()->get_parameter("position_target_fallback_min").as_double();
    position_target_fallback_max_ =
      get_node()->get_parameter("position_target_fallback_max").as_double();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Invalid controller configuration: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (joint_names_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joints specified");
    return controller_interface::CallbackReturn::ERROR;
  }

  const std::unordered_set<std::string> unique_joint_names(
    joint_names_.begin(), joint_names_.end());
  if (unique_joint_names.size() != joint_names_.size()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Joint names must be unique");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (!std::isfinite(default_kp_) || !std::isfinite(default_kd_) ||
    !std::isfinite(kp_max_) || !std::isfinite(kd_max_) ||
    !std::isfinite(position_target_fallback_min_) ||
    !std::isfinite(position_target_fallback_max_))
  {
    RCLCPP_ERROR(get_node()->get_logger(), "Controller configuration values must be finite");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (kp_max_ < 0.0 || kd_max_ < 0.0 ||
    default_kp_ < 0.0 || default_kp_ > kp_max_ ||
    default_kd_ < 0.0 || default_kd_ > kd_max_ ||
    position_target_fallback_min_ >= position_target_fallback_max_)
  {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Invalid controller configuration ranges or default gains");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Limits are best-effort; joints without parsed limits use the configured
  // fallback range, with the loader logging the reason.
  (void)load_joint_limits_from_robot_description();
  // Reuse the same ordered regex-map convention as the other impedance
  // controllers. These values seed the runtime-editable per-joint sliders;
  // an explicit joint.<name>.{kp,kd} override still takes precedence when the
  // slider parameter is declared below.
  std::vector<double> initial_kp;
  std::vector<double> initial_kd;
  try {
    initial_kp = utils::resolve_gains_from_params(
      *get_node(), "kp", joint_names_, default_kp_);
    initial_kd = utils::resolve_gains_from_params(
      *get_node(), "kd", joint_names_, default_kd_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to resolve initial gain maps: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  // Pre-size setpoints and declare per-joint parameters so they are
  // visible in ``ros2 param list`` / rqt / Foxglove immediately after
  // configure. Each parameter is given a ParameterDescriptor with a
  // FloatingPointRange so rqt renders it as a bounded slider.
  {
    std::lock_guard<std::mutex> lock(setpoint_mutex_);
    joint_setpoints_.assign(joint_names_.size(), JointSetpoint{});
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      const auto & jn = joint_names_[i];

      // kp slider: [0, kp_max].
      const auto kp_name = parameter_name(jn, kKpSuffix);
      const auto kp_desc = make_slider_descriptor(
        "Per-joint stiffness for impedance control. Range: [0, kp_max]",
        0.0, kp_max_);
      if (!get_node()->has_parameter(kp_name)) {
        get_node()->declare_parameter(
          kp_name, rclcpp::ParameterValue(initial_kp[i]), kp_desc);
      }

      // kd slider: [0, kd_max].
      const auto kd_name = parameter_name(jn, kKdSuffix);
      const auto kd_desc = make_slider_descriptor(
        "Per-joint damping for impedance control. Range: [0, kd_max]",
        0.0, kd_max_);
      if (!get_node()->has_parameter(kd_name)) {
        get_node()->declare_parameter(
          kd_name, rclcpp::ParameterValue(initial_kd[i]), kd_desc);
      }

      // position_target slider: URDF [lower, upper] when available, else
      // the fallback range.
      double pos_min, pos_max;
      if (joint_limits_[i].has_limits) {
        pos_min = joint_limits_[i].lower;
        pos_max = joint_limits_[i].upper;
      } else {
        pos_min = position_target_fallback_min_;
        pos_max = position_target_fallback_max_;
      }
      const auto pos_desc = make_slider_descriptor(
        joint_limits_[i].has_limits ?
            "Per-joint commanded position (rad). Range from URDF <limit>" :
            "Per-joint commanded position (rad). URDF has no <limit>; using fallback range",
        pos_min, pos_max);
      // Clamp the declared default into the slider range. rclcpp validates
      // the default against the FloatingPointRange and throws
      // InvalidParameterValueException when 0.0 lies outside a joint's
      // travel. The measured position is re-seeded on activation anyway.
      const auto position_name = parameter_name(jn, kPositionSuffix);
      const double pos_default = std::clamp(0.0, pos_min, pos_max);
      if (!get_node()->has_parameter(position_name)) {
        get_node()->declare_parameter(
          position_name, rclcpp::ParameterValue(pos_default), pos_desc);
      }

      joint_setpoints_[i].kp =
        get_node()->get_parameter(kp_name).as_double();
      joint_setpoints_[i].kd =
        get_node()->get_parameter(kd_name).as_double();
      joint_setpoints_[i].position_target =
        get_node()->get_parameter(position_name).as_double();
    }
  }

  // Register the runtime parameter callback so CLI / GUI updates flow
  // through to joint_setpoints_ without needing reconfiguration.
  param_cb_handle_ = get_node()->add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & parameters) {
      return this->on_parameters_set(parameters);
    });

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Configured JointParameterController with %zu joints, default_kp=%.3f, default_kd=%.3f, "
    "kp_max=%.3f, kd_max=%.3f",
    joint_names_.size(), default_kp_, default_kd_, kp_max_, kd_max_);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
JointParameterController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Claim all impedance-control interfaces. The hardware side interprets
  // the combination kp/kd + position + zero velocity/effort feedforward
  // as standard Unitree-style impedance control.
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
JointParameterController::state_interface_configuration() const
{
  // Claim each joint's position state so on_activate can seed
  // position_target from the current measurement.
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint_name : joint_names_) {
    config.names.push_back(joint_name + "/position");
  }
  return config;
}

controller_interface::CallbackReturn JointParameterController::on_activate(
  const rclcpp_lifecycle::State &)
{
  std::lock_guard<std::mutex> interface_cache_lock(interface_cache_mutex_);
  is_active_.store(false);
  position_command_indices_.clear();
  velocity_command_indices_.clear();
  effort_command_indices_.clear();
  kp_command_indices_.clear();
  kd_command_indices_.clear();
  position_state_indices_.clear();

  std::vector<std::string> position_names, velocity_names, effort_names, kp_names, kd_names;
  position_names.reserve(joint_names_.size());
  velocity_names.reserve(joint_names_.size());
  effort_names.reserve(joint_names_.size());
  kp_names.reserve(joint_names_.size());
  kd_names.reserve(joint_names_.size());
  for (const auto & name : joint_names_) {
    position_names.push_back(name + "/position");
    velocity_names.push_back(name + "/velocity");
    effort_names.push_back(name + "/effort");
    kp_names.push_back(name + "/kp");
    kd_names.push_back(name + "/kd");
  }

  auto lookup_cmd = [&](const std::vector<std::string> & names, const char * label,
    std::vector<size_t> & out) -> bool {
      auto indices = utils::find_command_interface_indices(names, command_interfaces_);
      if (!indices.has_value()) {
        RCLCPP_ERROR(
          get_node()->get_logger(),
          "Failed to find %s command interfaces", label);
        return false;
      }
      out = std::move(indices.value());
      return true;
    };

  if (!lookup_cmd(position_names, "position", position_command_indices_) ||
    !lookup_cmd(velocity_names, "velocity", velocity_command_indices_) ||
    !lookup_cmd(effort_names, "effort", effort_command_indices_) ||
    !lookup_cmd(kp_names, "kp", kp_command_indices_) ||
    !lookup_cmd(kd_names, "kd", kd_command_indices_))
  {
    return controller_interface::CallbackReturn::ERROR;
  }

  // Look up the joint position state interfaces we claimed in
  // state_interface_configuration().
  auto state_indices = utils::find_state_interface_indices(position_names, state_interfaces_);
  if (!state_indices.has_value()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to find joint position state interfaces - cannot seed position_target safely");
    return controller_interface::CallbackReturn::ERROR;
  }
  position_state_indices_ = std::move(state_indices.value());

  // Seed each joint's position_target from the current measured position.
  // We push the values back through ``set_parameters`` so rqt /
  // Foxglove see the real starting q on the slider, and so the
  // on_parameters_set callback updates joint_setpoints_ atomically.
  std::vector<rclcpp::Parameter> seeded_params;
  seeded_params.reserve(joint_names_.size());
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto current_q =
      state_interfaces_[position_state_indices_[i]].get_optional<double>();
    if (!current_q.has_value() || !std::isfinite(current_q.value())) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Joint '%s' position state is unavailable or non-finite; refusing unsafe activation",
        joint_names_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    const double q = current_q.value();
    const double lower = joint_limits_[i].has_limits ?
      joint_limits_[i].lower : position_target_fallback_min_;
    const double upper = joint_limits_[i].has_limits ?
      joint_limits_[i].upper : position_target_fallback_max_;
    if (q < lower || q > upper) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Joint '%s' measured position %.6f is outside its safe seed range [%.6f, %.6f]",
        joint_names_[i].c_str(), q, lower, upper);
      return controller_interface::CallbackReturn::ERROR;
    }
    seeded_params.emplace_back(
      parameter_name(joint_names_[i], kPositionSuffix), q);
  }
  if (!seeded_params.empty()) {
    // Atomic set so the on_parameters_set callback updates joint_setpoints_
    // for all joints at once before any update() tick runs.
    const auto result = get_node()->set_parameters_atomically(seeded_params);
    if (!result.successful) {
      RCLCPP_ERROR(
        get_node()->get_logger(),
        "Failed to seed joint position targets; refusing unsafe activation: %s",
        result.reason.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    RCLCPP_INFO(
      get_node()->get_logger(),
      "Seeded position_target from current /joint_states for %zu / %zu joints",
      joint_names_.size(), joint_names_.size());
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "JointParameterController activated for %zu joints",
    joint_names_.size());
  is_active_.store(true);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn JointParameterController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  std::lock_guard<std::mutex> interface_cache_lock(interface_cache_mutex_);
  is_active_.store(false);
  position_command_indices_.clear();
  velocity_command_indices_.clear();
  effort_command_indices_.clear();
  kp_command_indices_.clear();
  kd_command_indices_.clear();
  position_state_indices_.clear();

  RCLCPP_INFO(get_node()->get_logger(), "JointParameterController deactivated");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn JointParameterController::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  // Per-joint parameters are statically typed and therefore cannot be
  // undeclared. Preserve them across cleanup; on_configure reuses their
  // existing values when rebuilding the runtime setpoints.

  // Drop the runtime parameter callback so it does not fire against a
  // stale joint list after cleanup.
  param_cb_handle_.reset();

  {
    std::lock_guard<std::mutex> lock(setpoint_mutex_);
    joint_setpoints_.clear();
  }
  joint_limits_.clear();
  joint_names_.clear();

  RCLCPP_INFO(get_node()->get_logger(), "JointParameterController cleaned up");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type JointParameterController::update(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  const size_t joint_count = joint_names_.size();
  // Snapshot the setpoints under the lock. A small copy keeps the update()
  // loop deterministic even while parameters are being set.
  std::vector<JointSetpoint> snapshot;
  {
    std::lock_guard<std::mutex> lock(setpoint_mutex_);
    snapshot = joint_setpoints_;
  }

  if (snapshot.size() != joint_count) {
    RCLCPP_ERROR_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "JointParameterController setpoints are not configured for every joint");
    return controller_interface::return_type::ERROR;
  }

  // Hold the cache lock for the entire command write. A lifecycle transition
  // obtains the same lock before it clears the index vectors, so update()
  // cannot index a cleared vector or continue writing after deactivation
  // returns.
  std::lock_guard<std::mutex> interface_cache_lock(interface_cache_mutex_);
  if (!is_active_.load() ||
    position_command_indices_.size() != joint_count ||
    velocity_command_indices_.size() != joint_count ||
    effort_command_indices_.size() != joint_count ||
    kp_command_indices_.size() != joint_count ||
    kd_command_indices_.size() != joint_count)
  {
    RCLCPP_ERROR_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "JointParameterController update called without active command interfaces");
    return controller_interface::return_type::ERROR;
  }

  for (size_t i = 0; i < joint_count; ++i) {
    const auto & sp = snapshot[i];
    (void)command_interfaces_[position_command_indices_[i]].set_value(sp.position_target);
    (void)command_interfaces_[velocity_command_indices_[i]].set_value(0.0);
    (void)command_interfaces_[effort_command_indices_[i]].set_value(0.0);
    (void)command_interfaces_[kp_command_indices_[i]].set_value(sp.kp);
    (void)command_interfaces_[kd_command_indices_[i]].set_value(sp.kd);
  }

  return controller_interface::return_type::OK;
}

rcl_interfaces::msg::SetParametersResult JointParameterController::on_parameters_set(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  std::lock_guard<std::mutex> lock(setpoint_mutex_);

  // Stage updates in a copy so a rejected batch never partially mutates the
  // live setpoints: rclcpp discards the whole batch when this callback returns
  // unsuccessful, so joint_setpoints_ must stay unchanged in that case.
  std::vector<JointSetpoint> staged = joint_setpoints_;

  // Only the per-joint parameters (joint.<j>.{kp,kd,position_target}) are
  // mutable at runtime. All top-level params are declared read_only in
  // on_init, so they will never reach this callback. We re-parse every
  // incoming name against the per-joint dotted names; anything that
  // doesn't match is silently ignored (other parameter callbacks may
  // own those names).
  for (const auto & param : parameters) {
    const std::string & name = param.get_name();
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      const auto & jn = joint_names_[i];
      const bool is_kp = (name == parameter_name(jn, kKpSuffix));
      const bool is_kd = (name == parameter_name(jn, kKdSuffix));
      const bool is_pos = (name == parameter_name(jn, kPositionSuffix));
      if (!is_kp && !is_kd && !is_pos) {
        continue;
      }

      // The FloatingPointRange descriptor does not reject NaN / inf, so a
      // ``ros2 param set ... NaN`` would otherwise reach the hardware.
      // Reject non-finite values here and skip storing them.
      const double value = param.as_double();
      if (!std::isfinite(value)) {
        result.successful = false;
        result.reason = name + " must be finite";
        continue;
      }

      if (is_kp) {
        staged[i].kp = value;
      } else if (is_kd) {
        staged[i].kd = value;
      } else {
        staged[i].position_target = value;
      }
    }
  }

  // Commit only when the entire batch validated, matching rclcpp's
  // all-or-nothing rejection semantics.
  if (result.successful) {
    joint_setpoints_ = std::move(staged);
  }

  return result;
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::controllers::JointParameterController,
  controller_interface::ControllerInterface)

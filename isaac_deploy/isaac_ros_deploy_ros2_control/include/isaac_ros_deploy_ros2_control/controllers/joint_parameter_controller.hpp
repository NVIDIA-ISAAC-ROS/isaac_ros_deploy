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

#include <numbers>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

/// Controller that exposes per-joint ``kp``, ``kd``, and ``position_target``
/// as ROS 2 parameters.
///
/// On every ``update`` cycle, the controller writes the current parameter
/// values into the ``position``, ``kp``, and ``kd`` command interfaces for
/// each joint, and holds ``velocity`` / ``effort`` feed-forward at zero.
///
/// Two features distinguish this from a plain "write parameter to interface"
/// controller:
///
///   * **Bounded sliders.** Per-joint ``position_target`` gets a
///     ``FloatingPointRange`` matching the URDF ``<limit>`` block (``kp`` /
///     ``kd`` use ``[0, kp_max]`` / ``[0, kd_max]``) so rqt / Foxglove render
///     sliders bounded to the joint's physical travel.
///   * **Seed-from-state on activation.** ``on_activate`` seeds each
///     ``position_target`` from the joint's measured ``q``, so switching in
///     from a "motors off" controller produces zero commanded delta on the
///     first tick.
///
/// Intended for manual bring-up / verification of ros2_control + hardware
/// integration on a real robot, not for high-rate tracking.
class JointParameterController : public controller_interface::ControllerInterface
{
public:
  JointParameterController();

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  /// Per-joint runtime setpoint bundle.
  struct JointSetpoint
  {
    double kp{0.0};
    double kd{0.0};
    double position_target{0.0};
  };

  /// Per-joint URDF position-limit bundle. ``has_limits`` is false for
  /// continuous joints or when the URDF does not have a parseable limit
  /// block (in which case we fall back to a wide default range so the
  /// parameter is still bounded enough for rqt to render a slider).
  struct JointLimits
  {
    bool has_limits{false};
    double lower{0.0};
    double upper{0.0};
  };

  /// Build the parameter name for (joint, suffix).
  ///
  /// Returns a dotted name like ``joint.<joint_name>.kp`` so the parameters
  /// appear grouped by joint in ``ros2 param list`` and in the Foxglove /
  /// rqt parameter GUIs.
  static std::string parameter_name(
    const std::string & joint_name, const std::string & suffix);

  /// Populate ``joint_limits_`` from controller_manager's cached robot
  /// description. Returns true when at least one joint has position limits.
  /// Joints without position limits get ``has_limits = false``.
  bool load_joint_limits_from_robot_description();

  /// ROS 2 parameter set callback — captures updates to any of the
  /// per-joint parameters into ``joint_setpoints_`` under the setpoint
  /// mutex, so ``update`` sees the new values on the next tick.
  rcl_interfaces::msg::SetParametersResult on_parameters_set(
    const std::vector<rclcpp::Parameter> & parameters);

  // Parameters.
  std::vector<std::string> joint_names_;
  double default_kp_{0.0};
  double default_kd_{0.0};
  // Slider upper bounds for kp / kd descriptors. Declared as top-level
  // params (``kp_max`` / ``kd_max``) so a robot with stiffer joints can
  // widen them without recompiling.
  double kp_max_{500.0};
  double kd_max_{20.0};
  // Fallback range when a joint has no URDF limits (continuous joint, or
  // unparseable). Wide enough to cover most use cases while still
  // bounded enough for rqt to render a slider.
  double position_target_fallback_min_{-std::numbers::pi};
  double position_target_fallback_max_{std::numbers::pi};

  // Per-joint URDF limits, indexed by ``joint_names_``.
  std::vector<JointLimits> joint_limits_;

  // Current setpoints, guarded by setpoint_mutex_. update() takes a copy
  // under the lock and releases it before writing to command interfaces
  // to keep the control loop non-blocking.
  std::mutex setpoint_mutex_;
  std::vector<JointSetpoint> joint_setpoints_;

  // Cached command-interface indices, populated on activate.
  std::vector<size_t> position_command_indices_;
  std::vector<size_t> velocity_command_indices_;
  std::vector<size_t> effort_command_indices_;
  std::vector<size_t> kp_command_indices_;
  std::vector<size_t> kd_command_indices_;

  // Cached state-interface indices for joint position, populated on
  // activate. Used by the seed-from-state logic.
  std::vector<size_t> position_state_indices_;

  // Protects the active flag and all cached interface indices as one
  // lifecycle-owned bundle. update() holds this lock while it writes command
  // interfaces, so on_deactivate() cannot clear indices or return while a
  // command write is still in progress.
  std::mutex interface_cache_mutex_;

  // update() must not access cached interface indices before a successful
  // activation or after deactivation has released them.
  std::atomic<bool> is_active_{false};

  // Parameter-callback handle (must outlive the controller node).
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

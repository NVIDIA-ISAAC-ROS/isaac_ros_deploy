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

#include "isaac_ros_deploy_ros2_control/hardware/topic_based_system_interface.hpp"

#include <cmath>
#include <stdexcept>

#include <pluginlib/class_list_macros.hpp>

namespace isaac_ros_deploy_ros2_control
{
namespace hardware
{

TopicBasedSystemInterface::TopicBasedSystemInterface() = default;

TopicBasedSystemInterface::~TopicBasedSystemInterface()
{
  executor_thread_.request_stop();
  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }
}

#if ROS_DISTRO_HUMBLE
hardware_interface::CallbackReturn TopicBasedSystemInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }
  const auto & hw_info = info;
#else
hardware_interface::CallbackReturn TopicBasedSystemInterface::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (hardware_interface::SystemInterface::on_init(params) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }
  const auto & hw_info = get_hardware_info();
#endif

  const auto logger = rclcpp::get_logger("TopicBasedSystemInterface");

  // Read hardware parameters with defaults.
  auto get_param = [&](const std::string & name, const std::string & default_val) -> std::string {
      const auto it = hw_info.hardware_parameters.find(name);
      return (it != hw_info.hardware_parameters.end()) ? it->second : default_val;
    };

  topics_.joint_states = get_param("joint_states_topic", "/isaac_joint_states");
  topics_.joint_commands = get_param("joint_commands_topic", "/isaac_joint_commands");
  topics_.joint_gains = get_param("joint_gains_topic", "/isaac_sim_drive_gains");
  topics_.imu = get_param("imu_topic", "/isaac_imu");

  // Initialize joint storage.
  const size_t num_joints = hw_info.joints.size();

  joint_state_positions_.resize(num_joints, 0.0);
  joint_state_velocities_.resize(num_joints, 0.0);
  joint_state_efforts_.resize(num_joints, 0.0);

  // Position storage initial-values are read from URDF state_interface
  // <param name="initial_value"> below; keep the default at 0.0 so a missing
  // initial_value behaves the same as before.
  joint_cmd_.positions.resize(num_joints, 0.0);
  // Velocity / effort / kp / kd are initialised to NaN so a controller that
  // never writes them is distinguishable from one that intentionally writes
  // zero. write() leaves the published JointState.velocity array empty in
  // the "never written" case so the Isaac Sim actuator pipeline does not
  // see a spurious target_vel = 0 and apply -kd * v_actual damping that
  // fights motion.
  joint_cmd_.velocities.assign(num_joints, std::nan(""));
  joint_cmd_.efforts.assign(num_joints, std::nan(""));
  joint_cmd_.kp.assign(num_joints, std::nan(""));
  joint_cmd_.kd.assign(num_joints, std::nan(""));

  // Initialize realtime buffers with default data.
  JointStateData initial_joint_data;
  initial_joint_data.positions.resize(num_joints, 0.0);
  initial_joint_data.velocities.resize(num_joints, 0.0);
  initial_joint_data.efforts.resize(num_joints, 0.0);
  joint_state_buf_.writeFromNonRT(initial_joint_data);

  // Build joint name to index mapping.
  for (size_t i = 0; i < num_joints; ++i) {
    joint_name_to_index_[hw_info.joints[i].name] = i;
  }

  // Read initial values from URDF state interface parameters.
  for (size_t i = 0; i < num_joints; ++i) {
    for (const auto & si : hw_info.joints[i].state_interfaces) {
      if (si.name == "position") {
        const auto it = si.parameters.find("initial_value");
        if (it != si.parameters.end()) {
          const double val = std::stod(it->second);
          joint_state_positions_[i] = val;
          initial_joint_data.positions[i] = val;
        }
      }
    }
  }
  joint_state_buf_.writeFromNonRT(initial_joint_data);

  // Build IMU interface name to index mapping.
  imu_name_to_index_ = {
    {"orientation.x", 0},
    {"orientation.y", 1},
    {"orientation.z", 2},
    {"orientation.w", 3},
    {"angular_velocity.x", 4},
    {"angular_velocity.y", 5},
    {"angular_velocity.z", 6},
    {"linear_acceleration.x", 7},
    {"linear_acceleration.y", 8},
    {"linear_acceleration.z", 9},
  };

  // Pre-allocate command message. Position is always published; velocity is
  // populated by write() only when at least one controller has written a
  // non-NaN target this cycle - otherwise the velocity array is left empty
  // so the Isaac Sim actuator pipeline treats it as "no velocity target".
  // The effort field is left unused: the actuator pipeline runs inside
  // Isaac Sim at physics rate.
  joint_cmd_msg_.name.resize(num_joints);
  joint_cmd_msg_.position.resize(num_joints, 0.0);
  joint_cmd_msg_.velocity.clear();
  for (size_t i = 0; i < num_joints; ++i) {
    joint_cmd_msg_.name[i] = hw_info.joints[i].name;
  }

  // Pre-allocate gains message. Convention: position[]=kp, velocity[]=kd.
  // Both fields are populated by write() only when at least one controller
  // has authored a non-NaN gain this cycle; otherwise both arrays are
  // cleared so the Isaac Sim launcher subscriber treats it as "no gains
  // this cycle" and keeps the previous values.
  joint_gains_msg_.name.resize(num_joints);
  joint_gains_msg_.position.clear();
  joint_gains_msg_.velocity.clear();
  for (size_t i = 0; i < num_joints; ++i) {
    joint_gains_msg_.name[i] = hw_info.joints[i].name;
  }

  RCLCPP_INFO(
    logger,
    "Initialized TopicBasedSystemInterface with %zu joints. "
    "Topics: states='%s', commands='%s', gains='%s', imu='%s'",
    num_joints, topics_.joint_states.c_str(), topics_.joint_commands.c_str(),
    topics_.joint_gains.c_str(), topics_.imu.c_str());

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
TopicBasedSystemInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  const auto & hw_info = get_hardware_info();

  // Joint state interfaces.
  for (size_t i = 0; i < hw_info.joints.size(); ++i) {
    const auto & joint = hw_info.joints[i];
    for (const auto & si : joint.state_interfaces) {
      double * value_ptr = nullptr;
      if (si.name == "position") {
        value_ptr = &joint_state_positions_[i];
      } else if (si.name == "velocity") {
        value_ptr = &joint_state_velocities_[i];
      } else if (si.name == "effort") {
        value_ptr = &joint_state_efforts_[i];
      }
      if (value_ptr != nullptr) {
        state_interfaces.emplace_back(joint.name, si.name, value_ptr);
      }
    }
  }

  // IMU sensor state interfaces.
  for (const auto & sensor : hw_info.sensors) {
    for (const auto & si : sensor.state_interfaces) {
      const auto it = imu_name_to_index_.find(si.name);
      if (it != imu_name_to_index_.end()) {
        state_interfaces.emplace_back(sensor.name, si.name, &imu_state_[it->second]);
      }
    }
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
TopicBasedSystemInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  const auto & hw_info = get_hardware_info();

  for (size_t i = 0; i < hw_info.joints.size(); ++i) {
    const auto & joint = hw_info.joints[i];
    for (const auto & ci : joint.command_interfaces) {
      double * value_ptr = nullptr;
      if (ci.name == "position") {
        value_ptr = &joint_cmd_.positions[i];
      } else if (ci.name == "velocity") {
        value_ptr = &joint_cmd_.velocities[i];
      } else if (ci.name == "effort") {
        value_ptr = &joint_cmd_.efforts[i];
      } else if (ci.name == "kp") {
        value_ptr = &joint_cmd_.kp[i];
      } else if (ci.name == "kd") {
        value_ptr = &joint_cmd_.kd[i];
      }
      if (value_ptr != nullptr) {
        command_interfaces.emplace_back(joint.name, ci.name, value_ptr);
      }
    }
  }

  return command_interfaces;
}

hardware_interface::CallbackReturn TopicBasedSystemInterface::on_activate(
  const rclcpp_lifecycle::State &)
{
  const auto & hw_info = get_hardware_info();
  const auto logger = rclcpp::get_logger("TopicBasedSystemInterface");

  node_ = std::make_shared<rclcpp::Node>("topic_based_system_" + hw_info.name);

  joint_state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    topics_.joint_states, rclcpp::SystemDefaultsQoS(),
    std::bind(&TopicBasedSystemInterface::joint_state_callback, this, std::placeholders::_1));

  imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
    topics_.imu, rclcpp::SystemDefaultsQoS(),
    std::bind(&TopicBasedSystemInterface::imu_callback, this, std::placeholders::_1));

  joint_cmd_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(
    topics_.joint_commands, rclcpp::SystemDefaultsQoS());

  joint_gains_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(
    topics_.joint_gains, rclcpp::SystemDefaultsQoS());

  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);

  executor_thread_ = std::jthread(
    [this](const std::stop_token & stop_token) {
      while (!stop_token.stop_requested()) {
        executor_->spin_some(std::chrono::milliseconds(10));
      }
    });

  RCLCPP_INFO(
    logger,
    "Activated TopicBasedSystemInterface. Subscribing to '%s' and '%s', "
    "publishing commands to '%s' and gains to '%s'",
    topics_.joint_states.c_str(), topics_.imu.c_str(),
    topics_.joint_commands.c_str(), topics_.joint_gains.c_str());

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn TopicBasedSystemInterface::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  executor_thread_.request_stop();
  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }

  if (executor_ && node_) {
    executor_->remove_node(node_);
  }
  joint_state_sub_.reset();
  imu_sub_.reset();
  joint_cmd_pub_.reset();
  joint_gains_pub_.reset();
  executor_.reset();
  node_.reset();

  RCLCPP_INFO(
    rclcpp::get_logger("TopicBasedSystemInterface"),
    "Deactivated TopicBasedSystemInterface");

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type TopicBasedSystemInterface::read(
  const rclcpp::Time & time,
  const rclcpp::Duration &)
{
  const auto & joint_data = *joint_state_buf_.readFromRT();
  joint_state_positions_ = joint_data.positions;
  joint_state_velocities_ = joint_data.velocities;
  joint_state_efforts_ = joint_data.efforts;
  const auto joint_stamp = joint_data.stamp;

  const auto & imu_data = *imu_buf_.readFromRT();
  imu_state_ = imu_data.values;
  const auto imu_stamp = imu_data.stamp;

  if (joint_stamp.nanoseconds() > 0 && imu_stamp.nanoseconds() > 0) {
    const double joint_age_ms = (time - joint_stamp).seconds() * 1000.0;
    const double imu_age_ms = (time - imu_stamp).seconds() * 1000.0;
    if (joint_age_ms > 10.0 || imu_age_ms > 10.0) {
      RCLCPP_WARN_THROTTLE(
        rclcpp::get_logger("TopicBasedSystemInterface"),
        *node_->get_clock(), 1000,
        "Stale sensor data: joint_age=%.1fms, imu_age=%.1fms (cm_time=%.3fs)",
        joint_age_ms, imu_age_ms, time.seconds());
    }
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type TopicBasedSystemInterface::write(
  const rclcpp::Time & time,
  const rclcpp::Duration &)
{
  if (!joint_cmd_pub_) {
    return hardware_interface::return_type::OK;
  }

  const size_t num_joints = joint_cmd_.positions.size();

  // Per-step: ship position targets, plus velocity targets only when at
  // least one controller wrote a non-NaN velocity this cycle. Effort is
  // intentionally left unused - the Isaac Sim actuators extension runs the
  // actuator pipeline at physics rate.
  joint_cmd_msg_.header.stamp = time;
  bool any_velocity_target = false;
  for (size_t i = 0; i < num_joints; ++i) {
    joint_cmd_msg_.position[i] = joint_cmd_.positions[i];
    if (!std::isnan(joint_cmd_.velocities[i])) {
      any_velocity_target = true;
    }
  }
  if (any_velocity_target) {
    joint_cmd_msg_.velocity.resize(num_joints);
    for (size_t i = 0; i < num_joints; ++i) {
      // Joints whose controller did not author a velocity target get 0.0,
      // matching the per-joint default that ros2_control would apply if the
      // command interface had been zero-initialised. The bridge consumer
      // expects a fixed-size array when velocity is populated.
      const double v = joint_cmd_.velocities[i];
      joint_cmd_msg_.velocity[i] = std::isnan(v) ? 0.0 : v;
    }
  } else {
    joint_cmd_msg_.velocity.clear();
  }
  // Gains channel: publish kp/kd only when at least one controller wrote a
  // non-NaN gain this cycle. NaN slots are forwarded as NaN so the launcher
  // subscriber can skip them ("keep previous value") -- the controller stack
  // routinely writes some joints' gains and not others depending on which
  // controller is active.
  if (joint_gains_pub_) {
    bool any_gain = false;
    for (size_t i = 0; i < num_joints; ++i) {
      if (!std::isnan(joint_cmd_.kp[i]) || !std::isnan(joint_cmd_.kd[i])) {
        any_gain = true;
        break;
      }
    }
    if (any_gain) {
      joint_gains_msg_.header.stamp = time;
      joint_gains_msg_.position.resize(num_joints);
      joint_gains_msg_.velocity.resize(num_joints);
      for (size_t i = 0; i < num_joints; ++i) {
        joint_gains_msg_.position[i] = joint_cmd_.kp[i];  // NaN preserved
        joint_gains_msg_.velocity[i] = joint_cmd_.kd[i];  // NaN preserved
      }
      joint_gains_pub_->publish(joint_gains_msg_);
    }
  }

  joint_cmd_pub_->publish(joint_cmd_msg_);

  return hardware_interface::return_type::OK;
}

void TopicBasedSystemInterface::joint_state_callback(
  const sensor_msgs::msg::JointState::SharedPtr msg)
{
  JointStateData data;
  data.stamp = rclcpp::Time(msg->header.stamp);
  data.positions.resize(joint_name_to_index_.size(), 0.0);
  data.velocities.resize(joint_name_to_index_.size(), 0.0);
  data.efforts.resize(joint_name_to_index_.size(), 0.0);

  for (size_t i = 0; i < msg->name.size(); ++i) {
    const auto it = joint_name_to_index_.find(msg->name[i]);
    if (it == joint_name_to_index_.end()) {
      continue;
    }
    const size_t idx = it->second;
    if (i < msg->position.size()) {
      data.positions[idx] = msg->position[i];
    }
    if (i < msg->velocity.size()) {
      data.velocities[idx] = msg->velocity[i];
    }
    if (i < msg->effort.size()) {
      data.efforts[idx] = msg->effort[i];
    }
  }

  joint_state_buf_.writeFromNonRT(data);
}

void TopicBasedSystemInterface::imu_callback(
  const sensor_msgs::msg::Imu::SharedPtr msg)
{
  ImuData data;
  data.stamp = rclcpp::Time(msg->header.stamp);
  data.values[0] = msg->orientation.x;
  data.values[1] = msg->orientation.y;
  data.values[2] = msg->orientation.z;
  data.values[3] = msg->orientation.w;
  data.values[4] = msg->angular_velocity.x;
  data.values[5] = msg->angular_velocity.y;
  data.values[6] = msg->angular_velocity.z;
  data.values[7] = msg->linear_acceleration.x;
  data.values[8] = msg->linear_acceleration.y;
  data.values[9] = msg->linear_acceleration.z;

  imu_buf_.writeFromNonRT(data);
}

}  // namespace hardware
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::hardware::TopicBasedSystemInterface,
  hardware_interface::SystemInterface)

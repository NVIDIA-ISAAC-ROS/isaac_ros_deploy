// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hardware_interface/sensor_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/version.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "std_msgs/msg/float64.hpp"

// Check ROS distro for API compatibility
#define ROS_DISTRO_HUMBLE (HARDWARE_INTERFACE_VERSION_MAJOR < 3)

namespace isaac_ros_deploy_ros2_control
{
namespace hardware
{

/// Hardware interface that subscribes to a Float64 topic and exposes it as a state interface.
///
/// This is useful for controllers that need to receive values from ROS topics
/// (like a slider) but want to access them through the standard ros2_control state interface.
///
/// URDF configuration example:
/// <ros2_control name="slider_sensor" type="sensor">
///   <hardware>
///     <plugin>isaac_ros_deploy_ros2_control/TopicSensorInterface</plugin>
///     <param name="topic">/safety_slider</param>
///     <param name="default_value">1.0</param>
///   </hardware>
///   <sensor name="slider">
///     <state_interface name="value"/>
///   </sensor>
/// </ros2_control>
class TopicSensorInterface : public hardware_interface::SensorInterface
{
public:
  TopicSensorInterface();
  ~TopicSensorInterface() override;

#if ROS_DISTRO_HUMBLE
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;
#else
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;
#endif

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

#if ROS_DISTRO_HUMBLE
  const hardware_interface::HardwareInfo & get_hardware_info() const
  {
    return info_;
  }
#endif

private:
  void topic_callback(const std_msgs::msg::Float64::SharedPtr msg);

  // Configuration
  std::string topic_name_;
  double default_value_{0.0};

  // ROS node and subscription (runs in separate thread)
  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  std::thread executor_thread_;
  std::atomic<bool> running_{false};
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr subscription_;

  // State value (thread-safe access)
  std::mutex value_mutex_;
  double current_value_{0.0};

  // Exposed state interface value
  double state_value_{0.0};
};

}  // namespace hardware
}  // namespace isaac_ros_deploy_ros2_control

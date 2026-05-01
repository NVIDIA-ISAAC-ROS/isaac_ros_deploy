// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/hardware/topic_sensor_interface.hpp"

#include <pluginlib/class_list_macros.hpp>

namespace isaac_ros_deploy_ros2_control
{
namespace hardware
{

TopicSensorInterface::TopicSensorInterface() = default;

TopicSensorInterface::~TopicSensorInterface()
{
  // Ensure cleanup on destruction
  running_ = false;
  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }
}

#if ROS_DISTRO_HUMBLE
hardware_interface::CallbackReturn TopicSensorInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SensorInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }
  const auto & hw_info = info;
#else
hardware_interface::CallbackReturn TopicSensorInterface::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (hardware_interface::SensorInterface::on_init(params) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }
  const auto & hw_info = get_hardware_info();
#endif

  // Get topic name from parameters
  auto it = hw_info.hardware_parameters.find("topic");
  if (it == hw_info.hardware_parameters.end()) {
    RCLCPP_ERROR(
      rclcpp::get_logger("TopicSensorInterface"),
      "Missing required parameter 'topic'");
    return hardware_interface::CallbackReturn::ERROR;
  }
  topic_name_ = it->second;

  // Get optional default value
  it = hw_info.hardware_parameters.find("default_value");
  if (it != hw_info.hardware_parameters.end()) {
    default_value_ = std::stod(it->second);
  }

  current_value_ = default_value_;
  state_value_ = default_value_;

  RCLCPP_INFO(
    rclcpp::get_logger("TopicSensorInterface"),
    "Initialized TopicSensorInterface for topic '%s' with default value %f",
    topic_name_.c_str(), default_value_);

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> TopicSensorInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  const auto & hw_info = get_hardware_info();

  // Export state interface for each sensor defined in URDF
  for (const auto & sensor : hw_info.sensors) {
    for (const auto & interface : sensor.state_interfaces) {
      state_interfaces.emplace_back(
        hardware_interface::StateInterface(sensor.name, interface.name, &state_value_));
      RCLCPP_INFO(
        rclcpp::get_logger("TopicSensorInterface"),
        "Exporting state interface: %s/%s", sensor.name.c_str(), interface.name.c_str());
    }
  }

  return state_interfaces;
}

hardware_interface::CallbackReturn TopicSensorInterface::on_activate(
  const rclcpp_lifecycle::State &)
{
  const auto & hw_info = get_hardware_info();

  // Create a separate node for the subscription
  node_ = std::make_shared<rclcpp::Node>("topic_sensor_" + hw_info.name);

  // Create subscription
  subscription_ = node_->create_subscription<std_msgs::msg::Float64>(
    topic_name_, rclcpp::SystemDefaultsQoS(),
    std::bind(&TopicSensorInterface::topic_callback, this, std::placeholders::_1));

  // Create executor and start spinning thread
  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);

  running_ = true;
  executor_thread_ = std::thread(
    [this]() {
      while (running_) {
        executor_->spin_some(std::chrono::milliseconds(10));
      }
    });

  RCLCPP_INFO(
    rclcpp::get_logger("TopicSensorInterface"),
    "Activated TopicSensorInterface, subscribing to '%s'", topic_name_.c_str());

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn TopicSensorInterface::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  // Stop executor thread
  running_ = false;
  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }

  // Clean up
  if (executor_ && node_) {
    executor_->remove_node(node_);
  }
  subscription_.reset();
  executor_.reset();
  node_.reset();

  RCLCPP_INFO(
    rclcpp::get_logger("TopicSensorInterface"),
    "Deactivated TopicSensorInterface");

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type TopicSensorInterface::read(
  const rclcpp::Time &,
  const rclcpp::Duration &)
{
  // Copy the latest value from the subscription thread
  std::lock_guard<std::mutex> lock(value_mutex_);
  state_value_ = current_value_;
  return hardware_interface::return_type::OK;
}

void TopicSensorInterface::topic_callback(const std_msgs::msg::Float64::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(value_mutex_);
  current_value_ = msg->data;
}

}  // namespace hardware
}  // namespace isaac_ros_deploy_ros2_control

PLUGINLIB_EXPORT_CLASS(
  isaac_ros_deploy_ros2_control::hardware::TopicSensorInterface,
  hardware_interface::SensorInterface)

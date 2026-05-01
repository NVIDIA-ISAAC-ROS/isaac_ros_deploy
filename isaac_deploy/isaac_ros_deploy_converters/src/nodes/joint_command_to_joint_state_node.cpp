// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include "isaac_ros_deploy_interfaces/msg/joint_command.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace isaac_ros_deploy_converters
{

/// Node that converts JointCommand messages to JointState messages for visualization.
///
/// This enables visualizing commanded joint positions using robot_state_publisher
/// with a TF prefix, creating a "ghost" robot showing the commanded pose.
///
/// Subscribes to: ~/input (isaac_ros_deploy_interfaces/JointCommand)
/// Publishes to: ~/output (sensor_msgs/JointState)
class JointCommandToJointStateNode : public rclcpp::Node
{
public:
  explicit JointCommandToJointStateNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("joint_command_to_joint_state_node", options)
  {
    subscription_ = create_subscription<isaac_ros_deploy_interfaces::msg::JointCommand>(
      "~/input", 10,
      std::bind(&JointCommandToJointStateNode::callback, this, std::placeholders::_1));

    publisher_ = create_publisher<sensor_msgs::msg::JointState>("~/output", 10);

    RCLCPP_INFO(get_logger(), "JointCommandToJointStateNode initialized");
  }

private:
  void callback(const isaac_ros_deploy_interfaces::msg::JointCommand::SharedPtr msg)
  {
    sensor_msgs::msg::JointState output;
    output.header = msg->header;
    output.name = msg->names;
    output.position = msg->position;
    output.velocity = msg->velocity;
    output.effort = msg->effort;

    publisher_->publish(output);
  }

  rclcpp::Subscription<isaac_ros_deploy_interfaces::msg::JointCommand>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
};

}  // namespace isaac_ros_deploy_converters

RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_deploy_converters::JointCommandToJointStateNode)

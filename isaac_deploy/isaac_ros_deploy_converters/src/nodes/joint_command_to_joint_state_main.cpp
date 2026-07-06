// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_deploy_interfaces/msg/joint_command.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace isaac_ros_deploy_converters
{

/// Node that converts JointCommand messages to JointState messages for visualization.
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

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<isaac_ros_deploy_converters::JointCommandToJointStateNode>());
  rclcpp::shutdown();
  return 0;
}

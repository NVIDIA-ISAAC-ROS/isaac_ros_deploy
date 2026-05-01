// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_deploy_converters/nodes/input_builder_node.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<isaac_ros_deploy_converters::InputBuilderNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

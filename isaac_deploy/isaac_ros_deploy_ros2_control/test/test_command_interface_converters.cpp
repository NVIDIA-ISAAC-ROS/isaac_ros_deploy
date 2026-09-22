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

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include <hardware_interface/handle.hpp>
#include <hardware_interface/loaned_command_interface.hpp>

#include "isaac_ros_deploy_ros2_control/adapters/command_interface_adapter.hpp"
#include "isaac_ros_deploy_ros2_control/converters/command_interface_converter.hpp"

namespace isaac_ros_deploy_ros2_control
{

TEST(CommandInterfaceConverterTest, RegistryContainsBuiltinKinds)
{
  initialize_command_interface_converters();
  auto & reg = CommandInterfaceConverterRegistry::instance();

  EXPECT_TRUE(reg.contains("target/joint/position"));
  EXPECT_TRUE(reg.contains("kp"));
  EXPECT_TRUE(reg.contains("kd"));
  EXPECT_FALSE(reg.contains("actions"));
  EXPECT_FALSE(reg.contains("nonexistent"));
}

TEST(CommandInterfaceConverterTest, JointPositionInterfaces)
{
  initialize_command_interface_converters();
  const auto converter = CommandInterfaceConverterRegistry::instance().create_for_kind(
    "target/joint/position");
  ASSERT_NE(converter, nullptr);

  const std::vector<std::vector<std::string>> element_names = {{}, {"hip", "knee"}};
  const auto interfaces = converter->get_required_command_interfaces(element_names, "", "");
  ASSERT_EQ(interfaces.size(), 2u);
  EXPECT_EQ(interfaces[0], "hip/position");
  EXPECT_EQ(interfaces[1], "knee/position");
}

TEST(CommandInterfaceConverterTest, JointPositionWithPrefixSuffix)
{
  initialize_command_interface_converters();
  const auto converter = CommandInterfaceConverterRegistry::instance().create_for_kind(
    "target/joint/position");

  const std::vector<std::vector<std::string>> element_names = {{}, {"hip", "knee"}};
  const auto interfaces = converter->get_required_command_interfaces(element_names, "safety",
      "_raw");
  ASSERT_EQ(interfaces.size(), 2u);
  EXPECT_EQ(interfaces[0], "safety/hip/position_raw");
  EXPECT_EQ(interfaces[1], "safety/knee/position_raw");
}

TEST(CommandInterfaceConverterTest, KpInterfaces)
{
  initialize_command_interface_converters();
  const auto converter = CommandInterfaceConverterRegistry::instance().create_for_kind("kp");
  ASSERT_NE(converter, nullptr);

  const std::vector<std::vector<std::string>> element_names = {{}, {"hip"}};
  const auto interfaces = converter->get_required_command_interfaces(element_names, "", "");
  ASSERT_EQ(interfaces.size(), 1u);
  EXPECT_EQ(interfaces[0], "hip/kp");
}

TEST(CommandInterfaceConverterTest, KdInterfaces)
{
  initialize_command_interface_converters();
  const auto converter = CommandInterfaceConverterRegistry::instance().create_for_kind("kd");
  ASSERT_NE(converter, nullptr);

  const std::vector<std::vector<std::string>> element_names = {{}, {"hip"}};
  const auto interfaces = converter->get_required_command_interfaces(element_names, "", "");
  ASSERT_EQ(interfaces.size(), 1u);
  EXPECT_EQ(interfaces[0], "hip/kd");
}

TEST(CommandInterfaceConverterTest, PoseRelativeUsesLeappDeltaElementNames)
{
  initialize_command_interface_converters();
  const auto converter = CommandInterfaceConverterRegistry::instance().create_for_kind(
    "target/body/pose_relative");
  ASSERT_NE(converter, nullptr);

  const auto interfaces = converter->get_required_command_interfaces({}, "safety/arm_action",
      "_raw");
  EXPECT_EQ(
    interfaces,
    (std::vector<std::string>{
      "safety/arm_action/delta_x_raw",
      "safety/arm_action/delta_y_raw",
      "safety/arm_action/delta_z_raw",
      "safety/arm_action/delta_axis_angle_x_raw",
      "safety/arm_action/delta_axis_angle_y_raw",
      "safety/arm_action/delta_axis_angle_z_raw"}));

  const auto spec = converter->get_tensor_spec({});
  EXPECT_EQ(
    spec.names,
    (std::vector<std::vector<std::string>>{
      {},
      {"delta_x", "delta_y", "delta_z",
        "delta_axis_angle_x", "delta_axis_angle_y", "delta_axis_angle_z"}}));
}

TEST(CommandInterfaceConverterTest, TensorSpecPreservesElementNames)
{
  initialize_command_interface_converters();
  const auto converter = CommandInterfaceConverterRegistry::instance().create_for_kind(
    "target/joint/position");

  const std::vector<std::vector<std::string>> element_names = {{}, {"hip", "knee"}};
  const auto spec = converter->get_tensor_spec(element_names);
  ASSERT_EQ(spec.names.size(), 2u);
  EXPECT_EQ(spec.names[1][0], "hip");
  EXPECT_EQ(spec.names[1][1], "knee");
}

TEST(CommandInterfaceConverterTest, AdapterPrefixesHardwareNamesAndPreservesTargets)
{
  initialize_command_interface_converters();
  isaac_deploy_core::OutputTermConfig config;
  config.name = "joint_pos_target";
  config.kind = "target/joint/position";
  config.shape = {1, 2};
  config.element_names = {{}, {"joint1", "joint2"}};

  CommandInterfaceAdapter adapter({config}, "safety", "_raw", "robot_");
  EXPECT_EQ(
    adapter.get_required_command_interfaces(),
    (std::vector<std::string>{
      "safety/robot_joint1/position_raw", "safety/robot_joint2/position_raw"}));

  double joint1 = 0.0;
  double joint2 = 0.0;
  auto joint1_interface = std::make_shared<hardware_interface::CommandInterface>(
    "safety/robot_joint1", "position_raw", &joint1);
  auto joint2_interface = std::make_shared<hardware_interface::CommandInterface>(
    "safety/robot_joint2", "position_raw", &joint2);
  std::vector<hardware_interface::LoanedCommandInterface> interfaces;
  interfaces.emplace_back(joint1_interface);
  interfaces.emplace_back(joint2_interface);
  adapter.set_command_interfaces(interfaces);

  isaac_deploy_core::NamedTensor target{
    .name = "joint_pos_target",
    .timestamp_ns = 0,
    .tensor = torch::tensor({{0.45F, -0.15F}}),
  };
  adapter.write_tensor(0, target);
  EXPECT_NEAR(joint1, 0.45, 1e-6);
  EXPECT_NEAR(joint2, -0.15, 1e-6);

  const auto spec = adapter.get_tensor_spec(0);
  EXPECT_EQ(spec.names[1], (std::vector<std::string>{"joint1", "joint2"}));

  adapter.invalidate_command_interfaces();
  EXPECT_TRUE(std::isnan(joint1));
  EXPECT_TRUE(std::isnan(joint2));
}

}  // namespace isaac_ros_deploy_ros2_control

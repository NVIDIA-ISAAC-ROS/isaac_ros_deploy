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
  const auto interfaces = converter->get_required_command_interfaces(element_names, "safety", "_raw");
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

}  // namespace isaac_ros_deploy_ros2_control

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "isaac_ros_deploy_ros2_control/converters/state_interface_converter.hpp"

namespace isaac_ros_deploy_ros2_control
{

TEST(StateInterfaceConverterTest, RegistryContainsBuiltinKinds)
{
  initialize_state_interface_converters();
  auto & reg = StateInterfaceConverterRegistry::instance();

  EXPECT_TRUE(reg.contains("state/joint/position"));
  EXPECT_TRUE(reg.contains("state/joint/velocity"));
  EXPECT_TRUE(reg.contains("state/body/rotation"));
  EXPECT_TRUE(reg.contains("state/body/angular_velocity"));
  EXPECT_FALSE(reg.contains("nonexistent"));
}

TEST(StateInterfaceConverterTest, JointPositionInterfaces)
{
  initialize_state_interface_converters();
  const auto converter = StateInterfaceConverterRegistry::instance().create_for_kind(
    "state/joint/position");
  ASSERT_NE(converter, nullptr);

  const std::vector<std::vector<std::string>> element_names = {{}, {"hip", "knee"}};
  const auto interfaces = converter->get_required_state_interfaces(element_names);
  ASSERT_EQ(interfaces.size(), 2u);
  EXPECT_EQ(interfaces[0], "hip/position");
  EXPECT_EQ(interfaces[1], "knee/position");
}

TEST(StateInterfaceConverterTest, JointVelocityInterfaces)
{
  initialize_state_interface_converters();
  const auto converter = StateInterfaceConverterRegistry::instance().create_for_kind(
    "state/joint/velocity");
  ASSERT_NE(converter, nullptr);

  const std::vector<std::vector<std::string>> element_names = {{}, {"hip", "knee"}};
  const auto interfaces = converter->get_required_state_interfaces(element_names);
  ASSERT_EQ(interfaces.size(), 2u);
  EXPECT_EQ(interfaces[0], "hip/velocity");
  EXPECT_EQ(interfaces[1], "knee/velocity");
}

TEST(StateInterfaceConverterTest, JointPositionTensorSpec)
{
  initialize_state_interface_converters();
  const auto converter = StateInterfaceConverterRegistry::instance().create_for_kind(
    "state/joint/position");

  const std::vector<std::vector<std::string>> element_names = {{}, {"hip", "knee"}};
  const auto spec = converter->get_tensor_spec(element_names);

  // Joint converters return element_names as-is (hardware order == YAML order).
  ASSERT_EQ(spec.names.size(), 2u);
  EXPECT_TRUE(spec.names[0].empty());
  ASSERT_EQ(spec.names[1].size(), 2u);
  EXPECT_EQ(spec.names[1][0], "hip");
  EXPECT_EQ(spec.names[1][1], "knee");
}

TEST(StateInterfaceConverterTest, ImuOrientationInterfaces)
{
  initialize_state_interface_converters();
  const auto converter = StateInterfaceConverterRegistry::instance().create_for_kind(
    "state/body/rotation");
  ASSERT_NE(converter, nullptr);

  // IMU interfaces are fixed, element_names are ignored.
  const std::vector<std::vector<std::string>> element_names = {{}, {"w", "x", "y", "z"}};
  const auto interfaces = converter->get_required_state_interfaces(element_names);
  ASSERT_EQ(interfaces.size(), 4u);
  EXPECT_EQ(interfaces[0], "imu/orientation.x");
  EXPECT_EQ(interfaces[1], "imu/orientation.y");
  EXPECT_EQ(interfaces[2], "imu/orientation.z");
  EXPECT_EQ(interfaces[3], "imu/orientation.w");
}

TEST(StateInterfaceConverterTest, ImuOrientationTensorSpec)
{
  initialize_state_interface_converters();
  const auto converter = StateInterfaceConverterRegistry::instance().create_for_kind(
    "state/body/rotation");

  // IMU TensorSpec returns hardware order (x, y, z, w), not YAML element_names.
  const std::vector<std::vector<std::string>> element_names = {{}, {"w", "x", "y", "z"}};
  const auto spec = converter->get_tensor_spec(element_names);

  ASSERT_EQ(spec.names.size(), 2u);
  ASSERT_EQ(spec.names[1].size(), 4u);
  EXPECT_EQ(spec.names[1][0], "x");
  EXPECT_EQ(spec.names[1][1], "y");
  EXPECT_EQ(spec.names[1][2], "z");
  EXPECT_EQ(spec.names[1][3], "w");
}

TEST(StateInterfaceConverterTest, ImuAngularVelocityInterfaces)
{
  initialize_state_interface_converters();
  const auto converter = StateInterfaceConverterRegistry::instance().create_for_kind(
    "state/body/angular_velocity");
  ASSERT_NE(converter, nullptr);

  const std::vector<std::vector<std::string>> element_names = {{}, {"x", "y", "z"}};
  const auto interfaces = converter->get_required_state_interfaces(element_names);
  ASSERT_EQ(interfaces.size(), 3u);
  EXPECT_EQ(interfaces[0], "imu/angular_velocity.x");
  EXPECT_EQ(interfaces[1], "imu/angular_velocity.y");
  EXPECT_EQ(interfaces[2], "imu/angular_velocity.z");
}

}  // namespace isaac_ros_deploy_ros2_control

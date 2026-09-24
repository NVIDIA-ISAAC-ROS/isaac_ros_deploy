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
#include <hardware_interface/loaned_state_interface.hpp>

#include "isaac_ros_deploy_ros2_control/adapters/state_interface_adapter.hpp"
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
  EXPECT_TRUE(reg.contains("state/body/rotation_6d"));
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

TEST(StateInterfaceConverterTest, BodyPositionDefaultsToEefInterfaces)
{
  initialize_state_interface_converters();
  const auto converter = StateInterfaceConverterRegistry::instance().create_for_kind(
    "state/body/position");
  ASSERT_NE(converter, nullptr);

  EXPECT_EQ(
    converter->get_required_state_interfaces({{}, {"x", "y", "z"}}),
    (std::vector<std::string>{
      "eef/position.x", "eef/position.y", "eef/position.z"}));
}

TEST(StateInterfaceConverterTest, BodyRotation6DIdentityOutput)
{
  initialize_state_interface_converters();
  const auto converter = StateInterfaceConverterRegistry::instance().create_for_kind(
    "state/body/rotation_6d");
  ASSERT_NE(converter, nullptr);

  const auto interfaces = converter->get_required_state_interfaces({{}, {"r00", "r01", "r02",
      "r10", "r11", "r12"}});
  EXPECT_EQ(
    interfaces,
    (std::vector<std::string>{
      "eef/orientation.x", "eef/orientation.y",
      "eef/orientation.z", "eef/orientation.w"}));

  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double qw = 1.0;
  auto qx_interface = std::make_shared<hardware_interface::StateInterface>(
    "eef", "orientation.x", &qx);
  auto qy_interface = std::make_shared<hardware_interface::StateInterface>(
    "eef", "orientation.y", &qy);
  auto qz_interface = std::make_shared<hardware_interface::StateInterface>(
    "eef", "orientation.z", &qz);
  auto qw_interface = std::make_shared<hardware_interface::StateInterface>(
    "eef", "orientation.w", &qw);
  std::vector<hardware_interface::LoanedStateInterface> loaned_interfaces;
  loaned_interfaces.emplace_back(qx_interface);
  loaned_interfaces.emplace_back(qy_interface);
  loaned_interfaces.emplace_back(qz_interface);
  loaned_interfaces.emplace_back(qw_interface);

  auto tensor = torch::zeros({1, 6}, torch::kFloat32);
  converter->read(loaned_interfaces, {0, 1, 2, 3}, tensor);

  const auto accessor = tensor.accessor<float, 2>();
  EXPECT_FLOAT_EQ(accessor[0][0], 1.0F);
  EXPECT_FLOAT_EQ(accessor[0][1], 0.0F);
  EXPECT_FLOAT_EQ(accessor[0][2], 0.0F);
  EXPECT_FLOAT_EQ(accessor[0][3], 0.0F);
  EXPECT_FLOAT_EQ(accessor[0][4], 1.0F);
  EXPECT_FLOAT_EQ(accessor[0][5], 0.0F);

  const auto spec = converter->get_tensor_spec({});
  EXPECT_EQ(
    spec.names,
    (std::vector<std::vector<std::string>>{
      {}, {"r00", "r01", "r02", "r10", "r11", "r12"}}));
}

TEST(StateInterfaceConverterTest, BodyRotation6DYaw90Output)
{
  initialize_state_interface_converters();
  const auto converter = StateInterfaceConverterRegistry::instance().create_for_kind(
    "state/body/rotation_6d");
  ASSERT_NE(converter, nullptr);

  double qx = 0.0;
  double qy = 0.0;
  double qz = std::sqrt(0.5);
  double qw = std::sqrt(0.5);
  auto qx_interface = std::make_shared<hardware_interface::StateInterface>(
    "eef", "orientation.x", &qx);
  auto qy_interface = std::make_shared<hardware_interface::StateInterface>(
    "eef", "orientation.y", &qy);
  auto qz_interface = std::make_shared<hardware_interface::StateInterface>(
    "eef", "orientation.z", &qz);
  auto qw_interface = std::make_shared<hardware_interface::StateInterface>(
    "eef", "orientation.w", &qw);
  std::vector<hardware_interface::LoanedStateInterface> loaned_interfaces;
  loaned_interfaces.emplace_back(qx_interface);
  loaned_interfaces.emplace_back(qy_interface);
  loaned_interfaces.emplace_back(qz_interface);
  loaned_interfaces.emplace_back(qw_interface);

  auto tensor = torch::zeros({1, 6}, torch::kFloat32);
  converter->read(loaned_interfaces, {0, 1, 2, 3}, tensor);

  const auto accessor = tensor.accessor<float, 2>();
  EXPECT_NEAR(accessor[0][0], 0.0F, 1e-6F);
  EXPECT_NEAR(accessor[0][1], -1.0F, 1e-6F);
  EXPECT_NEAR(accessor[0][2], 0.0F, 1e-6F);
  EXPECT_NEAR(accessor[0][3], 1.0F, 1e-6F);
  EXPECT_NEAR(accessor[0][4], 0.0F, 1e-6F);
  EXPECT_NEAR(accessor[0][5], 0.0F, 1e-6F);
}

TEST(StateInterfaceConverterTest, AdapterPrefixesHardwareNamesWithoutChangingTensorValues)
{
  initialize_state_interface_converters();
  isaac_deploy_core::InputTermConfig config;
  config.name = "joint_pos";
  config.source = "joint_pos";
  config.kind = "state/joint/position";
  config.shape = {1, 2};
  config.element_names = {{}, {"joint1", "joint2"}};

  StateInterfaceAdapter adapter({config}, "robot_");
  EXPECT_EQ(
    adapter.get_required_state_interfaces(),
    (std::vector<std::string>{"robot_joint1/position", "robot_joint2/position"}));

  double joint1 = 0.125;
  double joint2 = -0.375;
  auto joint1_interface = std::make_shared<hardware_interface::StateInterface>(
    "robot_joint1", "position", &joint1);
  auto joint2_interface = std::make_shared<hardware_interface::StateInterface>(
    "robot_joint2", "position", &joint2);
  std::vector<hardware_interface::LoanedStateInterface> interfaces;
  interfaces.emplace_back(joint1_interface);
  interfaces.emplace_back(joint2_interface);
  adapter.set_state_interfaces(interfaces);

  auto tensor = torch::zeros({1, 2}, torch::kFloat32);
  adapter.read_tensor(0, tensor);
  EXPECT_FLOAT_EQ(tensor[0][0].item<float>(), 0.125F);
  EXPECT_FLOAT_EQ(tensor[0][1].item<float>(), -0.375F);

  const auto spec = adapter.get_tensor_spec(0);
  EXPECT_EQ(spec.names[1], (std::vector<std::string>{"joint1", "joint2"}));
}

}  // namespace isaac_ros_deploy_ros2_control

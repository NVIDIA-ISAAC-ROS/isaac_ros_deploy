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
#include <torch/torch.h>

#include "isaac_ros_deploy_converters/converters/tensor_to_message_converter.hpp"
#include "isaac_ros_deploy_interfaces/msg/joint_command_trajectory.hpp"

namespace
{

using isaac_ros_deploy_converters::OutputConverterRegistry;
using isaac_ros_deploy_converters::initialize_output_converters;
using isaac_ros_deploy_interfaces::msg::JointCommandTrajectory;

TEST(JointCommandTrajectoryConverterTest, ShapeRank3SelectsTrajectoryConverter)
{
  initialize_output_converters();
  auto & reg = OutputConverterRegistry::instance();
  auto conv = reg.create_for_kind("target/joint/position", {1, 30, 7});
  ASSERT_NE(conv, nullptr);
  EXPECT_EQ(conv->get_message_type(),
            "isaac_ros_deploy_interfaces/msg/JointCommandTrajectory");
}

TEST(JointCommandTrajectoryConverterTest, ShapeRank2StillSelectsSingleStep)
{
  initialize_output_converters();
  auto & reg = OutputConverterRegistry::instance();
  auto conv = reg.create_for_kind("target/joint/position", {1, 7});
  ASSERT_NE(conv, nullptr);
  EXPECT_EQ(conv->get_message_type(),
            "isaac_ros_deploy_interfaces/msg/JointCommand");
}

TEST(JointCommandTrajectoryConverterTest, WritePopulatesFlatArrays)
{
  initialize_output_converters();
  auto & reg = OutputConverterRegistry::instance();
  auto conv = reg.create_for_kind("target/joint/position", {1, 2, 3});
  ASSERT_NE(conv, nullptr);

  // [B=1, H=2, N=3] row-major: step 0 [1,2,3], step 1 [4,5,6]
  auto tensor = torch::tensor(
    {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}}, torch::kFloat32);

  auto msg = std::make_shared<JointCommandTrajectory>();
  const std::vector<std::string> element_names = {"j0", "j1", "j2"};
  conv->write(tensor, element_names, msg, /*timestamp_ns=*/1'000'000'000);

  EXPECT_EQ(msg->horizon, 2u);
  EXPECT_EQ(msg->names, element_names);
  ASSERT_EQ(msg->position.size(), 6u);
  EXPECT_DOUBLE_EQ(msg->position[0], 1.0);
  EXPECT_DOUBLE_EQ(msg->position[1], 2.0);
  EXPECT_DOUBLE_EQ(msg->position[2], 3.0);
  EXPECT_DOUBLE_EQ(msg->position[3], 4.0);
  EXPECT_DOUBLE_EQ(msg->position[4], 5.0);
  EXPECT_DOUBLE_EQ(msg->position[5], 6.0);
}

TEST(JointCommandTrajectoryConverterTest, MultipleConvertersShareHorizonAndNames)
{
  initialize_output_converters();
  auto & reg = OutputConverterRegistry::instance();
  auto pos_conv = reg.create_for_kind("target/joint/position", {1, 2, 3});
  auto eff_conv = reg.create_for_kind("target/joint/effort", {1, 2, 3});
  ASSERT_NE(pos_conv, nullptr);
  ASSERT_NE(eff_conv, nullptr);

  auto pos_tensor = torch::tensor(
    {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}}, torch::kFloat32);
  auto eff_tensor = torch::tensor(
    {{{10.0f, 20.0f, 30.0f}, {40.0f, 50.0f, 60.0f}}}, torch::kFloat32);

  auto msg = std::make_shared<JointCommandTrajectory>();
  const std::vector<std::string> element_names = {"j0", "j1", "j2"};
  pos_conv->write(pos_tensor, element_names, msg, 1'000'000'000);
  eff_conv->write(eff_tensor, element_names, msg, 1'000'000'000);

  EXPECT_EQ(msg->horizon, 2u);
  EXPECT_EQ(msg->names.size(), 3u);
  EXPECT_EQ(msg->position.size(), 6u);
  EXPECT_EQ(msg->effort.size(), 6u);
  EXPECT_DOUBLE_EQ(msg->position[3], 4.0);
  EXPECT_DOUBLE_EQ(msg->effort[3], 40.0);
}

TEST(JointCommandTrajectoryConverterTest, WrongRankTensorThrows)
{
  initialize_output_converters();
  auto & reg = OutputConverterRegistry::instance();
  auto conv = reg.create_for_kind("target/joint/position", {1, 2, 3});
  ASSERT_NE(conv, nullptr);

  // 2D tensor given to trajectory converter -> wrong rank.
  auto tensor = torch::tensor({{1.0f, 2.0f, 3.0f}}, torch::kFloat32);
  auto msg = std::make_shared<JointCommandTrajectory>();
  EXPECT_THROW(
    conv->write(tensor, {"j0", "j1", "j2"}, msg, 0),
    std::runtime_error);
}

TEST(JointCommandTrajectoryConverterTest, MismatchedHorizonAcrossConvertersThrows)
{
  initialize_output_converters();
  auto & reg = OutputConverterRegistry::instance();
  auto pos_conv = reg.create_for_kind("target/joint/position", {1, 2, 3});
  auto eff_conv = reg.create_for_kind("target/joint/effort", {1, 4, 3});
  ASSERT_NE(pos_conv, nullptr);
  ASSERT_NE(eff_conv, nullptr);

  auto pos_tensor = torch::tensor(
    {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}}, torch::kFloat32);
  // 4-step effort tensor.
  auto eff_tensor = torch::zeros({1, 4, 3}, torch::kFloat32);

  auto msg = std::make_shared<JointCommandTrajectory>();
  const std::vector<std::string> names = {"j0", "j1", "j2"};
  pos_conv->write(pos_tensor, names, msg, 0);

  EXPECT_THROW(eff_conv->write(eff_tensor, names, msg, 0), std::runtime_error);
}

TEST(JointCommandTrajectoryConverterTest, MultipleConvertersDisjointJointSubsets)
{
  initialize_output_converters();
  auto & reg = OutputConverterRegistry::instance();
  auto left_conv = reg.create_for_kind("target/joint/position", {1, 2, 2});
  auto right_conv = reg.create_for_kind("target/joint/position", {1, 2, 3});
  ASSERT_NE(left_conv, nullptr);
  ASSERT_NE(right_conv, nullptr);

  // Left: [H=2, N=2] — step 0 [1, 2], step 1 [3, 4]
  auto left_tensor = torch::tensor(
    {{{1.0f, 2.0f}, {3.0f, 4.0f}}}, torch::kFloat32);
  // Right: [H=2, N=3] — step 0 [10, 20, 30], step 1 [40, 50, 60]
  auto right_tensor = torch::tensor(
    {{{10.0f, 20.0f, 30.0f}, {40.0f, 50.0f, 60.0f}}}, torch::kFloat32);

  auto msg = std::make_shared<JointCommandTrajectory>();
  left_conv->write(left_tensor, {"L0", "L1"}, msg, 0);
  right_conv->write(right_tensor, {"R0", "R1", "R2"}, msg, 0);

  EXPECT_EQ(msg->horizon, 2u);
  ASSERT_EQ(msg->names.size(), 5u);
  EXPECT_EQ(msg->names[0], "L0");
  EXPECT_EQ(msg->names[4], "R2");
  ASSERT_EQ(msg->position.size(), 2u * 5u);  // H * N_total
  // Row 0: [1, 2, 10, 20, 30]
  EXPECT_DOUBLE_EQ(msg->position[0], 1.0);
  EXPECT_DOUBLE_EQ(msg->position[1], 2.0);
  EXPECT_DOUBLE_EQ(msg->position[2], 10.0);
  EXPECT_DOUBLE_EQ(msg->position[3], 20.0);
  EXPECT_DOUBLE_EQ(msg->position[4], 30.0);
  // Row 1: [3, 4, 40, 50, 60]
  EXPECT_DOUBLE_EQ(msg->position[5], 3.0);
  EXPECT_DOUBLE_EQ(msg->position[6], 4.0);
  EXPECT_DOUBLE_EQ(msg->position[7], 40.0);
  EXPECT_DOUBLE_EQ(msg->position[8], 50.0);
  EXPECT_DOUBLE_EQ(msg->position[9], 60.0);
}

TEST(JointCommandTrajectoryConverterTest, PartialOverlapThrows)
{
  initialize_output_converters();
  auto & reg = OutputConverterRegistry::instance();
  auto c1 = reg.create_for_kind("target/joint/position", {1, 2, 2});
  auto c2 = reg.create_for_kind("target/joint/position", {1, 2, 2});
  ASSERT_NE(c1, nullptr);
  ASSERT_NE(c2, nullptr);

  auto t = torch::zeros({1, 2, 2}, torch::kFloat32);
  auto msg = std::make_shared<JointCommandTrajectory>();
  c1->write(t, {"A", "B"}, msg, 0);
  // Partial overlap: B appears in both.
  EXPECT_THROW(c2->write(t, {"B", "C"}, msg, 0), std::runtime_error);
}

TEST(JointCommandTrajectoryConverterTest, SubsetScatterWriteToSeparateField)
{
  initialize_output_converters();
  auto & reg = OutputConverterRegistry::instance();
  auto pos_conv = reg.create_for_kind("target/joint/position", {1, 2, 3});
  auto eff_conv = reg.create_for_kind("target/joint/effort", {1, 2, 2});
  ASSERT_NE(pos_conv, nullptr);
  ASSERT_NE(eff_conv, nullptr);

  // Position covers 3 joints.
  auto pos_tensor = torch::tensor(
    {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}}, torch::kFloat32);
  // Effort covers 2 of those 3 joints.
  auto eff_tensor = torch::tensor(
    {{{10.0f, 20.0f}, {30.0f, 40.0f}}}, torch::kFloat32);

  auto msg = std::make_shared<JointCommandTrajectory>();
  pos_conv->write(pos_tensor, {"j0", "j1", "j2"}, msg, 0);
  eff_conv->write(eff_tensor, {"j0", "j2"}, msg, 0);

  EXPECT_EQ(msg->horizon, 2u);
  ASSERT_EQ(msg->names.size(), 3u);
  ASSERT_EQ(msg->effort.size(), 6u);
  // Effort columns at j0 and j2 are filled; j1 stays zero.
  EXPECT_DOUBLE_EQ(msg->effort[0], 10.0);  // row 0, j0
  EXPECT_DOUBLE_EQ(msg->effort[1], 0.0);   // row 0, j1 — zero
  EXPECT_DOUBLE_EQ(msg->effort[2], 20.0);  // row 0, j2
  EXPECT_DOUBLE_EQ(msg->effort[3], 30.0);  // row 1, j0
  EXPECT_DOUBLE_EQ(msg->effort[4], 0.0);   // row 1, j1
  EXPECT_DOUBLE_EQ(msg->effort[5], 40.0);  // row 1, j2
}

}  // namespace

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

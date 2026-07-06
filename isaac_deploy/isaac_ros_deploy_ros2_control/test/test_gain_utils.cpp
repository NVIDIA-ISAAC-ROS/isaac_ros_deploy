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

#include "isaac_ros_deploy_ros2_control/utils/gain_utils.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace utils
{

TEST(ResolveGainsTest, CatchAllPatternSetsAllJoints)
{
  const std::vector<std::pair<std::string, double>> patterns = {
    {".*", 100.0},
  };
  const std::vector<std::string> joints = {"joint_a", "joint_b", "joint_c"};

  auto gains = resolve_gains(patterns, joints);

  ASSERT_EQ(gains.size(), 3u);
  EXPECT_DOUBLE_EQ(gains[0], 100.0);
  EXPECT_DOUBLE_EQ(gains[1], 100.0);
  EXPECT_DOUBLE_EQ(gains[2], 100.0);
}

TEST(ResolveGainsTest, SpecificPatternOverridesCatchAll)
{
  const std::vector<std::pair<std::string, double>> patterns = {
    {".*", 100.0},
    {"joint_b", 50.0},
  };
  const std::vector<std::string> joints = {"joint_a", "joint_b", "joint_c"};

  auto gains = resolve_gains(patterns, joints);

  ASSERT_EQ(gains.size(), 3u);
  EXPECT_DOUBLE_EQ(gains[0], 100.0);
  EXPECT_DOUBLE_EQ(gains[1], 50.0);
  EXPECT_DOUBLE_EQ(gains[2], 100.0);
}

TEST(ResolveGainsTest, RegexPatternMatchesSubset)
{
  const std::vector<std::pair<std::string, double>> patterns = {
    {".*", 100.0},
    {".*_wrist_.*", 20.0},
  };
  const std::vector<std::string> joints = {
    "left_wrist_roll", "right_wrist_pitch", "shoulder_pan", "elbow_flex"};

  auto gains = resolve_gains(patterns, joints);

  ASSERT_EQ(gains.size(), 4u);
  EXPECT_DOUBLE_EQ(gains[0], 20.0);
  EXPECT_DOUBLE_EQ(gains[1], 20.0);
  EXPECT_DOUBLE_EQ(gains[2], 100.0);
  EXPECT_DOUBLE_EQ(gains[3], 100.0);
}

TEST(ResolveGainsTest, MultipleOverrideLevels)
{
  const std::vector<std::pair<std::string, double>> patterns = {
    {".*", 100.0},
    {".*_wrist_.*", 20.0},
    {"waist_yaw_joint", 250.0},
  };
  const std::vector<std::string> joints = {
    "left_wrist_roll", "waist_yaw_joint", "shoulder_pan"};

  auto gains = resolve_gains(patterns, joints);

  ASSERT_EQ(gains.size(), 3u);
  EXPECT_DOUBLE_EQ(gains[0], 20.0);
  EXPECT_DOUBLE_EQ(gains[1], 250.0);
  EXPECT_DOUBLE_EQ(gains[2], 100.0);
}

TEST(ResolveGainsTest, EmptyPatternsReturnsZeros)
{
  const std::vector<std::pair<std::string, double>> patterns = {};
  const std::vector<std::string> joints = {"joint_a", "joint_b"};

  auto gains = resolve_gains(patterns, joints);

  ASSERT_EQ(gains.size(), 2u);
  EXPECT_DOUBLE_EQ(gains[0], 0.0);
  EXPECT_DOUBLE_EQ(gains[1], 0.0);
}

TEST(ResolveGainsTest, NoMatchingPatternLeavesDefaultZero)
{
  const std::vector<std::pair<std::string, double>> patterns = {
    {"wrist_.*", 20.0},
  };
  const std::vector<std::string> joints = {"shoulder_pan", "elbow_flex"};

  auto gains = resolve_gains(patterns, joints);

  ASSERT_EQ(gains.size(), 2u);
  EXPECT_DOUBLE_EQ(gains[0], 0.0);
  EXPECT_DOUBLE_EQ(gains[1], 0.0);
}

TEST(ResolveGainsTest, EmptyJointsReturnsEmptyVector)
{
  const std::vector<std::pair<std::string, double>> patterns = {
    {".*", 100.0},
  };
  const std::vector<std::string> joints = {};

  auto gains = resolve_gains(patterns, joints);

  EXPECT_TRUE(gains.empty());
}

}  // namespace utils
}  // namespace isaac_ros_deploy_ros2_control

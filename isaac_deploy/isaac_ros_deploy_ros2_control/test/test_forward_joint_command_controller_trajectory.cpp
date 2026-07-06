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

// Minimal smoke test for ForwardJointCommandController trajectory-mode math.
//
// The trajectory branch's heavy lifting is the chunk-sampler call + per-joint
// linear interpolation.  Since a full controller-manager harness is heavy,
// this test exercises the underlying sampling/interpolation on the same data
// layout the controller feeds to `chunk_sampler.hpp` — guaranteeing the
// "halfway between two steps returns the mean" invariant end-to-end.

#include <gtest/gtest.h>

#include <span>
#include <cstddef>

#include "isaac_deploy_core/chunk_sampler.hpp"
#include "isaac_ros_deploy_interfaces/msg/joint_command_trajectory.hpp"
#include "rclcpp/rclcpp.hpp"

TEST(ForwardJointCommandControllerTrajectory, HalfwayInterpolation)
{
  using isaac_ros_deploy_interfaces::msg::JointCommandTrajectory;

  JointCommandTrajectory traj;
  traj.header.stamp = rclcpp::Time(static_cast<int64_t>(1'000'000'000), RCL_ROS_TIME);  // 1.0 s
  traj.step_dt = rclcpp::Duration::from_seconds(0.02);            // 50 Hz step
  traj.horizon = 2;
  traj.names = {"j0", "j1"};
  // Row-major [H=2, N=2] — step 0 = [1, 2], step 1 = [3, 4].
  traj.position = {1.0, 2.0, 3.0, 4.0};

  const auto pos = isaac_deploy_core::compute_chunk_position(
    rclcpp::Time(traj.header.stamp, RCL_ROS_TIME),
    rclcpp::Duration(traj.step_dt),
    static_cast<std::size_t>(traj.horizon),
    rclcpp::Time(static_cast<int64_t>(1'010'000'000), RCL_ROS_TIME));  // 1.01 s -> halfway

  EXPECT_FALSE(pos.past_end);
  EXPECT_FALSE(pos.before_start);
  EXPECT_EQ(pos.lo, 0u);
  EXPECT_EQ(pos.hi, 1u);
  EXPECT_DOUBLE_EQ(pos.alpha, 0.5);

  std::span<const double> field(traj.position.data(), traj.position.size());
  const std::size_t n = traj.names.size();
  EXPECT_DOUBLE_EQ(isaac_deploy_core::interpolate_field(field, n, 0, pos), 2.0);
  EXPECT_DOUBLE_EQ(isaac_deploy_core::interpolate_field(field, n, 1, pos), 3.0);
}

TEST(ForwardJointCommandControllerTrajectory, PastEndClampsToLastStep)
{
  using isaac_ros_deploy_interfaces::msg::JointCommandTrajectory;

  JointCommandTrajectory traj;
  traj.header.stamp = rclcpp::Time(static_cast<int64_t>(1'000'000'000), RCL_ROS_TIME);
  traj.step_dt = rclcpp::Duration::from_seconds(0.02);
  traj.horizon = 2;
  traj.names = {"j0"};
  traj.position = {10.0, 20.0};

  // Far past the last step (step H-1 @ 1.02 s).
  const auto pos = isaac_deploy_core::compute_chunk_position(
    rclcpp::Time(traj.header.stamp, RCL_ROS_TIME),
    rclcpp::Duration(traj.step_dt),
    static_cast<std::size_t>(traj.horizon),
    rclcpp::Time(static_cast<int64_t>(2'000'000'000), RCL_ROS_TIME));

  EXPECT_TRUE(pos.past_end);
  EXPECT_EQ(pos.lo, 1u);
  EXPECT_EQ(pos.hi, 1u);

  std::span<const double> field(traj.position.data(), traj.position.size());
  EXPECT_DOUBLE_EQ(isaac_deploy_core::interpolate_field(field, 1, 0, pos), 20.0);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

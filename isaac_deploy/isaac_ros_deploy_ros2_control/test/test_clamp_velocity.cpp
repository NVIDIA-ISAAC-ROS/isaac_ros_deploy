// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/safety_controller/strategies/clamp_velocity.hpp"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  TEST(ClampVelocityTest, CreateSuccess) {
    ClampVelocityConfig config {.max_velocities = {1.0, 2.0, 3.0}};
    auto result = ClampVelocity::create(config);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->name(), "ClampVelocity");
  }

  TEST(ClampVelocityTest, CreateFailsWithEmptyVelocities) {
    ClampVelocityConfig config {.max_velocities = {}};
    auto result = ClampVelocity::create(config);
    EXPECT_FALSE(result.has_value());
  }

  TEST(ClampVelocityTest, NoClampingNeeded) {
    ClampVelocityConfig config {.max_velocities = {10.0, 10.0}};
    auto strategy = *ClampVelocity::create(config);

    auto command = torch::tensor({1.0f, 1.0f});
    auto current = torch::tensor({0.0f, 0.0f});
    double blend_ratio = 1.0;
    double dt = 1.0;

    auto result = strategy->apply(command, current, blend_ratio, dt);
    ASSERT_TRUE(result.has_value());

    // Command is within max_velocity * dt, so no clamping.
    EXPECT_TRUE(torch::allclose(*result, command));
  }

  TEST(ClampVelocityTest, ClampingApplied) {
    ClampVelocityConfig config {.max_velocities = {0.5, 0.5}};
    auto strategy = *ClampVelocity::create(config);

    auto command = torch::tensor({2.0f, 2.0f});
    auto current = torch::tensor({0.0f, 0.0f});
    double blend_ratio = 1.0;
    double dt = 1.0;

    auto result = strategy->apply(command, current, blend_ratio, dt);
    ASSERT_TRUE(result.has_value());

    // Command is clamped to max_velocity * dt = 0.5.
    auto expected = torch::tensor({0.5f, 0.5f});
    EXPECT_TRUE(torch::allclose(*result, expected));
  }

  TEST(ClampVelocityTest, BlendRatioInterpolation) {
    ClampVelocityConfig config {.max_velocities = {10.0, 10.0}};
    auto strategy = *ClampVelocity::create(config);

    auto command = torch::tensor({2.0f, 2.0f});
    auto current = torch::tensor({0.0f, 0.0f});
    double blend_ratio = 0.5;
    double dt = 1.0;

    auto result = strategy->apply(command, current, blend_ratio, dt);
    ASSERT_TRUE(result.has_value());

    // With blend_ratio = 0.5, output = lerp(current, command, 0.5) = 1.0.
    auto expected = torch::tensor({1.0f, 1.0f});
    EXPECT_TRUE(torch::allclose(*result, expected));
  }

}  // namespace isaac_deploy_core

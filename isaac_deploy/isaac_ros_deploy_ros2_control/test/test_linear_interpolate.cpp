// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/safety_controller/strategies/linear_interpolate.hpp"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  TEST(LinearInterpolateTest, CreateSuccess) {
    LinearInterpolateConfig config;
    auto result = LinearInterpolate::create(config);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->name(), "LinearInterpolate");
  }

  TEST(LinearInterpolateTest, FirstCallSetsActivationPosition) {
    LinearInterpolateConfig config;
    auto strategy = *LinearInterpolate::create(config);

    auto command = torch::tensor({2.0f, 2.0f});
    auto current = torch::tensor({0.0f, 0.0f});
    double blend_ratio = 0.5;
    double dt = 0.01;

    auto result = strategy->apply(command, current, blend_ratio, dt);
    ASSERT_TRUE(result.has_value());

    // First call: activation_pos = current = {0, 0}
    // output = lerp(activation_pos, command, 0.5) = {1, 1}
    auto expected = torch::tensor({1.0f, 1.0f});
    EXPECT_TRUE(torch::allclose(*result, expected));
  }

  TEST(LinearInterpolateTest, SubsequentCallsUseActivationPosition) {
    LinearInterpolateConfig config;
    auto strategy = *LinearInterpolate::create(config);

    // First call - sets activation position.
    auto current1 = torch::tensor({0.0f, 0.0f});
    auto command1 = torch::tensor({2.0f, 2.0f});
    (void)strategy->apply(command1, current1, 0.5, 0.01);

    // Second call - activation position should still be {0, 0}.
    auto current2 = torch::tensor({1.0f, 1.0f});  // Robot moved.
    auto command2 = torch::tensor({4.0f, 4.0f});
    double blend_ratio = 0.5;

    auto result = strategy->apply(command2, current2, blend_ratio, 0.01);
    ASSERT_TRUE(result.has_value());

    // output = lerp(activation_pos={0,0}, command={4,4}, 0.5) = {2, 2}
    auto expected = torch::tensor({2.0f, 2.0f});
    EXPECT_TRUE(torch::allclose(*result, expected));
  }

  TEST(LinearInterpolateTest, ResetClearsActivationPosition) {
    LinearInterpolateConfig config;
    auto strategy = *LinearInterpolate::create(config);

    // First call.
    auto current1 = torch::tensor({0.0f, 0.0f});
    auto command1 = torch::tensor({2.0f, 2.0f});
    (void)strategy->apply(command1, current1, 1.0, 0.01);

    // Reset.
    strategy->reset();

    // Next call should set new activation position.
    auto current2 = torch::tensor({5.0f, 5.0f});
    auto command2 = torch::tensor({10.0f, 10.0f});
    auto result = strategy->apply(command2, current2, 0.5, 0.01);

    // output = lerp(activation_pos={5,5}, command={10,10}, 0.5) = {7.5, 7.5}
    auto expected = torch::tensor({7.5f, 7.5f});
    EXPECT_TRUE(torch::allclose(*result, expected));
  }

}  // namespace isaac_deploy_core

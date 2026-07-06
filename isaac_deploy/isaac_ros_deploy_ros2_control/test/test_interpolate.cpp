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

#include <limits>

#include "isaac_ros_deploy_ros2_control/safety_controller/strategies/interpolate.hpp"

namespace isaac_deploy_core
{

TEST(InterpolateTest, CreateSuccess) {
    InterpolateConfig config {.max_velocities = {1.0, 2.0, 3.0}};
    auto result = Interpolate::create(config);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->name(), "Interpolate");
}

TEST(InterpolateTest, CreateFailsWithEmptyVelocities) {
    InterpolateConfig config {.max_velocities = {}};
    auto result = Interpolate::create(config);
    EXPECT_FALSE(result.has_value());
}

TEST(InterpolateTest, NoClampingNeeded) {
    InterpolateConfig config {.max_velocities = {10.0, 10.0}};
    auto strategy = *Interpolate::create(config);

    auto command = torch::tensor({1.0f, 1.0f});
    auto current = torch::tensor({0.0f, 0.0f});
    double blend_ratio = 1.0;
    double dt = 1.0;

    auto result = strategy->apply(command, current, blend_ratio, dt);
    ASSERT_TRUE(result.has_value());

    // Command is within max_velocity * dt, so no clamping.
    EXPECT_TRUE(torch::allclose(*result, command));
}

TEST(InterpolateTest, ClampingApplied) {
    InterpolateConfig config {.max_velocities = {0.5, 0.5}};
    auto strategy = *Interpolate::create(config);

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

TEST(InterpolateTest, BlendRatioInterpolation) {
    InterpolateConfig config {.max_velocities = {10.0, 10.0}};
    auto strategy = *Interpolate::create(config);

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

TEST(InterpolateTest, IntegratorAdvancesOpenLoop) {
    InterpolateConfig config {
    .max_velocities = {0.5, 0.5},
    };
    auto strategy = *Interpolate::create(config);

    auto command = torch::tensor({2.0f, 2.0f});
    auto current = torch::tensor({0.0f, 0.0f});

    auto step1 = strategy->apply(command, current, 1.0, 1.0);
    ASSERT_TRUE(step1.has_value());
    EXPECT_TRUE(torch::allclose(*step1, torch::tensor({0.5f, 0.5f})));

    auto step2 = strategy->apply(command, current, 1.0, 1.0);
    ASSERT_TRUE(step2.has_value());
    EXPECT_TRUE(torch::allclose(*step2, torch::tensor({1.0f, 1.0f})));
}

TEST(InterpolateTest, ResetReseedsIntegrator) {
    InterpolateConfig config {
    .max_velocities = {0.5, 0.5},
    };
    auto strategy = *Interpolate::create(config);

    auto command = torch::tensor({2.0f, 2.0f});
    auto current = torch::tensor({0.0f, 0.0f});

    ASSERT_TRUE(strategy->apply(command, current, 1.0, 1.0).has_value());
    strategy->reset();

    auto result = strategy->apply(command, current, 1.0, 1.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(torch::allclose(*result, torch::tensor({0.5f, 0.5f})));
}

TEST(InterpolateTest, HomeTargetAtZeroBlend) {
    InterpolateConfig config {
    .max_velocities = {0.5, 0.5},
    .default_position = {1.0, 1.0},
    };
    auto strategy = *Interpolate::create(config);

    auto command = torch::tensor({2.0f, 2.0f});
    auto current = torch::tensor({0.0f, 0.0f});

    auto step1 = strategy->apply(command, current, 0.0, 1.0);
    ASSERT_TRUE(step1.has_value());
    EXPECT_TRUE(torch::allclose(*step1, torch::tensor({0.5f, 0.5f})));

    auto step2 = strategy->apply(command, current, 0.0, 1.0);
    ASSERT_TRUE(step2.has_value());
    EXPECT_TRUE(torch::allclose(*step2, torch::tensor({1.0f, 1.0f})));

    auto step3 = strategy->apply(command, current, 0.0, 1.0);
    ASSERT_TRUE(step3.has_value());
    EXPECT_TRUE(torch::allclose(*step3, torch::tensor({1.0f, 1.0f})));
}

TEST(InterpolateTest, HoldsActivationWhenNoHome) {
    InterpolateConfig config {.max_velocities = {0.5, 0.5}};
    auto strategy = *Interpolate::create(config);

    auto command = torch::tensor({2.0f, 2.0f});
    auto current = torch::tensor({0.0f, 0.0f});

    auto step1 = strategy->apply(command, current, 0.0, 1.0);
    ASSERT_TRUE(step1.has_value());
    EXPECT_TRUE(torch::allclose(*step1, torch::tensor({0.0f, 0.0f})));
}

TEST(InterpolateTest, PartialHomePosition) {
    InterpolateConfig config {
    .max_velocities = {0.5, 0.5},
    .default_position = {0.0, std::numeric_limits<double>::quiet_NaN()},
    };
    auto strategy = *Interpolate::create(config);

    auto command = torch::tensor({2.0f, 2.0f});
    auto current = torch::tensor({1.0f, 1.0f});

    auto step1 = strategy->apply(command, current, 0.0, 1.0);
    ASSERT_TRUE(step1.has_value());
    EXPECT_TRUE(torch::allclose(*step1, torch::tensor({0.5f, 1.0f})));
}

TEST(InterpolateTest, CreateFailsOnHomeSizeMismatch) {
    InterpolateConfig config {
    .max_velocities = {0.5, 0.5},
    .default_position = {1.0},  // wrong size
    };
    auto result = Interpolate::create(config);
    EXPECT_FALSE(result.has_value());
}

TEST(InterpolateTest, NonPositiveVelocitySkipsClamp) {
    InterpolateConfig config {.max_velocities = {-1.0, 0.0, 0.5}};
    auto strategy = *Interpolate::create(config);

    auto command = torch::tensor({2.0f, 2.0f, 2.0f});
    auto current = torch::tensor({0.0f, 0.0f, 0.0f});

    auto step1 = strategy->apply(command, current, 1.0, 1.0);
    ASSERT_TRUE(step1.has_value());
    EXPECT_TRUE(torch::allclose(*step1, torch::tensor({2.0f, 2.0f, 0.5f})));
}

}  // namespace isaac_deploy_core

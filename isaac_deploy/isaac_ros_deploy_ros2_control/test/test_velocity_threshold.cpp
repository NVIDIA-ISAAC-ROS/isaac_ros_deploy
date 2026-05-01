// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/safety_controller/detectors/velocity_threshold.hpp"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  TEST(VelocityThresholdDetectorTest, CreateSuccess) {
    VelocityThresholdConfig config {
      .input_names = {"joint_velocities"},
      .max_velocity = 10.0,
      .mean_velocity = 5.0,
    };
    auto result = VelocityThresholdDetector::create(config);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->name(), "velocity_threshold");
  }

  TEST(VelocityThresholdDetectorTest, PassesWhenBelowMaxThreshold) {
    VelocityThresholdConfig config {
      .input_names = {"joint_velocities"},
      .max_velocity = 10.0,
      .mean_velocity = 0.0,
    };
    auto detector = *VelocityThresholdDetector::create(config);

    std::vector < NamedTensor > inputs = {
      {.name = "joint_velocities", .tensor = torch::tensor({5.0f, 3.0f, 2.0f})},
    };
    auto result = detector->check(inputs);
    EXPECT_TRUE(result.has_value());
  }

  TEST(VelocityThresholdDetectorTest, FailsWhenAboveMaxThreshold) {
    VelocityThresholdConfig config {
      .input_names = {"joint_velocities"},
      .max_velocity = 10.0,
      .mean_velocity = 0.0,
    };
    auto detector = *VelocityThresholdDetector::create(config);

    std::vector < NamedTensor > inputs = {
      {.name = "joint_velocities", .tensor = torch::tensor({5.0f, 15.0f, 2.0f})},
    };
    auto result = detector->check(inputs);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, Error::Code::kFailedPrecondition);
  }

  TEST(VelocityThresholdDetectorTest, PassesWhenBelowMeanThreshold) {
    VelocityThresholdConfig config {
      .input_names = {"joint_velocities"},
      .max_velocity = 0.0,
      .mean_velocity = 5.0,
    };
    auto detector = *VelocityThresholdDetector::create(config);

    std::vector < NamedTensor > inputs = {
      {.name = "joint_velocities", .tensor = torch::tensor({3.0f, 4.0f, 2.0f})},
    };
    auto result = detector->check(inputs);
    EXPECT_TRUE(result.has_value());
  }

  TEST(VelocityThresholdDetectorTest, FailsWhenAboveMeanThreshold) {
    VelocityThresholdConfig config {
      .input_names = {"joint_velocities"},
      .max_velocity = 0.0,
      .mean_velocity = 5.0,
    };
    auto detector = *VelocityThresholdDetector::create(config);

    std::vector < NamedTensor > inputs = {
      {.name = "joint_velocities", .tensor = torch::tensor({6.0f, 7.0f, 8.0f})},
    };
    auto result = detector->check(inputs);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, Error::Code::kFailedPrecondition);
  }

  TEST(VelocityThresholdDetectorTest, PassesWithNoConfiguredInputs) {
    VelocityThresholdConfig config {
      .input_names = {},
      .max_velocity = 1.0,
      .mean_velocity = 1.0,
    };
    auto detector = *VelocityThresholdDetector::create(config);

    std::vector < NamedTensor > inputs = {
      {.name = "joint_velocities", .tensor = torch::tensor({100.0f, 100.0f, 100.0f})},
    };
    auto result = detector->check(inputs);
    EXPECT_TRUE(result.has_value());
  }

  TEST(VelocityThresholdDetectorTest, HandlesNegativeVelocities) {
    VelocityThresholdConfig config {
      .input_names = {"joint_velocities"},
      .max_velocity = 10.0,
      .mean_velocity = 0.0,
    };
    auto detector = *VelocityThresholdDetector::create(config);

    std::vector < NamedTensor > inputs = {
      {.name = "joint_velocities", .tensor = torch::tensor({-15.0f, 3.0f, 2.0f})},
    };
    auto result = detector->check(inputs);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, Error::Code::kFailedPrecondition);
  }

}  // namespace isaac_deploy_core

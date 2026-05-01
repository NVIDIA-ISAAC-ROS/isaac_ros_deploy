// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/safety_controller/safety_controller.hpp"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  TEST(SafetyControllerTest, CreateWithClampVelocity) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kClampVelocity, .max_velocities = {1.0, 2.0, 3.0}},
    };
    auto result = SafetyController::create(config);
    ASSERT_TRUE(result.has_value());
  }

  TEST(SafetyControllerTest, CreateWithInterpolatePosition) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kLinearInterpolate},
    };
    auto result = SafetyController::create(config);
    ASSERT_TRUE(result.has_value());
  }

  TEST(SafetyControllerTest, CreateWithNoPostProcessing) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kNoPostProcessing},
    };
    auto result = SafetyController::create(config);
    ASSERT_TRUE(result.has_value());
  }

  TEST(SafetyControllerTest, ActivateAndAdvance) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kClampVelocity,
        .max_velocities = {1000.0, 1000.0, 1000.0}},
    };
    auto controller_result = SafetyController::create(config);
    ASSERT_TRUE(controller_result.has_value());
    auto & controller = *controller_result;

    std::vector < TensorSpec > input_specs;
    std::vector < NamedTensor > inputs = {
      {.name = "command_positions", .tensor = torch::tensor({1.0f, 2.0f, 3.0f})},
      {.name = "current_positions", .tensor = torch::tensor({0.0f, 0.0f, 0.0f})},
      {.name = "blend_ratio", .tensor = torch::tensor(1.0)},
      {.name = "dt", .tensor = torch::tensor(0.01)},
    };
    std::vector < TensorSpec > output_specs;
    std::vector < NamedTensor > outputs = {
      {.name = "safe_positions", .tensor = torch::zeros({3})},
    };

    auto activate_result = controller.activate(input_specs, inputs, output_specs, outputs);
    ASSERT_TRUE(activate_result.has_value());

    auto advance_result = controller.advance(0, inputs, outputs);
    ASSERT_TRUE(advance_result.has_value());

    // With high max_velocity and blend_ratio=1, output should match command.
    EXPECT_TRUE(torch::allclose(outputs[0].tensor, inputs[0].tensor));
  }

  TEST(SafetyControllerTest, InputNames) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kNoPostProcessing},
    };
    auto controller = *SafetyController::create(config);

    auto input_names = controller.input_names();
    EXPECT_EQ(input_names.size(), 4);
    EXPECT_EQ(input_names[0], "command_positions");
    EXPECT_EQ(input_names[1], "current_positions");
    EXPECT_EQ(input_names[2], "blend_ratio");
    EXPECT_EQ(input_names[3], "dt");
  }

  TEST(SafetyControllerTest, OutputNames) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kNoPostProcessing},
    };
    auto controller = *SafetyController::create(config);

    auto output_names = controller.output_names();
    EXPECT_EQ(output_names.size(), 1);
    EXPECT_EQ(output_names[0], "safe_positions");
  }

  TEST(SafetyControllerTest, VelocityCheckPassesWhenBelowMaxThreshold) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kNoPostProcessing},
      .out_of_domain_detection = {.velocity_threshold = {.input_names = {"joint_velocities"},
          .max_velocity = 10.0,
          .mean_velocity = 0.0}},
    };
    auto controller = *SafetyController::create(config);

    std::vector < TensorSpec > specs;
    std::vector < NamedTensor > inputs = {
      {.name = "command_positions", .tensor = torch::tensor({1.0f, 2.0f, 3.0f})},
      {.name = "current_positions", .tensor = torch::tensor({0.0f, 0.0f, 0.0f})},
      {.name = "blend_ratio", .tensor = torch::tensor(1.0)},
      {.name = "dt", .tensor = torch::tensor(0.01)},
      {.name = "joint_velocities", .tensor = torch::tensor({5.0f, 3.0f, 2.0f})},
    };
    std::vector < NamedTensor > outputs = {{.name = "safe_positions", .tensor = torch::zeros({3})}};

    (void)controller.activate(specs, inputs, specs, outputs);
    auto result = controller.advance(0, inputs, outputs);
    EXPECT_TRUE(result.has_value());
  }

  TEST(SafetyControllerTest, VelocityCheckFailsWhenAboveMaxThreshold) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kNoPostProcessing},
      .out_of_domain_detection = {.velocity_threshold = {.input_names = {"joint_velocities"},
          .max_velocity = 10.0,
          .mean_velocity = 0.0}},
    };
    auto controller = *SafetyController::create(config);

    std::vector < TensorSpec > specs;
    std::vector < NamedTensor > inputs = {
      {.name = "command_positions", .tensor = torch::tensor({1.0f, 2.0f, 3.0f})},
      {.name = "current_positions", .tensor = torch::tensor({0.0f, 0.0f, 0.0f})},
      {.name = "blend_ratio", .tensor = torch::tensor(1.0)},
      {.name = "dt", .tensor = torch::tensor(0.01)},
      {.name = "joint_velocities", .tensor = torch::tensor({5.0f, 15.0f, 2.0f})},
    };
    std::vector < NamedTensor > outputs = {{.name = "safe_positions", .tensor = torch::zeros({3})}};

    (void)controller.activate(specs, inputs, specs, outputs);
    auto result = controller.advance(0, inputs, outputs);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, Error::Code::kFailedPrecondition);
  }

  TEST(SafetyControllerTest, VelocityCheckPassesWhenBelowMeanThreshold) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kNoPostProcessing},
      .out_of_domain_detection = {.velocity_threshold = {.input_names = {"joint_velocities"},
          .max_velocity = 0.0,
          .mean_velocity = 5.0}},
    };
    auto controller = *SafetyController::create(config);

    std::vector < TensorSpec > specs;
    std::vector < NamedTensor > inputs = {
      {.name = "command_positions", .tensor = torch::tensor({1.0f, 2.0f, 3.0f})},
      {.name = "current_positions", .tensor = torch::tensor({0.0f, 0.0f, 0.0f})},
      {.name = "blend_ratio", .tensor = torch::tensor(1.0)},
      {.name = "dt", .tensor = torch::tensor(0.01)},
      {.name = "joint_velocities", .tensor = torch::tensor({3.0f, 4.0f, 2.0f})},
    };
    std::vector < NamedTensor > outputs = {{.name = "safe_positions", .tensor = torch::zeros({3})}};

    (void)controller.activate(specs, inputs, specs, outputs);
    auto result = controller.advance(0, inputs, outputs);
    EXPECT_TRUE(result.has_value());
  }

  TEST(SafetyControllerTest, VelocityCheckFailsWhenAboveMeanThreshold) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kNoPostProcessing},
      .out_of_domain_detection = {.velocity_threshold = {.input_names = {"joint_velocities"},
          .max_velocity = 0.0,
          .mean_velocity = 5.0}},
    };
    auto controller = *SafetyController::create(config);

    std::vector < TensorSpec > specs;
    std::vector < NamedTensor > inputs = {
      {.name = "command_positions", .tensor = torch::tensor({1.0f, 2.0f, 3.0f})},
      {.name = "current_positions", .tensor = torch::tensor({0.0f, 0.0f, 0.0f})},
      {.name = "blend_ratio", .tensor = torch::tensor(1.0)},
      {.name = "dt", .tensor = torch::tensor(0.01)},
      {.name = "joint_velocities", .tensor = torch::tensor({6.0f, 7.0f, 8.0f})},
    };
    std::vector < NamedTensor > outputs = {{.name = "safe_positions", .tensor = torch::zeros({3})}};

    (void)controller.activate(specs, inputs, specs, outputs);
    auto result = controller.advance(0, inputs, outputs);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, Error::Code::kFailedPrecondition);
  }

  TEST(SafetyControllerTest, VelocityCheckPassesWithNoConfiguredInputs) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kNoPostProcessing},
      .out_of_domain_detection = {.velocity_threshold = {.input_names = {},
          .max_velocity = 1.0,
          .mean_velocity = 1.0}},
    };
    auto controller = *SafetyController::create(config);

    std::vector < TensorSpec > specs;
    std::vector < NamedTensor > inputs = {
      {.name = "command_positions", .tensor = torch::tensor({1.0f, 2.0f, 3.0f})},
      {.name = "current_positions", .tensor = torch::tensor({0.0f, 0.0f, 0.0f})},
      {.name = "blend_ratio", .tensor = torch::tensor(1.0)},
      {.name = "dt", .tensor = torch::tensor(0.01)},
      {.name = "joint_velocities", .tensor = torch::tensor({100.0f, 100.0f, 100.0f})},
    };
    std::vector < NamedTensor > outputs = {{.name = "safe_positions", .tensor = torch::zeros({3})}};

    (void)controller.activate(specs, inputs, specs, outputs);
    auto result = controller.advance(0, inputs, outputs);
    EXPECT_TRUE(result.has_value());
  }

  TEST(SafetyControllerTest, VelocityCheckHandlesNegativeVelocities) {
    SafetyControllerConfig config {
      .blend_ratio = {.type = BlendStrategy::kNoPostProcessing},
      .out_of_domain_detection = {.velocity_threshold = {.input_names = {"joint_velocities"},
          .max_velocity = 10.0,
          .mean_velocity = 0.0}},
    };
    auto controller = *SafetyController::create(config);

    std::vector < TensorSpec > specs;
    std::vector < NamedTensor > inputs = {
      {.name = "command_positions", .tensor = torch::tensor({1.0f, 2.0f, 3.0f})},
      {.name = "current_positions", .tensor = torch::tensor({0.0f, 0.0f, 0.0f})},
      {.name = "blend_ratio", .tensor = torch::tensor(1.0)},
      {.name = "dt", .tensor = torch::tensor(0.01)},
      {.name = "joint_velocities", .tensor = torch::tensor({-15.0f, 3.0f, 2.0f})},
    };
    std::vector < NamedTensor > outputs = {{.name = "safe_positions", .tensor = torch::zeros({3})}};

    (void)controller.activate(specs, inputs, specs, outputs);
    auto result = controller.advance(0, inputs, outputs);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, Error::Code::kFailedPrecondition);
  }

}  // namespace isaac_deploy_core

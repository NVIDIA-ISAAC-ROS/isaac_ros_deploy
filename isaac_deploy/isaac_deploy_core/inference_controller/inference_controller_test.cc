// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/inference_controller.h"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  namespace {

    /// Create a minimal InferenceControllerConfig for testing.
    InferenceControllerConfig make_test_config(const std::string & runner_type)
    {
      return {
               .inputs = {.terms = {{.name = "input", .kind = "input", .shape = {1, 3}}}},
               .outputs = {.terms = {{.name = "output", .kind = "output", .shape = {1, 3}}}},
               .runner = {.model_path = "", .runner_type = runner_type},
      };
    }

  }  // namespace

  TEST(InferenceControllerTest, CreateFailsWithInvalidRunner) {
    auto result = InferenceController::create(make_test_config("invalid"));
    EXPECT_FALSE(result.has_value());
  }

  TEST(InferenceControllerTest, CreateSuccess) {
    auto result = InferenceController::create(make_test_config("mock"));
    ASSERT_TRUE(result.has_value());
  }

  TEST(InferenceControllerTest, ActivateAndAdvance) {
    auto controller_result = InferenceController::create(make_test_config("mock"));
    ASSERT_TRUE(controller_result.has_value());
    auto & controller = *controller_result;

    std::vector < NamedTensor > inputs = {
      {.name = "input", .tensor = torch::tensor({{1.0f, 2.0f, 3.0f}})},
    };
    std::vector < TensorSpec > input_specs = {{}};
    std::vector < TensorSpec > output_specs = {{}};
    std::vector < NamedTensor > outputs = {{.name = "output", .tensor = torch::zeros({1, 3})}};

    auto activate_result = controller.activate(inputs, input_specs, output_specs, outputs);
    ASSERT_TRUE(activate_result.has_value());

    auto advance_result = controller.advance(inputs, outputs);
    ASSERT_TRUE(advance_result.has_value());

    // Mock runner copies input to output.
    EXPECT_TRUE(torch::allclose(outputs[0].tensor, torch::tensor({{1.0f, 2.0f, 3.0f}})));
  }

  TEST(InferenceControllerTest, Deactivate) {
    auto controller_result = InferenceController::create(make_test_config("mock"));
    ASSERT_TRUE(controller_result.has_value());
    auto & controller = *controller_result;

    std::vector < NamedTensor > inputs = {
      {.name = "input", .tensor = torch::ones({1, 3})},
    };
    std::vector < TensorSpec > input_specs = {{}};
    std::vector < TensorSpec > output_specs = {{}};
    std::vector < NamedTensor > outputs = {{.name = "output", .tensor = torch::zeros({1, 3})}};

    auto activate_result = controller.activate(inputs, input_specs, output_specs, outputs);
    ASSERT_TRUE(activate_result.has_value());

    auto deactivate_result = controller.deactivate();
    ASSERT_TRUE(deactivate_result.has_value());
  }

}  // namespace isaac_deploy_core

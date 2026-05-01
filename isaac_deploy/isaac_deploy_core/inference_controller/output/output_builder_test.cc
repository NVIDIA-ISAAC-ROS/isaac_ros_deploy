// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/output/output_builder.h"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  TEST(OutputBuilderTest, CreateSuccess) {
    OutputBuilder::Config config {
      .terms = {{.name = "pos_targets", .kind = "pos", .shape = {1, 3}},
        {.name = "vel_targets", .kind = "vel", .shape = {1, 3}}}
    };
    auto result = OutputBuilder::create(config);
    ASSERT_TRUE(result.has_value());

    auto keys = result->get_input_names();
    EXPECT_EQ(keys.size(), 2);
    EXPECT_EQ(keys[0], "pos_targets");
    EXPECT_EQ(keys[1], "vel_targets");
  }

  TEST(OutputBuilderTest, ActivateAndAdvance) {
    OutputBuilder::Config config {
      .terms = {{.name = "pos_targets", .kind = "pos", .shape = {1, 3}}}};
    auto builder_result = OutputBuilder::create(config);
    ASSERT_TRUE(builder_result.has_value());
    auto & builder = *builder_result;

    std::vector < TensorSpec > specs = {{.names = {}}};
    std::vector < NamedTensor > outputs = {{.name = "pos_targets", .tensor = torch::zeros({1, 3})}};

    auto activate_result = builder.activate(specs, outputs);
    ASSERT_TRUE(activate_result.has_value());

    TensorDict nn_outputs;
    nn_outputs["pos_targets"] = torch::tensor({{1.0f, 2.0f, 3.0f}});

    auto advance_result = builder.advance(nn_outputs, outputs);
    ASSERT_TRUE(advance_result.has_value());

    EXPECT_TRUE(torch::allclose(outputs[0].tensor, nn_outputs["pos_targets"]));
  }

  TEST(OutputBuilderTest, MultipleOutputs) {
    OutputBuilder::Config config {.terms = {{.name = "pos", .kind = "pos", .shape = {1, 2}},
        {.name = "vel", .kind = "vel", .shape = {1, 2}}}};
    auto builder_result = OutputBuilder::create(config);
    ASSERT_TRUE(builder_result.has_value());
    auto & builder = *builder_result;

    std::vector < TensorSpec > specs = {{}, {}};
    std::vector < NamedTensor > outputs = {{.name = "pos", .tensor = torch::zeros({1, 2})},
      {.name = "vel", .tensor = torch::zeros({1, 2})}};

    auto activate_result = builder.activate(specs, outputs);
    ASSERT_TRUE(activate_result.has_value());

    TensorDict nn_outputs;
    nn_outputs["pos"] = torch::tensor({{1.0f, 2.0f}});
    nn_outputs["vel"] = torch::tensor({{3.0f, 4.0f}});

    auto advance_result = builder.advance(nn_outputs, outputs);
    ASSERT_TRUE(advance_result.has_value());

    EXPECT_TRUE(torch::allclose(outputs[0].tensor, torch::tensor({{1.0f, 2.0f}})));
    EXPECT_TRUE(torch::allclose(outputs[1].tensor, torch::tensor({{3.0f, 4.0f}})));
  }

}  // namespace isaac_deploy_core

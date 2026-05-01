// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/input/input_builder.h"

#include <gtest/gtest.h>

#include "isaac_deploy_core/inference_controller/config_parser.h"

namespace isaac_deploy_core {

  TEST(InputBuilderTest, CreateSuccess) {
    InputBuilder::Config config {
      .terms = {
        {.name = "joint_pos", .kind = "joint_pos", .shape = {1, 3}},
        {.name = "joint_vel", .kind = "joint_vel", .shape = {1, 3}},
      }
    };
    auto result = InputBuilder::create(config);
    ASSERT_TRUE(result.has_value());

    auto keys = result->get_output_names();
    EXPECT_EQ(keys.size(), 2);
    EXPECT_EQ(keys[0], "joint_pos");
    EXPECT_EQ(keys[1], "joint_vel");
  }

  TEST(InputBuilderTest, ActivateAndAdvance) {
    InputBuilder::Config config {
      .terms = {
        {.name = "joint_pos", .kind = "joint_pos", .shape = {1, 3}},
      }
    };
    auto builder_result = InputBuilder::create(config);
    ASSERT_TRUE(builder_result.has_value());
    auto & builder = *builder_result;

    std::vector < NamedTensor > inputs = {
      {.name = "joint_pos", .tensor = torch::tensor({{1.0f, 2.0f, 3.0f}})},
    };
    std::vector < TensorSpec > specs = {{}};

    auto activate_result = builder.activate(inputs, specs);
    ASSERT_TRUE(activate_result.has_value());

    TensorDict outputs;
    auto advance_result = builder.advance(inputs, outputs);
    ASSERT_TRUE(advance_result.has_value());

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_GT(outputs.count("joint_pos"), 0u);
    EXPECT_TRUE(torch::allclose(outputs["joint_pos"], torch::tensor({{1.0f, 2.0f, 3.0f}})));
  }

  TEST(InputBuilderTest, MultipleInputs) {
    InputBuilder::Config config {
      .terms = {
        {.name = "pos", .kind = "pos", .shape = {1, 2}},
        {.name = "vel", .kind = "vel", .shape = {1, 2}},
      }
    };
    auto builder_result = InputBuilder::create(config);
    ASSERT_TRUE(builder_result.has_value());
    auto & builder = *builder_result;

    std::vector < NamedTensor > inputs = {
      {.name = "pos", .tensor = torch::tensor({{1.0f, 2.0f}})},
      {.name = "vel", .tensor = torch::tensor({{3.0f, 4.0f}})},
    };
    std::vector < TensorSpec > specs = {{}, {}};

    auto activate_result = builder.activate(inputs, specs);
    ASSERT_TRUE(activate_result.has_value());

    TensorDict outputs;
    auto advance_result = builder.advance(inputs, outputs);
    ASSERT_TRUE(advance_result.has_value());

    EXPECT_EQ(outputs.size(), 2);
    EXPECT_TRUE(torch::allclose(outputs["pos"], torch::tensor({{1.0f, 2.0f}})));
    EXPECT_TRUE(torch::allclose(outputs["vel"], torch::tensor({{3.0f, 4.0f}})));
  }

  TEST(InputBuilderTest, GetRequiredInputKinds) {
    InputBuilder::Config config {
      .terms = {
        {.name = "joint_pos", .kind = "joint_pos", .shape = {1, 3}},
        {.name = "last_actions", .kind = "last_actions", .shape = {1, 3},
          .output_key = "out"},
      }
    };
    auto builder_result = InputBuilder::create(config);
    ASSERT_TRUE(builder_result.has_value());

    // Kinds exclude feedback (used for registry converter lookup).
    auto kinds = builder_result->get_unique_kinds();
    EXPECT_EQ(kinds.size(), 1);
    EXPECT_EQ(kinds[0], "joint_pos");

    // Sources include all (external + feedback).
    // Feedback term source defaults to output_key ("out").
    auto sources = builder_result->get_unique_source_names();
    EXPECT_EQ(sources.size(), 2);
  }

  TEST(InputBuilderTest, KindBasedRouting) {
    // Two terms share the same kind: current value and history.
    InputBuilder::Config config {
      .terms = {
        {.name = "joint_pos", .kind = "joint_pos", .shape = {1, 3}},
        {.name = "joint_pos_history", .kind = "joint_pos", .shape = {1, 2, 3},
          .history_length = 2, .include_current_in_history = true},
      }
    };
    auto builder_result = InputBuilder::create(config);
    ASSERT_TRUE(builder_result.has_value());
    auto & builder = *builder_result;

    // Only one kind needed from outside.
    auto kinds = builder.get_unique_kinds();
    EXPECT_EQ(kinds.size(), 1);
    EXPECT_EQ(kinds[0], "joint_pos");

    // Activate with a single input.
    std::vector < NamedTensor > inputs = {
      {.name = "joint_pos", .tensor = torch::tensor({{1.0f, 2.0f, 3.0f}})},
    };
    std::vector < TensorSpec > specs = {{}};

    auto activate_result = builder.activate(inputs, specs);
    ASSERT_TRUE(activate_result.has_value());

    // Advance with updated tensor — both terms should be populated.
    inputs[0].tensor = torch::tensor({{4.0f, 5.0f, 6.0f}});
    TensorDict outputs;
    auto advance_result = builder.advance(inputs, outputs);
    ASSERT_TRUE(advance_result.has_value());

    EXPECT_EQ(outputs.size(), 2);
    EXPECT_GT(outputs.count("joint_pos"), 0u);
    EXPECT_GT(outputs.count("joint_pos_history"), 0u);

    // Current value term should have the input directly.
    EXPECT_TRUE(torch::allclose(outputs["joint_pos"], torch::tensor({{4.0f, 5.0f, 6.0f}})));

    // History term should have shape [1, 2, 3].
    EXPECT_EQ(outputs["joint_pos_history"].sizes(), std::vector < int64_t > ({1, 2, 3}));
  }

  TEST(InputBuilderTest, GetRequiredInputKindsDeduplicates) {
    // Multiple terms with the same kind should produce a single required kind.
    InputBuilder::Config config {
      .terms = {
        {.name = "joint_pos", .kind = "joint_pos", .shape = {1, 3}},
        {.name = "joint_pos_history", .kind = "joint_pos", .shape = {1, 2, 3},
          .history_length = 2},
      }
    };
    auto builder_result = InputBuilder::create(config);
    ASSERT_TRUE(builder_result.has_value());

    auto kinds = builder_result->get_unique_kinds();
    EXPECT_EQ(kinds.size(), 1);
    EXPECT_EQ(kinds[0], "joint_pos");
  }

  TEST(InputBuilderTest, SourceBasedRouting) {
    // Two terms with the same kind but different sources.
    InputBuilder::Config config {
      .terms = {
        {.name = "left_img", .kind = "image", .source = "left_image",
          .shape = {1, 3}},
        {.name = "right_img", .kind = "image", .source = "right_image",
          .shape = {1, 3}},
      }
    };
    auto builder_result = InputBuilder::create(config);
    ASSERT_TRUE(builder_result.has_value());
    auto & builder = *builder_result;

    // Required sources should be two distinct sources.
    auto sources = builder.get_unique_source_names();
    EXPECT_EQ(sources.size(), 2);

    // Required kinds should be just one (both have kind "image").
    auto kinds = builder.get_unique_kinds();
    EXPECT_EQ(kinds.size(), 1);
    EXPECT_EQ(kinds[0], "image");

    // Source→kind map should have two entries, both mapping to "image".
    auto source_to_kind = builder.get_source_to_kind_map();
    EXPECT_EQ(source_to_kind.size(), 2);
    EXPECT_EQ(source_to_kind["left_image"], "image");
    EXPECT_EQ(source_to_kind["right_image"], "image");

    // Activate with two inputs named by source.
    std::vector < NamedTensor > inputs = {
      {.name = "left_image", .tensor = torch::tensor({{1.0f, 2.0f, 3.0f}})},
      {.name = "right_image", .tensor = torch::tensor({{4.0f, 5.0f, 6.0f}})},
    };
    std::vector < TensorSpec > specs = {{}, {}};

    auto activate_result = builder.activate(inputs, specs);
    ASSERT_TRUE(activate_result.has_value());

    TensorDict outputs;
    auto advance_result = builder.advance(inputs, outputs);
    ASSERT_TRUE(advance_result.has_value());

    EXPECT_EQ(outputs.size(), 2);
    EXPECT_TRUE(torch::allclose(outputs["left_img"], torch::tensor({{1.0f, 2.0f, 3.0f}})));
    EXPECT_TRUE(torch::allclose(outputs["right_img"], torch::tensor({{4.0f, 5.0f, 6.0f}})));
  }

  TEST(InputBuilderTest, SourceDefaultsToKind) {
    // When source is not specified, it defaults to kind.
    InputBuilder::Config config {
      .terms = {
        {.name = "joint_pos", .kind = "joint_pos", .shape = {1, 3}},
      }
    };
    auto builder_result = InputBuilder::create(config);
    ASSERT_TRUE(builder_result.has_value());

    auto sources = builder_result->get_unique_source_names();
    EXPECT_EQ(sources.size(), 1);
    EXPECT_EQ(sources[0], "joint_pos");

    auto source_to_kind = builder_result->get_source_to_kind_map();
    EXPECT_EQ(source_to_kind.size(), 1);
    EXPECT_EQ(source_to_kind["joint_pos"], "joint_pos");
  }

  TEST(InputBuilderTest, FeedbackTermAsRegularInput) {
    // Feedback term (output_key set) is treated as a regular input.
    InputBuilder::Config config {
      .terms = {
        {.name = "joint_pos", .kind = "joint_pos", .shape = {1, 3}},
        {.name = "last_actions", .kind = "last_actions", .shape = {1, 3},
          .output_key = "prev_output"},
      }
    };
    auto builder_result = InputBuilder::create(config);
    ASSERT_TRUE(builder_result.has_value());
    auto & builder = *builder_result;

    // Feedback term source defaults to output_key.
    auto feedback = builder.get_feedback_input_names();
    EXPECT_EQ(feedback.size(), 1);
    EXPECT_EQ(feedback[0], "prev_output");

    // Activate with both external and feedback inputs.
    std::vector < NamedTensor > inputs = {
      {.name = "joint_pos", .tensor = torch::tensor({{1.0f, 2.0f, 3.0f}})},
      {.name = "prev_output", .tensor = torch::zeros({1, 3})},
    };
    std::vector < TensorSpec > specs = {{}, {}};

    auto activate_result = builder.activate(inputs, specs);
    ASSERT_TRUE(activate_result.has_value());

    // Advance — feedback tensor is at position in inputs vector.
    inputs[1].tensor = torch::tensor({{7.0f, 8.0f, 9.0f}});
    TensorDict outputs;
    auto advance_result = builder.advance(inputs, outputs);
    ASSERT_TRUE(advance_result.has_value());

    EXPECT_EQ(outputs.size(), 2);
    EXPECT_TRUE(torch::allclose(outputs["joint_pos"], torch::tensor({{1.0f, 2.0f, 3.0f}})));
    EXPECT_TRUE(torch::allclose(outputs["last_actions"], torch::tensor({{7.0f, 8.0f, 9.0f}})));
  }

  TEST(InputBuilderTest, CreateFromYamlWithFeedbackConnections) {
    // Feedback connections are applied by create_from_model_config from ModelConfig.
    auto yaml =
      YAML::Load(
      R"(
      models:
        policy:
          inputs:
          - name: joint_pos
            kind: joint_pos
            shape: [1, 3]
            type: tensor
          - name: last_actions
            kind: last_actions
            shape: [1, 3]
            type: tensor
          outputs:
          - name: actions
            kind: actions
            shape: [1, 3]
      pipeline:
        feedback_connections:
          policy/actions: [policy/last_actions]
        data_flow: {}
    )");

    auto graph_result = parse_graph_config(yaml);
    ASSERT_TRUE(graph_result.has_value());
    auto model_config_result = merge_graph_to_model_config(*graph_result, yaml);
    ASSERT_TRUE(model_config_result.has_value());

    auto config_result = InputBuilder::Config::create_from_model_config(*model_config_result);
    ASSERT_TRUE(config_result.has_value());

    auto builder_result = InputBuilder::create(*config_result);
    ASSERT_TRUE(builder_result.has_value());

    // The feedback term should have output_key set by feedback_connections.
    auto feedback = builder_result->get_feedback_input_names();
    EXPECT_EQ(feedback.size(), 1);
    EXPECT_EQ(feedback[0], "actions");

    // External kinds should exclude the feedback term.
    auto kinds = builder_result->get_unique_kinds();
    EXPECT_EQ(kinds.size(), 1);
    EXPECT_EQ(kinds[0], "joint_pos");

    // Source of feedback term should be the output name.
    auto sources = builder_result->get_unique_source_names();
    EXPECT_EQ(sources.size(), 2);
  }

}  // namespace isaac_deploy_core

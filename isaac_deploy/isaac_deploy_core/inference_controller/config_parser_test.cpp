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

#include "isaac_deploy_core/inference_controller/config_parser.hpp"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  // Helper: parse_graph_config + merge_graph_to_model_config in one step.
  expected < ModelConfig > parse_and_merge(const YAML::Node & yaml)
  {
    auto graph_result = parse_graph_config(yaml);
    if (!graph_result) {
      return tl::unexpected(graph_result.error());
    }
    return merge_graph_to_model_config(*graph_result, yaml);
  }

  TEST(ConfigParserTest, ParseMinimalConfig) {
    auto yaml =
      YAML::Load(
      R"(
      models:
        policy:
          inputs:
          - name: input
            kind: joint_pos
            shape: [1, 3]
          outputs:
          - name: output
            kind: joint_pos_targets
            shape: [1, 3]
    )");
    auto result = parse_and_merge(yaml);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->inputs.IsSequence());
    EXPECT_TRUE(result->outputs.IsSequence());
    EXPECT_TRUE(result->feedback_connections.empty());
  }

  TEST(ConfigParserTest, ParseFeedbackConnections) {
    auto yaml =
      YAML::Load(
      R"(
      models:
        policy:
          inputs:
          - name: joint_pos
            kind: joint_pos
            shape: [1, 3]
          - name: last_actions
            kind: last_actions
            shape: [1, 3]
          outputs:
          - name: actions
            kind: actions
            shape: [1, 3]
      pipeline:
        inputs:
          policy: [joint_pos]
        outputs:
          policy: [actions]
        feedback_connections:
          policy/actions: [policy/last_actions]
        data_flow: {}
    )");
    auto result = parse_and_merge(yaml);
    ASSERT_TRUE(result.has_value());

    // Model prefix should be stripped.
    EXPECT_EQ(result->feedback_connections.size(), 1);
    EXPECT_EQ(result->feedback_connections.count("actions"), 1);
    EXPECT_EQ(result->feedback_connections.at("actions").size(), 1);
    EXPECT_EQ(result->feedback_connections.at("actions")[0], "last_actions");
  }

  TEST(ConfigParserTest, ParseFeedbackConnectionsMultipleInputs) {
    auto yaml =
      YAML::Load(
      R"(
      models:
        policy:
          inputs:
          - name: a
            kind: a
            shape: [1, 3]
          - name: b
            kind: b
            shape: [1, 3]
          outputs:
          - name: out
            kind: out
            shape: [1, 3]
      pipeline:
        feedback_connections:
          policy/out: [policy/a, policy/b]
        data_flow: {}
    )");
    auto result = parse_and_merge(yaml);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->feedback_connections.at("out").size(), 2);
    EXPECT_EQ(result->feedback_connections.at("out")[0], "a");
    EXPECT_EQ(result->feedback_connections.at("out")[1], "b");
  }

  TEST(ConfigParserTest, NoModelsSection) {
    auto yaml = YAML::Load("pipeline: {}");
    auto result = parse_graph_config(yaml);
    EXPECT_FALSE(result.has_value());
  }

  TEST(ConfigParserTest, MultiModelRequiresPipeline) {
    auto yaml =
      YAML::Load(
      R"(
      models:
        a:
          inputs: []
          outputs: []
        b:
          inputs: []
          outputs: []
    )");
    auto graph_result = parse_graph_config(yaml);
    ASSERT_TRUE(graph_result.has_value());
    auto result = merge_graph_to_model_config(*graph_result, yaml);
    EXPECT_FALSE(result.has_value());
  }

  TEST(ConfigParserTest, ParseParameters) {
    auto yaml =
      YAML::Load(
      R"(
      models:
        policy:
          inputs:
          - name: a
            kind: a
            shape: [1, 3]
          outputs:
          - name: b
            kind: b
            shape: [1, 3]
          parameters:
            model_path: /path/to/model.onnx
            backend: triton
    )");
    auto result = parse_and_merge(yaml);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->model_path, "/path/to/model.onnx");
    EXPECT_EQ(result->backend, "triton");
  }

  TEST(ConfigParserTest, FeedbackConnectionsScalarRejected) {
    auto yaml =
      YAML::Load(
      R"(
      models:
        policy:
          inputs:
          - name: last_actions
            kind: last_actions
            shape: [1, 3]
          outputs:
          - name: actions
            kind: actions
            shape: [1, 3]
      pipeline:
        feedback_connections:
          policy/actions: policy/last_actions
        data_flow: {}
    )");
    auto result = parse_graph_config(yaml);
    EXPECT_FALSE(result.has_value());
  }

  TEST(ConfigParserTest, FeedbackConnectionsUnknownOutputRejected) {
    auto yaml =
      YAML::Load(
      R"(
      models:
        policy:
          inputs:
          - name: last_actions
            kind: last_actions
            shape: [1, 3]
          outputs:
          - name: actions
            kind: actions
            shape: [1, 3]
      pipeline:
        feedback_connections:
          policy/nonexistent: [policy/last_actions]
        data_flow: {}
    )");
    auto result = parse_graph_config(yaml);
    EXPECT_FALSE(result.has_value());
  }

  TEST(ConfigParserTest, MultiModelMergedConfig) {
    auto yaml =
      YAML::Load(
      R"(
      models:
        model_a:
          inputs:
          - name: ai1
            kind: state/joint/position
            shape: [1, 1]
          - name: ai2
            kind: feedback/ao2
            shape: [1, 1]
          - name: ai3
            kind: feedback/bo3
            shape: [1, 1]
          outputs:
          - name: ao1
            kind: ao1
            shape: [1, 1]
          - name: ao2
            kind: ao2
            shape: [1, 1]
          parameters:
            model_path: model_a.onnx
            backend: triton
        model_b:
          inputs:
          - name: bi1
            kind: data_flow/ao1
            shape: [1, 1]
          - name: bi2
            kind: feedback/bo2
            shape: [1, 1]
          outputs:
          - name: bo1
            kind: joint_pos_targets
            shape: [1, 1]
          - name: bo2
            kind: bo2
            shape: [1, 1]
          - name: bo3
            kind: bo3
            shape: [1, 1]
          parameters:
            model_path: model_b.onnx
            backend: triton
      pipeline:
        inputs:
          model_a: [ai1]
        outputs:
          model_b: [bo1]
        feedback_connections:
          model_a/ao2: [model_a/ai2]
          model_b/bo3: [model_a/ai3]
          model_b/bo2: [model_b/bi2]
        data_flow:
          model_a/ao1: [model_b/bi1]
    )");
    auto result = parse_and_merge(yaml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Merged inputs: ai1 (dangling) only. Feedback targets (ai2, ai3, bi2)
    // and data_flow targets (bi1) are excluded.
    ASSERT_TRUE(result->inputs.IsSequence());
    EXPECT_EQ(result->inputs.size(), 1);

    std::vector < std::string > input_names;
    for (const auto & input : result->inputs) {
      input_names.push_back(input["name"].as < std::string > ());
    }
    EXPECT_EQ(input_names, (std::vector < std::string > {"ai1"}));

    // Merged outputs: bo1 (dangling) only. Feedback sources (ao2, bo2, bo3)
    // are handled by InferenceRunnerNodes directly.
    ASSERT_TRUE(result->outputs.IsSequence());
    EXPECT_EQ(result->outputs.size(), 1);

    std::vector < std::string > output_names;
    for (const auto & output : result->outputs) {
      output_names.push_back(output["name"].as < std::string > ());
    }
    EXPECT_EQ(output_names, (std::vector < std::string > {"bo1"}));

    // Feedback connections are empty in merged config (handled by runners).
    EXPECT_TRUE(result->feedback_connections.empty());

    // model_path and backend should be empty for merged config.
    EXPECT_TRUE(result->model_path.empty());
    EXPECT_TRUE(result->backend.empty());
  }

  TEST(ConfigParserTest, ParseGraphConfigSingleModel) {
    auto yaml =
      YAML::Load(
      R"(
      models:
        policy:
          inputs:
          - name: joint_pos
            kind: joint_pos
            shape: [1, 3]
          outputs:
          - name: actions
            kind: actions
            shape: [1, 3]
          parameters:
            model_path: /path/to/model.onnx
            backend: triton
      pipeline:
        inputs:
          policy: [joint_pos]
        outputs:
          policy: [actions]
        feedback_connections: {}
        data_flow: {}
    )");
    auto result = parse_graph_config(yaml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_EQ(result->models.size(), 1);
    EXPECT_EQ(result->models[0].first, "policy");
    EXPECT_EQ(result->models[0].second.model_path, "/path/to/model.onnx");
    EXPECT_EQ(result->models[0].second.backend, "triton");
    EXPECT_TRUE(result->data_flow.empty());
  }

  TEST(ConfigParserTest, ParseGraphConfigMultiModel) {
    auto yaml =
      YAML::Load(
      R"(
      models:
        model_a:
          inputs:
          - name: ai1
            kind: state/joint/position
            shape: [1, 1]
          outputs:
          - name: ao1
            kind: ao1
            shape: [1, 1]
          parameters:
            model_path: a.onnx
            backend: triton
        model_b:
          inputs:
          - name: bi1
            kind: data_flow/ao1
            shape: [1, 1]
          outputs:
          - name: bo1
            kind: joint_pos_targets
            shape: [1, 1]
          parameters:
            model_path: b.onnx
            backend: triton
      pipeline:
        inputs:
          model_a: [ai1]
        outputs:
          model_b: [bo1]
        feedback_connections:
          model_a/ao1: [model_a/ai1]
        data_flow:
          model_a/ao1: [model_b/bi1]
    )");
    auto result = parse_graph_config(yaml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Two models in insertion order.
    ASSERT_EQ(result->models.size(), 2);
    EXPECT_EQ(result->models[0].first, "model_a");
    EXPECT_EQ(result->models[0].second.model_path, "a.onnx");
    EXPECT_EQ(result->models[1].first, "model_b");
    EXPECT_EQ(result->models[1].second.model_path, "b.onnx");

    // Per-model feedback: model_a has ao1->[ai1], model_b has none.
    EXPECT_EQ(result->models[0].second.feedback_connections.size(), 1);
    EXPECT_EQ(
      result->models[0].second.feedback_connections.at("ao1"),
      (std::vector < std::string > {"ai1"}));
    EXPECT_TRUE(result->models[1].second.feedback_connections.empty());

    // Data flow preserved with full prefixed keys.
    EXPECT_EQ(result->data_flow.size(), 1);
    EXPECT_EQ(
      result->data_flow.at("model_a/ao1"),
      (std::vector < std::string > {"model_b/bi1"}));
  }

}  // namespace isaac_deploy_core

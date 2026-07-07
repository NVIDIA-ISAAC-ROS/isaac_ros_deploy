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

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "isaac_deploy_core/core/error.hpp"

namespace isaac_deploy_core {

/// Model configuration extracted from a graph-pipeline YAML file.
///
/// Each model entry in the YAML contains `inputs`, `outputs`, and optionally
/// `parameters` sections.
  struct ModelConfig
  {
    /// YAML sequence of input term configurations.
    YAML::Node inputs;
    /// YAML sequence of output term configurations.
    YAML::Node outputs;
    /// Path to the model file (empty if not specified).
    std::filesystem::path model_path;
    /// Inference backend, e.g. "onnx" (empty if not specified).
    std::string backend;
    /// Feedback connections: output_name -> [input_name, ...] (model prefix stripped).
    std::unordered_map < std::string, std::vector < std::string >> feedback_connections;
  };

/// Multi-model graph configuration extracted from a graph-pipeline YAML file.
///
/// Contains per-model configs and pipeline-level connectivity (data flow
/// between models, dangling inputs/outputs, feedback connections).
  struct GraphConfig
  {
    /// Model configs in YAML insertion order (name, config).
    std::vector < std::pair < std::string, ModelConfig >> models;

    /// Data flow connections: "model_a/output" -> ["model_b/input", ...].
    std::unordered_map < std::string, std::vector < std::string >> data_flow;
  };

/// Parse the full graph configuration from a graph-pipeline YAML config.
///
/// Supports single-model and multi-model configs. Returns per-model configs
/// and pipeline connectivity information.
  expected < GraphConfig > parse_graph_config(const YAML::Node & root);

/// Merge a GraphConfig into a single ModelConfig for InputBuilder/OutputBuilder.
///
/// For single-model graphs, returns the model as-is.
/// For multi-model graphs, merges inputs/outputs into a single ModelConfig
/// using the pipeline section to determine which inputs/outputs are dangling.
/// Data-flow target inputs and feedback inputs are excluded from the merged
/// config (they are handled by InferenceRunnerNodes directly).
  expected < ModelConfig > merge_graph_to_model_config(
    const GraphConfig & graph,
    const YAML::Node & root);

}  // namespace isaac_deploy_core

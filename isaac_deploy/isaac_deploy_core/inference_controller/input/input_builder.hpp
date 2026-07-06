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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaac_deploy_core/core/error.hpp"
#include "isaac_deploy_core/core/types.hpp"
#include "isaac_deploy_core/inference_controller/config_parser.hpp"
#include "isaac_deploy_core/inference_controller/input/input_term.hpp"

namespace isaac_deploy_core {

/// InputBuilder manages multiple InputTerms and builds the full input for neural network.
///
/// Terms are grouped by "source" — multiple terms can share the same source (e.g., a
/// current-value term and a history term both with source "joint_pos"). The builder
/// routes a single external input per source to all terms that share it.
  class InputBuilder {
public:
    /// Configuration for InputBuilder.
    struct Config
    {
      /// Configurations for all input terms.
      std::vector < InputTermConfig > terms;

      /// Create a Config from a parsed ModelConfig.
      /// Uses model inputs and feedback_connections from the ModelConfig.
      static expected < Config > create_from_model_config(const ModelConfig & model_config);
    };

    /// Create an InputBuilder from configuration.
    static expected < InputBuilder > create(const Config & config);

    /// Activate the builder with initial inputs and their specifications.
    /// @param inputs Initial input tensors (one per unique source, .name = source).
    /// @param input_specs TensorSpec per input (positionally aligned with inputs).
    expected < void > activate(
      const std::vector < NamedTensor > &inputs,
      const std::vector < TensorSpec > &input_specs);

    /// Deactivate the builder.
    expected < void > deactivate();

    /// Build neural network inputs from middleware inputs.
    /// Inputs must be in the same order as passed to activate().
    /// This method is real-time safe (only positional lookups, no allocations).
    /// @param inputs Current tensor values (positionally aligned with activate).
    /// @param outputs Neural network input tensors (written in-place).
    expected < void > advance(
      const std::vector < NamedTensor > &inputs,
      TensorDict & outputs);

    /// Get the unique sources of all inputs that this builder requires.
    /// Includes both external and feedback (output-dependent) sources.
    /// Each source maps to one NamedTensor position (.name = source).
    std::vector < std::string > get_unique_source_names() const;

    /// Get source→kind mapping for all external (non-output-dependent) inputs.
    std::unordered_map < std::string, std::string > get_source_to_kind_map() const;

    /// Get the unique kinds of external inputs that this builder requires.
    /// Excludes output-dependent (feedback) terms.
    std::vector < std::string > get_unique_kinds() const;

    /// Get the names of feedback inputs (previous outputs) that this builder depends on.
    std::vector < std::string > get_feedback_input_names() const;

    /// Get feedback input shapes (same order as get_feedback_input_names).
    std::vector < std::vector < int64_t >> get_feedback_input_shapes() const;

    /// Get the names of the tensors that this builder produces (NN input keys).
    std::vector < std::string > get_output_names() const;

private:
    explicit InputBuilder(std::vector < InputTerm > terms);

    std::vector < InputTerm > terms_;
    // Positional mapping: input_to_terms_[i] = term indices fed by inputs[i].
    std::vector < std::vector < size_t >> input_to_terms_;
    // Pre-allocated output tensors for each term.
    std::vector < torch::Tensor > term_outputs_;
  };

}  // namespace isaac_deploy_core

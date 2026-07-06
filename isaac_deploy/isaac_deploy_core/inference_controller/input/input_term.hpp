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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "torch/torch.h"
#include "yaml-cpp/yaml.h"

#include "isaac_deploy_core/core/error.hpp"
#include "isaac_deploy_core/core/types.hpp"

namespace isaac_deploy_core {

/// Configuration for an InputTerm.
  struct InputTermConfig
  {
    /// Name of this input term (used as the neural network input slot identifier).
    std::string name;
    /// Kind of input (e.g., "joint_pos").
    ///
    /// Can be used by a middleware to know what message type to subscribe to.
    std::string kind;
    /// Name of the tensor that should be used for this input term.
    ///
    /// If multiple input terms use the same source, they will use the same
    /// tensor. Defaults to kind if empty.
    ///
    /// Example: If you have one input term for the current joint position and
    /// one input term for the history joint positions, then you could use the
    /// same source for both and then they would use the same tensor (but only
    /// the second input term would concatenate the history).
    ///
    /// Whenever possible this 1-to-n mapping should already be done in the
    /// neural network model and then every input term would have a unique
    /// source.
    std::string source;
    /// Expected output shape for the neural network.
    std::vector < int64_t > shape;
    /// Number of historical timesteps to concatenate (0 = no history).
    int history_length = 0;
    /// Whether to include the current value in the history buffer.
    bool include_current_in_history = true;
    /// Element order expected by the neural network. During activate(), this is compared
    /// against TensorSpec.names to compute reorder indices.
    std::vector < std::vector < std::string >> element_names;
    /// For inputs that depend on outputs (e.g., last_actions), the output name to use.
    std::string output_key;

    /// Create an InputTermConfig from a YAML node.
    static expected < InputTermConfig > create_from_yaml(const YAML::Node & yaml);
  };

/// InputTerm transforms raw input data into the format expected by the neural network.
///
/// Features:
/// - Reorders tensor elements to match the neural network's expected order
/// - Maintains history buffers (ring buffer of past values)
/// - Reshapes tensors to expected shape
  class InputTerm {
public:
    /// Create an InputTerm from configuration.
    static expected < InputTerm > create(const InputTermConfig & config);

    /// Get the name of this term (also the neural network input slot identifier).
    const std::string & name() const {return config_.name;}

    /// Get the kind of this term.
    const std::string & kind() const {return config_.kind;}

    /// Get the expected output shape.
    const std::vector < int64_t > & shape() const {
      return config_.shape;
    }

    /// Get the history length (0 = no history).
    int history_length() const {return config_.history_length;}

    /// Get the output key (for output-dependent inputs).
    const std::string & output_key() const {return config_.output_key;}

    /// Get the source.
    const std::string & source() const {return config_.source;}

    /// Prepare the term with input specs and initial values.
    /// This computes reorder indices and pre-allocates buffers.
    /// @param input_spec Specification of the input tensor from middleware.
    /// @param initial_value Initial input tensor (used for shape inference).
    expected < void > activate(const TensorSpec & input_spec, const torch::Tensor & initial_value);

    /// Clean up after the controller stops.
    expected < void > deactivate();

    /// Compute the tensor for this input term.
    /// @param input Input tensor from middleware.
    /// @param output Output tensor (pre-allocated, written in-place).
    expected < void > advance(
      const torch::Tensor & input,
      torch::Tensor & output);

    /// Check if this term depends on an output from a previous step.
    bool depends_on_output() const {return !config_.output_key.empty();}

private:
    explicit InputTerm(InputTermConfig config);

    InputTermConfig config_;

    // Pre-computed reorder operations (one per dimension that needs reordering).
    struct ReorderOp
    {
      torch::Tensor indices;
      int64_t dim;
    };
    std::vector < ReorderOp > reorder_ops_;

    // History buffer (ring buffer of past values).
    torch::Tensor history_buffer_;
    int history_write_idx_ = 0;
    bool history_initialized_ = false;
  };

}  // namespace isaac_deploy_core

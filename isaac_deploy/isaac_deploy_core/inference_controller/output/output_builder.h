// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "isaac_deploy_core/core/error.h"
#include "isaac_deploy_core/core/types.h"
#include "isaac_deploy_core/inference_controller/config_parser.h"
#include "isaac_deploy_core/inference_controller/output/output_term.h"

namespace isaac_deploy_core {

/// OutputBuilder manages multiple OutputTerms and builds the full output for middleware.
  class OutputBuilder {
public:
    /// Configuration for OutputBuilder.
    struct Config
    {
      /// Configurations for all output terms.
      std::vector < OutputTermConfig > terms;

      /// Create a Config from a parsed ModelConfig.
      static expected < Config > create_from_model_config(const ModelConfig & model_config);
    };

    /// Create an OutputBuilder from configuration.
    static expected < OutputBuilder > create(const Config & config);

    /// Activate the builder with output specifications.
    /// @param output_specs Specs for each output tensor expected by middleware.
    /// @param outputs Initial output tensors.
    expected < void > activate(
      const std::vector < TensorSpec > &output_specs,
      std::vector < NamedTensor > &outputs);

    /// Deactivate the builder.
    expected < void > deactivate();

    /// Build middleware outputs from neural network outputs.
    /// @param nn_outputs Neural network output tensors.
    /// @param outputs Middleware output tensors (written in-place).
    expected < void > advance(const TensorDict & nn_outputs, std::vector < NamedTensor > &outputs);

    /// Get the names of input tensors that this builder expects from the neural network.
    std::vector < std::string > get_input_names() const;

    /// Get the names of outputs this builder produces.
    std::vector < std::string > get_output_names() const;

    /// Get the shape of each output term (aligned with get_output_names()).
    std::vector < std::vector < int64_t >> get_output_shapes() const;

    /// Get output name -> kind mapping for all output terms.
    std::unordered_map < std::string, std::string > get_output_to_kind_map() const;

    /// Get the flattened element names for a given output term (last non-empty dim).
    /// Used by converters to populate message field names (e.g., joint names in JointCommand).
    std::vector < std::string > get_element_names(const std::string & output_name) const;

private:
    explicit OutputBuilder(std::vector < OutputTerm > terms);

    std::vector < OutputTerm > terms_;
    // Map from output name to term index.
    std::unordered_map < std::string, size_t > name_to_term_;
  };

}  // namespace isaac_deploy_core

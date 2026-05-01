// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "torch/torch.h"
#include "yaml-cpp/yaml.h"

#include "isaac_deploy_core/core/error.h"
#include "isaac_deploy_core/core/types.h"

namespace isaac_deploy_core {

/// Configuration for an OutputTerm.
  struct OutputTermConfig
  {
    /// Name of this output term (also the neural network output slot identifier).
    std::string name;
    /// Kind of output (e.g., "joint_pos_targets", "actions").
    std::string kind;
    /// Expected shape from the neural network.
    std::vector < int64_t > shape;
    /// Element order as produced by the neural network. During activate(), this is compared
    /// against TensorSpec.names (the middleware's expected order) to compute reorder indices.
    std::vector < std::vector < std::string >> element_names;

    /// Create an OutputTermConfig from a YAML node.
    static expected < OutputTermConfig > create_from_yaml(const YAML::Node & yaml);
  };

/// OutputTerm transforms neural network outputs into the format expected by the middleware.
///
/// Features:
/// - Reorders tensor elements from neural network order to middleware order
  class OutputTerm {
public:
    /// Create an OutputTerm from configuration.
    static expected < OutputTerm > create(const OutputTermConfig & config);

    /// Get the name of this term (also the neural network output slot identifier).
    const std::string & name() const {return config_.name;}

    /// Get the kind of this term.
    const std::string & kind() const {return config_.kind;}

    /// Get the expected shape from neural network.
    const std::vector < int64_t > & shape() const {
      return config_.shape;
    }

    /// Get the element names (per-dimension, from YAML config).
    const std::vector < std::vector < std::string >> & element_names() const {
      return config_.element_names;
    }

    /// Prepare the term with output specs.
    /// This computes reorder indices.
    /// @param output_spec Specification of the output tensor expected by middleware.
    /// @param output Initial output tensor (for shape/device info).
    expected < void > activate(const TensorSpec & output_spec, NamedTensor & output);

    /// Clean up after the controller stops.
    expected < void > deactivate();

    /// Write the neural network output to the output tensor. Real-time safe.
    /// @param nn_output Output tensor from neural network.
    /// @param output Output tensor for middleware (written in-place).
    expected < void > advance(const torch::Tensor & nn_output, NamedTensor & output);

private:
    explicit OutputTerm(OutputTermConfig config);

    OutputTermConfig config_;

    // Pre-computed reorder indices (empty if no reordering needed).
    torch::Tensor reorder_indices_;

    // Dimension to reorder along (typically the last dimension).
    int64_t reorder_dim_ = -1;
  };

}  // namespace isaac_deploy_core

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/output/output_term.h"

#include <utility>

#include "isaac_deploy_core/core/tensor_utils.h"

namespace isaac_deploy_core {

  namespace {

    /// Check if a reorder index vector is the identity permutation (0, 1, 2, ...).
    bool is_identity(const std::vector < int64_t > &indices, size_t source_size)
    {
      if (indices.size() != source_size) {
        return false;
      }
      for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] != static_cast < int64_t > (i)) {
          return false;
        }
      }
      return true;
    }

  }  // namespace

  expected < OutputTermConfig > OutputTermConfig::create_from_yaml(const YAML::Node & yaml) {
    OutputTermConfig config;

    if (!yaml["name"]) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "OutputTermConfig: 'name' is required"));
    }
    config.name = yaml["name"].as < std::string > ();

    if (yaml["kind"]) {
      config.kind = yaml["kind"].as < std::string > ();
    }

    if (!yaml["shape"]) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "OutputTermConfig: 'shape' is required"));
    }
    config.shape = yaml["shape"].as < std::vector < int64_t >> ();

    if (yaml["element_names"]) {
      for (const auto & dim : yaml["element_names"]) {
        config.element_names.push_back(dim.as < std::vector < std::string >> ());
      }
      // Right-align element_names with shape dimensions: pad with empty lists at front.
      // A user may omit the batch dimension.
      while (config.element_names.size() < config.shape.size()) {
        config.element_names.insert(config.element_names.begin(), std::vector<std::string>{});
      }
    }

    return config;
  }

  OutputTerm::OutputTerm(OutputTermConfig config)
  : config_(std::move(config)) {}

  expected < OutputTerm > OutputTerm::create(const OutputTermConfig & config) {
    if (config.name.empty()) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "OutputTerm name cannot be empty"));
    }
    if (config.shape.empty()) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "OutputTerm shape cannot be empty"));
    }
    return OutputTerm(config);
  }

  expected < void > OutputTerm::activate(const TensorSpec & output_spec, NamedTensor & /*output*/) {
    // Compute reorder indices if element_names are specified.
    // For output, we reorder FROM neural network order TO middleware order.
    if (!config_.element_names.empty() && !output_spec.names.empty()) {
      // Find the dimension to reorder (last non-empty dimension in element_names).
      for (int64_t dim = static_cast < int64_t > (config_.element_names.size()) - 1; dim >= 0;
        --dim)
      {
        if (!config_.element_names[dim].empty()) {
          reorder_dim_ = dim;
          break;
        }
      }

      if (reorder_dim_ >= 0 && reorder_dim_ < static_cast < int64_t > (output_spec.names.size())) {
        const auto & source_names = config_.element_names[reorder_dim_];  // NN order
        const auto & target_names = output_spec.names[reorder_dim_];  // Middleware order

        if (!source_names.empty() && !target_names.empty()) {
          auto indices_result = compute_reorder_indices(source_names, target_names);
          if (!indices_result.has_value()) {
            return tl::unexpected(
              make_error(
                Error::Code::kInvalidArgument,
                "OutputTerm '" + config_.name + "': reorder failed on dim " +
                std::to_string(reorder_dim_) + " — source has " +
                std::to_string(source_names.size()) + " names, target has " +
                std::to_string(target_names.size()) + ": " +
                indices_result.error().message));
          }
          if (!is_identity(*indices_result, source_names.size())) {
            reorder_indices_ = torch::tensor(*indices_result, torch::kLong);
          }
        }
      }
    }

    return expected < void > ();
  }

  expected < void > OutputTerm::deactivate() {
    reorder_indices_ = torch::Tensor();
    return expected < void > ();
  }

  expected < void > OutputTerm::advance(const torch::Tensor & nn_output, NamedTensor & output) {
    torch::Tensor processed = nn_output;

    // Apply reordering if indices are computed.
    if (reorder_indices_.defined() && reorder_dim_ >= 0) {
      processed = reorder_tensor(processed, reorder_indices_, reorder_dim_);
    }

    output.tensor = processed;
    return expected < void > ();
  }

}  // namespace isaac_deploy_core

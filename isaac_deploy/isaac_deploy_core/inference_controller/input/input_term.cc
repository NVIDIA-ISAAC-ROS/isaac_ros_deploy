// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/input/input_term.h"

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

  expected < InputTermConfig > InputTermConfig::create_from_yaml(const YAML::Node & yaml) {
    InputTermConfig config;

    if (!yaml["name"]) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "InputTermConfig: 'name' is required"));
    }
    config.name = yaml["name"].as < std::string > ();

    if (yaml["kind"]) {
      config.kind = yaml["kind"].as < std::string > ();
    }

    if (!yaml["shape"]) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "InputTermConfig: 'shape' is required"));
    }
    config.shape = yaml["shape"].as < std::vector < int64_t >> ();

    // Accept both 'history_length' (preferred) and 'history' (protomotions format).
    if (yaml["history_length"]) {
      config.history_length = yaml["history_length"].as < int > ();
    } else if (yaml["history"]) {
      config.history_length = yaml["history"].as < int > ();
    }

    // Accept both 'include_current_in_history' (preferred) and
    // 'include_current_value_in_history' (protomotions format).
    if (yaml["include_current_in_history"]) {
      config.include_current_in_history = yaml["include_current_in_history"].as < bool > ();
    } else if (yaml["include_current_value_in_history"]) {
      config.include_current_in_history = yaml["include_current_value_in_history"].as < bool > ();
    }

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

    if (yaml["source"]) {
      config.source = yaml["source"].as < std::string > ();
    }

    return config;
  }

  InputTerm::InputTerm(InputTermConfig config)
  : config_(std::move(config)) {}

  expected < InputTerm > InputTerm::create(const InputTermConfig & config) {
    if (config.name.empty()) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "InputTerm name cannot be empty"));
    }
    // kind is optional: feedback terms use output_key, topic-only terms use name as source.
    if (config.shape.empty()) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "InputTerm shape cannot be empty"));
    }

    InputTermConfig cfg = config;
    if (cfg.source.empty()) {
      if (!cfg.output_key.empty()) {
        cfg.source = cfg.output_key;
      } else if (!cfg.kind.empty()) {
        cfg.source = cfg.kind;
      } else {
        cfg.source = cfg.name;
      }
    }

    return InputTerm(std::move(cfg));
  }

  expected < void > InputTerm::activate(
    const TensorSpec & input_spec, const torch::Tensor & initial_value) {
    // Compute reorder indices for each dimension that has element names in both
    // the config and the input spec. For each target dimension with names, find the
    // matching source dimension (where target names are a subset of source names).
    reorder_ops_.clear();
    if (!config_.element_names.empty() && !input_spec.names.empty()) {
      // Track which source dimensions have already been matched.
      std::vector < bool > source_used(input_spec.names.size(), false);

      for (size_t td = 0; td < config_.element_names.size(); ++td) {
        const auto & target_names = config_.element_names[td];
        if (target_names.empty()) {
          continue;
        }

        for (size_t sd = 0; sd < input_spec.names.size(); ++sd) {
          if (source_used[sd]) {
            continue;
          }
          const auto & source_names = input_spec.names[sd];
          if (source_names.empty()) {
            continue;
          }

          auto indices_result = compute_reorder_indices(source_names, target_names);
          if (indices_result.has_value()) {
            // Skip identity reorders (same order, same count).
            if (!is_identity(*indices_result, source_names.size())) {
              reorder_ops_.push_back(
                {torch::tensor(*indices_result, torch::kLong),
                  static_cast < int64_t > (sd)});
            }
            source_used[sd] = true;
            break;  // Found match for this target dim.
          }
        }
      }
    }

    // Pre-allocate history buffer if needed.
    if (config_.history_length > 0) {
      // Shape for history: [history_length, ...feature_dims].
      std::vector < int64_t > history_shape;
      history_shape.push_back(config_.history_length);

      // Get feature dimensions from input tensor or config shape.
      if (initial_value.dim() > 1) {
        // Use actual input tensor dimensions (skip batch dim).
        // If a reorder op targets a dimension, use the target size
        // (the input may have more elements than the policy expects).
        for (int64_t i = 1; i < initial_value.dim(); ++i) {
          int64_t dim_size = initial_value.size(i);
          for (const auto & op : reorder_ops_) {
            if (op.dim == i) {
              dim_size = op.indices.size(0);
              break;
            }
          }
          history_shape.push_back(dim_size);
        }
      } else if (config_.shape.size() > 1) {
        // For output-dependent inputs, use expected shape from config.
        // Config shape: [batch, history, ...features] or [batch, ...features].
        // Skip batch dim (index 0) and history dim if present (3 dims total).
        size_t feature_start = (config_.shape.size() == 3) ? 2 : 1;
        for (size_t i = feature_start; i < config_.shape.size(); ++i) {
          history_shape.push_back(config_.shape[i]);
        }
      }

      auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
      history_buffer_ = torch::zeros(history_shape, tensor_opts);
      history_write_idx_ = 0;
      history_initialized_ = false;
    }

    return expected < void > ();
  }

  expected < void > InputTerm::deactivate() {
    reorder_ops_.clear();
    history_buffer_ = torch::Tensor();
    history_write_idx_ = 0;
    history_initialized_ = false;
    return expected < void > ();
  }

  expected < void > InputTerm::advance(
    const torch::Tensor & input,
    torch::Tensor & output) {
    // Apply all reorder operations.
    torch::Tensor reordered = input;
    for (const auto & op : reorder_ops_) {
      reordered = reorder_tensor(reordered, op.indices, op.dim);
    }

    torch::Tensor processed = reordered;

    // Handle history.
    if (config_.history_length > 0) {
      if (!history_initialized_) {
        // Initialize history with the first value.
        for (int i = 0; i < config_.history_length; ++i) {
          history_buffer_[i] = reordered.squeeze(0);
        }
        history_initialized_ = true;
      }

      if (config_.include_current_in_history) {
        // Update history with current value.
        history_buffer_[history_write_idx_] = reordered.squeeze(0);
        history_write_idx_ = (history_write_idx_ + 1) % config_.history_length;
      }

      // Build output from history (oldest to newest).
      std::vector < torch::Tensor > history_slices;
      history_slices.reserve(config_.history_length);
      for (int i = 0; i < config_.history_length; ++i) {
        int idx = (history_write_idx_ + i) % config_.history_length;
        history_slices.push_back(history_buffer_[idx]);
      }
      processed = torch::stack(history_slices, 0).unsqueeze(0);

      if (!config_.include_current_in_history) {
        // Update history after building output.
        history_buffer_[history_write_idx_] = reordered.squeeze(0);
        history_write_idx_ = (history_write_idx_ + 1) % config_.history_length;
      }
    }

    // Reshape to expected shape if needed.
    if (processed.sizes().vec() != config_.shape) {
      processed = processed.reshape(config_.shape);
    }

    output = processed;
    return expected < void > ();
  }

}  // namespace isaac_deploy_core

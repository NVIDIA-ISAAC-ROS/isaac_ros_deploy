// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/inference_controller.h"

#include <unordered_map>
#include <utility>

namespace isaac_deploy_core {

  InferenceController::InferenceController(
    InputBuilder input_builder,
    std::unique_ptr < InferenceRunner > runner,
    OutputBuilder output_builder)
  : input_builder_(std::move(input_builder)),
    runner_(std::move(runner)),
    output_builder_(std::move(output_builder)) {}

  expected < InferenceController > InferenceController::create(InferenceControllerConfig config) {
    auto input_builder_result = InputBuilder::create(config.inputs);
    if (!input_builder_result.has_value()) {
      return tl::unexpected(input_builder_result.error());
    }

    auto output_builder_result = OutputBuilder::create(config.outputs);
    if (!output_builder_result.has_value()) {
      return tl::unexpected(output_builder_result.error());
    }

    auto runner_result = InferenceRunner::create(config.runner);
    if (!runner_result.has_value()) {
      return tl::unexpected(runner_result.error());
    }

    return InferenceController(
      std::move(*input_builder_result), std::move(*runner_result),
      std::move(*output_builder_result));
  }

  expected < void > InferenceController::activate(
    const std::vector < NamedTensor > &inputs,
    const std::vector < TensorSpec > &input_specs,
    const std::vector < TensorSpec > &output_specs,
    std::vector < NamedTensor > &outputs) {
    // Build the full inputs vector (external + feedback).
    auto all_sources = input_builder_.get_unique_source_names();
    auto feedback_names = input_builder_.get_feedback_input_names();
    auto feedback_shapes = input_builder_.get_feedback_input_shapes();

    // Build feedback name -> shape lookup (combined set + shape map).
    std::unordered_map < std::string, std::vector < int64_t >> feedback_shape_map;
    for (size_t i = 0; i < feedback_names.size(); ++i) {
      feedback_shape_map[feedback_names[i]] = feedback_shapes[i];
    }

    // Build name-based lookup for caller's inputs.
    std::unordered_map < std::string, size_t > ext_name_to_idx;
    for (size_t i = 0; i < inputs.size(); ++i) {
      ext_name_to_idx[inputs[i].name] = i;
    }

    all_inputs_.clear();
    std::vector < TensorSpec > all_specs;
    external_positions_.clear();
    external_positions_.reserve(inputs.size());
    feedback_mappings_.clear();
    input_names_.clear();

    // Track which external inputs are consumed so we can detect unused ones.
    std::vector < bool > ext_used(inputs.size(), false);

    for (const auto & source : all_sources) {
      size_t all_idx = all_inputs_.size();
      auto feedback_it = feedback_shape_map.find(source);

      if (feedback_it != feedback_shape_map.end()) {
        all_inputs_.push_back(
          {.name = source,
            .tensor = torch::zeros(feedback_it->second)});
        all_specs.push_back({});
        feedback_mappings_.push_back(
          {.input_idx = all_idx,
            .output_key = source});
      } else {
        auto ext_it = ext_name_to_idx.find(source);
        if (ext_it == ext_name_to_idx.end()) {
          return tl::unexpected(
            make_error(
              Error::Code::kNotFound,
              "External input not found for source: " + source));
        }
        size_t ext_idx = ext_it->second;
        all_inputs_.push_back(inputs[ext_idx]);
        all_specs.push_back(input_specs[ext_idx]);
        external_positions_.push_back(
          {.ext_idx = ext_idx, .all_idx = all_idx});
        ext_used[ext_idx] = true;
        input_names_.push_back(source);
      }
    }

    // Verify all external inputs are consumed by the policy.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!ext_used[i]) {
        return tl::unexpected(
          make_error(
            Error::Code::kInvalidArgument,
            "External input '" + inputs[i].name +
            "' is not referenced by the policy config"));
      }
    }

    // Activate input builder with all inputs.
    auto input_result = input_builder_.activate(all_inputs_, all_specs);
    if (!input_result.has_value()) {
      return tl::unexpected(input_result.error());
    }

    // Activate output builder.
    auto output_result = output_builder_.activate(output_specs, outputs);
    if (!output_result.has_value()) {
      return tl::unexpected(output_result.error());
    }

    output_names_ = output_builder_.get_output_names();

    // Reset neural network I/O buffers.
    nn_inputs_.clear();
    nn_outputs_.clear();
    return expected < void > ();
  }

  expected < void > InferenceController::deactivate() {
    auto input_result = input_builder_.deactivate();
    if (!input_result.has_value()) {
      return tl::unexpected(input_result.error());
    }

    auto output_result = output_builder_.deactivate();
    if (!output_result.has_value()) {
      return tl::unexpected(output_result.error());
    }

    nn_inputs_.clear();
    nn_outputs_.clear();
    all_inputs_.clear();
    external_positions_.clear();
    feedback_mappings_.clear();

    return expected < void > ();
  }

  expected < void > InferenceController::advance(
    const std::vector < NamedTensor > &inputs,
    std::vector < NamedTensor > &outputs) {
    // Copy external inputs into their positions.
    for (const auto & pos : external_positions_) {
      all_inputs_[pos.all_idx].tensor = inputs[pos.ext_idx].tensor;
    }

    // Build neural network inputs (fully positional, RT-safe).
    auto input_result = input_builder_.advance(all_inputs_, nn_inputs_);
    if (!input_result.has_value()) {
      return tl::unexpected(input_result.error());
    }

    // Run inference.
    auto run_result = runner_->run(nn_inputs_, nn_outputs_);
    if (!run_result.has_value()) {
      return tl::unexpected(run_result.error());
    }

    // Update feedback positions with NN outputs for next cycle.
    for (const auto & mapping : feedback_mappings_) {
      auto it = nn_outputs_.find(mapping.output_key);
      if (it == nn_outputs_.end()) {
        return tl::unexpected(
          make_error(
            Error::Code::kNotFound,
            "Feedback output key '" + mapping.output_key +
            "' not found in neural network outputs"));
      }
      all_inputs_[mapping.input_idx].tensor = it->second;
    }

    // Build middleware outputs.
    auto output_result = output_builder_.advance(nn_outputs_, outputs);
    if (!output_result.has_value()) {
      return tl::unexpected(output_result.error());
    }

    return expected < void > ();
  }

}  // namespace isaac_deploy_core

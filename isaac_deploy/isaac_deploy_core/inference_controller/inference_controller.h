// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "isaac_deploy_core/core/error.h"
#include "isaac_deploy_core/core/types.h"
#include "isaac_deploy_core/inference_controller/input/input_builder.h"
#include "isaac_deploy_core/inference_controller/output/output_builder.h"
#include "isaac_deploy_core/inference_controller/runner/inference_runner.h"

namespace isaac_deploy_core {

/// Configuration for the InferenceController.
  struct InferenceControllerConfig
  {
    /// Configuration of the inputs of the controller.
    InputBuilder::Config inputs;
    /// Configuration of the outputs of the controller.
    OutputBuilder::Config outputs;
    /// Configuration for the inference runner.
    InferenceRunner::Config runner;
  };

/// InferenceController runs neural network inference with input/output processing.
///
/// The controller orchestrates:
/// 1. InputBuilder: Preprocesses inputs (reordering, history, reshaping)
/// 2. InferenceRunner: Executes the neural network
/// 3. OutputBuilder: Postprocesses outputs (reordering)
///
/// The advance() method is real-time safe (does not allocate) after activate() is called.
  class InferenceController {
public:
    /// Create the controller from configuration.
    static expected < InferenceController > create(InferenceControllerConfig config);

    /// Activate the controller.
    /// Call this to prepare the controller just before it starts running.
    /// This calculates remapping indices for all inputs and outputs based on
    /// the provided specs and pre-allocates all internal buffers.
    /// @param inputs Initial input tensors (one per unique source, .name = source).
    /// @param input_specs TensorSpec per input (positionally aligned with inputs).
    /// @param output_specs Specs describing the expected structure of outputs.
    /// @param outputs Pre-allocated output tensors (controller writes into these).
    expected < void > activate(
      const std::vector < NamedTensor > &inputs,
      const std::vector < TensorSpec > &input_specs,
      const std::vector < TensorSpec > &output_specs,
      std::vector < NamedTensor > &outputs);

    /// Deactivate the controller.
    /// Call this to clean up after the controller stops running.
    expected < void > deactivate();

    /// Advance the controller by one timestep.
    /// This method is real-time safe (does not allocate).
    /// Inputs must be in the same order as passed to activate().
    /// @param inputs Input tensors (positionally aligned with activate).
    /// @param outputs Output tensors (written in-place by the controller).
    expected < void > advance(
      const std::vector < NamedTensor > &inputs,
      std::vector < NamedTensor > &outputs);

    /// Get the names of required inputs.
    const std::vector < std::string > & input_names() const {
      return input_names_;
    }

    /// Get the names of produced outputs.
    const std::vector < std::string > & output_names() const {
      return output_names_;
    }

private:
    InferenceController(
      InputBuilder input_builder, std::unique_ptr < InferenceRunner > runner,
      OutputBuilder output_builder);

    InputBuilder input_builder_;
    std::unique_ptr < InferenceRunner > runner_;
    OutputBuilder output_builder_;

    std::vector < std::string > input_names_;
    std::vector < std::string > output_names_;

    // Pre-allocated buffers for neural network I/O.
    TensorDict nn_inputs_;
    TensorDict nn_outputs_;

    // All inputs to InputBuilder (external + feedback), positionally aligned.
    std::vector < NamedTensor > all_inputs_;
    // Maps external input index to position in all_inputs_.
    struct ExternalPosition
    {
      size_t ext_idx;
      size_t all_idx;
    };
    std::vector < ExternalPosition > external_positions_;
    // Maps feedback positions in all_inputs_ to NN output keys.
    struct FeedbackMapping
    {
      size_t input_idx;
      std::string output_key;
    };
    std::vector < FeedbackMapping > feedback_mappings_;
  };

}  // namespace isaac_deploy_core

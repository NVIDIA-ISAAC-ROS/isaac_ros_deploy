// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "isaac_deploy_core/core/error.h"
#include "isaac_deploy_core/core/types.h"

namespace isaac_deploy_core {

/// Abstract interface for neural network inference.
///
/// InferenceRunner implementations execute models using different backends
/// (ONNX Runtime, Triton, TensorRT, etc.). All implementations must support
/// allocation-free run() after initialization.
  class InferenceRunner {
public:
    /// Configuration for creating an InferenceRunner.
    struct Config
    {
      /// Path to the model file.
      std::filesystem::path model_path;
      /// Type of runner to use (e.g., "onnx").
      std::string runner_type;
    };

    /// Create an InferenceRunner from configuration.
    /// @param config Configuration specifying the model and runner type.
    /// @return The runner or an error.
    static expected < std::unique_ptr < InferenceRunner >> create(const Config & config);

    virtual ~InferenceRunner() = default;

    /// Run inference with named inputs, writing results to outputs.
    /// This method is real-time safe (does not allocate) if outputs are pre-allocated.
    /// @param inputs Map of input name to tensor.
    /// @param outputs Map of output name to tensor (written in-place).
    /// @return Success or error status.
    virtual expected < void > run(const TensorDict & inputs, TensorDict & outputs) = 0;

    /// Get the names of expected inputs.
    virtual std::vector < std::string > get_input_names() const = 0;

    /// Get the names of produced outputs.
    virtual std::vector < std::string > get_output_names() const = 0;

    /// Reset any internal state (for recurrent/stateful models).
    virtual void reset() = 0;
  };

}  // namespace isaac_deploy_core

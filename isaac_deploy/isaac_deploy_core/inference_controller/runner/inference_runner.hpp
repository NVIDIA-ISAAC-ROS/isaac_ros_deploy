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
#include <memory>
#include <string>
#include <vector>

#include "isaac_deploy_core/core/error.hpp"
#include "isaac_deploy_core/core/types.hpp"

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

    /// Run one inference with zero-filled dummy inputs to trigger JIT compilation.
    /// Call once after creation (before the real-time loop) to avoid a stall on the
    /// first real advance() call.  Default implementation is a no-op.
    virtual expected<void> warmup() { return {}; }
  };

}  // namespace isaac_deploy_core

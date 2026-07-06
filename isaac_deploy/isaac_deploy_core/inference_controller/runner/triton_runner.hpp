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

#include "isaac_deploy_core/inference_controller/runner/inference_runner.hpp"

// Forward-declare Triton opaque types to avoid leaking the C API header.
struct TRITONSERVER_Server;
struct TRITONSERVER_ResponseAllocator;

namespace isaac_deploy_core {

/// Configuration for Triton Runner.
struct TritonRunnerConfig {
  /// Path to the .onnx model file.
  std::filesystem::path model_path;
  /// Path to Triton backend directory. Auto-detected from TRITON_BACKEND_DIRECTORY env var if
  /// empty.
  std::filesystem::path backend_dir;
};

/// Triton Inference Server inference backend.
///
/// Implements neural network inference using the Triton C API with an embedded in-process server.
/// The ONNX model is loaded via Triton's onnxruntime backend with automatic model configuration.
class TritonRunner : public InferenceRunner {
 public:
  ~TritonRunner() override;

  /// Create a TritonRunner from configuration.
  /// @param config Configuration for the runner.
  /// @return The runner or an error.
  static expected<std::unique_ptr<TritonRunner>> create(const TritonRunnerConfig& config);

  /// Run inference with named inputs.
  /// @param inputs Map of input name to tensor.
  /// @param outputs Map of output name to tensor (written in-place).
  /// @return Success or error status.
  expected<void> run(const TensorDict& inputs, TensorDict& outputs) override;

  /// Get the names of expected inputs.
  std::vector<std::string> get_input_names() const override;

  /// Get the names of produced outputs.
  std::vector<std::string> get_output_names() const override;

  /// Reset internal state (no-op for Triton models).
  void reset() override;

  /// Run one inference with zero-filled dummy inputs to trigger TensorRT JIT compilation.
  expected<void> warmup() override;

  /// Get the expected input shapes.
  const std::vector<std::vector<int64_t>>& get_input_shapes() const { return input_shapes_; }

  /// Get the expected output shapes.
  const std::vector<std::vector<int64_t>>& get_output_shapes() const { return output_shapes_; }

 private:
  explicit TritonRunner(const TritonRunnerConfig& config);

  /// Initialize the runner: set up model repo, start server, query metadata.
  expected<void> init();

  /// Create a temporary model repository with the ONNX model symlinked.
  expected<void> setup_model_repo();

  /// Start the embedded Triton server and wait for readiness.
  expected<void> start_server();

  /// Query model metadata to extract input/output names and shapes.
  expected<void> query_model_metadata();

  TritonRunnerConfig config_;
  std::filesystem::path model_repo_dir_;
  std::string model_name_{"model"};
  TRITONSERVER_Server* server_{nullptr};
  TRITONSERVER_ResponseAllocator* allocator_{nullptr};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
};

}  // namespace isaac_deploy_core

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "isaac_deploy_core/inference_controller/runner/inference_runner.h"

namespace isaac_deploy_core {

/// Mock inference runner for testing.
///
/// This runner simply copies the first input tensor to the "output" key.
  class MockRunner: public InferenceRunner {
public:
    /// Create a MockRunner.
    /// @return The runner.
    static expected < std::unique_ptr < MockRunner >> create();

    /// Run inference - copies first input to output.
    expected < void > run(const TensorDict & inputs, TensorDict & outputs) override;

    /// Get the names of expected inputs.
    std::vector < std::string > get_input_names() const override;

    /// Get the names of produced outputs.
    std::vector < std::string > get_output_names() const override;

    /// Reset internal state (no-op).
    void reset() override;
  };

}  // namespace isaac_deploy_core

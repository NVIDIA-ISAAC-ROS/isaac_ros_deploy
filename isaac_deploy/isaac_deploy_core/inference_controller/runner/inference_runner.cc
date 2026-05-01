// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/runner/inference_runner.h"

#include "isaac_deploy_core/inference_controller/runner/mock_runner.h"
#include "isaac_deploy_core/inference_controller/runner/triton_runner.h"

namespace isaac_deploy_core {

  expected < std::unique_ptr < InferenceRunner >> InferenceRunner::create(const Config & config) {
    if (config.runner_type == "triton") {
      TritonRunnerConfig triton_config{.model_path = config.model_path};
      return TritonRunner::create(triton_config);
    }

    if (config.runner_type == "mock") {
      return MockRunner::create();
    }

    return tl::unexpected(
      make_error(
        Error::Code::kInvalidArgument,
        "Unknown runner type: " + config.runner_type));
  }

}  // namespace isaac_deploy_core

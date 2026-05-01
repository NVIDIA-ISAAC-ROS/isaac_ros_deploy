// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/runner/mock_runner.h"

namespace isaac_deploy_core {

  expected < std::unique_ptr < MockRunner >> MockRunner::create() {
    return std::unique_ptr < MockRunner > (new MockRunner());
  }

  expected < void > MockRunner::run(const TensorDict & inputs, TensorDict & outputs) {
    if (inputs.empty()) {
      return tl::unexpected(make_error(Error::Code::kInvalidArgument, "No inputs provided"));
    }
    outputs["output"] = inputs.begin()->second.clone();
    return expected < void > ();
  }

  std::vector < std::string > MockRunner::get_input_names() const {
    return {"input"};
  }

  std::vector < std::string > MockRunner::get_output_names() const {
    return {"output"};
  }

  void MockRunner::reset() {}

}  // namespace isaac_deploy_core

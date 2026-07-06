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

#include "isaac_deploy_core/inference_controller/runner/mock_runner.hpp"

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

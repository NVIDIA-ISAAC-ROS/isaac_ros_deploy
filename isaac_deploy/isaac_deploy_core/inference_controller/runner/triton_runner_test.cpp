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

#include "isaac_deploy_core/inference_controller/runner/triton_runner.hpp"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

TEST(TritonRunnerTest, CreateFailsWithNonexistentFile) {
  TritonRunnerConfig config{.model_path = "/nonexistent/path/model.onnx"};
  auto result = TritonRunner::create(config);
  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code, Error::Code::kNotFound);
}

}  // namespace isaac_deploy_core

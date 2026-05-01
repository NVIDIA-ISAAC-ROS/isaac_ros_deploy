// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/runner/triton_runner.h"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

TEST(TritonRunnerTest, CreateFailsWithNonexistentFile) {
  TritonRunnerConfig config{.model_path = "/nonexistent/path/model.onnx"};
  auto result = TritonRunner::create(config);
  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code, Error::Code::kNotFound);
}

}  // namespace isaac_deploy_core

// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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

#include "isaac_ros_deploy_ros2_control/controllers/leapp_backend_mapping.hpp"

#include <gtest/gtest.h>

namespace isaac_ros_deploy_ros2_control {
namespace controllers {

TEST(LeappBackendToRunnerTypeTest, MapsOnnxToTriton) {
  auto result = leapp_backend_to_runner_type("onnx");
  ASSERT_TRUE(result.has_value()) << result.error().message;
  EXPECT_EQ(*result, "triton");
}

TEST(LeappBackendToRunnerTypeTest, RejectsMissingBackend) {
  auto result = leapp_backend_to_runner_type("");
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message.find("parameters.backend"), std::string::npos)
      << result.error().message;
}

TEST(LeappBackendToRunnerTypeTest, RejectsJitBackend) {
  auto result = leapp_backend_to_runner_type("jit");
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message.find("backend 'jit'"), std::string::npos)
      << result.error().message;
}

TEST(LeappBackendToRunnerTypeTest, RejectsUnknownBackend) {
  auto result = leapp_backend_to_runner_type("custom");
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message.find("backend 'custom'"), std::string::npos)
      << result.error().message;
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

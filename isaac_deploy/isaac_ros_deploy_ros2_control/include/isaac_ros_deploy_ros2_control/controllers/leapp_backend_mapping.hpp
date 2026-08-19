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

#pragma once

#include <string>

#include "isaac_deploy_core/core/error.hpp"

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

inline isaac_deploy_core::expected<std::string> leapp_backend_to_runner_type(
  const std::string & backend)
{
  if (backend == "onnx") {
    return "triton";
  }
  if (backend.empty()) {
    return tl::unexpected(
        isaac_deploy_core::make_error(
          isaac_deploy_core::Error::Code::kInvalidArgument,
          "LEAPP model parameters.backend is missing. "
          "Export the LEAPP graph with the ONNX backend."));
  }
  return tl::unexpected(
      isaac_deploy_core::make_error(
        isaac_deploy_core::Error::Code::kInvalidArgument,
        "LEAPP model backend '" + backend +
        "' is not supported by ros2_control InferenceController. "
        "Export the LEAPP graph with the ONNX backend."));
}

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

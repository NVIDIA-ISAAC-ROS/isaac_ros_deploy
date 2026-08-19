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

#include <filesystem>

#include "isaac_deploy_core/core/error.hpp"
#include "isaac_deploy_core/core/types.hpp"
#include "isaac_deploy_core/inference_controller/config_parser.hpp"

namespace isaac_deploy_core
{

/// Load tensors from a safetensors file.
expected<TensorDict> load_safetensors(const std::filesystem::path & path);

/// Load and map LEAPP feedback initial values to runtime feedback source names.
///
/// LEAPP stores initial values by fully-prefixed feedback target input name
/// (`model/input`). The runtime consumes feedback by the source output name
/// (`output`, with the model prefix stripped), matching InputBuilder sources.
expected<TensorDict> load_feedback_initial_values(
  const GraphConfig & graph,
  const std::filesystem::path & config_path);

}  // namespace isaac_deploy_core

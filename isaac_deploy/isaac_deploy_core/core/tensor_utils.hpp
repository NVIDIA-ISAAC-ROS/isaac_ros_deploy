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

#include <string>
#include <vector>

#include "torch/torch.h"

#include "isaac_deploy_core/core/error.hpp"
#include "isaac_deploy_core/core/types.hpp"

namespace isaac_deploy_core {

/// Compute reorder indices from source order to target order.
/// @param source_names Names in source order.
/// @param target_names Names in target order.
/// @return Vector of indices such that target[i] = source[result[i]], or error if names don't
/// match.
  expected < std::vector < int64_t >> compute_reorder_indices(
    const std::vector < std::string > &source_names,
    const std::vector < std::string > &target_names);

/// Reorder a tensor along the specified dimension using precomputed indices.
/// @param tensor Input tensor.
/// @param indices Reorder indices from compute_reorder_indices.
/// @param dim Dimension to reorder along.
/// @return Reordered tensor.
  torch::Tensor reorder_tensor(
    const torch::Tensor & tensor, const std::vector < int64_t > &indices,
    int64_t dim);

/// Reorder a tensor along the specified dimension using precomputed index tensor.
/// This version is allocation-free if index_tensor is pre-allocated.
/// @param tensor Input tensor.
/// @param index_tensor Pre-allocated tensor containing reorder indices.
/// @param dim Dimension to reorder along.
/// @return Reordered tensor.
  torch::Tensor reorder_tensor(
    const torch::Tensor & tensor, const torch::Tensor & index_tensor,
    int64_t dim);

}  // namespace isaac_deploy_core

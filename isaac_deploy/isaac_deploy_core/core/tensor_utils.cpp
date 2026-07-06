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

#include "isaac_deploy_core/core/tensor_utils.hpp"

#include <algorithm>
#include <unordered_map>

namespace isaac_deploy_core {

  expected < std::vector < int64_t >> compute_reorder_indices(
    const std::vector < std::string > &source_names,
    const std::vector < std::string > &target_names) {
    // Build index map from source names to their positions.
    std::unordered_map < std::string, int64_t > source_index_map;
    for (size_t i = 0; i < source_names.size(); ++i) {
      if (source_index_map.contains(source_names[i])) {
        return tl::unexpected(
          make_error(
            Error::Code::kInvalidArgument,
            "Duplicate name in source: " + source_names[i]));
      }
      source_index_map[source_names[i]] = static_cast < int64_t > (i);
    }

    // Compute indices for target order.
    std::vector < int64_t > indices;
    indices.reserve(target_names.size());
    for (const auto & name : target_names) {
      auto it = source_index_map.find(name);
      if (it == source_index_map.end()) {
        return tl::unexpected(
          make_error(
            Error::Code::kNotFound,
            "Name not found in source: " + name));
      }
      indices.push_back(it->second);
    }

    return indices;
  }

  torch::Tensor reorder_tensor(
    const torch::Tensor & tensor, const std::vector < int64_t > &indices,
    int64_t dim)
  {
    auto index_tensor =
      torch::tensor(indices, torch::TensorOptions().dtype(torch::kLong).device(tensor.device()));
    return tensor.index_select(dim, index_tensor);
  }

  torch::Tensor reorder_tensor(
    const torch::Tensor & tensor, const torch::Tensor & index_tensor,
    int64_t dim)
  {
    return tensor.index_select(dim, index_tensor);
  }

}  // namespace isaac_deploy_core

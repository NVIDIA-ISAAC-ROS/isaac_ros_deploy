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

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  TEST(ComputeReorderIndicesTest, SimpleReorder) {
    std::vector < std::string > source = {"a", "b", "c"};
    std::vector < std::string > target = {"c", "a", "b"};

    auto result = compute_reorder_indices(source, target);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, std::vector < int64_t > ({2, 0, 1}));
  }

  TEST(ComputeReorderIndicesTest, IdentityReorder) {
    std::vector < std::string > names = {"x", "y", "z"};

    auto result = compute_reorder_indices(names, names);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, std::vector < int64_t > ({0, 1, 2}));
  }

  TEST(ComputeReorderIndicesTest, SizeMismatch) {
    std::vector < std::string > source = {"a", "b"};
    std::vector < std::string > target = {"a", "b", "c"};

    auto result = compute_reorder_indices(source, target);
    EXPECT_FALSE(result.has_value());
  }

  TEST(ComputeReorderIndicesTest, SubsetSelection) {
    std::vector < std::string > source = {"a", "b", "c", "d", "e"};
    std::vector < std::string > target = {"c", "a", "e"};

    auto result = compute_reorder_indices(source, target);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, std::vector < int64_t > ({2, 0, 4}));
  }

  TEST(ComputeReorderIndicesTest, MissingName) {
    std::vector < std::string > source = {"a", "b", "c"};
    std::vector < std::string > target = {"a", "b", "d"};

    auto result = compute_reorder_indices(source, target);
    EXPECT_FALSE(result.has_value());
  }

  TEST(ComputeReorderIndicesTest, DuplicateInSource) {
    std::vector < std::string > source = {"a", "a", "c"};
    std::vector < std::string > target = {"a", "c", "a"};

    auto result = compute_reorder_indices(source, target);
    EXPECT_FALSE(result.has_value());
  }

  TEST(ReorderTensorTest, Reorder1D) {
    auto tensor = torch::tensor({1.0f, 2.0f, 3.0f});
    std::vector < int64_t > indices = {2, 0, 1};

    auto result = reorder_tensor(tensor, indices, 0);
    auto expected = torch::tensor({3.0f, 1.0f, 2.0f});
    EXPECT_TRUE(torch::allclose(result, expected));
  }

  TEST(ReorderTensorTest, Reorder2DAlongDim1) {
    auto tensor = torch::tensor({{1.0f, 2.0f, 3.0f}});
    std::vector < int64_t > indices = {2, 0, 1};

    auto result = reorder_tensor(tensor, indices, 1);
    auto expected = torch::tensor({{3.0f, 1.0f, 2.0f}});
    EXPECT_TRUE(torch::allclose(result, expected));
  }

  TEST(ReorderTensorTest, ReorderWithPreallocatedIndices) {
    auto tensor = torch::tensor({{1.0f, 2.0f, 3.0f}});
    auto index_tensor = torch::tensor({2, 0, 1}, torch::kLong);

    auto result = reorder_tensor(tensor, index_tensor, 1);
    auto expected = torch::tensor({{3.0f, 1.0f, 2.0f}});
    EXPECT_TRUE(torch::allclose(result, expected));
  }

}  // namespace isaac_deploy_core

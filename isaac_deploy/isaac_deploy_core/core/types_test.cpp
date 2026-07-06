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

#include "isaac_deploy_core/core/types.hpp"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  TEST(TensorSpecTest, EmptySpec) {
    TensorSpec spec;
    EXPECT_TRUE(spec.names.empty());
  }

  TEST(TensorSpecTest, JointSpec) {
    TensorSpec spec {.names = {{"batch"}, {"joint1", "joint2", "joint3"}}};
    EXPECT_EQ(spec.names.size(), 2);
    EXPECT_EQ(spec.names[1].size(), 3);
    EXPECT_EQ(spec.names[1][0], "joint1");
  }

  TEST(NamedTensorTest, Construction) {
    NamedTensor nt {
      .name = "test",
      .timestamp_ns = 12345,
      .tensor = torch::zeros({1, 3}),
    };
    EXPECT_EQ(nt.name, "test");
    EXPECT_EQ(nt.timestamp_ns, 12345);
    EXPECT_EQ(nt.tensor.sizes(), std::vector < int64_t > ({1, 3}));
  }

  TEST(TensorDictTest, InsertAndAccess) {
    TensorDict dict;
    dict["a"] = torch::ones({2, 3});
    dict["b"] = torch::zeros({4});

    EXPECT_EQ(dict.size(), 2);
    EXPECT_EQ(dict["a"].sizes(), std::vector < int64_t > ({2, 3}));
    EXPECT_EQ(dict["b"].sizes(), std::vector < int64_t > ({4}));
  }

}  // namespace isaac_deploy_core

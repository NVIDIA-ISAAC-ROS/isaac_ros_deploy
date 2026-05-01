// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/core/types.h"

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

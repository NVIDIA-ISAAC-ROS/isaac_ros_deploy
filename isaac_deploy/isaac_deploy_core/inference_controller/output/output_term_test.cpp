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

#include "isaac_deploy_core/inference_controller/output/output_term.hpp"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  TEST(OutputTermTest, CreateSuccess) {
    OutputTermConfig config {
      .name = "joint_pos_targets",
      .kind = "joint_pos_targets",
      .shape = {1, 3},
    };
    auto result = OutputTerm::create(config);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->name(), "joint_pos_targets");
  }

  TEST(OutputTermTest, CreateFailsWithEmptyName) {
    OutputTermConfig config {.name = "", .kind = "out", .shape = {1, 3}};
    auto result = OutputTerm::create(config);
    EXPECT_FALSE(result.has_value());
  }

  TEST(OutputTermTest, SimplePassthrough) {
    OutputTermConfig config {.name = "out", .kind = "out", .shape = {1, 3}};
    auto term_result = OutputTerm::create(config);
    ASSERT_TRUE(term_result.has_value());
    auto & term = *term_result;

    TensorSpec spec;
    NamedTensor output {.name = "out", .tensor = torch::zeros({1, 3})};

    auto activate_result = term.activate(spec, output);
    ASSERT_TRUE(activate_result.has_value());

    auto nn_output = torch::tensor({{1.0f, 2.0f, 3.0f}});
    auto advance_result = term.advance(nn_output, output);
    ASSERT_TRUE(advance_result.has_value());

    EXPECT_TRUE(torch::allclose(output.tensor, nn_output));
  }

  TEST(OutputTermTest, Reordering) {
    OutputTermConfig config {
      .name = "out",
      .kind = "out",
      .shape = {1, 3},
      .element_names = {{"batch"}, {"c", "a", "b"}},  // NN order
    };
    auto term_result = OutputTerm::create(config);
    ASSERT_TRUE(term_result.has_value());
    auto & term = *term_result;

    TensorSpec spec {.names = {{"batch"}, {"a", "b", "c"}}};  // Middleware order
    NamedTensor output {.name = "out", .tensor = torch::zeros({1, 3})};

    auto activate_result = term.activate(spec, output);
    ASSERT_TRUE(activate_result.has_value());

    // NN output: c=1, a=2, b=3
    auto nn_output = torch::tensor({{1.0f, 2.0f, 3.0f}});
    auto advance_result = term.advance(nn_output, output);
    ASSERT_TRUE(advance_result.has_value());

    // Expected middleware output: a=2, b=3, c=1
    auto expected = torch::tensor({{2.0f, 3.0f, 1.0f}});
    EXPECT_TRUE(torch::allclose(output.tensor, expected));
  }

}  // namespace isaac_deploy_core

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/input/input_term.h"

#include <gtest/gtest.h>

namespace isaac_deploy_core {

  TEST(InputTermTest, CreateSuccess) {
    InputTermConfig config {
      .name = "joint_pos",
      .kind = "joint_pos",
      .shape = {1, 3},
    };
    auto result = InputTerm::create(config);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->name(), "joint_pos");
    EXPECT_EQ(result->kind(), "joint_pos");
  }

  TEST(InputTermTest, CreateFailsWithEmptyName) {
    InputTermConfig config {.name = "", .kind = "k", .shape = {1, 3}};
    auto result = InputTerm::create(config);
    EXPECT_FALSE(result.has_value());
  }

  TEST(InputTermTest, EmptyKindDefaultsSourceToName) {
    InputTermConfig config {.name = "name", .kind = "", .shape = {1, 3}};
    auto result = InputTerm::create(config);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->source(), "name");
  }

  TEST(InputTermTest, SimplePassthrough) {
    InputTermConfig config {.name = "test", .kind = "test", .shape = {1, 3}};
    auto term_result = InputTerm::create(config);
    ASSERT_TRUE(term_result.has_value());
    auto & term = *term_result;

    TensorSpec spec;
    auto input = torch::tensor({{1.0f, 2.0f, 3.0f}});

    auto activate_result = term.activate(spec, input);
    ASSERT_TRUE(activate_result.has_value());

    torch::Tensor output;
    auto advance_result = term.advance(input, output);
    ASSERT_TRUE(advance_result.has_value());

    EXPECT_TRUE(torch::allclose(output, input));
  }

  TEST(InputTermTest, Reordering) {
    InputTermConfig config {
      .name = "test",
      .kind = "test",
      .shape = {1, 3},
      .element_names = {{"batch"}, {"c", "a", "b"}},  // Target order
    };
    auto term_result = InputTerm::create(config);
    ASSERT_TRUE(term_result.has_value());
    auto & term = *term_result;

    TensorSpec spec {.names = {{"batch"}, {"a", "b", "c"}}};  // Source order
    auto input = torch::tensor({{1.0f, 2.0f, 3.0f}});

    auto activate_result = term.activate(spec, input);
    ASSERT_TRUE(activate_result.has_value());

    torch::Tensor output;
    auto advance_result = term.advance(input, output);
    ASSERT_TRUE(advance_result.has_value());

    // Expected: c=3, a=1, b=2
    auto expected = torch::tensor({{3.0f, 1.0f, 2.0f}});
    EXPECT_TRUE(torch::allclose(output, expected));
  }

  TEST(InputTermTest, HistoryBuffer) {
    InputTermConfig config {
      .name = "test",
      .kind = "test",
      .shape = {1, 3, 2},
      .history_length = 3,
      .include_current_in_history = true,
    };
    auto term_result = InputTerm::create(config);
    ASSERT_TRUE(term_result.has_value());
    auto & term = *term_result;

    TensorSpec spec;
    auto input = torch::tensor({{1.0f, 2.0f}});

    auto activate_result = term.activate(spec, input);
    ASSERT_TRUE(activate_result.has_value());

    // First advance - history should be all 1s.
    torch::Tensor output;
    input = torch::tensor({{1.0f, 1.0f}});
    auto advance_result = term.advance(input, output);
    ASSERT_TRUE(advance_result.has_value());
    EXPECT_EQ(output.sizes(), std::vector < int64_t > ({1, 3, 2}));

    // Second advance with new value.
    input = torch::tensor({{2.0f, 2.0f}});
    advance_result = term.advance(input, output);
    ASSERT_TRUE(advance_result.has_value());
  }

  TEST(InputTermTest, OutputDependency) {
    InputTermConfig config {
      .name = "last_actions",
      .kind = "last_actions",
      .shape = {1, 3},
      .output_key = "actions",
    };
    auto term_result = InputTerm::create(config);
    ASSERT_TRUE(term_result.has_value());

    EXPECT_TRUE(term_result->depends_on_output());
    EXPECT_EQ(term_result->output_key(), "actions");
  }

}  // namespace isaac_deploy_core

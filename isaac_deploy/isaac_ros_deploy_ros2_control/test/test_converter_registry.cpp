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

#include <gtest/gtest.h>

#include "isaac_ros_deploy_ros2_control/converters/converter_registry.hpp"

namespace isaac_ros_deploy_ros2_control
{

class DummyConverter
{
public:
  virtual ~DummyConverter() = default;
  std::string label;
};

using DummyRegistry = ConverterRegistry<DummyConverter>;

TEST(ConverterRegistryTest, ContainsReturnsFalseForUnknown)
{
  auto & reg = DummyRegistry::instance();
  EXPECT_FALSE(reg.contains("nonexistent_kind_xyz"));
}

TEST(ConverterRegistryTest, RegisterAndCreate)
{
  auto & reg = DummyRegistry::instance();
  reg.register_converter("test_kind", []() {
    auto c = std::make_shared<DummyConverter>();
    c->label = "hello";
    return c;
  });

  EXPECT_TRUE(reg.contains("test_kind"));

  auto converter = reg.create_for_kind("test_kind");
  ASSERT_NE(converter, nullptr);
  EXPECT_EQ(converter->label, "hello");
}

TEST(ConverterRegistryTest, CreateReturnsNullptrForUnknown)
{
  auto & reg = DummyRegistry::instance();
  EXPECT_EQ(reg.create_for_kind("nonexistent_kind_xyz"), nullptr);
}

}  // namespace isaac_ros_deploy_ros2_control

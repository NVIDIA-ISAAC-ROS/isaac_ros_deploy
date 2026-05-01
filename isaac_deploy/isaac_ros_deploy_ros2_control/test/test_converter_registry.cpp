// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

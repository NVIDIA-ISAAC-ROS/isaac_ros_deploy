// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "rclcpp/serialization.hpp"

#include "isaac_ros_deploy_converters/converters/message_to_tensor_converter.hpp"

namespace isaac_ros_deploy_converters
{

class MessageToTensorConverterTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    initialize_input_converters();
  }
};

// --- Registry tests ---

TEST_F(MessageToTensorConverterTest, DefaultConverterForVelocityCommandIsTwist)
{
  auto & registry = MessageToTensorConverterRegistry::instance();
  const auto converter = registry.create_for_kind("command/body/velocity");
  ASSERT_NE(converter, nullptr);
  EXPECT_EQ(converter->get_message_type(), "geometry_msgs/msg/Twist");
}

TEST_F(MessageToTensorConverterTest, ExplicitTwistStampedSelection)
{
  auto & registry = MessageToTensorConverterRegistry::instance();
  const auto converter = registry.create_for_kind(
    "command/body/velocity", {}, "geometry_msgs/msg/TwistStamped");
  ASSERT_NE(converter, nullptr);
  EXPECT_EQ(converter->get_message_type(), "geometry_msgs/msg/TwistStamped");
}

TEST_F(MessageToTensorConverterTest, ExplicitTwistSelection)
{
  auto & registry = MessageToTensorConverterRegistry::instance();
  const auto converter = registry.create_for_kind(
    "command/body/velocity", {}, "geometry_msgs/msg/Twist");
  ASSERT_NE(converter, nullptr);
  EXPECT_EQ(converter->get_message_type(), "geometry_msgs/msg/Twist");
}

TEST_F(MessageToTensorConverterTest, UnknownMessageTypeReturnsNullptr)
{
  auto & registry = MessageToTensorConverterRegistry::instance();
  const auto converter = registry.create_for_kind(
    "command/body/velocity", {}, "geometry_msgs/msg/Nonexistent");
  EXPECT_EQ(converter, nullptr);
}

TEST_F(MessageToTensorConverterTest, UnknownKindReturnsNullptr)
{
  auto & registry = MessageToTensorConverterRegistry::instance();
  EXPECT_EQ(registry.create_for_kind("nonexistent_kind"), nullptr);
}

// --- Converter output tests ---

template<typename MsgT>
std::shared_ptr<rclcpp::SerializedMessage> serialize(const MsgT & msg)
{
  auto serialized = std::make_shared<rclcpp::SerializedMessage>();
  rclcpp::Serialization<MsgT> serializer;
  serializer.serialize_message(&msg, serialized.get());
  return serialized;
}

TEST_F(MessageToTensorConverterTest, TwistConverterOutput)
{
  auto & registry = MessageToTensorConverterRegistry::instance();
  const auto converter = registry.create_for_kind("command/body/velocity");

  geometry_msgs::msg::Twist twist;
  twist.linear.x = 1.0;
  twist.linear.y = 2.0;
  twist.linear.z = 3.0;
  twist.angular.x = 4.0;
  twist.angular.y = 5.0;
  twist.angular.z = 6.0;

  const auto tensor = converter->convert(serialize(twist));
  ASSERT_EQ(tensor.sizes(), (std::vector<int64_t>{1, 6}));

  const auto accessor = tensor.accessor<float, 2>();
  EXPECT_FLOAT_EQ(accessor[0][0], 1.0f);
  EXPECT_FLOAT_EQ(accessor[0][1], 2.0f);
  EXPECT_FLOAT_EQ(accessor[0][2], 3.0f);
  EXPECT_FLOAT_EQ(accessor[0][3], 4.0f);
  EXPECT_FLOAT_EQ(accessor[0][4], 5.0f);
  EXPECT_FLOAT_EQ(accessor[0][5], 6.0f);
}

TEST_F(MessageToTensorConverterTest, TwistStampedConverterOutput)
{
  auto & registry = MessageToTensorConverterRegistry::instance();
  const auto converter = registry.create_for_kind(
    "command/body/velocity", {}, "geometry_msgs/msg/TwistStamped");

  geometry_msgs::msg::TwistStamped twist_stamped;
  twist_stamped.twist.linear.x = 1.0;
  twist_stamped.twist.linear.y = 2.0;
  twist_stamped.twist.linear.z = 3.0;
  twist_stamped.twist.angular.x = 4.0;
  twist_stamped.twist.angular.y = 5.0;
  twist_stamped.twist.angular.z = 6.0;

  const auto tensor = converter->convert(serialize(twist_stamped));
  ASSERT_EQ(tensor.sizes(), (std::vector<int64_t>{1, 6}));

  const auto accessor = tensor.accessor<float, 2>();
  EXPECT_FLOAT_EQ(accessor[0][0], 1.0f);
  EXPECT_FLOAT_EQ(accessor[0][1], 2.0f);
  EXPECT_FLOAT_EQ(accessor[0][2], 3.0f);
  EXPECT_FLOAT_EQ(accessor[0][3], 4.0f);
  EXPECT_FLOAT_EQ(accessor[0][4], 5.0f);
  EXPECT_FLOAT_EQ(accessor[0][5], 6.0f);
}

TEST_F(MessageToTensorConverterTest, BothConvertersProduceSameTensorSpec)
{
  auto & registry = MessageToTensorConverterRegistry::instance();
  const auto twist_conv = registry.create_for_kind("command/body/velocity");
  const auto stamped_conv = registry.create_for_kind(
    "command/body/velocity", {}, "geometry_msgs/msg/TwistStamped");

  const auto twist_spec = twist_conv->get_tensor_spec();
  const auto stamped_spec = stamped_conv->get_tensor_spec();
  EXPECT_EQ(twist_spec.names, stamped_spec.names);
}

}  // namespace isaac_ros_deploy_converters

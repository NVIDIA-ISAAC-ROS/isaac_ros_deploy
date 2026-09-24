// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cuda_runtime_api.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "isaac_ros_deploy_converters/utils/tensor_list_utils.hpp"

namespace isaac_ros_deploy_converters
{

TEST(TensorListUtilsTest, TorchToTensorMessagePopulatesDLPackMetadata)
{
  const auto tensor = torch::tensor(
    {{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}}, torch::kFloat32);

  const auto msg = torch_to_tensor_msg(tensor);

  EXPECT_EQ(msg.shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(msg.strides, (std::vector<int64_t>{3, 1}));
  EXPECT_EQ(msg.dtype_code, 2);
  EXPECT_EQ(msg.dtype_bits, 32);
  EXPECT_EQ(msg.dtype_lanes, 1);
  EXPECT_EQ(msg.byte_offset, 0U);
  ASSERT_EQ(msg.data.size(), tensor.numel() * tensor.element_size());
  EXPECT_EQ(std::memcmp(msg.data.data(), tensor.data_ptr(), msg.data.size()), 0);
}

TEST(TensorListUtilsTest, DictToTensorListUsesParallelNamesAndTensors)
{
  isaac_deploy_core::TensorDict dict;
  dict["policy"] = torch::tensor({1, 2, 3}, torch::kInt32);

  const auto msg = dict_to_tensor_list(dict, rclcpp::Time(123), "pelvis");

  EXPECT_EQ(msg.header.frame_id, "pelvis");
  ASSERT_EQ(msg.names.size(), 1U);
  ASSERT_EQ(msg.tensors.size(), 1U);
  EXPECT_EQ(msg.names[0], "policy");
  EXPECT_EQ(msg.tensors[0].shape, (std::vector<int64_t>{3}));
}

TEST(TensorListUtilsTest, TensorListToDictRejectsMismatchedNamesAndTensors)
{
  isaac_ros_tensor_msgs::msg::TensorList msg;
  msg.names.push_back("missing_tensor");

  EXPECT_THROW(tensor_list_to_dict(msg), std::runtime_error);
}

TEST(TensorListUtilsTest, TensorListToNamedTensorsRejectsMismatchedNamesAndTensors)
{
  isaac_ros_tensor_msgs::msg::TensorList msg;
  msg.tensors.emplace_back();

  EXPECT_THROW(tensor_list_to_named_tensors(msg), std::runtime_error);
}

TEST(TensorListUtilsTest, TensorMessageRoundTripUsesCudaBufferBackend)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  const auto expected = torch::tensor(
    {{1.25F, -2.5F}, {3.75F, 4.0F}}, torch::kFloat32);
  const auto actual = tensor_msg_to_torch(torch_to_tensor_msg(expected));

  EXPECT_TRUE(torch::equal(actual, expected));
}

}  // namespace isaac_ros_deploy_converters

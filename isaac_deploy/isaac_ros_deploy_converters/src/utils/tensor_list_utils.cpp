// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_deploy_converters/utils/tensor_list_utils.hpp"

#include <cuda_runtime_api.h>

#include <cstring>
#include <limits>
#include <stdexcept>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_tensor_msgs/tensor_utils.hpp"

namespace isaac_ros_deploy_converters
{

namespace
{

torch::ScalarType dlpack_to_scalar_type(uint8_t code, uint8_t bits, uint16_t lanes)
{
  if (lanes != 1) {
    throw std::runtime_error("Vector-lane tensor dtypes are not supported");
  }
  if (code == 0 && bits == 8) {return torch::kInt8;}
  if (code == 0 && bits == 16) {return torch::kInt16;}
  if (code == 0 && bits == 32) {return torch::kInt32;}
  if (code == 0 && bits == 64) {return torch::kInt64;}
  if (code == 1 && bits == 8) {return torch::kUInt8;}
  if (code == 2 && bits == 32) {return torch::kFloat32;}
  if (code == 2 && bits == 64) {return torch::kFloat64;}
  throw std::runtime_error(
          "Unsupported DLPack dtype: code=" + std::to_string(code) +
          ", bits=" + std::to_string(bits));
}

void set_dlpack_dtype(
  tensor_msgs::msg::ExperimentalTensor & msg, torch::ScalarType scalar_type)
{
  msg.dtype_lanes = 1;
  switch (scalar_type) {
    case torch::kInt8: msg.dtype_code = 0; msg.dtype_bits = 8; return;
    case torch::kInt16: msg.dtype_code = 0; msg.dtype_bits = 16; return;
    case torch::kInt32: msg.dtype_code = 0; msg.dtype_bits = 32; return;
    case torch::kInt64: msg.dtype_code = 0; msg.dtype_bits = 64; return;
    case torch::kUInt8: msg.dtype_code = 1; msg.dtype_bits = 8; return;
    case torch::kFloat32: msg.dtype_code = 2; msg.dtype_bits = 32; return;
    case torch::kFloat64: msg.dtype_code = 2; msg.dtype_bits = 64; return;
    default:
      throw std::runtime_error("Unsupported scalar type for tensor message conversion");
  }
}

void check_cuda(cudaError_t result, const char * operation)
{
  if (result != cudaSuccess) {
    throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

}  // namespace

torch::Tensor tensor_msg_to_torch(const tensor_msgs::msg::ExperimentalTensor & msg)
{
  if (!isaac_ros_tensor_msgs::IsContiguousRowMajor(msg)) {
    throw std::runtime_error("Only contiguous row-major tensors are supported");
  }
  const auto dtype = dlpack_to_scalar_type(msg.dtype_code, msg.dtype_bits, msg.dtype_lanes);

  // Validate the metadata against the backing buffer before allocating, so a bogus
  // shape is rejected instead of first reserving memory for it. The element count
  // matches torch::empty(msg.shape): an empty shape is a scalar with one element.
  // dlpack_to_scalar_type above has already rejected lanes != 1.
  size_t element_count = 1;
  for (const auto dim : msg.shape) {
    if (dim < 0) {
      throw std::runtime_error("Tensor dimensions must be non-negative");
    }
    const size_t extent = static_cast<size_t>(dim);
    if (extent != 0 && element_count > std::numeric_limits<size_t>::max() / extent) {
      throw std::runtime_error("Tensor storage size overflow");
    }
    element_count *= extent;
  }
  const size_t element_size = msg.dtype_bits / 8;
  if (element_size != 0 && element_count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::runtime_error("Tensor storage size overflow");
  }
  const size_t byte_count = element_count * element_size;
  if (msg.byte_offset > msg.data.size() || byte_count > msg.data.size() - msg.byte_offset) {
    throw std::runtime_error("Tensor metadata exceeds its backing buffer");
  }

  auto options = torch::TensorOptions().dtype(dtype);
  torch::Tensor tensor = torch::empty(msg.shape, options);
  if (byte_count != 0) {
    auto read_handle = cuda_buffer_backend::from_input_buffer(msg.data, cudaStreamPerThread);
    check_cuda(
      cudaMemcpyAsync(
        tensor.data_ptr(), read_handle.get_ptr() + msg.byte_offset, byte_count,
        cudaMemcpyDeviceToHost, cudaStreamPerThread),
      "cudaMemcpyAsync");
    check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "cudaStreamSynchronize");
  }
  return tensor;
}

tensor_msgs::msg::ExperimentalTensor torch_to_tensor_msg(const torch::Tensor & tensor)
{
  tensor_msgs::msg::ExperimentalTensor msg;
  torch::Tensor contiguous = tensor.contiguous().cpu();
  msg.shape.assign(contiguous.sizes().begin(), contiguous.sizes().end());
  for (int64_t i = 0; i < contiguous.dim(); ++i) {
    msg.strides.push_back(contiguous.stride(i));
  }
  set_dlpack_dtype(msg, contiguous.scalar_type());
  msg.byte_offset = 0;
  size_t data_size = contiguous.numel() * contiguous.element_size();
  msg.data.resize(data_size);
  std::memcpy(msg.data.data(), contiguous.data_ptr(), data_size);
  return msg;
}

isaac_deploy_core::TensorDict tensor_list_to_dict(
  const isaac_ros_tensor_msgs::msg::TensorList & msg)
{
  isaac_deploy_core::TensorDict dict;
  if (msg.names.size() != msg.tensors.size()) {
    throw std::runtime_error("TensorList names and tensors must have the same length");
  }
  for (size_t i = 0; i < msg.names.size(); ++i) {
    dict[msg.names[i]] = tensor_msg_to_torch(msg.tensors[i]);
  }
  return dict;
}

isaac_ros_tensor_msgs::msg::TensorList dict_to_tensor_list(
  const isaac_deploy_core::TensorDict & dict,
  const rclcpp::Time & stamp,
  const std::string & frame_id)
{
  isaac_ros_tensor_msgs::msg::TensorList msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;

  for (const auto & [name, tensor] : dict) {
    msg.names.push_back(name);
    msg.tensors.push_back(torch_to_tensor_msg(tensor));
  }

  return msg;
}

isaac_ros_tensor_msgs::msg::TensorList named_tensors_to_tensor_list(
  const std::unordered_map<std::string, isaac_deploy_core::NamedTensor> & named_tensors,
  const rclcpp::Time & stamp,
  const std::string & frame_id)
{
  isaac_ros_tensor_msgs::msg::TensorList msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;

  for (const auto & [name, named_tensor] : named_tensors) {
    msg.names.push_back(named_tensor.name);
    msg.tensors.push_back(torch_to_tensor_msg(named_tensor.tensor));
  }

  return msg;
}

std::unordered_map<std::string, isaac_deploy_core::NamedTensor> tensor_list_to_named_tensors(
  const isaac_ros_tensor_msgs::msg::TensorList & msg)
{
  std::unordered_map<std::string, isaac_deploy_core::NamedTensor> named_tensors;

  // Convert header timestamp to nanoseconds.
  int64_t timestamp_ns = rclcpp::Time(msg.header.stamp).nanoseconds();

  if (msg.names.size() != msg.tensors.size()) {
    throw std::runtime_error("TensorList names and tensors must have the same length");
  }
  for (size_t i = 0; i < msg.names.size(); ++i) {
    isaac_deploy_core::NamedTensor named_tensor;
    named_tensor.name = msg.names[i];
    named_tensor.timestamp_ns = timestamp_ns;
    named_tensor.tensor = tensor_msg_to_torch(msg.tensors[i]);
    named_tensors[msg.names[i]] = std::move(named_tensor);
  }

  return named_tensors;
}

isaac_ros_tensor_msgs::msg::TensorList tensor_to_tensor_list(
  const std::string & name,
  const torch::Tensor & tensor,
  const rclcpp::Time & stamp)
{
  isaac_ros_tensor_msgs::msg::TensorList msg;
  msg.header.stamp = stamp;
  msg.names.push_back(name);
  msg.tensors.push_back(torch_to_tensor_msg(tensor));
  return msg;
}

torch::Tensor tensor_list_to_tensor(
  const isaac_ros_tensor_msgs::msg::TensorList & msg)
{
  if (msg.tensors.empty()) {
    return torch::empty({});
  }
  return tensor_msg_to_torch(msg.tensors[0]);
}

}  // namespace isaac_ros_deploy_converters

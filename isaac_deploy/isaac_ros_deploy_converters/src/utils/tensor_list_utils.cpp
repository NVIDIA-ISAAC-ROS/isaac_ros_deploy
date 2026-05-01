// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_converters/utils/tensor_list_utils.hpp"

#include <cstring>
#include <stdexcept>

namespace isaac_ros_deploy_converters
{

namespace
{

/// Data types matching isaac_ros_tensor_list_interfaces/Tensor.msg.
/// Using an enum class so the compiler warns about unhandled cases in switch statements.
/// Note: float16 is not supported by this conversion layer. If a model outputs float16,
/// the conversion will throw at runtime.
enum class DataType : int32_t
{
  kInt8 = 1,
  kUInt8 = 2,
  kInt16 = 3,
  kUInt16 = 4,
  kInt32 = 5,
  kUInt32 = 6,
  kInt64 = 7,
  kUInt64 = 8,
  kFloat32 = 9,
  kFloat64 = 10,
};

torch::ScalarType data_type_to_scalar_type(int32_t data_type)
{
  switch (static_cast<DataType>(data_type)) {
    case DataType::kInt8:
      return torch::kInt8;
    case DataType::kUInt8:
      return torch::kUInt8;
    case DataType::kInt16:
      return torch::kInt16;
    case DataType::kUInt16:
      throw std::runtime_error("UInt16 tensor data type is not supported by torch");
    case DataType::kInt32:
      return torch::kInt32;
    case DataType::kUInt32:
      throw std::runtime_error("UInt32 tensor data type is not supported by torch");
    case DataType::kInt64:
      return torch::kInt64;
    case DataType::kUInt64:
      throw std::runtime_error("UInt64 tensor data type is not supported by torch");
    case DataType::kFloat32:
      return torch::kFloat32;
    case DataType::kFloat64:
      return torch::kFloat64;
  }
  throw std::runtime_error("Unknown data type: " + std::to_string(data_type));
}

int32_t scalar_type_to_data_type(torch::ScalarType scalar_type)
{
  switch (scalar_type) {
    case torch::kInt8:
      return static_cast<int32_t>(DataType::kInt8);
    case torch::kUInt8:
      return static_cast<int32_t>(DataType::kUInt8);
    case torch::kInt16:
      return static_cast<int32_t>(DataType::kInt16);
    case torch::kInt32:
      return static_cast<int32_t>(DataType::kInt32);
    case torch::kInt64:
      return static_cast<int32_t>(DataType::kInt64);
    case torch::kFloat32:
      return static_cast<int32_t>(DataType::kFloat32);
    case torch::kFloat64:
      return static_cast<int32_t>(DataType::kFloat64);
    default:
      throw std::runtime_error("Unsupported scalar type for tensor message conversion");
  }
}

}  // namespace

torch::Tensor tensor_msg_to_torch(const isaac_ros_tensor_list_interfaces::msg::Tensor & msg)
{
  // Get tensor shape.
  std::vector<int64_t> shape;
  shape.reserve(msg.shape.dims.size());
  for (const auto & dim : msg.shape.dims) {
    shape.push_back(static_cast<int64_t>(dim));
  }

  // Get scalar type.
  torch::ScalarType dtype = data_type_to_scalar_type(msg.data_type);

  // Create tensor from data.
  auto options = torch::TensorOptions().dtype(dtype);
  torch::Tensor tensor = torch::empty(shape, options);

  // Copy data.
  if (!msg.data.empty()) {
    std::memcpy(tensor.data_ptr(), msg.data.data(), msg.data.size());
  }

  return tensor;
}

isaac_ros_tensor_list_interfaces::msg::Tensor torch_to_tensor_msg(
  const torch::Tensor & tensor,
  const std::string & name)
{
  isaac_ros_tensor_list_interfaces::msg::Tensor msg;
  msg.name = name;

  // Ensure tensor is contiguous and on CPU.
  torch::Tensor contiguous = tensor.contiguous().cpu();

  // Set shape (rank + dims).
  msg.shape.rank = static_cast<uint8_t>(contiguous.dim());
  for (int64_t i = 0; i < contiguous.dim(); ++i) {
    msg.shape.dims.push_back(static_cast<uint32_t>(contiguous.size(i)));
  }

  // Set data type.
  msg.data_type = scalar_type_to_data_type(contiguous.scalar_type());

  // Set strides (in bytes).
  for (int64_t i = 0; i < contiguous.dim(); ++i) {
    msg.strides.push_back(
      static_cast<uint64_t>(contiguous.stride(i)) * contiguous.element_size());
  }

  // Copy data.
  size_t data_size = contiguous.numel() * contiguous.element_size();
  msg.data.resize(data_size);
  std::memcpy(msg.data.data(), contiguous.data_ptr(), data_size);

  return msg;
}

isaac_deploy_core::TensorDict tensor_list_to_dict(
  const isaac_ros_tensor_list_interfaces::msg::TensorList & msg)
{
  isaac_deploy_core::TensorDict dict;
  for (const auto & tensor_msg : msg.tensors) {
    dict[tensor_msg.name] = tensor_msg_to_torch(tensor_msg);
  }
  return dict;
}

isaac_ros_tensor_list_interfaces::msg::TensorList dict_to_tensor_list(
  const isaac_deploy_core::TensorDict & dict,
  const rclcpp::Time & stamp,
  const std::string & frame_id)
{
  isaac_ros_tensor_list_interfaces::msg::TensorList msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;

  for (const auto & [name, tensor] : dict) {
    msg.tensors.push_back(torch_to_tensor_msg(tensor, name));
  }

  return msg;
}

isaac_ros_tensor_list_interfaces::msg::TensorList named_tensors_to_tensor_list(
  const std::unordered_map<std::string, isaac_deploy_core::NamedTensor> & named_tensors,
  const rclcpp::Time & stamp,
  const std::string & frame_id)
{
  isaac_ros_tensor_list_interfaces::msg::TensorList msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;

  for (const auto & [name, named_tensor] : named_tensors) {
    msg.tensors.push_back(torch_to_tensor_msg(named_tensor.tensor, named_tensor.name));
  }

  return msg;
}

std::unordered_map<std::string, isaac_deploy_core::NamedTensor> tensor_list_to_named_tensors(
  const isaac_ros_tensor_list_interfaces::msg::TensorList & msg)
{
  std::unordered_map<std::string, isaac_deploy_core::NamedTensor> named_tensors;

  // Convert header timestamp to nanoseconds.
  int64_t timestamp_ns = rclcpp::Time(msg.header.stamp).nanoseconds();

  for (const auto & tensor_msg : msg.tensors) {
    isaac_deploy_core::NamedTensor named_tensor;
    named_tensor.name = tensor_msg.name;
    named_tensor.timestamp_ns = timestamp_ns;
    named_tensor.tensor = tensor_msg_to_torch(tensor_msg);
    named_tensors[tensor_msg.name] = std::move(named_tensor);
  }

  return named_tensors;
}

isaac_ros_tensor_list_interfaces::msg::TensorList tensor_to_tensor_list(
  const std::string & name,
  const torch::Tensor & tensor,
  const rclcpp::Time & stamp)
{
  isaac_ros_tensor_list_interfaces::msg::TensorList msg;
  msg.header.stamp = stamp;
  msg.tensors.push_back(torch_to_tensor_msg(tensor, name));
  return msg;
}

torch::Tensor tensor_list_to_tensor(
  const isaac_ros_tensor_list_interfaces::msg::TensorList & msg)
{
  if (msg.tensors.empty()) {
    return torch::empty({});
  }
  return tensor_msg_to_torch(msg.tensors[0]);
}

}  // namespace isaac_ros_deploy_converters

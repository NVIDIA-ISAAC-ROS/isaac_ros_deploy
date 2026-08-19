// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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

#include "isaac_deploy_core/inference_controller/safetensors_loader.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace isaac_deploy_core
{

namespace
{

std::string strip_model_prefix(const std::string & s)
{
  auto pos = s.find('/');
  if (pos == std::string::npos) {
    return s;
  }
  return s.substr(pos + 1);
}

expected<std::vector<char>> read_binary_file(const std::filesystem::path & path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return tl::unexpected(
        make_error(
          Error::Code::kNotFound,
          "Failed to open safetensors file: " + path.string()));
  }

  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0) {
    return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "Failed to determine safetensors file size: " + path.string()));
  }
  input.seekg(0, std::ios::beg);

  std::vector<char> bytes(static_cast<size_t>(size));
  input.read(bytes.data(), size);
  if (!input) {
    return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "Failed to read safetensors file: " + path.string()));
  }
  return bytes;
}

uint64_t read_u64_le(const char * data)
{
  uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(static_cast<unsigned char>(*data++)) << shift;
  }
  return value;
}

expected<std::pair<torch::ScalarType, size_t>> dtype_info(const std::string & dtype)
{
  if (dtype == "F64") {return std::pair{torch::kFloat64, size_t{8}};}
  if (dtype == "F32") {return std::pair{torch::kFloat32, size_t{4}};}
  if (dtype == "F16") {return std::pair{torch::kFloat16, size_t{2}};}
  if (dtype == "BF16") {return std::pair{torch::kBFloat16, size_t{2}};}
  if (dtype == "I64") {return std::pair{torch::kInt64, size_t{8}};}
  if (dtype == "I32") {return std::pair{torch::kInt32, size_t{4}};}
  if (dtype == "I16") {return std::pair{torch::kInt16, size_t{2}};}
  if (dtype == "I8") {return std::pair{torch::kInt8, size_t{1}};}
  if (dtype == "U8") {return std::pair{torch::kUInt8, size_t{1}};}
  if (dtype == "BOOL") {return std::pair{torch::kBool, size_t{1}};}

  return tl::unexpected(
      make_error(
        Error::Code::kInvalidArgument,
        "Unsupported safetensors dtype: " + dtype));
}

expected<torch::Tensor> load_tensor(
  const std::string & name,
  const nlohmann::json & metadata,
  const char * data_begin,
  size_t data_size)
{
  if (!metadata.contains("dtype") ||
    !metadata.contains("shape") ||
    !metadata.contains("data_offsets"))
  {
    return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "Invalid safetensors metadata for tensor: " + name));
  }

  const auto dtype_result = dtype_info(metadata.at("dtype").get<std::string>());
  if (!dtype_result) {
    return tl::unexpected(dtype_result.error());
  }
  const auto [scalar_type, element_size] = *dtype_result;

  const auto shape = metadata.at("shape").get<std::vector<int64_t>>();
  const auto offsets = metadata.at("data_offsets").get<std::vector<size_t>>();
  if (offsets.size() != 2 || offsets[1] < offsets[0] || offsets[1] > data_size) {
    return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "Invalid safetensors data offsets for tensor: " + name));
  }

  size_t element_count = 1;
  for (const auto dim : shape) {
    if (dim < 0) {
      return tl::unexpected(
          make_error(
            Error::Code::kInvalidArgument,
            "Negative safetensors shape dimension for tensor: " + name));
    }
    element_count *= static_cast<size_t>(dim);
  }
  const auto expected_size = element_count * element_size;
  if (offsets[1] - offsets[0] != expected_size) {
    return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "Safetensors byte size does not match shape for tensor: " + name));
  }

  auto options = torch::TensorOptions().dtype(scalar_type).device(torch::kCPU);
  return torch::from_blob(
    const_cast<char *>(data_begin + offsets[0]), shape, options).clone();
}

}  // namespace

expected<TensorDict> load_safetensors(const std::filesystem::path & path)
{
  auto bytes_result = read_binary_file(path);
  if (!bytes_result) {
    return tl::unexpected(bytes_result.error());
  }
  const auto & bytes = *bytes_result;
  if (bytes.size() < 8) {
    return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "Invalid safetensors file, missing header length: " + path.string()));
  }

  const auto header_size = read_u64_le(bytes.data());
  if (header_size > bytes.size() - 8) {
    return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "Invalid safetensors header length: " + path.string()));
  }

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(bytes.data() + 8, bytes.data() + 8 + header_size);
  } catch (const std::exception & e) {
    return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "Failed to parse safetensors header: " + std::string(e.what())));
  }

  TensorDict tensors;
  const char * data_begin = bytes.data() + 8 + header_size;
  const size_t data_size = bytes.size() - 8 - header_size;
  for (const auto & [name, metadata] : header.items()) {
    if (name == "__metadata__") {
      continue;
    }
    auto tensor_result = load_tensor(name, metadata, data_begin, data_size);
    if (!tensor_result) {
      return tl::unexpected(tensor_result.error());
    }
    tensors[name] = std::move(*tensor_result);
  }

  return tensors;
}

expected<TensorDict> load_feedback_initial_values(
  const GraphConfig & graph,
  const std::filesystem::path & config_path)
{
  TensorDict result;
  if (graph.initial_values_path.empty()) {
    return result;
  }

  std::filesystem::path initial_values_path = graph.initial_values_path;
  if (!initial_values_path.is_absolute()) {
    initial_values_path = config_path.parent_path() / initial_values_path;
  }

  auto tensors_result = load_safetensors(initial_values_path);
  if (!tensors_result) {
    return tl::unexpected(tensors_result.error());
  }

  const auto & tensors = *tensors_result;
  for (const auto & [source_key, target_keys] : graph.feedback_flow) {
    const auto source_name = strip_model_prefix(source_key);
    for (const auto & target_key : target_keys) {
      const auto tensor_it = tensors.find(target_key);
      if (tensor_it != tensors.end()) {
        result[source_name] = tensor_it->second;
      }
    }
  }

  return result;
}

}  // namespace isaac_deploy_core

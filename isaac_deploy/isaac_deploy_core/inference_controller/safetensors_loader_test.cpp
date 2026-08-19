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

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace isaac_deploy_core
{

namespace
{

void write_u64_le(std::ofstream & out, uint64_t value)
{
  for (int shift = 0; shift < 64; shift += 8) {
    const auto byte = static_cast<char>((value >> shift) & 0xff);
    out.write(&byte, 1);
  }
}

std::filesystem::path test_path(const std::string & name)
{
  return std::filesystem::temp_directory_path() /
         ("isaac_deploy_safetensors_" + name + ".safetensors");
}

void write_bytes(const std::filesystem::path & path, const std::string & bytes)
{
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_safetensors(
  const std::filesystem::path & path,
  const std::string & header,
  const std::string & payload = "")
{
  std::ofstream out(path, std::ios::binary);
  write_u64_le(out, header.size());
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

std::string payload_from_floats(const std::vector<float> & values)
{
  return std::string(
    reinterpret_cast<const char *>(values.data()),
    values.size() * sizeof(float));
}

}  // namespace

TEST(SafetensorsLoaderTest, LoadsFloat32Tensor) {
  const auto path = test_path("loads_float32_tensor");
  const std::string header =
    R"({"policy/feedback":{"dtype":"F32","shape":[1,3],"data_offsets":[0,12]}})";
  write_safetensors(path, header, payload_from_floats({1.0f, 2.0f, 3.0f}));

  auto result = load_safetensors(path);
  ASSERT_TRUE(result.has_value()) << result.error().message;
  ASSERT_EQ(result->size(), 1);
  ASSERT_TRUE(result->contains("policy/feedback"));
  EXPECT_TRUE(torch::allclose(
      result->at("policy/feedback"), torch::tensor({{1.0f, 2.0f, 3.0f}})));

  std::filesystem::remove(path);
}

TEST(SafetensorsLoaderTest, RejectsMissingHeaderLength) {
  const auto path = test_path("missing_header_length");
  write_bytes(path, "short");

  auto result = load_safetensors(path);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message.find("missing header length"), std::string::npos)
    << result.error().message;

  std::filesystem::remove(path);
}

TEST(SafetensorsLoaderTest, RejectsTruncatedHeader) {
  const auto path = test_path("truncated_header");
  std::ofstream out(path, std::ios::binary);
  write_u64_le(out, 64);
  out.write("{}", 2);
  out.close();

  auto result = load_safetensors(path);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message.find("header length"), std::string::npos)
    << result.error().message;

  std::filesystem::remove(path);
}

TEST(SafetensorsLoaderTest, RejectsMalformedJsonHeader) {
  const auto path = test_path("malformed_json_header");
  write_safetensors(path, "{not json");

  auto result = load_safetensors(path);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message.find("Failed to parse safetensors header"), std::string::npos)
    << result.error().message;

  std::filesystem::remove(path);
}

TEST(SafetensorsLoaderTest, RejectsUnsupportedDtype) {
  const auto path = test_path("unsupported_dtype");
  const std::string header =
    R"({"policy/feedback":{"dtype":"F8","shape":[1,1],"data_offsets":[0,1]}})";
  write_safetensors(path, header, std::string(1, '\0'));

  auto result = load_safetensors(path);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message.find("Unsupported safetensors dtype: F8"), std::string::npos)
    << result.error().message;

  std::filesystem::remove(path);
}

TEST(SafetensorsLoaderTest, RejectsShapeByteSizeMismatch) {
  const auto path = test_path("shape_byte_size_mismatch");
  const std::string header =
    R"({"policy/feedback":{"dtype":"F32","shape":[1,3],"data_offsets":[0,8]}})";
  write_safetensors(path, header, std::string(8, '\0'));

  auto result = load_safetensors(path);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message.find("byte size does not match shape"), std::string::npos)
    << result.error().message;

  std::filesystem::remove(path);
}

TEST(SafetensorsLoaderTest, SkipsMetadataEntry) {
  const auto path = test_path("skips_metadata");
  const std::string header =
    R"({"__metadata__":{"format":"pt"},"policy/feedback":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}})";
  write_safetensors(path, header, payload_from_floats({7.0f}));

  auto result = load_safetensors(path);
  ASSERT_TRUE(result.has_value()) << result.error().message;
  ASSERT_EQ(result->size(), 1);
  EXPECT_FALSE(result->contains("__metadata__"));
  ASSERT_TRUE(result->contains("policy/feedback"));
  EXPECT_TRUE(torch::allclose(result->at("policy/feedback"), torch::tensor({{7.0f}})));

  std::filesystem::remove(path);
}

}  // namespace isaac_deploy_core

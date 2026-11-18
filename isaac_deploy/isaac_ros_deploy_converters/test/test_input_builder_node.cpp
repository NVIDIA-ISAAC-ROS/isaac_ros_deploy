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

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_deploy_converters/nodes/input_builder_node.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"

namespace isaac_ros_deploy_converters
{
namespace
{

using namespace std::chrono_literals;

class TemporaryConfig
{
public:
  explicit TemporaryConfig(const std::optional<double> frequency_hz)
  {
    const auto path_template =
      (std::filesystem::temp_directory_path() / "input_builder_node_XXXXXX").string();
    std::vector<char> writable_path(path_template.begin(), path_template.end());
    writable_path.push_back('\0');
    const int fd = mkstemp(writable_path.data());
    if (fd == -1) {
      throw std::runtime_error("Failed to create temporary InputBuilder config");
    }
    close(fd);
    path_ = writable_path.data();

    std::ofstream config(path_);
    config <<
      R"(models:
  policy:
    inputs: []
    outputs: []
    parameters:
      backend: onnx
)";
    if (frequency_hz.has_value()) {
      config << "pipeline:\n"
             << "  configs:\n"
             << "    frequency: " << *frequency_hz << "\n";
    }
  }

  ~TemporaryConfig()
  {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  std::string path() const {return path_.string();}

private:
  std::filesystem::path path_;
};

class InputBuilderNodeTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }

  static void TearDownTestSuite()
  {
    rclcpp::shutdown();
  }

  void expect_publish_rate(
    const std::optional<double> graph_frequency_hz,
    const std::optional<double> publish_rate_hz,
    const double expected_rate_hz,
    const std::size_t sample_count)
  {
    TemporaryConfig config(graph_frequency_hz);
    const std::string suffix =
      std::to_string(getpid()) + "_" + std::to_string(test_counter_++);
    const std::string output_topic = "input_tensors_" + suffix;

    std::vector<rclcpp::Parameter> parameters{
      rclcpp::Parameter("config_path", config.path()),
      rclcpp::Parameter("output_topic", output_topic),
    };
    if (publish_rate_hz.has_value()) {
      parameters.emplace_back("publish_rate", *publish_rate_hz);
    }

    const auto options = rclcpp::NodeOptions().parameter_overrides(parameters);
    auto input_builder = std::make_shared<InputBuilderNode>(options);
    auto io_node = std::make_shared<rclcpp::Node>("input_builder_node_test_" + suffix);

    std::vector<std::chrono::steady_clock::time_point> receive_times;
    auto subscription = io_node->create_subscription<isaac_ros_tensor_msgs::msg::TensorList>(
      output_topic, 10,
      [&receive_times](const isaac_ros_tensor_msgs::msg::TensorList::SharedPtr) {
        receive_times.push_back(std::chrono::steady_clock::now());
      });

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(input_builder);
    executor.add_node(io_node);

    const auto discovery_deadline = std::chrono::steady_clock::now() + 5s;
    while (subscription->get_publisher_count() == 0 &&
      std::chrono::steady_clock::now() < discovery_deadline)
    {
      executor.spin_some();
      std::this_thread::sleep_for(1ms);
    }
    ASSERT_GT(subscription->get_publisher_count(), 0U);

    const auto receive_deadline = std::chrono::steady_clock::now() + 5s;
    while (receive_times.size() < sample_count &&
      std::chrono::steady_clock::now() < receive_deadline)
    {
      executor.spin_some();
      std::this_thread::sleep_for(1ms);
    }
    ASSERT_GE(receive_times.size(), sample_count);

    const double observed_span_seconds =
      std::chrono::duration<double>(
      receive_times[sample_count - 1] - receive_times.front()).count();
    const double expected_span_seconds =
      static_cast<double>(sample_count - 1) / expected_rate_hz;
    const double tolerance_seconds = expected_span_seconds * 0.5 + 0.02;
    EXPECT_NEAR(observed_span_seconds, expected_span_seconds, tolerance_seconds);
  }

  void expect_configuration_error(
    const double graph_frequency_hz, const std::string & expected_message)
  {
    TemporaryConfig config(graph_frequency_hz);
    const auto options = rclcpp::NodeOptions().parameter_overrides({
          rclcpp::Parameter("config_path", config.path()),
    });
    try {
      std::make_shared<InputBuilderNode>(options);
      FAIL() << "Expected InputBuilderNode configuration to fail";
    } catch (const std::runtime_error & error) {
      EXPECT_NE(std::string(error.what()).find(expected_message), std::string::npos)
        << error.what();
    }
  }

  inline static std::atomic_uint test_counter_{0};
};

TEST_F(InputBuilderNodeTest, UsesGraphFrequencyWhenPublishRateIsZero)
{
  expect_publish_rate(10.0, 0.0, 10.0, 5);
}

TEST_F(InputBuilderNodeTest, FallsBackTo50HzWhenGraphFrequencyIsMissing)
{
  expect_publish_rate(std::nullopt, std::nullopt, 50.0, 8);
}

TEST_F(InputBuilderNodeTest, ExplicitPublishRateOverridesGraphFrequency)
{
  expect_publish_rate(5.0, 25.0, 25.0, 6);
}

TEST_F(InputBuilderNodeTest, RejectsFrequencyOutsideTimerRange)
{
  expect_configuration_error(
    1e-12, "pipeline.configs.frequency is outside the supported timer range");
}

}  // namespace
}  // namespace isaac_ros_deploy_converters

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

#include "isaac_ros_deploy_converters/nodes/output_builder_node.hpp"
#include "isaac_ros_deploy_converters/utils/tensor_list_utils.hpp"
#include "isaac_ros_deploy_interfaces/msg/joint_command_trajectory.hpp"

namespace isaac_ros_deploy_converters
{
namespace
{

using namespace std::chrono_literals;
using isaac_ros_deploy_interfaces::msg::JointCommandTrajectory;

class TemporaryConfig
{
public:
  explicit TemporaryConfig(
    const std::optional<double> temporal_period_ms,
    const double policy_dt_seconds = 0.02,
    const std::string & shape = "[1, 2, 2]",
    const std::optional<std::string> & element_names = std::nullopt)
  {
    const auto path_template =
      (std::filesystem::temp_directory_path() / "output_builder_node_XXXXXX").string();
    std::vector<char> writable_path(path_template.begin(), path_template.end());
    writable_path.push_back('\0');
    const int fd = mkstemp(writable_path.data());
    if (fd == -1) {
      throw std::runtime_error("Failed to create temporary OutputBuilder config");
    }
    close(fd);
    path_ = writable_path.data();

    std::ofstream config(path_);
    config <<
      "models:\n"
      "  policy:\n"
      "    inputs: []\n"
      "    outputs:\n"
      "    - name: actions\n"
      "      kind: actions\n"
      "      shape: " << shape << "\n"
      "      type: tensor\n";
    if (element_names.has_value()) {
      config << "      element_names: " << *element_names << "\n";
    } else if (temporal_period_ms.has_value()) {
      config << "      element_names: [__temporal_axis__, [left, right]]\n";
    } else {
      config << "      element_names: [[], [], [left, right]]\n";
    }
    if (temporal_period_ms.has_value()) {
      config << "      temporal_period_ms: " << *temporal_period_ms << "\n";
    }
    config << R"(    parameters:
      backend: onnx
semantic:
  scene:
    dt: )" << policy_dt_seconds << "\n";
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

class OutputBuilderNodeTest : public ::testing::Test
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

  void expect_step_dt(
    const std::optional<double> temporal_period_ms, const double expected_seconds)
  {
    TemporaryConfig config(temporal_period_ms);
    const std::string suffix =
      std::to_string(getpid()) + "_" + std::to_string(test_counter_++);
    const std::string input_topic = "output_tensors_" + suffix;
    const std::string output_topic = "joint_trajectory_" + suffix;

    const auto options = rclcpp::NodeOptions().parameter_overrides({
          rclcpp::Parameter("config_path", config.path()),
          rclcpp::Parameter("input_topic", input_topic),
          rclcpp::Parameter("output_to_topic.actions", output_topic),
    });
    auto output_builder = std::make_shared<OutputBuilderNode>(options);
    auto io_node = std::make_shared<rclcpp::Node>("output_builder_node_test_" + suffix);

    JointCommandTrajectory::SharedPtr received;
    auto subscription = io_node->create_subscription<JointCommandTrajectory>(
      output_topic, 10,
      [&received](const JointCommandTrajectory::SharedPtr msg) {received = msg;});
    auto publisher = io_node->create_publisher<isaac_ros_tensor_msgs::msg::TensorList>(
      input_topic, 10);

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(output_builder);
    executor.add_node(io_node);

    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while ((publisher->get_subscription_count() == 0 ||
      subscription->get_publisher_count() == 0) &&
      std::chrono::steady_clock::now() < deadline)
    {
      executor.spin_some();
      std::this_thread::sleep_for(10ms);
    }
    ASSERT_GT(publisher->get_subscription_count(), 0U);
    ASSERT_GT(subscription->get_publisher_count(), 0U);

    auto tensor_list = tensor_to_tensor_list(
      "actions", torch::zeros({1, 2, 2}, torch::kFloat32), rclcpp::Time(0));
    publisher->publish(tensor_list);
    const auto receive_deadline = std::chrono::steady_clock::now() + 5s;
    while (!received && std::chrono::steady_clock::now() < receive_deadline) {
      executor.spin_some();
      std::this_thread::sleep_for(10ms);
    }

    ASSERT_NE(received, nullptr);
    EXPECT_NEAR(rclcpp::Duration(received->step_dt).seconds(), expected_seconds, 1e-9);
  }

  void expect_configuration_error(
    const std::optional<double> temporal_period_ms,
    const double policy_dt_seconds,
    const std::string & shape,
    const std::optional<std::string> & element_names,
    const std::string & expected_message)
  {
    TemporaryConfig config(
      temporal_period_ms, policy_dt_seconds, shape, element_names);
    const auto options = rclcpp::NodeOptions().parameter_overrides({
          rclcpp::Parameter("config_path", config.path()),
    });
    try {
      std::make_shared<OutputBuilderNode>(options);
      FAIL() << "Expected OutputBuilderNode configuration to fail";
    } catch (const std::runtime_error & error) {
      EXPECT_NE(std::string(error.what()).find(expected_message), std::string::npos)
        << error.what();
    }
  }

  inline static std::atomic_uint test_counter_{0};
};

TEST_F(OutputBuilderNodeTest, UsesTemporalPeriodForTrajectoryStepDt)
{
  expect_step_dt(50.0, 0.05);
}

TEST_F(OutputBuilderNodeTest, FallsBackToPolicyDtForTrajectoryStepDt)
{
  expect_step_dt(std::nullopt, 0.02);
}

TEST_F(OutputBuilderNodeTest, RejectsNonPositivePolicyDt)
{
  expect_configuration_error(std::nullopt, 0.0, "[1, 2, 2]", std::nullopt, "must be positive");
}

TEST_F(OutputBuilderNodeTest, RejectsTemporalMetadataOnNonTrajectoryOutput)
{
  expect_configuration_error(
    50.0, 0.02, "[1, 2]", "[__temporal_axis__, [left, right]]",
    "requires a rank-3 JointCommandTrajectory output");
}

TEST_F(OutputBuilderNodeTest, RejectsUnsupportedTrajectoryTemporalAxis)
{
  expect_configuration_error(
    50.0, 0.02, "[1, 2, 2]", "[[left, right], __temporal_axis__]",
    "requires temporal axis 1");
}

}  // namespace
}  // namespace isaac_ros_deploy_converters

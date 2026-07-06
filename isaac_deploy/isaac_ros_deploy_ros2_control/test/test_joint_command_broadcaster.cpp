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

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "isaac_ros_deploy_ros2_control/controllers/joint_command_broadcaster.hpp"

#include <hardware_interface/loaned_state_interface.hpp>
#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_deploy_interfaces/msg/joint_command.hpp"

using isaac_ros_deploy_ros2_control::controllers::JointCommandBroadcaster;

class JointCommandBroadcasterTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    // Force localhost-only, UDP-only DDS to avoid network discovery and
    // shared-memory hangs in sandboxed remote execution (Bazel RBE).
    setenv("ROS_LOCALHOST_ONLY", "1", 1);
    setenv("ROS_DOMAIN_ID", "99", 1);
    // Disable FastDDS shared-memory transport — /dev/shm may be unavailable
    // or too small in the remote execution sandbox.
    setenv("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4", 1);
    rclcpp::init(0, nullptr);
  }

  static void TearDownTestSuite()
  {
    rclcpp::shutdown();
  }

  void SetUp() override
  {
    broadcaster_ = std::make_unique<JointCommandBroadcaster>();
  }

  /// Initialize the broadcaster's underlying ROS node.
  void init_broadcaster(const std::string & name = "test_broadcaster")
  {
    ASSERT_EQ(
      broadcaster_->init(
        name, "", 100, "",
        rclcpp::NodeOptions()
        .allow_undeclared_parameters(true)
        .automatically_declare_parameters_from_overrides(true)),
      controller_interface::return_type::OK);
  }

  /// Configure the broadcaster with test parameters and inject state interfaces.
  void configure_and_activate(
    const std::vector<std::string> & joint_names,
    const std::string & command_prefix = "safety_controller",
    const std::string & command_suffix = "_raw",
    const std::string & topic_name = "/test_commanded_joint_states")
  {
    init_broadcaster("test_joint_command_broadcaster");

    // Set parameters before configure.
    broadcaster_->get_node()->set_parameter(
      rclcpp::Parameter("joints", joint_names));
    broadcaster_->get_node()->set_parameter(
      rclcpp::Parameter("topic_name", topic_name));
    broadcaster_->get_node()->set_parameter(
      rclcpp::Parameter("command_prefix", command_prefix));
    broadcaster_->get_node()->set_parameter(
      rclcpp::Parameter("command_suffix", command_suffix));

    // Configure.
    ASSERT_EQ(
      broadcaster_->on_configure(rclcpp_lifecycle::State()),
      controller_interface::CallbackReturn::SUCCESS);

    // Prepare mock state interface values and loaned interfaces.
    // Layout per joint: position, velocity, effort, kp, kd.
    constexpr size_t kFieldsPerJoint = 5;
    values_.resize(joint_names.size() * kFieldsPerJoint, 0.0);
    state_interfaces_.clear();
    for (size_t i = 0; i < joint_names.size(); ++i) {
      const auto base = i * kFieldsPerJoint;
      state_interfaces_.emplace_back(hardware_interface::StateInterface(
        command_prefix, joint_names[i] + "/position" + command_suffix, &values_[base + 0]));
      state_interfaces_.emplace_back(hardware_interface::StateInterface(
        command_prefix, joint_names[i] + "/velocity" + command_suffix, &values_[base + 1]));
      state_interfaces_.emplace_back(hardware_interface::StateInterface(
        command_prefix, joint_names[i] + "/effort" + command_suffix, &values_[base + 2]));
      state_interfaces_.emplace_back(hardware_interface::StateInterface(
        command_prefix, joint_names[i] + "/kp" + command_suffix, &values_[base + 3]));
      state_interfaces_.emplace_back(hardware_interface::StateInterface(
        command_prefix, joint_names[i] + "/kd" + command_suffix, &values_[base + 4]));
    }

    // Assign state interfaces to the controller.
    std::vector<hardware_interface::LoanedStateInterface> loaned;
    for (auto & si : state_interfaces_) {
      loaned.emplace_back(si);
    }
    broadcaster_->assign_interfaces({}, std::move(loaned));

    // Activate.
    ASSERT_EQ(
      broadcaster_->on_activate(rclcpp_lifecycle::State()),
      controller_interface::CallbackReturn::SUCCESS);
  }

  std::unique_ptr<JointCommandBroadcaster> broadcaster_;
  std::vector<double> values_;
  std::vector<hardware_interface::StateInterface> state_interfaces_;
};

TEST_F(JointCommandBroadcasterTest, ConfigureSucceedsWithValidParams)
{
  std::vector<std::string> joints = {"joint_a", "joint_b"};
  configure_and_activate(joints);
}

TEST_F(JointCommandBroadcasterTest, ConfigureFailsWithEmptyJoints)
{
  init_broadcaster();

  // Set empty joints list.
  broadcaster_->get_node()->set_parameter(
    rclcpp::Parameter("joints", std::vector<std::string>{}));

  EXPECT_EQ(
    broadcaster_->on_configure(rclcpp_lifecycle::State()),
    controller_interface::CallbackReturn::ERROR);
}

TEST_F(JointCommandBroadcasterTest, UpdatePublishesAllFields)
{
  std::vector<std::string> joints = {"joint_a", "joint_b"};
  configure_and_activate(joints);

  // Set command values: [pos, vel, eff, kp, kd] per joint.
  constexpr size_t kFieldsPerJoint = 5;
  // joint_a
  values_[0 * kFieldsPerJoint + 0] = 1.0;    // position
  values_[0 * kFieldsPerJoint + 1] = 0.5;    // velocity
  values_[0 * kFieldsPerJoint + 2] = 0.1;    // effort
  values_[0 * kFieldsPerJoint + 3] = 100.0;  // kp
  values_[0 * kFieldsPerJoint + 4] = 10.0;   // kd
  // joint_b
  values_[1 * kFieldsPerJoint + 0] = 2.0;
  values_[1 * kFieldsPerJoint + 1] = -0.3;
  values_[1 * kFieldsPerJoint + 2] = 0.0;
  values_[1 * kFieldsPerJoint + 3] = 200.0;
  values_[1 * kFieldsPerJoint + 4] = 20.0;

  // Subscribe to the topic to verify published data.
  isaac_ros_deploy_interfaces::msg::JointCommand received_msg;
  bool msg_received = false;
  auto subscription =
    broadcaster_->get_node()->create_subscription<isaac_ros_deploy_interfaces::msg::JointCommand>(
    "/test_commanded_joint_states", rclcpp::SystemDefaultsQoS(),
    [&](isaac_ros_deploy_interfaces::msg::JointCommand::SharedPtr msg) {
      received_msg = *msg;
      msg_received = true;
    });

  // Run update.
  rclcpp::Time now(1000000000, 0, RCL_ROS_TIME);  // 1 second
  rclcpp::Duration period(0, 5000000);  // 5ms
  ASSERT_EQ(broadcaster_->update(now, period), controller_interface::return_type::OK);

  // Spin with retries — the realtime publisher may need a few cycles.
  for (int i = 0; i < 50 && !msg_received; ++i) {
    broadcaster_->update(now, period);
    rclcpp::spin_some(broadcaster_->get_node()->get_node_base_interface());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_TRUE(msg_received) << "No JointCommand message received on topic";
  ASSERT_EQ(received_msg.names.size(), 2u);
  EXPECT_EQ(received_msg.names[0], "joint_a");
  EXPECT_EQ(received_msg.names[1], "joint_b");

  // Position.
  ASSERT_EQ(received_msg.position.size(), 2u);
  EXPECT_DOUBLE_EQ(received_msg.position[0], 1.0);
  EXPECT_DOUBLE_EQ(received_msg.position[1], 2.0);
  // Velocity.
  ASSERT_EQ(received_msg.velocity.size(), 2u);
  EXPECT_DOUBLE_EQ(received_msg.velocity[0], 0.5);
  EXPECT_DOUBLE_EQ(received_msg.velocity[1], -0.3);
  // Effort.
  ASSERT_EQ(received_msg.effort.size(), 2u);
  EXPECT_DOUBLE_EQ(received_msg.effort[0], 0.1);
  EXPECT_DOUBLE_EQ(received_msg.effort[1], 0.0);
  // Kp.
  ASSERT_EQ(received_msg.kp.size(), 2u);
  EXPECT_DOUBLE_EQ(received_msg.kp[0], 100.0);
  EXPECT_DOUBLE_EQ(received_msg.kp[1], 200.0);
  // Kd.
  ASSERT_EQ(received_msg.kd.size(), 2u);
  EXPECT_DOUBLE_EQ(received_msg.kd[0], 10.0);
  EXPECT_DOUBLE_EQ(received_msg.kd[1], 20.0);
}

TEST_F(JointCommandBroadcasterTest, CommandInterfaceConfigIsNone)
{
  std::vector<std::string> joints = {"joint_a"};
  configure_and_activate(joints);

  auto config = broadcaster_->command_interface_configuration();
  EXPECT_EQ(config.type, controller_interface::interface_configuration_type::NONE);
}

TEST_F(JointCommandBroadcasterTest, StateInterfaceConfigRequestsAllFields)
{
  init_broadcaster();

  broadcaster_->get_node()->set_parameter(
    rclcpp::Parameter("joints", std::vector<std::string>{"joint_a", "joint_b"}));
  broadcaster_->get_node()->set_parameter(
    rclcpp::Parameter("command_prefix", "my_controller"));
  broadcaster_->get_node()->set_parameter(
    rclcpp::Parameter("command_suffix", "_cmd"));

  ASSERT_EQ(
    broadcaster_->on_configure(rclcpp_lifecycle::State()),
    controller_interface::CallbackReturn::SUCCESS);

  auto config = broadcaster_->state_interface_configuration();
  EXPECT_EQ(config.type, controller_interface::interface_configuration_type::INDIVIDUAL);
  // 5 interfaces per joint (position, velocity, effort, kp, kd) x 2 joints = 10.
  ASSERT_EQ(config.names.size(), 10u);
  EXPECT_EQ(config.names[0], "my_controller/joint_a/position_cmd");
  EXPECT_EQ(config.names[1], "my_controller/joint_a/velocity_cmd");
  EXPECT_EQ(config.names[2], "my_controller/joint_a/effort_cmd");
  EXPECT_EQ(config.names[3], "my_controller/joint_a/kp_cmd");
  EXPECT_EQ(config.names[4], "my_controller/joint_a/kd_cmd");
  EXPECT_EQ(config.names[5], "my_controller/joint_b/position_cmd");
  EXPECT_EQ(config.names[6], "my_controller/joint_b/velocity_cmd");
  EXPECT_EQ(config.names[7], "my_controller/joint_b/effort_cmd");
  EXPECT_EQ(config.names[8], "my_controller/joint_b/kp_cmd");
  EXPECT_EQ(config.names[9], "my_controller/joint_b/kd_cmd");
}

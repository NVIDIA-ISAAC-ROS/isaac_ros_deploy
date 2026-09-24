// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hardware_interface/handle.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>
#include <controller_interface/controller_interface_params.hpp>
#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_deploy_ros2_control/controllers/joint_parameter_controller.hpp"

namespace
{

controller_interface::ControllerInterfaceParams make_controller_params(
  const rclcpp::NodeOptions & options)
{
  controller_interface::ControllerInterfaceParams params;
  params.controller_name = "test_joint_parameter_controller";
  params.controller_manager_update_rate = 100;
  params.update_rate = 100;
  params.node_options = options;
  return params;
}

using isaac_ros_deploy_ros2_control::controllers::JointParameterController;

class JointParameterControllerTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    setenv("ROS_LOCALHOST_ONLY", "1", 1);
    setenv("ROS_DOMAIN_ID", "99", 1);
    setenv("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4", 1);
    rclcpp::init(0, nullptr);
  }

  static void TearDownTestSuite()
  {
    rclcpp::shutdown();
  }

  void SetUp() override
  {
    controller_ = std::make_unique<JointParameterController>();
  }

  void TearDown() override
  {
    if (controller_) {
      controller_->release_interfaces();
      controller_.reset();
    }
  }

  void initialize(const std::vector<std::string> & joints = {"joint_a", "joint_b"})
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        rclcpp::Parameter("joints", joints),
        rclcpp::Parameter("default_kp", 10.0),
        rclcpp::Parameter("default_kd", 1.0),
        rclcpp::Parameter("kp_max", 100.0),
        rclcpp::Parameter("kd_max", 10.0)});
    ASSERT_EQ(
      controller_->init(make_controller_params(options)),
      controller_interface::return_type::OK);
    ASSERT_EQ(
      controller_->on_configure(rclcpp_lifecycle::State()),
      controller_interface::CallbackReturn::SUCCESS);
  }

  controller_interface::CallbackReturn configure_with_overrides(
    const std::vector<rclcpp::Parameter> & parameter_overrides)
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides(parameter_overrides);
    EXPECT_EQ(
      controller_->init(make_controller_params(options)),
      controller_interface::return_type::OK);
    return controller_->on_configure(rclcpp_lifecycle::State());
  }

  void assign_interfaces(
    const std::vector<std::string> & joints,
    const std::vector<double> & positions)
  {
    ASSERT_EQ(joints.size(), positions.size());
    state_values_ = positions;
    command_values_.assign(joints.size() * kInterfacesPerJoint, -1.0);

    for (size_t i = 0; i < joints.size(); ++i) {
      state_interfaces_.emplace_back(std::make_shared<hardware_interface::StateInterface>(
        joints[i], "position", &state_values_[i]));
      for (size_t j = 0; j < kInterfacesPerJoint; ++j) {
        command_interfaces_.emplace_back(std::make_shared<hardware_interface::CommandInterface>(
          joints[i], kCommandInterfaces[j], &command_values_[i * kInterfacesPerJoint + j]));
      }
    }

    std::vector<hardware_interface::LoanedStateInterface> loaned_state_interfaces;
    loaned_state_interfaces.reserve(state_interfaces_.size());
    for (const auto & interface : state_interfaces_) {
      loaned_state_interfaces.emplace_back(
        std::const_pointer_cast<const hardware_interface::StateInterface>(interface));
    }
    std::vector<hardware_interface::LoanedCommandInterface> loaned_command_interfaces;
    loaned_command_interfaces.reserve(command_interfaces_.size());
    for (const auto & interface : command_interfaces_) {
      loaned_command_interfaces.emplace_back(interface, [] {});
    }
    controller_->assign_interfaces(
      std::move(loaned_command_interfaces), std::move(loaned_state_interfaces));
  }

  static constexpr size_t kInterfacesPerJoint = 5;
  static constexpr const char * kCommandInterfaces[kInterfacesPerJoint] = {
    "position", "velocity", "effort", "kp", "kd"};

  std::unique_ptr<JointParameterController> controller_;
  std::vector<double> state_values_;
  std::vector<double> command_values_;
  std::vector<hardware_interface::StateInterface::SharedPtr> state_interfaces_;
  std::vector<hardware_interface::CommandInterface::SharedPtr> command_interfaces_;
};

TEST_F(JointParameterControllerTest, DeclaresImpedanceInterfacesForEveryJoint)
{
  initialize();

  const auto command_config = controller_->command_interface_configuration();
  EXPECT_EQ(command_config.type, controller_interface::interface_configuration_type::INDIVIDUAL);
  EXPECT_EQ(
    command_config.names,
    (std::vector<std::string>{
      "joint_a/position", "joint_a/velocity", "joint_a/effort", "joint_a/kp", "joint_a/kd",
      "joint_b/position", "joint_b/velocity", "joint_b/effort", "joint_b/kp", "joint_b/kd"}));

  const auto state_config = controller_->state_interface_configuration();
  EXPECT_EQ(state_config.type, controller_interface::interface_configuration_type::INDIVIDUAL);
  EXPECT_EQ(state_config.names, (std::vector<std::string>{"joint_a/position", "joint_b/position"}));
}

TEST_F(JointParameterControllerTest, RegexGainMapsInitializePerJointSliders)
{
  const std::vector<std::string> joints{
    "left_sharpa_index_MCP_FE_joint",
    "left_sharpa_index_PIP_joint",
    "right_sharpa_thumb_IP_joint"};
  ASSERT_EQ(
    configure_with_overrides({
      rclcpp::Parameter("joints", joints),
      rclcpp::Parameter("default_kp", 1.0),
      rclcpp::Parameter("default_kd", 0.1),
      rclcpp::Parameter("kp..*_(IP|PIP|DIP)_joint", 0.2),
      rclcpp::Parameter("kd..*_(IP|PIP|DIP)_joint", 0.02)}),
    controller_interface::CallbackReturn::SUCCESS);

  EXPECT_DOUBLE_EQ(
    controller_->get_node()->get_parameter(
      "joint.left_sharpa_index_MCP_FE_joint.kp").as_double(), 1.0);
  EXPECT_DOUBLE_EQ(
    controller_->get_node()->get_parameter(
      "joint.left_sharpa_index_PIP_joint.kp").as_double(), 0.2);
  EXPECT_DOUBLE_EQ(
    controller_->get_node()->get_parameter(
      "joint.right_sharpa_thumb_IP_joint.kd").as_double(), 0.02);
}

TEST_F(JointParameterControllerTest, RuntimeParametersDriveImpedanceCommands)
{
  const std::vector<std::string> joints{"joint_a", "joint_b"};
  initialize(joints);
  assign_interfaces(joints, {0.25, -0.5});
  ASSERT_EQ(
    controller_->on_activate(rclcpp_lifecycle::State()),
    controller_interface::CallbackReturn::SUCCESS);

  const auto results = controller_->get_node()->set_parameters_atomically({
      rclcpp::Parameter("joint.joint_a.position_target", 0.5),
      rclcpp::Parameter("joint.joint_a.kp", 20.0),
      rclcpp::Parameter("joint.joint_a.kd", 2.0),
      rclcpp::Parameter("joint.joint_b.position_target", -0.75),
      rclcpp::Parameter("joint.joint_b.kp", 30.0),
      rclcpp::Parameter("joint.joint_b.kd", 3.0)});
  ASSERT_TRUE(results.successful) << results.reason;

  EXPECT_EQ(
    controller_->update(rclcpp::Time(0), rclcpp::Duration::from_seconds(0.01)),
    controller_interface::return_type::OK);

  EXPECT_DOUBLE_EQ(command_values_[0], 0.5);
  EXPECT_DOUBLE_EQ(command_values_[1], 0.0);
  EXPECT_DOUBLE_EQ(command_values_[2], 0.0);
  EXPECT_DOUBLE_EQ(command_values_[3], 20.0);
  EXPECT_DOUBLE_EQ(command_values_[4], 2.0);
  EXPECT_DOUBLE_EQ(command_values_[5], -0.75);
  EXPECT_DOUBLE_EQ(command_values_[6], 0.0);
  EXPECT_DOUBLE_EQ(command_values_[7], 0.0);
  EXPECT_DOUBLE_EQ(command_values_[8], 30.0);
  EXPECT_DOUBLE_EQ(command_values_[9], 3.0);
}

TEST_F(JointParameterControllerTest, ActivationSeedsPositionTargetsFromMeasuredState)
{
  const std::vector<std::string> joints{"joint_a", "joint_b"};
  initialize(joints);
  assign_interfaces(joints, {0.25, -0.5});

  ASSERT_EQ(
    controller_->on_activate(rclcpp_lifecycle::State()),
    controller_interface::CallbackReturn::SUCCESS);
  EXPECT_DOUBLE_EQ(controller_->get_node()->get_parameter(
      "joint.joint_a.position_target").as_double(), 0.25);
  EXPECT_DOUBLE_EQ(controller_->get_node()->get_parameter(
      "joint.joint_b.position_target").as_double(), -0.5);
}

TEST_F(JointParameterControllerTest, ActivationRejectsNonFiniteMeasuredState)
{
  initialize({"joint_a"});
  assign_interfaces({"joint_a"}, {std::numeric_limits<double>::quiet_NaN()});

  EXPECT_EQ(
    controller_->on_activate(rclcpp_lifecycle::State()),
    controller_interface::CallbackReturn::ERROR);
}

TEST_F(JointParameterControllerTest, ActivationRejectsMeasuredStateOutsideSafeRange)
{
  initialize({"joint_a"});
  assign_interfaces({"joint_a"}, {4.0});

  EXPECT_EQ(
    controller_->on_activate(rclcpp_lifecycle::State()),
    controller_interface::CallbackReturn::ERROR);
  EXPECT_EQ(
    controller_->update(rclcpp::Time(0), rclcpp::Duration::from_seconds(0.01)),
    controller_interface::return_type::ERROR);
}

TEST_F(JointParameterControllerTest, ConfigurationRejectsDuplicateJointNames)
{
  EXPECT_EQ(
    configure_with_overrides({
      rclcpp::Parameter("joints", std::vector<std::string>{"joint_a", "joint_a"})}),
    controller_interface::CallbackReturn::ERROR);
}

TEST_F(JointParameterControllerTest, ConfigurationRejectsNonFiniteGain)
{
  EXPECT_EQ(
    configure_with_overrides({
      rclcpp::Parameter("joints", std::vector<std::string>{"joint_a"}),
      rclcpp::Parameter("default_kp", std::numeric_limits<double>::quiet_NaN())}),
    controller_interface::CallbackReturn::ERROR);
}

TEST_F(JointParameterControllerTest, ConfigurationRejectsGainOutsideSliderRange)
{
  EXPECT_EQ(
    configure_with_overrides({
      rclcpp::Parameter("joints", std::vector<std::string>{"joint_a"}),
      rclcpp::Parameter("default_kp", 101.0),
      rclcpp::Parameter("kp_max", 100.0)}),
    controller_interface::CallbackReturn::ERROR);
}

TEST_F(JointParameterControllerTest, ConfigurationRejectsInvertedFallbackRange)
{
  EXPECT_EQ(
    configure_with_overrides({
      rclcpp::Parameter("joints", std::vector<std::string>{"joint_a"}),
      rclcpp::Parameter("position_target_fallback_min", 1.0),
      rclcpp::Parameter("position_target_fallback_max", 1.0)}),
    controller_interface::CallbackReturn::ERROR);
}

TEST_F(JointParameterControllerTest, UpdateIsSafeBeforeActivationAndAfterDeactivation)
{
  initialize({"joint_a"});
  EXPECT_EQ(
    controller_->update(rclcpp::Time(0), rclcpp::Duration::from_seconds(0.01)),
    controller_interface::return_type::ERROR);

  assign_interfaces({"joint_a"}, {0.25});
  ASSERT_EQ(
    controller_->on_activate(rclcpp_lifecycle::State()),
    controller_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(
    controller_->on_deactivate(rclcpp_lifecycle::State()),
    controller_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(
    controller_->update(rclcpp::Time(0), rclcpp::Duration::from_seconds(0.01)),
    controller_interface::return_type::ERROR);
}

TEST_F(JointParameterControllerTest, ReconfigureAfterCleanupPreservesParameters)
{
  initialize();
  ASSERT_TRUE(
    controller_->get_node()->set_parameter(
      rclcpp::Parameter("joint.joint_a.kp", 42.0)).successful);
  ASSERT_EQ(
    controller_->on_cleanup(rclcpp_lifecycle::State()),
    controller_interface::CallbackReturn::SUCCESS);
  EXPECT_TRUE(controller_->get_node()->has_parameter("joint.joint_a.kp"));
  EXPECT_EQ(
    controller_->on_configure(rclcpp_lifecycle::State()),
    controller_interface::CallbackReturn::SUCCESS);
  EXPECT_DOUBLE_EQ(
    controller_->get_node()->get_parameter("joint.joint_a.kp").as_double(), 42.0);
}

}  // namespace

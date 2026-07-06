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

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <controller_interface/chainable_controller_interface.hpp>
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <rclcpp/rclcpp.hpp>

#include "isaac_deploy_core/core/types.hpp"
#include "isaac_ros_deploy_ros2_control/controllers/safety_controller.hpp"
#include "isaac_ros_deploy_ros2_control/safety_controller/safety_controller.hpp"
#include "isaac_ros_deploy_ros2_control/utils/realtime_service_client.hpp"
#include "isaac_ros_deploy_ros2_control/utils/gain_utils.hpp"

using isaac_ros_deploy_ros2_control::controllers::SafetyController;

namespace isaac_ros_deploy_ros2_control
{
namespace controllers
{

class SafetyControllerTestAccess
{
public:
  static std::vector<std::string> & joint_names(SafetyController & controller)
  {
    return controller.joint_names_;
  }

  static bool & velocity_threshold_enabled(SafetyController & controller)
  {
    return controller.velocity_threshold_enabled_;
  }

  static double & max_joint_velocity(SafetyController & controller)
  {
    return controller.max_joint_velocity_;
  }

  static double & mean_joint_velocity(SafetyController & controller)
  {
    return controller.mean_joint_velocity_;
  }

  static std::vector<std::string> & excluded_joint_patterns(SafetyController & controller)
  {
    return controller.excluded_joint_patterns_;
  }

  static std::vector<size_t> & excluded_joint_indices(SafetyController & controller)
  {
    return controller.excluded_joint_indices_;
  }

  static std::optional<size_t> & joint_velocities_input_index(SafetyController & controller)
  {
    return controller.joint_velocities_input_index_;
  }

  static std::optional<isaac_deploy_core::SafetyController> & safety_controller(
    SafetyController & controller)
  {
    return controller.safety_controller_;
  }

  static std::vector<isaac_deploy_core::NamedTensor> & inputs(SafetyController & controller)
  {
    return controller.inputs_;
  }

  static std::vector<isaac_deploy_core::TensorSpec> & input_specs(SafetyController & controller)
  {
    return controller.input_specs_;
  }

  static std::string & emergency_controller(SafetyController & controller)
  {
    return controller.emergency_controller_;
  }

  static double & emergency_switch_timeout_s(SafetyController & controller)
  {
    return controller.emergency_switch_timeout_s_;
  }

  static std::vector<std::string> & configured_emergency_deactivate_controllers(
    SafetyController & controller)
  {
    return controller.configured_emergency_deactivate_controllers_;
  }

  static isaac_ros_deploy_ros2_control::utils::RealtimeServiceClient<
    controller_manager_msgs::srv::SwitchController> & emergency_switch_client(
    SafetyController & controller)
  {
    return controller.emergency_switch_client_;
  }

  static std::string emergency_reason(SafetyController & controller)
  {
    std::lock_guard<std::mutex> lock(controller.emergency_reason_mutex_);
    return controller.emergency_reason_;
  }

  static bool create_safety_controller(SafetyController & controller)
  {
    return controller.create_safety_controller();
  }

  static void resolve_excluded_joint_indices(SafetyController & controller)
  {
    controller.resolve_excluded_joint_indices();
  }

  static controller_manager_msgs::srv::SwitchController::Request build_emergency_switch_request(
    const SafetyController & controller)
  {
    return controller.build_emergency_switch_request();
  }

  static void latch_emergency(SafetyController & controller, const std::string & reason)
  {
    controller.latch_emergency(reason);
  }
};

}  // namespace controllers
}  // namespace isaac_ros_deploy_ros2_control

using isaac_ros_deploy_ros2_control::controllers::SafetyControllerTestAccess;

TEST(RosSafetyControllerTest, StateInterfacesRequestOnlyPositionsWhenVelocityThresholdDisabled)
{
  SafetyController controller;
  SafetyControllerTestAccess::joint_names(controller) = {"joint_a", "joint_b"};
  SafetyControllerTestAccess::velocity_threshold_enabled(controller) = false;

  const auto config = controller.state_interface_configuration();
  ASSERT_EQ(config.type, controller_interface::interface_configuration_type::INDIVIDUAL);
  ASSERT_EQ(config.names.size(), 2u);
  EXPECT_EQ(config.names[0], "joint_a/position");
  EXPECT_EQ(config.names[1], "joint_b/position");
}

TEST(RosSafetyControllerTest, CreateSafetyControllerOmitsVelocityInputWhenThresholdDisabled)
{
  SafetyController controller;
  SafetyControllerTestAccess::joint_names(controller) = {"joint_a", "joint_b"};
  SafetyControllerTestAccess::velocity_threshold_enabled(controller) = false;

  ASSERT_TRUE(SafetyControllerTestAccess::create_safety_controller(controller));
  ASSERT_FALSE(SafetyControllerTestAccess::joint_velocities_input_index(controller).has_value());
  ASSERT_TRUE(SafetyControllerTestAccess::safety_controller(controller).has_value());
  EXPECT_EQ(SafetyControllerTestAccess::safety_controller(controller)->input_names().size(), 4u);
  EXPECT_EQ(SafetyControllerTestAccess::inputs(controller).size(), 4u);
  EXPECT_EQ(SafetyControllerTestAccess::input_specs(controller).size(), 4u);
}

TEST(RosSafetyControllerTest, CreateSafetyControllerAddsVelocityInputWhenThresholdEnabled)
{
  SafetyController controller;
  SafetyControllerTestAccess::joint_names(controller) = {"joint_a", "joint_b"};
  SafetyControllerTestAccess::velocity_threshold_enabled(controller) = true;
  SafetyControllerTestAccess::max_joint_velocity(controller) = 30.0;
  SafetyControllerTestAccess::mean_joint_velocity(controller) = 2.0;

  ASSERT_TRUE(SafetyControllerTestAccess::create_safety_controller(controller));
  ASSERT_TRUE(SafetyControllerTestAccess::joint_velocities_input_index(controller).has_value());
  const auto velocity_input_index =
    SafetyControllerTestAccess::joint_velocities_input_index(controller).value();
  ASSERT_LT(velocity_input_index, SafetyControllerTestAccess::inputs(controller).size());
  ASSERT_TRUE(SafetyControllerTestAccess::safety_controller(controller).has_value());
  EXPECT_EQ(SafetyControllerTestAccess::safety_controller(controller)->input_names().size(), 4u);
  EXPECT_EQ(SafetyControllerTestAccess::inputs(controller).size(), 5u);
  EXPECT_EQ(SafetyControllerTestAccess::input_specs(controller).size(), 5u);
  EXPECT_EQ(
    SafetyControllerTestAccess::inputs(controller)[velocity_input_index].name, "joint_velocities");
  EXPECT_EQ(SafetyControllerTestAccess::inputs(controller)[velocity_input_index].tensor.size(0), 1);
  EXPECT_EQ(SafetyControllerTestAccess::inputs(controller)[velocity_input_index].tensor.size(1), 2);
}

TEST(RosSafetyControllerTest, ResolveExcludedJointIndicesSortsAndDeduplicates)
{
  SafetyController controller;
  SafetyControllerTestAccess::joint_names(controller) = {
    "left_hip_pitch_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_1_joint",
    "right_knee_joint"};
  SafetyControllerTestAccess::excluded_joint_patterns(controller) = {
    "right_hand_.*", "left_hand_.*", ".*hand.*"};

  SafetyControllerTestAccess::resolve_excluded_joint_indices(controller);

  ASSERT_EQ(SafetyControllerTestAccess::excluded_joint_indices(controller).size(), 2u);
  EXPECT_EQ(SafetyControllerTestAccess::excluded_joint_indices(controller)[0], 1u);
  EXPECT_EQ(SafetyControllerTestAccess::excluded_joint_indices(controller)[1], 2u);
}

TEST(RosSafetyControllerTest, StateInterfacesRequestVelocitiesWhenVelocityThresholdEnabled)
{
  SafetyController controller;
  SafetyControllerTestAccess::joint_names(controller) = {"joint_a", "joint_b"};
  SafetyControllerTestAccess::velocity_threshold_enabled(controller) = true;

  const auto config = controller.state_interface_configuration();
  ASSERT_EQ(config.type, controller_interface::interface_configuration_type::INDIVIDUAL);
  ASSERT_EQ(config.names.size(), 4u);
  EXPECT_EQ(config.names[0], "joint_a/position");
  EXPECT_EQ(config.names[1], "joint_b/position");
  EXPECT_EQ(config.names[2], "joint_a/velocity");
  EXPECT_EQ(config.names[3], "joint_b/velocity");
}

TEST(RosSafetyControllerTest, LatchEmergencyKeepsFirstReasonAndDoesNotMarkSecondSwitchPending)
{
  SafetyController controller;
  SafetyControllerTestAccess::emergency_controller(controller) = "freeze_controller";

  SafetyControllerTestAccess::latch_emergency(controller, "velocity threshold exceeded");

  EXPECT_TRUE(SafetyControllerTestAccess::emergency_switch_client(controller).latched());
  EXPECT_EQ(SafetyControllerTestAccess::emergency_reason(controller), "velocity threshold exceeded");
  EXPECT_TRUE(SafetyControllerTestAccess::emergency_switch_client(controller).pending());
  EXPECT_FALSE(SafetyControllerTestAccess::emergency_switch_client(controller).in_flight());

  SafetyControllerTestAccess::latch_emergency(controller, "second violation");

  EXPECT_TRUE(SafetyControllerTestAccess::emergency_switch_client(controller).latched());
  EXPECT_EQ(SafetyControllerTestAccess::emergency_reason(controller), "velocity threshold exceeded");
  EXPECT_TRUE(SafetyControllerTestAccess::emergency_switch_client(controller).pending());
  EXPECT_FALSE(SafetyControllerTestAccess::emergency_switch_client(controller).in_flight());
}

TEST(RosSafetyControllerTest, BuildEmergencySwitchRequestActivatesEmergencyController)
{
  SafetyController controller;
  SafetyControllerTestAccess::emergency_controller(controller) = "freeze_controller";
  SafetyControllerTestAccess::emergency_switch_timeout_s(controller) = 1.25;

  const auto request = SafetyControllerTestAccess::build_emergency_switch_request(controller);

  ASSERT_EQ(request.activate_controllers.size(), 1u);
  EXPECT_EQ(request.activate_controllers[0], "freeze_controller");
  EXPECT_TRUE(request.deactivate_controllers.empty());
  EXPECT_EQ(
    request.strictness,
    controller_manager_msgs::srv::SwitchController::Request::FORCE_AUTO);
  EXPECT_TRUE(request.activate_asap);
  EXPECT_EQ(request.timeout.sec, 1);
  EXPECT_EQ(request.timeout.nanosec, 250000000u);
}

TEST(RosSafetyControllerTest, BuildEmergencySwitchRequestUsesConfiguredDeactivateControllers)
{
  SafetyController controller;
  SafetyControllerTestAccess::emergency_controller(controller) = "freeze_controller";
  SafetyControllerTestAccess::emergency_switch_timeout_s(controller) = 1.25;
  SafetyControllerTestAccess::configured_emergency_deactivate_controllers(controller) = {
    "joint_command_broadcaster",
    "safety_controller",
    "walking_controller",
  };

  const auto request = SafetyControllerTestAccess::build_emergency_switch_request(controller);

  EXPECT_EQ(request.deactivate_controllers, std::vector<std::string>({
    "joint_command_broadcaster",
    "safety_controller",
    "walking_controller",
  }));
  EXPECT_EQ(
    request.strictness,
    controller_manager_msgs::srv::SwitchController::Request::BEST_EFFORT);
}

TEST(RosSafetyControllerTest, RealtimeServiceClientTriggerOnceLatchesAndMarksPending)
{
  isaac_ros_deploy_ros2_control::utils::RealtimeServiceClient<
    controller_manager_msgs::srv::SwitchController> client;

  EXPECT_TRUE(client.trigger_once());
  EXPECT_TRUE(client.latched());
  EXPECT_TRUE(client.pending());
  EXPECT_FALSE(client.in_flight());

  EXPECT_FALSE(client.trigger_once());
  EXPECT_TRUE(client.latched());
  EXPECT_TRUE(client.pending());
  EXPECT_FALSE(client.in_flight());

  client.reset();

  EXPECT_FALSE(client.latched());
  EXPECT_FALSE(client.pending());
  EXPECT_FALSE(client.in_flight());
}

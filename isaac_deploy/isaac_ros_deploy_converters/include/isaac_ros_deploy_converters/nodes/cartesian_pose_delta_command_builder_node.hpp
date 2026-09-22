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

#pragma once

#include <atomic>
#include <array>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "isaac_ros_deploy_interfaces/msg/cartesian_pose_delta_command.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/empty.hpp"

namespace isaac_ros_deploy_converters
{

/// Builds Cartesian pose-delta commands by anchoring policy outputs to their input observation.
///
/// Two parameters exist for hardware. ``enable_topic`` gates the node so an
/// orchestrator can hold the policy closed and know that no queued action can fire
/// the moment it opens again. ``max_policy_delay_s`` drops an action whose
/// observation is already too old to act on, and asks the input builder to restart
/// the recurrent state rather than resume a policy rollout with a hole in it.
class CartesianPoseDeltaCommandBuilderNode : public rclcpp::Node
{
public:
  explicit CartesianPoseDeltaCommandBuilderNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  struct PendingAction
  {
    builtin_interfaces::msg::Time stamp;
    std::vector<double> action;
  };

  struct PendingObservation
  {
    builtin_interfaces::msg::Time stamp;
    geometry_msgs::msg::PoseStamped pose;
  };

  void observation_tensors_callback(
    const isaac_ros_tensor_msgs::msg::TensorList::SharedPtr msg);

  std::optional<geometry_msgs::msg::PoseStamped> tensor_list_to_observation_pose(
    const isaac_ros_tensor_msgs::msg::TensorList & msg) const;

  std::optional<geometry_msgs::msg::PoseStamped> pop_matching_observation(double stamp_s);

  bool is_action_tensor(const std::string & name) const;

  bool is_observation_position_tensor(
    const std::string & name) const;

  bool is_observation_rotation_tensor(
    const std::string & name) const;

  void action_tensors_callback(
    const isaac_ros_tensor_msgs::msg::TensorList::SharedPtr msg);

  void publish_command(
    const PendingAction & pending_action,
    const geometry_msgs::msg::PoseStamped & observation);

  void publish_ready_commands();

  void enable_callback(const std_msgs::msg::Bool::SharedPtr msg);

  void clear_pending(const std::string & reason);

  void request_feedback_reset();

  std::string action_tensor_name_;
  std::string action_tensor_base_name_;
  std::string observation_position_tensor_name_;
  std::string observation_position_tensor_base_name_;
  std::string observation_rotation_tensor_name_;
  std::string observation_rotation_tensor_base_name_;
  std::string observation_frame_id_;
  double sync_tolerance_s_{0.01};
  double max_policy_delay_s_{0.0};
  std::atomic_bool enabled_{true};
  size_t max_observation_queue_{512};
  size_t published_count_{0};
  std::deque<PendingAction> pending_actions_;
  std::deque<PendingObservation> observations_;
  rclcpp::Publisher<isaac_ros_deploy_interfaces::msg::CartesianPoseDeltaCommand>::SharedPtr
    publisher_;
  rclcpp::Subscription<isaac_ros_tensor_msgs::msg::TensorList>::SharedPtr
    observation_sub_;
  rclcpp::Subscription<isaac_ros_tensor_msgs::msg::TensorList>::SharedPtr action_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_sub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr feedback_reset_pub_;
};

}  // namespace isaac_ros_deploy_converters

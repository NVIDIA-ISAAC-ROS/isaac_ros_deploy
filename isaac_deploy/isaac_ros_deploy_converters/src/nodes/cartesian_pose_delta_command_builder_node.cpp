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

#include "isaac_ros_deploy_converters/nodes/cartesian_pose_delta_command_builder_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "rclcpp_components/register_node_macro.hpp"

#include "isaac_ros_deploy_converters/utils/tensor_list_utils.hpp"

namespace isaac_ros_deploy_converters
{

namespace
{

constexpr size_t kCartesianPoseDeltaWidth = 6;

double stamp_to_seconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

std::string strip_model_prefix(const std::string & name)
{
  const auto pos = name.rfind('/');
  if (pos == std::string::npos) {
    return name;
  }
  return name.substr(pos + 1);
}

std::vector<double> tensor_to_doubles(
  const tensor_msgs::msg::ExperimentalTensor & tensor,
  const std::string & name)
{
  // Read through tensor_msg_to_torch rather than tensor.data directly: the payload
  // may be backed by device memory, and a raw host copy of it yields silent garbage
  // rather than a visible failure. tensor_msg_to_torch also validates the dtype,
  // the row-major layout, and the metadata against the backing buffer.
  try {
    const torch::Tensor values =
      tensor_msg_to_torch(tensor).reshape({-1}).to(torch::kFloat64).contiguous();
    const double * begin = values.data_ptr<double>();
    return std::vector<double>(begin, begin + values.numel());
  } catch (const std::exception & e) {
    throw std::runtime_error("Tensor '" + name + "': " + e.what());
  }
}

std::array<double, 3> normalize(const std::array<double, 3> & vector)
{
  const double norm = std::sqrt(
    vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
  if (norm <= 0.0 || !std::isfinite(norm)) {
    throw std::runtime_error("Cannot normalize invalid rotation_6d vector");
  }
  return {vector[0] / norm, vector[1] / norm, vector[2] / norm};
}

double dot_product(
  const std::array<double, 3> & lhs,
  const std::array<double, 3> & rhs)
{
  return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

std::array<double, 3> cross_product(
  const std::array<double, 3> & lhs,
  const std::array<double, 3> & rhs)
{
  return {
    lhs[1] * rhs[2] - lhs[2] * rhs[1],
    lhs[2] * rhs[0] - lhs[0] * rhs[2],
    lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

geometry_msgs::msg::Quaternion rotation_matrix_rows_to_quaternion(
  const std::array<double, 3> & row0,
  const std::array<double, 3> & row1,
  const std::array<double, 3> & row2)
{
  geometry_msgs::msg::Quaternion quat;
  const double trace = row0[0] + row1[1] + row2[2];
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    quat.w = 0.25 * s;
    quat.x = (row2[1] - row1[2]) / s;
    quat.y = (row0[2] - row2[0]) / s;
    quat.z = (row1[0] - row0[1]) / s;
  } else if (row0[0] > row1[1] && row0[0] > row2[2]) {
    const double s = std::sqrt(1.0 + row0[0] - row1[1] - row2[2]) * 2.0;
    quat.w = (row2[1] - row1[2]) / s;
    quat.x = 0.25 * s;
    quat.y = (row0[1] + row1[0]) / s;
    quat.z = (row0[2] + row2[0]) / s;
  } else if (row1[1] > row2[2]) {
    const double s = std::sqrt(1.0 + row1[1] - row0[0] - row2[2]) * 2.0;
    quat.w = (row0[2] - row2[0]) / s;
    quat.x = (row0[1] + row1[0]) / s;
    quat.y = 0.25 * s;
    quat.z = (row1[2] + row2[1]) / s;
  } else {
    const double s = std::sqrt(1.0 + row2[2] - row0[0] - row1[1]) * 2.0;
    quat.w = (row1[0] - row0[1]) / s;
    quat.x = (row0[2] + row2[0]) / s;
    quat.y = (row1[2] + row2[1]) / s;
    quat.z = 0.25 * s;
  }

  const double norm = std::sqrt(
    quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w);
  if (norm <= 0.0 || !std::isfinite(norm)) {
    throw std::runtime_error("Cannot convert invalid rotation_6d tensor to quaternion");
  }
  quat.x /= norm;
  quat.y /= norm;
  quat.z /= norm;
  quat.w /= norm;
  return quat;
}

geometry_msgs::msg::PoseStamped make_pose_from_position_and_rotation_6d(
  const std::vector<double> & position,
  const std::vector<double> & rotation_6d,
  const builtin_interfaces::msg::Time & stamp,
  const std::string & frame_id)
{
  if (position.size() < 3) {
    throw std::runtime_error("Observation position tensor must have at least 3 values");
  }
  if (rotation_6d.size() < 6) {
    throw std::runtime_error("Observation rotation_6d tensor must have at least 6 values");
  }

  const std::array<double, 3> raw_row0{rotation_6d[0], rotation_6d[1], rotation_6d[2]};
  const std::array<double, 3> raw_row1{rotation_6d[3], rotation_6d[4], rotation_6d[5]};
  const auto row0 = normalize(raw_row0);
  const double row1_projection = dot_product(row0, raw_row1);
  const std::array<double, 3> row1_orthogonal{
    raw_row1[0] - row1_projection * row0[0],
    raw_row1[1] - row1_projection * row0[1],
    raw_row1[2] - row1_projection * row0[2]};
  const auto row1 = normalize(row1_orthogonal);
  const auto row2 = normalize(cross_product(row0, row1));

  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = frame_id;
  pose.pose.position.x = position[0];
  pose.pose.position.y = position[1];
  pose.pose.position.z = position[2];
  pose.pose.orientation = rotation_matrix_rows_to_quaternion(row0, row1, row2);
  return pose;
}

}  // namespace

CartesianPoseDeltaCommandBuilderNode::CartesianPoseDeltaCommandBuilderNode(
  const rclcpp::NodeOptions & options)
: Node("cartesian_pose_delta_command_builder_node", options)
{
  declare_parameter<std::string>("action_tensor_topic", "output_tensors");
  declare_parameter<std::string>("observation_tensor_topic", "input_tensors");
  declare_parameter<std::string>("output_topic", "pose_delta_command");
  declare_parameter<std::string>("action_tensor_name", "arm_action");
  declare_parameter<std::string>("observation_position_tensor_name", "eef_pos");
  declare_parameter<std::string>("observation_rotation_6d_tensor_name", "eef_rot_6d");
  declare_parameter<std::string>("observation_frame_id", "world");
  declare_parameter<double>("sync_tolerance_s", 0.01);
  declare_parameter<int>("max_observation_queue", 512);
  declare_parameter<double>("max_policy_delay_s", 0.0);
  declare_parameter<std::string>("enable_topic", "");
  declare_parameter<bool>("enabled_on_start", true);
  declare_parameter<std::string>("feedback_reset_topic", "");

  const auto action_tensor_topic = get_parameter("action_tensor_topic").as_string();
  const auto observation_tensor_topic = get_parameter("observation_tensor_topic").as_string();
  const auto output_topic = get_parameter("output_topic").as_string();
  action_tensor_name_ = get_parameter("action_tensor_name").as_string();
  action_tensor_base_name_ = strip_model_prefix(action_tensor_name_);
  observation_position_tensor_name_ =
    get_parameter("observation_position_tensor_name").as_string();
  observation_position_tensor_base_name_ = strip_model_prefix(observation_position_tensor_name_);
  observation_rotation_tensor_name_ =
    get_parameter("observation_rotation_6d_tensor_name").as_string();
  observation_rotation_tensor_base_name_ = strip_model_prefix(observation_rotation_tensor_name_);
  observation_frame_id_ = get_parameter("observation_frame_id").as_string();
  sync_tolerance_s_ = get_parameter("sync_tolerance_s").as_double();
  if (sync_tolerance_s_ < 0.0) {
    throw std::runtime_error("sync_tolerance_s must be non-negative");
  }
  max_observation_queue_ = static_cast<size_t>(
    std::max<int64_t>(0, get_parameter("max_observation_queue").as_int()));
  if (max_observation_queue_ == 0) {
    throw std::runtime_error("max_observation_queue must be positive");
  }
  max_policy_delay_s_ = get_parameter("max_policy_delay_s").as_double();
  if (max_policy_delay_s_ < 0.0) {
    throw std::runtime_error("max_policy_delay_s must be non-negative");
  }
  enabled_ = get_parameter("enabled_on_start").as_bool();

  auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
  publisher_ = create_publisher<isaac_ros_deploy_interfaces::msg::CartesianPoseDeltaCommand>(
    output_topic, 10);
  observation_sub_ = create_subscription<isaac_ros_tensor_msgs::msg::TensorList>(
    observation_tensor_topic, qos,
    std::bind(&CartesianPoseDeltaCommandBuilderNode::observation_tensors_callback, this,
    std::placeholders::_1));
  action_sub_ = create_subscription<isaac_ros_tensor_msgs::msg::TensorList>(
    action_tensor_topic, qos,
    std::bind(&CartesianPoseDeltaCommandBuilderNode::action_tensors_callback, this,
    std::placeholders::_1));

  const auto enable_topic = get_parameter("enable_topic").as_string();
  if (!enable_topic.empty()) {
    enable_sub_ = create_subscription<std_msgs::msg::Bool>(
      enable_topic, 10,
      std::bind(
        &CartesianPoseDeltaCommandBuilderNode::enable_callback, this, std::placeholders::_1));
  } else {
    // With no gate topic there is nothing that could ever open the gate, so an
    // absent gate means always enabled rather than permanently mute.
    if (!enabled_) {
      RCLCPP_WARN(
        get_logger(),
        "Ignoring enabled_on_start=false because enable_topic is empty: with no gate topic "
        "nothing could ever enable this node. Set enable_topic to gate publication.");
    }
    enabled_ = true;
  }
  const auto feedback_reset_topic = get_parameter("feedback_reset_topic").as_string();
  if (!feedback_reset_topic.empty()) {
    feedback_reset_pub_ = create_publisher<std_msgs::msg::Empty>(feedback_reset_topic, 10);
  }

  RCLCPP_INFO(
    get_logger(),
    "Configured CartesianPoseDeltaCommandBuilderNode: action_tensor_topic=%s, "
    "observation_tensor_topic=%s, output_topic=%s, action_tensor_name=%s, "
    "observation_position_tensor_name=%s, observation_rotation_6d_tensor_name=%s, "
    "observation_frame_id=%s, sync_tolerance_s=%.6f, max_policy_delay_s=%.6f, "
    "enable_topic=%s, enabled=%s",
    action_tensor_topic.c_str(), observation_tensor_topic.c_str(), output_topic.c_str(),
    action_tensor_name_.c_str(), observation_position_tensor_name_.c_str(),
    observation_rotation_tensor_name_.c_str(), observation_frame_id_.c_str(),
    sync_tolerance_s_, max_policy_delay_s_,
    enable_topic.empty() ? "<always enabled>" : enable_topic.c_str(),
    enabled_ ? "true" : "false");
}

void CartesianPoseDeltaCommandBuilderNode::observation_tensors_callback(
  const isaac_ros_tensor_msgs::msg::TensorList::SharedPtr msg)
{
  if (!enabled_) {
    return;
  }

  const auto observation_pose = tensor_list_to_observation_pose(*msg);
  if (!observation_pose) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Waiting for observation tensors '%s' and '%s'",
      observation_position_tensor_name_.c_str(), observation_rotation_tensor_name_.c_str());
    return;
  }

  observations_.push_back(PendingObservation{msg->header.stamp, *observation_pose});
  while (observations_.size() > max_observation_queue_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Discarding stale task-space observation pose because no matching action arrived; "
      "expected briefly while inference warms up, otherwise the action stream is falling behind");
    observations_.pop_front();
  }
  publish_ready_commands();
}

std::optional<geometry_msgs::msg::PoseStamped>
CartesianPoseDeltaCommandBuilderNode::tensor_list_to_observation_pose(
  const isaac_ros_tensor_msgs::msg::TensorList & msg) const
{
  std::optional<std::vector<double>> position;
  std::optional<std::vector<double>> rotation_6d;
  for (size_t i = 0; i < msg.tensors.size() && i < msg.names.size(); ++i) {
    const auto & name = msg.names[i];
    const auto & tensor = msg.tensors[i];
    try {
      if (is_observation_position_tensor(name)) {
        position = tensor_to_doubles(tensor, name);
      } else if (is_observation_rotation_tensor(name)) {
        rotation_6d = tensor_to_doubles(tensor, name);
      }
    } catch (const std::exception & exc) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", exc.what());
      return std::nullopt;
    }
  }

  if (!position || !rotation_6d) {
    return std::nullopt;
  }

  try {
    return make_pose_from_position_and_rotation_6d(
      *position, *rotation_6d, msg.header.stamp, observation_frame_id_);
  } catch (const std::exception & exc) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", exc.what());
    return std::nullopt;
  }
}

std::optional<geometry_msgs::msg::PoseStamped>
CartesianPoseDeltaCommandBuilderNode::pop_matching_observation(const double stamp_s)
{
  if (observations_.empty()) {
    return std::nullopt;
  }

  auto best_it = observations_.end();
  double best_diff = std::numeric_limits<double>::infinity();
  for (auto it = observations_.begin(); it != observations_.end(); ++it) {
    const double diff = std::abs(stamp_to_seconds(it->stamp) - stamp_s);
    if (diff < best_diff) {
      best_diff = diff;
      best_it = it;
    }
  }

  if (best_it == observations_.end() || best_diff > sync_tolerance_s_) {
    return std::nullopt;
  }

  auto result = best_it->pose;
  observations_.erase(best_it);
  return result;
}

bool CartesianPoseDeltaCommandBuilderNode::is_action_tensor(
  const std::string & name) const
{
  const auto base_name = strip_model_prefix(name);
  return name == action_tensor_name_ || name == action_tensor_base_name_ ||
         base_name == action_tensor_name_ || base_name == action_tensor_base_name_;
}

bool CartesianPoseDeltaCommandBuilderNode::is_observation_position_tensor(
  const std::string & name) const
{
  const auto base_name = strip_model_prefix(name);
  return name == observation_position_tensor_name_ ||
         name == observation_position_tensor_base_name_ ||
         base_name == observation_position_tensor_name_ ||
         base_name == observation_position_tensor_base_name_;
}

bool CartesianPoseDeltaCommandBuilderNode::is_observation_rotation_tensor(
  const std::string & name) const
{
  const auto base_name = strip_model_prefix(name);
  return name == observation_rotation_tensor_name_ ||
         name == observation_rotation_tensor_base_name_ ||
         base_name == observation_rotation_tensor_name_ ||
         base_name == observation_rotation_tensor_base_name_;
}

void CartesianPoseDeltaCommandBuilderNode::action_tensors_callback(
  const isaac_ros_tensor_msgs::msg::TensorList::SharedPtr msg)
{
  if (!enabled_) {
    return;
  }

  if (max_policy_delay_s_ > 0.0) {
    const double delay_s = now().seconds() - stamp_to_seconds(msg->header.stamp);
    if (delay_s > max_policy_delay_s_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Dropping task-space action because the policy input-output delay %.6fs exceeds "
        "max_policy_delay_s=%.6fs", delay_s, max_policy_delay_s_);
      clear_pending("policy output exceeded the input-output delay limit");
      request_feedback_reset();
      return;
    }
  }

  std::optional<std::vector<double>> action;
  for (size_t i = 0; i < msg->tensors.size() && i < msg->names.size(); ++i) {
    const auto & name = msg->names[i];
    const auto & tensor = msg->tensors[i];
    if (!is_action_tensor(name)) {
      continue;
    }
    try {
      action = tensor_to_doubles(tensor, name);
    } catch (const std::exception & exc) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", exc.what());
      return;
    }
    break;
  }

  if (!action) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Waiting for action tensor '%s'",
      action_tensor_name_.c_str());
    return;
  }
  if (action->size() < kCartesianPoseDeltaWidth) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Action tensor has %zu values; expected at least %zu",
      action->size(), kCartesianPoseDeltaWidth);
    return;
  }

  pending_actions_.push_back(PendingAction{msg->header.stamp, *action});
  while (pending_actions_.size() > max_observation_queue_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Discarding stale task-space action because no matching observation pose arrived");
    pending_actions_.pop_front();
  }
  publish_ready_commands();
}

void CartesianPoseDeltaCommandBuilderNode::clear_pending(const std::string & reason)
{
  if (pending_actions_.empty() && observations_.empty()) {
    return;
  }
  RCLCPP_WARN_THROTTLE(
    get_logger(), *get_clock(), 1000,
    "Cleared %zu pending action(s) and %zu observation(s): %s",
    pending_actions_.size(), observations_.size(), reason.c_str());
  pending_actions_.clear();
  observations_.clear();
}

void CartesianPoseDeltaCommandBuilderNode::request_feedback_reset()
{
  if (!feedback_reset_pub_) {
    return;
  }
  feedback_reset_pub_->publish(std_msgs::msg::Empty());
}

void CartesianPoseDeltaCommandBuilderNode::enable_callback(
  const std_msgs::msg::Bool::SharedPtr msg)
{
  const bool enabled = msg->data;
  if (enabled == enabled_) {
    return;
  }
  enabled_ = enabled;
  // Drop the queues in both directions: a queued action from before the gate
  // closed must not fire when it opens again.
  clear_pending("the enable gate changed");
  RCLCPP_INFO(
    get_logger(), "Cartesian pose-delta command building %s",
    enabled_ ? "enabled" : "disabled");
}

void CartesianPoseDeltaCommandBuilderNode::publish_command(
  const PendingAction & pending_action,
  const geometry_msgs::msg::PoseStamped & observation)
{
  isaac_ros_deploy_interfaces::msg::CartesianPoseDeltaCommand command;
  command.header.stamp = pending_action.stamp;
  command.header.frame_id = observation.header.frame_id;
  command.delta_position.x = pending_action.action[0];
  command.delta_position.y = pending_action.action[1];
  command.delta_position.z = pending_action.action[2];
  command.delta_axis_angle.x = pending_action.action[3];
  command.delta_axis_angle.y = pending_action.action[4];
  command.delta_axis_angle.z = pending_action.action[5];
  command.observation_pose = observation.pose;
  publisher_->publish(command);

  ++published_count_;
  if (published_count_ == 1 || published_count_ % 300 == 0) {
    RCLCPP_DEBUG(
      get_logger(),
      "Published CartesianPoseDeltaCommand count=%zu, stamp=%.6f, "
      "delta_position=[%.6f, %.6f, %.6f], delta_axis_angle=[%.6f, %.6f, %.6f]",
      published_count_, stamp_to_seconds(pending_action.stamp),
      pending_action.action[0], pending_action.action[1], pending_action.action[2],
      pending_action.action[3], pending_action.action[4], pending_action.action[5]);
  }
}

void CartesianPoseDeltaCommandBuilderNode::publish_ready_commands()
{
  for (auto it = pending_actions_.begin(); it != pending_actions_.end(); ) {
    auto observation = pop_matching_observation(stamp_to_seconds(it->stamp));
    if (!observation) {
      ++it;
      continue;
    }
    publish_command(*it, *observation);
    it = pending_actions_.erase(it);
  }
}

}  // namespace isaac_ros_deploy_converters

RCLCPP_COMPONENTS_REGISTER_NODE(
  isaac_ros_deploy_converters::CartesianPoseDeltaCommandBuilderNode)

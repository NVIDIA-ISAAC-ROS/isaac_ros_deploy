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

#include "isaac_ros_deploy_converters/nodes/cartesian_pose_delta_decoder_node.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/bool.hpp"

namespace isaac_ros_deploy_converters
{

namespace
{

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;

Vec3 as_vec3(const std::vector<double> & values, const std::string & name)
{
  if (values.size() != 3) {
    throw std::runtime_error(name + " must contain 3 values");
  }
  for (const auto value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(name + " must contain only finite values");
    }
  }
  return {values[0], values[1], values[2]};
}

Quat normalize_quat_xyzw(const Quat & quat)
{
  const double norm = std::sqrt(
    quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
  if (norm <= 0.0 || !std::isfinite(norm)) {
    return {0.0, 0.0, 0.0, 1.0};
  }
  return {quat[0] / norm, quat[1] / norm, quat[2] / norm, quat[3] / norm};
}

Quat multiply_quat_xyzw(const Quat & lhs, const Quat & rhs)
{
  const auto a = normalize_quat_xyzw(lhs);
  const auto b = normalize_quat_xyzw(rhs);
  return normalize_quat_xyzw({
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  });
}

std::array<std::array<double, 3>, 3> quat_to_matrix_xyzw(const Quat & quat)
{
  const auto q = normalize_quat_xyzw(quat);
  const double x = q[0];
  const double y = q[1];
  const double z = q[2];
  const double w = q[3];
  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;
  const double wx = w * x;
  const double wy = w * y;
  const double wz = w * z;
  return {{
    {{1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)}},
    {{2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)}},
    {{2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)}},
  }};
}

Vec3 rotate_by_quat_xyzw(const Quat & quat, const Vec3 & vector)
{
  const auto matrix = quat_to_matrix_xyzw(quat);
  return {
    matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
    matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
    matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
  };
}

Quat axis_angle_to_quat_xyzw(const Vec3 & axis_angle)
{
  const double angle = std::sqrt(
    axis_angle[0] * axis_angle[0] +
    axis_angle[1] * axis_angle[1] +
    axis_angle[2] * axis_angle[2]);
  if (angle <= 1e-12 || !std::isfinite(angle)) {
    return {0.0, 0.0, 0.0, 1.0};
  }
  const double half_angle = 0.5 * angle;
  const double sin_half = std::sin(half_angle);
  return normalize_quat_xyzw({
        axis_angle[0] / angle * sin_half,
        axis_angle[1] / angle * sin_half,
        axis_angle[2] / angle * sin_half,
        std::cos(half_angle),
  });
}

std::pair<Vec3, Quat> pose_to_arrays(const geometry_msgs::msg::Pose & pose)
{
  return {
    {pose.position.x, pose.position.y, pose.position.z},
    normalize_quat_xyzw({
          pose.orientation.x,
          pose.orientation.y,
          pose.orientation.z,
          pose.orientation.w,
    }),
  };
}

geometry_msgs::msg::Pose arrays_to_pose(const Vec3 & pos, const Quat & quat_xyzw)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = pos[0];
  pose.position.y = pos[1];
  pose.position.z = pos[2];
  const auto quat = normalize_quat_xyzw(quat_xyzw);
  pose.orientation.x = quat[0];
  pose.orientation.y = quat[1];
  pose.orientation.z = quat[2];
  pose.orientation.w = quat[3];
  return pose;
}

std::pair<Vec3, Quat> decode_cartesian_pose_delta(
  const isaac_ros_deploy_interfaces::msg::CartesianPoseDeltaCommand & command,
  const double action_scale,
  const Vec3 & output_tcp_offset)
{
  if (!std::isfinite(action_scale)) {
    throw std::runtime_error("action_scale must be finite");
  }
  if (!std::isfinite(command.delta_position.x) ||
    !std::isfinite(command.delta_position.y) ||
    !std::isfinite(command.delta_position.z))
  {
    throw std::runtime_error("delta_position must be finite");
  }
  if (!std::isfinite(command.delta_axis_angle.x) ||
    !std::isfinite(command.delta_axis_angle.y) ||
    !std::isfinite(command.delta_axis_angle.z))
  {
    throw std::runtime_error("delta_axis_angle must be finite");
  }
  if (!std::isfinite(command.observation_pose.position.x) ||
    !std::isfinite(command.observation_pose.position.y) ||
    !std::isfinite(command.observation_pose.position.z))
  {
    throw std::runtime_error("observation_pose position must be finite");
  }
  // normalize_quat_xyzw() falls back to identity for a non-finite or zero
  // quaternion, which would silently decode against the wrong orientation.
  const auto & observation_orientation = command.observation_pose.orientation;
  if (!std::isfinite(observation_orientation.x) ||
    !std::isfinite(observation_orientation.y) ||
    !std::isfinite(observation_orientation.z) ||
    !std::isfinite(observation_orientation.w))
  {
    throw std::runtime_error("observation_pose orientation must be finite");
  }
  if (std::sqrt(
      observation_orientation.x * observation_orientation.x +
      observation_orientation.y * observation_orientation.y +
      observation_orientation.z * observation_orientation.z +
      observation_orientation.w * observation_orientation.w) <= 0.0)
  {
    throw std::runtime_error("observation_pose orientation must have non-zero norm");
  }

  const auto [observation_pos, observation_quat] = pose_to_arrays(command.observation_pose);
  const Vec3 scaled_delta_position{
    action_scale * command.delta_position.x,
    action_scale * command.delta_position.y,
    action_scale * command.delta_position.z,
  };
  const Vec3 scaled_delta_axis_angle{
    action_scale * command.delta_axis_angle.x,
    action_scale * command.delta_axis_angle.y,
    action_scale * command.delta_axis_angle.z,
  };

  const Vec3 target_tcp_pos{
    observation_pos[0] + scaled_delta_position[0],
    observation_pos[1] + scaled_delta_position[1],
    observation_pos[2] + scaled_delta_position[2],
  };
  const auto target_tcp_quat = multiply_quat_xyzw(
    axis_angle_to_quat_xyzw(scaled_delta_axis_angle),
    observation_quat);
  const auto rotated_tcp_offset = rotate_by_quat_xyzw(target_tcp_quat, output_tcp_offset);
  const Vec3 target_pos{
    target_tcp_pos[0] - rotated_tcp_offset[0],
    target_tcp_pos[1] - rotated_tcp_offset[1],
    target_tcp_pos[2] - rotated_tcp_offset[2],
  };
  return {target_pos, target_tcp_quat};
}

}  // namespace

CartesianPoseDeltaDecoderNode::CartesianPoseDeltaDecoderNode(
  const rclcpp::NodeOptions & options)
: Node("cartesian_pose_delta_decoder_node", options)
{
  declare_parameter<std::string>("input_topic", "pose_delta_command");
  declare_parameter<std::string>("output_topic", "target_pose");
  declare_parameter<double>("action_scale", 1.0);
  declare_parameter<std::vector<double>>("output_tcp_offset", {0.0, 0.0, 0.0});
  declare_parameter<std::string>("output_tcp_offset_topic", "");
  declare_parameter<std::string>("output_frame_id", "");
  declare_parameter<std::string>("enable_topic", "");
  declare_parameter<bool>("enabled_on_start", true);

  const auto input_topic = get_parameter("input_topic").as_string();
  const auto output_topic = get_parameter("output_topic").as_string();
  action_scale_ = get_parameter("action_scale").as_double();
  output_tcp_offset_ = as_vec3(
    get_parameter("output_tcp_offset").as_double_array(), "output_tcp_offset");
  output_frame_id_ = get_parameter("output_frame_id").as_string();
  enabled_ = get_parameter("enabled_on_start").as_bool();

  publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>(output_topic, 10);
  subscription_ =
    create_subscription<isaac_ros_deploy_interfaces::msg::CartesianPoseDeltaCommand>(
    input_topic, 10,
    std::bind(&CartesianPoseDeltaDecoderNode::command_callback, this, std::placeholders::_1));

  const auto enable_topic = get_parameter("enable_topic").as_string();
  if (!enable_topic.empty()) {
    enable_sub_ = create_subscription<std_msgs::msg::Bool>(
      enable_topic, 10,
      std::bind(&CartesianPoseDeltaDecoderNode::enable_callback, this, std::placeholders::_1));
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

  const auto tcp_offset_topic = get_parameter("output_tcp_offset_topic").as_string();
  if (!tcp_offset_topic.empty()) {
    // Latched by the pose source, which resolves the offset once at startup.
    tcp_offset_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
      tcp_offset_topic,
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
      std::bind(
        &CartesianPoseDeltaDecoderNode::tcp_offset_callback, this, std::placeholders::_1));
  }

  RCLCPP_INFO(
    get_logger(),
    "Configured CartesianPoseDeltaDecoderNode: input_topic=%s, output_topic=%s, "
    "action_scale=%.6f, output_tcp_offset=[%.6f, %.6f, %.6f], output_tcp_offset_topic=%s, "
    "enable_topic=%s, enabled=%s",
    input_topic.c_str(), output_topic.c_str(), action_scale_,
    output_tcp_offset_[0], output_tcp_offset_[1], output_tcp_offset_[2],
    tcp_offset_topic.empty() ? "<static>" : tcp_offset_topic.c_str(),
    enable_topic.empty() ? "<always enabled>" : enable_topic.c_str(),
    enabled_ ? "true" : "false");
}

void CartesianPoseDeltaDecoderNode::enable_callback(const std_msgs::msg::Bool::SharedPtr msg)
{
  const bool enabled = msg->data;
  if (enabled == enabled_) {
    return;
  }
  enabled_ = enabled;
  RCLCPP_INFO(
    get_logger(), "Cartesian target publication %s", enabled_ ? "enabled" : "disabled");
}

void CartesianPoseDeltaDecoderNode::tcp_offset_callback(
  const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
{
  const Vec3 offset{msg->vector.x, msg->vector.y, msg->vector.z};
  for (const auto value : offset) {
    if (!std::isfinite(value)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000, "Ignoring non-finite TCP offset update");
      return;
    }
  }
  if (offset == output_tcp_offset_) {
    return;
  }
  output_tcp_offset_ = offset;
  RCLCPP_INFO(
    get_logger(), "Using measured output_tcp_offset=[%.6f, %.6f, %.6f]",
    offset[0], offset[1], offset[2]);
}

void CartesianPoseDeltaDecoderNode::command_callback(
  const isaac_ros_deploy_interfaces::msg::CartesianPoseDeltaCommand::SharedPtr command)
{
  if (!enabled_) {
    return;
  }

  Vec3 target_pos;
  Quat target_quat;
  try {
    std::tie(target_pos, target_quat) = decode_cartesian_pose_delta(
      *command, action_scale_, output_tcp_offset_);
  } catch (const std::exception & exc) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", exc.what());
    return;
  }

  geometry_msgs::msg::PoseStamped target;
  target.header.stamp = get_clock()->now();
  target.header.frame_id = output_frame_id_.empty() ? command->header.frame_id : output_frame_id_;
  target.pose = arrays_to_pose(target_pos, target_quat);
  publisher_->publish(target);

  ++published_count_;
  if (published_count_ == 1 || published_count_ % 300 == 0) {
    RCLCPP_DEBUG(
      get_logger(),
      "Published Cartesian target pose count=%zu, frame=%s, "
      "position=[%.6f, %.6f, %.6f], quaternion_xyzw=[%.6f, %.6f, %.6f, %.6f]",
      published_count_, target.header.frame_id.c_str(),
      target_pos[0], target_pos[1], target_pos[2],
      target_quat[0], target_quat[1], target_quat[2], target_quat[3]);
  }
}

}  // namespace isaac_ros_deploy_converters

RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_deploy_converters::CartesianPoseDeltaDecoderNode)

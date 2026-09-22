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

#include "isaac_ros_deploy_converters/converters/message_to_tensor_converter.hpp"

#include <array>
#include <cmath>
#include <mutex>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "isaac_ros_deploy_interfaces/msg/body_command.hpp"
#include "isaac_ros_deploy_interfaces/msg/joint_command.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace isaac_ros_deploy_converters
{

namespace
{

constexpr std::array<const char *, 6> kRotation6DNames{
  "r00", "r01", "r02", "r10", "r11", "r12"};

std::array<double, 4> normalize_quat_xyzw(double x, double y, double z, double w)
{
  const double norm = std::sqrt(x * x + y * y + z * z + w * w);
  if (norm <= 0.0 || !std::isfinite(norm)) {
    return {0.0, 0.0, 0.0, 1.0};
  }
  return {x / norm, y / norm, z / norm, w / norm};
}

std::array<float, 6> quat_xyzw_to_rot6d_rows(double x, double y, double z, double w)
{
  const auto q = normalize_quat_xyzw(x, y, z, w);
  x = q[0];
  y = q[1];
  z = q[2];
  w = q[3];
  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;
  const double wx = w * x;
  const double wy = w * y;
  const double wz = w * z;
  return {
    static_cast<float>(1.0 - 2.0 * (yy + zz)),
    static_cast<float>(2.0 * (xy - wz)),
    static_cast<float>(2.0 * (xz + wy)),
    static_cast<float>(2.0 * (xy + wz)),
    static_cast<float>(1.0 - 2.0 * (xx + zz)),
    static_cast<float>(2.0 * (yz - wx)),
  };
}

isaac_deploy_core::TensorSpec rotation_6d_tensor_spec()
{
  return {.names = {{}, std::vector<std::string>(
      kRotation6DNames.begin(), kRotation6DNames.end())}};
}

// ============================================================================
// Shared deserialization helpers
// ============================================================================

struct DeserializedJointState
{
  std::vector<std::string> names;
  std::vector<double> position;
  std::vector<double> velocity;
};

DeserializedJointState deserialize_joint_state(
  const std::shared_ptr<rclcpp::SerializedMessage> & msg)
{
  sensor_msgs::msg::JointState joint_state;
  rclcpp::Serialization<sensor_msgs::msg::JointState> serializer;
  serializer.deserialize_message(msg.get(), &joint_state);
  return {
    std::move(joint_state.name),
    std::move(joint_state.position),
    std::move(joint_state.velocity)
  };
}

struct DeserializedImu
{
  double orientation_x, orientation_y, orientation_z, orientation_w;
  double angular_velocity_x, angular_velocity_y, angular_velocity_z;
  double linear_acceleration_x, linear_acceleration_y, linear_acceleration_z;
};

DeserializedImu deserialize_imu(const std::shared_ptr<rclcpp::SerializedMessage> & msg)
{
  sensor_msgs::msg::Imu imu;
  rclcpp::Serialization<sensor_msgs::msg::Imu> serializer;
  serializer.deserialize_message(msg.get(), &imu);
  return {
    imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w,
    imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
    imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z
  };
}

torch::Tensor doubles_to_tensor(const std::vector<double> & data)
{
  std::vector<float> float_data(data.begin(), data.end());
  return torch::tensor(float_data, torch::kFloat32).unsqueeze(0);  // [1, N]
}

// ============================================================================
// JointState converters
// ============================================================================

class JointPosConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "state/joint/position";}
  std::string get_message_type() const override {return "sensor_msgs/msg/JointState";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    auto js = deserialize_joint_state(msg);
    message_joint_names_ = std::move(js.names);
    return doubles_to_tensor(js.position);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    if (message_joint_names_.empty()) {return {};}
    return {.names = {{}, message_joint_names_}};
  }

private:
  std::vector<std::string> message_joint_names_;
};

class JointVelConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "state/joint/velocity";}
  std::string get_message_type() const override {return "sensor_msgs/msg/JointState";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    auto js = deserialize_joint_state(msg);
    message_joint_names_ = std::move(js.names);
    return doubles_to_tensor(js.velocity);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    if (message_joint_names_.empty()) {return {};}
    return {.names = {{}, message_joint_names_}};
  }

private:
  std::vector<std::string> message_joint_names_;
};

// ============================================================================
// IMU converters
// ============================================================================

class AnchorBodyRotConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "state/body/rotation";}
  std::string get_message_type() const override {return "sensor_msgs/msg/Imu";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    auto imu = deserialize_imu(msg);
    return torch::tensor(
      {{static_cast<float>(imu.orientation_x),
        static_cast<float>(imu.orientation_y),
        static_cast<float>(imu.orientation_z),
        static_cast<float>(imu.orientation_w)}},
      torch::kFloat32);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    return {.names = {{}, {"x", "y", "z", "w"}}};
  }
};

class RootBodyAngVelConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "state/body/angular_velocity";}
  std::string get_message_type() const override {return "sensor_msgs/msg/Imu";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    auto imu = deserialize_imu(msg);
    return torch::tensor(
      {{static_cast<float>(imu.angular_velocity_x),
        static_cast<float>(imu.angular_velocity_y),
        static_cast<float>(imu.angular_velocity_z)}},
      torch::kFloat32);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    return {.names = {{}, {"x", "y", "z"}}};
  }
};

class PoseStampedPositionConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "state/body/position";}
  std::string get_message_type() const override {return "geometry_msgs/msg/PoseStamped";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    geometry_msgs::msg::PoseStamped pose_stamped;
    rclcpp::Serialization<geometry_msgs::msg::PoseStamped> serializer;
    serializer.deserialize_message(msg.get(), &pose_stamped);

    const auto & pos = pose_stamped.pose.position;
    return torch::tensor(
      {{static_cast<float>(pos.x),
        static_cast<float>(pos.y),
        static_cast<float>(pos.z)}},
      torch::kFloat32);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    return {.names = {{}, {"x", "y", "z"}}};
  }
};

class PoseStampedRotationConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "state/body/rotation";}
  std::string get_message_type() const override {return "geometry_msgs/msg/PoseStamped";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    geometry_msgs::msg::PoseStamped pose_stamped;
    rclcpp::Serialization<geometry_msgs::msg::PoseStamped> serializer;
    serializer.deserialize_message(msg.get(), &pose_stamped);

    const auto & ori = pose_stamped.pose.orientation;
    return torch::tensor(
      {{static_cast<float>(ori.x),
        static_cast<float>(ori.y),
        static_cast<float>(ori.z),
        static_cast<float>(ori.w)}},
      torch::kFloat32);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    return {.names = {{}, {"qx", "qy", "qz", "qw"}}};
  }
};

class PoseStampedRotation6DConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "state/body/rotation_6d";}
  std::string get_message_type() const override {return "geometry_msgs/msg/PoseStamped";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    geometry_msgs::msg::PoseStamped pose_stamped;
    rclcpp::Serialization<geometry_msgs::msg::PoseStamped> serializer;
    serializer.deserialize_message(msg.get(), &pose_stamped);

    const auto & ori = pose_stamped.pose.orientation;
    const auto rot6d = quat_xyzw_to_rot6d_rows(ori.x, ori.y, ori.z, ori.w);
    return torch::tensor(
      {{rot6d[0], rot6d[1], rot6d[2], rot6d[3], rot6d[4], rot6d[5]}},
      torch::kFloat32);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    return rotation_6d_tensor_spec();
  }
};

// ============================================================================
// ReferenceMotion converters
// ============================================================================

class RefMotionBodyRotConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "command/body/rotation";}
  std::string get_message_type() const override
  {
    return "isaac_ros_deploy_interfaces/msg/BodyCommand";
  }

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    isaac_ros_deploy_interfaces::msg::BodyCommand body_command;
    rclcpp::Serialization<isaac_ros_deploy_interfaces::msg::BodyCommand> serializer;
    serializer.deserialize_message(msg.get(), &body_command);

    message_body_names_ = body_command.names;

    size_t num_bodies = body_command.pose.size();
    torch::Tensor tensor = torch::zeros(
      {1, static_cast<int64_t>(num_bodies), 4}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 3>();

    for (size_t i = 0; i < num_bodies; ++i) {
      const auto & pose = body_command.pose[i];
      accessor[0][static_cast<int64_t>(i)][0] = static_cast<float>(pose.orientation.x);
      accessor[0][static_cast<int64_t>(i)][1] = static_cast<float>(pose.orientation.y);
      accessor[0][static_cast<int64_t>(i)][2] = static_cast<float>(pose.orientation.z);
      accessor[0][static_cast<int64_t>(i)][3] = static_cast<float>(pose.orientation.w);
    }
    return tensor;
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    if (message_body_names_.empty()) {
      return {.names = {{}, {}, {"x", "y", "z", "w"}}};
    }
    return {.names = {{}, message_body_names_, {"x", "y", "z", "w"}}};
  }

private:
  std::vector<std::string> message_body_names_;
};

class RefMotionJointPosConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "command/joint/position";}
  std::string get_message_type() const override
  {
    return "isaac_ros_deploy_interfaces/msg/JointCommand";
  }

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    isaac_ros_deploy_interfaces::msg::JointCommand joint_command;
    rclcpp::Serialization<isaac_ros_deploy_interfaces::msg::JointCommand> serializer;
    serializer.deserialize_message(msg.get(), &joint_command);

    message_joint_names_ = joint_command.names;
    return doubles_to_tensor(joint_command.position);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    if (message_joint_names_.empty()) {return {};}
    return {.names = {{}, message_joint_names_}};
  }

private:
  std::vector<std::string> message_joint_names_;
};

class RefMotionJointVelConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "command/joint/velocity";}
  std::string get_message_type() const override
  {
    return "isaac_ros_deploy_interfaces/msg/JointCommand";
  }

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    isaac_ros_deploy_interfaces::msg::JointCommand joint_command;
    rclcpp::Serialization<isaac_ros_deploy_interfaces::msg::JointCommand> serializer;
    serializer.deserialize_message(msg.get(), &joint_command);

    message_joint_names_ = joint_command.names;
    return doubles_to_tensor(joint_command.velocity);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    if (message_joint_names_.empty()) {return {};}
    return {.names = {{}, message_joint_names_}};
  }

private:
  std::vector<std::string> message_joint_names_;
};

// ============================================================================
// Twist converter
// ============================================================================

class TwistConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "command/body/velocity";}
  std::string get_message_type() const override {return "geometry_msgs/msg/Twist";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    geometry_msgs::msg::Twist twist;
    rclcpp::Serialization<geometry_msgs::msg::Twist> serializer;
    serializer.deserialize_message(msg.get(), &twist);
    return torch::tensor(
      {{static_cast<float>(twist.linear.x),
        static_cast<float>(twist.linear.y),
        static_cast<float>(twist.linear.z),
        static_cast<float>(twist.angular.x),
        static_cast<float>(twist.angular.y),
        static_cast<float>(twist.angular.z)}},
      torch::kFloat32);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    return {.names = {{},
        {"lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"}}};
  }
};

class TwistStampedConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "command/body/velocity";}
  std::string get_message_type() const override {return "geometry_msgs/msg/TwistStamped";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    geometry_msgs::msg::TwistStamped twist_stamped;
    rclcpp::Serialization<geometry_msgs::msg::TwistStamped> serializer;
    serializer.deserialize_message(msg.get(), &twist_stamped);
    const auto & t = twist_stamped.twist;
    return torch::tensor(
      {{static_cast<float>(t.linear.x),
        static_cast<float>(t.linear.y),
        static_cast<float>(t.linear.z),
        static_cast<float>(t.angular.x),
        static_cast<float>(t.angular.y),
        static_cast<float>(t.angular.z)}},
      torch::kFloat32);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    return {.names = {{},
        {"lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"}}};
  }
};

/// Reads a Twist message and outputs [lin_vel_x, lin_vel_y, ang_vel_z, height].
class VelocityHeightCommandsConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "command/body/velocity_height";}
  std::string get_message_type() const override {return "geometry_msgs/msg/Twist";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    geometry_msgs::msg::Twist twist;
    rclcpp::Serialization<geometry_msgs::msg::Twist> serializer;
    serializer.deserialize_message(msg.get(), &twist);
    // TODO(lgulich): Height is hardcoded to 0.7, to clean this up we need
    // to decide what message to use to send the height command.
    constexpr float kDefaultHeight = 0.7f;
    return torch::tensor(
      {{static_cast<float>(twist.linear.x),
        static_cast<float>(twist.linear.y),
        static_cast<float>(twist.angular.z),
        kDefaultHeight}},
      torch::kFloat32);
  }
};

/// Reads an Odometry message and extracts root body linear velocity [x, y, z].
class RootLinVelConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "state/body/linear_velocity";}
  std::string get_message_type() const override {return "nav_msgs/msg/Odometry";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    nav_msgs::msg::Odometry odom;
    rclcpp::Serialization<nav_msgs::msg::Odometry> serializer;
    serializer.deserialize_message(msg.get(), &odom);
    return torch::tensor(
      {{static_cast<float>(odom.twist.twist.linear.x),
        static_cast<float>(odom.twist.twist.linear.y),
        static_cast<float>(odom.twist.twist.linear.z)}},
      torch::kFloat32);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    return {.names = {{}, {"x", "y", "z"}}};
  }
};

// ============================================================================
// Image converter
// ============================================================================

class ImageConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "state/camera/image";}
  std::string get_message_type() const override {return "sensor_msgs/msg/Image";}

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    sensor_msgs::msg::Image image;
    rclcpp::Serialization<sensor_msgs::msg::Image> serializer;
    serializer.deserialize_message(msg.get(), &image);

    const int64_t h = image.height;
    const int64_t w = image.width;

    const bool is_bgr = (image.encoding == "bgr8");
    const bool is_rgb = (image.encoding == "rgb8");
    if (!is_bgr && !is_rgb) {
      throw std::runtime_error(
        "ImageConverter only supports rgb8 and bgr8 encodings, got: " + image.encoding);
    }
    const int64_t channels = 3;

    // Convert uint8 pixel data to float32 tensor [1, H, W, C].
    auto tensor = torch::from_blob(
      image.data.data(), {h, w, channels}, torch::kUInt8).to(torch::kFloat32);

    // Convert BGR to RGB if needed.
    if (is_bgr) {
      tensor = tensor.index(
        {torch::indexing::Slice(), torch::indexing::Slice(),
          torch::tensor({2, 1, 0})});
    }

    return tensor.unsqueeze(0);  // [1, H, W, C]
  }
};

// ============================================================================
// Single-body command converters (BodyCommand with one entry)
// ============================================================================

class BodyCommandPositionConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "command/body/target_position";}
  std::string get_message_type() const override
  {
    return "isaac_ros_deploy_interfaces/msg/BodyCommand";
  }

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    isaac_ros_deploy_interfaces::msg::BodyCommand body_command;
    rclcpp::Serialization<isaac_ros_deploy_interfaces::msg::BodyCommand> serializer;
    serializer.deserialize_message(msg.get(), &body_command);

    if (body_command.pose.empty()) {
      throw std::runtime_error("BodyCommand message has no pose entries");
    }

    const auto & pos = body_command.pose[0].position;
    return torch::tensor(
      {{static_cast<float>(pos.x),
        static_cast<float>(pos.y),
        static_cast<float>(pos.z)}},
      torch::kFloat32);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    return {.names = {{}, {"x", "y", "z"}}};
  }
};

class BodyCommandRotationConverter : public MessageToTensorConverter
{
public:
  std::string get_kind() const override {return "command/body/target_rotation";}
  std::string get_message_type() const override
  {
    return "isaac_ros_deploy_interfaces/msg/BodyCommand";
  }

  torch::Tensor convert(const std::shared_ptr<rclcpp::SerializedMessage> & msg) override
  {
    isaac_ros_deploy_interfaces::msg::BodyCommand body_command;
    rclcpp::Serialization<isaac_ros_deploy_interfaces::msg::BodyCommand> serializer;
    serializer.deserialize_message(msg.get(), &body_command);

    if (body_command.pose.empty()) {
      throw std::runtime_error("BodyCommand message has no pose entries");
    }

    const auto & ori = body_command.pose[0].orientation;
    return torch::tensor(
      {{static_cast<float>(ori.x),
        static_cast<float>(ori.y),
        static_cast<float>(ori.z),
        static_cast<float>(ori.w)}},
      torch::kFloat32);
  }

  isaac_deploy_core::TensorSpec get_tensor_spec() const override
  {
    return {.names = {{}, {"x", "y", "z", "w"}}};
  }
};

}  // namespace

void initialize_input_converters()
{
  static std::once_flag flag;
  std::call_once(
    flag, []() {
      auto & registry = MessageToTensorConverterRegistry::instance();

      registry.register_converter(
        "state/joint/position", "sensor_msgs/msg/JointState",
        [](const std::string &) {
          return std::make_shared<JointPosConverter>();
        });
      registry.register_converter(
        "state/joint/velocity", "sensor_msgs/msg/JointState",
        [](const std::string &) {
          return std::make_shared<JointVelConverter>();
        });
      registry.register_converter(
        "state/body/rotation", "sensor_msgs/msg/Imu",
        [](const std::string &) {
          return std::make_shared<AnchorBodyRotConverter>();
        });
      registry.register_converter(
        "state/body/position", "geometry_msgs/msg/PoseStamped",
        [](const std::string &) {
          return std::make_shared<PoseStampedPositionConverter>();
        });
      registry.register_converter(
        "state/body/rotation", "geometry_msgs/msg/PoseStamped",
        [](const std::string &) {
          return std::make_shared<PoseStampedRotationConverter>();
        });
      registry.register_converter(
        "state/body/rotation_6d", "geometry_msgs/msg/PoseStamped",
        [](const std::string &) {
          return std::make_shared<PoseStampedRotation6DConverter>();
        });
      registry.register_converter(
        "state/body/angular_velocity", "sensor_msgs/msg/Imu",
        [](const std::string &) {
          return std::make_shared<RootBodyAngVelConverter>();
        });
      registry.register_converter(
        "command/body/rotation", "isaac_ros_deploy_interfaces/msg/BodyCommand",
        [](const std::string &) {
          return std::make_shared<RefMotionBodyRotConverter>();
        });
      registry.register_converter(
        "command/joint/position", "isaac_ros_deploy_interfaces/msg/JointCommand",
        [](const std::string &) {
          return std::make_shared<RefMotionJointPosConverter>();
        });
      registry.register_converter(
        "command/joint/velocity", "isaac_ros_deploy_interfaces/msg/JointCommand",
        [](const std::string &) {
          return std::make_shared<RefMotionJointVelConverter>();
        });
      // Default for velocity_command: Twist (registered first).
      registry.register_converter(
        "command/body/velocity", "geometry_msgs/msg/Twist",
        [](const std::string &) {
          return std::make_shared<TwistConverter>();
        });
      // Alternative for velocity_command: TwistStamped.
      registry.register_converter(
        "command/body/velocity", "geometry_msgs/msg/TwistStamped",
        [](const std::string &) {
          return std::make_shared<TwistStampedConverter>();
        });
      registry.register_converter(
        "command/body/velocity_height", "geometry_msgs/msg/Twist",
        [](const std::string &) {
          return std::make_shared<VelocityHeightCommandsConverter>();
        });
      registry.register_converter(
        "state/body/linear_velocity", "nav_msgs/msg/Odometry",
        [](const std::string &) {
          return std::make_shared<RootLinVelConverter>();
        });
      registry.register_converter(
        "state/camera/image", "sensor_msgs/msg/Image",
        [](const std::string &) {
          return std::make_shared<ImageConverter>();
        });
      registry.register_converter(
        "command/body/target_position", "isaac_ros_deploy_interfaces/msg/BodyCommand",
        [](const std::string &) {
          return std::make_shared<BodyCommandPositionConverter>();
        });
      registry.register_converter(
        "command/body/target_rotation", "isaac_ros_deploy_interfaces/msg/BodyCommand",
        [](const std::string &) {
          return std::make_shared<BodyCommandRotationConverter>();
        });
    });
}

}  // namespace isaac_ros_deploy_converters

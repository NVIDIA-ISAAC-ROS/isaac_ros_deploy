// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <mutex>

#include "isaac_ros_deploy_converters/converters/tensor_to_message_converter.hpp"
#include "isaac_ros_deploy_interfaces/msg/body_command.hpp"
#include "isaac_ros_deploy_interfaces/msg/joint_command.hpp"

namespace isaac_ros_deploy_converters
{

namespace
{

// ============================================================================
// Shared helpers
// ============================================================================

void write_stamp(std_msgs::msg::Header & header, int64_t timestamp_ns)
{
  header.stamp.sec = static_cast<int32_t>(timestamp_ns / 1000000000);
  header.stamp.nanosec = static_cast<uint32_t>(timestamp_ns % 1000000000);
}

void write_tensor_to_doubles(
  std::vector<double> & out,
  const torch::Tensor & tensor,
  size_t count)
{
  auto flat = tensor.contiguous().cpu().view({-1});
  auto accessor = flat.accessor<float, 1>();
  int64_t flat_size = flat.size(0);
  out.resize(count, 0.0);
  for (size_t i = 0; i < count && static_cast<int64_t>(i) < flat_size; ++i) {
    out[i] = static_cast<double>(accessor[static_cast<int64_t>(i)]);
  }
}

// ============================================================================
// JointCommand per-kind converter
// ============================================================================

/// Pointer-to-member selecting which JointCommand field to write into.
using JointCommandField =
  std::vector<double> isaac_ros_deploy_interfaces::msg::JointCommand::*;

/// Generic converter for any JointCommand field. Each instance handles one kind
/// (e.g., "joint_pos_targets") and writes into one field (e.g., &JointCommand::position).
class JointCommandConverter
  : public TypedOutputConverter<isaac_ros_deploy_interfaces::msg::JointCommand>
{
public:
  JointCommandConverter(std::string kind, JointCommandField field)
  : kind_(std::move(kind)), field_(field) {}

  std::string get_kind() const override {return kind_;}
  std::string get_message_type() const override
  {
    return "isaac_ros_deploy_interfaces/msg/JointCommand";
  }

  void write(
    const torch::Tensor & tensor,
    const std::vector<std::string> & element_names,
    const std::shared_ptr<void> & msg,
    int64_t timestamp_ns) const override
  {
    auto & m = *static_cast<isaac_ros_deploy_interfaces::msg::JointCommand *>(msg.get());
    write_stamp(m.header, timestamp_ns);
    m.names = element_names;
    write_tensor_to_doubles(m.*field_, tensor, element_names.size());
  }

private:
  std::string kind_;
  JointCommandField field_;
};

// ============================================================================
// BodyCommand per-kind converters
// ============================================================================

class BodyRotTargetConverter
  : public TypedOutputConverter<isaac_ros_deploy_interfaces::msg::BodyCommand>
{
public:
  std::string get_kind() const override {return "body_rot_target";}
  std::string get_message_type() const override
  {
    return "isaac_ros_deploy_interfaces/msg/BodyCommand";
  }

  void write(
    const torch::Tensor & tensor,
    const std::vector<std::string> & element_names,
    const std::shared_ptr<void> & msg,
    int64_t timestamp_ns) const override
  {
    auto & m = *static_cast<isaac_ros_deploy_interfaces::msg::BodyCommand *>(msg.get());
    write_stamp(m.header, timestamp_ns);
    m.names = element_names;

    size_t num_bodies = element_names.empty() ? 1 : element_names.size();
    m.pose.resize(num_bodies);

    auto flat = tensor.contiguous().cpu().view({-1});
    auto accessor = flat.accessor<float, 1>();
    int64_t flat_size = flat.size(0);

    for (size_t i = 0; i < num_bodies; ++i) {
      int64_t offset = static_cast<int64_t>(i * 4);
      if (offset + 3 < flat_size) {
        m.pose[i].orientation.x = static_cast<double>(accessor[offset + 0]);
        m.pose[i].orientation.y = static_cast<double>(accessor[offset + 1]);
        m.pose[i].orientation.z = static_cast<double>(accessor[offset + 2]);
        m.pose[i].orientation.w = static_cast<double>(accessor[offset + 3]);
      }
    }
  }
};

class BodyAngVelTargetConverter
  : public TypedOutputConverter<isaac_ros_deploy_interfaces::msg::BodyCommand>
{
public:
  std::string get_kind() const override {return "body_ang_vel_target";}
  std::string get_message_type() const override
  {
    return "isaac_ros_deploy_interfaces/msg/BodyCommand";
  }

  void write(
    const torch::Tensor & tensor,
    const std::vector<std::string> & element_names,
    const std::shared_ptr<void> & msg,
    int64_t timestamp_ns) const override
  {
    auto & m = *static_cast<isaac_ros_deploy_interfaces::msg::BodyCommand *>(msg.get());
    write_stamp(m.header, timestamp_ns);
    m.names = element_names;

    size_t num_bodies = element_names.empty() ? 1 : element_names.size();
    m.twist.resize(num_bodies);

    auto flat = tensor.contiguous().cpu().view({-1});
    auto accessor = flat.accessor<float, 1>();
    int64_t flat_size = flat.size(0);

    for (size_t i = 0; i < num_bodies; ++i) {
      int64_t offset = static_cast<int64_t>(i * 3);
      if (offset + 2 < flat_size) {
        m.twist[i].angular.x = static_cast<double>(accessor[offset + 0]);
        m.twist[i].angular.y = static_cast<double>(accessor[offset + 1]);
        m.twist[i].angular.z = static_cast<double>(accessor[offset + 2]);
      }
    }
  }
};

}  // namespace

void initialize_output_converters()
{
  static std::once_flag flag;
  std::call_once(flag, []() {

  auto & registry = OutputConverterRegistry::instance();

  using JC = isaac_ros_deploy_interfaces::msg::JointCommand;

  // JointCommand output converters.
  registry.register_converter(
    "joint_pos_targets", []() {
      return std::make_shared<JointCommandConverter>("joint_pos_targets", &JC::position);
    });
  registry.register_converter(
    "target/joint/position", []() {
      return std::make_shared<JointCommandConverter>("target/joint/position", &JC::position);
    });
  registry.register_converter(
    "joint_vel_targets", []() {
      return std::make_shared<JointCommandConverter>("joint_vel_targets", &JC::velocity);
    });
  registry.register_converter(
    "actions", []() {
      return std::make_shared<JointCommandConverter>("actions", &JC::position);
    });
  registry.register_converter(
    "stiffness_targets", []() {
      return std::make_shared<JointCommandConverter>("stiffness_targets", &JC::kp);
    });
  registry.register_converter(
    "kp", []() {
      return std::make_shared<JointCommandConverter>("kp", &JC::kp);
    });
  registry.register_converter(
    "damping_targets", []() {
      return std::make_shared<JointCommandConverter>("damping_targets", &JC::kd);
    });
  registry.register_converter(
    "kd", []() {
      return std::make_shared<JointCommandConverter>("kd", &JC::kd);
    });

  // BodyCommand output converters.
  registry.register_converter(
    "body_rot_target", []() {
      return std::make_shared<BodyRotTargetConverter>();
    });
  registry.register_converter(
    "body_ang_vel_target", []() {
      return std::make_shared<BodyAngVelTargetConverter>();
    });
  });
}

}  // namespace isaac_ros_deploy_converters

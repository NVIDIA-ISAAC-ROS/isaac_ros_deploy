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

#include <algorithm>
#include <limits>
#include <mutex>

#include "isaac_ros_deploy_converters/converters/tensor_to_message_converter.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "isaac_ros_deploy_interfaces/msg/body_command.hpp"
#include "isaac_ros_deploy_interfaces/msg/joint_command.hpp"
#include "isaac_ros_deploy_interfaces/msg/joint_command_trajectory.hpp"

namespace isaac_ros_deploy_converters
{

namespace
{

// ============================================================================
// Shared helpers
// ============================================================================

void validate_float32_dtype(const torch::Tensor & tensor, const char * context)
{
  if (tensor.scalar_type() != torch::kFloat32) {
    throw std::runtime_error(
      std::string(context) + ": expected tensor dtype float32, got " +
      std::string(torch::toString(tensor.scalar_type())));
  }
}

void write_stamp(std_msgs::msg::Header & header, int64_t timestamp_ns)
{
  header.stamp.sec = static_cast<int32_t>(timestamp_ns / 1000000000);
  header.stamp.nanosec = static_cast<uint32_t>(timestamp_ns % 1000000000);
}

void append_tensor_to_doubles(
  std::vector<double> & out,
  const torch::Tensor & tensor,
  size_t count)
{
  const auto flat = tensor.contiguous().cpu().view({-1});
  validate_float32_dtype(flat, "append_tensor_to_doubles");
  const auto accessor = flat.accessor<float, 1>();
  const int64_t flat_size = flat.size(0);
  const size_t start = out.size();
  out.resize(start + count, 0.0);
  for (size_t i = 0; i < count && static_cast<int64_t>(i) < flat_size; ++i) {
    out[start + i] = static_cast<double>(accessor[static_cast<int64_t>(i)]);
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
    append_tensor_to_doubles(m.*field_, tensor, element_names.size());
    // Keep `names` in sync with the longest populated field.  Two patterns:
    // (1) Converters write different joint *subsets* to the same field
    //     (e.g. left_arm + right_arm -> position): names must grow so
    //     downstream consumers can look up joints by name.
    // (2) Converters write the *same joint set* to different fields
    //     (e.g. joint_pos_targets -> position, joint_vel_targets ->
    //     velocity): names must not duplicate.
    // Extending `names` only when the just-written field overshoots the
    // current `names` length handles both cases.
    if ((m.*field_).size() > m.names.size()) {
      m.names.insert(
        m.names.end(), element_names.begin(), element_names.end());
    }
  }

private:
  std::string kind_;
  JointCommandField field_;
};

// ============================================================================
// JointCommandTrajectory per-kind converter
// ============================================================================

/// Pointer-to-member selecting which JointCommandTrajectory field to write into.
using JointCommandTrajectoryField =
  std::vector<double> isaac_ros_deploy_interfaces::msg::JointCommandTrajectory::*;

/// Chunked sibling of JointCommandConverter.  For 3D tensors [1, H, N];
/// writes H*N floats row-major into one field of a JointCommandTrajectory.
/// Multiple instances (position + effort + kp + kd) may write into the same
/// message, populating different fields, while sharing `names` and `horizon`.
class JointCommandTrajectoryConverter
  : public TypedOutputConverter<
    isaac_ros_deploy_interfaces::msg::JointCommandTrajectory>
{
public:
  JointCommandTrajectoryConverter(
    std::string kind, JointCommandTrajectoryField field)
  : kind_(std::move(kind)), field_(field) {}

  std::string get_kind() const override {return kind_;}
  std::string get_message_type() const override
  {
    return "isaac_ros_deploy_interfaces/msg/JointCommandTrajectory";
  }

  void write(
    const torch::Tensor & tensor,
    const std::vector<std::string> & element_names,
    const std::shared_ptr<void> & msg,
    int64_t timestamp_ns) const override
  {
    using JCT = isaac_ros_deploy_interfaces::msg::JointCommandTrajectory;
    auto & m = *static_cast<JCT *>(msg.get());
    write_stamp(m.header, timestamp_ns);

    // Expect [1, H, N] with H >= 0, N == element_names.size(); reject
    // negative dims so the size_t casts below don't underflow.
    const auto sizes = tensor.sizes();
    if (sizes.size() != 3 || sizes[0] != 1 ||
      sizes[1] < 0 || sizes[2] < 0 ||
      static_cast<std::size_t>(sizes[2]) != element_names.size())
    {
      throw std::runtime_error(
        "JointCommandTrajectoryConverter expects [1, H, N] tensor with "
        "N == element_names.size()");
    }
    const std::size_t horizon = static_cast<std::size_t>(sizes[1]);
    const std::size_t n_new = element_names.size();

    // Set `horizon` on first write; validate on subsequent writes.
    if (m.horizon == 0) {
      m.horizon = static_cast<uint32_t>(horizon);
    } else if (m.horizon != horizon) {
      throw std::runtime_error(
        "JointCommandTrajectory horizon mismatch across converters");
    }

    auto flat = tensor.contiguous().cpu().view({-1});
    validate_float32_dtype(flat, "JointCommandTrajectoryConverter");
    const auto accessor = flat.accessor<float, 1>();

    auto & target = m.*field_;
    const std::size_t n_existing = m.names.size();

    // Case A — first write into this message.  Establish `names` and
    // populate the target field in a contiguous row-major copy.
    if (n_existing == 0) {
      m.names = element_names;
      target.resize(horizon * n_new);
      for (std::size_t i = 0; i < horizon * n_new; ++i) {
        target[i] = static_cast<double>(accessor[static_cast<int64_t>(i)]);
      }
      return;
    }

    // Classify element_names against m.names: each incoming name is
    // either already present (in_count) or new (out_count).  The four
    // cases then split by (in_count, out_count):
    //   (n_new, 0)   -> Case B: subset (includes exact match).
    //   (0, n_new)   -> Case C: fully disjoint — extend m.names.
    //   (>0, >0)     -> genuine partial overlap — still a bug.
    std::vector<std::size_t> col_indices;
    col_indices.reserve(element_names.size());
    std::size_t in_count = 0;
    std::size_t out_count = 0;
    for (const auto & name : element_names) {
      auto it = std::find(m.names.begin(), m.names.end(), name);
      if (it != m.names.end()) {
        col_indices.push_back(
          static_cast<std::size_t>(std::distance(m.names.begin(), it)));
        ++in_count;
      } else {
        col_indices.push_back(std::numeric_limits<std::size_t>::max());
        ++out_count;
      }
    }

    if (in_count > 0 && out_count > 0) {
      throw std::runtime_error(
        "JointCommandTrajectory names partial overlap across converters");
    }

    // Case B — all element_names are already in m.names (subset or exact
    // match).  Scatter-write into the target field at the matching
    // columns.  Lazily allocates the target field (first write to this
    // field) and zero-pads any columns not covered by this converter.
    if (out_count == 0) {
      target.resize(horizon * n_existing, 0.0);
      for (std::size_t r = 0; r < horizon; ++r) {
        for (std::size_t c = 0; c < n_new; ++c) {
          target[r * n_existing + col_indices[c]] =
            static_cast<double>(accessor[static_cast<int64_t>(r * n_new + c)]);
        }
      }
      return;
    }

    // Case C — all element_names are disjoint from m.names.  Another
    // converter is writing to the same message but for a different group
    // of joints (e.g. GR00T's per-group decode_action outputs).

    const std::size_t n_total = n_existing + n_new;

    // Grow every already-populated field from [H, n_existing] to
    // [H, n_total] row-major, preserving existing columns in the first
    // `n_existing` positions of each row and zero-filling the new columns.
    // For the field being written, also populate the new `n_new` columns
    // from the incoming tensor.
    auto grow_field = [&](std::vector<double> & f, bool is_target) {
        if (f.empty() && !is_target) {
        // Field not populated; leave empty.
          return;
        }
        std::vector<double> expanded(horizon * n_total, 0.0);
        for (std::size_t r = 0; r < horizon; ++r) {
          if (!f.empty()) {
            std::copy(
            f.begin() + r * n_existing,
            f.begin() + (r + 1) * n_existing,
            expanded.begin() + r * n_total);
          }
          if (is_target) {
            for (std::size_t c = 0; c < n_new; ++c) {
              expanded[r * n_total + n_existing + c] =
                static_cast<double>(accessor[static_cast<int64_t>(r * n_new + c)]);
            }
          }
        }
        f = std::move(expanded);
      };

    grow_field(m.position, field_ == &JCT::position);
    grow_field(m.velocity, field_ == &JCT::velocity);
    grow_field(m.effort, field_ == &JCT::effort);
    grow_field(m.kp, field_ == &JCT::kp);
    grow_field(m.kd, field_ == &JCT::kd);

    m.names.insert(m.names.end(), element_names.begin(), element_names.end());
  }

private:
  std::string kind_;
  JointCommandTrajectoryField field_;
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
    // Multiple BodyCommand converters (rot + ang_vel) may write into the
    // same message. Only set `names` on the first write so subsequent
    // converters don't duplicate entries.
    if (m.names.empty()) {
      m.names = element_names;
    }

    size_t num_bodies = element_names.empty() ? 1 : element_names.size();
    m.pose.resize(num_bodies);

    const auto flat = tensor.contiguous().cpu().view({-1});
    validate_float32_dtype(flat, "BodyRotTargetConverter");
    const auto accessor = flat.accessor<float, 1>();
    const int64_t flat_size = flat.size(0);

    for (size_t i = 0; i < num_bodies; ++i) {
      const int64_t offset = static_cast<int64_t>(i * 4);
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
    // Multiple BodyCommand converters (rot + ang_vel) may write into the
    // same message. Only set `names` on the first write so subsequent
    // converters don't duplicate entries.
    if (m.names.empty()) {
      m.names = element_names;
    }

    size_t num_bodies = element_names.empty() ? 1 : element_names.size();
    m.twist.resize(num_bodies);

    const auto flat = tensor.contiguous().cpu().view({-1});
    validate_float32_dtype(flat, "BodyAngVelTargetConverter");
    const auto accessor = flat.accessor<float, 1>();
    const int64_t flat_size = flat.size(0);

    for (size_t i = 0; i < num_bodies; ++i) {
      const int64_t offset = static_cast<int64_t>(i * 3);
      if (offset + 2 < flat_size) {
        m.twist[i].angular.x = static_cast<double>(accessor[offset + 0]);
        m.twist[i].angular.y = static_cast<double>(accessor[offset + 1]);
        m.twist[i].angular.z = static_cast<double>(accessor[offset + 2]);
      }
    }
  }
};

// ============================================================================
// VelocityCommand converter (velocity_command → geometry_msgs/Twist)
// ============================================================================

/// Converts a [lin_x, lin_y, ang_z] tensor to a Twist message.
class VelocityCommandConverter
  : public TypedOutputConverter<geometry_msgs::msg::Twist>
{
public:
  std::string get_kind() const override {return "velocity_command";}
  std::string get_message_type() const override
  {
    return "geometry_msgs/msg/Twist";
  }

  void write(
    const torch::Tensor & tensor,
    const std::vector<std::string> &,
    const std::shared_ptr<void> & msg,
    int64_t) const override
  {
    auto & m = *static_cast<geometry_msgs::msg::Twist *>(msg.get());
    const auto flat = tensor.contiguous().cpu().view({-1});
    validate_float32_dtype(flat, "VelocityCommandConverter");
    const auto accessor = flat.accessor<float, 1>();
    const int64_t n = flat.size(0);

    if (n >= 1) {m.linear.x = static_cast<double>(accessor[0]);}
    if (n >= 2) {m.linear.y = static_cast<double>(accessor[1]);}
    if (n >= 3) {m.angular.z = static_cast<double>(accessor[2]);}
  }
};

}  // namespace

void initialize_output_converters()
{
  static std::once_flag flag;
  std::call_once(flag, []() {
      auto & registry = OutputConverterRegistry::instance();

      using JCT = isaac_ros_deploy_interfaces::msg::JointCommandTrajectory;
      using JC = isaac_ros_deploy_interfaces::msg::JointCommand;

  // JointCommand output converters.  Shape-dispatched: 3D tensors [1, H, N]
  // route to JointCommandTrajectoryConverter; 2D tensors fall back to the
  // single-step JointCommandConverter.
      registry.register_converter(
    "joint_pos_targets",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {
            return std::make_shared<JointCommandTrajectoryConverter>(
              "joint_pos_targets", &JCT::position);
          }
          return std::make_shared<JointCommandConverter>("joint_pos_targets", &JC::position);
    });
      registry.register_converter(
    "target/joint/position",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {
            return std::make_shared<JointCommandTrajectoryConverter>(
              "target/joint/position", &JCT::position);
          }
          return std::make_shared<JointCommandConverter>("target/joint/position", &JC::position);
    });
      registry.register_converter(
    "joint_vel_targets",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {
            return std::make_shared<JointCommandTrajectoryConverter>(
              "joint_vel_targets", &JCT::velocity);
          }
          return std::make_shared<JointCommandConverter>("joint_vel_targets", &JC::velocity);
    });
      registry.register_converter(
    "actions",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {
            return std::make_shared<JointCommandTrajectoryConverter>(
              "actions", &JCT::position);
          }
          return std::make_shared<JointCommandConverter>("actions", &JC::position);
    });
      registry.register_converter(
    "stiffness_targets",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {
            return std::make_shared<JointCommandTrajectoryConverter>(
              "stiffness_targets", &JCT::kp);
          }
          return std::make_shared<JointCommandConverter>("stiffness_targets", &JC::kp);
    });
      registry.register_converter(
    "kp",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {
            return std::make_shared<JointCommandTrajectoryConverter>("kp", &JCT::kp);
          }
          return std::make_shared<JointCommandConverter>("kp", &JC::kp);
    });
      registry.register_converter(
    "damping_targets",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {
            return std::make_shared<JointCommandTrajectoryConverter>(
              "damping_targets", &JCT::kd);
          }
          return std::make_shared<JointCommandConverter>("damping_targets", &JC::kd);
    });
      registry.register_converter(
    "kd",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {
            return std::make_shared<JointCommandTrajectoryConverter>("kd", &JCT::kd);
          }
          return std::make_shared<JointCommandConverter>("kd", &JC::kd);
    });
      registry.register_converter(
    "joint_effort_targets",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {
            return std::make_shared<JointCommandTrajectoryConverter>(
              "joint_effort_targets", &JCT::effort);
          }
          return std::make_shared<JointCommandConverter>("joint_effort_targets", &JC::effort);
    });
      registry.register_converter(
    "target/joint/effort",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {
            return std::make_shared<JointCommandTrajectoryConverter>(
              "target/joint/effort", &JCT::effort);
          }
          return std::make_shared<JointCommandConverter>("target/joint/effort", &JC::effort);
    });

  // BodyCommand output converters.  No chunk analog today; return nullptr
  // for 3D shapes so OutputBuilder's filter logs a warning.
      registry.register_converter(
    "body_rot_target",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {return nullptr;}
          return std::make_shared<BodyRotTargetConverter>();
    });
      registry.register_converter(
    "body_ang_vel_target",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          if (shape.size() == 3) {return nullptr;}
          return std::make_shared<BodyAngVelTargetConverter>();
    });

  // VelocityCommand output converter (velocity_command → Twist).  Handles both
  // 2D [1, 3] and 3D [1, H, 3] tensor shapes; for 3D chunks, flatten picks up
  // step 0's (lin_x, lin_y, ang_z) as the single-step target.  Full chunk
  // playback for the base-velocity channel is a future task.
      registry.register_converter(
    "velocity_command",
        [](const std::vector<int64_t> & shape) -> std::shared_ptr<TensorToMessageConverter> {
          (void)shape;
          return std::make_shared<VelocityCommandConverter>();
    });
  });
}

}  // namespace isaac_ros_deploy_converters

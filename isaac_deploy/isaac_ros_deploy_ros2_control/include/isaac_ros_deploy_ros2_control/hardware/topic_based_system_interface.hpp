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

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/version.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

// Check ROS distro for API compatibility.
#define ROS_DISTRO_HUMBLE (HARDWARE_INTERFACE_VERSION_MAJOR < 3)

namespace isaac_ros_deploy_ros2_control
{
namespace hardware
{

/// Hardware system interface that exchanges joint targets with Isaac Sim over
/// ROS 2 topics.
///
/// Subscribes to sensor_msgs/JointState for joint state feedback and
/// sensor_msgs/Imu for IMU data. Publishes sensor_msgs/JointState with
/// per-joint position + velocity targets on the configured commands topic.
///
/// The actuator pipeline (PD / DC-motor clamping / delay) runs inside Isaac
/// Sim via the `isaacsim.core.experimental.actuators` extension, which is
/// built and attached to the articulation by a separate Python launcher.
/// That extension consumes the position+velocity targets published here and
/// drives the articulation at physics rate on either the PhysX or Newton
/// backend - no DDS round-trip in the inner control loop, no effort
/// computation in ROS for this hardware path.
///
/// A second JointState publisher carries live kp/kd gains on the
/// `joint_gains_topic` parameter so the policy's per-step gain output reaches
/// the Isaac Sim actuator pipeline (the launcher subscribes and writes the
/// values into the Newton ControllerPD's kp/kd warp arrays). Convention on
/// that topic:
///   - `name[i]`     = joint name
///   - `position[i]` = kp for joint i
///   - `velocity[i]` = kd for joint i
/// NaN entries mean "no gain authored this cycle, leave previous value".
///
/// The `effort` field of the published command JointState is left unused.
///
/// URDF configuration example:
///   <ros2_control name="IsaacSimSystem" type="system">
///     <hardware>
///       <plugin>isaac_ros_deploy_ros2_control/TopicBasedSystemInterface</plugin>
///       <param name="joint_states_topic">/isaac_joint_states</param>
///       <param name="joint_commands_topic">/isaac_joint_commands</param>
///       <param name="joint_gains_topic">/isaac_sim_drive_gains</param>
///       <param name="imu_topic">/isaac_imu</param>
///     </hardware>
///     <joint name="..."> ... </joint>
///     <sensor name="imu"> ... </sensor>
///   </ros2_control>
class TopicBasedSystemInterface : public hardware_interface::SystemInterface
{
public:
  TopicBasedSystemInterface();
  ~TopicBasedSystemInterface() override;

#if ROS_DISTRO_HUMBLE
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;
#else
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;
#endif

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

#if ROS_DISTRO_HUMBLE
  const hardware_interface::HardwareInfo & get_hardware_info() const
  {
    return info_;
  }
#endif

private:
  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

  // Configuration: the topic names read from the URDF hardware parameters.
  struct Topics
  {
    std::string joint_states;
    std::string joint_commands;
    std::string joint_gains;
    std::string imu;
  };
  Topics topics_;

  // Transparent hasher so the string-keyed maps below can be looked up with a
  // std::string_view (e.g. the interface names handed to us by ros2_control)
  // without forcing a std::string allocation on every find().
  struct StringHash
  {
    using is_transparent = void;
    [[nodiscard]] size_t operator()(std::string_view sv) const noexcept
    {
      return std::hash<std::string_view>{}(sv);
    }
  };
  template<typename Value>
  using StringMap = std::unordered_map<std::string, Value, StringHash, std::equal_to<>>;

  // Joint name to index mapping (built from URDF joint order).
  StringMap<size_t> joint_name_to_index_;

  // Joint state storage (backing state interfaces, written only in read()).
  std::vector<double> joint_state_positions_;
  std::vector<double> joint_state_velocities_;
  std::vector<double> joint_state_efforts_;

  // Joint command storage (backing command interfaces, written by controllers).
  // effort/kp/kd are exported for compatibility with the controller stack
  // (forward_joint_command_controller, safety_controller); on the Isaac Sim
  // path they are not forwarded - only position and velocity targets are
  // published.
  struct JointCommands
  {
    std::vector<double> positions;
    std::vector<double> velocities;
    std::vector<double> efforts;
    std::vector<double> kp;
    std::vector<double> kd;
  };
  JointCommands joint_cmd_;

  // IMU state storage (backing state interfaces, written only in read()).
  // Order: orientation(x,y,z,w), angular_velocity(x,y,z), linear_acceleration(x,y,z)
  static constexpr size_t kImuStateCount{10};
  std::array<double, kImuStateCount> imu_state_{};
  StringMap<size_t> imu_name_to_index_;

  // Buffered sensor data from subscription callbacks (lock-free via RealtimeBuffer).
  struct JointStateData
  {
    std::vector<double> positions;
    std::vector<double> velocities;
    std::vector<double> efforts;
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  };
  realtime_tools::RealtimeBuffer<JointStateData> joint_state_buf_;

  struct ImuData
  {
    std::array<double, kImuStateCount> values{};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  };
  realtime_tools::RealtimeBuffer<ImuData> imu_buf_;

  // Pre-allocated command message: per-joint position + velocity targets.
  // The effort field is left unused; the Isaac Sim actuators extension runs
  // the actuator pipeline at physics rate.
  sensor_msgs::msg::JointState joint_cmd_msg_;

  // Pre-allocated gains message: per-joint kp + kd carried in the position
  // and velocity slots respectively (see class docstring).
  sensor_msgs::msg::JointState joint_gains_msg_;

  // ROS node and subscriptions/publisher (runs in separate thread).
  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  // std::jthread auto-joins on destruction and carries a stop token, so the
  // spin loop exits without a separate atomic flag.
  std::jthread executor_thread_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_gains_pub_;
};

}  // namespace hardware
}  // namespace isaac_ros_deploy_ros2_control

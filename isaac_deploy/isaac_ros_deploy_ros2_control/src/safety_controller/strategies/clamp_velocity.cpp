// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/safety_controller/strategies/clamp_velocity.hpp"

#include <utility>

namespace isaac_deploy_core {

  ClampVelocity::ClampVelocity(torch::Tensor max_velocities)
  : max_velocities_(std::move(max_velocities)) {}

  expected < std::unique_ptr <
  ClampVelocity >> ClampVelocity::create(const ClampVelocityConfig & config) {
    if (config.max_velocities.empty()) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "max_velocities cannot be empty"));
    }

    auto max_velocities = torch::tensor(config.max_velocities, torch::kFloat32);
    return std::unique_ptr < ClampVelocity > (new ClampVelocity(std::move(max_velocities)));
  }

  expected < torch::Tensor > ClampVelocity::apply(
    const torch::Tensor & command_positions,
    const torch::Tensor & current_positions,
    double blend_ratio, double dt) {
    // Compute maximum allowed delta per joint.
    auto max_delta = max_velocities_ * dt;

    // Clamp the commanded positions to be within max_delta of current.
    auto delta = command_positions - current_positions;
    auto clamped_delta = torch::clamp(delta, -max_delta, max_delta);
    auto clamped_positions = current_positions + clamped_delta;

    // Interpolate between current and clamped positions based on blend ratio.
    auto output = current_positions + (clamped_positions - current_positions) * blend_ratio;

    return output;
  }

  void ClampVelocity::reset()
  {
    // No internal state to reset.
  }

}  // namespace isaac_deploy_core

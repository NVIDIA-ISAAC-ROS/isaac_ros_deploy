// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_deploy_ros2_control/safety_controller/safety_strategy.hpp"

namespace isaac_deploy_core {

/// Configuration for ClampVelocity strategy.
  struct ClampVelocityConfig
  {
    /// Maximum velocities per joint (rad/s or m/s).
    std::vector < double > max_velocities;
  };

/// ClampVelocity safety strategy.
///
/// Limits position change per timestep based on maximum velocity:
///   max_delta = max_velocity * dt
///   clamped_pos = clamp(target_pos, current_pos - max_delta, current_pos + max_delta)
///   output_pos = lerp(current_pos, clamped_pos, blend_ratio)
  class ClampVelocity: public SafetyStrategy {
public:
    /// Create a ClampVelocity strategy from configuration.
    static expected < std::unique_ptr < ClampVelocity >> create(const ClampVelocityConfig & config);

    expected < torch::Tensor > apply(
      const torch::Tensor & command_positions,
      const torch::Tensor & current_positions, double blend_ratio,
      double dt) override;

    void reset() override;

    std::string name() const override {return "ClampVelocity";}

private:
    explicit ClampVelocity(torch::Tensor max_velocities);

    torch::Tensor max_velocities_;
  };

}  // namespace isaac_deploy_core

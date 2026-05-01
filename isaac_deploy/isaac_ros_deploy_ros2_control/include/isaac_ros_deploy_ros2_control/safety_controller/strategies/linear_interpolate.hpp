// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include "isaac_ros_deploy_ros2_control/safety_controller/safety_strategy.hpp"

namespace isaac_deploy_core {

/// Configuration for LinearInterpolate strategy.
  struct LinearInterpolateConfig
  {
    // No additional configuration needed.
  };

/// LinearInterpolate safety strategy.
///
/// Linearly interpolates between the position when the controller was activated
/// and the target position:
///   output_pos = lerp(activation_pos, target_pos, blend_ratio)
  class LinearInterpolate: public SafetyStrategy {
public:
    /// Create a LinearInterpolate strategy.
    static expected < std::unique_ptr < LinearInterpolate >> create(
      const LinearInterpolateConfig & config);

    expected < torch::Tensor > apply(
      const torch::Tensor & command_positions,
      const torch::Tensor & current_positions, double blend_ratio,
      double dt) override;

    void reset() override;

    std::string name() const override {return "LinearInterpolate";}

    void set_activation_position(const torch::Tensor & position) override;

private:
    LinearInterpolate() = default;

    torch::Tensor activation_position_;
    bool initialized_ = false;
  };

}  // namespace isaac_deploy_core

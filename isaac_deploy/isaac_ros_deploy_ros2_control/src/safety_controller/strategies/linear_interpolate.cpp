// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/safety_controller/strategies/linear_interpolate.hpp"

#include <memory>

namespace isaac_deploy_core {

  expected < std::unique_ptr < LinearInterpolate >> LinearInterpolate::create(
    const LinearInterpolateConfig & /*config*/) {
    return std::unique_ptr < LinearInterpolate > (new LinearInterpolate());
  }

  expected < torch::Tensor > LinearInterpolate::apply(
    const torch::Tensor & command_positions,
    const torch::Tensor & current_positions,
    double blend_ratio, double /*dt*/) {
    // Initialize activation position on first call.
    if (!initialized_) {
      activation_position_ = current_positions.clone();
      initialized_ = true;
    }

    // Interpolate between activation position and commanded position.
    auto output = activation_position_ + (command_positions - activation_position_) * blend_ratio;

    return output;
  }

  void LinearInterpolate::reset()
  {
    activation_position_ = torch::Tensor();
    initialized_ = false;
  }

  void LinearInterpolate::set_activation_position(const torch::Tensor & position)
  {
    activation_position_ = position.clone();
    initialized_ = true;
  }

}  // namespace isaac_deploy_core

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <utility>

#include "torch/torch.h"

#include "isaac_deploy_core/core/error.h"

namespace isaac_deploy_core {

/// Enum for blending strategies used by the safety controller.
  enum class BlendStrategy
  {
    kNoPostProcessing,
    kClampVelocity,
    kLinearInterpolate,
  };

/// Abstract interface for safety constraint strategies.
  class SafetyStrategy {
public:
    virtual ~SafetyStrategy() = default;

    /// Apply safety constraints to commanded positions.
    /// @param command_positions Target positions from controller.
    /// @param current_positions Current joint positions.
    /// @param blend_ratio Interpolation value (0.0 = current, 1.0 = target).
    /// @param dt Time step in seconds.
    /// @return Safe positions to send to the robot.
    virtual expected < torch::Tensor > apply(
      const torch::Tensor & command_positions,
      const torch::Tensor & current_positions, double blend_ratio,
      double dt) = 0;

    /// Reset any internal state.
    virtual void reset() = 0;

    /// Set the activation position (called when controller is activated).
    /// Default implementation is a no-op; strategies that need the activation
    /// position (e.g., InterpolatePosition) override this.
    virtual void set_activation_position(const torch::Tensor & /*position*/) {}

    /// Get the name of this strategy.
    virtual std::string name() const = 0;
  };

}  // namespace isaac_deploy_core

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

#pragma once

#include <memory>
#include <string>
#include <utility>

#include "torch/torch.h"

#include "isaac_deploy_core/core/error.hpp"

namespace isaac_deploy_core {

/// Enum for blending strategies used by the safety controller.
  enum class BlendStrategy
  {
    kNoPostProcessing,
    kInterpolate,
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

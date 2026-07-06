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
#include <vector>

#include "isaac_ros_deploy_ros2_control/safety_controller/safety_strategy.hpp"

namespace isaac_deploy_core
{

/// Configuration for Interpolate strategy.
struct InterpolateConfig
{
  /// Non-positive value disables clamping for that joint.
  std::vector<double> max_velocities;
  /// Optional per-joint home pose. NaN = use measured activation pose for that joint.
  /// Must match max_velocities in size when provided.
  std::vector<double> default_position;
};

/// Velocity-limited slew: output = reference + clamp(target - reference, -max_delta, max_delta)
/// where target = home + blend_ratio * (command - home).
/// INVARIANT: max_velocity is always respected regardless of blend_ratio magnitude or step size.
class Interpolate : public SafetyStrategy {
public:
  static expected<std::unique_ptr<Interpolate>> create(const InterpolateConfig & config);

  expected<torch::Tensor> apply(
    const torch::Tensor & command_positions,
    const torch::Tensor & current_positions, double blend_ratio,
    double dt) override;

  void reset() override;

  std::string name() const override {return "Interpolate";}

private:
  Interpolate(
    torch::Tensor max_velocities, torch::Tensor home_position,
    torch::Tensor configured_home_mask);

  torch::Tensor max_velocities_;
  torch::Tensor home_position_;       // NaN-free; valid where configured_home_mask_ is true
  torch::Tensor configured_home_mask_;  // true = joint has YAML-configured home
  torch::Tensor integrated_position_;
  bool integrated_initialized_ = false;
  torch::Tensor activation_position_;  // resolved home: configured or measured at reset()
};

}  // namespace isaac_deploy_core

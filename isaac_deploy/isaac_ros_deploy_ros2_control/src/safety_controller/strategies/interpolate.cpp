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

#include "isaac_ros_deploy_ros2_control/safety_controller/strategies/interpolate.hpp"

#include <limits>
#include <utility>

namespace isaac_deploy_core
{

Interpolate::Interpolate(
  torch::Tensor max_velocities, torch::Tensor home_position,
  torch::Tensor configured_home_mask)
: max_velocities_(std::move(max_velocities)),
  home_position_(std::move(home_position)),
  configured_home_mask_(std::move(configured_home_mask)) {}

expected<std::unique_ptr<
    Interpolate>> Interpolate::create(const InterpolateConfig & config)
{
  if (config.max_velocities.empty()) {
    return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "max_velocities cannot be empty"));
  }

  const auto n = static_cast<int64_t>(config.max_velocities.size());
  torch::Tensor home_position;
  torch::Tensor configured_home_mask;
  if (!config.default_position.empty()) {
    if (config.default_position.size() != config.max_velocities.size()) {
      return tl::unexpected(
          make_error(
            Error::Code::kInvalidArgument,
            "default_position size must match max_velocities size"));
    }
    auto raw = torch::tensor(config.default_position, torch::kFloat32);
    configured_home_mask = torch::logical_not(torch::isnan(raw));
    home_position = torch::where(configured_home_mask, raw, torch::zeros_like(raw));
  } else {
    configured_home_mask = torch::zeros(n, torch::kBool);
    home_position = torch::zeros(n, torch::kFloat32);
  }

  auto max_velocities = torch::tensor(config.max_velocities, torch::kFloat32);
  return std::unique_ptr<Interpolate>(
    new Interpolate(
      std::move(max_velocities), std::move(home_position),
      std::move(configured_home_mask)));
}

expected<torch::Tensor> Interpolate::apply(
  const torch::Tensor & command_positions,
  const torch::Tensor & current_positions,
  double blend_ratio, double dt)
{
  const auto max_delta = max_velocities_ * dt;

  if (!integrated_initialized_) {
    const auto home_reshaped = home_position_.reshape_as(current_positions);
    const auto mask_reshaped = configured_home_mask_.reshape_as(current_positions);
    activation_position_ =
      torch::where(mask_reshaped, home_reshaped, current_positions).clone();
    integrated_position_ = current_positions.clone();
    integrated_initialized_ = true;
  }
  const torch::Tensor & reference = integrated_position_;
  const auto target =
    activation_position_ + blend_ratio * (command_positions - activation_position_);

  // Non-positive max velocity → unbounded (passes through).
  const auto bound = torch::where(
    max_delta <= 0,
    torch::full_like(max_delta, std::numeric_limits<float>::infinity()),
    max_delta);
  const auto clamped_delta = torch::clamp(target - reference, -bound, bound);
  const auto output = reference + clamped_delta;

  integrated_position_.copy_(output);

  return output;
}

void Interpolate::reset()
{
  integrated_position_ = torch::Tensor();
  integrated_initialized_ = false;
  activation_position_ = torch::Tensor();
}

}  // namespace isaac_deploy_core

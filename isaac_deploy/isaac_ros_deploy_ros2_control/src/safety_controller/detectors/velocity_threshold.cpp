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

#include "isaac_ros_deploy_ros2_control/safety_controller/detectors/velocity_threshold.hpp"

#include <utility>

#include "torch/torch.h"

namespace isaac_deploy_core {

  VelocityThresholdDetector::VelocityThresholdDetector(VelocityThresholdConfig config)
  : config_(std::move(config)) {}

  expected < std::unique_ptr < VelocityThresholdDetector >> VelocityThresholdDetector::create(
    const VelocityThresholdConfig & config) {
    return std::unique_ptr < VelocityThresholdDetector > (new VelocityThresholdDetector(config));
  }

  expected < void > VelocityThresholdDetector::check(const std::vector < NamedTensor > &inputs) {
    // Skip check if no velocity inputs configured or thresholds disabled.
    if (config_.input_names.empty()) {
      return expected < void > ();
    }
    const bool max_check_enabled = config_.max_velocity > 0.0;
    const bool mean_check_enabled = config_.mean_velocity > 0.0;
    if (!max_check_enabled && !mean_check_enabled) {
      return expected < void > ();
    }

    // Check each configured velocity input.
    for (const auto & velocity_name : config_.input_names) {
      for (const auto & input : inputs) {
        if (input.name != velocity_name) {
          continue;
        }

        // Take absolute values for velocity magnitude check.
        const torch::Tensor abs_velocities = torch::abs(input.tensor);

        // Check max velocity threshold.
        if (max_check_enabled) {
          const double max_velocity = abs_velocities.max().item < double > ();
          if (max_velocity > config_.max_velocity) {
            return tl::unexpected(
              make_error(
                Error::Code::kFailedPrecondition,
                "Velocity safety check failed: max velocity " +
                std::to_string(max_velocity) + " rad/s exceeds threshold " +
                std::to_string(config_.max_velocity) + " rad/s in input '" +
                velocity_name + "'"));
          }
        }

        // Check mean velocity threshold.
        if (mean_check_enabled) {
          const double mean_velocity = abs_velocities.mean().item < double > ();
          if (mean_velocity > config_.mean_velocity) {
            return tl::unexpected(
              make_error(
                Error::Code::kFailedPrecondition,
                "Velocity safety check failed: mean velocity " +
                std::to_string(mean_velocity) + " rad/s exceeds threshold " +
                std::to_string(config_.mean_velocity) + " rad/s in input '" +
                velocity_name + "'"));
          }
        }
      }
    }

    return expected < void > ();
  }

  void VelocityThresholdDetector::reset()
  {
    // No internal state to reset.
  }

  std::string VelocityThresholdDetector::name() const {return "velocity_threshold";}

}  // namespace isaac_deploy_core

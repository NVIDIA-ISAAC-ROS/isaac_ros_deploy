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

#include "isaac_ros_deploy_ros2_control/safety_controller/out_of_domain_detector.hpp"

namespace isaac_deploy_core {

/// Configuration for VelocityThresholdDetector.
  struct VelocityThresholdConfig
  {
    /// Names of input tensors containing velocity data to check.
    std::vector < std::string > input_names;
    /// Maximum allowed velocity for any single joint (rad/s). Disabled if <= 0.
    double max_velocity = 0.0;
    /// Maximum allowed mean velocity across all joints (rad/s). Disabled if <= 0.
    double mean_velocity = 0.0;
  };

/// Detects out-of-domain conditions based on velocity thresholds.
///
/// Checks if joint velocities exceed configured maximum or mean thresholds.
  class VelocityThresholdDetector: public OutOfDomainDetector {
public:
    /// Create a VelocityThresholdDetector from configuration.
    static expected < std::unique_ptr < VelocityThresholdDetector >> create(
      const VelocityThresholdConfig & config);

    /// Check if velocities are within safe limits.
    expected < void > check(const std::vector < NamedTensor > &inputs) override;

    /// Reset internal state (no-op for this detector).
    void reset() override;

    /// Get the name of this detector.
    std::string name() const override;

private:
    explicit VelocityThresholdDetector(VelocityThresholdConfig config);

    VelocityThresholdConfig config_;
  };

}  // namespace isaac_deploy_core

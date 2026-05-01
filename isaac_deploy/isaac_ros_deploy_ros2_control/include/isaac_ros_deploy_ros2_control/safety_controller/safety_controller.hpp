// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "isaac_deploy_core/core/error.h"
#include "isaac_deploy_core/core/types.h"
#include "isaac_ros_deploy_ros2_control/safety_controller/detectors/velocity_threshold.hpp"
#include "isaac_ros_deploy_ros2_control/safety_controller/out_of_domain_detector.hpp"
#include "isaac_ros_deploy_ros2_control/safety_controller/safety_strategy.hpp"

namespace isaac_deploy_core {

/// Configuration for the SafetyController.
  struct SafetyControllerConfig
  {
    /// Blend strategy configuration.
    struct BlendConfig
    {
      /// Type of blend strategy to use.
      BlendStrategy type = BlendStrategy::kLinearInterpolate;
      /// Maximum velocities per joint (for kClampVelocity strategy).
      std::vector < double > max_velocities;
    };

    /// Out-of-domain detection configuration.
    struct OutOfDomainDetectionConfig
    {
      /// Velocity threshold detection configuration.
      VelocityThresholdConfig velocity_threshold;
    };

    /// Blend strategy configuration.
    BlendConfig blend_ratio;

    /// Out-of-domain detection configuration.
    OutOfDomainDetectionConfig out_of_domain_detection;

    /// Create a SafetyControllerConfig from a YAML node.
    static expected < SafetyControllerConfig > create_from_yaml(const YAML::Node & yaml);
  };

/// SafetyController applies safety constraints to joint commands.
///
/// The controller has two main components:
/// 1. Blend strategy: Interpolates between current and commanded positions
/// 2. Out-of-domain detection: Checks if operation is safe (e.g., velocity limits)
///
/// The controller processes inputs:
/// - command_positions: Target positions from the inference controller
/// - current_positions: Current measured joint positions
/// - blend_ratio: Interpolation value (0.0 = stay at current, 1.0 = go to target)
/// - dt: Time step in seconds
///
/// And produces:
/// - safe_positions: Positions that satisfy safety constraints
  class SafetyController {
public:
    /// Create the controller from configuration.
    static expected < SafetyController > create(const SafetyControllerConfig & config);

    /// Activate the controller.
    /// @param input_specs Specs for input tensors.
    /// @param inputs Initial input tensors.
    /// @param output_specs Specs for output tensors.
    /// @param outputs Pre-allocated output tensors.
    expected < void > activate(
      const std::vector < TensorSpec > &input_specs,
      const std::vector < NamedTensor > &inputs,
      const std::vector < TensorSpec > &output_specs,
      std::vector < NamedTensor > &outputs);

    /// Deactivate the controller.
    expected < void > deactivate();

    /// Advance the controller by one timestep.
    /// This method is real-time safe (does not allocate).
    /// @param timestamp_ns Current timestamp in nanoseconds.
    /// @param inputs Input tensors: command_positions, current_positions, blend_ratio, dt.
    /// @param outputs Output tensors: safe_positions.
    expected < void > advance(
      int64_t timestamp_ns, const std::vector < NamedTensor > &inputs,
      std::vector < NamedTensor > &outputs);

    /// Get the names of required inputs.
    const std::vector < std::string > & input_names() const {
      return input_names_;
    }

    /// Get the names of produced outputs.
    const std::vector < std::string > & output_names() const {
      return output_names_;
    }

private:
    SafetyController(
      std::unique_ptr < SafetyStrategy > blend_strategy,
      std::unique_ptr < OutOfDomainDetector > out_of_domain_detector);

    std::unique_ptr < SafetyStrategy > blend_strategy_;
    std::unique_ptr < OutOfDomainDetector > out_of_domain_detector_;
    std::vector < std::string > input_names_;
    std::vector < std::string > output_names_;
    int64_t last_timestamp_ns_ = 0;
  };

}  // namespace isaac_deploy_core

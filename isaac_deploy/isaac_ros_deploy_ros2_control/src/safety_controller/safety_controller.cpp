// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_deploy_ros2_control/safety_controller/safety_controller.hpp"

#include <utility>

#include "isaac_ros_deploy_ros2_control/safety_controller/strategies/clamp_velocity.hpp"
#include "isaac_ros_deploy_ros2_control/safety_controller/strategies/linear_interpolate.hpp"

namespace isaac_deploy_core {

  expected < SafetyControllerConfig >
  SafetyControllerConfig::create_from_yaml(const YAML::Node & yaml) {
    SafetyControllerConfig config;

    // Parse blend_ratio config.
    if (yaml["blend_ratio"]) {
      const auto & blend_yaml = yaml["blend_ratio"];
      if (blend_yaml["type"]) {
        const std::string type_str = blend_yaml["type"].as < std::string > ();
        if (type_str == "clamp_velocity") {
          config.blend_ratio.type = BlendStrategy::kClampVelocity;
        } else if (type_str == "linear_interpolate") {
          config.blend_ratio.type = BlendStrategy::kLinearInterpolate;
        } else if (type_str == "no_post_processing") {
          config.blend_ratio.type = BlendStrategy::kNoPostProcessing;
        } else {
          return tl::unexpected(
            make_error(
              Error::Code::kInvalidArgument,
              "Unknown blend strategy type: " + type_str));
        }
      }
      if (blend_yaml["max_velocities"]) {
        config.blend_ratio.max_velocities = blend_yaml["max_velocities"].as < std::vector <
          double >> ();
      }
    }

    // Parse out-of-domain detection config.
    if (yaml["out_of_domain_detection"]) {
      const auto & ood_yaml = yaml["out_of_domain_detection"];
      if (ood_yaml["velocity_threshold"]) {
        const auto & vt_yaml = ood_yaml["velocity_threshold"];
        if (vt_yaml["input_names"]) {
          config.out_of_domain_detection.velocity_threshold.input_names =
            vt_yaml["input_names"].as < std::vector < std::string >> ();
        }
        if (vt_yaml["max_velocity"]) {
          config.out_of_domain_detection.velocity_threshold.max_velocity =
            vt_yaml["max_velocity"].as < double > ();
        }
        if (vt_yaml["mean_velocity"]) {
          config.out_of_domain_detection.velocity_threshold.mean_velocity =
            vt_yaml["mean_velocity"].as < double > ();
        }
      }
    }

    return config;
  }

  SafetyController::SafetyController(
    std::unique_ptr < SafetyStrategy > blend_strategy,
    std::unique_ptr < OutOfDomainDetector > out_of_domain_detector)
  : blend_strategy_(std::move(blend_strategy)),
    out_of_domain_detector_(std::move(out_of_domain_detector)),
    input_names_({"command_positions", "current_positions", "blend_ratio", "dt"}),
    output_names_({"safe_positions"}) {}

  expected < SafetyController > SafetyController::create(const SafetyControllerConfig & config) {
    // Create blend strategy.
    std::unique_ptr < SafetyStrategy > blend_strategy;

    switch (config.blend_ratio.type) {
      case BlendStrategy::kNoPostProcessing:
        // No strategy - just pass through commands.
        blend_strategy = nullptr;
        break;

      case BlendStrategy::kClampVelocity: {
          ClampVelocityConfig clamp_config {.max_velocities = config.blend_ratio.max_velocities};
          auto result = ClampVelocity::create(clamp_config);
          if (!result.has_value()) {
            return tl::unexpected(result.error());
          }
          blend_strategy = std::move(*result);
          break;
        }

      case BlendStrategy::kLinearInterpolate: {
          LinearInterpolateConfig interp_config;
          auto result = LinearInterpolate::create(interp_config);
          if (!result.has_value()) {
            return tl::unexpected(result.error());
          }
          blend_strategy = std::move(*result);
          break;
        }
    }

    // Create out-of-domain detector.
    std::unique_ptr < OutOfDomainDetector > out_of_domain_detector;
    const auto & vt_config = config.out_of_domain_detection.velocity_threshold;
    if (!vt_config.input_names.empty() &&
      (vt_config.max_velocity > 0 || vt_config.mean_velocity > 0))
    {
      auto result = VelocityThresholdDetector::create(vt_config);
      if (!result.has_value()) {
        return tl::unexpected(result.error());
      }
      out_of_domain_detector = std::move(*result);
    }

    return SafetyController(std::move(blend_strategy), std::move(out_of_domain_detector));
  }

  expected < void > SafetyController::activate(
    const std::vector < TensorSpec > &/*input_specs*/,
    const std::vector < NamedTensor > &/*inputs*/,
    const std::vector < TensorSpec > &/*output_specs*/,
    std::vector < NamedTensor > &/*outputs*/) {
    if (blend_strategy_) {
      blend_strategy_->reset();
    }
    if (out_of_domain_detector_) {
      out_of_domain_detector_->reset();
    }
    last_timestamp_ns_ = 0;
    return expected < void > ();
  }

  expected < void > SafetyController::deactivate() {
    if (blend_strategy_) {
      blend_strategy_->reset();
    }
    if (out_of_domain_detector_) {
      out_of_domain_detector_->reset();
    }
    return expected < void > ();
  }

  expected < void > SafetyController::advance(
    int64_t timestamp_ns,
    const std::vector < NamedTensor > &inputs,
    std::vector < NamedTensor > &outputs) {
    // Check out-of-domain detection before processing.
    if (out_of_domain_detector_) {
      auto check_result = out_of_domain_detector_->check(inputs);
      if (!check_result.has_value()) {
        return tl::unexpected(check_result.error());
      }
    }

    // Find required inputs.
    const torch::Tensor * command_positions = nullptr;
    const torch::Tensor * current_positions = nullptr;
    double blend_ratio = 1.0;
    double dt = 0.005;  // Default 200Hz.

    for (const auto & input : inputs) {
      if (input.name == "command_positions") {
        command_positions = &input.tensor;
      } else if (input.name == "current_positions") {
        current_positions = &input.tensor;
      } else if (input.name == "blend_ratio") {
        blend_ratio = input.tensor.item < double > ();
      } else if (input.name == "dt") {
        dt = input.tensor.item < double > ();
      }
    }

    if (!command_positions) {
      return tl::unexpected(
        make_error(
          Error::Code::kNotFound,
          "command_positions input not found"));
    }
    if (!current_positions) {
      return tl::unexpected(
        make_error(
          Error::Code::kNotFound,
          "current_positions input not found"));
    }

    // Compute dt from timestamps if not provided.
    if (last_timestamp_ns_ > 0 && dt <= 0) {
      dt = static_cast < double > (timestamp_ns - last_timestamp_ns_) * 1e-9;
    }
    last_timestamp_ns_ = timestamp_ns;

    // Apply blend strategy.
    torch::Tensor safe_positions;
    if (blend_strategy_) {
      auto result =
        blend_strategy_->apply(*command_positions, *current_positions, blend_ratio, dt);
      if (!result.has_value()) {
        return tl::unexpected(result.error());
      }
      safe_positions = *result;
    } else {
      // No strategy - pass through commands (with blend ratio interpolation).
      safe_positions = *current_positions + (*command_positions - *current_positions) *
        blend_ratio;
    }

    // Write output.
    for (auto & output : outputs) {
      if (output.name == "safe_positions") {
        output.tensor = safe_positions;
        output.timestamp_ns = timestamp_ns;
      }
    }

    return expected < void > ();
  }

}  // namespace isaac_deploy_core

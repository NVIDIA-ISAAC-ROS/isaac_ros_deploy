// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef ISAAC_DEPLOY_CORE__CHUNK_SAMPLER_HPP_
#define ISAAC_DEPLOY_CORE__CHUNK_SAMPLER_HPP_

#include <span>
#include <cstddef>

#include "rclcpp/rclcpp.hpp"

namespace isaac_deploy_core
{

/// Interpolation position within a uniformly-spaced chunk.
struct ChunkSamplePosition
{
  std::size_t lo;          ///< lower step index
  std::size_t hi;          ///< upper step index (== lo when clamped)
  double alpha;            ///< 0.0..1.0 weight on `hi`
  bool past_end;           ///< true if `now` > step H-1
  bool before_start;       ///< true if `now` < chunk_start
};

/// Compute where `now` falls within a chunk of `horizon` steps, each
/// `step_dt` apart, starting at `chunk_start`.  Out-of-range is clamped to
/// the nearest endpoint; the `past_end` / `before_start` flags let the
/// caller warn or branch.
inline ChunkSamplePosition compute_chunk_position(
  const rclcpp::Time & chunk_start,
  const rclcpp::Duration & step_dt,
  std::size_t horizon,
  const rclcpp::Time & now)
{
  if (horizon == 0) {
    return {0, 0, 0.0, true, true};
  }
  const double step_seconds = step_dt.seconds();
  const double elapsed = (now - chunk_start).seconds();
  const double fstep = elapsed / step_seconds;
  const std::size_t last = horizon - 1;

  // Check past_end first so that horizon == 1 (last == 0) always reports
  // past_end when fstep >= 0.
  if (fstep >= static_cast<double>(last)) {
    return {last, last, 0.0, true, false};
  }
  if (fstep <= 0.0) {
    return {0, 0, 0.0, false, fstep < 0.0};
  }
  const std::size_t lo = static_cast<std::size_t>(fstep);
  const double alpha = fstep - static_cast<double>(lo);
  return {lo, lo + 1, alpha, false, false};
}

/// Linearly interpolate one column of a row-major [H, N] field at a
/// previously-computed sample position.  Callers typically loop over
/// columns (N joints) with the same `pos`.
inline double interpolate_field(
  std::span<const double> row_major_field,
  std::size_t n,
  std::size_t col,
  const ChunkSamplePosition & pos)
{
  const double a = row_major_field[pos.lo * n + col];
  const double b = row_major_field[pos.hi * n + col];
  return a + pos.alpha * (b - a);
}

}  // namespace isaac_deploy_core

#endif  // ISAAC_DEPLOY_CORE__CHUNK_SAMPLER_HPP_

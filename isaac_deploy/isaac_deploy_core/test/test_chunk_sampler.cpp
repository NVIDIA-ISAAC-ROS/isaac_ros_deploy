// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <gtest/gtest.h>

#include "isaac_deploy_core/chunk_sampler.hpp"
#include "rclcpp/rclcpp.hpp"

namespace
{

using isaac_deploy_core::compute_chunk_position;
using isaac_deploy_core::interpolate_field;

constexpr double kEps = 1e-9;

rclcpp::Time t(double sec)
{
  return rclcpp::Time(static_cast<int64_t>(sec * 1e9));
}
rclcpp::Duration d(double sec)
{
  return rclcpp::Duration::from_seconds(sec);
}

TEST(ChunkSamplerTest, PositionAtStep0)
{
  auto pos = compute_chunk_position(t(1.0), d(0.02), 30, t(1.0));
  EXPECT_EQ(pos.lo, 0u);
  EXPECT_EQ(pos.hi, 0u);
  EXPECT_NEAR(pos.alpha, 0.0, kEps);
  EXPECT_FALSE(pos.past_end);
  EXPECT_FALSE(pos.before_start);
}

TEST(ChunkSamplerTest, PositionHalfwayBetweenSteps)
{
  // 1.03s is halfway between step 1 (1.02s) and step 2 (1.04s).
  auto pos = compute_chunk_position(t(1.0), d(0.02), 30, t(1.03));
  EXPECT_EQ(pos.lo, 1u);
  EXPECT_EQ(pos.hi, 2u);
  EXPECT_NEAR(pos.alpha, 0.5, kEps);
  EXPECT_FALSE(pos.past_end);
  EXPECT_FALSE(pos.before_start);
}

TEST(ChunkSamplerTest, PositionPastEndClampsToLastStep)
{
  // H=5 means last step is step 4 at t=1.08s.  Query at 1.5s.
  auto pos = compute_chunk_position(t(1.0), d(0.02), 5, t(1.5));
  EXPECT_EQ(pos.lo, 4u);
  EXPECT_EQ(pos.hi, 4u);
  EXPECT_NEAR(pos.alpha, 0.0, kEps);
  EXPECT_TRUE(pos.past_end);
  EXPECT_FALSE(pos.before_start);
}

TEST(ChunkSamplerTest, PositionBeforeStartClampsToStep0)
{
  auto pos = compute_chunk_position(t(5.0), d(0.02), 30, t(1.0));
  EXPECT_EQ(pos.lo, 0u);
  EXPECT_EQ(pos.hi, 0u);
  EXPECT_NEAR(pos.alpha, 0.0, kEps);
  EXPECT_FALSE(pos.past_end);
  EXPECT_TRUE(pos.before_start);
}

TEST(ChunkSamplerTest, InterpolateFieldHalfway)
{
  // 2 steps x 3 joints.  Halfway: (a+b)/2.
  std::vector<double> field = {
    10.0, 20.0, 30.0,   // step 0
    12.0, 24.0, 36.0,   // step 1
  };
  isaac_deploy_core::ChunkSamplePosition pos{
    /*lo=*/0, /*hi=*/1, /*alpha=*/0.5,
    /*past_end=*/false, /*before_start=*/false};

  EXPECT_NEAR(interpolate_field(field, /*n=*/3, /*col=*/0, pos), 11.0, kEps);
  EXPECT_NEAR(interpolate_field(field, /*n=*/3, /*col=*/1, pos), 22.0, kEps);
  EXPECT_NEAR(interpolate_field(field, /*n=*/3, /*col=*/2, pos), 33.0, kEps);
}

TEST(ChunkSamplerTest, InterpolateFieldAtClampedEndpoint)
{
  std::vector<double> field = {1.0, 2.0, 3.0, 4.0};
  isaac_deploy_core::ChunkSamplePosition pos{
    /*lo=*/1, /*hi=*/1, /*alpha=*/0.0,
    /*past_end=*/true, /*before_start=*/false};

  EXPECT_NEAR(interpolate_field(field, /*n=*/2, /*col=*/0, pos), 3.0, kEps);
  EXPECT_NEAR(interpolate_field(field, /*n=*/2, /*col=*/1, pos), 4.0, kEps);
}

TEST(ChunkSamplerTest, HorizonOneIsAlwaysClampedPastEnd)
{
  auto pos = compute_chunk_position(t(1.0), d(0.02), 1, t(1.0));
  EXPECT_EQ(pos.lo, 0u);
  EXPECT_EQ(pos.hi, 0u);
  EXPECT_TRUE(pos.past_end);  // anything >= step 0 is past step H-1 when H=1
}

}  // namespace

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "isaac_deploy_core/core/error.hpp"

namespace isaac_deploy_core
{

struct TermMetadata
{
  std::vector<std::vector<std::string>> element_names;
  std::optional<std::size_t> temporal_axis;
  std::optional<double> temporal_period_ms;
};

namespace detail
{

inline tl::unexpected<Error> term_metadata_error(
  std::string_view context, std::string_view message)
{
  return tl::unexpected(
    make_error(
      Error::Code::kInvalidArgument,
      std::string(context) + ": " + std::string(message)));
}

inline expected<void> parse_element_names(
  const YAML::Node & element_names,
  const std::vector<int64_t> & shape,
  std::string_view context,
  TermMetadata & metadata)
{
  if (!element_names.IsDefined() || element_names.IsNull()) {
    return {};
  }
  if (!element_names.IsSequence()) {
    return term_metadata_error(context, "'element_names' must be a sequence");
  }

  for (const auto & dim : element_names) {
    if (dim.IsScalar()) {
      if (dim.as<std::string>() != "__temporal_axis__") {
        return term_metadata_error(
          context, "scalar element_names entries must be '__temporal_axis__'");
      }
      if (metadata.temporal_axis.has_value()) {
        return term_metadata_error(
          context, "element_names may contain at most one temporal axis");
      }
      metadata.temporal_axis = metadata.element_names.size();
      metadata.element_names.emplace_back();
    } else if (dim.IsNull()) {
      metadata.element_names.emplace_back();
    } else if (dim.IsSequence()) {
      try {
        metadata.element_names.push_back(dim.as<std::vector<std::string>>());
      } catch (const YAML::Exception &) {
        return term_metadata_error(
          context, "element_names sequences must contain only strings");
      }
    } else {
      return term_metadata_error(
        context, "element_names entries must be sequences, null, or the temporal sentinel");
    }
  }

  const auto padding = metadata.element_names.size() < shape.size() ?
    shape.size() - metadata.element_names.size() : 0;
  metadata.element_names.insert(
    metadata.element_names.begin(), padding, std::vector<std::string>{});
  if (metadata.temporal_axis.has_value()) {
    *metadata.temporal_axis += padding;
  }
  return {};
}

inline expected<std::optional<double>> parse_temporal_period(
  const YAML::Node & period, std::string_view context)
{
  if (!period.IsDefined() || period.IsNull()) {
    return std::nullopt;
  }
  try {
    const double period_ms = period.as<double>();
    if (!std::isfinite(period_ms) || period_ms <= 0.0) {
      return term_metadata_error(context, "temporal_period_ms must be positive");
    }
    return period_ms;
  } catch (const YAML::Exception &) {
    return term_metadata_error(context, "temporal_period_ms must be numeric");
  }
}

}  // namespace detail

inline expected<std::vector<std::vector<std::string>>> parse_term_element_names(
  const YAML::Node & yaml,
  const std::vector<int64_t> & shape,
  std::string_view context)
{
  TermMetadata metadata;
  auto result = detail::parse_element_names(
    yaml["element_names"], shape, context, metadata);
  if (!result.has_value()) {
    return tl::unexpected(result.error());
  }
  return std::move(metadata.element_names);
}

inline expected<TermMetadata> parse_term_metadata(
  const YAML::Node & yaml,
  const std::vector<int64_t> & shape,
  std::string_view context)
{
  TermMetadata metadata;
  auto element_names = detail::parse_element_names(
    yaml["element_names"], shape, context, metadata);
  if (!element_names.has_value()) {
    return tl::unexpected(element_names.error());
  }

  auto temporal_period = detail::parse_temporal_period(yaml["temporal_period_ms"], context);
  if (!temporal_period.has_value()) {
    return tl::unexpected(temporal_period.error());
  }
  metadata.temporal_period_ms = *temporal_period;
  // Incomplete temporal metadata was ignored before temporal support was added. Preserve that
  // behavior so older bundles keep loading; only a complete pair is consumed.
  if (metadata.temporal_axis.has_value() != metadata.temporal_period_ms.has_value()) {
    metadata.temporal_axis.reset();
    metadata.temporal_period_ms.reset();
  }
  return metadata;
}

}  // namespace isaac_deploy_core

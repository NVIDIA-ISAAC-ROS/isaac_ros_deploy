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

#include "isaac_ros_deploy_ros2_control/utils/gain_utils.hpp"

#include <regex>

namespace isaac_ros_deploy_ros2_control
{
namespace utils
{

std::vector<double> resolve_gains(
  const std::vector<std::pair<std::string, double>> & patterns,
  const std::vector<std::string> & joint_names,
  double unmatched_default)
{
  std::vector<double> gains(joint_names.size(), unmatched_default);

  for (const auto & [pattern_str, value] : patterns) {
    std::regex pattern(pattern_str);
    for (size_t i = 0; i < joint_names.size(); ++i) {
      if (std::regex_match(joint_names[i], pattern)) {
        gains[i] = value;
      }
    }
  }

  return gains;
}

}  // namespace utils
}  // namespace isaac_ros_deploy_ros2_control

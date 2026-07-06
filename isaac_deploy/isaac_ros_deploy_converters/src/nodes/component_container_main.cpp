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

// Custom component_container that links against libtorch and onnxruntime.
//
// The standard rclcpp_components::component_container does not have these
// libraries in its DT_RPATH.  When it dlopen's composable node .so files
// that depend on libtorch/onnxruntime, the dynamic linker cannot find them.
// This binary is identical in behavior but carries the necessary RPATH
// entries via its Bazel deps.

#include <memory>

#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/utilities.hpp"

#include "rclcpp_components/component_manager.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto exec = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  auto node = std::make_shared<rclcpp_components::ComponentManager>(exec);
  exec->add_node(node);
  exec->spin();
  return 0;
}

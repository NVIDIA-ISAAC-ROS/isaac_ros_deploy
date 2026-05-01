// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Custom component_container that links against libtorch and onnxruntime.
//
// The standard rclcpp_components::component_container does not have these
// libraries in its DT_RPATH.  When it dlopen's composable node .so files
// that depend on libtorch/onnxruntime, the dynamic linker cannot find them.
// This binary is identical in behaviour but carries the necessary RPATH
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

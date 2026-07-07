# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(isaac_ros_deploy_bringup)
find_package(ament_cmake REQUIRED)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  find_package(ament_cmake_ros REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  function(add_ros_isolated_launch_test path)
    set(RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
    add_launch_test("${path}" RUNNER "${RUNNER}" ${ARGN})
  endfunction()

  install(DIRECTORY test/data/ DESTINATION share/${PROJECT_NAME}/test/data)
  install(DIRECTORY test/scripts/ DESTINATION share/${PROJECT_NAME}/test/scripts USE_SOURCE_PERMISSIONS)

  add_ros_isolated_launch_test(test/test_single_model_inference_graph.py TIMEOUT 180)
  add_ros_isolated_launch_test(test/test_multi_model_inference_graph.py TIMEOUT 180)
endif()

ament_package()

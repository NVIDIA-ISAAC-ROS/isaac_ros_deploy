# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(isaac_ros_deploy_interfaces)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Explicitly use the Python from the current environment
# to avoid other Python versions leaking into PATH
execute_process(
  COMMAND which python3
  OUTPUT_VARIABLE PYTHON3_EXECUTABLE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Python3_EXECUTABLE ${PYTHON3_EXECUTABLE} CACHE FILEPATH "Python interpreter")
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BodyCommand.msg"
  "msg/JointCommand.msg"
  "msg/ReferenceMotionState.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

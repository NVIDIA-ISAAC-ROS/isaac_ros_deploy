# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(isaac_ros_deploy_converters)

# C++20 to match isaac_deploy_core.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force RPATH instead of RUNPATH for better plugin loading.
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# Disable RUNPATH (use RPATH instead).
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers)
endif()

# Find dependencies.
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Find isaac_deploy_core first - it exports LibTorch and ONNX Runtime for us.
find_package(isaac_deploy_core REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(isaac_ros_deploy_interfaces REQUIRED)
find_package(isaac_ros_tensor_list_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# LibTorch is provided by isaac_deploy_core via isaac_deploy_core-extras.cmake.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# =============================================================================
# Converters Library (message-to-tensor, shared with isaac_ros_deploy_ros2_control)
# =============================================================================

add_library(converters SHARED
  src/converters/message_to_tensor_converters.cpp
)
target_include_directories(converters PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(converters
  rclcpp
  isaac_ros_deploy_interfaces
  sensor_msgs
  geometry_msgs
  nav_msgs
)
target_link_libraries(converters
  isaac_deploy_core::types
  ${TORCH_LIBRARIES}
)
set_target_properties(converters PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# =============================================================================
# Utilities Library
# =============================================================================

add_library(tensor_list_utils SHARED
  src/utils/tensor_list_utils.cpp
)
target_include_directories(tensor_list_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(tensor_list_utils
  rclcpp
  isaac_ros_tensor_list_interfaces
)
target_link_libraries(tensor_list_utils
  isaac_deploy_core::types
  isaac_deploy_core::tensor_utils
  ${TORCH_LIBRARIES}
)
set_target_properties(tensor_list_utils PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# =============================================================================
# Output Converters Library (tensor-to-message, used only by OutputBuilderNode)
# =============================================================================

add_library(output_converters SHARED
  src/converters/tensor_to_message_converters.cpp
)
target_include_directories(output_converters PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(output_converters
  rclcpp
  isaac_ros_deploy_interfaces
  sensor_msgs
  geometry_msgs
)
target_link_libraries(output_converters
  tensor_list_utils
  isaac_deploy_core::types
  ${TORCH_LIBRARIES}
  yaml-cpp
)
set_target_properties(output_converters PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# =============================================================================
# Nodes Library
# =============================================================================

add_library(isaac_ros_deploy_converters_lib SHARED
  src/nodes/input_builder_node.cpp
  src/nodes/output_builder_node.cpp
  src/nodes/joint_command_to_joint_state_node.cpp
)
target_include_directories(isaac_ros_deploy_converters_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(isaac_ros_deploy_converters_lib
  rclcpp
  rclcpp_components
  isaac_ros_deploy_interfaces
  isaac_ros_tensor_list_interfaces
  sensor_msgs
)
target_link_libraries(isaac_ros_deploy_converters_lib
  converters
  output_converters
  tensor_list_utils
  isaac_deploy_core::config_parser
  isaac_deploy_core::input_builder
  isaac_deploy_core::output_builder
  ${TORCH_LIBRARIES}
  yaml-cpp
)
set_target_properties(isaac_ros_deploy_converters_lib PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# Register components.
rclcpp_components_register_nodes(isaac_ros_deploy_converters_lib
  "isaac_ros_deploy_converters::InputBuilderNode"
  "isaac_ros_deploy_converters::OutputBuilderNode"
  "isaac_ros_deploy_converters::JointCommandToJointStateNode"
)

# =============================================================================
# Executables
# =============================================================================

add_executable(input_builder_node_exe src/nodes/input_builder_main.cpp)
target_link_libraries(input_builder_node_exe isaac_ros_deploy_converters_lib)
set_target_properties(input_builder_node_exe PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

add_executable(output_builder_node_exe src/nodes/output_builder_main.cpp)
target_link_libraries(output_builder_node_exe isaac_ros_deploy_converters_lib)
set_target_properties(output_builder_node_exe PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

add_executable(joint_command_to_joint_state_node src/nodes/joint_command_to_joint_state_main.cpp)
target_link_libraries(joint_command_to_joint_state_node isaac_ros_deploy_converters_lib)
set_target_properties(joint_command_to_joint_state_node PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# Custom component_container that carries libtorch/onnxruntime in its RPATH.
# The standard rclcpp_components component_container does not link against these
# libraries, so dlopen of composable node .so files that need them fails.
# TODO(lgulich): Remove this again
add_executable(component_container src/nodes/component_container_main.cpp)
ament_target_dependencies(component_container rclcpp rclcpp_components)
target_link_libraries(component_container isaac_ros_deploy_converters_lib)
set_target_properties(component_container PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;$ORIGIN/.."
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${onnxruntime_SOURCE_DIR}/lib"
)

# =============================================================================
# Python module (for launch file utilities)
# =============================================================================

ament_python_install_package(${PROJECT_NAME})

# =============================================================================
# Installation
# =============================================================================

install(
  TARGETS converters tensor_list_utils output_converters isaac_ros_deploy_converters_lib
  EXPORT isaac_ros_deploy_converters_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS
    input_builder_node_exe
    output_builder_node_exe
    joint_command_to_joint_state_node
    component_container
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(converters)
ament_export_targets(isaac_ros_deploy_converters_targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  isaac_deploy_core
  rclcpp
  rclcpp_components
  isaac_ros_deploy_interfaces
  isaac_ros_tensor_list_interfaces
  sensor_msgs
  geometry_msgs
  nav_msgs
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_message_to_tensor_converters
    test/test_message_to_tensor_converters.cpp
  )
  target_link_libraries(test_message_to_tensor_converters
    converters
    ${TORCH_LIBRARIES}
  )
  target_include_directories(test_message_to_tensor_converters PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  ament_target_dependencies(test_message_to_tensor_converters
    rclcpp
    geometry_msgs
  )
  set_target_properties(test_message_to_tensor_converters PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )
endif()

ament_package()

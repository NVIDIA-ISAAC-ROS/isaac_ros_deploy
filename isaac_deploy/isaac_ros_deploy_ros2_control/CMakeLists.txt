# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(isaac_ros_deploy_ros2_control)

# C++20 to match isaac_deploy_core.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force RPATH instead of RUNPATH for better plugin loading.
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# Disable RUNPATH (use RPATH instead)
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
find_package(isaac_ros_deploy_converters REQUIRED)

find_package(controller_interface REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(isaac_ros_deploy_interfaces REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(isaac_ros_inverse_dynamics REQUIRED)

# LibTorch is provided by isaac_deploy_core via isaac_deploy_core-extras.cmake.
# The Torch package is already found, and TORCH_INSTALL_PREFIX is set.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# =============================================================================
# Safety Controller - Interfaces
# =============================================================================

# safety_strategy (header-only)
add_library(safety_strategy INTERFACE)
target_include_directories(safety_strategy INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(safety_strategy INTERFACE
  isaac_deploy_core::expected
  ${TORCH_LIBRARIES}
)

# out_of_domain_detector (header-only)
add_library(out_of_domain_detector INTERFACE)
target_include_directories(out_of_domain_detector INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(out_of_domain_detector INTERFACE
  isaac_deploy_core::types
  isaac_deploy_core::expected
)

# =============================================================================
# Safety Controller - Detectors
# =============================================================================

# velocity_threshold
add_library(velocity_threshold SHARED
  src/safety_controller/detectors/velocity_threshold.cpp
)
target_include_directories(velocity_threshold PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(velocity_threshold PUBLIC
  out_of_domain_detector
  isaac_deploy_core::types
  isaac_deploy_core::expected
  ${TORCH_LIBRARIES}
)
set_target_properties(velocity_threshold PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# =============================================================================
# Safety Controller - Strategies
# =============================================================================

# interpolate
add_library(interpolate SHARED
  src/safety_controller/strategies/interpolate.cpp
)
target_include_directories(interpolate PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(interpolate PUBLIC
  safety_strategy
  ${TORCH_LIBRARIES}
)
set_target_properties(interpolate PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# =============================================================================
# Safety Controller
# =============================================================================

add_library(safety_controller SHARED
  src/safety_controller/safety_controller.cpp
)
target_include_directories(safety_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(safety_controller PUBLIC
  out_of_domain_detector
  safety_strategy
  isaac_deploy_core::types
  isaac_deploy_core::expected
  velocity_threshold
  interpolate
  yaml-cpp
)
set_target_properties(safety_controller PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# =============================================================================
# Utilities Library
# =============================================================================

add_library(tensor_interface_utils SHARED
  src/utils/tensor_interface_utils.cpp
)
target_include_directories(tensor_interface_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(tensor_interface_utils
  hardware_interface
)
target_link_libraries(tensor_interface_utils
  isaac_deploy_core::types
  isaac_deploy_core::tensor_utils
  ${TORCH_LIBRARIES}
)

# Set RPATH for LibTorch.
set_target_properties(tensor_interface_utils PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# Controllers Library.
add_library(isaac_ros_deploy_ros2_control_controllers SHARED
  src/converters/state_interface_converters.cpp
  src/converters/command_interface_converters.cpp
  src/adapters/state_interface_adapter.cpp
  src/adapters/command_interface_adapter.cpp
  src/controllers/disable_controller.cpp
  src/controllers/freeze_controller.cpp
  src/controllers/forward_joint_command_controller.cpp
  src/controllers/inference_controller.cpp
  src/controllers/joint_command_broadcaster.cpp
  src/controllers/safety_controller.cpp
  src/utils/gain_utils.cpp
)
target_include_directories(isaac_ros_deploy_ros2_control_controllers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(isaac_ros_deploy_ros2_control_controllers
  controller_manager_msgs
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  sensor_msgs
  std_msgs
  isaac_ros_deploy_interfaces
  isaac_ros_deploy_converters
)
target_link_libraries(isaac_ros_deploy_ros2_control_controllers
  tensor_interface_utils
  isaac_deploy_core::config_parser
  isaac_deploy_core::inference_controller
  isaac_ros_deploy_converters::converters
  safety_controller
  isaac_ros_inverse_dynamics::inverse_dynamics_solver
  ${TORCH_LIBRARIES}
  yaml-cpp
)

# Set RPATH for LibTorch.
set_target_properties(isaac_ros_deploy_ros2_control_controllers PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH   "${TORCH_INSTALL_PREFIX}/lib"
)

# Hardware Interfaces Library.
add_library(isaac_ros_deploy_ros2_control_hardware SHARED
  src/hardware/topic_sensor_interface.cpp
)
target_include_directories(isaac_ros_deploy_ros2_control_hardware PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(isaac_ros_deploy_ros2_control_hardware
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  std_msgs
)

# Export controller plugin.
pluginlib_export_plugin_description_file(
  controller_interface isaac_ros_deploy_ros2_control_controllers.xml)

# Export hardware interface plugin.
pluginlib_export_plugin_description_file(
  hardware_interface isaac_ros_deploy_ros2_control_hardware.xml)

# Install.
install(
  TARGETS
    safety_strategy
    out_of_domain_detector
    velocity_threshold
    interpolate
    safety_controller
    tensor_interface_utils
    isaac_ros_deploy_ros2_control_controllers
    isaac_ros_deploy_ros2_control_hardware
  EXPORT isaac_ros_deploy_ros2_control_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)


# Testing.
if(BUILD_TESTING)

  # Install test utilities.
  install(
    DIRECTORY test/test_utils/
    DESTINATION lib/${PROJECT_NAME}/test_utils
  )
  ament_get_python_install_dir(_python_install_dir)
  if(_python_install_dir)
    install(
      FILES test/test_utils/mujoco_test_helpers.py
      DESTINATION "${_python_install_dir}"
    )
  endif()
  find_package(ament_cmake_gtest REQUIRED)

  # Helper function to add a converter test target.
  function(add_converter_test test_name test_file)
    ament_add_gtest(${test_name} ${test_file})
    target_link_libraries(${test_name}
      isaac_ros_deploy_ros2_control_controllers
      ${TORCH_LIBRARIES}
    )
    target_include_directories(${test_name} PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    ament_target_dependencies(${test_name}
      hardware_interface
      rclcpp
    )
    set_target_properties(${test_name} PROPERTIES
      INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
      BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
  endfunction()

  add_converter_test(test_converter_registry test/test_converter_registry.cpp)
  add_converter_test(test_state_interface_converters test/test_state_interface_converters.cpp)
  add_converter_test(test_command_interface_converters test/test_command_interface_converters.cpp)
  add_converter_test(test_gain_utils test/test_gain_utils.cpp)

  # Safety controller tests.
  ament_add_gtest(velocity_threshold_test test/test_velocity_threshold.cpp)
  target_link_libraries(velocity_threshold_test velocity_threshold ${TORCH_LIBRARIES})
  set_target_properties(velocity_threshold_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  ament_add_gtest(interpolate_test test/test_interpolate.cpp)
  target_link_libraries(interpolate_test interpolate ${TORCH_LIBRARIES})
  set_target_properties(interpolate_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  ament_add_gtest(safety_controller_test test/test_safety_controller.cpp)
  target_link_libraries(safety_controller_test safety_controller ${TORCH_LIBRARIES})
  set_target_properties(safety_controller_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  # Joint command broadcaster test.
  ament_add_gtest(joint_command_broadcaster_test test/test_joint_command_broadcaster.cpp)
  target_link_libraries(joint_command_broadcaster_test
    isaac_ros_deploy_ros2_control_controllers
    ${TORCH_LIBRARIES}
  )
  target_include_directories(joint_command_broadcaster_test PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  ament_target_dependencies(joint_command_broadcaster_test
    controller_interface
    hardware_interface
    rclcpp
    rclcpp_lifecycle
    sensor_msgs
  )
  set_target_properties(joint_command_broadcaster_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  # ROS safety controller test.
  ament_add_gtest(ros_safety_controller_test test/test_ros_safety_controller.cpp)
  target_link_libraries(ros_safety_controller_test
    isaac_ros_deploy_ros2_control_controllers
    ${TORCH_LIBRARIES}
  )
  target_include_directories(ros_safety_controller_test PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  ament_target_dependencies(ros_safety_controller_test
    controller_interface
    controller_manager_msgs
    hardware_interface
    rclcpp
    rclcpp_lifecycle
  )
  set_target_properties(ros_safety_controller_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  # ForwardJointCommandController trajectory-mode sampling smoke test.
  ament_add_gtest(test_forward_joint_command_controller_trajectory
    test/test_forward_joint_command_controller_trajectory.cpp)
  ament_target_dependencies(test_forward_joint_command_controller_trajectory
    rclcpp
    isaac_ros_deploy_interfaces
    isaac_deploy_core
  )
endif()

ament_export_targets(isaac_ros_deploy_ros2_control_targets HAS_LIBRARY_TARGET)
ament_export_dependencies(isaac_deploy_core yaml-cpp isaac_ros_inverse_dynamics)

ament_package()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(isaac_deploy_core VERSION 0.1.0)

# Default to C++20 (matching Bazel config).
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Werror -Wno-missing-field-initializers)
endif()

# Find dependencies.
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tl_expected REQUIRED)

# Find LibTorch (installed via pip in the Docker image).
# Use Python to discover torch's cmake config path since pip installs
# place it outside CMake's default search paths.
execute_process(
  COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE _torch_result
)
if(_torch_result EQUAL 0 AND TORCH_CMAKE_PREFIX_PATH)
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Download and find Triton Inference Server.
include(FetchContent)
if(NOT TRITON_SERVER_TARBALL_URL)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(_TRITON_BASE_URL "https://github.com/triton-inference-server/server/releases/download")
    set(TRITON_SERVER_TARBALL_URL "${_TRITON_BASE_URL}/v2.60.0/tritonserver2.60.0-agx.tar")
  else()
    set(_TRITON_BASE_URL "https://artifactory.pdx.nvidia.com/artifactory")
    set(_TRITON_PATH "sw-isaac-ros-generic-local/triton/tritonserver.2.60.tar.gz")
    set(TRITON_SERVER_TARBALL_URL "${_TRITON_BASE_URL}/${_TRITON_PATH}")
  endif()
endif()
fetchcontent_declare(
  tritonserver
  URL ${TRITON_SERVER_TARBALL_URL}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
fetchcontent_populate(tritonserver)

file(GLOB_RECURSE _TRITON_HDR_CANDIDATES "${tritonserver_SOURCE_DIR}/*/tritonserver.h")
list(GET _TRITON_HDR_CANDIDATES 0 _TRITONSERVER_H)
get_filename_component(_TRITON_CORE_DIR "${_TRITONSERVER_H}" DIRECTORY)
get_filename_component(_TRITON_DIR "${_TRITON_CORE_DIR}" DIRECTORY)
get_filename_component(_TRITON_INC_DIR "${_TRITON_DIR}" DIRECTORY)

file(GLOB_RECURSE _TRITON_LIB_CANDIDATES "${tritonserver_SOURCE_DIR}/*/libtritonserver.so")
list(GET _TRITON_LIB_CANDIDATES 0 _TRITONSERVER_SO)
get_filename_component(_TRITON_LIB_DIR "${_TRITONSERVER_SO}" DIRECTORY)

add_library(tritonserver::tritonserver SHARED IMPORTED)
set_target_properties(tritonserver::tritonserver PROPERTIES
  IMPORTED_LOCATION "${_TRITON_LIB_DIR}/libtritonserver.so"
  INTERFACE_INCLUDE_DIRECTORIES "${_TRITON_INC_DIR}"
)

# =============================================================================
# Include Directory Setup
# =============================================================================

# Create symlinks for build-time includes that match the install structure.
# This allows both internal includes (core/types.hpp) and external includes
# (isaac_deploy_core/core/types.hpp) to work during build.
set(BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${BUILD_INCLUDE_DIR}/isaac_deploy_core)
file(CREATE_LINK
  ${CMAKE_CURRENT_SOURCE_DIR}/core
  ${BUILD_INCLUDE_DIR}/isaac_deploy_core/core
  SYMBOLIC
)
file(CREATE_LINK
  ${CMAKE_CURRENT_SOURCE_DIR}/inference_controller
  ${BUILD_INCLUDE_DIR}/isaac_deploy_core/inference_controller
  SYMBOLIC
)

# =============================================================================
# Core Libraries
# =============================================================================

# core::types (header-only)
add_library(types INTERFACE)
target_include_directories(types INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(types INTERFACE ${TORCH_LIBRARIES})

# core::expected (header-only)
add_library(expected INTERFACE)
target_include_directories(expected INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(expected INTERFACE tl_expected::tl_expected)

# core::tensor_utils
add_library(tensor_utils SHARED
  core/tensor_utils.cpp
)
target_include_directories(tensor_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensor_utils PUBLIC
  types
  expected
  ${TORCH_LIBRARIES}
)
set_target_properties(tensor_utils PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# =============================================================================
# Inference Controller - Input
# =============================================================================

# inference_controller/input::input_term
add_library(input_term SHARED
  inference_controller/input/input_term.cpp
)
target_include_directories(input_term PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(input_term PUBLIC
  types
  expected
  tensor_utils
  ${TORCH_LIBRARIES}
  yaml-cpp
)
set_target_properties(input_term PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# inference_controller/input::input_builder
add_library(input_builder SHARED
  inference_controller/input/input_builder.cpp
)
target_include_directories(input_builder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(input_builder PUBLIC
  input_term
  config_parser
  types
  expected
)
set_target_properties(input_builder PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# =============================================================================
# Inference Controller - Output
# =============================================================================

# inference_controller/output::output_term
add_library(output_term SHARED
  inference_controller/output/output_term.cpp
)
target_include_directories(output_term PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(output_term PUBLIC
  types
  expected
  tensor_utils
  ${TORCH_LIBRARIES}
  yaml-cpp
)
set_target_properties(output_term PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# inference_controller/output::output_builder
add_library(output_builder SHARED
  inference_controller/output/output_builder.cpp
)
target_include_directories(output_builder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(output_builder PUBLIC
  output_term
  types
  expected
  config_parser
)
set_target_properties(output_builder PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# =============================================================================
# Inference Controller - Runner
# =============================================================================

# inference_controller/runner::inference_runner_base (header-only)
add_library(inference_runner_base INTERFACE)
target_include_directories(inference_runner_base INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(inference_runner_base INTERFACE
  types
  expected
)

# inference_controller/runner::mock_runner
add_library(mock_runner SHARED
  inference_controller/runner/mock_runner.cpp
)
target_include_directories(mock_runner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(mock_runner PUBLIC
  inference_runner_base
  types
  expected
  ${TORCH_LIBRARIES}
)
set_target_properties(mock_runner PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# inference_controller/runner::triton_runner
add_library(triton_runner SHARED
  inference_controller/runner/triton_runner.cpp
)
target_include_directories(triton_runner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(triton_runner PUBLIC
  inference_runner_base
  types
  expected
  ${TORCH_LIBRARIES}
  tritonserver::tritonserver
  dl
  pthread
)
set_target_properties(triton_runner PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_TRITON_LIB_DIR}"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_TRITON_LIB_DIR}"
)

# inference_controller/runner::inference_runner
add_library(inference_runner SHARED
  inference_controller/runner/inference_runner.cpp
)
target_include_directories(inference_runner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(inference_runner PUBLIC
  mock_runner
  triton_runner
  types
  expected
)
set_target_properties(inference_runner PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_TRITON_LIB_DIR}"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_TRITON_LIB_DIR}"
)

# =============================================================================
# Inference Controller - Config Parser
# =============================================================================

# inference_controller::config_parser
add_library(config_parser SHARED
  inference_controller/config_parser.cpp
)
target_include_directories(config_parser PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(config_parser PUBLIC
  expected
  yaml-cpp
)

# =============================================================================
# Inference Controller
# =============================================================================

add_library(inference_controller SHARED
  inference_controller/inference_controller.cpp
)
target_include_directories(inference_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(inference_controller PUBLIC
  types
  expected
  input_builder
  output_builder
  inference_runner
)
set_target_properties(inference_controller PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_TRITON_LIB_DIR}"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_TRITON_LIB_DIR}"
)

# =============================================================================
# Installation
# =============================================================================

# Install libraries.
install(
  TARGETS
    types
    expected
    tensor_utils
    input_term
    input_builder
    output_term
    output_builder
    inference_runner_base
    mock_runner
    triton_runner
    inference_runner
    config_parser
    inference_controller
  EXPORT isaac_deploy_core_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers.
install(
  DIRECTORY core/
  DESTINATION include/isaac_deploy_core/core
  FILES_MATCHING PATTERN "*.hpp"
)
install(
  DIRECTORY inference_controller/
  DESTINATION include/isaac_deploy_core/inference_controller
  FILES_MATCHING PATTERN "*.hpp"
)
install(
  DIRECTORY include/isaac_deploy_core/
  DESTINATION include/isaac_deploy_core
  FILES_MATCHING PATTERN "*.hpp"
)

# Install test data.
install(
  DIRECTORY test_data/
  DESTINATION share/${PROJECT_NAME}/test_data
)

# Export targets for downstream packages.
ament_export_targets(isaac_deploy_core_targets HAS_LIBRARY_TARGET)
ament_export_dependencies(yaml-cpp tl_expected rclcpp)

# Export LibTorch and ONNX Runtime for downstream packages.
# Configure the extras file from template to the build directory
configure_file(
  cmake/isaac_deploy_core-extras.cmake.in
  cmake/isaac_deploy_core-extras.cmake
  @ONLY
)

# Install the configured extras file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/isaac_deploy_core-extras.cmake
  DESTINATION share/${PROJECT_NAME}/cmake
)

# Register the configured extras file (relative to build directory)
list(
  APPEND ${PROJECT_NAME}_CONFIG_EXTRAS
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/isaac_deploy_core-extras.cmake"
)

# =============================================================================
# Testing
# =============================================================================

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # Core tests.
  ament_add_gtest(types_test core/types_test.cpp)
  target_link_libraries(types_test types ${TORCH_LIBRARIES})
  set_target_properties(types_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  ament_add_gtest(tensor_utils_test core/tensor_utils_test.cpp)
  target_link_libraries(tensor_utils_test tensor_utils ${TORCH_LIBRARIES})
  set_target_properties(tensor_utils_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  # Input tests.
  ament_add_gtest(input_term_test inference_controller/input/input_term_test.cpp)
  target_link_libraries(input_term_test input_term ${TORCH_LIBRARIES})
  set_target_properties(input_term_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  ament_add_gtest(input_builder_test inference_controller/input/input_builder_test.cpp)
  target_link_libraries(input_builder_test input_builder config_parser ${TORCH_LIBRARIES})
  set_target_properties(input_builder_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  # Output tests.
  ament_add_gtest(output_term_test inference_controller/output/output_term_test.cpp)
  target_link_libraries(output_term_test output_term ${TORCH_LIBRARIES})
  set_target_properties(output_term_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  ament_add_gtest(output_builder_test inference_controller/output/output_builder_test.cpp)
  target_link_libraries(output_builder_test output_builder ${TORCH_LIBRARIES})
  set_target_properties(output_builder_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )

  # Runner tests.
  ament_add_gtest(triton_runner_test inference_controller/runner/triton_runner_test.cpp)
  target_link_libraries(triton_runner_test triton_runner ${TORCH_LIBRARIES})
  set_target_properties(triton_runner_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_TRITON_LIB_DIR}"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_TRITON_LIB_DIR}"
  )

  # Config parser test.
  ament_add_gtest(config_parser_test inference_controller/config_parser_test.cpp)
  target_link_libraries(config_parser_test config_parser yaml-cpp)

  # Inference controller test.
  ament_add_gtest(inference_controller_test inference_controller/inference_controller_test.cpp)
  target_link_libraries(inference_controller_test inference_controller ${TORCH_LIBRARIES})
  set_target_properties(inference_controller_test PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_TRITON_LIB_DIR}"
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_TRITON_LIB_DIR}"
  )

  # Chunk sampler test (header-only utility).
  ament_add_gtest(test_chunk_sampler test/test_chunk_sampler.cpp)
  target_include_directories(test_chunk_sampler PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  ament_target_dependencies(test_chunk_sampler rclcpp)
endif()

ament_package()

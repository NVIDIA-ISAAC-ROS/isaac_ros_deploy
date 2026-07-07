cmake_minimum_required(VERSION 3.8)
project(isaac_ros_inverse_dynamics)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(pinocchio REQUIRED)

add_library(inverse_dynamics_solver SHARED
  src/inverse_dynamics_solver.cpp
)
target_include_directories(inverse_dynamics_solver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(inverse_dynamics_solver PUBLIC cxx_std_17)
target_link_libraries(inverse_dynamics_solver PUBLIC
  pinocchio::pinocchio
)

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS inverse_dynamics_solver
  EXPORT inverse_dynamics_solver_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(inverse_dynamics_solver_targets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(pinocchio)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(isaac_ros_deploy_reference_applications)
find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/displayport_flexiv_task_space_passthrough.py
  scripts/displayport_task_space_pose_source.py
  scripts/flexiv_set_cartesian_impedance.py
  scripts/flexiv_task_space_repeated_insertion_loop.py
  scripts/policy_io_logger.py
  scripts/task_space_policy_io_logger.py
  DESTINATION lib/${PROJECT_NAME}
)

# Imported by the scripts above from their own install directory, so it has to
# land next to them rather than in share/.
install(FILES
  scripts/task_space_pose_math.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/displayport_task_space_isaac_sim.py
  scripts/displayport_task_space_isaac_sim.sh
  DESTINATION share/${PROJECT_NAME}/scripts
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_pytest_test(
    test_task_space_pose_math
    test/test_task_space_pose_math.py
  )
  ament_add_pytest_test(
    test_displayport_flexiv_task_space_passthrough
    test/test_displayport_flexiv_task_space_passthrough.py
  )
  ament_add_pytest_test(
    test_task_space_policy_io_logger
    test/test_task_space_policy_io_logger.py
  )
endif()

ament_package()

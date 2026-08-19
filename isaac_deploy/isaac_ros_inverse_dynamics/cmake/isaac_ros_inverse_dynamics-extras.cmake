# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(CMakeFindDependencyMacro)

find_dependency(Eigen3 REQUIRED)
find_dependency(pinocchio REQUIRED)

if(TARGET Eigen3::Eigen)
  get_target_property(_eigen_include_dirs Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  if(_eigen_include_dirs)
    list(APPEND isaac_ros_inverse_dynamics_INCLUDE_DIRS "${_eigen_include_dirs}")
  endif()

  list(APPEND isaac_ros_inverse_dynamics_LIBRARIES Eigen3::Eigen)
endif()

if(TARGET pinocchio::pinocchio)
  get_target_property(
    _pinocchio_include_dirs pinocchio::pinocchio INTERFACE_INCLUDE_DIRECTORIES)
  if(_pinocchio_include_dirs)
    list(APPEND isaac_ros_inverse_dynamics_INCLUDE_DIRS "${_pinocchio_include_dirs}")
  endif()

  list(APPEND isaac_ros_inverse_dynamics_LIBRARIES pinocchio::pinocchio)
endif()

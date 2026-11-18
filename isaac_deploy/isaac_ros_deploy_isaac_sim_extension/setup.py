# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup

package_name = 'isaac_ros_deploy_isaac_sim_extension'

setup(
    name=package_name,
    version='4.5.0',
    packages=[package_name, package_name + '.scripts'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Kit extension manifest -- co-installed so the package can be
        # used as an omni.ext extension in addition to the standalone
        # CLI launcher.
        ('share/' + package_name + '/config',
            [package_name + '/config/extension.toml']),
    ],
    package_data={
        package_name: ['config/extension.toml', 'py.typed'],
    },
    include_package_data=True,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Isaac ROS Maintainers',
    maintainer_email='isaac-ros-maintainers@nvidia.com',
    description=(
        'Kit extension that builds an ArticulationActuators view from the '
        'NewtonActuator prims baked into the robot USD, plus a live kp/kd '
        'injector driven by ros2_control.'
    ),
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
)

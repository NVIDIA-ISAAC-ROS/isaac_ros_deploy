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
"""
Executable entry points for ``isaac_ros_deploy_isaac_sim_extension``.

Modules in this subpackage are intended to be invoked from the Isaac Sim
Python interpreter (``<ISAAC_PATH>/python.sh -m
isaac_ros_deploy_isaac_sim_extension.scripts.<name>``) and therefore import Kit
runtime modules at top level. They are kept separate from the importable
library so that ``isaac_ros_deploy_isaac_sim_extension`` can be loaded outside of
Isaac Sim (unit tests, linters) without side effects.
"""

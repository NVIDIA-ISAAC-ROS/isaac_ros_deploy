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
Newton actuator bridge for Isaac Sim deployment.

Provides a Kit extension that, on stage open, builds an
``ArticulationActuators`` view from the robot USD's ``NewtonActuator``
prims and a :class:`~isaac_ros_deploy_isaac_sim_extension.gain_injector.GainInjector`
that streams runtime kp/kd from ``/isaac_sim_drive_gains`` into the Newton
ControllerPD warp arrays.
"""

# Re-export the IExt subclass so Kit discovers it when the package loads as
# an extension. The omni.ext import in extension.py is guarded, so the
# package stays importable outside Kit. GainInjector is imported directly
# from its submodule by callers, keeping warp off the package-import path.
from isaac_ros_deploy_isaac_sim_extension.extension import (
    IsaacsimActuatorBridgeExtension,
)

__all__ = [
    'IsaacsimActuatorBridgeExtension',
]

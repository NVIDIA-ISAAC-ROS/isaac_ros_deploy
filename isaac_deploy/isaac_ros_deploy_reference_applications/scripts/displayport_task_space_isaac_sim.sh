#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ -z ${ISAAC_SIM_PACKAGE_PATH:-} ]]; then
  echo "ISAAC_SIM_PACKAGE_PATH must point to an Isaac Sim installation." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_BRIDGE_LIB_DIR="${ISAAC_SIM_PACKAGE_PATH}/exts/isaacsim.ros2.core/jazzy/lib"

if [[ -d ${ROS2_BRIDGE_LIB_DIR} ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ROS2_BRIDGE_LIB_DIR}"
fi

exec "${ISAAC_SIM_PACKAGE_PATH}/python.sh" \
  "${SCRIPT_DIR}/displayport_task_space_isaac_sim.py" \
  "$@"

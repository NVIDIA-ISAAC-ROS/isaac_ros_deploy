# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Body command message for reference motion tracking.
# Contains poses and twists for all tracked bodies.

std_msgs/Header header

# Body names (determines array ordering).
string[] names

# Body poses (position + orientation) in world frame.
geometry_msgs/Pose[] pose

# Body twists (linear + angular velocity) in world frame.
geometry_msgs/Twist[] twist

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Compatibility wrapper: the ROS2 tl_expected package provides headers under
// tl_expected/, while the upstream/BCR tl-expected uses tl/.
#pragma once
#include "tl/expected.hpp"

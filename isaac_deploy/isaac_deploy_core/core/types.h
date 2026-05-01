// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "torch/torch.h"

namespace isaac_deploy_core {

/// Specification for a tensor's named dimensions.
/// Used during activate() to compute reorder indices between middleware and neural network order.
  struct TensorSpec
  {
    /// Names corresponding to every element of each dimension.
    /// E.g., for a [1, 29] joint tensor: {{""}, {"joint1", "joint2", ...}}.
    std::vector < std::vector < std::string >> names;
  };

/// A named tensor with timestamp.
  struct NamedTensor
  {
    /// Name of the tensor.
    std::string name;
    /// Timestamp of the tensor in nanoseconds.
    int64_t timestamp_ns = 0;
    /// Data of the tensor.
    torch::Tensor tensor;
  };

/// Dictionary mapping names to tensors.
  using TensorDict = std::unordered_map < std::string, torch::Tensor >;

}  // namespace isaac_deploy_core

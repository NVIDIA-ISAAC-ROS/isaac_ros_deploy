// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "isaac_deploy_core/core/error.h"
#include "isaac_deploy_core/core/types.h"

namespace isaac_deploy_core {

/// Abstract interface for out-of-domain detection strategies.
///
/// Out-of-domain detectors check if the current state is safe for operation.
/// If the state is unsafe (out of domain), the detector returns an error.
  class OutOfDomainDetector {
public:
    virtual ~OutOfDomainDetector() = default;

    /// Check if the current inputs are within safe operating domain.
    /// @param inputs Input tensors containing state information.
    /// @return Success if within domain, error if out of domain.
    virtual expected < void > check(const std::vector < NamedTensor > &inputs) = 0;

    /// Reset any internal state.
    virtual void reset() = 0;

    /// Get the name of this detector.
    virtual std::string name() const = 0;
  };

}  // namespace isaac_deploy_core

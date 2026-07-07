// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>

#include "isaac_deploy_core/core/error.hpp"
#include "isaac_deploy_core/core/types.hpp"

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

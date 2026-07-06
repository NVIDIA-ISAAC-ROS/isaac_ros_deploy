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
#include <string_view>
#include <utility>

#include "tl_expected/expected.hpp"

namespace isaac_deploy_core {

/// Error type for expected<T, E>.
  struct Error
  {
    enum class Code
    {
      kInvalidArgument = 1,
      kInternal = 2,
      kNotFound = 3,
      kFailedPrecondition = 4,
    };

    Code code;
    std::string message;
  };

  template < typename T >
  using expected = tl::expected < T, Error >;

/// Create an error.
  inline Error make_error(Error::Code code, std::string_view message = "")
  {
    return Error {code, std::string(message)};
  }

/// Macro for early return on error.
#define RETURN_IF_ERROR(expr) \
        do { \
          auto _status_or_value = (expr); \
          if (!_status_or_value.has_value()) { \
            return tl::unexpected(_status_or_value.error()); \
          } \
        } while (0)

/// Helper macros for unique variable names.
#define _IDC_CONCAT_IMPL(a, b) a ## b
#define _IDC_CONCAT(a, b) _IDC_CONCAT_IMPL(a, b)

/// Macro for assigning value or returning error.
/// Note: Cannot be wrapped in do/while because lhs must remain in scope.
/// Must be used inside braces (not after an unbraced if/else).
#define ASSIGN_OR_RETURN(lhs, expr) \
        auto _IDC_CONCAT(_result_, __LINE__) = (expr); \
        if (!_IDC_CONCAT(_result_, __LINE__).has_value()) { \
          return tl::unexpected(_IDC_CONCAT(_result_, __LINE__).error()); \
        } \
        lhs = std::move(*_IDC_CONCAT(_result_, __LINE__))

}  // namespace isaac_deploy_core

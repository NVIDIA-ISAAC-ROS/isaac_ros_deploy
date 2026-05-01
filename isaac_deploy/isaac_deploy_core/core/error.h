// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

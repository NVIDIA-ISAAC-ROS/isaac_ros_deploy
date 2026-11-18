// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/// Bridges Triton's inference trace activities onto NVTX, so a capture shows named phases instead
/// of anonymous kernels. Activities arrive on the worker thread, so spans are process-scoped.

#include <cstdint>

#include "triton/core/tritonserver.h"

#include "isaac_common/tracing/nvtx_scope.hpp"

#if defined(USE_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

namespace isaac_deploy_core::nvtx_trace
{

/// Phase durations as marks, not ranges: per tritonbackend.h the four COMPUTE_* timestamps are
/// reported in one burst after the fact, so a range across them would measure the burst.
enum Phase : int
{
  kPhaseAccept = 0,   ///< REQUEST_START -> QUEUE_START: request accepted into the scheduler.
  kPhaseQueue,        ///< QUEUE_START -> COMPUTE_START: scheduler backlog.
  kPhaseInput,        ///< COMPUTE_START -> COMPUTE_INPUT_END: input staging, incl. H2D.
  kPhaseExecute,      ///< COMPUTE_INPUT_END -> COMPUTE_OUTPUT_START: the model forward pass.
  kPhaseOutput,       ///< COMPUTE_OUTPUT_START -> COMPUTE_END: output extraction, incl. D2H.
  kPhaseReportLag,    ///< COMPUTE_END -> REQUEST_END: anchors the phases to the mark timestamp.
  kPhaseCount
};

/// Per-request state, heap-allocated in TritonRunner::run() and freed by release_fn.
struct TraceState
{
#if defined(USE_NVTX)
  nvtxRangeId_t request_range{};
#endif
  /// Triton's timestamps by activity. TENSOR_* and CUSTOM are not requested at TIMESTAMPS level.
  uint64_t ts[TRITONSERVER_TRACE_REQUEST_END + 1]{};
};

#if defined(USE_NVTX)

namespace detail
{

/// A dedicated domain keeps these clear of any other NVTX producer in the process.
inline nvtxDomainHandle_t domain()
{
  static nvtxDomainHandle_t handle = nvtxDomainCreateA("isaac_deploy.triton");
  return handle;
}

inline nvtxStringHandle_t request_name()
{
  static const nvtxStringHandle_t name =
    nvtxDomainRegisterStringA(domain(), "wbc:triton_request");
  return name;
}

/// Phase mark names, prefixed `wbc:` to stay distinct from the GR00T node's ensemble spans.
inline const nvtxStringHandle_t * phase_names()
{
  static const nvtxStringHandle_t names[kPhaseCount] = {
    nvtxDomainRegisterStringA(domain(), "wbc:triton_accept_ns (triton clock)"),
    nvtxDomainRegisterStringA(domain(), "wbc:triton_queue_ns (triton clock)"),
    nvtxDomainRegisterStringA(domain(), "wbc:triton_input_ns (triton clock)"),
    nvtxDomainRegisterStringA(domain(), "wbc:ort_execute_ns (triton clock)"),
    nvtxDomainRegisterStringA(domain(), "wbc:triton_output_ns (triton clock)"),
    nvtxDomainRegisterStringA(domain(), "wbc:triton_report_lag_ns (triton clock)"),
  };
  return names;
}

inline void open_request(TraceState * state)
{
  nvtxEventAttributes_t attr{};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.colorType = NVTX_COLOR_ARGB;
  attr.color = 0xFF808080;
  attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  attr.message.registered = request_name();
  state->request_range = nvtxDomainRangeStartEx(domain(), &attr);
}

inline void close_request(TraceState * state)
{
  if (state->request_range != 0) {
    nvtxDomainRangeEnd(domain(), state->request_range);
    state->request_range = 0;
  }
}

/// One phase duration from Triton's clock. Skipped if an activity is missing or the pair inverts.
inline void mark_phase_ns(
  const TraceState * state, Phase phase,
  TRITONSERVER_InferenceTraceActivity from, TRITONSERVER_InferenceTraceActivity to)
{
  const uint64_t t0 = state->ts[from];
  const uint64_t t1 = state->ts[to];
  // Zero-length phases are emitted, not skipped; see mark_phase() in triton_node.cpp.
  if (t0 == 0 || t1 < t0) {
    return;
  }

  nvtxEventAttributes_t attr{};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  attr.message.registered = phase_names()[phase];
  attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  attr.payload.ullValue = t1 - t0;
  nvtxDomainMarkEx(domain(), &attr);
}

}  // namespace detail

#endif  // USE_NVTX

/// Forwards to the one gate in nvtx_scope.hpp; Triton's trace machinery is not free.
inline bool enabled()
{
  return ::nvidia::isaac_ros::tracing::enabled();
}

/// Activity callback, invoked on whichever thread performed the activity, chronologically per
/// trace. Assumes TritonRunner's single-model config; an ensemble would need per-trace keying.
inline void activity_fn(
  TRITONSERVER_InferenceTrace * trace, TRITONSERVER_InferenceTraceActivity activity,
  uint64_t timestamp_ns, void * userp)
{
  (void)trace;
  auto * state = static_cast<TraceState *>(userp);
  if (state == nullptr) {
    return;
  }

#if defined(USE_NVTX)
  using detail::close_request;
  using detail::mark_phase_ns;
  using detail::open_request;

  if (activity <= TRITONSERVER_TRACE_REQUEST_END) {
    state->ts[activity] = timestamp_ns;
  }

  switch (activity) {
    case TRITONSERVER_TRACE_REQUEST_START:
      // Names the scheduler thread. Distinct from the GR00T node's -- different processes.
      ::nvidia::isaac_ros::tracing::name_thread("wbc:triton_sched");
      open_request(state);
      break;
    case TRITONSERVER_TRACE_COMPUTE_START:
      // Names the backend thread. Label only -- COMPUTE_* arrives as a burst after the fact.
      ::nvidia::isaac_ros::tracing::name_thread("wbc:triton_backend");
      break;
    case TRITONSERVER_TRACE_REQUEST_END:
      close_request(state);
      mark_phase_ns(
        state, kPhaseAccept, TRITONSERVER_TRACE_REQUEST_START,
        TRITONSERVER_TRACE_QUEUE_START);
      mark_phase_ns(
        state, kPhaseQueue, TRITONSERVER_TRACE_QUEUE_START, TRITONSERVER_TRACE_COMPUTE_START);
      mark_phase_ns(
        state, kPhaseInput, TRITONSERVER_TRACE_COMPUTE_START,
        TRITONSERVER_TRACE_COMPUTE_INPUT_END);
      mark_phase_ns(
        state, kPhaseExecute, TRITONSERVER_TRACE_COMPUTE_INPUT_END,
        TRITONSERVER_TRACE_COMPUTE_OUTPUT_START);
      mark_phase_ns(
        state, kPhaseOutput, TRITONSERVER_TRACE_COMPUTE_OUTPUT_START,
        TRITONSERVER_TRACE_COMPUTE_END);
      mark_phase_ns(
        state, kPhaseReportLag, TRITONSERVER_TRACE_COMPUTE_END,
        TRITONSERVER_TRACE_REQUEST_END);
      break;
    default:
      // QUEUE_START and the COMPUTE_* activities contribute timestamps only, recorded above.
      break;
  }
#else
  (void)activity;
  (void)timestamp_ns;
#endif
}

/// Release callback. Per the Triton C API it takes ownership of the trace and must delete it;
/// ServerInferAsync takes ownership unconditionally, so the caller must never delete it.
inline void release_fn(TRITONSERVER_InferenceTrace * trace, void * userp)
{
  auto * state = static_cast<TraceState *>(userp);
#if defined(USE_NVTX)
  if (state != nullptr) {
    detail::close_request(state);
  }
#endif
  delete state;

  if (trace != nullptr) {
    TRITONSERVER_Error * err = TRITONSERVER_InferenceTraceDelete(trace);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

}  // namespace isaac_deploy_core::nvtx_trace

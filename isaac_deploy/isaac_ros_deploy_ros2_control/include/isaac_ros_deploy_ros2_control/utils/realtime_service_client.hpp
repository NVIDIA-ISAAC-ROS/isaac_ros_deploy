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

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <rclcpp/rclcpp.hpp>

namespace isaac_ros_deploy_ros2_control
{
namespace utils
{

/// RT-safe one-shot trigger that sends the actual ROS service request from a non-RT timer.
template<typename ServiceT>
class RealtimeServiceClient
{
public:
  using Request = typename ServiceT::Request;
  using Client = rclcpp::Client<ServiceT>;
  using BuildRequestCallback = std::function<Request()>;
  // Return false to keep the one-shot request latched and retry it on the next timer tick.
  using ResponseCallback = std::function<bool(typename Client::SharedFuture)>;
  using FailureCallback = std::function<void(const std::string &)>;
  using TimeoutCallback = std::function<void(double)>;
  using ServiceUnavailableCallback = std::function<void()>;

  RealtimeServiceClient()
  : state_(std::make_shared<State>())
  {
  }
  ~RealtimeServiceClient()
  {
    reset();
  }

  template<typename NodeT>
  void configure(
    const std::shared_ptr<NodeT> & node,
    const std::string & service_name,
    std::chrono::milliseconds poll_period,
    double response_timeout_s,
    BuildRequestCallback build_request,
    ResponseCallback on_response,
    ServiceUnavailableCallback on_service_unavailable,
    FailureCallback on_send_failure,
    TimeoutCallback on_response_timeout)
  {
    reset();

    auto state = std::make_shared<State>();
    state->client = node->template create_client<ServiceT>(service_name);
    state->build_request = std::move(build_request);
    state->on_response = std::move(on_response);
    state->on_service_unavailable = std::move(on_service_unavailable);
    state->on_send_failure = std::move(on_send_failure);
    state->on_response_timeout = std::move(on_response_timeout);
    state->response_timeout_s = response_timeout_s;

    std::weak_ptr<State> weak_state = state;
    timer_ = node->create_wall_timer(
      poll_period, [weak_state]() {
        if (const auto locked_state = weak_state.lock()) {
          process(locked_state);
        }
      });
    state_ = std::move(state);
  }

  bool trigger_once()
  {
    const auto state = state_;
    if (!state || !state->active.load()) {
      return false;
    }

    bool expected_unlatched = false;
    if (!state->latched.compare_exchange_strong(expected_unlatched, true)) {
      return false;
    }
    state->pending.store(true);
    return true;
  }

  void reset()
  {
    const auto state = state_;
    if (state) {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->active.store(false);
      state->latched.store(false);
      state->pending.store(false);
      state->in_flight.store(false);
      state->client.reset();
    }

    if (timer_) {
      timer_->cancel();
      timer_.reset();
    }
    state_.reset();
  }

  bool active() const
  {
    const auto state = state_;
    return state && state->active.load();
  }

  bool latched() const
  {
    const auto state = state_;
    return state && state->latched.load();
  }

  bool pending() const
  {
    const auto state = state_;
    return state && state->pending.load();
  }

  bool in_flight() const
  {
    const auto state = state_;
    return state && state->in_flight.load();
  }

private:
  struct State
  {
    std::mutex mutex;
    std::atomic_bool active{true};
    std::atomic_bool latched{false};
    std::atomic_bool pending{false};
    std::atomic_bool in_flight{false};
    std::chrono::steady_clock::time_point in_flight_started_at{};
    double response_timeout_s{1.0};
    typename Client::SharedPtr client;
    BuildRequestCallback build_request;
    ResponseCallback on_response;
    ServiceUnavailableCallback on_service_unavailable;
    FailureCallback on_send_failure;
    TimeoutCallback on_response_timeout;
  };

  static void process(const std::shared_ptr<State> & state)
  {
    if (!state || !state->active.load()) {
      return;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    if (!state->active.load()) {
      return;
    }

    if (state->in_flight.load()) {
      const auto elapsed = std::chrono::steady_clock::now() - state->in_flight_started_at;
      if (elapsed > std::chrono::duration<double>(state->response_timeout_s)) {
        state->in_flight.store(false);
        state->pending.store(true);
        if (state->on_response_timeout) {
          state->on_response_timeout(state->response_timeout_s);
        }
      }
      return;
    }

    if (!state->pending.load()) {
      return;
    }

    bool expected_in_flight = false;
    if (!state->in_flight.compare_exchange_strong(expected_in_flight, true)) {
      return;
    }

    if (!state->client || !state->client->service_is_ready()) {
      state->in_flight.store(false);
      if (state->on_service_unavailable) {
        state->on_service_unavailable();
      }
      return;
    }

    auto request = std::make_shared<Request>(state->build_request());
    std::weak_ptr<State> weak_state = state;
    state->pending.store(false);
    state->in_flight_started_at = std::chrono::steady_clock::now();
    try {
      (void)state->client->async_send_request(
        request,
        [weak_state](typename Client::SharedFuture future) {
          const auto locked_state = weak_state.lock();
          if (!locked_state || !locked_state->active.load()) {
            return;
          }
          locked_state->in_flight.store(false);
          if (locked_state->on_response) {
            if (!locked_state->on_response(std::move(future)) && locked_state->active.load()) {
              locked_state->pending.store(true);
            }
          }
        });
    } catch (const std::exception & e) {
      state->pending.store(true);
      state->in_flight.store(false);
      if (state->on_send_failure) {
        state->on_send_failure(e.what());
      }
    }
  }

  std::shared_ptr<State> state_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace utils
}  // namespace isaac_ros_deploy_ros2_control

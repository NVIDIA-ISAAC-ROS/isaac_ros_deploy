// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/runner/triton_runner.h"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <sstream>
#include <thread>

#include "triton/core/tritonserver.h"

namespace isaac_deploy_core {

namespace {

/// Maximum number of poll attempts when waiting for server/model readiness.
constexpr int kMaxReadinessPollAttempts{2000};

/// Interval between readiness poll attempts.
constexpr auto kReadinessPollInterval{std::chrono::milliseconds(5)};

/// Check a Triton C API call and return our Error type on failure.
#define RETURN_TRITON_ERROR(expr)                                         \
  do {                                                                    \
    TRITONSERVER_Error* _triton_err = (expr);                             \
    if (_triton_err != nullptr) {                                         \
      const std::string _msg = TRITONSERVER_ErrorMessage(_triton_err);    \
      TRITONSERVER_ErrorDelete(_triton_err);                              \
      return tl::unexpected(make_error(Error::Code::kInternal, _msg));    \
    }                                                                     \
  } while (0)

/// CPU-only response allocator: allocates output buffers with malloc.
TRITONSERVER_Error* ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name, size_t byte_size,
    TRITONSERVER_MemoryType preferred_memory_type, int64_t preferred_memory_type_id, void* userp,
    void** buffer, void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id) {
  (void)allocator;
  (void)tensor_name;
  (void)preferred_memory_type;
  (void)preferred_memory_type_id;
  (void)userp;

  // Always allocate on CPU regardless of preference.
  *actual_memory_type = TRITONSERVER_MEMORY_CPU;
  *actual_memory_type_id = 0;
  *buffer_userp = nullptr;

  if (byte_size == 0) {
    *buffer = nullptr;
    return nullptr;
  }

  *buffer = malloc(byte_size);
  if (*buffer == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "failed to allocate output buffer");
  }
  return nullptr;
}

/// CPU-only response release: frees buffers allocated by ResponseAlloc.
TRITONSERVER_Error* ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp, size_t byte_size,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id) {
  (void)allocator;
  (void)buffer_userp;
  (void)byte_size;
  (void)memory_type;
  (void)memory_type_id;

  free(buffer);
  return nullptr;
}

/// Callback invoked when inference completes. Sets the promise with the response.
void InferenceComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp) {
  if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0) {
    auto* promise =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    if (promise != nullptr) {
      promise->set_value(response);
    }
  }
}

/// Callback invoked when an inference request is released. Sets a barrier promise.
void InferRequestRelease(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp) {
  (void)request;
  (void)flags;
  auto* barrier = reinterpret_cast<std::promise<void>*>(userp);
  if (barrier != nullptr) {
    barrier->set_value();
  }
}

/// Parse a JSON array of objects from Triton metadata to extract name and shape for each
/// input/output. The expected format is:
///   [{"name":"obs","datatype":"FP32","shape":[-1,42]}, ...]
///
/// This is a minimal parser that avoids adding a JSON library dependency.
struct TensorMetadata {
  std::string name;
  std::vector<int64_t> shape;
};

std::vector<TensorMetadata> parse_tensor_array(const std::string& json, const std::string& key) {
  std::vector<TensorMetadata> result;

  // Find the key, e.g. "inputs":[ or "outputs":[
  const std::string search_key = "\"" + key + "\"";
  const auto key_pos = json.find(search_key);
  if (key_pos == std::string::npos) {
    return result;
  }

  // Find the opening bracket of the array.
  const auto array_start = json.find('[', key_pos + search_key.size());
  if (array_start == std::string::npos) {
    return result;
  }

  // Find the matching closing bracket (handles nested brackets in shape arrays).
  int depth{1};
  auto array_end = array_start + 1;
  for (; array_end < json.size() && depth > 0; ++array_end) {
    if (json[array_end] == '[') {
      ++depth;
    } else if (json[array_end] == ']') {
      --depth;
    }
  }

  const std::string array_str = json.substr(array_start, array_end - array_start);

  // Parse each object within the array.
  size_t pos{0};
  while (pos < array_str.size()) {
    const auto obj_start = array_str.find('{', pos);
    if (obj_start == std::string::npos) {
      break;
    }
    const auto obj_end = array_str.find('}', obj_start);
    if (obj_end == std::string::npos) {
      break;
    }
    const std::string obj = array_str.substr(obj_start, obj_end - obj_start + 1);

    TensorMetadata meta;

    // Extract "name":"value"
    const auto name_key_pos = obj.find("\"name\"");
    if (name_key_pos != std::string::npos) {
      const auto colon_pos = obj.find(':', name_key_pos + 6);
      if (colon_pos != std::string::npos) {
        const auto quote1 = obj.find('"', colon_pos + 1);
        if (quote1 != std::string::npos) {
          const auto quote2 = obj.find('"', quote1 + 1);
          if (quote2 != std::string::npos) {
            meta.name = obj.substr(quote1 + 1, quote2 - quote1 - 1);
          }
        }
      }
    }

    // Extract "shape":[d1,d2,...]
    const auto shape_key_pos = obj.find("\"shape\"");
    if (shape_key_pos != std::string::npos) {
      const auto bracket_start = obj.find('[', shape_key_pos + 7);
      if (bracket_start != std::string::npos) {
        const auto bracket_end = obj.find(']', bracket_start);
        if (bracket_end != std::string::npos) {
          const std::string shape_str =
              obj.substr(bracket_start + 1, bracket_end - bracket_start - 1);
          std::istringstream ss(shape_str);
          std::string token;
          while (std::getline(ss, token, ',')) {
            // Trim whitespace.
            const auto start = token.find_first_not_of(" \t");
            if (start == std::string::npos) {
              continue;
            }
            const auto trimmed = token.substr(start);
            meta.shape.push_back(std::stoll(trimmed));
          }
        }
      }
    }

    if (!meta.name.empty()) {
      result.push_back(std::move(meta));
    }
    pos = obj_end + 1;
  }

  return result;
}

}  // namespace

TritonRunner::TritonRunner(const TritonRunnerConfig& config) : config_(config) {}

TritonRunner::~TritonRunner() {
  if (server_ != nullptr) {
    TRITONSERVER_ServerStop(server_);

    // Poll until the server is no longer live.
    bool server_live{true};
    for (int i = 0; i < kMaxReadinessPollAttempts && server_live; ++i) {
      TRITONSERVER_Error* err = TRITONSERVER_ServerIsLive(server_, &server_live);
      if (err != nullptr) {
        TRITONSERVER_ErrorDelete(err);
        break;
      }
      if (server_live) {
        std::this_thread::sleep_for(kReadinessPollInterval);
      }
    }

    TRITONSERVER_ServerDelete(server_);
    server_ = nullptr;
  }

  if (allocator_ != nullptr) {
    TRITONSERVER_ResponseAllocatorDelete(allocator_);
    allocator_ = nullptr;
  }

  // Clean up the temporary model repository.
  if (!model_repo_dir_.empty()) {
    std::error_code ec;
    std::filesystem::remove_all(model_repo_dir_, ec);
  }
}

expected<std::unique_ptr<TritonRunner>> TritonRunner::create(const TritonRunnerConfig& config) {
  if (!std::filesystem::exists(config.model_path)) {
    return tl::unexpected(
        make_error(Error::Code::kNotFound,
                   "Model file not found: " + config.model_path.string()));
  }

  try {
    auto runner = std::unique_ptr<TritonRunner>(new TritonRunner(config));
    RETURN_IF_ERROR(runner->init());
    return runner;
  } catch (const std::exception& e) {
    return tl::unexpected(
        make_error(Error::Code::kInternal,
                   std::string("Failed to create Triton runner: ") + e.what()));
  }
}

expected<void> TritonRunner::init() {
  RETURN_IF_ERROR(setup_model_repo());
  RETURN_IF_ERROR(start_server());

  // Create the response allocator (used for all inference requests).
  RETURN_TRITON_ERROR(
      TRITONSERVER_ResponseAllocatorNew(&allocator_, ResponseAlloc, ResponseRelease, nullptr));

  RETURN_IF_ERROR(query_model_metadata());
  return {};
}

expected<void> TritonRunner::setup_model_repo() {
  // Create /tmp/triton_model_repo_<pid>_<counter>/model/1/
  static std::atomic<int> counter{0};
  const std::string repo_name =
      "triton_model_repo_" + std::to_string(getpid()) + "_" + std::to_string(counter++);
  model_repo_dir_ = std::filesystem::temp_directory_path() / repo_name;

  const auto model_version_dir = model_repo_dir_ / model_name_ / "1";
  std::error_code ec;
  std::filesystem::create_directories(model_version_dir, ec);
  if (ec) {
    return tl::unexpected(
        make_error(Error::Code::kInternal,
                   "Failed to create model repo directory: " + ec.message()));
  }

  // Symlink the .onnx file into the version directory.
  const auto target = model_version_dir / "model.onnx";
  const auto source = std::filesystem::absolute(config_.model_path);
  std::filesystem::create_symlink(source, target, ec);
  if (ec) {
    return tl::unexpected(
        make_error(Error::Code::kInternal,
                   "Failed to symlink model file: " + ec.message()));
  }

  return {};
}

expected<void> TritonRunner::start_server() {
  TRITONSERVER_ServerOptions* options{nullptr};
  RETURN_TRITON_ERROR(TRITONSERVER_ServerOptionsNew(&options));

  // RAII guard ensures options are freed on all paths.
  struct OptionsDeleter {
    void operator()(TRITONSERVER_ServerOptions* opts) const {
      if (opts != nullptr) {
        TRITONSERVER_ServerOptionsDelete(opts);
      }
    }
  };
  std::unique_ptr<TRITONSERVER_ServerOptions, OptionsDeleter> options_guard(options);

  // Set model repository path.
  RETURN_TRITON_ERROR(
      TRITONSERVER_ServerOptionsSetModelRepositoryPath(
          options, model_repo_dir_.c_str()));

  // Automatic model configuration (no config.pbtxt needed).
  RETURN_TRITON_ERROR(TRITONSERVER_ServerOptionsSetStrictModelConfig(options, false));

  // Set backend directory: prefer explicit config, then env var.
  const auto resolve_backend_dir = [this]() -> std::filesystem::path {
    if (!config_.backend_dir.empty()) {
      return config_.backend_dir;
    }
    const char* env_val = std::getenv("TRITON_BACKEND_DIRECTORY");
    return (env_val != nullptr) ? std::filesystem::path(env_val) : std::filesystem::path{};
  };
  const auto backend_dir = resolve_backend_dir();
  if (!backend_dir.empty() && std::filesystem::is_directory(backend_dir)) {
    RETURN_TRITON_ERROR(
        TRITONSERVER_ServerOptionsSetBackendDirectory(options, backend_dir.c_str()));
  }

  // Quiet logging: warnings only.
  RETURN_TRITON_ERROR(TRITONSERVER_ServerOptionsSetLogVerbose(options, 0));
  RETURN_TRITON_ERROR(TRITONSERVER_ServerOptionsSetLogInfo(options, false));
  RETURN_TRITON_ERROR(
      TRITONSERVER_ServerOptionsSetServerId(options, "isaac_deploy_triton"));

  // Create the server. The guard ensures options are freed whether this succeeds or fails.
  TRITONSERVER_Error* err = TRITONSERVER_ServerNew(&server_, options);
  options_guard.reset();  // Free options now (no longer needed after ServerNew).
  if (err != nullptr) {
    const std::string msg = TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
    return tl::unexpected(
        make_error(Error::Code::kInternal, "Failed to create Triton server: " + msg));
  }

  // Wait for the server and model to become ready.
  bool server_ready{false};
  for (int i = 0; i < kMaxReadinessPollAttempts; ++i) {
    TRITONSERVER_Error* poll_err = TRITONSERVER_ServerIsReady(server_, &server_ready);
    if (poll_err != nullptr) {
      TRITONSERVER_ErrorDelete(poll_err);
    }
    if (server_ready) {
      break;
    }
    std::this_thread::sleep_for(kReadinessPollInterval);
  }
  if (!server_ready) {
    return tl::unexpected(
        make_error(Error::Code::kInternal, "Triton server failed to become ready"));
  }

  bool model_ready{false};
  for (int i = 0; i < kMaxReadinessPollAttempts; ++i) {
    TRITONSERVER_Error* poll_err =
        TRITONSERVER_ServerModelIsReady(server_, model_name_.c_str(), -1, &model_ready);
    if (poll_err != nullptr) {
      TRITONSERVER_ErrorDelete(poll_err);
    }
    if (model_ready) {
      break;
    }
    std::this_thread::sleep_for(kReadinessPollInterval);
  }
  if (!model_ready) {
    return tl::unexpected(
        make_error(Error::Code::kInternal,
                   "Triton model '" + model_name_ + "' failed to become ready"));
  }

  return {};
}

expected<void> TritonRunner::query_model_metadata() {
  TRITONSERVER_Message* metadata_msg{nullptr};
  RETURN_TRITON_ERROR(
      TRITONSERVER_ServerModelMetadata(server_, model_name_.c_str(), -1, &metadata_msg));

  const char* json_buf{nullptr};
  size_t json_size{0};
  TRITONSERVER_Error* err =
      TRITONSERVER_MessageSerializeToJson(metadata_msg, &json_buf, &json_size);
  if (err != nullptr) {
    const std::string msg = TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
    TRITONSERVER_MessageDelete(metadata_msg);
    return tl::unexpected(make_error(Error::Code::kInternal, msg));
  }

  const std::string json(json_buf, json_size);
  TRITONSERVER_MessageDelete(metadata_msg);

  // Parse inputs.
  const auto inputs = parse_tensor_array(json, "inputs");
  for (const auto& input : inputs) {
    input_names_.push_back(input.name);
    auto shape = input.shape;
    // Replace -1 (dynamic dims) with 1.
    for (auto& dim : shape) {
      if (dim == -1) {
        dim = 1;
      }
    }
    input_shapes_.push_back(std::move(shape));
  }

  // Parse outputs.
  const auto outputs = parse_tensor_array(json, "outputs");
  for (const auto& output : outputs) {
    output_names_.push_back(output.name);
    auto shape = output.shape;
    for (auto& dim : shape) {
      if (dim == -1) {
        dim = 1;
      }
    }
    output_shapes_.push_back(std::move(shape));
  }

  if (input_names_.empty()) {
    return tl::unexpected(
        make_error(Error::Code::kInternal, "No inputs found in model metadata"));
  }
  if (output_names_.empty()) {
    return tl::unexpected(
        make_error(Error::Code::kInternal, "No outputs found in model metadata"));
  }

  return {};
}

expected<void> TritonRunner::run(const TensorDict& inputs, TensorDict& outputs) {
  // Create inference request.
  TRITONSERVER_InferenceRequest* request{nullptr};
  RETURN_TRITON_ERROR(
      TRITONSERVER_InferenceRequestNew(&request, server_, model_name_.c_str(), -1));

  // Ensure request is cleaned up on all paths.
  struct RequestDeleter {
    void operator()(TRITONSERVER_InferenceRequest* req) const {
      if (req != nullptr) {
        TRITONSERVER_InferenceRequestDelete(req);
      }
    }
  };
  std::unique_ptr<TRITONSERVER_InferenceRequest, RequestDeleter> request_guard(request);

  // Keep contiguous tensors alive until inference completes.
  std::vector<torch::Tensor> input_tensors;
  input_tensors.reserve(input_names_.size());

  // Add inputs.
  for (size_t i = 0; i < input_names_.size(); ++i) {
    const auto& input_name = input_names_[i];
    const auto it = inputs.find(input_name);
    if (it == inputs.end()) {
      return tl::unexpected(
          make_error(Error::Code::kNotFound, "Input key not found: " + input_name));
    }

    torch::Tensor tensor = it->second.to(torch::kFloat32).contiguous();
    const auto& expected_shape = input_shapes_[i];

    // Verify element count matches.
    int64_t expected_numel{1};
    for (const auto dim : expected_shape) {
      expected_numel *= dim;
    }
    if (tensor.numel() != expected_numel) {
      std::ostringstream oss;
      oss << "Input '" << input_name << "' size mismatch: got " << tensor.numel()
          << " elements, expected " << expected_numel;
      return tl::unexpected(make_error(Error::Code::kInvalidArgument, oss.str()));
    }

    tensor = tensor.reshape(expected_shape);
    input_tensors.push_back(tensor);

    RETURN_TRITON_ERROR(TRITONSERVER_InferenceRequestAddInput(
        request, input_name.c_str(), TRITONSERVER_TYPE_FP32, expected_shape.data(),
        expected_shape.size()));

    const auto byte_size = static_cast<size_t>(tensor.numel()) * sizeof(float);
    RETURN_TRITON_ERROR(TRITONSERVER_InferenceRequestAppendInputData(
        request, input_name.c_str(), tensor.data_ptr<float>(), byte_size,
        TRITONSERVER_MEMORY_CPU, 0));
  }

  // Request outputs.
  for (const auto& output_name : output_names_) {
    RETURN_TRITON_ERROR(
        TRITONSERVER_InferenceRequestAddRequestedOutput(request, output_name.c_str()));
  }

  // Set up synchronous completion via promise/future.
  std::promise<TRITONSERVER_InferenceResponse*> response_promise;
  std::future<TRITONSERVER_InferenceResponse*> response_future = response_promise.get_future();

  std::promise<void> release_barrier;
  std::future<void> release_future = release_barrier.get_future();

  RETURN_TRITON_ERROR(TRITONSERVER_InferenceRequestSetReleaseCallback(
      request, InferRequestRelease, reinterpret_cast<void*>(&release_barrier)));

  RETURN_TRITON_ERROR(TRITONSERVER_InferenceRequestSetResponseCallback(
      request, allocator_, nullptr, InferenceComplete,
      reinterpret_cast<void*>(&response_promise)));

  // Release ownership: Triton takes over the request lifecycle.
  RETURN_TRITON_ERROR(TRITONSERVER_ServerInferAsync(server_, request_guard.release(), nullptr));

  // Block until the response arrives.
  TRITONSERVER_InferenceResponse* response = response_future.get();

  // Check for inference errors. The error is owned by the response, so do not delete it
  // separately — TRITONSERVER_InferenceResponseDelete handles cleanup.
  TRITONSERVER_Error* resp_err = TRITONSERVER_InferenceResponseError(response);
  if (resp_err != nullptr) {
    const std::string msg = TRITONSERVER_ErrorMessage(resp_err);
    TRITONSERVER_InferenceResponseDelete(response);
    return tl::unexpected(make_error(Error::Code::kInternal, "Inference failed: " + msg));
  }

  // Read output tensors from the response.
  uint32_t output_count{0};
  TRITONSERVER_Error* count_err =
      TRITONSERVER_InferenceResponseOutputCount(response, &output_count);
  if (count_err != nullptr) {
    const std::string msg = TRITONSERVER_ErrorMessage(count_err);
    TRITONSERVER_ErrorDelete(count_err);
    TRITONSERVER_InferenceResponseDelete(response);
    return tl::unexpected(make_error(Error::Code::kInternal, msg));
  }

  for (uint32_t i = 0; i < output_count; ++i) {
    const char* name{nullptr};
    TRITONSERVER_DataType dtype{};
    const int64_t* shape{nullptr};
    uint64_t dims_count{0};
    const void* buffer{nullptr};
    size_t byte_size{0};
    TRITONSERVER_MemoryType memory_type{};
    int64_t memory_type_id{0};
    void* userp{nullptr};

    TRITONSERVER_Error* out_err = TRITONSERVER_InferenceResponseOutput(
        response, i, &name, &dtype, &shape, &dims_count, &buffer, &byte_size, &memory_type,
        &memory_type_id, &userp);
    if (out_err != nullptr) {
      const std::string msg = TRITONSERVER_ErrorMessage(out_err);
      TRITONSERVER_ErrorDelete(out_err);
      TRITONSERVER_InferenceResponseDelete(response);
      return tl::unexpected(make_error(Error::Code::kInternal, msg));
    }

    // Build shape vector and create a torch tensor by copying the data.
    const std::vector<int64_t> tensor_shape(shape, shape + dims_count);
    const torch::Tensor torch_output =
        torch::from_blob(
            const_cast<void*>(buffer), torch::IntArrayRef(tensor_shape.data(), tensor_shape.size()),
            torch::kFloat32)
            .clone();

    outputs[std::string(name)] = torch_output;
  }

  // Wait for request release and clean up the response.
  release_future.get();
  TRITONSERVER_InferenceResponseDelete(response);

  return {};
}

std::vector<std::string> TritonRunner::get_input_names() const { return input_names_; }

std::vector<std::string> TritonRunner::get_output_names() const { return output_names_; }

void TritonRunner::reset() {
  // Triton models are stateless, nothing to reset.
}

}  // namespace isaac_deploy_core

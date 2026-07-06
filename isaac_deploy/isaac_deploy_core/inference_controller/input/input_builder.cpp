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

#include "isaac_deploy_core/inference_controller/input/input_builder.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace isaac_deploy_core {

  namespace {

    std::vector < std::string > remove_duplicates(std::vector < std::string > values) {
      std::sort(values.begin(), values.end());
      values.erase(std::unique(values.begin(), values.end()), values.end());
      return values;
    }

  }  // namespace

  expected < InputBuilder::Config >
  InputBuilder::Config::create_from_model_config(const ModelConfig & model_config) {
    if (!model_config.inputs.IsSequence()) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "InputBuilder::Config: model must contain an 'inputs' sequence"));
    }

    Config config;

    for (const auto & term_yaml : model_config.inputs) {
      auto term_result = InputTermConfig::create_from_yaml(term_yaml);
      if (!term_result.has_value()) {
        return tl::unexpected(term_result.error());
      }
      config.terms.push_back(std::move(*term_result));
    }

    // Apply feedback_connections: for each output_name -> [input_names],
    // find matching input terms by name and set their output_key + source.
    for (const auto &[output_name, input_names] : model_config.feedback_connections) {
      for (const auto & input_name : input_names) {
        auto it = std::find_if(
          config.terms.begin(), config.terms.end(),
          [&](const auto & term) {return term.name == input_name;});
        if (it == config.terms.end()) {
          return tl::unexpected(
            make_error(
              Error::Code::kInvalidArgument,
              "feedback_connections references unknown input '" + input_name + "'"));
        }
        it->output_key = output_name;
        // Reject if source was explicitly set in YAML to something different.
        if (!it->source.empty() && it->source != output_name) {
          return tl::unexpected(
            make_error(
              Error::Code::kInvalidArgument,
              "feedback_connections: input '" + input_name +
              "' has explicit source '" + it->source +
              "' which conflicts with feedback output '" + output_name + "'"));
        }
        it->source = output_name;
      }
    }

    // Apply source defaults now that feedback_connections are resolved.
    for (auto & term : config.terms) {
      if (term.source.empty()) {
        if (!term.kind.empty()) {
          term.source = term.kind;
        } else {
          term.source = term.name;
        }
      }
    }

    return config;
  }

  InputBuilder::InputBuilder(std::vector < InputTerm > terms)
  : terms_(std::move(terms)) {}

  expected < InputBuilder > InputBuilder::create(const Config & config) {
    std::vector < InputTerm > terms;
    terms.reserve(config.terms.size());

    for (const auto & term_config : config.terms) {
      auto term_result = InputTerm::create(term_config);
      if (!term_result.has_value()) {
        return tl::unexpected(term_result.error());
      }
      terms.push_back(std::move(*term_result));
    }

    return InputBuilder(std::move(terms));
  }

  expected < void > InputBuilder::activate(
    const std::vector < NamedTensor > &inputs,
    const std::vector < TensorSpec > &input_specs) {
    if (inputs.size() != input_specs.size()) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "inputs and input_specs must have the same size"));
    }

    // Build source→input index lookup from the inputs vector.
    std::unordered_map < std::string, size_t > source_to_input_idx;
    for (size_t i = 0; i < inputs.size(); ++i) {
      source_to_input_idx[inputs[i].name] = i;
    }

    input_to_terms_.clear();
    input_to_terms_.resize(inputs.size());
    term_outputs_.resize(terms_.size());

    for (size_t i = 0; i < terms_.size(); ++i) {
      auto & term = terms_[i];

      auto idx_it = source_to_input_idx.find(term.source());
      if (idx_it == source_to_input_idx.end()) {
        return tl::unexpected(
          make_error(
            Error::Code::kNotFound,
            "Input not found for source: " + term.source()));
      }
      size_t input_idx = idx_it->second;
      input_to_terms_[input_idx].push_back(i);

      auto result = term.activate(input_specs[input_idx], inputs[input_idx].tensor);
      if (!result.has_value()) {
        return tl::unexpected(result.error());
      }

      // Pre-allocate output tensor.
      term_outputs_[i] = torch::zeros(term.shape());
    }

    return expected < void > ();
  }

  expected < void > InputBuilder::deactivate() {
    for (auto & term : terms_) {
      auto result = term.deactivate();
      if (!result.has_value()) {
        return tl::unexpected(result.error());
      }
    }
    input_to_terms_.clear();
    term_outputs_.clear();
    return expected < void > ();
  }

  expected < void > InputBuilder::advance(
    const std::vector < NamedTensor > &inputs,
    TensorDict & outputs) {
    for (size_t i = 0; i < input_to_terms_.size(); ++i) {
      for (size_t term_idx : input_to_terms_[i]) {
        auto result = terms_[term_idx].advance(inputs[i].tensor, term_outputs_[term_idx]);
        if (!result.has_value()) {
          return tl::unexpected(result.error());
        }
        outputs[terms_[term_idx].name()] = term_outputs_[term_idx];
      }
    }

    return expected < void > ();
  }

  std::vector < std::string > InputBuilder::get_output_names() const {
    std::vector < std::string > names;
    names.reserve(terms_.size());
    for (const auto & term : terms_) {
      names.push_back(term.name());
    }
    return names;
  }

  std::vector < std::string > InputBuilder::get_unique_source_names() const {
    std::vector < std::string > sources;
    for (const auto & term : terms_) {
      sources.push_back(term.source());
    }
    return remove_duplicates(std::move(sources));
  }

  std::unordered_map < std::string, std::string > InputBuilder::get_source_to_kind_map() const {
    std::unordered_map < std::string, std::string > result;
    for (const auto & term : terms_) {
      if (term.depends_on_output()) {
        continue;
      }
      result[term.source()] = term.kind();
    }
    return result;
  }

  std::vector < std::string > InputBuilder::get_unique_kinds() const {
    std::vector < std::string > kinds;
    for (const auto & term : terms_) {
      if (!term.depends_on_output()) {
        kinds.push_back(term.kind());
      }
    }
    return remove_duplicates(std::move(kinds));
  }

  std::vector < std::string > InputBuilder::get_feedback_input_names() const {
    std::vector < std::string > names;
    for (const auto & term : terms_) {
      if (term.depends_on_output()) {
        names.push_back(term.output_key());
      }
    }
    return names;
  }

  std::vector < std::vector < int64_t >> InputBuilder::get_feedback_input_shapes() const {
    std::vector < std::vector < int64_t >> shapes;
    for (const auto & term : terms_) {
      if (term.depends_on_output()) {
        const auto & shape = term.shape();
        if (term.history_length() > 0 && shape.size() >= 3) {
          // Strip the history dimension: [batch, history, ...features] -> [batch, ...features].
          std::vector < int64_t > per_advance = {shape[0]};
          per_advance.insert(per_advance.end(), shape.begin() + 2, shape.end());
          shapes.push_back(std::move(per_advance));
        } else {
          shapes.push_back(shape);
        }
      }
    }
    return shapes;
  }

}  // namespace isaac_deploy_core

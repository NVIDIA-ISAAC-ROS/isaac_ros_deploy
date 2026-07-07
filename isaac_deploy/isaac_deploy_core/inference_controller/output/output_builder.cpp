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

#include "isaac_deploy_core/inference_controller/output/output_builder.hpp"

#include <utility>

namespace isaac_deploy_core {

  expected < OutputBuilder::Config >
  OutputBuilder::Config::create_from_model_config(const ModelConfig & model_config) {
    Config config;

    if (!model_config.outputs.IsSequence()) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "OutputBuilder::Config: model must contain an 'outputs' sequence"));
    }

    for (const auto & term_yaml : model_config.outputs) {
      auto term_result = OutputTermConfig::create_from_yaml(term_yaml);
      if (!term_result.has_value()) {
        return tl::unexpected(term_result.error());
      }
      config.terms.push_back(std::move(*term_result));
    }

    return config;
  }

  OutputBuilder::OutputBuilder(std::vector < OutputTerm > terms, double policy_dt_seconds)
  : terms_(std::move(terms)), policy_dt_seconds_(policy_dt_seconds) {}

  expected < OutputBuilder > OutputBuilder::create(const Config & config) {
    std::vector < OutputTerm > terms;
    terms.reserve(config.terms.size());

    for (const auto & term_config : config.terms) {
      auto term_result = OutputTerm::create(term_config);
      if (!term_result.has_value()) {
        return tl::unexpected(term_result.error());
      }
      terms.push_back(std::move(*term_result));
    }

    return OutputBuilder(std::move(terms), config.policy_dt_seconds);
  }

  expected < void > OutputBuilder::activate(
    const std::vector < TensorSpec > &output_specs,
    std::vector < NamedTensor > &outputs) {
    // Build name to output map.
    std::unordered_map < std::string, size_t > name_to_output;
    std::unordered_map < std::string, const TensorSpec * > name_to_spec;

    for (size_t i = 0; i < outputs.size(); ++i) {
      name_to_output[outputs[i].name] = i;
      if (i < output_specs.size()) {
        name_to_spec[outputs[i].name] = &output_specs[i];
      }
    }

    // Activate each term.
    name_to_term_.clear();
    for (size_t i = 0; i < terms_.size(); ++i) {
      auto & term = terms_[i];
      name_to_term_[term.name()] = i;

      auto output_it = name_to_output.find(term.name());
      if (output_it == name_to_output.end()) {
        return tl::unexpected(
          make_error(
            Error::Code::kNotFound,
            "Output not found for term: " + term.name()));
      }

      TensorSpec empty_spec;
      const TensorSpec * spec = &empty_spec;
      if (auto spec_it = name_to_spec.find(term.name()); spec_it != name_to_spec.end()) {
        spec = spec_it->second;
      }

      auto result = term.activate(*spec, outputs[output_it->second]);
      if (!result.has_value()) {
        return tl::unexpected(result.error());
      }
    }

    return expected < void > ();
  }

  expected < void > OutputBuilder::deactivate() {
    for (auto & term : terms_) {
      auto result = term.deactivate();
      if (!result.has_value()) {
        return tl::unexpected(result.error());
      }
    }
    name_to_term_.clear();
    return expected < void > ();
  }

  expected < void > OutputBuilder::advance(
    const TensorDict & nn_outputs,
    std::vector < NamedTensor > &outputs) {
    // Build name to output map.
    std::unordered_map < std::string, NamedTensor * > name_to_output;
    for (auto & output : outputs) {
      name_to_output[output.name] = &output;
    }

    // Process each term.
    for (auto & term : terms_) {
      auto nn_it = nn_outputs.find(term.name());
      if (nn_it == nn_outputs.end()) {
        return tl::unexpected(
          make_error(
            Error::Code::kNotFound,
            "Neural network output not found: " + term.name()));
      }

      auto output_it = name_to_output.find(term.name());
      if (output_it == name_to_output.end()) {
        return tl::unexpected(
          make_error(
            Error::Code::kNotFound,
            "Output not found: " + term.name()));
      }

      auto result = term.advance(nn_it->second, *output_it->second);
      if (!result.has_value()) {
        return tl::unexpected(result.error());
      }
    }

    return expected < void > ();
  }

  std::vector < std::string > OutputBuilder::get_input_names() const {
    std::vector < std::string > names;
    names.reserve(terms_.size());
    for (const auto & term : terms_) {
      names.push_back(term.name());
    }
    return names;
  }

  std::vector < std::string > OutputBuilder::get_output_names() const {
    std::vector < std::string > names;
    names.reserve(terms_.size());
    for (const auto & term : terms_) {
      names.push_back(term.name());
    }
    return names;
  }

  std::vector < std::vector < int64_t >> OutputBuilder::get_output_shapes() const {
    std::vector < std::vector < int64_t >> shapes;
    shapes.reserve(terms_.size());
    for (const auto & term : terms_) {
      shapes.push_back(term.shape());
    }
    return shapes;
  }

  std::unordered_map < std::string, std::string > OutputBuilder::get_output_to_kind_map() const {
    std::unordered_map < std::string, std::string > map;
    for (const auto & term : terms_) {
      map[term.name()] = term.kind();
    }
    return map;
  }

  std::vector < std::string > OutputBuilder::get_element_names(
    const std::string & output_name) const {
    for (const auto & term : terms_) {
      if (term.name() == output_name) {
        const auto & names = term.element_names();
        // Return the last non-empty dimension's names.
        for (auto it = names.rbegin(); it != names.rend(); ++it) {
          if (!it->empty()) {
            return *it;
          }
        }
        return {};
      }
    }
    return {};
  }

  double OutputBuilder::get_policy_dt_seconds() const
  {
    return policy_dt_seconds_;
  }

}  // namespace isaac_deploy_core

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_deploy_core/inference_controller/config_parser.h"

#include <unordered_set>
#include <utility>

namespace isaac_deploy_core {

  namespace {

    std::string strip_model_prefix(const std::string & s)
    {
      auto pos = s.find('/');
      if (pos == std::string::npos) {
        return s;
      }
      return s.substr(pos + 1);
    }

    std::string get_model_prefix(const std::string & s)
    {
      auto pos = s.find('/');
      if (pos == std::string::npos) {
        return "";
      }
      return s.substr(0, pos);
    }

    /// Parse a map<string, vector<string>> from a YAML map node.
    /// Keys and values are preserved as-is (including any model prefixes).
    expected < std::unordered_map < std::string, std::vector < std::string >> >
    parse_connection_map(const YAML::Node & node, const std::string & section_name)
    {
      std::unordered_map < std::string, std::vector < std::string >> result;
      if (!node.IsDefined()) {
        return result;
      }
      if (!node.IsMap()) {
        return tl::unexpected(
          make_error(
            Error::Code::kInvalidArgument,
            section_name + " must be a map"));
      }
      for (const auto & entry : node) {
        const std::string key = entry.first.as < std::string > ();
        if (!entry.second.IsSequence()) {
          return tl::unexpected(
            make_error(
              Error::Code::kInvalidArgument,
              section_name + " value for '" + key +
              "' must be a sequence"));
        }
        std::vector < std::string > values;
        for (const auto & v : entry.second) {
          values.push_back(v.as < std::string > ());
        }
        result[key] = std::move(values);
      }
      return result;
    }

  }  // namespace

  expected < GraphConfig > parse_graph_config(const YAML::Node & root)
  {
    if (!root["models"].IsDefined()) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "Config must contain a 'models' section"));
    }

    const auto & models = root["models"];
    if (!models.IsMap() || models.size() == 0) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "'models' section must be a non-empty map"));
    }

    GraphConfig graph;

    // Parse each model in YAML insertion order.
    for (const auto & model_entry : models) {
      const std::string model_name = model_entry.first.as < std::string > ();
      const auto & model = model_entry.second;

      ModelConfig config;

      if (!model["inputs"].IsDefined()) {
        return tl::unexpected(
          make_error(
            Error::Code::kInvalidArgument,
            "Model '" + model_name + "' must contain an 'inputs' section"));
      }
      config.inputs = model["inputs"];

      if (!model["outputs"].IsDefined()) {
        return tl::unexpected(
          make_error(
            Error::Code::kInvalidArgument,
            "Model '" + model_name + "' must contain an 'outputs' section"));
      }
      config.outputs = model["outputs"];

      if (model["parameters"].IsDefined()) {
        const auto & params = model["parameters"];
        if (params["model_path"].IsDefined()) {
          config.model_path = params["model_path"].as < std::string > ();
        }
        if (params["backend"].IsDefined()) {
          config.backend = params["backend"].as < std::string > ();
        }
      }

      // Per-model feedback_connections (from pipeline section).
      if (root["pipeline"].IsDefined()) {
        const auto & pipeline = root["pipeline"];
        if (pipeline["feedback_connections"].IsDefined()) {
          const auto & fc = pipeline["feedback_connections"];
          for (const auto & entry : fc) {
            const std::string full_key = entry.first.as < std::string > ();
            const std::string key_model = get_model_prefix(full_key);
            const std::string key_name = strip_model_prefix(full_key);
            if (key_model == model_name) {
              // Validate that the output name exists in this model's outputs.
              bool output_exists = false;
              if (model["outputs"].IsSequence()) {
                for (const auto & output : model["outputs"]) {
                  if (output["name"].as < std::string > () == key_name) {
                    output_exists = true;
                    break;
                  }
                }
              }
              if (!output_exists) {
                return tl::unexpected(
                  make_error(
                    Error::Code::kInvalidArgument,
                    "feedback_connections key '" + full_key +
                    "' refers to unknown output '" + key_name +
                    "' in model '" + model_name + "'"));
              }
              if (!entry.second.IsSequence()) {
                return tl::unexpected(
                  make_error(
                    Error::Code::kInvalidArgument,
                    "feedback_connections value for '" + full_key +
                    "' must be a sequence"));
              }
              std::vector < std::string > input_names;
              for (const auto & inp : entry.second) {
                const std::string input_name = strip_model_prefix(inp.as < std::string > ());

                input_names.push_back(input_name);
              }
              config.feedback_connections[key_name] = std::move(input_names);
            }
          }
        }
      }

      graph.models.emplace_back(model_name, std::move(config));
    }

    // Parse data_flow from pipeline section (keep full prefixed keys).
    if (root["pipeline"].IsDefined()) {
      const auto & pipeline = root["pipeline"];
      auto df_result = parse_connection_map(pipeline["data_flow"], "data_flow");
      if (!df_result) {
        return tl::unexpected(df_result.error());
      }
      graph.data_flow = std::move(*df_result);
    }

    return graph;
  }

  expected < ModelConfig > merge_graph_to_model_config(
    const GraphConfig & graph,
    const YAML::Node & root)
  {
    // Single-model: return as-is.
    if (graph.models.size() == 1) {
      return graph.models[0].second;
    }

    // Multi-model: merge into a single ModelConfig for InputBuilder/OutputBuilder.
    //
    // Build sets from pipeline section to decide which inputs/outputs to include.
    const auto & pipeline = root["pipeline"];
    if (!pipeline.IsDefined()) {
      return tl::unexpected(
        make_error(
          Error::Code::kInvalidArgument,
          "Multi-model configs require a 'pipeline' section"));
    }

    // Parse pipeline connection maps.
    auto inputs_result = parse_connection_map(
      pipeline["inputs"], "inputs");
    if (!inputs_result) {
      return tl::unexpected(inputs_result.error());
    }
    const auto & inputs = *inputs_result;

    auto outputs_result = parse_connection_map(
      pipeline["outputs"], "outputs");
    if (!outputs_result) {
      return tl::unexpected(outputs_result.error());
    }
    const auto & outputs = *outputs_result;

    auto data_flow_raw_result = parse_connection_map(
      pipeline["data_flow"], "data_flow");
    if (!data_flow_raw_result) {
      return tl::unexpected(data_flow_raw_result.error());
    }
    const auto & data_flow_raw = *data_flow_raw_result;

    // Build per-model name sets.
    // dangling_input_names[model] = set of input names from inputs
    std::unordered_map < std::string, std::unordered_set < std::string >> dangling_input_names;
    for (const auto &[model, names] : inputs) {
      dangling_input_names[model].insert(names.begin(), names.end());
    }

    // data_flow_target_names[model] = set of input names from data_flow values (to EXCLUDE)
    std::unordered_map < std::string, std::unordered_set < std::string >> data_flow_target_names;
    for (const auto &[key, targets] : data_flow_raw) {
      for (const auto & target : targets) {
        const std::string target_model = get_model_prefix(target);
        const std::string target_name = strip_model_prefix(target);
        data_flow_target_names[target_model].insert(target_name);
      }
    }

    // dangling_output_names[model] = set of output names from outputs
    std::unordered_map < std::string, std::unordered_set < std::string >> dangling_output_names_map;
    for (const auto &[model, names] : outputs) {
      dangling_output_names_map[model].insert(names.begin(), names.end());
    }

    // Merge into single ModelConfig.
    ModelConfig config;

    // Build merged inputs and outputs as YAML sequences.
    YAML::Node merged_inputs(YAML::NodeType::Sequence);
    YAML::Node merged_outputs(YAML::NodeType::Sequence);

    for (const auto &[model_name, model_config] : graph.models) {
      // Merge inputs: include only dangling inputs that are not data_flow targets.
      // Feedback and data_flow inputs are handled by InferenceRunnerNodes directly.
      if (model_config.inputs.IsSequence()) {
        for (const auto & input : model_config.inputs) {
          const std::string name = input["name"].as < std::string > ();
          const bool is_dangling = dangling_input_names[model_name].contains(name);
          const bool is_data_flow_target = data_flow_target_names[model_name].contains(name);

          if (is_dangling && !is_data_flow_target) {
            merged_inputs.push_back(input);
          }
        }
      }

      // Merge outputs: include only dangling outputs.
      // Feedback source outputs are published by InferenceRunnerNodes directly.
      if (model_config.outputs.IsSequence()) {
        for (const auto & output : model_config.outputs) {
          const std::string name = output["name"].as < std::string > ();
          const bool is_dangling = dangling_output_names_map[model_name].contains(name);

          if (is_dangling) {
            merged_outputs.push_back(output);
          }
        }
      }
    }

    config.inputs = merged_inputs;
    config.outputs = merged_outputs;

    // Feedback connections are handled by InferenceRunnerNodes, not InputBuilder.
    // Leave feedback_connections empty in the merged config.

    // model_path and backend left empty for merged config.

    return config;
  }

}  // namespace isaac_deploy_core

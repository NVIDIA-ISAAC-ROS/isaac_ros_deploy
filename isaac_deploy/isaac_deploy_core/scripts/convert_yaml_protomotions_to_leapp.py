#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a unified_pipeline YAML to the new multi-model graph pipeline format.

The old format has a single model described by policy_inputs/policy_outputs/_runtime.
The new format has a models dict (each with formatting, inputs, outputs, parameters)
and a pipeline section with inputs, outputs, and data_flow.
"""

import argparse
import sys
from pathlib import Path

import yaml


# Keys that belong to the old format's structural sections (not robot metadata).
_OLD_FORMAT_KEYS = {
    "type",
    "policy_inputs",
    "policy_outputs",
    "_runtime",
    "metadata",
}


def _convert_input(inp: dict) -> dict:
    """Convert a single policy_input entry to new format."""
    result = {
        "name": inp["key"],
        "kind": inp["kind"],
        "shape": inp["shape"],
        "type": "tensor",
    }

    if "element_names" in inp:
        result["element_names"] = inp["element_names"]

    history = inp.get("history", inp.get("history_length", 0))
    if history > 0:
        result["history"] = history

    if "include_current_value_in_history" in inp:
        result["include_current_value_in_history"] = inp[
            "include_current_value_in_history"
        ]
    elif "include_current_in_history" in inp:
        result["include_current_value_in_history"] = inp[
            "include_current_in_history"
        ]

    if "source" in inp:
        result["source"] = inp["source"]

    return result


_OUTPUT_KIND_MAP = {
    "joint_pos_targets": "target/joint/position",
    "stiffness_targets": "kp",
    "damping_targets": "kd",
}


def _convert_output(out: dict) -> dict:
    """Convert a single policy_output entry to new format."""
    result = {
        "name": out["key"],
        "kind": _OUTPUT_KIND_MAP.get(out["kind"], out["kind"]),
        "shape": out["shape"],
        "type": "tensor",
    }

    if "element_names" in out:
        result["element_names"] = out["element_names"]

    return result


def _build_input_format(
    policy_inputs: list, runtime: dict | None
) -> list:
    """Build the formatting.input_format list.

    Uses _runtime.onnx_in_names for ordering if available. Handles the case
    where ONNX names differ from pipeline keys (via onnx_name_to_in_key).
    """
    if runtime and "onnx_in_names" in runtime:
        onnx_in_names = runtime["onnx_in_names"]
        onnx_to_key = runtime.get("onnx_name_to_in_key", {})

        result = []
        for onnx_name in onnx_in_names:
            key = onnx_to_key.get(onnx_name, onnx_name)
            if onnx_name != key:
                # Pipeline uses key, tensor uses onnx_name -> {key: onnx_name}
                result.append({key: onnx_name})
            else:
                result.append(onnx_name)
        return result

    # Fallback: use key order from policy_inputs.
    return [inp["key"] for inp in policy_inputs]


def _build_output_format(
    policy_outputs: list, runtime: dict | None
) -> dict | str:
    """Build the formatting.output_format.

    Returns a string if single output, dict otherwise.
    """
    if runtime and "onnx_out_names" in runtime:
        names = runtime["onnx_out_names"]
    else:
        names = [out["key"] for out in policy_outputs]

    if len(names) == 1:
        return names[0]

    return {name: name for name in names}


def _build_parameters(protomotions_yaml: dict) -> dict:
    """Build model parameters from metadata."""
    params = {
        "backend": "triton",
        "device": "cuda",
    }

    metadata = protomotions_yaml.get("metadata", {})
    if "checkpoint" in metadata:
        params["model_path"] = metadata["checkpoint"]

    return params


def _derive_model_name(protomotions_yaml: dict) -> str:
    """Derive a model name from the pipeline type, falling back to 'policy'."""
    pipeline_type = protomotions_yaml.get("type", "")
    if pipeline_type and pipeline_type not in ("unified_pipeline",):
        return pipeline_type
    return "policy"


def convert(protomotions_yaml: dict) -> dict:
    """Convert protomotions yaml format to the LEAPP yaml format."""
    policy_inputs = protomotions_yaml["policy_inputs"]
    policy_outputs = protomotions_yaml["policy_outputs"]
    runtime = protomotions_yaml.get("_runtime")

    model_name = _derive_model_name(protomotions_yaml)
    inputs = [_convert_input(inp) for inp in policy_inputs]
    outputs = [_convert_output(out) for out in policy_outputs]

    input_format = _build_input_format(policy_inputs, runtime)
    output_format = _build_output_format(policy_outputs, runtime)

    model = {
        "formatting": {
            "input_format": input_format,
            "output_format": output_format,
        },
        "inputs": inputs,
        "outputs": outputs,
        "parameters": _build_parameters(protomotions_yaml),
    }

    # Build feedback_connections from old output_key fields.
    feedback_connections = {}
    feedback_input_names = set()
    for old_inp in policy_inputs:
        if "output_key" in old_inp:
            output_key = old_inp["output_key"]
            input_name = old_inp["key"]
            fc_key = f"{model_name}/{output_key}"
            feedback_connections.setdefault(fc_key, []).append(
                f"{model_name}/{input_name}"
            )
            feedback_input_names.add(input_name)

    # Dangling inputs exclude feedback inputs (they are internally connected).
    input_names = [inp["name"] for inp in inputs if inp["name"] not in feedback_input_names]
    output_names = [out["name"] for out in outputs]

    pipeline = {
        "inputs": {model_name: input_names},
        "outputs": {model_name: output_names},
    }
    if feedback_connections:
        pipeline["feedback_connections"] = feedback_connections
    pipeline["data_flow"] = {}

    # Preserve top-level robot metadata (joint_names, body_names, etc.).
    result = {}
    for key, value in protomotions_yaml.items():
        if key not in _OLD_FORMAT_KEYS:
            result[key] = value

    result["models"] = {model_name: model}
    result["pipeline"] = pipeline

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert unified_pipeline YAML to graph pipeline format."
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Path to the protomotions YAML file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the generated LEAPP yaml file",
    )
    args = parser.parse_args()

    protomotions_yaml = yaml.safe_load(args.input.read_text())

    result = convert(protomotions_yaml)

    yaml_str = yaml.dump(
        result,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )

    if args.output:
        args.output.write_text(yaml_str)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(yaml_str)


if __name__ == "__main__":
    main()

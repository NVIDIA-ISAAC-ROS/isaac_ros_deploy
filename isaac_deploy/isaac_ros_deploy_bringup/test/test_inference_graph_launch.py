#!/usr/bin/env python3

# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for inference_graph.launch.py helper and wiring logic."""

import ast
from pathlib import Path


BRINGUP_ROOT = Path(__file__).parents[1]


def _launch_tree():
    return ast.parse((BRINGUP_ROOT / "launch" / "inference_graph.launch.py").read_text())


def _load_launch_function(function_name):
    launch_file = BRINGUP_ROOT / "launch" / "inference_graph.launch.py"
    tree = ast.parse(launch_file.read_text())
    function_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.Module(body=[function_def], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(launch_file), "exec"), namespace)
    return namespace[function_name]


def test_csv_string_to_list_strips_empty_and_whitespace_entries():
    csv_string_to_list = _load_launch_function("csv_string_to_list")

    assert csv_string_to_list("") == []
    assert csv_string_to_list(" ") == []
    assert csv_string_to_list("policy") == ["policy"]
    assert csv_string_to_list(" policy , critic , ") == ["policy", "critic"]


def test_triton_cpu_models_are_forwarded_to_model_repo_creation():
    tree = _launch_tree()

    for call in [node for node in ast.walk(tree) if isinstance(node, ast.Call)]:
        if getattr(call.func, "id", "") != "create_triton_model_repo":
            continue
        keyword_names = {keyword.arg for keyword in call.keywords}
        assert "cpu_models" in keyword_names
        return

    raise AssertionError("create_triton_model_repo call not found")

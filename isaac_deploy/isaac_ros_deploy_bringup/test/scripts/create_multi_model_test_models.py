#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create two ONNX models for multi-model pipeline testing.

Model A: ao1 = ao2 = ai1 + ai2 + ai3   (3 inputs, 2 outputs)
Model B: bo1 = bo2 = bo3 = bi1 + bi2    (2 inputs, 3 outputs)

All tensors are shape [1, 1].
"""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def create_model_a(output_path: Path):
    """Create Model A: ao1 = ao2 = ai1 + ai2 + ai3."""
    # Inputs
    ai1 = helper.make_tensor_value_info("ai1", TensorProto.FLOAT, [1, 1])
    ai2 = helper.make_tensor_value_info("ai2", TensorProto.FLOAT, [1, 1])
    ai3 = helper.make_tensor_value_info("ai3", TensorProto.FLOAT, [1, 1])

    # Outputs
    ao1 = helper.make_tensor_value_info("ao1", TensorProto.FLOAT, [1, 1])
    ao2 = helper.make_tensor_value_info("ao2", TensorProto.FLOAT, [1, 1])

    # Nodes: sum12 = ai1 + ai2, internal = sum12 + ai3
    add_12 = helper.make_node(
        "Add", inputs=["ai1", "ai2"], outputs=["sum12"], name="add_12"
    )
    add_all = helper.make_node(
        "Add", inputs=["sum12", "ai3"], outputs=["internal"], name="add_all"
    )
    identity_ao1 = helper.make_node(
        "Identity", inputs=["internal"], outputs=["ao1"], name="identity_ao1"
    )
    identity_ao2 = helper.make_node(
        "Identity", inputs=["internal"], outputs=["ao2"], name="identity_ao2"
    )

    graph = helper.make_graph(
        nodes=[add_12, add_all, identity_ao1, identity_ao2],
        name="model_a",
        inputs=[ai1, ai2, ai3],
        outputs=[ao1, ao2],
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    print(f"Created Model A at: {output_path}")


def create_model_b(output_path: Path):
    """Create Model B: bo1 = bo2 = bo3 = bi1 + bi2."""
    # Inputs
    bi1 = helper.make_tensor_value_info("bi1", TensorProto.FLOAT, [1, 1])
    bi2 = helper.make_tensor_value_info("bi2", TensorProto.FLOAT, [1, 1])

    # Outputs
    bo1 = helper.make_tensor_value_info("bo1", TensorProto.FLOAT, [1, 1])
    bo2 = helper.make_tensor_value_info("bo2", TensorProto.FLOAT, [1, 1])
    bo3 = helper.make_tensor_value_info("bo3", TensorProto.FLOAT, [1, 1])

    # Nodes: internal = bi1 + bi2
    add_inputs = helper.make_node(
        "Add", inputs=["bi1", "bi2"], outputs=["internal"], name="add_inputs"
    )
    identity_bo1 = helper.make_node(
        "Identity", inputs=["internal"], outputs=["bo1"], name="identity_bo1"
    )
    identity_bo2 = helper.make_node(
        "Identity", inputs=["internal"], outputs=["bo2"], name="identity_bo2"
    )
    identity_bo3 = helper.make_node(
        "Identity", inputs=["internal"], outputs=["bo3"], name="identity_bo3"
    )

    graph = helper.make_graph(
        nodes=[add_inputs, identity_bo1, identity_bo2, identity_bo3],
        name="model_b",
        inputs=[bi1, bi2],
        outputs=[bo1, bo2, bo3],
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    print(f"Created Model B at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create multi-model ONNX test models."
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Output directory for the ONNX models",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    create_model_a(args.output_dir / "model_a.onnx")
    create_model_b(args.output_dir / "model_b.onnx")


if __name__ == "__main__":
    main()

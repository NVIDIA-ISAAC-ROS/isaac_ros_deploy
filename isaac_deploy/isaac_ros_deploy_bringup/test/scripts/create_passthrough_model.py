#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Create a passthrough ONNX model for testing the inference graph."""

import argparse
from pathlib import Path

import onnx
from onnx import helper, TensorProto


def create_passthrough_onnx(output_path: Path, num_joints: int = 3):
    """Create a passthrough ONNX model using onnx helper.

    Args:
        output_path: Path to save the ONNX model.
        num_joints: Number of joints (default 3).
    """
    # Define inputs
    joint_pos = helper.make_tensor_value_info(
        "joint_pos", TensorProto.FLOAT, [1, num_joints]
    )
    joint_vel = helper.make_tensor_value_info(
        "joint_vel", TensorProto.FLOAT, [1, num_joints]
    )
    body_rot = helper.make_tensor_value_info(
        "body_rot", TensorProto.FLOAT, [1, 4]
    )
    body_ang_vel = helper.make_tensor_value_info(
        "body_ang_vel", TensorProto.FLOAT, [1, 3]
    )
    last_joint_pos = helper.make_tensor_value_info(
        "last_joint_pos", TensorProto.FLOAT, [1, num_joints]
    )

    # Define outputs
    joint_pos_targets = helper.make_tensor_value_info(
        "joint_pos_targets", TensorProto.FLOAT, [1, num_joints]
    )
    joint_vel_targets = helper.make_tensor_value_info(
        "joint_vel_targets", TensorProto.FLOAT, [1, num_joints]
    )
    body_rot_target = helper.make_tensor_value_info(
        "body_rot_target", TensorProto.FLOAT, [1, 4]
    )
    body_ang_vel_target = helper.make_tensor_value_info(
        "body_ang_vel_target", TensorProto.FLOAT, [1, 3]
    )

    # Create Identity nodes (passthrough)
    identity_pos = helper.make_node(
        "Identity",
        inputs=["joint_pos"],
        outputs=["joint_pos_targets"],
        name="identity_pos",
    )
    identity_vel = helper.make_node(
        "Identity",
        inputs=["joint_vel"],
        outputs=["joint_vel_targets"],
        name="identity_vel",
    )
    identity_body_rot = helper.make_node(
        "Identity",
        inputs=["body_rot"],
        outputs=["body_rot_target"],
        name="identity_body_rot",
    )
    identity_body_ang_vel = helper.make_node(
        "Identity",
        inputs=["body_ang_vel"],
        outputs=["body_ang_vel_target"],
        name="identity_body_ang_vel",
    )

    # Create graph (last_joint_pos is an input but not connected to any output —
    # it exists so the model accepts the feedback tensor from InputBuilder).
    graph = helper.make_graph(
        nodes=[identity_pos, identity_vel, identity_body_rot, identity_body_ang_vel],
        name="passthrough",
        inputs=[joint_pos, joint_vel, body_rot, body_ang_vel, last_joint_pos],
        outputs=[
            joint_pos_targets, joint_vel_targets,
            body_rot_target, body_ang_vel_target,
        ],
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    print(f"Created passthrough ONNX model at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a passthrough ONNX model for testing."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "passthrough_model.onnx",
        help="Output path for the ONNX model",
    )
    parser.add_argument(
        "--num-joints",
        "-n",
        type=int,
        default=3,
        help="Number of joints (default: 3)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_passthrough_onnx(args.output, args.num_joints)


if __name__ == "__main__":
    main()

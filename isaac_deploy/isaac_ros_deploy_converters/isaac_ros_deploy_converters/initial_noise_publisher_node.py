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

"""ROS2 node that publishes random noise tensors for diffusion-based policies.

Some policies (e.g. GR00T) use a diffusion action head that requires
random initial noise as an input. This node finds dangling inputs named
``initial_noise`` (i.e. listed under ``pipeline.inputs.<model>`` in the
exported config — meaning they are not fed by any other model and must be
supplied externally), extracts their shape/dtype, and publishes seeded
random tensors at a fixed rate.

Usage:
    ros2 run isaac_ros_deploy_converters initial_noise_publisher_node \
        --ros-args -p config_path:=/path/to/gr00t_config.yaml \
                   -p publish_rate:=50.0 \
                   -p seed:=42
"""

from pathlib import Path

from isaac_ros_deploy_converters.tensor_utils import make_tensor
from isaac_ros_tensor_list_interfaces.msg import TensorList
import numpy as np
import rclpy
from rclpy.node import Node
import yaml


_INITIAL_NOISE_INPUT_NAME = 'initial_noise'


def _find_initial_noise_inputs(config: dict) -> list[dict]:
    """Find dangling model inputs named ``initial_noise``.

    Dangling inputs are those listed under ``pipeline.inputs.<model>``
    in the config — they are not produced by any upstream model via
    ``data_flow`` and therefore must be published by an external source.
    """
    pipeline_inputs = (config.get('pipeline') or {}).get('inputs') or {}
    models = config.get('models') or {}
    results = []
    for model_name, dangling_names in pipeline_inputs.items():
        if _INITIAL_NOISE_INPUT_NAME not in (dangling_names or []):
            continue
        for inp in (models.get(model_name) or {}).get('inputs', []):
            if inp.get('name') == _INITIAL_NOISE_INPUT_NAME:
                results.append(inp)
                break
    return results


class InitialNoisePublisherNode(Node):
    """Publishes seeded random noise tensors for diffusion-based policies."""

    def __init__(self):
        super().__init__('initial_noise_publisher_node')

        self.declare_parameter('config_path', '')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('seed', 42)
        self.declare_parameter('output_topic', 'initial_noise')

        config_path = Path(self.get_parameter('config_path').value)
        if not config_path.name:
            raise RuntimeError("'config_path' parameter is required but was not set")
        publish_rate = self.get_parameter('publish_rate').value
        seed = self.get_parameter('seed').value
        output_topic = self.get_parameter('output_topic').value

        if publish_rate <= 0:
            raise RuntimeError(
                f"'publish_rate' must be positive, got {publish_rate}"
            )

        config = yaml.safe_load(config_path.read_text())
        self.noise_inputs = _find_initial_noise_inputs(config)
        if not self.noise_inputs:
            raise RuntimeError(
                f"No dangling input named '{_INITIAL_NOISE_INPUT_NAME}' "
                f'found under pipeline.inputs in {config_path}'
            )

        self.rng = np.random.default_rng(seed=seed)

        for inp in self.noise_inputs:
            self.get_logger().info(
                f'Publishing noise: name={inp["name"]}, '
                f'shape={inp["shape"]}, dtype={inp.get("dtype", "float32")}'
            )

        self.publisher = self.create_publisher(TensorList, output_topic, 10)
        self.timer = self.create_timer(1.0 / publish_rate, self._publish_callback)

    def _publish_callback(self):
        """Publish one TensorList with random noise tensors."""
        tensors = []
        for inp in self.noise_inputs:
            shape = tuple(inp['shape'])
            dtype = np.dtype(inp.get('dtype', 'float32'))
            noise = self.rng.standard_normal(shape).astype(dtype)
            tensors.append(make_tensor(inp['name'], noise))

        msg = TensorList()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.tensors = tensors
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = InitialNoisePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

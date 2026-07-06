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

"""Shared utilities for building TensorList messages from numpy arrays."""

from isaac_ros_tensor_list_interfaces.msg import Tensor, TensorShape
import numpy as np

# numpy dtype string -> TensorList data_type enum value.
# Values from isaac_ros_tensor_list_interfaces/msg/Tensor.msg.
DTYPE_TO_ENUM = {
    'float32': 9,
    'float64': 10,
    'int32': 5,
    'int64': 7,
}


def make_tensor(name: str, data: np.ndarray) -> Tensor:
    """Create a Tensor message from a numpy array."""
    tensor = Tensor()
    tensor.name = name
    tensor.shape = TensorShape()
    tensor.shape.rank = len(data.shape)
    tensor.shape.dims = list(data.shape)
    dtype_str = str(data.dtype)
    if dtype_str not in DTYPE_TO_ENUM:
        raise ValueError(
            f"Unsupported numpy dtype '{dtype_str}' for TensorList message. "
            f"Supported dtypes: {list(DTYPE_TO_ENUM.keys())}"
        )
    tensor.data_type = DTYPE_TO_ENUM[dtype_str]
    data = np.ascontiguousarray(data)
    tensor.strides = list(data.strides)
    tensor.data = data.tobytes()
    return tensor

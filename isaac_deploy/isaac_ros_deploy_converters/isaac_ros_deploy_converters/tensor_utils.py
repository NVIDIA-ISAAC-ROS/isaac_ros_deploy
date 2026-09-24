# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared utilities for building CUDA-buffer-compatible tensor messages."""

import numpy as np
from tensor_msgs.msg import ExperimentalTensor

# numpy dtype string -> DLPack (code, bits). Lanes are scalar (1).
DTYPE_TO_DLPACK = {
    'float32': (2, 32),
    'float64': (2, 64),
    'int8': (0, 8),
    'int16': (0, 16),
    'int32': (0, 32),
    'int64': (0, 64),
    'uint8': (1, 8),
}


def make_tensor(data: np.ndarray) -> ExperimentalTensor:
    """Create a contiguous ExperimentalTensor message from a numpy array."""
    tensor = ExperimentalTensor()
    data = np.ascontiguousarray(data)
    tensor.shape = list(data.shape)
    dtype_str = str(data.dtype)
    if dtype_str not in DTYPE_TO_DLPACK:
        raise ValueError(
            f"Unsupported numpy dtype '{dtype_str}' for TensorList message. "
            f'Supported dtypes: {list(DTYPE_TO_DLPACK.keys())}'
        )
    tensor.dtype_code, tensor.dtype_bits = DTYPE_TO_DLPACK[dtype_str]
    tensor.dtype_lanes = 1
    # DLPack permits an empty stride vector for contiguous row-major tensors.
    # This also avoids NumPy's noncanonical zero strides on size-one axes.
    tensor.strides = []
    tensor.byte_offset = 0
    tensor.data = data.tobytes()
    return tensor

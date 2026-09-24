# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaac_ros_deploy_converters.tensor_utils import make_tensor
import numpy as np


def test_make_tensor_uses_implicit_contiguous_strides():
    # Adding a size-one axis can leave NumPy with a zero stride even though the
    # resulting array is C-contiguous. DLPack's implicit representation avoids
    # publishing that noncanonical explicit stride.
    data = np.zeros((16, 3), dtype=np.float32)[np.newaxis, :, :]

    tensor = make_tensor(data)

    assert tensor.shape == [1, 16, 3]
    assert tensor.strides == []
    assert bytes(tensor.data) == data.tobytes()

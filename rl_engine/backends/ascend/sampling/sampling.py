# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from rl_engine.backends.cpu.sampling import PyTorchSamplingOp
from rl_engine.platforms.constants import BackendLib


class AscendSamplingOp(PyTorchSamplingOp):
    """Sample from the shared distribution using PyTorch on Ascend tensors."""

    backend_id = "rlkernel.ascend.sampling.v1"
    backend_lib = BackendLib.CANN.value

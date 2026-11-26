# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from rl_engine.platforms.constants import BackendLib


class PyTorchSamplingOp:
    """Categorical draw on devices supporting PyTorch multinomial."""

    backend_id = "rlkernel.reference.native_sampling.v1"
    backend_lib = BackendLib.NATIVE.value

    def sample(self, probs, *, generator=None, deterministic=True):
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

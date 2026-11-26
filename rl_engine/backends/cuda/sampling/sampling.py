# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from rl_engine.platforms.constants import BackendLib


class FlashInferSamplingOp:
    """Draw tokens with FlashInfer from the shared policy probabilities."""

    backend_id = "rlkernel.cuda.flashinfer_sampling.v1"
    backend_lib = BackendLib.FLASHINFER.value
    legacy_probability_definition_id = "rlkernel.sampling.flashinfer_legacy.v1"

    def __init__(self):
        from flashinfer.sampling import (
            sampling_from_probs,
            top_k_renorm_probs,
            top_p_sampling_from_probs,
        )

        self._sample = sampling_from_probs
        self._top_k = top_k_renorm_probs
        self._top_p = top_p_sampling_from_probs

    def sample(self, probs, *, generator=None, deterministic=True):
        return self._sample(probs, generator=generator, deterministic=deterministic).long()

    def sample_legacy(
        self, logits, temperature, top_k, top_p, *, generator=None, deterministic=True
    ):
        """Use the vendor's fused filtering for the original CUDA sampling call."""

        scaled = logits if temperature == 1.0 else logits / temperature
        probs = scaled.float().contiguous().softmax(dim=-1)
        rows, vocab = logits.shape
        if isinstance(top_k, torch.Tensor):
            top_k = top_k.to(logits.device).reshape(-1)
            top_k = torch.where(top_k > 0, top_k, vocab).clamp(max=vocab)
            top_k = top_k.to(torch.int32).expand(rows).contiguous()
        elif top_k == -1:
            top_k = None
        elif top_k is not None:
            top_k = min(top_k, vocab)
        if top_k is not None:
            probs = self._top_k(probs, top_k)
        if isinstance(top_p, torch.Tensor):
            top_p = top_p.to(device=logits.device, dtype=torch.float32)
            top_p = top_p.reshape(-1).expand(rows).contiguous()
        if top_p is not None:
            return self._top_p(probs, top_p, generator=generator, deterministic=deterministic)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Sampling dispatcher and multi-backend routing (#563 Task 49)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from rl_engine.ops.sampling.policy import (
    GREEDY_TEMPERATURE,
    SAMPLING_PROBABILITY_DEFINITION_ID,
    SamplingPolicyConfig,
    transformed_sampling_logprobs,
    transformed_sampling_probs,
)
from rl_engine.runtime.semantic_registry import _JsonRecord


@dataclass(frozen=True)
class SamplingDispatchSelection(_JsonRecord):
    """Traceable registration and selection record for a platform sampling backend."""

    platform: str
    backend_id: str
    backend_lib: str
    probability_definition_id: str = SAMPLING_PROBABILITY_DEFINITION_ID


def _sampling_backend(device: str | torch.device | None):
    from rl_engine.runtime.registry import kernel_registry

    if isinstance(device, str) and device.split(":", 1)[0] == "ascend":
        device = device.replace("ascend", "npu", 1)
    op = kernel_registry.get_op("sampling", device)
    selection = SamplingDispatchSelection(
        kernel_registry._platform_for_device(device), op.backend_id, op.backend_lib
    )
    return op, selection


def resolve_sampling_dispatch(
    device_or_platform: str | torch.device | None = None,
) -> SamplingDispatchSelection:
    return _sampling_backend(device_or_platform)[1]


class SamplerBackend(nn.Module):
    def __init__(self, device: str | torch.device | None = None):
        super().__init__()
        self.dispatch_selection = resolve_sampling_dispatch(device)
        self._platform = None if device is None else self.dispatch_selection.platform

    @property
    def backend(self) -> str:
        return self.dispatch_selection.backend_lib

    @torch.inference_mode()
    def sample(
        self,
        logits,
        top_k=None,
        top_p=None,
        temperature=1.0,
        deterministic=True,
        *,
        policy: SamplingPolicyConfig | None = None,
        generator: torch.Generator | None = None,
        rng_seed: int | None = None,
    ):
        op, selection = _sampling_backend(logits.device)
        if self._platform is not None and self._platform != selection.platform:
            raise ValueError(
                f"configured sampling platform {self._platform!r} does not match logits"
            )
        seed = policy.rng_seed if rng_seed is None and policy is not None else rng_seed
        if generator is None and seed is not None:
            generator = torch.Generator(device=logits.device).manual_seed(seed)

        legacy = getattr(op, "sample_legacy", None)
        if (
            policy is None
            and legacy is not None
            and isinstance(temperature, (int, float))
            and temperature >= GREEDY_TEMPERATURE
        ):
            self.dispatch_selection = replace(
                selection, probability_definition_id=op.legacy_probability_definition_id
            )
            return legacy(
                logits,
                temperature,
                top_k,
                top_p,
                generator=generator,
                deterministic=deterministic,
            )
        self.dispatch_selection = selection
        policy = policy or SamplingPolicyConfig(temperature=temperature, top_k=top_k, top_p=top_p)
        probs, _ = transformed_sampling_probs(logits, policy)
        return op.sample(probs, generator=generator, deterministic=deterministic)

    @torch.inference_mode()
    def compute_logp(self, logits, token_ids, *, policy: SamplingPolicyConfig | None = None):
        if policy is not None:
            return transformed_sampling_logprobs(logits, token_ids, policy)
        batch_size = logits.shape[0]
        out_logp = logits.new_empty(logits.shape[:-1])
        for i in range(0, batch_size, 4096):
            c_logits = logits[i : i + 4096]
            c_token_ids = token_ids[i : i + 4096]
            out_logp[i : i + 4096] = (
                torch.log_softmax(c_logits, dim=-1)
                .gather(-1, c_token_ids.unsqueeze(-1))
                .squeeze(-1)
            )
        return out_logp


__all__ = [
    "SamplerBackend",
    "SamplingDispatchSelection",
    "resolve_sampling_dispatch",
]

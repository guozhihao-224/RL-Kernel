# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import sys
import types

import pytest
import torch

from rl_engine.ops.sampling.api import SamplerBackend
from rl_engine.ops.sampling.policy import (
    SAMPLING_PROBABILITY_DEFINITION_ID,
    SamplingPolicyConfig,
    sampling_keep_mask,
)
from rl_engine.runtime import registry


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_flashinfer_dispatch_preserves_legacy_and_explicit_policy(monkeypatch, temperature):
    captured = {}
    sampling = types.ModuleType("flashinfer.sampling")

    def sample(probs, *, generator, deterministic):
        captured.update(probs=probs.clone(), generator=generator)
        return torch.multinomial(probs, 1, generator=generator).squeeze(1)

    def top_p_sample(probs, top_p, **kwargs):
        captured["top_p"] = top_p
        return sample(probs, **kwargs)

    sampling.sampling_from_probs = sample
    sampling.top_p_sampling_from_probs = top_p_sample
    sampling.top_k_renorm_probs = lambda probs, top_k: probs
    monkeypatch.setitem(sys.modules, "flashinfer.sampling", sampling)
    monkeypatch.setattr(
        registry.KernelRegistry, "_platform_for_device", lambda self, device: "cuda"
    )
    monkeypatch.setattr(registry, "kernel_registry", registry.KernelRegistry())

    logits = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 4
    generator = torch.Generator().manual_seed(7)
    sampler = SamplerBackend(device="cuda")
    top_p = torch.tensor([0.9, 0.8])

    def no_scalar_read(self):
        pytest.fail("legacy filtering must keep per-row parameters on the device")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", no_scalar_read)
        sampler.sample(logits, top_p=top_p, temperature=temperature, generator=generator)
    torch.testing.assert_close(captured.pop("top_p"), top_p)
    torch.testing.assert_close(captured["probs"], (logits / temperature).softmax(-1))
    assert sampler.dispatch_selection.probability_definition_id.endswith("flashinfer_legacy.v1")

    policy = SamplingPolicyConfig(top_p=0.9, temperature=temperature)
    sampler.sample(logits, policy=policy, generator=generator)
    keep = sampling_keep_mask(logits, temperature=temperature, top_p=0.9)
    torch.testing.assert_close(
        captured["probs"], (logits / temperature).masked_fill(~keep, -torch.inf).softmax(-1)
    )
    assert captured["generator"] is generator
    assert "top_p" not in captured
    assert (
        sampler.dispatch_selection.probability_definition_id == SAMPLING_PROBABILITY_DEFINITION_ID
    )

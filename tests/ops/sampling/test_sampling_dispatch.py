# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.ops.sampling.api import SamplerBackend, resolve_sampling_dispatch
from rl_engine.ops.sampling.policy import (
    SamplingPolicyConfig,
    transformed_sampling_logprobs,
    transformed_sampling_probs,
)


def test_sample_and_recompute_share_token_space_and_probabilities():
    sampler = SamplerBackend()
    logits = torch.tensor([[1.0, 3.0, 100.0], [2.0, 0.0, 100.0]])
    policy = SamplingPolicyConfig(
        temperature=torch.tensor([0.0, 2.0]), real_vocab_size=2, rng_seed=42
    )

    expected = torch.tensor([[0.0, 1.0, 0.0], [0.7310586, 0.2689414, 0.0]])
    probs, _ = transformed_sampling_probs(logits, policy)
    torch.testing.assert_close(probs, expected)
    tokens = sampler.sample(logits, policy=policy)
    assert sampler.backend == "native"
    generator = torch.Generator().manual_seed(policy.rng_seed)
    assert torch.equal(tokens, torch.multinomial(expected, 1, generator=generator).squeeze(1))
    logp = sampler.compute_logp(logits, tokens, policy=policy)
    torch.testing.assert_close(logp, expected.gather(1, tokens[:, None]).squeeze(1).log())

    logits.requires_grad_()
    transformed_sampling_logprobs(logits, tokens, policy).sum().backward()
    assert torch.equal(logits.grad[:, 2], torch.zeros(2))


def test_explicit_ascend_route_requires_ascend_logits():
    assert resolve_sampling_dispatch("npu").backend_lib == "cann"
    with pytest.raises(ValueError, match="does not match logits"):
        SamplerBackend("npu").sample(torch.ones((1, 2)))


@pytest.mark.parametrize(("field", "value"), [("temperature", -0.1), ("top_p", 0.0), ("top_k", 0)])
def test_policy_rejects_invalid_scalar_and_row_values(field, value):
    for param in (value, torch.tensor([value])):
        with pytest.raises(ValueError, match=field):
            SamplingPolicyConfig(**{field: param})


def test_equivalent_greedy_policies_advance_generator_identically():
    sampler = SamplerBackend("cpu")
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    first = torch.Generator().manual_seed(19)
    second = torch.Generator().manual_seed(19)
    scalar = sampler.sample(logits, policy=SamplingPolicyConfig(temperature=0), generator=first)
    tensor = sampler.sample(
        logits, policy=SamplingPolicyConfig(temperature=torch.tensor([0.0])), generator=second
    )
    assert torch.equal(scalar, tensor)
    assert torch.equal(first.get_state(), second.get_state())
    assert sampler.compute_logp(logits, scalar).shape == scalar.shape


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA"
)
def test_cuda_filtered_policy_accepts_strided_logits():
    logits = torch.arange(64, device="cuda", dtype=torch.float32).reshape(16, 4).T
    policy = SamplingPolicyConfig(top_k=2, top_p=0.9, rng_seed=8)
    sampler = SamplerBackend()
    assert torch.equal(
        sampler.sample(logits, policy=policy), sampler.sample(logits.contiguous(), policy=policy)
    )

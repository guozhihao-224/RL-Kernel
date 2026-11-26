# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Complete top-k/top-p support, independent of API top-logprob limits.

The support follows vLLM's native ascending-sort contract on the real
vocabulary. Only the boolean selection is detached; scoring retains autograd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

SAMPLING_PROBABILITY_DEFINITION_ID = "rlkernel.sampling.transformed_support_fp32.v1"
GREEDY_TEMPERATURE = 1e-5


@dataclass(frozen=True)
class SamplingPolicyConfig:
    """Sampling transform and RNG contract shared by rollout and training recomputation."""

    temperature: float | torch.Tensor = 1.0
    top_k: int | torch.Tensor | None = None
    top_p: float | torch.Tensor | None = None
    real_vocab_size: int | None = None
    rng_seed: int | None = None

    def __post_init__(self) -> None:
        temp = self.temperature
        if isinstance(temp, torch.Tensor):
            valid_temp = (temp.isfinite() & (temp >= 0)).all().item()
        else:
            valid_temp = math.isfinite(temp) and temp >= 0
        if not valid_temp:
            raise ValueError("temperature must be finite and >= 0.0")
        p = self.top_p
        if p is not None and not _all((p > 0) & (p <= 1)):
            raise ValueError("top_p must be in (0.0, 1.0]")
        k = self.top_k
        if isinstance(k, torch.Tensor) and k.dtype not in (torch.int32, torch.int64):
            raise ValueError("top_k tensor must use int32 or int64")
        if k is not None and not _all((k == -1) | (k > 0)):
            raise ValueError("top_k must be -1, None, or a positive integer")
        if self.real_vocab_size is not None and self.real_vocab_size <= 0:
            raise ValueError("real_vocab_size must be a positive integer or None")


def _all(condition: bool | torch.Tensor) -> bool:
    return bool(condition.all().item()) if isinstance(condition, torch.Tensor) else condition


def _scale_sampling_logits(logits: torch.Tensor, temperature) -> tuple[torch.Tensor, torch.Tensor]:
    temp = torch.as_tensor(temperature, dtype=torch.float32, device=logits.device).reshape(-1)
    greedy = (temp < GREEDY_TEMPERATURE).expand(logits.size(0))
    values = logits.float()
    if isinstance(temperature, (int, float)) and (
        temperature < GREEDY_TEMPERATURE or temperature == 1
    ):
        return values, greedy
    return values / torch.where(greedy, 1.0, temp).unsqueeze(1), greedy


def _masked_policy_logits(
    logits: torch.Tensor, policy: SamplingPolicyConfig
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if logits.ndim != 2:
        raise ValueError("sampling logits must be [rows, vocab]")
    real_vocab = logits.size(1) if policy.real_vocab_size is None else policy.real_vocab_size
    if not (0 < real_vocab <= logits.size(1)):
        raise ValueError(f"real_vocab_size ({real_vocab}) out of range for width {logits.size(1)}")
    active = logits if real_vocab == logits.size(1) else logits[:, :real_vocab]
    values, greedy = _scale_sampling_logits(active, policy.temperature)
    scalar_greedy = isinstance(policy.temperature, (int, float)) and (
        policy.temperature < GREEDY_TEMPERATURE
    )
    if scalar_greedy:
        keep = torch.isfinite(values)
    else:
        keep = _sampling_keep_mask(values, greedy, top_p=policy.top_p, top_k=policy.top_k)
    masked = values.masked_fill(~keep, float("-inf"))
    if scalar_greedy or isinstance(policy.temperature, torch.Tensor):
        winner = masked.argmax(dim=-1, keepdim=True)
        winners = torch.arange(real_vocab, device=logits.device) == winner
        masked = masked.masked_fill(greedy[:, None] & ~winners, float("-inf"))
    return masked, greedy, real_vocab


def transformed_sampling_probs(
    logits: torch.Tensor, policy: SamplingPolicyConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute policy-transformed FP32 probabilities and per-row greedy flags."""

    masked, greedy, real_vocab = _masked_policy_logits(logits, policy)
    probs = torch.softmax(masked, dim=-1)
    if real_vocab == int(logits.size(1)):
        return probs, greedy
    out = torch.zeros_like(logits, dtype=torch.float32)
    out[:, :real_vocab] = probs
    return out, greedy


def transformed_sampling_logprobs(
    logits: torch.Tensor, target_ids: torch.Tensor, policy: SamplingPolicyConfig
) -> torch.Tensor:
    """Recompute target log-probabilities under the exact policy transformation."""

    masked, _, _ = _masked_policy_logits(logits, policy)
    # Rollout tokens may have been allocated under inference_mode.
    ids = target_ids.reshape(-1).to(
        device=logits.device,
        dtype=torch.long,
        copy=masked.requires_grad and target_ids.is_inference(),
    )
    if ids.numel() != logits.size(0):
        raise ValueError("target_ids must have one token per logit row")
    return torch.log_softmax(masked, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def sampling_keep_mask(logits, *, temperature=1.0, top_p=None, top_k=None):
    """Return the full support for scalar or per-row sampling parameters."""
    if logits.ndim != 2:
        raise ValueError("sampling logits must be [rows, real_vocab]")
    values, greedy = _scale_sampling_logits(logits, temperature)
    return _sampling_keep_mask(values, greedy, top_p=top_p, top_k=top_k)


@torch.no_grad()
def _sampling_keep_mask(values, greedy, *, top_p, top_k):
    if top_p is None and top_k is None:
        return torch.isfinite(values)
    cuda_scatter = values.is_cuda and torch.version.hip is None
    if cuda_scatter and values.stride(1) != 1:
        values = values.clone(memory_format=torch.contiguous_format)
    sorted_values, ids = values.sort(dim=-1, descending=False)
    if top_k is not None:
        k = torch.as_tensor(top_k, dtype=torch.long, device=values.device).reshape(-1)
        k = torch.where(k <= 0, values.size(1), k).clamp(max=values.size(1))
        k = k.expand(values.size(0))
        threshold = sorted_values.gather(1, (values.size(1) - k).unsqueeze(1))
        sorted_values.masked_fill_(sorted_values < threshold, float("-inf"))
    if top_p is not None:
        p = torch.as_tensor(top_p, dtype=torch.float32, device=values.device)
        probabilities = sorted_values.softmax(dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        removed = cumulative <= 1 - p.reshape(-1, 1)
        removed[:, -1] = False
        sorted_values.masked_fill_(removed, float("-inf"))
    if cuda_scatter:
        from rl_engine.backends.cuda.sampling.unique_scatter import scatter_permuted_columns

        mask = scatter_permuted_columns(torch.isfinite(sorted_values), ids)
    else:
        mask = torch.zeros_like(values, dtype=torch.bool).scatter_(
            1, ids, torch.isfinite(sorted_values)
        )
    return torch.where(greedy.reshape(-1, 1), torch.isfinite(values), mask)


@torch.no_grad()
def vocab_parallel_sampling_keep_mask(
    local_logits,
    *,
    real_vocab_size,
    tp_group=None,
    temperature=1.0,
    top_p=None,
    top_k=None,
    active_rows=None,
    chunk_size=None,
):
    """Compute global support in bounded chunks, then retain this TP shard.

    TP ranks have identical token ownership. CP groups deliberately do not
    participate: their token rows are independent. Inactive packed/prompt
    rows stay unmasked so discarded targets do not introduce infinities.
    """
    if chunk_size is None:
        # Each token row is independent. Amortize TP transport and sorting
        # dispatches on CUDA while bounding the larger vocabulary workspace.
        # Keep the existing ROCm/CPU workspace and explicit caller overrides.
        chunk_size = (
            max(32, min(128, (512 * 1024 * 1024) // (28 * real_vocab_size)))
            if local_logits.is_cuda and torch.version.hip is None and real_vocab_size > 0
            else 32
        )
    if chunk_size <= 0:
        raise ValueError("sampling mask chunk size must be positive")
    initialized = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size(tp_group) if initialized else 1
    rank = dist.get_rank(tp_group) if initialized else 0
    width = local_logits.size(1)
    if not 0 < real_vocab_size <= width * world:
        raise ValueError("real vocabulary must fit the TP shards")
    keep = torch.ones_like(local_logits, dtype=torch.bool)
    indices = (
        torch.arange(local_logits.size(0), device=local_logits.device)
        if active_rows is None
        else active_rows.nonzero().reshape(-1)
    )
    for chunk in indices.split(chunk_size):
        local = local_logits.detach().index_select(0, chunk).contiguous()
        if world > 1:
            shards = [torch.empty_like(local) for _ in range(world)]
            dist.all_gather(shards, local, group=tp_group)
            complete = torch.cat(shards, dim=1)[:, :real_vocab_size]
        else:
            complete = local[:, :real_vocab_size]
        mask = sampling_keep_mask(complete, temperature=temperature, top_p=top_p, top_k=top_k)
        start = rank * width
        count = max(0, min(width, real_vocab_size - start))
        if local.is_cuda and torch.version.hip is None and count == width:
            selected = mask[:, start : start + count]
        else:
            selected = torch.zeros_like(local, dtype=torch.bool)
            selected[:, :count] = mask[:, start : start + count]
        if keep.is_cuda and torch.version.hip is None and keep.stride(1) == 1:
            from rl_engine.backends.cuda.sampling.unique_scatter import copy_unique_rows_

            copy_unique_rows_(keep, chunk, selected)
        else:
            keep.index_copy_(0, chunk, selected)
    return keep

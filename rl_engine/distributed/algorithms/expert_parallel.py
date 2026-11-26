# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic Expert Parallel (EP) plans, dispatch, and return (#563 Task 24 / #642 Task 39)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rl_engine.reference.moe.combine import (
    combine_returned_slots_fixed_order,
    fixed_slot_tree_reduce_fp32,
)


@dataclass(frozen=True)
class ExpertParallelPartition:
    """Logical contiguous ownership of experts across EP ranks."""

    num_experts: int
    ep_world_size: int
    ep_rank: int

    def __post_init__(self) -> None:
        if self.num_experts <= 0 or self.ep_world_size <= 0:
            raise ValueError("num_experts and ep_world_size must be positive")
        if self.num_experts % self.ep_world_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"ep_world_size ({self.ep_world_size})"
            )
        if not (0 <= self.ep_rank < self.ep_world_size):
            raise ValueError(f"ep_rank ({self.ep_rank}) out of range [0, {self.ep_world_size})")

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.ep_world_size

    @property
    def expert_start(self) -> int:
        return self.ep_rank * self.experts_per_rank

    @property
    def expert_end(self) -> int:
        return self.expert_start + self.experts_per_rank

    def owner_rank(self, expert_id: int) -> int:
        if not (0 <= expert_id < self.num_experts):
            raise ValueError(f"expert_id ({expert_id}) out of range [0, {self.num_experts})")
        return expert_id // self.experts_per_rank


@dataclass(frozen=True)
class LocalExpertDispatchPlan:
    """Deterministic local token/slot to expert permutation and send counts."""

    partition: ExpertParallelPartition
    num_tokens: int
    top_k: int
    flat_token_indices: torch.Tensor
    flat_slot_indices: torch.Tensor
    flat_expert_ids: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    send_counts: torch.Tensor
    send_offsets: torch.Tensor


@dataclass(frozen=True)
class ExpertReceivePlan:
    """Source-rank splits for one rank's expert dispatch receive buffer."""

    recv_counts: torch.Tensor
    recv_offsets: torch.Tensor


def _offsets(counts: torch.Tensor) -> torch.Tensor:
    return torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))


def build_receive_plan(
    recv_counts: torch.Tensor, partition: ExpertParallelPartition
) -> ExpertReceivePlan:
    """Use one exchanged dispatch count per source rank, in rank order."""

    if recv_counts.shape != (partition.ep_world_size,):
        raise ValueError("recv_counts must contain one count per EP rank")
    counts = recv_counts.clone()
    return ExpertReceivePlan(counts, _offsets(counts))


def build_local_dispatch_plan(
    selected_experts: torch.Tensor,
    partition: ExpertParallelPartition,
) -> LocalExpertDispatchPlan:
    """Build a stable token/slot dispatch mapping sorted by global expert ID."""

    if selected_experts.ndim != 2 or selected_experts.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected_experts must be an int32/int64 tensor of shape [tokens, top_k]")
    num_tokens, top_k = int(selected_experts.size(0)), int(selected_experts.size(1))
    if not (1 <= top_k <= partition.num_experts):
        raise ValueError(f"top_k ({top_k}) must be in [1, {partition.num_experts}]")

    device = selected_experts.device
    experts_long = selected_experts.to(dtype=torch.long, copy=True)
    if num_tokens > 0:
        if ((experts_long < 0) | (experts_long >= partition.num_experts)).any():
            raise ValueError("selected_experts contains out-of-bounds expert ID")
        if top_k > 1:
            sorted_row, _ = torch.sort(experts_long, dim=-1)
            if bool((sorted_row[:, 1:] == sorted_row[:, :-1]).any()):
                raise ValueError("duplicate expert IDs within the same token's top_k slots")

    flat_experts = experts_long.reshape(-1)
    coordinates = torch.arange(flat_experts.numel(), dtype=torch.long, device=device)
    permutation = torch.argsort(flat_experts, stable=True)
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = coordinates

    expert_counts = torch.bincount(flat_experts, minlength=partition.num_experts)
    send_counts = expert_counts.reshape(partition.ep_world_size, partition.experts_per_rank).sum(1)

    return LocalExpertDispatchPlan(
        partition=partition,
        num_tokens=num_tokens,
        top_k=top_k,
        flat_token_indices=coordinates // top_k,
        flat_slot_indices=coordinates % top_k,
        flat_expert_ids=flat_experts,
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        expert_counts=expert_counts,
        expert_offsets=_offsets(expert_counts),
        send_counts=send_counts,
        send_offsets=_offsets(send_counts),
    )


def _restore_slots(
    rows: torch.Tensor, inverse_permutation: torch.Tensor, num_tokens: int, top_k: int
) -> torch.Tensor:
    return rows.index_select(0, inverse_permutation).reshape(num_tokens, top_k, rows.size(1))


class _PackDispatchTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, plan):
        ctx.save_for_backward(plan.inverse_permutation)
        ctx.slot_shape = (plan.num_tokens, plan.top_k)
        token_indices = plan.flat_token_indices.index_select(0, plan.permutation)
        return hidden_states.index_select(0, token_indices)

    @staticmethod
    def backward(ctx, grad_rows):
        (inverse_permutation,) = ctx.saved_tensors
        slots = _restore_slots(grad_rows, inverse_permutation, *ctx.slot_shape)
        return fixed_slot_tree_reduce_fp32(slots), None


def pack_dispatch_tokens(
    hidden_states: torch.Tensor, plan: LocalExpertDispatchPlan
) -> torch.Tensor:
    """Dispatch expert-major rows and restore input gradients in fixed slot order."""

    if hidden_states.ndim != 2 or int(hidden_states.size(0)) != plan.num_tokens:
        raise ValueError(f"hidden_states must have shape [{plan.num_tokens}, hidden_dim]")
    return _PackDispatchTokens.apply(hidden_states, plan)


def unpack_returned_expert_outputs(
    permuted_expert_outputs: torch.Tensor, plan: LocalExpertDispatchPlan
) -> torch.Tensor:
    """Restore expert outputs from dispatch order to ``[num_tokens, top_k, hidden_dim]``."""

    total = plan.num_tokens * plan.top_k
    if permuted_expert_outputs.ndim != 2 or int(permuted_expert_outputs.size(0)) != total:
        raise ValueError(f"permuted_expert_outputs must have shape [{total}, hidden_dim]")
    return _restore_slots(
        permuted_expert_outputs, plan.inverse_permutation, plan.num_tokens, plan.top_k
    )


def combine_backward_input_gradients_fixed_order(
    slot_input_grads: torch.Tensor,
    plan: LocalExpertDispatchPlan,
    *,
    shared_input_grad: torch.Tensor | None = None,
    residual_grad: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Unpack (if 2D) and combine expert ``dX`` in fixed binary-tree FP32 order."""

    unpacked = (
        unpack_returned_expert_outputs(slot_input_grads, plan)
        if slot_input_grads.ndim == 2
        else slot_input_grads
    )
    if unpacked.shape[:2] != (plan.num_tokens, plan.top_k):
        raise ValueError("input gradients must match the dispatch plan's token and slot counts")
    return fixed_slot_tree_reduce_fp32(
        unpacked,
        shared=shared_input_grad,
        residual=residual_grad,
        output_dtype=slot_input_grads.dtype if output_dtype is None else output_dtype,
    )


@dataclass(frozen=True)
class ExpertReturnPacket:
    """One returned chunk or message carrying explicit ``(token, slot)`` coordinates."""

    token_indices: torch.Tensor
    slot_indices: torch.Tensor
    contributions: torch.Tensor


def reconstruct_slots_arrival_invariant(
    packets: Sequence[ExpertReturnPacket],
    *,
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reconstruct ``[num_tokens, top_k, hidden_dim]`` independent of packet arrival order."""

    total = num_tokens * top_k
    if not packets:
        if total:
            raise RuntimeError("missing (token, slot) contribution in EP return")
        return torch.empty((num_tokens, top_k, hidden_dim), device=device, dtype=dtype)

    if device is None:
        device = packets[0].contributions.device

    for packet in packets:
        count = packet.token_indices.numel()
        if (
            packet.token_indices.shape != (count,)
            or packet.slot_indices.shape != (count,)
            or packet.contributions.shape != (count, hidden_dim)
        ):
            raise ValueError("packet coordinates and contributions must have matching row counts")

    tokens = torch.cat([packet.token_indices.to(device) for packet in packets])
    slots = torch.cat([packet.slot_indices.to(device) for packet in packets])
    if tokens.dtype not in (torch.int32, torch.int64) or slots.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("packet coordinates must be integer indices")
    if ((tokens < 0) | (tokens >= num_tokens) | (slots < 0) | (slots >= top_k)).any():
        raise ValueError("packet token or slot index is out of bounds")

    coordinates, order = (tokens.long() * top_k + slots).sort()
    if (coordinates[1:] == coordinates[:-1]).any():
        raise RuntimeError("duplicate (token, slot) contribution in EP return")
    if coordinates.numel() != total:
        raise RuntimeError("missing (token, slot) contribution in EP return")

    rows = torch.cat([packet.contributions.to(device=device, dtype=dtype) for packet in packets])
    return rows.index_select(0, order).reshape(num_tokens, top_k, hidden_dim)


__all__ = [
    "ExpertParallelPartition",
    "ExpertReceivePlan",
    "ExpertReturnPacket",
    "LocalExpertDispatchPlan",
    "build_local_dispatch_plan",
    "build_receive_plan",
    "combine_backward_input_gradients_fixed_order",
    "combine_returned_slots_fixed_order",
    "fixed_slot_tree_reduce_fp32",
    "pack_dispatch_tokens",
    "reconstruct_slots_arrival_invariant",
    "unpack_returned_expert_outputs",
]

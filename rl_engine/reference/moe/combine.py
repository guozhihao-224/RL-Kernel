# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fixed-order expert slot arithmetic shared by local and distributed paths."""

import torch


def fixed_slot_tree_reduce_fp32(
    slot_tensors: torch.Tensor,
    *,
    shared: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Adjacent-pair binary tree reduction over slot dim 1 in FP32 with optional branches."""

    if slot_tensors.ndim != 3 or int(slot_tensors.size(1)) < 1:
        raise ValueError("slot_tensors must have shape [num_tokens, top_k >= 1, hidden_dim]")
    target_dtype = slot_tensors.dtype if output_dtype is None else output_dtype
    level = [slot_tensors[:, idx, :].float() for idx in range(int(slot_tensors.size(1)))]
    while len(level) > 1:
        paired = [level[i] + level[i + 1] for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            paired.append(level[-1])
        level = paired
    acc = level[0]
    if shared is not None:
        acc = acc + shared.float()
    if residual is not None:
        acc = acc + residual.float()
    return acc.to(dtype=target_dtype)


def combine_returned_slots_fixed_order(
    slot_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    shared_expert_output: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Weight each slot in FP32 and reduce in fixed slot-tree order."""

    if routing_weights.shape != slot_outputs.shape[:2]:
        raise ValueError("routing_weights shape must match slot_outputs[:2]")
    weighted = slot_outputs.float() * routing_weights.float().unsqueeze(-1)
    return fixed_slot_tree_reduce_fp32(
        weighted,
        shared=shared_expert_output,
        residual=residual,
        output_dtype=slot_outputs.dtype if output_dtype is None else output_dtype,
    )

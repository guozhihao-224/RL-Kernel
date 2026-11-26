# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for deterministic Expert Parallel (EP) plans, dispatch, and return (#563 Task 24)."""

from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.distributed.algorithms.expert_parallel import (
    ExpertParallelPartition,
    ExpertReturnPacket,
    build_local_dispatch_plan,
    build_receive_plan,
    combine_backward_input_gradients_fixed_order,
    combine_returned_slots_fixed_order,
    pack_dispatch_tokens,
    reconstruct_slots_arrival_invariant,
    unpack_returned_expert_outputs,
)


def test_dispatch_unpack_and_empty_ranks():
    part = ExpertParallelPartition(num_experts=8, ep_world_size=4, ep_rank=0)
    selected = torch.tensor([[7, 0, 4], [0, 7, 4], [4, 0, 7], [7, 4, 0]], dtype=torch.long)
    plan = build_local_dispatch_plan(selected, part)

    assert torch.equal(plan.send_counts, torch.tensor([4, 0, 4, 4], dtype=torch.long))
    hidden = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    packed = pack_dispatch_tokens(hidden, plan)
    unpacked = unpack_returned_expert_outputs(packed, plan)
    assert torch.equal(unpacked[:, 0], hidden)

    # Zero-token empty rank
    empty_plan = build_local_dispatch_plan(torch.empty((0, 2), dtype=torch.long), part)
    assert empty_plan.num_tokens == 0
    assert pack_dispatch_tokens(torch.empty((0, 6)), empty_plan).shape == (0, 6)

    counts_by_rank = torch.stack([plan.send_counts] + [empty_plan.send_counts] * 3)
    receiving = build_receive_plan(counts_by_rank[:, 2], replace(part, ep_rank=2))
    assert receiving.recv_counts.tolist() == [4, 0, 0, 0]
    assert receiving.recv_offsets.tolist() == [0, 4, 4, 4, 4]
    counts_by_rank.zero_()
    assert receiving.recv_counts.tolist() == [4, 0, 0, 0]
    selected.fill_(2)
    assert plan.flat_expert_ids[:3].tolist() == [7, 0, 4]


def test_arrival_order_invariance_and_backward_combine():
    s0 = torch.tensor([[1.0e20, 2.0]], dtype=torch.float32)
    s1 = torch.tensor([[-1.0e20, 3.0]], dtype=torch.float32)
    s2 = torch.tensor([[5.0, 4.0]], dtype=torch.float32)
    pkts = [
        ExpertReturnPacket(torch.tensor([0]), torch.tensor([i]), s)
        for i, s in enumerate((s0, s1, s2))
    ]
    recon_a = reconstruct_slots_arrival_invariant(pkts, num_tokens=1, top_k=3, hidden_dim=2)
    recon_b = reconstruct_slots_arrival_invariant(
        [pkts[2], pkts[0], pkts[1]], num_tokens=1, top_k=3, hidden_dim=2
    )
    assert torch.equal(recon_a, recon_b)
    rounded = ExpertReturnPacket(
        torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([[1.001], [-1.0]])
    )
    precise = reconstruct_slots_arrival_invariant([rounded], num_tokens=1, top_k=2, hidden_dim=1)
    assert precise.dtype == torch.float32
    assert combine_returned_slots_fixed_order(precise, torch.ones((1, 2))).item() == pytest.approx(
        0.001, abs=1e-7
    )

    out = combine_returned_slots_fixed_order(
        recon_a,
        torch.ones((1, 3)),
        shared_expert_output=torch.tensor([[0.5, 0.5]]),
        residual=torch.tensor([[0.25, -0.25]]),
    )
    assert out[0, 0].item() == pytest.approx(5.75)

    part = ExpertParallelPartition(num_experts=3, ep_world_size=1, ep_rank=0)
    plan = build_local_dispatch_plan(torch.tensor([[0, 2, 1]]), part)
    hidden = torch.ones((1, 1), requires_grad=True)
    dx = torch.tensor([[1e20], [5.0], [-1e20]])
    pack_dispatch_tokens(hidden, plan).backward(dx)
    assert hidden.grad.item() == 5.0
    assert torch.equal(hidden.grad, combine_backward_input_gradients_fixed_order(dx, plan))

    with pytest.raises(RuntimeError, match="missing"):
        reconstruct_slots_arrival_invariant(
            pkts[:2], num_tokens=1, top_k=3, hidden_dim=2, dtype=torch.float32
        )

    with pytest.raises(RuntimeError, match="duplicate"):
        reconstruct_slots_arrival_invariant(pkts + pkts[:1], num_tokens=1, top_k=3, hidden_dim=2)


def _ep_gloo_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=3, timeout=timedelta(seconds=20)
    )
    try:
        partition = ExpertParallelPartition(6, 3, rank)
        selected = torch.tensor([[[5, 0], [1, 4]], [], [[0, 1]]][rank], dtype=torch.long).reshape(
            -1, 2
        )
        hidden = torch.arange(selected.size(0)).float()[:, None] + 10 * rank
        plan = build_local_dispatch_plan(selected, partition)
        recv_counts = torch.empty_like(plan.send_counts)
        dist.all_to_all_single(recv_counts, plan.send_counts)
        receive = build_receive_plan(recv_counts, partition)
        send, recv = plan.send_counts.tolist(), receive.recv_counts.tolist()
        assert (sum(send), sum(recv)) == [(4, 4), (0, 0), (2, 2)][rank]
        packed = pack_dispatch_tokens(hidden, plan)
        expert_send = plan.flat_expert_ids.index_select(0, plan.permutation)
        inputs = hidden.new_empty((sum(recv), 1))
        experts = expert_send.new_empty(sum(recv))
        dist.all_to_all_single(inputs, packed, recv, send)
        dist.all_to_all_single(experts, expert_send, recv, send)
        assert ((experts >= partition.expert_start) & (experts < partition.expert_end)).all()

        returned = hidden.new_empty((sum(send), 1))
        dist.all_to_all_single(returned, inputs + experts[:, None], send, recv)
        slots = unpack_returned_expert_outputs(returned, plan)
        assert torch.equal(slots[..., 0], hidden + selected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_dispatch_and_return_with_an_empty_rank(tmp_path):
    mp.spawn(_ep_gloo_worker, args=((tmp_path / "store").as_uri(),), nprocs=3, join=True)

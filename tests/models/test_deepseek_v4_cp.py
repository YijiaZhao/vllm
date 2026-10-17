# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 context-parallel (PCP + DCP) layout helpers."""

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4.common import cp


def _owner(pos: int, world: int, interleave: int) -> int:
    return (pos // interleave) % world


@pytest.mark.parametrize("world,interleave", [(2, 128), (4, 128), (8, 128), (8, 256)])
def test_local_count_matches_ownership(world, interleave):
    n = torch.arange(0, 5 * world * interleave + 7)
    for rank in range(world):
        expected = torch.tensor(
            [
                sum(_owner(p, world, interleave) == rank for p in range(int(k)))
                for k in n[:: interleave // 4]
            ]
        )
        got = cp.local_count(n[:: interleave // 4], rank, world, interleave)
        assert torch.equal(got, expected)
    total = sum(cp.local_count(n, r, world, interleave) for r in range(world))
    assert torch.equal(total, n)


@pytest.mark.parametrize(
    "world,interleave,ratio", [(8, 128, 4), (8, 128, 128), (4, 256, 4)]
)
def test_compressed_entries_live_on_their_last_token_owner(world, interleave, ratio):
    """A compressed entry, its tokens and its local index agree on one rank."""
    ci = interleave // ratio
    num_entries = 3 * world * ci + 5
    for k in range(num_entries):
        tokens = range(k * ratio, (k + 1) * ratio)
        owners = {_owner(p, world, interleave) for p in tokens}
        assert owners == {_owner(k, world, ci)}
    # Local index of an entry == number of entries the rank owns before it.
    total = 11
    idx = cp.gathered_compressed_index(num_entries, world, ci, total, "cpu")
    for k in range(num_entries):
        rank = _owner(k, world, ci)
        local = sum(_owner(j, world, ci) == rank for j in range(k))
        assert int(idx[k]) == rank * total + local


def test_prefill_gather_plan_is_rank_independent():
    # Two prefill requests split into two rows each, plus one decode row.
    req_idx = np.array([5, 9, 9, 2, 2])
    ub = torch.tensor([0, 600, 600, 130, 130])
    plan = cp.build_dcp_prefill_gather_plan(
        req_idx,
        ub,
        seq_lens_cpu=torch.zeros(5, dtype=torch.int32),
        num_decodes=1,
        num_reqs=5,
        compress_ratio=4,
        world_size=8,
        interleave=128,
        device="cpu",
    )
    # Requests sorted by persistent index: 2 then 9.
    assert plan.req_rows.tolist() == [3, 1]
    assert plan.shard_rows.tolist() == [130 // 4, 600 // 4]
    assert plan.offsets.tolist() == [0, 130 // 4]
    assert plan.total == 130 // 4 + 600 // 4
    assert plan.row_to_req.tolist() == [1, 1, 0, 0]


def test_owner_mask():
    pos = torch.arange(0, 2048)
    masks = [cp.owner_mask(pos, r, 8, 128) for r in range(8)]
    owners = sum(m.int() for m in masks)
    assert torch.equal(owners, torch.ones_like(pos, dtype=torch.int))
    assert masks[1][128:256].all() and not masks[1][:128].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_replicated_slot_mapping():
    device = "cuda"
    block_size = 4
    block_table = torch.tensor([[7, 3, 9], [2, 5, 1]], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 5, 9, 2, 6, 0], dtype=torch.int64, device=device)
    token_to_req = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.int32, device=device)
    out = torch.empty(8, dtype=torch.int64, device=device)
    slots = cp.replicated_slot_mapping(
        out, positions, token_to_req, block_table, block_size, num_actual_tokens=5
    )
    expected = [7 * 4 + 0, 3 * 4 + 1, 9 * 4 + 1, 2 * 4 + 2, 5 * 4 + 2, -1]
    assert slots.tolist() == expected

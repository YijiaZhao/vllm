# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


def apply_attn_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    sink = attn_sink[: out.shape[1]].to(dtype=lse.dtype)
    output_lse = torch.logaddexp(lse, sink.unsqueeze(0))
    scale = torch.exp(lse - output_lse).to(dtype=out.dtype)
    return out * scale.unsqueeze(-1)


def dcp_merge_flashmla_output(
    local_out: torch.Tensor,
    local_lse: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    group: "GroupCoordinator",
    use_a2a: bool = True,
) -> None:
    if use_a2a:
        out, lse = dcp_a2a_lse_reduce(
            local_out,
            local_lse,
            group,
            return_lse=True,
        )
    else:
        out, lse = cp_lse_ag_out_rs(
            local_out,
            local_lse,
            group,
            return_lse=True,
        )
    output[:, : out.shape[1], :].copy_(apply_attn_sink(out, lse, attn_sink))


def dcp_softmax_reduce(
    local_max: torch.Tensor,
    local_sum: torch.Tensor,
    local_weighted_value: torch.Tensor,
    group: "GroupCoordinator",
) -> torch.Tensor:
    """Merge numerically stable partial softmax statistics across DCP ranks."""
    valid = local_sum > 0
    local_max = torch.where(
        valid,
        local_max,
        torch.full_like(local_max, -float("inf")),
    )
    gathered_max = group.all_gather(local_max, dim=0).reshape(
        (group.world_size,) + local_max.shape
    )
    global_max = gathered_max.max(dim=0).values

    scale = torch.exp(local_max - global_max)
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    reduce_payload = torch.stack(
        (
            torch.where(valid, local_sum * scale, torch.zeros_like(local_sum)),
            torch.where(
                valid,
                local_weighted_value * scale,
                torch.zeros_like(local_weighted_value),
            ),
        )
    )
    global_sum, global_weighted_value = group.all_reduce(reduce_payload).unbind(0)
    return torch.where(
        global_sum > 0,
        global_weighted_value / global_sum,
        torch.zeros_like(global_weighted_value),
    )


def dcp_softmax_reduce_packed(
    local_max: torch.Tensor,
    local_sum: torch.Tensor,
    local_weighted_value: torch.Tensor,
    group: "GroupCoordinator",
) -> torch.Tensor:
    """Same result as ``dcp_softmax_reduce`` with a single all_gather.

    Ranks with no local entries carry (max=-inf, sum=0, value=0).
    """
    packed = torch.stack((local_max, local_sum, local_weighted_value))
    gathered = group.all_gather(packed.unsqueeze(0), dim=0)  # [W, 3, N, D]
    m, s, v = gathered.unbind(1)
    global_max = m.max(dim=0).values
    scale = torch.exp(m - global_max.clamp_min(-3.0e38))
    scale = torch.where(s > 0, scale, torch.zeros_like(scale))
    global_sum = (s * scale).sum(dim=0)
    global_value = (v * scale).sum(dim=0)
    return torch.where(
        global_sum > 0, global_value / global_sum, torch.zeros_like(global_value)
    )

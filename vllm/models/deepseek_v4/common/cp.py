# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill/decode context parallelism (PCP + DCP) for DeepSeek-V4.

Layout, with DCP spanning the PCP axis (``dcp == pcp``):

* The compressed MLA KV and the indexer K are DCP-sharded by token position
  with ``cp_kv_cache_interleave_size`` (a multiple of every compress ratio), so
  a compressed entry lives on the rank that owns its group's last token.
* The SWA KV and the compressor states are small and replicated: every rank
  writes every token of the step into them.
* Decode rows are replicated on every rank. A step without prefill rows writes
  the caches from the rank-local tokens; a step with prefill rows all-gathers
  the cache inputs across PCP ("gathered layout") so every rank sees every
  token, and writes the compressed entries it owns.
* Decode attention is a partial over this rank's compressed entries (the SWA
  window is counted on DCP rank 0 only), merged across ranks by LSE; the sink
  is applied once after the merge. Prefill attention gathers the compressed KV
  of each prefill request from all ranks and runs locally.
"""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import CommonAttentionMetadata

PAD_SLOT_ID = -1


def dsv4_uses_cp(vllm_config: VllmConfig) -> bool:
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.prefill_context_parallel_size > 1
        or parallel_config.decode_context_parallel_size > 1
    )


def validate_dsv4_cp_config(vllm_config: VllmConfig) -> None:
    """Reject CP layouts the DeepSeek-V4 CP path does not implement."""
    if not dsv4_uses_cp(vllm_config):
        return
    parallel_config = vllm_config.parallel_config
    pcp = parallel_config.prefill_context_parallel_size
    dcp = parallel_config.decode_context_parallel_size
    if pcp != dcp:
        raise NotImplementedError(
            "DeepSeek-V4 context parallelism requires the DCP group to span the "
            f"PCP axis (decode_context_parallel_size == "
            f"prefill_context_parallel_size); got PCP={pcp}, DCP={dcp}."
        )
    interleave = parallel_config.cp_kv_cache_interleave_size
    ratios = {
        int(r)
        for r in getattr(vllm_config.model_config.hf_config, "compress_ratios", [])
        if int(r) > 1
    }
    # The sharded caches (compressed MLA KV, indexer K) use 256-token pages.
    if 256 % interleave != 0:
        raise ValueError(
            "DeepSeek-V4 context parallelism requires "
            "--cp-kv-cache-interleave-size to divide the 256-token KV page; "
            f"got {interleave}."
        )
    for ratio in ratios:
        if interleave % ratio != 0:
            raise ValueError(
                "DeepSeek-V4 context parallelism requires "
                "--cp-kv-cache-interleave-size to be a multiple of every compress "
                f"ratio {sorted(ratios)}; got {interleave}."
            )
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if (
        capability is not None
        and capability.major == 12
        and parallel_config.decode_context_parallel_size > 1
        and interleave != 256
    ):
        # The SM120 prefill gathers whole compressed pages across DCP ranks.
        raise ValueError(
            "DeepSeek-V4 DCP on SM120 requires --cp-kv-cache-interleave-size 256; "
            f"got {interleave}."
        )
    unsupported = []
    if vllm_config.speculative_config is not None:
        unsupported.append("speculative decoding")
    if vllm_config.cache_config.enable_prefix_caching:
        unsupported.append("prefix caching")
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is not None:
        # P/D: NixlPushConnector moves whole 256-token pages between DCP
        # ranks, which needs one page per interleave chunk on both sides.
        if kv_transfer_config.kv_role not in ("kv_consumer", "kv_producer") or (
            kv_transfer_config.kv_connector != "NixlPushConnector"
        ):
            unsupported.append("KV transfer other than NixlPushConnector")
        elif interleave != 256:
            raise ValueError(
                "DeepSeek-V4 context parallelism with KV transfer requires "
                f"--cp-kv-cache-interleave-size 256; got {interleave}."
            )
    if parallel_config.pipeline_parallel_size != 1:
        unsupported.append("pipeline parallelism")
    if parallel_config.use_ubatching:
        unsupported.append("microbatching")
    if unsupported:
        raise NotImplementedError(
            "DeepSeek-V4 context parallelism does not support "
            + ", ".join(unsupported)
            + "."
        )


def owner_mask(
    positions: torch.Tensor, rank: int, world_size: int, interleave: int
) -> torch.Tensor:
    """Whether this DCP rank owns the KV of each token position."""
    return (positions // interleave) % world_size == rank


def local_count(
    num_tokens: torch.Tensor, rank: int, world_size: int, interleave: int
) -> torch.Tensor:
    """Tokens among positions [0, num_tokens) that this DCP rank owns."""
    span = interleave * world_size
    rem = num_tokens % span - rank * interleave
    return num_tokens // span * interleave + rem.clamp(min=0, max=interleave)


@triton.jit
def _replicated_slots_kernel(
    out_ptr,
    positions_ptr,
    token_to_req_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    num_actual_tokens,
    num_tokens,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_tokens
    valid = offs < num_actual_tokens
    pos = tl.load(positions_ptr + offs, mask=valid, other=0)
    req = tl.load(token_to_req_ptr + offs, mask=valid, other=0)
    block = tl.load(
        block_table_ptr + req.to(tl.int64) * block_table_stride + pos // block_size,
        mask=valid,
        other=0,
    )
    slot = block.to(tl.int64) * block_size + pos % block_size
    tl.store(out_ptr + offs, tl.where(valid, slot, -1), mask=in_range)


def replicated_slot_mapping(
    out: torch.Tensor,
    positions: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_actual_tokens: int,
) -> torch.Tensor:
    """Rank-local slots of a replicated (non DCP-sharded) cache group.

    Under PCP the runner hands every group a gathered slot mapping in which a
    replicated row (a decode, or a prefill too short to split) is only written
    from PCP rank 0's copy. Queries need the slots of their own tokens, so
    rebuild them from the block table; padding tokens get PAD_SLOT_ID.
    """
    num_tokens = positions.shape[0]
    out = out[:num_tokens]
    if num_tokens == 0:
        return out
    BLOCK = 1024
    _replicated_slots_kernel[(triton.cdiv(num_tokens, BLOCK),)](
        out,
        positions,
        token_to_req,
        block_table,
        block_table.stride(0),
        block_size,
        num_actual_tokens,
        num_tokens,
        BLOCK=BLOCK,
    )
    return out


def padded_token_to_req(
    cm: "CommonAttentionMetadata", buffer: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """Token -> request row for the PCP-padded local tokens, in ``buffer``.

    ``token_to_req_indices`` covers the actual tokens only and may return a
    mapping cached in another builder's buffer.
    """
    mapping = cm.token_to_req_indices(buffer)
    out = buffer[:num_tokens]
    if mapping.data_ptr() != out.data_ptr():
        out[: mapping.shape[0]].copy_(mapping)
    return out


def gather_token_rows(
    token_to_req: torch.Tensor,
    num_tokens: int,
    num_actual_tokens: int,
    req_idx: np.ndarray | None,
    num_reqs: int,
    lookup: torch.Tensor,
) -> torch.Tensor:
    """Map the PCP-gathered tokens to this rank's request rows.

    Every rank holds a row for every request of the step (decodes and short
    prefills are replicated, split prefills give each rank two chunks), so a
    gathered token's request is found through its persistent request index.
    """
    # num_tokens is the PCP-padded local count, equal on every rank.
    device = token_to_req.device
    if req_idx is None:
        req_idx = np.arange(num_reqs)
    row_req = torch.from_numpy(np.ascontiguousarray(req_idx[:num_reqs], np.int32)).to(
        device, non_blocking=True
    )
    token_req = torch.full((num_tokens,), -1, dtype=torch.int32, device=device)
    token_req[:num_actual_tokens] = row_req[token_to_req[:num_actual_tokens].long()]
    gathered = get_pcp_group().all_gather(token_req, dim=0)
    lookup.fill_(0)
    lookup[row_req.long()] = torch.arange(num_reqs, dtype=torch.int32, device=device)
    return lookup[gathered.clamp(min=0).long()]


@dataclass
class DCPPrefillGatherPlan:
    """Per-step plan to gather the compressed KV of every prefill request.

    Every rank dequantizes its own entries of each prefill request into a
    ``[sum(shard_rows), D]`` buffer (requests sorted by persistent index), then
    all-gathers it. Global compressed entry ``k`` of request ``i`` sits at
    ``rank(k) * total + offset[i] + local(k)`` of the gathered buffer.
    """

    # One local request row per prefill request, in gather order.
    req_rows: torch.Tensor  # [num_prefill_reqs] (device)
    shard_rows: torch.Tensor  # [num_prefill_reqs] compressed entries per rank
    offsets: torch.Tensor  # [num_prefill_reqs] (device) start in a rank's shard
    total: int  # sum(shard_rows): one rank's shard length
    # For every prefill row (rows after the decodes): its prefill request.
    row_to_req: torch.Tensor  # [num_prefill_rows] (device)


def build_dcp_prefill_gather_plan(
    req_idx: np.ndarray | None,
    shard_tokens_upper_bound: torch.Tensor | None,
    seq_lens_cpu: torch.Tensor,
    num_decodes: int,
    num_reqs: int,
    compress_ratio: int,
    world_size: int,
    interleave: int,
    device: torch.device,
) -> DCPPrefillGatherPlan:
    if req_idx is None:
        req_idx = np.arange(num_reqs)
    prefill_req = np.asarray(req_idx[num_decodes:num_reqs])
    if shard_tokens_upper_bound is None:
        # Dummy batches bypass the PCP manager: one row per request.
        span = interleave * world_size
        n = seq_lens_cpu[num_decodes:num_reqs].numpy().astype(np.int64)
        ub = n // span * interleave + np.minimum(n % span, interleave)
    else:
        ub = shard_tokens_upper_bound[num_decodes:num_reqs].numpy().astype(np.int64)
    uniq, first_row, inverse = np.unique(
        prefill_req, return_index=True, return_inverse=True
    )
    shard_rows = np.maximum(ub[first_row] // compress_ratio, 1)
    offsets = np.zeros_like(shard_rows)
    offsets[1:] = np.cumsum(shard_rows)[:-1]
    return DCPPrefillGatherPlan(
        req_rows=torch.from_numpy((first_row + num_decodes).astype(np.int32)).to(
            device, non_blocking=True
        ),
        shard_rows=torch.from_numpy(shard_rows.astype(np.int32)),
        offsets=torch.from_numpy(offsets.astype(np.int64)).to(
            device, non_blocking=True
        ),
        total=int(shard_rows.sum()),
        row_to_req=torch.from_numpy(inverse.astype(np.int64)).to(
            device, non_blocking=True
        ),
    )


def gathered_compressed_index(
    num_entries: int,
    world_size: int,
    compressed_interleave: int,
    total: int,
    device: torch.device,
) -> torch.Tensor:
    """Gathered-buffer offset of global compressed entries [0, num_entries)."""
    k = torch.arange(num_entries, device=device, dtype=torch.int64)
    span = compressed_interleave * world_size
    rank = (k // compressed_interleave) % world_size
    local = k // span * compressed_interleave + k % compressed_interleave
    return rank * total + local


def apply_attn_sink(
    out: torch.Tensor, lse: torch.Tensor, attn_sink: torch.Tensor
) -> torch.Tensor:
    """Scale a sink-free attention output by exp(lse) / (exp(lse) + exp(sink))."""
    sink = attn_sink[: out.shape[1]].to(dtype=lse.dtype)
    scale = torch.sigmoid(lse - sink.unsqueeze(0))
    return out * scale.unsqueeze(-1).to(out.dtype)


def dcp_merge_decode(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    """Merge per-rank partial decode attention [T, H, D] by LSE ([T, H], base e).

    Rows a rank has no entries for must carry (out=0, lse=-inf).
    """
    group = get_dcp_group()
    lses = group.all_gather(lse.contiguous(), dim=0).view(group.world_size, *lse.shape)
    lse_max = lses.max(dim=0).values
    lse_max = torch.where(torch.isfinite(lse_max), lse_max, torch.zeros_like(lse_max))
    total = torch.exp(lses - lse_max).sum(dim=0)
    lse_total = torch.log(total) + lse_max
    scale = torch.exp(lse - lse_total)
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    merged = group.all_reduce(out.float() * scale.unsqueeze(-1))
    return apply_attn_sink(merged, lse_total, attn_sink).to(out.dtype)


@triton.jit
def _dcp_local_topk_kernel(
    out_ptr,
    out_stride,
    lens_ptr,
    topk_ptr,
    topk_stride,
    token_to_req_ptr,
    block_table_ptr,
    block_table_stride,
    is_valid_token_ptr,
    block_size,
    dcp_rank,
    TOPK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    req = tl.load(token_to_req_ptr + token).to(tl.int64)
    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, TOPK, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        in_range = offs < TOPK
        g = tl.load(topk_ptr + token * topk_stride + offs, mask=in_range, other=-1)
        owned = (g >= 0) & ((g // INTERLEAVE) % DCP_WORLD_SIZE == dcp_rank)
        local = g // (INTERLEAVE * DCP_WORLD_SIZE) * INTERLEAVE + g % INTERLEAVE
        local = tl.where(owned, local, 0)
        block = tl.load(
            block_table_ptr + req * block_table_stride + local // block_size,
            mask=owned,
            other=0,
        )
        slot = block * block_size + local % block_size
        dst = count + tl.cumsum(owned.to(tl.int32), 0) - 1
        tl.store(out_ptr + token * out_stride + dst, slot, mask=owned)
        count += tl.sum(owned.to(tl.int32), axis=0)
    # Unused tail stays -1 (the caller pre-fills); padding tokens get length 0.
    is_valid = tl.load(is_valid_token_ptr + token)
    tl.store(lens_ptr + token, tl.where(is_valid, count, 0))


def dcp_local_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
    compressed_interleave: int,
    out: torch.Tensor,
    lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep this rank's global compressed top-k ids as compacted local slots."""
    group = get_dcp_group()
    num_tokens, topk = topk_indices.shape
    out = out[:num_tokens, :topk]
    lens = lens[:num_tokens]
    out.fill_(-1)
    if num_tokens == 0:
        return out, lens
    _dcp_local_topk_kernel[(num_tokens,)](
        out,
        out.stride(0),
        lens,
        topk_indices,
        topk_indices.stride(0),
        token_to_req,
        block_table,
        block_table.stride(0),
        is_valid_token,
        block_size,
        group.rank_in_group,
        TOPK=topk,
        DCP_WORLD_SIZE=group.world_size,
        INTERLEAVE=compressed_interleave,
        BLOCK=1024,
    )
    return out, lens

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
* Decode-only steps shard the rest of the layer across DCP ("decode SP", TP=1
  with EP only): the residual stream, mHC and MoE hold 1/DCP of the tokens;
  attention all-gathers its input, computes q/kv/indexer/compressor for every
  token (each rank needs every head's query over its KV shard), reduce-scatters
  the merged output by head group, runs wo_a/wo_b on its own groups and
  reduce-scatters the partial sums back to token shards. Steps with prefill
  rows (and PIECEWISE graphs, which mixed steps replay) keep every token on
  every rank.
* With a MoE that takes replicated tokens (not MegaMoE), decode-only steps
  keep every token on every rank instead, like TP: attention's per-group
  wo_a/wo_b partial sums are all-reduced, and the MoE runs this rank's experts
  on every token and all-reduces ("decode AR"); no token all-gathers.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config import CUDAGraphMode
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
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


def real_token_count(cm: "CommonAttentionMetadata") -> int:
    """Tokens of the step's requests, excluding CUDA graph padding.

    ``num_actual_tokens`` counts the padded tokens of a FULL graph batch, whose
    request rows map to request 0 and whose positions are 0: writing their
    replicated cache slots would land in that request's stale (since freed and
    reassigned) sliding-window blocks.
    """
    return int(cm.query_start_loc_cpu[-1])


def dcp_decode_sp_supported(vllm_config: VllmConfig) -> bool:
    """Whether decode-only steps shard the non-attention work across DCP."""
    parallel_config = vllm_config.parallel_config
    dcp = parallel_config.decode_context_parallel_size
    hf_config = vllm_config.model_config.hf_config
    return (
        dcp > 1
        and os.environ.get("VLLM_DSV4_CP_DECODE_SP", "1") == "1"
        and parallel_config.tensor_parallel_size == 1
        and parallel_config.data_parallel_size == 1
        and parallel_config.enable_expert_parallel
        and hf_config.o_groups % dcp == 0
    )


def dcp_decode_ar_supported(vllm_config: VllmConfig) -> bool:
    """Decode SP with replicated tokens: the MoE is not a token-sharded MegaMoE."""
    from vllm.config.kernel import MEGA_MOE_BACKENDS

    return (
        dcp_decode_sp_supported(vllm_config)
        and vllm_config.kernel_config.moe_backend not in MEGA_MOE_BACKENDS
        and os.environ.get("VLLM_DSV4_CP_DECODE_AR", "1") == "1"
    )


def dcp_decode_ar_active() -> bool:
    """Set by the model on decode-AR steps; read by the MoE runner."""
    return is_forward_context_available() and getattr(
        get_forward_context(), "dsv4_dcp_decode_ar", False
    )


def dcp_decode_sp_active() -> bool:
    """Whether this step runs decode SP (see the module docstring).

    Every rank decides the same: decode-only steps of a FULL graph or eager
    run, from the step's attention metadata. PIECEWISE graphs are shared with
    mixed steps, so they never do.
    """
    if not is_forward_context_available():
        return False
    ctx = get_forward_context()
    active = getattr(ctx, "dsv4_dcp_decode_sp", None)
    if active is None:
        active = False
        attn_metadata = ctx.attn_metadata
        if ctx.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE and isinstance(
            attn_metadata, dict
        ):
            for metadata in attn_metadata.values():
                num_prefills = getattr(metadata, "num_prefills", None)
                if num_prefills is not None:
                    active = num_prefills == 0 and metadata.num_decodes > 0
                    break
        ctx.dsv4_dcp_decode_sp = active
    return active


def _dcp_sp_pad(x: torch.Tensor, value=0) -> torch.Tensor:
    pad = (-x.shape[0]) % get_dcp_group().world_size
    if pad == 0:
        return x
    return torch.nn.functional.pad(x, (0, 0) * (x.ndim - 1) + (0, pad), value=value)


def dcp_sp_shard(x: torch.Tensor) -> torch.Tensor:
    """This DCP rank's token shard of a replicated ``[T, ...]`` tensor."""
    group = get_dcp_group()
    x = _dcp_sp_pad(x)
    chunk = x.shape[0] // group.world_size
    return x[group.rank_in_group * chunk : (group.rank_in_group + 1) * chunk]


def dcp_sp_padding_mask(is_padding: torch.Tensor | None, num_tokens: int, device):
    """The padding mask of this rank's token shard; shard padding is padding."""
    if is_padding is None:
        is_padding = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    return dcp_sp_shard(_dcp_sp_pad(is_padding[:num_tokens], value=True))


# Decode SP collectives are small and latency-bound. vLLM's custom (P2P)
# all-gather / reduce-scatter beat NCCL there, also over PCIe, where vLLM only
# enables them for the TP group of fully connected GPUs. Byte bounds per call
# (SM120 x8 PCIe: 3-6x faster than NCCL below them, slower well above).
_DCP_CUSTOM_COMM = None
_CUSTOM_AG_MAX_INPUT_BYTES = 64 * 1024
_CUSTOM_RS_MAX_INPUT_BYTES = 512 * 1024
_CUSTOM_AR_MAX_INPUT_BYTES = 2 * 1024 * 1024


def init_dcp_custom_comm(device: torch.device) -> None:
    """Create the DCP group's P2P communicator once (collective, at init)."""
    global _DCP_CUSTOM_COMM
    if _DCP_CUSTOM_COMM is not None:
        return
    _DCP_CUSTOM_COMM = False
    if os.environ.get("VLLM_DSV4_CP_CUSTOM_COMM", "1") != "1":
        return
    device_communicator = get_dcp_group().device_communicator
    existing = getattr(device_communicator, "ca_comm", None)
    if existing is not None:
        if not existing.disabled:
            _DCP_CUSTOM_COMM = existing
        return
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm.platforms import current_platform

    # CustomAllreduce refuses PCIe-only groups above two GPUs up front; its
    # P2P test still guards hosts without peer access (cross-socket .2).
    is_fully_connected = current_platform.is_fully_connected
    current_platform.is_fully_connected = lambda physical_device_ids: True
    try:
        comm = CustomAllreduce(get_dcp_group().cpu_group, device)
    finally:
        current_platform.is_fully_connected = is_fully_connected
    if not comm.disabled:
        _DCP_CUSTOM_COMM = comm
        # As the group's custom all-reduce it joins graph capture, which
        # registers the buffers captured all-reduces read from.
        device_communicator.ca_comm = comm
    _init_dcp_pcie_ipc_all_reduce(device)


def _init_dcp_pcie_ipc_all_reduce(device: torch.device) -> None:
    """FlashInfer's PCIe all-reduce for the DCP group: push-based, it beats the
    custom (pull) all-reduce 2-3x on PCIe-only GPUs (SM12x x8: 11/19/46 us
    vs 16/43/129 us at 1/32/128 x 4096 bf16). Set up by kernel warmup, it
    joins graph capture as the group's PCIe IPC backend."""
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if (
        os.environ.get("VLLM_DSV4_CP_PCIE_IPC_AR", "1") != "1"
        or capability is None
        or capability.major != 12
    ):
        return
    from vllm.distributed.device_communicators.flashinfer_pcie_ipc_all_reduce import (  # noqa: E501
        FlashInferPcieIpcAllReduce,
    )

    group = get_dcp_group()
    device_communicator = group.device_communicator
    if getattr(device_communicator, "fi_pcie_ipc_ar_comm", None) is not None:
        return
    comm = FlashInferPcieIpcAllReduce(
        group=group.device_group, tune_group=group.cpu_group, device=device
    )
    if not comm.disabled:
        device_communicator.fi_pcie_ipc_ar_comm = comm


def dcp_small_all_gather(x: torch.Tensor) -> torch.Tensor | None:
    """Dim-0 all-gather over the DCP P2P communicator, or None if too large."""
    if _DCP_CUSTOM_COMM and x.nbytes <= _CUSTOM_AG_MAX_INPUT_BYTES:
        return _DCP_CUSTOM_COMM.custom_all_gather(x.contiguous())
    return None


def dcp_sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    out = dcp_small_all_gather(x)
    if out is not None:
        return out
    return get_dcp_group().all_gather(x, dim=0)


def dcp_sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    x = _dcp_sp_pad(x)
    if _DCP_CUSTOM_COMM and x.nbytes <= _CUSTOM_RS_MAX_INPUT_BYTES:
        out = _DCP_CUSTOM_COMM.custom_reduce_scatter(x.contiguous())
        if out is not None:
            return out
    return get_dcp_group().reduce_scatter(x, dim=0)


def dcp_all_reduce(x: torch.Tensor) -> torch.Tensor:
    pcie_comm = getattr(get_dcp_group().device_communicator, "fi_pcie_ipc_ar_comm", None)
    if pcie_comm is not None:
        x = x.contiguous()
        if pcie_comm.should_use(x):
            return pcie_comm.all_reduce(x)
    if _DCP_CUSTOM_COMM and x.nbytes <= _CUSTOM_AR_MAX_INPUT_BYTES:
        out = _DCP_CUSTOM_COMM.custom_all_reduce(x.contiguous())
        if out is not None:
            return out
    return get_dcp_group().all_reduce(x)


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


def get_dcp_decode_merge_workspace(
    vllm_config: VllmConfig,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Direct (symmetric-memory P2P) head-scattering merge for decode SP.

    One peer-to-peer exchange of (out, lse) replaces the NCCL lse all-gather
    plus output reduce-scatter. None where symmetric memory is unavailable.
    """
    from vllm.v1.attention.ops.dcp import (
        get_dcp_workspace_max_num_tokens,
        get_direct_dcp_a2a_workspace,
    )

    group = get_dcp_group()
    return get_direct_dcp_a2a_workspace(
        group,
        device,
        get_dcp_workspace_max_num_tokens(vllm_config),
        num_heads // group.world_size,
        head_dim,
        dtype,
        1,
    )


def dcp_merge_decode(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    scatter_heads: bool = False,
    workspace=None,
) -> torch.Tensor:
    """Merge per-rank partial decode attention [T, H, D] by LSE ([T, H], base e).

    Rows a rank has no entries for must carry (out=0, lse=-inf). With
    ``scatter_heads`` each rank gets only its 1/DCP of the heads, [T, H/DCP, D],
    through ``workspace`` (get_dcp_decode_merge_workspace) when given.
    """
    group = get_dcp_group()
    if scatter_heads and workspace is not None:
        if group.rank_in_group == 0:
            # The sink is one more softmax term with a zero value: fold it
            # into rank 0's partial so the plain LSE merge applies it.
            sink = attn_sink[: out.shape[1]].to(torch.float32)
            lse = lse.float()
            sink_lse = torch.logaddexp(lse, sink.unsqueeze(0))
            out = out.float() * torch.exp(lse - sink_lse).unsqueeze(-1)
            out, lse = out.to(workspace_dtype(workspace)), sink_lse
        return workspace.lse_reduce(
            out.contiguous(), lse.float().contiguous(), is_lse_base_on_e=True
        )
    lses = group.all_gather(lse.contiguous(), dim=0).view(group.world_size, *lse.shape)
    lse_max = lses.max(dim=0).values
    lse_max = torch.where(torch.isfinite(lse_max), lse_max, torch.zeros_like(lse_max))
    total = torch.exp(lses - lse_max).sum(dim=0)
    lse_total = torch.log(total) + lse_max
    scale = torch.exp(lse - lse_total)
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    weighted = out.float() * scale.unsqueeze(-1)
    if scatter_heads:
        heads = out.shape[1] // group.world_size
        start = group.rank_in_group * heads
        merged = group.reduce_scatter(weighted.transpose(0, 1).contiguous(), dim=0)
        return apply_attn_sink(
            merged.transpose(0, 1),
            lse_total[:, start : start + heads],
            attn_sink[start:],
        ).to(out.dtype)
    merged = group.all_reduce(weighted)
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


@triton.jit
def _dcp_sp_merge_prep_kernel(
    out_ptr,
    out_stride_t,
    out_stride_h,
    lse_ptr,
    lse_stride_t,
    lens_a_ptr,
    lens_b_ptr,
    sink_ptr,
    out_c_ptr,
    lse_c_ptr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FOLD_SINK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)
    empty = (tl.load(lens_a_ptr + token) + tl.load(lens_b_ptr + token)) == 0
    lse = tl.load(lse_ptr + token * lse_stride_t + head).to(tl.float32)
    out = tl.load(out_ptr + token * out_stride_t + head * out_stride_h + d).to(
        tl.float32
    )
    lse = tl.where(empty, float("-inf"), lse)
    out = tl.where(empty, 0.0, out)
    if FOLD_SINK:
        sink = tl.load(sink_ptr + head).to(tl.float32)
        hi = tl.maximum(lse, sink)
        folded = hi + tl.log(tl.exp(lse - hi) + tl.exp(sink - hi))
        out = out * tl.exp(lse - folded)
        lse = folded
    dst = token * NUM_HEADS + head
    tl.store(out_c_ptr + dst * HEAD_DIM + d, out.to(out_c_ptr.dtype.element_ty))
    tl.store(lse_c_ptr + dst, lse)


def dcp_sp_merge_decode(
    out: torch.Tensor,
    lse: torch.Tensor,
    lens_a: torch.Tensor,
    lens_b: torch.Tensor,
    attn_sink: torch.Tensor,
    workspace,
) -> torch.Tensor:
    """Decode SP merge: this rank's heads of the LSE-merged attention.

    ``out`` [T, H, D] / ``lse`` [T, H] are this rank's partials (strided views
    are fine); rows with ``lens_a + lens_b == 0`` have no local entry and drop
    out of the merge. One kernel masks them, folds the sink into rank 0's
    partial (a zero-valued softmax term) and packs the inputs of the direct
    P2P exchange (get_dcp_decode_merge_workspace).
    """
    num_tokens, num_heads, head_dim = out.shape
    out_c = torch.empty(
        (num_tokens, num_heads, head_dim),
        dtype=workspace.received_output.dtype,
        device=out.device,
    )
    lse_c = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=out.device)
    if num_tokens > 0:
        assert out.stride(2) == 1
        _dcp_sp_merge_prep_kernel[(num_tokens, num_heads)](
            out,
            out.stride(0),
            out.stride(1),
            lse,
            lse.stride(0),
            lens_a,
            lens_b,
            attn_sink,
            out_c,
            lse_c,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            FOLD_SINK=get_dcp_group().rank_in_group == 0,
        )
    return workspace.lse_reduce(out_c, lse_c, is_lse_base_on_e=True)


@triton.jit
def _dcp_local_lse_kernel(
    lse_ptr,
    lse_stride_t,
    lens_a_ptr,
    lens_b_ptr,
    lse_c_ptr,
    NUM_HEADS: tl.constexpr,
):
    token = tl.program_id(0)
    h = tl.arange(0, NUM_HEADS)
    empty = (tl.load(lens_a_ptr + token) + tl.load(lens_b_ptr + token)) == 0
    lse = tl.load(lse_ptr + token * lse_stride_t + h).to(tl.float32)
    lse = tl.where(empty, float("-inf"), lse)
    tl.store(lse_c_ptr + token * NUM_HEADS + h, lse)


@triton.jit
def _dcp_local_scale_kernel(
    out_ptr,
    out_stride_t,
    out_stride_h,
    lse_all_ptr,
    sink_ptr,
    out_s_ptr,
    num_tokens,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)
    r = tl.arange(0, WORLD)
    lses = tl.load(lse_all_ptr + (r * num_tokens + token) * NUM_HEADS + head)
    sink = tl.load(sink_ptr + head).to(tl.float32)
    hi = tl.maximum(tl.max(lses, axis=0), sink)
    total = hi + tl.log(tl.sum(tl.exp(lses - hi), axis=0) + tl.exp(sink - hi))
    own = tl.load(lse_all_ptr + (RANK * num_tokens + token) * NUM_HEADS + head)
    weight = tl.exp(own - total)
    out = tl.load(out_ptr + token * out_stride_t + head * out_stride_h + d).to(
        tl.float32
    )
    out = tl.where(own == float("-inf"), 0.0, out * weight)
    dst = (token * NUM_HEADS + head) * HEAD_DIM
    tl.store(out_s_ptr + dst + d, out.to(out_s_ptr.dtype.element_ty))


# From this many decode tokens on, decode SP skips the head exchange of the
# merge: ranks share only their LSEs, and each runs wo_a/wo_b over all heads
# of its own partial, scaled by its global softmax weight. wo_b's partial sums
# then add up in the decoder layer's all-reduce / reduce-scatter. That trades
# the exchange, growing with the batch, for full o_proj weight reads.
_LOCAL_OPROJ_MIN_TOKENS = int(
    os.environ.get("VLLM_DSV4_CP_DECODE_LOCAL_OPROJ_MIN_TOKENS", "48")
)


def dcp_sp_local_scale_decode(
    out: torch.Tensor,
    lse: torch.Tensor,
    lens_a: torch.Tensor,
    lens_b: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor | None:
    """All heads of this rank's partial, weighted for an o_proj summed across
    DCP ranks, or None below the token threshold (see above)."""
    num_tokens, num_heads, head_dim = out.shape
    if num_tokens < _LOCAL_OPROJ_MIN_TOKENS or num_tokens == 0:
        return None
    assert out.stride(2) == 1
    group = get_dcp_group()
    lse_c = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=out.device)
    _dcp_local_lse_kernel[(num_tokens,)](
        lse, lse.stride(0), lens_a, lens_b, lse_c, NUM_HEADS=num_heads
    )
    lse_all = dcp_sp_all_gather(lse_c)
    out_s = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=out.dtype, device=out.device
    )
    _dcp_local_scale_kernel[(num_tokens, num_heads)](
        out,
        out.stride(0),
        out.stride(1),
        lse_all,
        attn_sink,
        out_s,
        num_tokens,
        WORLD=group.world_size,
        RANK=group.rank_in_group,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
    )
    return out_s


def workspace_dtype(workspace) -> torch.dtype:
    return workspace.received_output.dtype


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

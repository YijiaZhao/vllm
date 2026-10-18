# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 FlashMLA sparse backend, metadata, and metadata builder."""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.models.deepseek_v4.common.cp import (
    PAD_SLOT_ID,
    DCPPrefillGatherPlan,
    build_dcp_prefill_gather_plan,
    owner_mask,
    real_token_count,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

# Pad C128A topk width to this alignment. 128 covers both h_q=64 (B_TOPK=64) and
# h_q=128 (B_TOPK=128). FlashMLA decode asserts extra_topk % B_TOPK == 0;
# unaligned widths (e.g. 17 = ceil(2136/128)) crash the sm100 head64 kernel.
# Padded slots stay -1 and decode_lens caps them via topk_length, so the pad is a
# no-op at kernel level. Mirrors _SPARSE_PREFILL_TOPK_ALIGNMENT in cache_utils.py.
_C128A_TOPK_ALIGNMENT = 128


class DeepseekV4SparseMLABackend(AttentionBackend):
    """DeepSeek-V4 sparse-MLA backend base.

    Subclasses ``AttentionBackend`` directly (not the V3.2
    ``FlashMLASparseBackend``): DeepSeek-V4 runs its own attention layer
    (``DeepseekV4Attention``), so it does not reuse the V3.2 builder or impl, and
    only needs to declare its own metadata builder, KV-cache layout, and the
    sparse-MLA capability flags.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [256]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4SparseMLAMetadataBuilder"]:
        return DeepseekV4SparseMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[Any]:
        # DeepSeek-V4 runs its attention through ``DeepseekV4Attention.forward``,
        # not the generic ``Attention``/``MLAAttention`` layer, so the backend's
        # impl class is never instantiated.
        raise NotImplementedError(
            "DeepseekV4SparseMLABackend has no separate impl class; DeepSeek-V4 "
            "attention runs through DeepseekV4Attention."
        )

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V4 layout: 448 NoPE + 64 RoPE = 512.
        return [512]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [9, 10]


@dataclass
class DeepseekV4FlashMLAMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int
    topk_tokens: int

    # Pre-computed C128A metadata (compress_ratio == 128 only).
    # Decode: global slot ids + valid-entry counts (fused from positions).
    c128a_global_decode_topk_indices: torch.Tensor | None = None
    c128a_decode_topk_lens: torch.Tensor | None = None
    # Prefill: local topk indices (used by combine_topk_swa_indices).
    c128a_prefill_topk_indices: torch.Tensor | None = None
    # DCP: how to gather the prefill requests' compressed KV from every rank.
    dcp_prefill_plan: DCPPrefillGatherPlan | None = None


class DeepseekV4SparseMLAMetadataBuilder(
    AttentionMetadataBuilder[DeepseekV4FlashMLAMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        # Classify single-token queries (plus num_speculative_tokens via
        # supports_spec_as_decode=True) as decodes; longer queries go to prefill.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.topk_tokens = self.model_config.hf_config.index_topk

        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,), dtype=torch.int32, device=device
        )

        assert isinstance(self.kv_cache_spec.tokens_per_state, int)
        self.compress_ratio = self.kv_cache_spec.tokens_per_state

        # Pre-allocate compressed slot mapping buffer for CUDA graph address
        # stability when compress_ratio > 1.
        if self.compress_ratio > 1:
            self.compressed_slot_mapping_buffer = torch.empty(
                max_num_batched_tokens, dtype=torch.int64, device=device
            )

        # Context parallelism (see deepseek_v4/common/cp.py).
        parallel_config = vllm_config.parallel_config
        self.pcp_size = parallel_config.prefill_context_parallel_size
        self.dcp_size = parallel_config.decode_context_parallel_size
        self.use_cp = self.pcp_size > 1
        if self.use_cp and self.compress_ratio > 1:
            self.dcp_rank = get_dcp_group().rank_in_group
            self.cp_interleave = parallel_config.cp_kv_cache_interleave_size
            # Zeros: every local token counts as written when computing the
            # owner-agnostic compressed slots.
            self.cp_any_slot = torch.zeros(
                max_num_batched_tokens, dtype=torch.int64, device=device
            )
            self.cp_slot_mapping_buffer = torch.empty(
                self.pcp_size * max_num_batched_tokens,
                dtype=torch.int64,
                device=device,
            )
            self.cp_valid_token_buffer = torch.empty(
                max_num_batched_tokens, dtype=torch.int64, device=device
            )

        # Pre-allocate C128A topk buffers for CUDA graph address stability.
        if self.compress_ratio == 128:
            c128a_max_compressed = cdiv(
                self.model_config.max_model_len, self.compress_ratio
            )
            c128a_max_compressed = (
                cdiv(c128a_max_compressed, _C128A_TOPK_ALIGNMENT)
                * _C128A_TOPK_ALIGNMENT
            )
            # Stored so _build_c128a_metadata passes it as the kernel's
            # max_compressed_tokens, matching the buffer stride. Otherwise the
            # kernel's default 8192 iterates past row width and spills writes
            # into adjacent rows (present in both decode and prefill branches of
            # _build_c128a_topk_metadata_kernel).
            self.c128a_max_compressed = c128a_max_compressed
            self.c128a_global_decode_buffer = torch.empty(
                (max_num_batched_tokens, c128a_max_compressed),
                dtype=torch.int32,
                device=device,
            )
            self.c128a_decode_lens_buffer = torch.empty(
                max_num_batched_tokens, dtype=torch.int32, device=device
            )
            self.c128a_prefill_buffer = torch.empty(
                (max_num_batched_tokens, c128a_max_compressed),
                dtype=torch.int32,
                device=device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4FlashMLAMetadata:
        cm = common_attn_metadata
        req_id_per_token = cm.token_to_req_indices(self.req_id_per_token_buffer)

        slot_mapping = cm.slot_mapping
        dcp_prefill_plan = None
        if self.compress_ratio > 1 and self.use_cp:
            slot_mapping, dcp_prefill_plan = self._build_cp(cm)
        elif self.compress_ratio > 1:
            slot_mapping = get_compressed_slot_mapping(
                cm.num_actual_tokens,
                cm.slot_mapping,
                cm.query_start_loc,
                cm.seq_lens,
                cm.block_table_tensor.clamp_(min=0),
                int(self.kv_cache_spec.num_states),
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )

        c128a_fields: dict[str, torch.Tensor | None] = {}
        if self.compress_ratio == 128:
            c128a_fields = self._build_c128a_metadata(cm, req_id_per_token)

        return DeepseekV4FlashMLAMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            c128a_global_decode_topk_indices=c128a_fields.get(
                "c128a_global_decode_topk_indices"
            ),
            c128a_decode_topk_lens=c128a_fields.get("c128a_decode_topk_lens"),
            c128a_prefill_topk_indices=c128a_fields.get("c128a_prefill_topk_indices"),
            dcp_prefill_plan=dcp_prefill_plan,
        )

    def _build_cp(
        self, cm: CommonAttentionMetadata
    ) -> tuple[torch.Tensor, DCPPrefillGatherPlan | None]:
        """Compressed slot mapping of the cache tokens, and the prefill plan.

        The cache tokens are this rank's tokens in a decode-only step and the
        PCP-gathered tokens otherwise (see deepseek_v4/common/cp.py). Slots are
        computed owner-agnostically from each rank's own tokens (the block
        table is shared by the DCP ranks), then kept where this rank owns them.
        """
        assert cm.positions is not None
        num_local = cm.slot_mapping.shape[0] // self.pcp_size
        positions = cm.positions[:num_local]
        self.compressed_slot_mapping_buffer.fill_(-1)
        get_compressed_slot_mapping(
            real_token_count(cm),
            self.cp_any_slot[:num_local],
            cm.query_start_loc,
            cm.seq_lens,
            cm.block_table_tensor.clamp_(min=0),
            int(self.kv_cache_spec.num_states),
            self.compress_ratio,
            out=self.compressed_slot_mapping_buffer,
            dcp_world_size=self.dcp_size,
            cp_interleave=self.cp_interleave,
        )
        local = self.compressed_slot_mapping_buffer[:num_local]
        num_decodes, num_prefills, _, _ = split_decodes_and_prefills(
            cm,
            decode_threshold=self.reorder_batch_threshold or 1,
            treat_short_extends_as_decodes=False,
        )
        if num_prefills == 0:
            out = self.cp_slot_mapping_buffer[:num_local]
            owned = owner_mask(
                positions, self.dcp_rank, self.dcp_size, self.cp_interleave
            )
            out.copy_(local).masked_fill_(~owned, PAD_SLOT_ID)
            return out, None
        gathered = get_pcp_group().all_gather(local, dim=0)
        out = self.cp_slot_mapping_buffer[: gathered.shape[0]]
        # The runner's gathered slots are PAD where this rank does not own the
        # token or the row is a replicated copy written from another rank.
        out.copy_(gathered).masked_fill_(cm.slot_mapping == PAD_SLOT_ID, PAD_SLOT_ID)
        plan = None
        if self.dcp_size > 1:
            plan = build_dcp_prefill_gather_plan(
                cm.req_idx,
                cm.dcp_local_seq_lens_cpu_upper_bound,
                cm.seq_lens_cpu_upper_bound,
                num_decodes,
                cm.num_reqs,
                self.compress_ratio,
                self.dcp_size,
                self.cp_interleave,
                self.device,
            )
        return out, plan

    def _build_c128a_metadata(
        self,
        cm: CommonAttentionMetadata,
        req_id_per_token: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Pre-compute C128A topk indices for DeepseekV4 (compress_ratio >= 128)."""
        # Must match SWA's decode split (no `require_uniform=True`) so
        # `c128a_global_decode_topk_indices.shape[0]` lines up with q in
        # `_forward_decode`. The per-token C128A kernel handles non-uniform
        # query lengths.
        (num_decodes, _, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                cm,
                decode_threshold=self.reorder_batch_threshold or 1,
                treat_short_extends_as_decodes=not self.use_cp,
            )
        )

        num_total = num_decode_tokens + num_prefill_tokens
        if num_total == 0:
            return {}

        assert cm.positions is not None, (
            "positions is required for C128A metadata build"
        )
        active_topk_width = min(
            max(
                triton.next_power_of_2(max(cm.max_seq_len // self.compress_ratio, 1)),
                _C128A_TOPK_ALIGNMENT,
            ),
            self.c128a_max_compressed,
        )
        assert active_topk_width >= cm.max_seq_len // self.compress_ratio
        assert active_topk_width % _C128A_TOPK_ALIGNMENT == 0
        block_size = self.kv_cache_spec.block_size // self.compress_ratio
        token_slots = cm.slot_mapping
        dcp_kwargs: dict[str, int] = {}
        if self.use_cp:
            # The runner's slots are PAD for tokens this rank does not own;
            # decode validity is about padding only.
            num_local = cm.slot_mapping.shape[0] // self.pcp_size
            token_slots = self.cp_valid_token_buffer[:num_local]
            token_slots.fill_(-1)
            token_slots[: real_token_count(cm)] = 0
            if self.dcp_size > 1:
                dcp_kwargs = dict(
                    dcp_rank=self.dcp_rank,
                    dcp_world_size=self.dcp_size,
                    cp_interleave=self.cp_interleave,
                )
        global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
            cm.positions[:num_total],
            self.compress_ratio,
            num_decode_tokens,
            req_id_per_token,
            cm.block_table_tensor[:num_decodes],
            block_size,
            token_slots,
            self.c128a_global_decode_buffer,
            self.c128a_decode_lens_buffer,
            self.c128a_prefill_buffer,
            max_compressed_tokens=active_topk_width,
            **dcp_kwargs,
        )

        result: dict[str, torch.Tensor | None] = {}
        if num_decode_tokens > 0:
            result["c128a_global_decode_topk_indices"] = global_decode.view(
                num_decode_tokens, 1, -1
            )
            result["c128a_decode_topk_lens"] = decode_lens
        if num_prefill_tokens > 0:
            result["c128a_prefill_topk_indices"] = prefill_local
        return result


class DeepseekV4FlashMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekV4FlashMLABackend(DeepseekV4SparseMLABackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4FlashMLAMetadataBuilder]:
        return DeepseekV4FlashMLAMetadataBuilder


def build_c128a_topk_metadata(
    positions: torch.Tensor,
    compress_ratio: int,
    num_decode_tokens: int,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    global_decode_buffer: torch.Tensor,
    decode_lens_buffer: torch.Tensor,
    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_interleave: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single kernel for all C128A tokens (decode + prefill).

    Under DCP, decode tokens list this rank's compressed entries (local
    indices through its block table); prefill tokens keep global indices.

    Decode tokens: position → block_table lookup → global slot ids + topk_lens.
    Prefill tokens: position → local indices [0, ..., n-1, -1, ...].

    Writes into pre-allocated buffers for CUDA graph address stability.
    Returns views of the buffers.
    """
    num_tokens = positions.shape[0]
    num_prefill_tokens = num_tokens - num_decode_tokens
    assert max_compressed_tokens % _C128A_TOPK_ALIGNMENT == 0
    assert (
        0
        < max_compressed_tokens
        <= min(global_decode_buffer.shape[1], prefill_buffer.shape[1])
    )
    assert global_decode_buffer.stride(-1) == prefill_buffer.stride(-1) == 1

    # TODO: support adaptive-width decode on SM120 (needs the FlashInfer
    # SM120 kernel to accept a real row stride for eidx).
    capability = current_platform.get_device_capability()
    if capability is not None and capability.major == 12:
        global_decode = global_decode_buffer[:num_decode_tokens]
    else:
        global_decode = global_decode_buffer[:num_decode_tokens, :max_compressed_tokens]
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer[:num_prefill_tokens, :max_compressed_tokens]
    assert global_decode.stride(0) == global_decode_buffer.stride(0)
    assert prefill_local.stride(0) == prefill_buffer.stride(0)

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local

    _BUILD_C128A_TOPK_METADATA_KERNEL(
        global_decode_buffer,
        decode_lens_buffer,
        prefill_buffer,
        positions,
        compress_ratio,
        max_compressed_tokens,
        num_decode_tokens,
        token_to_req_indices,
        block_table,
        block_size,
        slot_mapping,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
    )
    return global_decode, decode_lens, prefill_local


class BuildC128ATopkMetadataKernel(
    VllmTritonJitKernel["BuildC128ATopkMetadataKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        compress_ratio: int
        max_compressed_tokens: int
        block_size: int
        triton_block_size: int

    @staticmethod
    @triton.jit(
        do_not_specialize=[
            "global_decode_stride",
            "prefill_local_stride",
            "num_decode_tokens",
            "block_table_stride",
        ]
    )
    def kernel(
        # Decode outputs
        global_decode_ptr,
        global_decode_stride,
        decode_lens_ptr,
        # Prefill output
        prefill_local_ptr,
        prefill_local_stride,
        # Inputs
        positions_ptr,
        compress_ratio,
        max_compressed_tokens,
        num_decode_tokens,
        token_to_req_indices_ptr,
        block_table_ptr,
        block_table_stride,
        block_size,
        slot_mapping_ptr,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        position = tl.load(positions_ptr + token_idx)
        num_compressed = (position + 1) // compress_ratio
        num_compressed = tl.minimum(num_compressed, max_compressed_tokens)
        is_decode = token_idx < num_decode_tokens

        if is_decode:
            if dcp_world_size > 1:
                # This rank's whole groups among tokens [0, position].
                span = cp_interleave * dcp_world_size
                rem = (position + 1) % span - dcp_rank * cp_interleave
                rem = tl.minimum(tl.maximum(rem, 0), cp_interleave)
                local_tokens = (position + 1) // span * cp_interleave + rem
                num_compressed = tl.minimum(
                    local_tokens // compress_ratio, max_compressed_tokens
                )
            # --- Decode: block-table lookup → global slot ids + count ---
            is_valid_token = tl.load(slot_mapping_ptr + token_idx) >= 0
            req_idx = tl.load(token_to_req_indices_ptr + token_idx)
            count = tl.zeros((), dtype=tl.int32)
            for i in range(0, max_compressed_tokens, BLOCK_SIZE):
                offset = i + tl.arange(0, BLOCK_SIZE)
                mask = offset < max_compressed_tokens
                is_valid = offset < num_compressed

                block_indices = offset // block_size
                block_numbers = tl.load(
                    block_table_ptr + req_idx * block_table_stride + block_indices,
                    mask=mask & is_valid,
                )
                block_offsets = offset % block_size
                slot_ids = block_numbers * block_size + block_offsets
                slot_ids = tl.where(is_valid, slot_ids, -1)
                tl.store(
                    global_decode_ptr + token_idx * global_decode_stride + offset,
                    slot_ids,
                    mask=mask,
                )
                count += tl.sum(is_valid.to(tl.int32), axis=0)

            tl.store(
                decode_lens_ptr + token_idx,
                tl.where(is_valid_token, count, 0),
            )
        else:
            # --- Prefill: write local indices ---
            pfx_idx = token_idx - num_decode_tokens
            for i in range(0, max_compressed_tokens, BLOCK_SIZE):
                offset = i + tl.arange(0, BLOCK_SIZE)
                mask = offset < max_compressed_tokens
                tl.store(
                    prefill_local_ptr + pfx_idx * prefill_local_stride + offset,
                    tl.where(offset < num_compressed, offset, -1),
                    mask=mask,
                )

    def dispatch(  # type: ignore[override]
        self,
        **compile_key_fields: int,
    ) -> CompileKey:
        return self.CompileKey(**compile_key_fields)

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        if getattr(hf_config, "model_type", None) != "deepseek_v4":
            return []

        compress_ratio = 128
        max_model_len = vllm_config.model_config.max_model_len
        if max_model_len <= 0:
            return []
        max_compressed_tokens = cdiv(max_model_len, compress_ratio)
        max_compressed_tokens = (
            cdiv(max_compressed_tokens, _C128A_TOPK_ALIGNMENT) * _C128A_TOPK_ALIGNMENT
        )
        return self._trace_dispatch(self.dispatch)(
            compress_ratio=compress_ratio,
            max_compressed_tokens=max_compressed_tokens,
            # DeepSeek V4 sparse MLA uses 256-token KV pages; C128A metadata
            # works in compressed-token units.
            block_size=256 // compress_ratio,
            triton_block_size=1024,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        return dict(
            global_decode_buffer=TritonWarmupTensor(
                torch.int32, shape=(1, 1), strides=(1, 1)
            ),
            decode_lens_buffer=int32_ptr,
            prefill_buffer=TritonWarmupTensor(
                torch.int32, shape=(1, 1), strides=(1, 1)
            ),
            positions=TritonWarmupTensor(torch.int64),
            compress_ratio=compile_key.compress_ratio,
            max_compressed_tokens=compile_key.max_compressed_tokens,
            num_decode_tokens=0,
            token_to_req_indices=int32_ptr,
            block_table=TritonWarmupTensor(torch.int32, shape=(1, 1), strides=(1, 1)),
            block_size=compile_key.block_size,
            slot_mapping=TritonWarmupTensor(torch.int64),
            dcp_rank=0,
            dcp_world_size=1,
            cp_interleave=1,
        )

    @kernel_launcher
    def __call__(
        self,
        global_decode_buffer: torch.Tensor,
        decode_lens_buffer: torch.Tensor,
        prefill_buffer: torch.Tensor,
        positions: torch.Tensor,
        compress_ratio: int,
        max_compressed_tokens: int,
        num_decode_tokens: int,
        token_to_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        slot_mapping: torch.Tensor,
        dcp_rank: int = 0,
        dcp_world_size: int = 1,
        cp_interleave: int = 1,
    ) -> LaunchSpec:
        return (positions.shape[0],), dict(
            global_decode_ptr=global_decode_buffer,
            global_decode_stride=global_decode_buffer.stride(0),
            decode_lens_ptr=decode_lens_buffer,
            prefill_local_ptr=prefill_buffer,
            prefill_local_stride=prefill_buffer.stride(0),
            block_table_stride=block_table.stride(0),
            BLOCK_SIZE=1024,
        )


_BUILD_C128A_TOPK_METADATA_KERNEL = BuildC128ATopkMetadataKernel()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.config.kernel import (
    FLASHINFER_MOE_EP_BACKENDS,
    FLASHINFER_MOE_EP_CAKE,
    FLASHINFER_MOE_EP_CUTEDSL,
    FLASHINFER_MOE_EP_DEEP_GEMM,
)
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import _import_deep_gemm

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

FlashInferMoeEpKernel = Literal["cutedsl", "deep_gemm", "cake"]
FlashInferMoeEpWeightFormat = Literal["nvfp4", "mxfp4"]
_WEIGHT_FORMATS: dict[QuantKey | None, FlashInferMoeEpWeightFormat] = {
    kNvfp4Static: "nvfp4",
    kMxfp4Static: "mxfp4",
}

_SM100_CAPABILITIES = frozenset({(10, 0), (10, 3)})
_SM120_CAPABILITIES = frozenset({(12, 0), (12, 1)})


def _current_capability() -> tuple[int, int] | None:
    capability = current_platform.get_device_capability()
    if capability is None:
        return None
    return (capability.major, capability.minor)


def supports_current_device() -> bool:
    capability = _current_capability()
    return any(capability in spec.capabilities for spec in _BACKEND_SPECS.values())


_E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class FlashInferMoeEpBackendSpec:
    kernel: FlashInferMoeEpKernel
    weight_formats: frozenset[FlashInferMoeEpWeightFormat]
    capabilities: frozenset[tuple[int, int]] = _SM100_CAPABILITIES


@dataclass(frozen=True)
class FlashInferMoeEpWeights:
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class FlashInferMoeEpEpilogue:
    input_norm_const: float = 1.0
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None
    fc1_norm_const: torch.Tensor | None = None


_BACKEND_SPECS = {
    FLASHINFER_MOE_EP_CUTEDSL: FlashInferMoeEpBackendSpec(
        kernel="cutedsl",
        weight_formats=frozenset({"nvfp4", "mxfp4"}),
    ),
    FLASHINFER_MOE_EP_DEEP_GEMM: FlashInferMoeEpBackendSpec(
        kernel="deep_gemm",
        weight_formats=frozenset({"mxfp4"}),
    ),
    FLASHINFER_MOE_EP_CAKE: FlashInferMoeEpBackendSpec(
        kernel="cake",
        weight_formats=frozenset({"mxfp4"}),
        capabilities=_SM120_CAPABILITIES,
    ),
}

assert frozenset(_BACKEND_SPECS) == FLASHINFER_MOE_EP_BACKENDS


def is_flashinfer_moe_ep_backend(moe_backend: str) -> bool:
    return moe_backend in _BACKEND_SPECS


def flashinfer_moe_ep_backend_spec(
    moe_backend: str,
) -> FlashInferMoeEpBackendSpec:
    try:
        return _BACKEND_SPECS[moe_backend]
    except KeyError:
        raise ValueError(
            f"{moe_backend!r} is not a FlashInfer MoE-EP backend; expected "
            f"one of {sorted(_BACKEND_SPECS)}"
        ) from None


def flashinfer_moe_ep_unsupported_reasons(
    moe: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[str, ...]:
    """Deployment constraints of the megakernel beyond the framework's checks.

    ``activation_key`` is ``None`` for MXFP4 checkpoints (the kernel quantizes
    bf16 activations itself) but marks a W4A16 recipe for NVFP4 weights.
    """
    spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)
    unsupported: list[str] = []
    capability = _current_capability()
    if capability not in spec.capabilities:
        unsupported.append(f"compute capability {capability}")
    weight_format = _WEIGHT_FORMATS.get(weight_key)
    if weight_format is None:
        unsupported.append(f"weight scheme {weight_key}")
    elif weight_format not in spec.weight_formats:
        unsupported.append(f"{weight_format.upper()} weights")
    if weight_key == kNvfp4Static and activation_key is None:
        unsupported.append("A16 activations with NVFP4 weights")
    if moe.skip_final_all_reduce:
        unsupported.append("skip_final_all_reduce")
    if moe.in_dtype != torch.bfloat16:
        unsupported.append(f"activation dtype {moe.in_dtype}")
    if moe.has_bias:
        unsupported.append("expert bias")
    if moe.swiglu_alpha is not None or moe.swiglu_beta is not None:
        unsupported.append("custom SwiGLU alpha or beta")
    if (
        spec.kernel == "deep_gemm"
        and moe.routing_method is not RoutingMethodType.DeepseekV4
    ):
        unsupported.append(f"routing method {moe.routing_method.name}")

    vllm_config = get_current_vllm_config()
    if vllm_config.weight_transfer_config is not None:
        unsupported.append("runtime weight transfer")
    if vllm_config.parallel_config.enable_dbo:
        unsupported.append("dual batch overlap")
    if spec.kernel == "cake":
        if moe.num_experts % moe.moe_parallel_config.ep_size:
            unsupported.append("experts not divisible by the EP size")
        limit = moe.swiglu_limit
        if limit is not None and (limit < 0 or limit != int(limit)):
            unsupported.append(f"non-integral swiglu_limit {limit}")
    if spec.kernel != "cutedsl" and vllm_config.parallel_config.enable_eplb:
        unsupported.append("EPLB")
    return tuple(unsupported)


def validate_flashinfer_moe_ep_layer(layer: RoutedExperts) -> None:
    if layer.expert_map_manager.placement_strategy != "linear":
        raise ValueError(
            f"{layer.moe_config.moe_backend} requires contiguous linear "
            "expert placement"
        )


def apply_topk_in_fc1(
    moe: FusedMoEConfig,
    *,
    apply_router_weight_on_input: bool = False,
) -> bool:
    return (
        apply_router_weight_on_input
        or moe.routing_method is RoutingMethodType.DeepseekV4
    )


def _dequant_mxfp4_ue8m0_gran32(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=raw.device)
    values = torch.empty(
        raw.shape[0],
        raw.shape[1] * 2,
        dtype=torch.float32,
        device=raw.device,
    )
    values[:, ::2] = lut[(raw & 0x0F).to(torch.int64)]
    values[:, 1::2] = lut[(raw >> 4).to(torch.int64)]
    encoded_scale = (
        scale
        if scale.dtype == torch.float8_e8m0fnu
        else scale.view(torch.float8_e8m0fnu)
    )
    decoded_scale = encoded_scale.float()
    return (values * decoded_scale.repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def _dequant_mxfp4_experts(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    num_experts, output_size, packed_input_size = weight.shape
    result = torch.empty(
        num_experts,
        output_size,
        packed_input_size * 2,
        dtype=torch.bfloat16,
        device=weight.device,
    )
    for expert_id in range(num_experts):
        result[expert_id] = _dequant_mxfp4_ue8m0_gran32(
            weight[expert_id], scale[expert_id]
        )
    return result


def mxfp4_moe_ep_weights(
    moe_backend: str,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> FlashInferMoeEpWeights:
    spec = flashinfer_moe_ep_backend_spec(moe_backend)
    if spec.kernel in ("deep_gemm", "cake"):
        return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale)
    return FlashInferMoeEpWeights(
        w13=_dequant_mxfp4_experts(w13, w13_scale),
        w2=_dequant_mxfp4_experts(w2, w2_scale),
    )


class _FlashInferMoeEpApi:
    def __init__(self) -> None:
        try:
            from flashinfer.moe_ep import (
                BootstrapConfig,
                DeepGemmMegaMoeConfig,
                FleetParams,
                MegaConfig,
                MoEEpMegaLayer,
                MoEEpTensors,
                MoEWeightPack,
                Nvfp4CutedslMegaMoeConfig,
            )
        except ImportError as err:
            raise RuntimeError(
                "FlashInfer MoE-EP requires the public API shipped with "
                "FlashInfer 0.7.0 or newer"
            ) from err

        self.BootstrapConfig = BootstrapConfig
        self.DeepGemmMegaMoeConfig = DeepGemmMegaMoeConfig
        self.FleetParams = FleetParams
        self.MegaConfig = MegaConfig
        self.MoEEpMegaLayer = MoEEpMegaLayer
        self.MoEEpTensors = MoEEpTensors
        self.MoEWeightPack = MoEWeightPack
        self.Nvfp4CutedslMegaMoeConfig = Nvfp4CutedslMegaMoeConfig
        try:
            from flashinfer.moe_ep import Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig
        except ImportError:
            Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig = None
        self.CakeMegaMoeConfig = Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig


def _load_flashinfer_moe_ep_api() -> _FlashInferMoeEpApi:
    return _FlashInferMoeEpApi()


def _expose_deep_gemm_to_flashinfer() -> None:
    deep_gemm = _import_deep_gemm()
    if deep_gemm is None:
        raise RuntimeError(
            "FlashInfer MoE-EP DeepGEMM requires a usable DeepGEMM installation"
        )
    sys.modules.setdefault("deep_gemm", deep_gemm)


class FlashInferMoeEp:
    def __init__(
        self,
        moe: FusedMoEConfig,
        weights: FlashInferMoeEpWeights,
        epilogue: FlashInferMoeEpEpilogue | None = None,
        *,
        apply_topk_in_fc1: bool,
    ) -> None:
        spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)
        if spec.kernel == "deep_gemm":
            _expose_deep_gemm_to_flashinfer()
        api = _load_flashinfer_moe_ep_api()
        if epilogue is None:
            epilogue = FlashInferMoeEpEpilogue()
        ep_group = get_ep_group()
        bootstrap = api.BootstrapConfig(
            world_size=ep_group.world_size,
            rank=ep_group.rank_in_group,
            process_group=ep_group.device_group,
            device=torch.accelerator.current_device_index(),
        )
        fleet_params = api.FleetParams(
            num_experts=moe.num_experts,
            max_tokens_per_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
        )
        weight_pack = api.MoEWeightPack(
            w13=weights.w13,
            w2=weights.w2,
            w13_scale=weights.w13_scale,
            w2_scale=weights.w2_scale,
        )
        if spec.kernel == "cake":
            if api.CakeMegaMoeConfig is None:
                raise RuntimeError(
                    f"{moe.moe_backend} requires a FlashInfer build exporting "
                    "Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig "
                    "(flashinfer-ai/flashinfer#6346)"
                )
            # The Cake kernel reads UE8M0 exponent bytes.
            weight_pack = api.MoEWeightPack(
                w13=weights.w13.view(torch.uint8),
                w2=weights.w2.view(torch.uint8),
                w13_scale=_ue8m0_bytes(weights.w13_scale),
                w2_scale=_ue8m0_bytes(weights.w2_scale),
            )
            megakernel = api.CakeMegaMoeConfig(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                swiglu_limit=int(moe.swiglu_limit or 0),
                transport_table=envs.VLLM_FLASHINFER_MOE_EP_CAKE_TRANSPORT_TABLE,
            )
        elif spec.kernel == "cutedsl":
            # In-kernel fc2 reduce keeps its default (off); FlashInfer 0.7.1rc1
            # renamed in_kernel_fc2_reduce to enable_in_kernel_fc2_reduce.
            megakernel = api.Nvfp4CutedslMegaMoeConfig(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                gate_up_clamp=moe.swiglu_limit,
                fast_math=True,
                apply_topk_in_fc1=apply_topk_in_fc1,
                combine_dtype="bf16",
                input_norm_const=epilogue.input_norm_const,
                fc1_alpha=None,
                fc2_alpha=None,
                fc1_norm_const=None,
                knobs=None,
            )
        else:
            megakernel = api.DeepGemmMegaMoeConfig(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                activation_clamp=moe.swiglu_limit,
                fast_math=True,
            )
        backend = api.MegaConfig(
            megakernel=megakernel,
            # Cake gets its MXFP8 activations from _cake_stage_inputs (one
            # kernel instead of FlashInfer's torch-op quantization).
            quantize_input=spec.kernel != "cake",
            preprocess_weights=True,
        )

        self._moe_ep_tensors_cls = api.MoEEpTensors
        self._mega_layer: Any | None = api.MoEEpMegaLayer(
            bootstrap,
            fleet_params,
            weight_pack,
            backend,
        )
        self._device = weights.w13.device
        self._hidden_size = moe.hidden_dim
        self._max_num_tokens = moe.max_num_tokens
        self._num_experts = moe.num_experts
        self._top_k = moe.experts_per_token
        self._epilogue: FlashInferMoeEpEpilogue | None = epilogue
        self._return_workspace_view = moe.routing_method is RoutingMethodType.DeepseekV4
        # Cake stages each call with a token bound shared by every EP rank, so
        # a call needs no routing exchange and can be captured in a graph.
        self._bucketed = spec.kernel == "cake"

    @property
    def can_overlap_shared_experts(self) -> bool:
        return False

    @property
    def output_is_reduced(self) -> bool:
        return True

    @property
    def topk_indices_dtype(self) -> torch.dtype:
        return torch.int32

    @property
    def is_monolithic(self) -> bool:
        return False

    def _tensors(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Any:
        epilogue = cast(FlashInferMoeEpEpilogue, self._epilogue)
        return self._moe_ep_tensors_cls(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            fc1_alpha=epilogue.fc1_alpha,
            fc2_alpha=epilogue.fc2_alpha,
            fc1_norm_const=epilogue.fc1_norm_const,
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE-EP layer was destroyed")
        if hidden_states.shape[0] > self._max_num_tokens:
            raise ValueError(
                f"FlashInfer MoE-EP got {hidden_states.shape[0]} tokens, but "
                f"the workspace supports at most {self._max_num_tokens}"
            )
        if self._bucketed:
            # Padded rows (CUDA graph / PCP padding) carry stale activations;
            # routing them can overflow the kernel's lossless BF16 return
            # codec, which sets protocol_error and desyncs the peers' signal
            # bookkeeping (a hang). -1 is the kernel's masked route.
            codes, scales, topk_ids = _cake_stage_inputs(
                hidden_states, topk_ids, _padding_mask(hidden_states.shape[0])
            )
            tensors = self._tensors(codes, topk_ids, topk_weights)
            tensors.scales = scales
            self._mega_layer.stage_inputs(
                tensors,
                compile_tokens_per_rank=_ep_token_bound(hidden_states.shape[0]),
            )
            return self._mega_layer.compute_staged(output=None)
        tensors = self._tensors(hidden_states, topk_ids, topk_weights)
        if self._return_workspace_view and getattr(
            self._mega_layer, "supports_output_view", False
        ):
            return self._mega_layer.forward(tensors, return_workspace_view=True)
        return self._mega_layer(tensors)

    @torch.inference_mode()
    def warmup(self) -> None:
        if self._mega_layer is None:
            return
        hidden_states = torch.zeros(
            1,
            self._hidden_size,
            dtype=torch.bfloat16,
            device=self._device,
        )
        topk_ids = (
            torch.arange(
                self._top_k,
                dtype=torch.int64 if self._bucketed else torch.int32,
                device=self._device,
            )
            .mul_(self._num_experts // self._top_k)
            .view(1, self._top_k)
        )
        topk_weights = torch.full(
            (1, self._top_k),
            1.0 / self._top_k,
            dtype=torch.float32,
            device=self._device,
        )
        if self._bucketed:
            codes, scales, topk_ids = _cake_stage_inputs(hidden_states, topk_ids, None)
            tensors = self._tensors(codes, topk_ids, topk_weights)
            tensors.scales = scales
        else:
            tensors = self._tensors(hidden_states, topk_ids, topk_weights)
        self._mega_layer.warmup(tensors)

    def kernel_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kernel-resident fc1 weight, fc1 scales, fc2 weight, fc2 scales.

        Contiguous per-expert byte tensors that alias the memory the megakernel
        reads, so a caller may permute experts in place (EPLB).
        """
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE-EP layer was destroyed")
        transformed = self._mega_layer._transformed
        if self._bucketed:
            return (
                transformed.w1_fp4.view(torch.uint8),
                transformed.w1_sfb,
                transformed.w2_fp4.view(torch.uint8),
                transformed.w2_sfb,
            )
        (fc1_weight, fc1_sf), (fc2_weight, fc2_sf) = transformed
        return (
            _expert_major_bytes(fc1_weight),
            _expert_major_bytes(fc1_sf),
            _expert_major_bytes(fc2_weight),
            _expert_major_bytes(fc2_sf),
        )

    def expert_weight_views(self) -> tuple[torch.Tensor, ...]:
        """Every per-expert tensor the kernel reads: weights plus epilogue."""
        epilogue = cast(FlashInferMoeEpEpilogue, self._epilogue)
        vectors = (epilogue.fc1_alpha, epilogue.fc2_alpha, epilogue.fc1_norm_const)
        return (*self.kernel_weights(), *(v for v in vectors if v is not None))

    def destroy(self) -> None:
        mega_layer = self._mega_layer
        if mega_layer is None:
            return
        mega_layer.destroy()
        self._mega_layer = None
        self._epilogue = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()


def _ue8m0_bytes(scale: torch.Tensor | None) -> torch.Tensor:
    """UE8M0 exponent bytes of MX block scales."""
    assert scale is not None
    if scale.dtype == torch.uint8:
        return scale
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.view(torch.uint8)
    exponents = scale.to(torch.float8_e8m0fnu)
    if not torch.equal(exponents.to(scale.dtype), scale):
        raise ValueError("MX block scales must be powers of two (UE8M0)")
    return exponents.view(torch.uint8)


@triton.jit
def _cake_stage_kernel(
    x_ptr,
    x_stride,
    codes_ptr,
    scales_ptr,
    ids_ptr,
    ids_stride,
    ids_out_ptr,
    padding_ptr,
    NUM_GROUPS: tl.constexpr,
    TOP_K: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    groups = tl.program_id(1) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    cols = tl.arange(0, 32)
    in_row = groups < NUM_GROUPS
    offs = groups[:, None] * 32 + cols[None, :]
    x = tl.load(x_ptr + row * x_stride + offs, mask=in_row[:, None], other=0.0).to(
        tl.float32
    )
    # UE8M0 scale: the power of two >= amax / 448 (FlashInfer's
    # quantize_activations_k32, which the kernel also uses in-kernel).
    amax = tl.max(tl.abs(x), axis=1)
    bits = tl.math.div_rn(amax, 448.0).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)
    exponent = tl.minimum(tl.maximum(exponent, 0), 254)
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    codes = (x * inverse[:, None]).to(tl.float8e4nv)
    tl.store(codes_ptr + row * NUM_GROUPS * 32 + offs, codes, mask=in_row[:, None])
    tl.store(
        scales_ptr + row * NUM_GROUPS + groups, exponent.to(tl.uint8), mask=in_row
    )
    if tl.program_id(1) == 0:
        k = tl.arange(0, PADDED_TOP_K)
        ids = tl.load(ids_ptr + row * ids_stride + k, mask=k < TOP_K).to(tl.int64)
        if HAS_PADDING:
            ids = tl.where(tl.load(padding_ptr + row) != 0, -1, ids)
        tl.store(ids_out_ptr + row * TOP_K + k, ids, mask=k < TOP_K)


def _cake_stage_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    is_padding: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP8 codes + UE8M0 scales of the activations and int64 routing ids
    (-1 on padding rows) for the Cake MegaMoE, in one kernel."""
    rows, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    codes = torch.empty(
        (rows, hidden), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
    scales = torch.empty(
        (rows, hidden // 32), dtype=torch.uint8, device=hidden_states.device
    )
    ids = torch.empty((rows, top_k), dtype=torch.int64, device=topk_ids.device)
    if rows == 0:
        return codes, scales, ids
    assert hidden_states.stride(1) == 1 and topk_ids.stride(1) == 1
    block_groups = 32
    _cake_stage_kernel[(rows, triton.cdiv(hidden // 32, block_groups))](
        hidden_states,
        hidden_states.stride(0),
        codes,
        scales,
        topk_ids,
        topk_ids.stride(0),
        ids,
        is_padding if is_padding is not None else ids,
        NUM_GROUPS=hidden // 32,
        TOP_K=top_k,
        PADDED_TOP_K=triton.next_power_of_2(top_k),
        HAS_PADDING=is_padding is not None,
        BLOCK_GROUPS=block_groups,
    )
    return codes, scales, ids


def _padding_mask(num_tokens: int) -> torch.Tensor | None:
    """The forward context's padding rows over this call's tokens, if any."""
    if not envs.VLLM_MOE_SKIP_PADDING or not is_forward_context_available():
        return None
    is_padding = get_forward_context().is_padding
    if is_padding is None or is_padding.shape[0] < num_tokens:
        return None
    return is_padding[:num_tokens]


def _ep_token_bound(num_tokens: int) -> int:
    """A per-call token bound equal on every EP rank.

    DP ranks pad to a common count only when they share a CUDA graph size, so
    take the largest; without DP the EP ranks are TP/PCP ranks of one batch,
    and PCP pads every rank's slice to the same length.
    """
    if is_forward_context_available():
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is not None:
            return max(num_tokens, int(dp_metadata.num_tokens_across_dp_cpu.max()))
    return num_tokens


def _expert_major_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Contiguous storage of a per-expert kernel tensor, 1-byte dtypes as uint8.

    The CuTeDSL kernel keeps (E, K, N) transpose views over (E, N, K) storage.
    """
    if not tensor.is_contiguous():
        tensor = tensor.transpose(1, 2)
    if not tensor.is_contiguous():
        raise ValueError(f"expert tensor is not expert-major: {tensor.stride()}")
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor

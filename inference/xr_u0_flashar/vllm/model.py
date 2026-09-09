"""UNISFlashARForCausalLM vLLM model class.

The class extends the Xiaomi-Robotics-U0 backbone with FlashAR vertical attention layers,
horizontal/vertical visual-token heads, and h/v fusion gates. vLLM still owns
the backbone forward and PagedAttention KV cache; the FlashAR runtime supplies
per-step visual-frame metadata so the vertical block has its own KV group.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import functools
from typing import Iterable, Optional, Union

import torch
from torch import nn

from vllm.attention import Attention, AttentionType
from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
from vllm.attention.layer import Attention as AttentionBase
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.unis import (
    UNISForCausalLM, UNISModel, UNISMLP,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata, subclass_attention_backend,
)
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec


# ---------------------------------------------------------------- vertical bits
# VerticalFullAttentionSpec inherits FullAttentionSpec so isinstance checks
# inside FullAttentionManager (used for KV slot read/write semantics — vertical
# uses identical paged KV layout) pass. To still split layers into a separate
# KVCacheGroup, we add a discriminator field (`_vertical_tag`) so the dataclass-
# generated __eq__/__hash__ differ from base FullAttentionSpec.
#
# We also override is_uniform_type / is_kv_cache_spec_uniform via spec_patch so
# vLLM treats this as a distinct "type" for grouping purposes. With both pieces
# in place, vertical layers route to their own KVCacheGroup but reuse all the
# FullAttentionManager / FullAttentionSpec machinery.
@dataclass(frozen=True)
class VerticalFullAttentionSpec(AttentionSpec):
    """Standalone spec for vertical_block KV. Distinct from FullAttentionSpec
    so it routes to its own KVCacheGroup. We register a paired
    VerticalAttentionManager (subclass of FullAttentionManager) under it so
    cache_hit / block alloc semantics match full attention."""
    sliding_window: Optional[int] = None
    attention_chunk_size: Optional[int] = None

    def max_memory_usage_bytes(self, vllm_config) -> int:
        from vllm.utils import cdiv
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @classmethod
    def merge(cls, specs):
        assert all(isinstance(s, VerticalFullAttentionSpec) for s in specs)
        from copy import deepcopy
        return deepcopy(specs[0])


# ----------------------------------------------- patches into vLLM internals
# Three small monkey-patches make vertical layers route to their own group.
# All three write to vllm.v1 internals; nothing in site-packages is touched.
def install_vertical_spec_patch():
    """Idempotent monkey-patch. Wires vertical layers into their own
    KVCacheGroup by:
      1. swapping FullAttentionSpec → VerticalFullAttentionSpec for vertical
         layers in get_kv_cache_spec();
      2. teaching is_uniform_type / is_kv_cache_spec_uniform to recognise
         VerticalFullAttentionSpec so they correctly return False in mixed
         backbone+vertical models, forcing vLLM down the multi-group path.
    """
    from vllm.v1.worker import gpu_model_runner as _gmr
    if getattr(_gmr.GPUModelRunner.get_kv_cache_spec,
               "_vertical_patched", False):
        return

    # ---- (1) get_kv_cache_spec post-process ----
    orig_spec = _gmr.GPUModelRunner.get_kv_cache_spec

    def patched_spec(self):
        spec_dict = orig_spec(self)
        n_swapped = 0
        for name, module in self.model.named_modules():
            if not isinstance(module, VerticalAttention):
                continue
            attn_layer_name = name
            old = spec_dict.get(attn_layer_name)
            if old is None or isinstance(old, VerticalFullAttentionSpec):
                continue
            assert isinstance(old, FullAttentionSpec), (
                f"unexpected spec type for {attn_layer_name}: {type(old)}")
            spec_dict[attn_layer_name] = VerticalFullAttentionSpec(
                block_size=old.block_size,
                num_kv_heads=old.num_kv_heads,
                head_size=old.head_size,
                dtype=old.dtype,
            )
            n_swapped += 1
        print(f"[vertical_spec_patch] swapped {n_swapped} layers to "
              f"VerticalFullAttentionSpec; total layers = {len(spec_dict)}")
        return spec_dict

    patched_spec._vertical_patched = True
    _gmr.GPUModelRunner.get_kv_cache_spec = patched_spec

    # ---- (2) UniformTypeKVCacheSpecs.is_uniform_type ----
    from vllm.v1 import kv_cache_interface as _kvi
    orig_is_uniform_type = _kvi.UniformTypeKVCacheSpecs.is_uniform_type

    @classmethod
    def patched_is_uniform_type(cls, kv_cache_specs):
        # If ANY layer is vertical and ANY is full-attention, force False so
        # vLLM goes down the page-size-uniform multi-group path.
        types = {type(s) for s in kv_cache_specs.values()}
        if VerticalFullAttentionSpec in types and len(types) > 1:
            return False
        # If all layers are vertical, treat as uniform.
        if types == {VerticalFullAttentionSpec}:
            block_sizes = {s.block_size for s in kv_cache_specs.values()}
            return len(block_sizes) == 1
        return orig_is_uniform_type.__func__(cls, kv_cache_specs)

    _kvi.UniformTypeKVCacheSpecs.is_uniform_type = patched_is_uniform_type

    # ---- (3) is_kv_cache_spec_uniform ----
    from vllm.v1.core import kv_cache_utils as _kvu
    orig_is_uniform = _kvu.is_kv_cache_spec_uniform

    def patched_is_uniform(kv_cache_spec):
        types = {type(s) for s in kv_cache_spec.values()}
        if VerticalFullAttentionSpec in types and len(types) > 1:
            return False
        return orig_is_uniform(kv_cache_spec)

    _kvu.is_kv_cache_spec_uniform = patched_is_uniform

    # ---- (4) spec_manager_map: route VerticalFullAttentionSpec ----
    from vllm.v1.core import single_type_kv_cache_manager as _stmgr
    if VerticalFullAttentionSpec not in _stmgr.spec_manager_map:
        # Subclass FullAttentionManager but bypass the type assertion in
        # find_longest_cache_hit (which only whitelists FullAttentionSpec /
        # ChunkedLocalAttentionSpec). Vertical KV layout is identical so the
        # rest of the manager's logic applies as-is.
        class VerticalAttentionManager(_stmgr.FullAttentionManager):
            @classmethod
            def find_longest_cache_hit(cls, block_hashes, max_length,
                                       kv_cache_group_ids, block_pool,
                                       kv_cache_spec, use_eagle,
                                       dcp_world_size: int = 1):
                assert isinstance(kv_cache_spec, VerticalFullAttentionSpec)
                # Reuse FullAttentionManager body by routing through a
                # FullAttentionSpec view.
                full_view = FullAttentionSpec(
                    block_size=kv_cache_spec.block_size,
                    num_kv_heads=kv_cache_spec.num_kv_heads,
                    head_size=kv_cache_spec.head_size,
                    dtype=kv_cache_spec.dtype,
                )
                return super().find_longest_cache_hit(
                    block_hashes, max_length, kv_cache_group_ids,
                    block_pool, full_view, use_eagle, dcp_world_size)

        _stmgr.spec_manager_map[VerticalFullAttentionSpec] = (
            VerticalAttentionManager)

    # ---- (5) HybridKVCacheCoordinator's "exactly one full attention type"
    # assertion treats VerticalFullAttentionSpec as the "other" group, while
    # the backbone FullAttentionSpec groups stay as the "full attention" set.
    # No patch needed there — the existing logic naturally splits them as long
    # as VerticalFullAttentionSpec is NOT a subclass of FullAttentionSpec.


install_vertical_spec_patch()


# Module-global hook for the wrapper to push per-step vertical metadata.
# VerticalAttentionBuilder reads this in build() since it has no direct path
# to the model instance.
_VERTICAL_META_HOOK: dict = {"per_step": None}


def set_vertical_meta_hook(per_step: Optional[dict]):
    """Wrapper-side: set the per-step vertical metadata before each forward.

    per_step format: dict with keys
      "req_id_to_visual": {req_id: {"visual_so_far": int,
                                     "visual_this_step": int}}
      "phase": optional string for logging (e.g. "prefill", "diag_3")
    OR None to clear (warmup / dummy steps).
    """
    _VERTICAL_META_HOOK["per_step"] = per_step


def get_vertical_intro_log() -> list:
    """Return per-step vertical metadata snapshots captured by the builder."""
    # Trigger import path so the inner closure exists; the log lives on
    # the inner class, not module-level — fish it out through one of
    # the underlying backends already cached by lru_cache.
    if not _VERTICAL_BUILDER_REF["cls"]:
        return []
    return list(_VERTICAL_BUILDER_REF["cls"]._intro_log)


def reset_vertical_intro_log() -> None:
    if _VERTICAL_BUILDER_REF["cls"]:
        _VERTICAL_BUILDER_REF["cls"]._intro_log.clear()


_VERTICAL_BUILDER_REF: dict = {"cls": None}


@functools.lru_cache
def create_vertical_attention_backend(
    underlying_attn_backend: AttentionBackend,
) -> type[AttentionBackend]:
    """Build a vertical-flavored backend.

    Override .build() to rewrite query_start_loc / seq_lens / slot_mapping
    into the visual-only frame, by reading _VERTICAL_META_HOOK populated
    by the wrapper before each engine_core.step().
    """
    prefix = "VerticalAttention_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class VerticalAttentionBuilder(underlying_builder):  # type: ignore
        # Diagnostic log (capped) — useful for verifying the rewrite is correct.
        _intro_log: list = []

        def build(self, common_prefix_len: int,
                  common_attn_metadata: CommonAttentionMetadata,
                  fast_build: bool = False) -> AttentionMetadata:
            per_step = _VERTICAL_META_HOOK.get("per_step")
            # Prefill / warmup / no-visual-tokens-this-step: vertical attention
            # is not invoked by patched_outer either, so the only contract we
            # have to honour with vLLM is "don't crash building metadata".
            # Returning passthrough metadata (with causal=False) is safe.
            if per_step is None or not any(
                    int(v.get("visual_this_step", 0))
                    for v in per_step.get("req_id_to_visual", {}).values()):
                new_md = copy(common_attn_metadata)
                new_md.causal = False
                return super().build(common_prefix_len, new_md, fast_build)

            req_id_to_visual = per_step.get("req_id_to_visual", {})
            batch_req_ids = per_step.get("batch_req_ids", [])
            num_reqs = common_attn_metadata.num_reqs

            # Resolve visual_so_far / visual_this_step per batch position.
            visual_so_far = [0] * num_reqs
            visual_this_step = [0] * num_reqs
            for i in range(num_reqs):
                if i < len(batch_req_ids):
                    info = req_id_to_visual.get(batch_req_ids[i])
                    if info is not None:
                        visual_so_far[i] = int(info["visual_so_far"])
                        visual_this_step[i] = int(info["visual_this_step"])

            # Build visual-frame query_start_loc (cumsum of visual_this_step).
            new_qsl_list = [0]
            for n in visual_this_step:
                new_qsl_list.append(new_qsl_list[-1] + n)
            num_visual_actual = new_qsl_list[-1]
            new_seq_lens_list = [
                visual_so_far[i] + visual_this_step[i] for i in range(num_reqs)
            ]
            num_computed_visual_list = [int(v) for v in visual_so_far]

            device = common_attn_metadata.query_start_loc.device
            new_qsl_cpu = torch.tensor(new_qsl_list, dtype=torch.int32)
            new_qsl = new_qsl_cpu.to(device, non_blocking=True)
            new_seq_lens_cpu = torch.tensor(new_seq_lens_list, dtype=torch.int32)
            new_seq_lens = new_seq_lens_cpu.to(device, non_blocking=True)
            new_num_computed_cpu = torch.tensor(num_computed_visual_list,
                                                dtype=torch.int32)

            # slot_mapping: recompute in visual-frame.
            # vLLM's default slot_mapping uses backbone-frame token positions
            # (= num_computed + arange), so its slots point at offsets
            # [prompt_len + visual_so_far .. prompt_len + visual_so_far +
            # visual_this_step - 1] in the vertical block_table. But vertical
            # KV semantically lives at offsets [visual_so_far ..
            # visual_so_far + visual_this_step - 1] (no prompt prefix).
            # Recompute using the same vLLM formula:
            #   slot = block_table[req][pos // block_size] * block_size
            #          + pos % block_size
            # but with visual-frame positions.
            block_size = common_attn_metadata.block_table_tensor.shape[1]  # placeholder
            # The right block_size is the vertical group's block size — read
            # from the spec's metadata via the underlying builder.
            block_size = self.kv_cache_spec.block_size
            block_table_tensor = common_attn_metadata.block_table_tensor
            # For each req i, generate visual_this_step[i] slot indices.
            slot_pieces = []
            for i in range(num_reqs):
                vstep = visual_this_step[i]
                if vstep == 0:
                    continue
                vso = visual_so_far[i]
                positions = torch.arange(vso, vso + vstep,
                                         dtype=torch.int64, device=device)
                # block_table_tensor[i, j] is the physical block number for
                # req i's logical block j. Slot = block_num * block_size + offset.
                block_idx = positions // block_size
                offsets = positions % block_size
                block_nums = block_table_tensor[i, block_idx]
                slot_pieces.append(block_nums * block_size + offsets)
            if slot_pieces:
                new_slot_mapping = torch.cat(slot_pieces, dim=0)
            else:
                new_slot_mapping = common_attn_metadata.slot_mapping[:0]

            new_max_query = max(visual_this_step) if visual_this_step else 0
            new_max_seq = max(new_seq_lens_list) if new_seq_lens_list else 0

            new_md = copy(common_attn_metadata)
            new_md.causal = False
            new_md.query_start_loc = new_qsl
            new_md.query_start_loc_cpu = new_qsl_cpu
            new_md.seq_lens = new_seq_lens
            new_md.seq_lens_cpu = new_seq_lens_cpu
            new_md.num_computed_tokens_cpu = new_num_computed_cpu
            new_md.num_actual_tokens = num_visual_actual
            new_md.max_query_len = new_max_query
            new_md.max_seq_len = new_max_seq
            new_md.slot_mapping = new_slot_mapping

            # Snapshot for diagnostics (capped).
            if len(VerticalAttentionBuilder._intro_log) < 16:
                VerticalAttentionBuilder._intro_log.append({
                    "phase": per_step.get("phase"),
                    "batch_req_ids": list(batch_req_ids),
                    "visual_so_far": list(visual_so_far),
                    "visual_this_step": list(visual_this_step),
                    "rewritten_qsl": new_qsl_list,
                    "rewritten_seq_lens": new_seq_lens_list,
                    "rewritten_num_actual_tokens": num_visual_actual,
                    "old_qsl": common_attn_metadata.query_start_loc_cpu.tolist()
                        if common_attn_metadata.query_start_loc_cpu is not None else None,
                    "old_seq_lens": common_attn_metadata.seq_lens_cpu.tolist()
                        if common_attn_metadata.seq_lens_cpu is not None else None,
                })

            return super().build(common_prefix_len, new_md, fast_build)

    _VERTICAL_BUILDER_REF["cls"] = VerticalAttentionBuilder
    return subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=VerticalAttentionBuilder,
    )


class VerticalAttention(Attention):
    """Standard vLLM Attention but with a vertical-tagged backend so the
    model_runner allocates a separate KV slab and rebuilds metadata.

    Marked attn_type=DECODER (we still want PagedAttention semantics — write +
    read KV, autocached prefix).
    """

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int,
                 cache_config=None, quant_config=None, prefix: str = "",
                 **kwargs):
        dtype = torch.get_default_dtype()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype, block_size = "auto", 16
        underlying = get_attn_backend(head_size, dtype, kv_cache_dtype,
                                      block_size)
        backend = create_vertical_attention_backend(underlying)
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_type=AttentionType.DECODER,
            attn_backend=backend,
            **kwargs,
        )


class VerticalSelfAttention(nn.Module):
    """vertical_block self-attention.

    VerticalAttention routes these layers to a separate vLLM KVCacheGroup.
    The runtime rewrites attention metadata into the visual-token frame before
    calling this block during diagonal decoding.

    Mirrors UNISAttention layout so the FlashAR ckpt's
    vertical_block.{i}.self_attn.{q_proj/k_proj/v_proj/o_proj/q_norm/k_norm}
    keys map cleanly via packed_modules_mapping.
    """

    def __init__(self, config, cache_config, quant_config, prefix: str):
        super().__init__()
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_pos = config.max_position_embeddings
        self.scale = head_dim ** -0.5
        self.layer_idx = int(prefix.split(".")[-2])  # vertical_block.{i}.self_attn

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size, head_dim,
            self.total_num_heads, self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim, self.hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            head_dim, rotary_dim=head_dim, max_position=max_pos,
            base=rope_theta, rope_scaling=rope_scaling)
        # vLLM Attention slot. KV is allocated here and used by the packed
        # vertical decode path.
        self.attn = VerticalAttention(
            num_heads=self.num_heads, head_size=head_dim,
            scale=self.scale, num_kv_heads=self.num_kv_heads,
            cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.attn")

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor,
                past_key_value=None):
        """Route through vLLM's PagedAttention via self.attn(q, k, v).

        Inputs are batched (B, N, D) for callers' convenience but we flatten
        to packed (B*N, D) before calling self.attn — vLLM's Attention takes
        packed-token tensors and reads visual-frame metadata from
        forward_context.attn_metadata[self.attn.layer_name] (set by
        VerticalAttentionBuilder.build()).

        Args:
            positions: (B, N) absolute position ids for RoPE
            hidden_states: (B, N, D) batched visual tokens (B = cond+uncond)
            past_key_value: ignored here (KV lives in vLLM PagedAttention)
        Returns:
            (output of shape (B, N, D), past_key_value)  — past_key_value is
            returned to match the wrapper's call sites; it's the
            same object passed in (or None) since vLLM owns the cache now.
        """
        assert hidden_states.dim() == 3, "vertical attn expects (B, N, D)"
        B, N, D = hidden_states.shape
        T = B * N

        # Flatten to packed (T, D) for vLLM Attention.
        h_flat = hidden_states.reshape(T, D).contiguous()
        qkv, _ = self.qkv_proj(h_flat)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # qk-norm per head (matches UNISAttention.forward in vLLM source).
        q = q.view(T, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = q.view(T, self.num_heads * self.head_dim)
        k = k.view(T, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = k.view(T, self.num_kv_heads * self.head_dim)

        # vLLM rotary_emb modifies q/k IN-PLACE; need contiguous.
        q = q.contiguous()
        k = k.contiguous()
        pos_flat = positions.reshape(-1).contiguous()
        q, k = self.rotary_emb(pos_flat, q, k)

        v = v.contiguous()
        # self.attn returns flat (T, num_heads * head_dim); KV write/read goes
        # through PagedAttention using the visual-frame metadata.
        attn_out = self.attn(q, k, v)
        out, _ = self.o_proj(attn_out)
        # Reshape back to (B, N, D)
        out = out.view(B, N, D)
        return out, past_key_value


class VerticalDecoderLayer(nn.Module):
    """One layer of vertical_block; structurally identical to UNISDecoderLayer
    but uses VerticalSelfAttention.

    Key match (FlashAR ckpt):
      vertical_block.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}
      vertical_block.{i}.mlp.{gate_proj,up_proj,down_proj}
      vertical_block.{i}.{input_layernorm,post_attention_layernorm}
    """

    def __init__(self, config, cache_config, quant_config, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = VerticalSelfAttention(
            config=config, cache_config=cache_config,
            quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = UNISMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor,
                residual: Optional[torch.Tensor],
                past_key_value=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states, past_key_value = self.self_attn(
            positions=positions, hidden_states=hidden_states,
            past_key_value=past_key_value)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, past_key_value


class UNISFlashARForCausalLM(UNISForCausalLM):
    """vLLM-registered FlashAR model.

    Layout vs UNISForCausalLM:
    - Inherits the full backbone (UNISModel with config.num_hidden_layers).
    - Adds model.vertical_block: ModuleList of N VerticalDecoderLayer.
      Each VerticalSelfAttention.attn is a VerticalAttention so vLLM auto-
      discovers it and the spec patch routes it to its own KVCacheGroup.
    - Adds model.vertical_norm.
    - Adds horizontal_head, vertical_head (logits projections to vocab).
    - Adds hv_gate_mlp, hv_gate_corner (per-position gate over h/v branches).

    Backbone path runs unchanged (vLLM compiles + paged-attentions it).
    vertical_block runs only on visual tokens (slice from full hidden) inside
    a custom forward path; see forward() for the visual-frame slicing logic.

    Compute logits is intercepted by the runtime: FlashAR samples visual tokens
    from packed horizontal/vertical logits and asks vLLM's default sampler to
    no-op for those steps.
    """

    # Backbone's packed_modules_mapping (qkv_proj, gate_up_proj) inherits
    # from UNISForCausalLM and applies to vertical_block's same-named
    # children automatically — AutoWeightsLoader matches by suffix.

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        n_vertical = getattr(config, "vertical_layers", 4)
        vert_prefix = maybe_prefix(prefix, "model.vertical_block")
        self.model.vertical_block = nn.ModuleList([
            VerticalDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{vert_prefix}.{i}",
            )
            for i in range(n_vertical)
        ])
        self.model.vertical_norm = RMSNorm(config.hidden_size,
                                           eps=config.rms_norm_eps)

        # FlashAR heads/gates. Top-level (not inside model.) to match
        # FlashAR ckpt key layout (horizontal_head.weight, hv_gate_corner.*).
        self.horizontal_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.vertical_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        gate_proj_dim = max(64, config.hidden_size // 8)
        self.hv_gate_mlp = nn.Sequential(
            nn.Linear(2 * config.hidden_size, gate_proj_dim, bias=False),
            nn.SiLU(),
            nn.Linear(gate_proj_dim, 1, bias=True),
        )
        self.hv_gate_corner = nn.Linear(config.hidden_size, 1, bias=True)

        # Per-parent vertical state (populated by wrapper before each step).
        # The VerticalAttentionBuilder reads this to construct visual-only
        # CommonAttentionMetadata for vertical layers.
        self._vertical_meta_per_step: dict = {}

    # ---- weight loading ----------------------------------------------------
    # FlashAR extras safetensors keys look like:
    #   horizontal_head.weight, vertical_head.weight, vertical_norm.weight,
    #   hv_gate_mlp.0.weight, hv_gate_mlp.2.{weight,bias}, hv_gate_corner.*
    #   vertical_block.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}
    #   vertical_block.{i}.mlp.{gate_proj,up_proj,down_proj}
    #   vertical_block.{i}.{input_layernorm,post_attention_layernorm}
    # The model owns these at:
    #   model.{embed_tokens,layers,norm} for backbone.* keys /
    #   model.vertical_block.{i}.* / model.vertical_norm /
    #   horizontal_head / vertical_head / hv_gate_mlp / hv_gate_corner
    # We translate the prefix when loading so AutoWeightsLoader can apply its
    # standard packed-modules merging to the renamed stream.
    _FLASHAR_EXTRAS_PREFIX_MAP = {
        "backbone.": "model.",
        "vertical_block.": "model.vertical_block.",
        "vertical_norm.": "model.vertical_norm.",
    }

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap(stream):
            for name, w in stream:
                for src, dst in self._FLASHAR_EXTRAS_PREFIX_MAP.items():
                    if name.startswith(src):
                        name = dst + name[len(src):]
                        break
                yield name, w
        loaded = super().load_weights(remap(weights))
        # FlashAR generation uses horizontal_head / vertical_head for visual
        # logits; public FlashAR checkpoints may omit the base LM head.
        loaded.add("lm_head.weight")
        return loaded

    # ---- per-step vertical metadata interface ------------------------------
    def set_vertical_meta(self, meta: dict) -> None:
        """Wrapper hooks call this BEFORE each engine_core.step() to provide
        per-request vertical-frame info: {parent_id: {visual_so_far: int,
        visual_this_step: int}}. The vertical attention builder reads it."""
        self._vertical_meta_per_step = meta

    def get_vertical_meta(self) -> dict:
        return self._vertical_meta_per_step

    # ---- vertical block forward step ---------------------------------------
    def vertical_forward_step(
        self,
        step_hidden: torch.Tensor,           # (B, N, D)
        past_key_value=None,                 # HF DynamicCache (per-parent) or None
        position_ids: Optional[torch.Tensor] = None,  # (B, N) absolute
    ):
        """Run vertical_block; returns (v_hidden_post_norm, updated_cache).

        Used by the eager-style vertical path with a per-parent DynamicCache.
        The vLLM runtime uses ``vertical_forward_step_packed`` instead.
        """
        from transformers import DynamicCache
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, n, layer_idx=0: \
                self.get_seq_length(layer_idx)
        assert step_hidden.dim() == 3, "step_hidden must be (B, N, D)"
        cache = past_key_value if past_key_value is not None else DynamicCache()
        B, N, _ = step_hidden.shape
        if position_ids is None:
            cache_offset = cache.get_seq_length(0) or 0
            pos = torch.arange(cache_offset, cache_offset + N,
                               device=step_hidden.device, dtype=torch.long)
            position_ids = pos.unsqueeze(0).expand(B, -1)

        v_hidden = step_hidden
        residual = None
        for layer in self.model.vertical_block:
            v_hidden, residual, cache = layer(
                positions=position_ids,
                hidden_states=v_hidden,
                residual=residual,
                past_key_value=cache,
            )
        # Final norm (RMSNorm in vLLM accepts (hidden, residual) tuple)
        if residual is not None:
            v_hidden, _ = self.model.vertical_norm(v_hidden, residual)
        else:
            v_hidden = self.model.vertical_norm(v_hidden)
        return v_hidden, cache

    def vertical_forward_step_packed(
        self,
        step_hidden: torch.Tensor,           # (T, D) packed across the whole batch
        position_ids: torch.Tensor,          # (T,) absolute RoPE positions
    ) -> torch.Tensor:
        """Run vertical_block once on the packed visual-token batch.

        Required so vLLM Attention sees q/k/v whose shape exactly matches the
        visual-frame attn_metadata (num_actual_tokens, query_start_loc,
        slot_mapping). Calling vertical_forward_step per-parent in a multi-
        parent batch corrupts PagedAttention because metadata is built once
        but self.attn() is invoked multiple times with mismatched shapes.

        Returns post-vertical_norm hidden of shape (T, D).
        """
        # VerticalSelfAttention.forward expects (B, N, D); use B=1, N=T.
        v_hidden = step_hidden.unsqueeze(0)               # (1, T, D)
        positions = position_ids.unsqueeze(0)             # (1, T)
        residual = None
        for layer in self.model.vertical_block:
            v_hidden, residual, _ = layer(
                positions=positions,
                hidden_states=v_hidden,
                residual=residual,
                past_key_value=None,
            )
        if residual is not None:
            v_hidden, _ = self.model.vertical_norm(v_hidden, residual)
        else:
            v_hidden = self.model.vertical_norm(v_hidden)
        return v_hidden.squeeze(0)                         # (T, D)

    # ---- diagonal step logits computation ----------------------------------
    def hv_gate_from_pair(
        self, h_feat: torch.Tensor, v_feat: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if h_feat.numel() == 0:
            shape = (*h_feat.shape[:-1], 1)
            return torch.full(shape, 0.5, device=h_feat.device, dtype=out_dtype)
        gate_dtype = self.hv_gate_mlp[0].weight.dtype
        feat = torch.cat([h_feat, v_feat], dim=-1).to(gate_dtype)
        gate = torch.sigmoid(self.hv_gate_mlp(feat))
        return gate.to(out_dtype)

    def hv_gate_from_corner(
        self, cond_hidden: torch.Tensor, out_dtype: torch.dtype,
    ) -> torch.Tensor:
        gate_dtype = self.hv_gate_corner.weight.dtype
        gate = torch.sigmoid(self.hv_gate_corner(cond_hidden.to(gate_dtype)))
        return gate.to(out_dtype)

    def compute_step_logits_from_prev(
        self,
        cond_horizontal_hidden: torch.Tensor,  # (B, 1, D)
        cond_vertical_hidden: torch.Tensor,    # (B, 1, D)
        prev_h_hidden: Optional[torch.Tensor], # (B, N_prev, D) or None
        prev_v_hidden: Optional[torch.Tensor], # (B, N_prev, D) or None
        step_positions: torch.Tensor,          # (N_s,) flat indices
        prev_positions: torch.Tensor,          # (N_prev,) flat indices
        height: int, width: int,
        # Optional pre-projected logits. When provided, the corresponding
        # head() call is skipped — used by the packed-projection path that
        # batches projections across parents before this method runs.
        cond_h_logits_pre: Optional[torch.Tensor] = None,  # (B, 1, V)
        cond_v_logits_pre: Optional[torch.Tensor] = None,  # (B, 1, V)
        h_prev_logits_pre: Optional[torch.Tensor] = None,  # (B, N_prev, V)
        v_prev_logits_pre: Optional[torch.Tensor] = None,  # (B, N_prev, V)
    ) -> torch.Tensor:
        """Port of FlashAR's _compute_step_logits_from_prev to the model class.

        Returns step_logits of shape (B, N_s, vocab_size).
        """
        device = cond_horizontal_hidden.device
        batch_size = int(cond_horizontal_hidden.size(0))

        cond_h_logits = (cond_h_logits_pre
                         if cond_h_logits_pre is not None
                         else self.horizontal_head(cond_horizontal_hidden))
        cond_v_logits = (cond_v_logits_pre
                         if cond_v_logits_pre is not None
                         else self.vertical_head(cond_vertical_hidden))

        step_logits = torch.empty(
            (batch_size, step_positions.numel(), self.config.vocab_size),
            device=device, dtype=cond_h_logits.dtype,
        )

        rows = step_positions // width
        cols = step_positions % width
        left_mask = cols > 0
        up_mask = rows > 0
        both_mask = left_mask & up_mask
        corner_mask = ~left_mask & ~up_mask
        h_only = left_mask & ~up_mask
        v_only = up_mask & ~left_mask

        if (prev_h_hidden is not None and prev_v_hidden is not None
                and prev_positions.numel() > 0):
            h_prev = (h_prev_logits_pre
                      if h_prev_logits_pre is not None
                      else self.horizontal_head(prev_h_hidden))
            v_prev = (v_prev_logits_pre
                      if v_prev_logits_pre is not None
                      else self.vertical_head(prev_v_hidden))
            total = int(height * width)
            pos_to_idx = torch.full((total,), -1, device=device,
                                    dtype=torch.long)
            pos_to_idx[prev_positions] = torch.arange(
                prev_positions.numel(), device=device)

            if h_only.any():
                left_idx = pos_to_idx[step_positions[h_only] - 1]
                step_logits[:, h_only, :] = h_prev[:, left_idx, :]
            if v_only.any():
                up_idx = pos_to_idx[step_positions[v_only] - width]
                step_logits[:, v_only, :] = v_prev[:, up_idx, :]
            if both_mask.any():
                left_idx = pos_to_idx[step_positions[both_mask] - 1]
                up_idx = pos_to_idx[step_positions[both_mask] - width]
                rw = self.hv_gate_from_pair(
                    prev_h_hidden[:, left_idx, :],
                    prev_v_hidden[:, up_idx, :],
                    out_dtype=h_prev.dtype,
                )
                step_logits[:, both_mask, :] = (
                    rw * h_prev[:, left_idx, :]
                    + (1.0 - rw) * v_prev[:, up_idx, :]
                )
        elif h_only.any() or v_only.any() or both_mask.any():
            raise RuntimeError(
                "prev_h/v_hidden missing for non-corner prediction.")

        if corner_mask.any():
            rw_corner = self.hv_gate_from_corner(
                cond_horizontal_hidden, out_dtype=cond_h_logits.dtype)
            cond_logits = (
                rw_corner * cond_h_logits[:, :1, :]
                + (1.0 - rw_corner) * cond_v_logits[:, :1, :]
            )
            step_logits[:, corner_mask, :] = cond_logits.expand(
                -1, int(corner_mask.sum().item()), -1)
        return step_logits

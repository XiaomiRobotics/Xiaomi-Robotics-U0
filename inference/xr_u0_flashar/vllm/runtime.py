"""Xiaomi-Robotics-U0-FlashAR diagonal decoding runtime on top of vLLM.

Architecture:
  - vLLM owns: backbone forward, KV cache (PagedAttention), attention impl.
  - The model class UNISFlashARForCausalLM owns
    the FlashAR heads, vertical_block, vertical_norm, hv_gate.
  - Wrapper owns: per-parent diagonal state, driver loop, CFG split, sampling.

Communication: collective_rpc into the worker process. Worker holds:
  - model_runner (vLLM)
  - model.last_aux / last_final (set by patched outer forward)
  - model._unis_flashar_states (per-parent FlashAR diagonal state)
"""

from __future__ import annotations

import os
import site
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from xr_u0_flashar.vllm.cfg import (
    COND_PREFIX,
    UNCOND_PREFIX,
    parent_id_from_child,
    role_of_child,
)
from xr_u0_flashar.vllm.debug_dump import LOGITS_DUMP
from xr_u0_flashar.vllm.diagonal import (
    diagonal_list,
    diagonal_positions,
    diagonal_prefix_sum,
    diagonal_tensor,
)
from xr_u0_flashar.vllm.sampling import (
    record_sample_summary,
    sample_diagonal_logits,
)


# ---------------------------------------------------------------- NVTX helper
# Gate via XR_U0_FLASHAR_NVTX=1 so eager runs pay nothing. Each range pushes a CUDA
# nvtx marker (visible in Nsight Systems / chrome trace under the "user" lane).
_NVTX_ON = os.environ.get("XR_U0_FLASHAR_NVTX") == "1"

# Pack horizontal/vertical head projections across parents into one matmul per
# head per step instead of one projection per parent. Default ON; set to "0" for
# controlled debugging.
_PACKED_LOGITS = os.environ.get("XR_U0_FLASHAR_PACKED_LOGITS", "1") != "0"


def _all_rpc_ok(info) -> bool:
    return bool(info) and all(isinstance(x, dict) and x.get("ok") for x in info)


def _ensure_vllm_plugin_entrypoint() -> None:
    """Expose FlashAR's model registration to spawned vLLM TP workers.

    vLLM loads `vllm.general_plugins` in helper subprocesses and worker
    processes. This repo is used source-tree style, so no installed
    distribution advertises our plugin. A tiny runtime .dist-info directory
    on PYTHONPATH is enough for importlib.metadata entry-point discovery.
    """
    plugin_root = Path("/tmp/xr_u0_flashar_vllm_plugin")
    dist_info = plugin_root / "xr_u0_flashar_vllm_plugin-0.0.dist-info"
    entry_points = dist_info / "entry_points.txt"
    metadata = dist_info / "METADATA"
    dist_info.mkdir(parents=True, exist_ok=True)
    entry_points.write_text(
        "[vllm.general_plugins]\n"
        "xr_u0_flashar_vllm = xr_u0_flashar.vllm.plugin:register_unis_flashar_model\n",
        encoding="utf-8",
    )
    metadata.write_text(
        "Metadata-Version: 2.1\n"
        "Name: xr-u0-flashar-vllm-plugin\n"
        "Version: 0.0\n",
        encoding="utf-8",
    )
    root = str(plugin_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    py_path = os.environ.get("PYTHONPATH", "")
    parts = [p for p in py_path.split(os.pathsep) if p]
    if root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([root] + parts)
    site.addsitedir(root)


class _NvtxRange:
    __slots__ = ("name",)
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        if _NVTX_ON:
            torch.cuda.nvtx.range_push(self.name)
        return self
    def __exit__(self, *exc):
        if _NVTX_ON:
            torch.cuda.nvtx.range_pop()
        return False


def _nvtx(name: str) -> _NvtxRange:
    return _NvtxRange(name)


# ---------------------------------------------------------------- worker side
# These functions run inside the vLLM worker process. They take `self` =
# Worker instance (with self.model_runner). Side-effects park state on the
# model so subsequent rpc calls can find it.


def worker_install_hooks(self, *_args, **_kwargs) -> dict:
    """Install the aux-capture hook on the registered UNISFlashARForCausalLM.

    The model class owns horizontal_head /
    vertical_head / hv_gate / vertical_block / vertical_norm. Wrapper no
    longer loads a UNISFlashARExtras module — those weights came in via the
    standard ckpt loader at LLM init time. Only the aux-layer hook is
    needed because vLLM's UNISModel.forward returns hidden + aux60 as a
    tuple when aux_hidden_state_layers is set.

    Idempotent.
    """
    m = self.model_runner.model
    if not hasattr(m, "set_aux_hidden_state_layers"):
        return {"ok": False, "err": "model lacks set_aux_hidden_state_layers"}
    if not hasattr(m, "vertical_forward_step"):
        return {"ok": False, "err": "model is not UNISFlashARForCausalLM"}

    if getattr(m, "_aux_hook_installed", False):
        return {"ok": True, "already": True}

    m.set_aux_hidden_state_layers((m.config.num_hidden_layers - 4,))
    inner = m.model

    def patched_outer(self_m, input_ids, positions,
                      intermediate_tensors=None, inputs_embeds=None):
        # 1) Backbone forward (vLLM-compiled).
        with _nvtx("backbone_forward"):
            out = inner(input_ids, positions, intermediate_tensors,
                        inputs_embeds)
        if isinstance(out, tuple):
            final, aux_list = out
            self_m.last_aux = aux_list
            self_m.last_final = final
        else:
            aux_list = None
            final = out
            self_m.last_aux = None
            self_m.last_final = out if hasattr(out, "shape") else None

        # 2) Run vertical_block inside the same
        #    forward_context, slicing visual tokens out of last_aux. Output
        #    stashed on the model so the post-forward compute_logits hook can
        #    read it. Skipped on prefill (visual_this_step=0).
        # If FLASHAR_DISABLE_VERTICAL=1, skip vertical entirely (debug aid).
        try:
            import os as _os
            if not _os.environ.get("FLASHAR_DISABLE_VERTICAL"):
                from xr_u0_flashar.vllm.model import _VERTICAL_META_HOOK
                ps = _VERTICAL_META_HOOK.get("per_step")
                if ps is not None and aux_list is not None:
                    with _nvtx("vertical_packed"):
                        _run_vertical_in_context(self_m, aux_list[0], positions, ps)
        except Exception as e:
            self_m._vertical_run_err = repr(e)

        return final

    m.forward = types.MethodType(patched_outer, m)
    m._aux_hook_installed = True
    return {"ok": True}


def _run_vertical_in_context(model, aux60_full, positions_full, per_step):
    """Run vertical_block on the visual-token subset of aux60_full ONCE for
    the entire batch (all parents packed together), store per-parent outputs
    on model._latest_vertical_outputs.

    Why one call: vLLM Attention reads its metadata from forward_context once
    per layer per step. If we called self.attn() multiple times per step (once
    per parent) the (q,k,v) shapes would mismatch the metadata's
    num_actual_tokens / query_start_loc, and PagedAttention's slot_mapping
    indexing would corrupt KV cache. Packing everything into a single call
    makes shapes match exactly.

    Token layout: visual tokens are gathered in the SAME order as
    batch_req_ids (which is what VerticalAttentionBuilder.build() used to
    build the visual-frame metadata).
    """
    req_id_to_visual = per_step.get("req_id_to_visual", {})
    batch_req_ids = per_step.get("batch_req_ids", [])
    if not batch_req_ids:
        return

    qmeta = getattr(model, "_query_meta_current", None)
    if qmeta is None:
        return
    qsl, _num_reqs, req_ids_in_batch = qmeta
    states = getattr(model, "_unis_flashar_states", None) or {}

    # Walk batch_req_ids in metadata order: collect aux slices + RoPE positions
    # contiguously, plus per-child cu_seqlens for splitting outputs back.
    aux_slices: list[torch.Tensor] = []
    pos_pieces: list[torch.Tensor] = []
    child_records: list[dict] = []  # one entry per child req with vstep > 0
    for i, rid in enumerate(req_ids_in_batch):
        parent = parent_id_from_child(rid)
        role = role_of_child(rid)
        if parent is None or role is None:
            continue
        info = req_id_to_visual.get(rid, {})
        vstep = int(info.get("visual_this_step", 0))
        if vstep == 0:
            continue
        st = states.get(parent)
        if st is None:
            continue
        # backbone-frame query slice: last vstep tokens of this child's query
        q_lo, q_hi = int(qsl[i]), int(qsl[i + 1])
        v_lo, v_hi = q_hi - vstep, q_hi
        aux_slices.append(aux60_full[v_lo:v_hi])  # (vstep, D)
        # RoPE: prompt_len + grid_flat_idx (raster order)
        just_diag = st["diag_idx_to_predict"] - 1
        H, W = st["height"], st["width"]
        diag_tensor = diagonal_tensor(just_diag, H, W, aux60_full.device)
        pos_pieces.append(diag_tensor + st["prompt_len"])
        child_records.append({"parent": parent, "role": role, "vstep": vstep})
        if os.environ.get("FLASHAR_DUMP_POS") and just_diag <= 1:
            _plmax = max(st["prompt_len"], st["uncond_len"])
            print(f"[POSDUMP-vert] diag={just_diag} role={role} "
                  f"uses_offset=prompt_len={st['prompt_len']} (pl_max={_plmax}) "
                  f"visual_pos[:5]={(diag_tensor[:5] + st['prompt_len']).tolist()}",
                  flush=True)

    if not aux_slices:
        return

    # Pack to (T, D) for vLLM Attention. Matches metadata.num_actual_tokens.
    v_packed = torch.cat(aux_slices, dim=0)                      # (T, D)
    pos_packed = torch.cat(pos_pieces, dim=0)                    # (T,)
    v_hidden_packed = model.vertical_forward_step_packed(
        v_packed, position_ids=pos_packed)                       # (T, D)

    # Split back per child, then group cond+uncond into (B=2, N, D) per parent.
    outputs: dict = {}
    parent_split: dict = defaultdict(dict)
    cursor = 0
    for rec in child_records:
        v = v_hidden_packed[cursor:cursor + rec["vstep"]]
        cursor += rec["vstep"]
        parent_split[rec["parent"]][rec["role"]] = v             # (N, D)
    for parent, roles in parent_split.items():
        if "cond" not in roles or "uncond" not in roles:
            continue
        N = roles["cond"].shape[0]
        if roles["uncond"].shape[0] != N:
            continue
        outputs[parent] = torch.stack(
            [roles["cond"], roles["uncond"]], dim=0)             # (2, N, D)
    model._latest_vertical_outputs = outputs



def worker_install_query_meta_capture(self) -> dict:
    """Patch model_runner._prepare_inputs to stash query_start_loc + req_ids
    onto m._unis_flashar_state["query_meta"] each step.

    Needed for slicing last_aux / last_final per-request when N_s > 1.
    """
    runner = self.model_runner
    if getattr(runner, "_query_meta_hook_installed", False):
        return {"ok": True, "already": True}
    orig_prep = type(runner)._prepare_inputs

    def patched_prep(self_runner, scheduler_output):
        m = self_runner.model
        # Publish the current batch's req_id ordering to
        # the vertical hook BEFORE build() runs (build is invoked from inside
        # orig_prep). input_batch.req_ids is already updated by
        # _update_states which runs strictly before _prepare_inputs.
        try:
            from xr_u0_flashar.vllm.model import _VERTICAL_META_HOOK
            ps = _VERTICAL_META_HOOK.get("per_step")
            if ps is not None:
                num_reqs_pre = self_runner.input_batch.num_reqs
                ps["batch_req_ids"] = list(
                    self_runner.input_batch.req_ids[:num_reqs_pre])
        except Exception as e:
            m._vertical_hook_err = repr(e)

        result = orig_prep(self_runner, scheduler_output)
        qsl = self_runner.query_start_loc.cpu.numpy().copy()
        num_reqs = self_runner.input_batch.num_reqs
        req_ids = list(self_runner.input_batch.req_ids[:num_reqs])
        m._query_meta_current = (qsl, num_reqs, req_ids)
        # Expose per-row prompt/computed counts so the
        # FlashAR diagonal hook can detect the "last prefill chunk" boundary
        # under chunked prefill (when V1 chunks a prompt into multiple
        # forwards, only the FINAL chunk's last token is the true prompt
        # last-token whose hidden becomes the cond_h/cond_v anchor).
        # Bisect knob: FLASHAR_DISABLE_PROMPT_LEN_HOOK=1 turns this off.
        if not os.environ.get("FLASHAR_DISABLE_PROMPT_LEN_HOOK"):
            ib = self_runner.input_batch
            m._query_prompt_lens = list(ib.num_prompt_tokens[:num_reqs])
            m._query_num_computed_after = [
                int(ib.num_computed_tokens_cpu[i])
                + int(scheduler_output.num_scheduled_tokens.get(req_ids[i], 0))
                for i in range(num_reqs)
            ]

        # Rewrite positions for visual decode steps.
        # FlashAR-eager uses RoPE position = pl_max + grid_flat_idx for each
        # visual token. vLLM's default is num_computed + arange — wrong for
        # diagonal scheduling (e.g. (1,0) gets pos 36 but should be 66).
        # We iterate ALL running req_ids, group cond+uncond by parent, and
        # override per-parent. With per-request state, multiple parents
        # at different diagonals can be active in the same step.
        try:
            states = getattr(m, "_unis_flashar_states", None)
            if states:
                qsl_arr = self_runner.query_start_loc.cpu.numpy()
                pos_cpu = self_runner.positions.cpu
                touched = False
                for i, rid in enumerate(req_ids):
                    parent = parent_id_from_child(rid)
                    if parent is None:
                        continue
                    st = states.get(parent)
                    if st is None or not st.get("phase", "").endswith("_pending_forward"):
                        continue
                    just_diag = st["diag_idx_to_predict"] - 1
                    just_pos_list = diagonal_list(just_diag, st["height"], st["width"])
                    pl_max = max(st["prompt_len"], st["uncond_len"])
                    q_lo, q_hi = int(qsl_arr[i]), int(qsl_arr[i + 1])
                    if (q_hi - q_lo) != len(just_pos_list):
                        # Off-cycle (e.g. prefill or all_done): vLLM's default
                        # positions are correct; skip.
                        continue
                    for k, flat in enumerate(just_pos_list):
                        pos_cpu[q_lo + k] = pl_max + flat
                    touched = True
                    if os.environ.get("FLASHAR_DUMP_POS") and just_diag <= 1:
                        _role = role_of_child(rid)
                        print(f"[POSDUMP-bb] diag={just_diag} role={_role} "
                              f"pl_max={pl_max} prompt_len={st['prompt_len']} "
                              f"uncond_len={st['uncond_len']} "
                              f"visual_pos[:5]={[pl_max + f for f in just_pos_list[:5]]}",
                              flush=True)
                if touched:
                    self_runner.positions.copy_to_gpu(
                        int(scheduler_output.total_num_scheduled_tokens))
        except Exception as e:
            m._pos_override_err = repr(e)
        return result

    runner._prepare_inputs = types.MethodType(patched_prep, runner)
    runner._query_meta_hook_installed = True
    return {"ok": True}


def worker_install_sample_skip(self) -> dict:
    """Patch model_runner._sample so that when m._unis_flashar_skip_sample is True,
    sampler returns an empty (num_reqs, 0) sampled_token_ids tensor. This
    lets the bookkeeping path no-op (no fake sampled token gets appended to
    request._all_token_ids) so the wrapper can pre-append real tokens later.
    """
    runner = self.model_runner
    if getattr(runner, "_sample_skip_hook_installed", False):
        return {"ok": True, "already": True}
    orig_sample = type(runner)._sample

    # SamplerOutput lives in vllm/v1/outputs.py
    from vllm.v1.outputs import SamplerOutput

    def patched_sample(self_runner, logits, spec_decode_metadata):
        if getattr(self_runner.model, "_unis_flashar_skip_sample", False):
            n = logits.shape[0] if logits is not None else 0
            # Shape (n, 1) so _bookkeeping_sync line max_gen_len==1 path runs
            # (avoids rejection_sampler call). Token id 0 is a placeholder;
            # we mark all reqs as 'discarded' so it gets cleared below.
            dummy_tokens = torch.zeros(
                (n, 1), dtype=torch.long,
                device=logits.device if logits is not None else "cpu",
            )
            # Force ALL request indices into discard set so _bookkeeping_sync
            # clears them at line 2186-2187.
            import numpy as np
            self_runner.num_discarded_requests = n
            self_runner.discard_request_indices.np[:n] = np.arange(n)
            return SamplerOutput(
                sampled_token_ids=dummy_tokens,
                logprobs_tensors=None,
            )
        return orig_sample(self_runner, logits, spec_decode_metadata)

    runner._sample = types.MethodType(patched_sample, runner)
    runner._sample_skip_hook_installed = True
    return {"ok": True}



def worker_install_non_causal_hook(self) -> dict:
    """Patch model_runner._prepare_inputs to set common_attn_metadata.causal=False
    on steps where m._unis_flashar_force_non_causal is True.

    FlashAR training used bidirectional attention on diagonal-step forwards
    (proximity mask was stored in attention_mask but content was all-zeros,
    and is_causal=False was set on the backbone). vLLM defaults to causal,
    making N_s diagonal tokens see each other in causal order — different
    from training. This hook flips causal=False per-step.
    """
    runner = self.model_runner
    if getattr(runner, "_non_causal_hook_installed", False):
        return {"ok": True, "already": True}
    orig_prep = type(runner)._prepare_inputs

    def patched_prep(self_runner, scheduler_output):
        result = orig_prep(self_runner, scheduler_output)
        # result is a tuple; metadata is built inside _prepare_inputs and
        # already passed to backend.build(). Too late to flip here.
        return result

    # Better approach: patch the FA backend metadata builder directly.
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
    if not getattr(FlashAttentionMetadataBuilder, "_diagonal_non_causal_patched", False):
        orig_build = FlashAttentionMetadataBuilder.build

        def patched_build(self_b, common_prefix_len, common_attn_metadata, fast_build=False):
            # Read the per-step flag from our model
            try:
                m = runner.model
                force_nc = getattr(m, "_unis_flashar_force_non_causal", False)
                if not hasattr(m, "_non_causal_log"):
                    m._non_causal_log = []
                m._non_causal_log.append({
                    "force_non_causal": force_nc,
                    "n_actual_tokens": int(common_attn_metadata.num_actual_tokens),
                    "max_query_len": int(common_attn_metadata.max_query_len),
                    "causal_before": bool(common_attn_metadata.causal),
                })
                if force_nc:
                    common_attn_metadata.causal = False
                m._non_causal_log[-1]["causal_after"] = bool(common_attn_metadata.causal)
            except Exception as e:
                pass
            attn_md = orig_build(self_b, common_prefix_len, common_attn_metadata, fast_build=fast_build)
            try:
                m._non_causal_log[-1]["attn_metadata_causal"] = bool(attn_md.causal)
            except Exception:
                pass
            return attn_md

        FlashAttentionMetadataBuilder.build = patched_build
        FlashAttentionMetadataBuilder._diagonal_non_causal_patched = True

    runner._non_causal_hook_installed = True
    return {"ok": True}

def worker_install_diagonal_decode(self) -> dict:
    """Install the multi-token-per-diagonal compute_logits hook (idempotent).

    This hook is installed once per worker. Each generate call registers its
    own per-parent state via ``worker_register_diagonal_request``.

    On every forward after prefill, this hook iterates all parents in
    m._unis_flashar_states whose req_ids are present in this step's query_meta:
      1. slices last_aux / last_final per parent (cond + uncond row) via query_meta
      2. caches just-forwarded diagonal hidden into per-parent arrays
      3. picks next diagonal's positions via diagonal helpers
      4. runs compute_step_logits_from_prev with prev/cur cached hidden
      5. CFG combine (cond + s*(c-u)) + visual mask + sample
      6. stores sampled tokens in next_pre_append_tokens
      7. enables _unis_flashar_skip_sample so vLLM sampler no-ops

    The driver loop reads each parent's next_pre_append_tokens before each
    step and pre-appends them to BOTH child Requests' _all_token_ids /
    _output_token_ids.
    """
    m = self.model_runner.model
    if not hasattr(m, "vertical_forward_step"):
        return {"ok": False, "err": "model is not UNISFlashARForCausalLM"}
    if not hasattr(m.model, "aux_hidden_state_layers") and not hasattr(m, "set_aux_hidden_state_layers"):
        return {"ok": False, "err": "aux hook missing"}
    if not hasattr(m, "_unis_flashar_states"):
        m._unis_flashar_states = {}

    if getattr(m, "_diagonal_hook_installed", False):
        return {"ok": True, "already": True}

    if not hasattr(m, "_orig_compute_logits"):
        m._orig_compute_logits = type(m).compute_logits

    def diagonal_compute_logits(self_m, hidden_states):
        """Post-forward FlashAR hook. Iterates all active parents.

        Returns dummy logits — vLLM's sampler is skipped via _unis_flashar_skip_sample.
        """
        states = self_m._unis_flashar_states
        if not states:
            # No active FlashAR generation. Fall through to vLLM's default.
            return self_m._orig_compute_logits(self_m, hidden_states)
        if _NVTX_ON:
            torch.cuda.nvtx.range_push("diag_compute_logits")

        # The model class owns the FlashAR computation methods
        # (vertical_forward_step, compute_step_logits_from_prev, hv_gate_*).
        # Keep this alias local so the packed and per-parent branches can share
        # call sites.
        ext = self_m
        last_aux_full = self_m.last_aux[0]   # (n_total_tokens, D)
        last_final_full = self_m.last_final  # (n_total_tokens, D)
        device = last_aux_full.device

        qmeta = getattr(self_m, "_query_meta_current", None)
        if qmeta is None:
            return _dummy_logits(hidden_states)
        qsl, _num_reqs, req_ids = qmeta
        # Per-child {"prompt_done": bool} used by the prefill branch to
        # detect partial prefill chunks under chunked_prefill. A child whose
        # last forwarded token is not yet the prompt's final token must
        # NOT capture the cond_h/v anchor — wait for the next chunk.
        prompt_lens = getattr(self_m, "_query_prompt_lens", None)
        num_computed_after = getattr(self_m, "_query_num_computed_after", None)
        child_prefill_done: dict = {}
        child_prefill_meta: dict = {}
        if prompt_lens is not None and num_computed_after is not None:
            for i, rid in enumerate(req_ids):
                if i < len(prompt_lens) and i < len(num_computed_after):
                    prompt_len_i = int(prompt_lens[i])
                    num_computed_after_i = int(num_computed_after[i])
                    child_prefill_done[rid] = (
                        num_computed_after_i >= prompt_len_i)
                    child_prefill_meta[rid] = {
                        "prompt_len": prompt_len_i,
                        "num_computed_after": num_computed_after_i,
                    }

        # Group child req slots by parent. Each entry: parent -> {"cond":(lo,hi), "uncond":(lo,hi)}
        parent_slots: "dict[str, dict[str, tuple[int, int]]]" = defaultdict(dict)
        for i, rid in enumerate(req_ids):
            parent = parent_id_from_child(rid)
            role = role_of_child(rid)
            if parent is None or role is None:
                continue
            parent_slots[parent][role] = (int(qsl[i]), int(qsl[i + 1]))

        # Per-parent prefill anchor capture happens before
        # the packed/per-parent decode dispatch. A parent's two children may
        # finish prefill in different steps (chunked prefill or single-prompt-
        # >-budget cases), so we stash each child's last-token hidden as soon
        # as its prefill chunk completes. Diag-0 sampling only fires once
        # both children's anchors are present, at which point the parent
        # transitions to diagonal_0_pending_forward.
        #
        # `just_emitted_diag0` collects parents that completed the prefill
        # → diag-0 transition in THIS step. The decode dispatcher must skip
        # them — their first decode-branch run happens on the NEXT step, when
        # the diag-0 token is forwarded.
        just_emitted_diag0: set = set()
        # Debug knob: FLASHAR_DECOUPLE_PREFILL=0 reverts to the
        # atomic-pair-only path (anchors captured by _step_one_parent
        # / _step_all_parents_packed prefill branch). Used to A/B-test the
        # decoupled path against the baseline path.
        if os.environ.get("FLASHAR_DECOUPLE_PREFILL", "1") != "0":
            for parent_id, slots in parent_slots.items():
                st = states.get(parent_id)
                if st is None:
                    continue
                if st["phase"] != "prefill_pending":
                    continue
                if _capture_prefill_anchors(
                        self_m, ext, st, parent_id, slots,
                        last_aux_full, last_final_full, device,
                        child_prefill_done, child_prefill_meta):
                    just_emitted_diag0.add(parent_id)
                    # diag 0 is always the single-corner token (1-wide), so
                    # it never needs the non-causal mask flag flipped here.

        if _PACKED_LOGITS:
            any_force_non_causal = _step_all_parents_packed(
                self_m, ext, states, parent_slots,
                last_aux_full, last_final_full, device,
                child_prefill_done=child_prefill_done,
                skip_parents=just_emitted_diag0)
        else:
            any_force_non_causal = False
            for parent_id, slots in parent_slots.items():
                st = states.get(parent_id)
                if st is None:
                    continue
                if parent_id in just_emitted_diag0:
                    # Anchors landed + diag-0 emitted in this same compute_logits
                    # invocation; the diag-0 forward happens NEXT step.
                    continue
                if st["phase"] == "prefill_pending":
                    # Still waiting on a sibling child to finish prefill.
                    continue
                if "cond" not in slots or "uncond" not in slots:
                    # Decode requires both children in the same step (atomic
                    # pair). Scheduler all_decode mode enforces this; defensive.
                    continue
                _step_one_parent(self_m, ext, st,
                                 last_aux_full, last_final_full,
                                 slots["cond"], slots["uncond"], device,
                                 cond_prefill_done=True,
                                 uncond_prefill_done=True)
                if st["phase"].endswith("_pending_forward"):
                    # Next forward will process the diagonal we just predicted.
                    # If that diagonal has > 1 token, we need non-causal mask.
                    next_diag_idx = st["diag_idx_to_predict"] - 1
                    if len(diagonal_list(next_diag_idx, st["height"], st["width"])) > 1:
                        any_force_non_causal = True

        # If any parent needs it, open non-causal attention globally for this step.
        # 1-token diagonals are unaffected (causal == non-causal at N=1), so
        # this is correct, just slightly broader than minimum-required.
        # This can be narrowed further with per-segment attention metadata.
        self_m._unis_flashar_skip_sample = True
        self_m._unis_flashar_force_non_causal = any_force_non_causal

        # Publish per-child visual-frame metadata for the next
        # forward. After diagonal_compute_logits runs, each parent's
        # diag_idx_to_predict has been incremented; visual_so_far is sum
        # of diagonals 0..diag_idx_to_predict-2 (those already FORWARDED),
        # visual_this_step is the size of the diagonal we're about to forward
        # (which is diag_idx_to_predict - 1).
        # The hook is published per CHILD req_id (cond + uncond) since the
        # builder will see two children per parent in query_start_loc.
        try:
            from xr_u0_flashar.vllm.model import set_vertical_meta_hook
            req_id_to_visual = {}
            for parent_id, st in states.items():
                ph = st.get("phase", "")
                if ph == "prefill_pending":
                    # Next forward is prefill (no visual)
                    visual_so_far = 0
                    visual_this_step = 0
                elif ph.endswith("_pending_forward"):
                    just_diag = st["diag_idx_to_predict"] - 1
                    H, W = st["height"], st["width"]
                    prefix = diagonal_prefix_sum(H, W)
                    visual_so_far = prefix[just_diag]
                    visual_this_step = prefix[just_diag + 1] - prefix[just_diag]
                else:
                    visual_so_far = 0
                    visual_this_step = 0
                for prefix in ("0_cfg_", "1_cfg_"):
                    req_id_to_visual[f"{prefix}{parent_id}"] = {
                        "visual_so_far": visual_so_far,
                        "visual_this_step": visual_this_step,
                    }
            set_vertical_meta_hook({
                "req_id_to_visual": req_id_to_visual,
                "phase": "post_compute_logits",
            })
        except Exception as e:
            self_m._vertical_hook_err = repr(e)

        if _NVTX_ON:
            torch.cuda.nvtx.range_pop()
        return _dummy_logits(hidden_states)

    m.compute_logits = types.MethodType(diagonal_compute_logits, m)
    m._diagonal_hook_installed = True
    return {"ok": True}


def worker_register_diagonal_request(
        self, parent_id: str, height: int, width: int,
        prompt_len: int, uncond_len: int,
        visual_token_offset: int, eoi_token_id: int,
        cfg_scale: float, temperature: float, top_k: int, top_p: float,
        seed: int) -> dict:
    """Register a per-parent FlashAR state into m._unis_flashar_states[parent_id].

    Called once per generate request, before submitting the parent to vLLM.
    """
    m = self.model_runner.model
    if not hasattr(m, "_unis_flashar_states"):
        m._unis_flashar_states = {}
    if parent_id in m._unis_flashar_states:
        return {"ok": False, "err": f"parent_id {parent_id} already registered"}
    m._unis_flashar_states[parent_id] = {
        "height": height,
        "width": width,
        "prompt_len": prompt_len,
        "uncond_len": uncond_len,
        "visual_token_offset": visual_token_offset,
        "eoi_token_id": eoi_token_id,
        "cfg_scale": cfg_scale,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        # Diagonal walk state
        "phase": "prefill_pending",
        "diag_idx_to_predict": 0,
        "next_pre_append_tokens": [],
        # Anchors set during prefill
        "cond_h_hidden": None,
        "cond_v_hidden": None,
        "vertical_kv_cache": None,
        # Prev diagonal hidden
        "prev_h_hidden": None,
        "prev_v_hidden": None,
        "prev_positions": None,
        # RNG (per-parent so concurrent generations don't share state)
        "generator": torch.Generator(device=next(m.parameters()).device).manual_seed(seed),
        # Diagnostics
        "diag_log": [],
        "anchor_log": [],
    }
    return {"ok": True}


def worker_unregister_diagonal_request(self, parent_id: str) -> dict:
    """Drop a parent's FlashAR state from m._unis_flashar_states. Call after all_done."""
    m = self.model_runner.model
    states = getattr(m, "_unis_flashar_states", None)
    if not states:
        return {"ok": True, "absent": True}
    states.pop(parent_id, None)
    return {"ok": True}


def worker_publish_initial_vertical_meta(self) -> dict:
    """Publish visual-frame meta for the NEXT forward, computed from
    the current m._unis_flashar_states. Called by the wrapper before the first
    engine_core.step() of a generate batch (so that prefill's build() sees
    visual_so_far=0, visual_this_step=0)."""
    from xr_u0_flashar.vllm.model import set_vertical_meta_hook
    m = self.model_runner.model
    states = getattr(m, "_unis_flashar_states", None) or {}
    req_id_to_visual = {}
    for parent_id, st in states.items():
        for prefix in ("0_cfg_", "1_cfg_"):
            req_id_to_visual[f"{prefix}{parent_id}"] = {
                "visual_so_far": 0,
                "visual_this_step": 0,
            }
    set_vertical_meta_hook({
        "req_id_to_visual": req_id_to_visual,
        "phase": "initial_prefill",
    })
    return {"ok": True, "n_reqs": len(req_id_to_visual)}


def worker_get_vertical_intro_log(self) -> list:
    """Return the per-step metadata log captured by VerticalAttentionBuilder."""
    from xr_u0_flashar.vllm.model import get_vertical_intro_log
    return get_vertical_intro_log()


def worker_reset_vertical_intro_log(self) -> dict:
    from xr_u0_flashar.vllm.model import reset_vertical_intro_log
    reset_vertical_intro_log()
    return {"ok": True}


def _capture_prefill_anchors(self_m, ext, st, parent_id, slots,
                             last_aux_full, last_final_full, device,
                             child_prefill_done: dict,
                             child_prefill_meta: "Optional[dict]" = None) -> bool:
    """Per-child anchor stash for prefill-decoupled scheduling.

    A CFG pair's two children may finish prefill in different steps under
    chunked prefill (e.g. cond completes in step N1, uncond in step N5).
    Each child's last-token hidden (the prompt-final-token output) is the
    anchor we use as cond_h_hidden / cond_v_hidden for the diagonal walk.

    On every step where this parent is in prefill_pending:
      * For each child whose forward chunk just landed AND completed prefill
        (num_computed_after >= num_prompt), clone its last-token hidden into
        st['_pending_anchors'][role]. We CLONE because last_aux_full /
        last_final_full are reused on the next forward, and the partner's
        anchor may have to live on for several steps before pairing.
      * If both children's anchors are present, sample diag-0 and flip
        the parent's phase to 'diagonal_0_pending_forward'.

    Returns True iff the diag-0 sample fired this step (caller marks the
    parent in `just_emitted_diag0`).
    """
    H, W = st["height"], st["width"]
    pending = st.setdefault("_pending_anchors", {})

    for role, q_slice in slots.items():
        if role in pending:
            # Already captured for this child in an earlier step.
            continue
        rid_key = ("0_cfg_" if role == "cond" else "1_cfg_") + parent_id
        if not child_prefill_done.get(rid_key, False):
            # This child's forwarded chunk did not yet end at prompt-last;
            # wait for the next chunk.
            continue
        lo, hi = q_slice
        if hi - lo <= 0:
            # No forward this step for this child (idle in scheduler);
            # nothing to capture from THIS step but the anchor may already
            # be captured from an earlier step.
            continue
        # last-token hidden = row (hi-1) within the packed forward.
        # CLONE because last_aux_full / last_final_full are reused next step.
        aux_last = last_aux_full[hi - 1:hi].clone()
        final_last = last_final_full[hi - 1:hi].clone()
        pending[role] = {"aux": aux_last, "final": final_last}
        meta = (child_prefill_meta or {}).get(rid_key, {})
        st.setdefault("anchor_log", []).append({
            "parent_id": parent_id,
            "role": role,
            "q_lo": int(lo),
            "q_hi": int(hi),
            "query_tokens": int(hi - lo),
            "prompt_len": int(meta.get("prompt_len", -1)),
            "num_computed_after": int(meta.get("num_computed_after", -1)),
        })

    if "cond" not in pending or "uncond" not in pending:
        # Wait for the other child.
        return False

    cond_aux_last = pending["cond"]["aux"]
    cond_final_last = pending["cond"]["final"]
    uncond_aux_last = pending["uncond"]["aux"]
    uncond_final_last = pending["uncond"]["final"]
    v_anchor = torch.stack([cond_aux_last, uncond_aux_last], dim=0)
    h_anchor = torch.stack([cond_final_last, uncond_final_last], dim=0)
    st["cond_h_hidden"] = h_anchor
    st["cond_v_hidden"] = v_anchor
    st["vertical_kv_cache"] = None

    step_pos = diagonal_tensor(0, H, W, device)
    prev_pos = torch.empty(0, device=device, dtype=torch.long)
    with _nvtx("compute_step_logits_d0"):
        step_logits = ext.compute_step_logits_from_prev(
            cond_horizontal_hidden=h_anchor,
            cond_vertical_hidden=v_anchor,
            prev_h_hidden=None, prev_v_hidden=None,
            step_positions=step_pos, prev_positions=prev_pos,
            height=H, width=W,
        )
    with _nvtx("diagonal_sample_d0"):
        sampled = sample_diagonal_logits(
            step_logits, st,
            dump_ctx={
                "parent_id": parent_id,
                "diag_idx": 0,
                "step_positions": step_pos,
            } if LOGITS_DUMP.is_active() else None,
        )
    st["next_pre_append_tokens"] = sampled.tolist()
    record_sample_summary(st, parent_id, 0, sampled)
    st["diag_idx_to_predict"] = 1
    st["phase"] = "diagonal_0_pending_forward"
    st["diag_log"].append({
        "phase": "prefill_done", "predicted_tokens": sampled.tolist(),
    })
    # Free the per-child stash; from here on we use cond_h_hidden / cond_v_hidden.
    st.pop("_pending_anchors", None)
    return True


def _step_all_parents_packed(self_m, ext, states, parent_slots,
                             last_aux_full, last_final_full, device,
                             child_prefill_done: "Optional[dict]" = None,
                             skip_parents: "Optional[set]" = None):
    """Like a fan-out of _step_one_parent across all active parents, but
    batches the 4 big head() projections across parents into single mms.

    Profiling showed per-parent calls to compute_step_logits_from_prev
    were ~91% of CUDA time (huge head projections to vocab=282926). Packing
    them yields O(1) launches per step instead of O(N_parents).

    Prefill anchors are captured by the caller via
    `_capture_prefill_anchors` BEFORE this dispatcher runs. Parents whose
    anchors+diag0 just landed this step are passed in `skip_parents` and
    excluded from the decode-payload pack (their diag-0 forward is the next
    step). Parents still waiting on a sibling's prefill simply have phase
    == "prefill_pending" and are skipped naturally.
    """
    skip_parents = skip_parents or set()
    # ---- Per-parent state prep: prefill, all_done, and decode eligibility.
    # Collect decode-path parents that still need to predict a next diagonal
    # (i.e. haven't just transitioned to all_done).
    decode_payloads = []  # list of dicts with cur_h, cur_v, st, parent_id
    any_force_non_causal = False
    for parent_id, slots in parent_slots.items():
        st = states.get(parent_id)
        if st is None:
            continue
        if parent_id in skip_parents:
            continue
        if st["phase"] == "all_done":
            continue
        if st["phase"] == "prefill_pending":
            # Still mid-prefill (sibling child not yet caught up). Anchor
            # capture is the caller's job.
            continue
        if "cond" not in slots or "uncond" not in slots:
            # Decode requires both children scheduled this step. Atomic
            # all_decode mode in the patched scheduler guarantees this for
            # post-prefill parents.
            continue

        # Decode: extract cur_h_hidden + cur_v_hidden, update st["prev_*"].
        H, W = st["height"], st["width"]
        cond_lo, cond_hi = slots["cond"]
        uncond_lo, uncond_hi = slots["uncond"]
        N_s_actual = cond_hi - cond_lo
        uncond_N_s_actual = uncond_hi - uncond_lo
        just_diag = st["diag_idx_to_predict"] - 1
        just_pos_list = diagonal_list(just_diag, H, W)
        N_just = len(just_pos_list)
        if uncond_N_s_actual != N_s_actual:
            raise RuntimeError(
                f"FlashAR CFG pair token mismatch for parent {parent_id}: "
                f"diag={just_diag}, cond_tokens={N_s_actual}, "
                f"uncond_tokens={uncond_N_s_actual}")
        if N_just != N_s_actual:
            raise RuntimeError(
                f"FlashAR diagonal token mismatch for parent {parent_id}: "
                f"diag={just_diag}, expected={N_just}, "
                f"forward_tokens={N_s_actual}, phase={st['phase']}")

        cond_final = last_final_full[cond_lo:cond_hi].unsqueeze(0)
        uncond_final = last_final_full[uncond_lo:uncond_hi].unsqueeze(0)
        cur_h_hidden = torch.cat([cond_final, uncond_final], dim=0)
        vouts = getattr(self_m, "_latest_vertical_outputs", {}) or {}
        cur_v_hidden = vouts.get(parent_id)
        if cur_v_hidden is None:
            raise RuntimeError(
                f"vertical output missing for parent {parent_id} "
                f"(diag {just_diag}, visual_this_step={N_just}); "
                f"available parents in vouts: {list(vouts.keys())}; "
                f"phase before this step: {st['phase']}")
        st["prev_h_hidden"] = cur_h_hidden
        st["prev_v_hidden"] = cur_v_hidden
        st["prev_positions"] = diagonal_tensor(just_diag, H, W, device)

        next_diag_idx = st["diag_idx_to_predict"]
        if next_diag_idx > H + W - 2:
            # Just-finished last diagonal → enqueue EOI, no logits projection.
            st["next_pre_append_tokens"] = [st["eoi_token_id"]]
            record_sample_summary(
                st, parent_id, next_diag_idx, st["next_pre_append_tokens"])
            st["phase"] = "all_done"
            st["diag_log"].append({"phase": "all_done"})
            continue

        decode_payloads.append({
            "parent_id": parent_id, "st": st,
            "cur_h_hidden": cur_h_hidden, "cur_v_hidden": cur_v_hidden,
            "next_diag_idx": next_diag_idx, "H": H, "W": W,
        })

    if not decode_payloads:
        return any_force_non_causal

    # ---- Packed projection path across all active parents.
    cond_h_list = [p["st"]["cond_h_hidden"] for p in decode_payloads]   # (2,1,D)
    cond_v_list = [p["st"]["cond_v_hidden"] for p in decode_payloads]   # (2,1,D)
    prev_h_list = [p["cur_h_hidden"] for p in decode_payloads]          # (2,N_i,D)
    prev_v_list = [p["cur_v_hidden"] for p in decode_payloads]          # (2,N_i,D)
    n_per_parent = [t.shape[1] for t in prev_h_list]

    K = len(decode_payloads)
    cond_h_packed = torch.cat(cond_h_list, dim=1)   # (2, K, D)
    cond_v_packed = torch.cat(cond_v_list, dim=1)
    prev_h_packed = torch.cat(prev_h_list, dim=1)   # (2, sum_N, D)
    prev_v_packed = torch.cat(prev_v_list, dim=1)

    with _nvtx("packed_head_proj"):
        cond_h_logits_p = ext.horizontal_head(cond_h_packed)   # (2, K, V)
        cond_v_logits_p = ext.vertical_head(cond_v_packed)
        prev_h_logits_p = ext.horizontal_head(prev_h_packed)   # (2, sum_N, V)
        prev_v_logits_p = ext.vertical_head(prev_v_packed)

    # ---- Per-parent sampling using the packed projection outputs.
    # compute_step_logits_from_prev path, then sample.
    offset = 0
    for k, p in enumerate(decode_payloads):
        st = p["st"]
        H, W = p["H"], p["W"]
        next_diag_idx = p["next_diag_idx"]
        n = n_per_parent[k]
        cond_h_pre = cond_h_logits_p[:, k:k + 1, :]
        cond_v_pre = cond_v_logits_p[:, k:k + 1, :]
        h_prev_pre = prev_h_logits_p[:, offset:offset + n, :]
        v_prev_pre = prev_v_logits_p[:, offset:offset + n, :]
        offset += n

        next_pos_list = diagonal_list(next_diag_idx, H, W)
        step_pos = diagonal_tensor(next_diag_idx, H, W, device)
        with _nvtx("compute_step_logits_packed"):
            step_logits = ext.compute_step_logits_from_prev(
                cond_horizontal_hidden=st["cond_h_hidden"],
                cond_vertical_hidden=st["cond_v_hidden"],
                prev_h_hidden=st["prev_h_hidden"],
                prev_v_hidden=st["prev_v_hidden"],
                step_positions=step_pos,
                prev_positions=st["prev_positions"],
                height=H, width=W,
                cond_h_logits_pre=cond_h_pre,
                cond_v_logits_pre=cond_v_pre,
                h_prev_logits_pre=h_prev_pre,
                v_prev_logits_pre=v_prev_pre,
            )
        with _nvtx("diagonal_sample_packed"):
            sampled = sample_diagonal_logits(
                step_logits, st,
                dump_ctx={
                    "parent_id": p["parent_id"],
                    "diag_idx": int(next_diag_idx),
                    "step_positions": step_pos,
                } if LOGITS_DUMP.is_active() else None,
            )
        st["next_pre_append_tokens"] = sampled.tolist()
        record_sample_summary(st, p["parent_id"], next_diag_idx, sampled)
        st["diag_idx_to_predict"] = next_diag_idx + 1
        st["phase"] = f"diagonal_{next_diag_idx}_pending_forward"
        st["diag_log"].append({
            "diag_just": next_diag_idx - 1,
            "diag_predicted": next_diag_idx,
            "n_just": n,
            "n_predicted": len(next_pos_list),
            "tokens": sampled.tolist(),
        })
        if len(next_pos_list) > 1:
            any_force_non_causal = True

    return any_force_non_causal


def _step_one_parent(self_m, ext, st,
                     last_aux_full, last_final_full,
                     cond_slice, uncond_slice, device,
                     cond_prefill_done: bool = True,
                     uncond_prefill_done: bool = True):
    """Run one diagonal step for a single parent. Mutates st in place.

    Prefill anchors are captured by `_capture_prefill_anchors`
    BEFORE this dispatcher runs; this function should only be invoked once
    the parent's phase is already `diagonal_*_pending_forward` (i.e. decode).
    The prefill branch is retained as a defensive fallback.
    """
    H, W = st["height"], st["width"]
    cond_lo, cond_hi = cond_slice
    uncond_lo, uncond_hi = uncond_slice
    N_s_actual = cond_hi - cond_lo
    uncond_N_s_actual = uncond_hi - uncond_lo

    # If parent is already all_done, this step is just the post-EOI forward
    # — nothing to compute. (Driver loop breaks after.)
    if st["phase"] == "all_done":
        return

    # ---- prefill: capture cond_h/v_hidden, predict diagonal 0 ----
    if st["phase"] == "prefill_pending":
        # Under prefill-decoupled scheduling, anchor capture is
        # the caller's responsibility (see _capture_prefill_anchors). This
        # branch is unreachable in the new path but kept for safety so that
        # if ever called through the fallback path we still produce a correct anchor.
        if N_s_actual == 0:
            return
        if not (cond_prefill_done and uncond_prefill_done):
            return
        cond_aux_last = last_aux_full[cond_hi - 1:cond_hi].clone()
        cond_final_last = last_final_full[cond_hi - 1:cond_hi].clone()
        uncond_aux_last = last_aux_full[uncond_hi - 1:uncond_hi].clone()
        uncond_final_last = last_final_full[uncond_hi - 1:uncond_hi].clone()
        v_anchor = torch.stack([cond_aux_last, uncond_aux_last], dim=0)
        h_anchor = torch.stack([cond_final_last, uncond_final_last], dim=0)
        st["cond_h_hidden"] = h_anchor
        st["cond_v_hidden"] = v_anchor
        st["vertical_kv_cache"] = None

        step_pos = diagonal_tensor(0, H, W, device)
        prev_pos = torch.empty(0, device=device, dtype=torch.long)
        with _nvtx("compute_step_logits_d0"):
            step_logits = ext.compute_step_logits_from_prev(
                cond_horizontal_hidden=h_anchor,
                cond_vertical_hidden=v_anchor,
                prev_h_hidden=None, prev_v_hidden=None,
                step_positions=step_pos, prev_positions=prev_pos,
                height=H, width=W,
            )
        # Find parent_id for dump (lookup states dict).
        _dump_pid_d0 = None
        if LOGITS_DUMP.is_active():
            for _pid, _pst in self_m._unis_flashar_states.items():
                if _pst is st:
                    _dump_pid_d0 = _pid
                    break
        with _nvtx("diagonal_sample_d0"):
            sampled = sample_diagonal_logits(
                step_logits, st,
                dump_ctx={
                    "parent_id": _dump_pid_d0,
                    "diag_idx": 0,
                    "step_positions": step_pos,
                } if _dump_pid_d0 is not None else None,
            )
        st["next_pre_append_tokens"] = sampled.tolist()
        record_sample_summary(st, _dump_pid_d0, 0, sampled)
        st["diag_idx_to_predict"] = 1
        st["phase"] = "diagonal_0_pending_forward"
        st["diag_log"].append({
            "phase": "prefill_done", "predicted_tokens": sampled.tolist(),
        })
        return

    # ---- decode: just-forwarded diagonal_k, predict diagonal_k+1 ----
    just_diag = st["diag_idx_to_predict"] - 1
    just_pos_list = diagonal_list(just_diag, H, W)
    N_just = len(just_pos_list)
    if uncond_N_s_actual != N_s_actual:
        raise RuntimeError(
            f"FlashAR CFG pair token mismatch: diag={just_diag}, "
            f"cond_tokens={N_s_actual}, uncond_tokens={uncond_N_s_actual}")
    if N_just != N_s_actual:
        raise RuntimeError(
            f"FlashAR diagonal token mismatch: diag={just_diag}, "
            f"expected={N_just}, forward_tokens={N_s_actual}, "
            f"phase={st['phase']}")

    cond_final = last_final_full[cond_lo:cond_hi].unsqueeze(0)
    uncond_final = last_final_full[uncond_lo:uncond_hi].unsqueeze(0)
    cur_h_hidden = torch.cat([cond_final, uncond_final], dim=0)

    # vertical_block already ran inside patched_outer
    # (i.e. inside vLLM's set_forward_context block, where vLLM Attention is
    # callable). Read its output for this parent.
    parent_id = None
    for pid, pst in self_m._unis_flashar_states.items():
        if pst is st:
            parent_id = pid
            break
    vouts = getattr(self_m, "_latest_vertical_outputs", {}) or {}
    cur_v_hidden = vouts.get(parent_id)
    if cur_v_hidden is None:
        raise RuntimeError(
            f"vertical output missing for parent {parent_id} "
            f"(diag {just_diag}, visual_this_step={N_just}); "
            f"available parents in vouts: {list(vouts.keys())}; "
            f"phase before this step: {st['phase']}")
    st["prev_h_hidden"] = cur_h_hidden
    st["prev_v_hidden"] = cur_v_hidden
    st["prev_positions"] = diagonal_tensor(just_diag, H, W, device)

    next_diag_idx = st["diag_idx_to_predict"]
    if next_diag_idx > H + W - 2:
        st["next_pre_append_tokens"] = [st["eoi_token_id"]]
        record_sample_summary(
            st, parent_id, next_diag_idx, st["next_pre_append_tokens"])
        st["phase"] = "all_done"
        st["diag_log"].append({"phase": "all_done"})
        return

    next_pos_list = diagonal_list(next_diag_idx, H, W)
    step_pos = diagonal_tensor(next_diag_idx, H, W, device)
    with _nvtx("compute_step_logits"):
        step_logits = ext.compute_step_logits_from_prev(
            cond_horizontal_hidden=st["cond_h_hidden"],
            cond_vertical_hidden=st["cond_v_hidden"],
            prev_h_hidden=cur_h_hidden,
            prev_v_hidden=cur_v_hidden,
            step_positions=step_pos,
            prev_positions=st["prev_positions"],
            height=H, width=W,
        )
    with _nvtx("diagonal_sample"):
        sampled = sample_diagonal_logits(
            step_logits, st,
            dump_ctx={
                "parent_id": parent_id,
                "diag_idx": int(next_diag_idx),
                "step_positions": step_pos,
            } if (LOGITS_DUMP.is_active() and parent_id is not None) else None,
        )
    st["next_pre_append_tokens"] = sampled.tolist()
    record_sample_summary(st, parent_id, next_diag_idx, sampled)
    st["diag_idx_to_predict"] = next_diag_idx + 1
    st["phase"] = f"diagonal_{next_diag_idx}_pending_forward"
    st["diag_log"].append({
        "diag_just": just_diag,
        "diag_predicted": next_diag_idx,
        "n_just": N_just,
        "n_predicted": len(next_pos_list),
        "tokens": sampled.tolist(),
    })


def _dummy_logits(hidden_states):
    """Return shape (B, V) of safe zeros. Sampler is skipped so values don't
    matter, but vLLM internals may still inspect shape/dtype."""
    B = hidden_states.shape[0]
    V = 282926  # FlashAR vocab; could read from m.config but hardcode is fine
    return torch.zeros((B, V), dtype=hidden_states.dtype, device=hidden_states.device)


def _gather_running_pair(scheduler, parent_request_id):
    """Return list of [cond_req, uncond_req] (order: index 0 = cond, 1 = uncond).

    Xiaomi-Robotics-U0 hybrid mode child request_id format: `{idx}_cfg_{parent}` where
    idx 0 = cond, idx 1 = uncond. The patched batch_scheduler enforces that
    they appear consecutively in `running`.

    NOTE: must match EXACTLY against `{idx}_cfg_{parent}` — using endswith
    on the parent suffix would match e.g. `0_cfg_unis_flashar_11` when looking
    for `unis_flashar_1`, which silently aliases distinct parents whose ids
    differ only by a digit. This bit us on N=128 multi-wave dispatch.
    """
    target_cond = f"0_cfg_{parent_request_id}"
    target_uncond = f"1_cfg_{parent_request_id}"
    matches = []
    for req in scheduler.running:
        if req.request_id == target_cond or req.request_id == target_uncond:
            matches.append(req)
    if len(matches) >= 2:
        matches.sort(key=lambda r: int(r.request_id.split("_cfg_")[0]))
        return matches[:2]
    return None


def _find_cond_request(scheduler, parent_request_id):
    """Find the cond request (idx == 0)."""
    target = f"0_cfg_{parent_request_id}"
    for req in list(scheduler.running) + list(scheduler.waiting):
        if req.request_id == target:
            return req
    requests_dict = getattr(scheduler, "requests", None)
    if requests_dict:
        cond_id = f"0_cfg_{parent_request_id}"
        if cond_id in requests_dict:
            return requests_dict[cond_id]
    return None


def _find_scheduler_request(scheduler, request_id: str):
    """Find a child Request anywhere the vLLM scheduler may hold it.

    FlashAR owns the visual-token sampling, but vLLM still owns request
    residency. Under chunked prefill / tight token budgets a CFG child can be
    present but unscheduled in a given step; the next sampled diagonal must be
    appended to the Request object before the scheduler next considers it,
    regardless of whether it currently lives in running or waiting.
    """
    for req in list(scheduler.running) + list(scheduler.waiting):
        if req.request_id == request_id:
            return req
    requests_dict = getattr(scheduler, "requests", None)
    if requests_dict and request_id in requests_dict:
        return requests_dict[request_id]
    return None


def worker_get_diagonal_states(self) -> dict:
    """Return per-parent state summary: parent_id -> {phase, tokens, diag_idx}.

    The driver loop iterates this snapshot and acts on each parent
    independently.
    """
    m = self.model_runner.model
    states = getattr(m, "_unis_flashar_states", None) or {}
    out = {}
    for parent_id, st in states.items():
        out[parent_id] = {
            "next_pre_append_tokens": list(st.get("next_pre_append_tokens", [])),
            "phase": st.get("phase"),
            "diag_idx_to_predict": st.get("diag_idx_to_predict"),
        }
    return {"ok": True, "states": out}


def worker_clear_diagonal_next_tokens(self, parent_id: str) -> dict:
    """Clear a parent's pending sampled-token buffer after driver append."""
    m = self.model_runner.model
    states = getattr(m, "_unis_flashar_states", None) or {}
    st = states.get(parent_id)
    if st is None:
        return {"ok": True, "absent": True}
    st["next_pre_append_tokens"] = []
    return {"ok": True}


def worker_get_tp_rank_state_summary(self) -> dict:
    """Return a compact per-rank FlashAR state summary for TP smoke tests."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    m = self.model_runner.model
    states = getattr(m, "_unis_flashar_states", None) or {}
    return {
        "ok": True,
        "tp_rank": int(get_tensor_model_parallel_rank()),
        "tp_size": int(get_tensor_model_parallel_world_size()),
        "n_active_parents": len(states),
        "parents": {
            parent_id: {
                "phase": st.get("phase"),
                "diag_idx_to_predict": st.get("diag_idx_to_predict"),
                "last_sample": st.get("_last_sample_summary"),
            } for parent_id, st in states.items()
        },
    }



# ---------------------------------------------------------------- driver side


@dataclass
class UNISFlashARVLLMConfig:
    backbone_dir: str
    tokenizer_dir: str
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    seed: int = 42
    enable_prefix_caching: bool = False
    max_num_seqs: int = 16
    # max_num_seqs=16 = up to 8 parents x 2 child (cond+uncond CFG pair).
    # Knobs surfaced for verification work:
    enable_chunked_prefill: bool = True
    # vLLM v1 ALWAYS sets enable_chunked_prefill=True for non-pooling runners
    # (regardless of what we pass), so this flag is effectively a hint for
    # `max_num_batched_tokens` sizing — short prompts are still kept in 1
    # chunk under the default 4096 budget.
    max_num_batched_tokens: int = 4096
    enable_log_stats: bool = False
    strict_visual_tokens: bool = True
    # Set True to enable scheduler.kv_cache_manager.prefix_cache_stats so
    # tools/verify_prefix_cache.py can read hit / query counters.


def build_unis_tokenizer(tokenizer_dir: str):
    from xr_u0_ar.hub_paths import resolve_tokenizer_path

    tokenizer_dir = resolve_tokenizer_path(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        special_tokens_file=os.path.join(tokenizer_dir, "unis_vision_tokens.txt"),
        trust_remote_code=True,
    )
    tokenizer.bos_token = "<|extra_203|>"
    tokenizer.eos_token = "<|extra_204|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.eof_token = "<|extra_201|>"
    tokenizer.tms_token = "<|extra_202|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"
    tokenizer.bss_token = "<|extra_100|>"
    tokenizer.ess_token = "<|extra_101|>"
    tokenizer.bog_token = "<|extra_60|>"
    tokenizer.eog_token = "<|extra_61|>"
    tokenizer.boc_token = "<|extra_50|>"
    tokenizer.eoc_token = "<|extra_51|>"
    return tokenizer


class UNISFlashARVLLMWrapper:
    """Driver for batched Xiaomi-Robotics-U0-FlashAR diagonal decoding on vLLM."""

    def __init__(self, cfg: UNISFlashARVLLMConfig):
        self.cfg = cfg
        self.tokenizer = build_unis_tokenizer(cfg.tokenizer_dir)

        # Register the model class and install the
        # vertical KVCacheGroup spec patch BEFORE LLM construction. The
        # backbone dir's config.json must list architectures=
        # ["UNISFlashARForCausalLM"] AND have a weight_map entry for the
        # FlashAR extras (heads, gates, vertical_block) — see
        # the public checkpoint layout for the required weight names.
        from vllm import ModelRegistry
        from xr_u0_flashar.vllm.model import install_vertical_spec_patch
        _ensure_vllm_plugin_entrypoint()
        install_vertical_spec_patch()
        ModelRegistry.register_model(
            "UNISFlashARForCausalLM",
            "xr_u0_flashar.vllm.model:UNISFlashARForCausalLM")

        resolution_chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*"]
        resolution_map = {self.tokenizer.encode(c)[0]: c for c in resolution_chars}

        self.llm = LLM(
            model=cfg.backbone_dir,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_seqs=cfg.max_num_seqs,
            # vLLM v1 forces enable_chunked_prefill=True for non-pooling
            # models regardless of what we pass; the flag here is a hint.
            # The FlashAR prefill hook now handles the multi-chunk case
            # (anchor capture deferred until prefill is fully complete).
            enable_chunked_prefill=cfg.enable_chunked_prefill,
            # Prefix caching can be flipped via cfg.enable_prefix_caching for
            # determinism testing. Keep the default OFF for FlashAR correctness;
            # cached prompt KV can corrupt the early diagonal anchor.
            enable_prefix_caching=cfg.enable_prefix_caching,
            disable_log_stats=not cfg.enable_log_stats,
            max_model_len=cfg.max_model_len,
            enforce_eager=True,
            seed=cfg.seed,
            scheduler_cls="vllm.v1.core.sched.batch_scheduler.Scheduler",
            additional_config={
                "boi_token_id": self.tokenizer.encode("<|image start|>")[0],
                "soi_token_id": self.tokenizer.encode("<|image token|>")[0],
                "eol_token_id": self.tokenizer.encode("<|extra_200|>")[0],
                "eoi_token_id": self.tokenizer.encode("<|image end|>")[0],
                "resolution_map": resolution_map,
            },
        )
        self.llm.set_tokenizer(self.tokenizer)

        self._install_hooks()

    def _install_hooks(self):
        info = self.llm.llm_engine.collective_rpc(worker_install_hooks)
        if not _all_rpc_ok(info):
            raise RuntimeError(f"worker hook install failed: {info}")
        print(f"[wrapper] worker hooks installed: {info}")
        # Capture query metadata and enable sample-skip/non-causal toggles
        self.llm.llm_engine.collective_rpc(worker_install_query_meta_capture)
        self.llm.llm_engine.collective_rpc(worker_install_sample_skip)
        self.llm.llm_engine.collective_rpc(worker_install_non_causal_hook)
        # Install the diagonal compute_logits hook once; per-parent state is
        # registered for each generate call.
        rpc_info = self.llm.llm_engine.collective_rpc(worker_install_diagonal_decode)
        if not _all_rpc_ok(rpc_info):
            raise RuntimeError(f"diagonal decode hook install failed: {rpc_info}")

    # ---------------------------------------- diagonal visual-token decoding

    def generate_diagonal(
        self,
        prompt: str,
        uncond_prompt: str,
        height: int = 32,
        width: int = 32,
        temperature: float = 1.0,
        top_k: int = 5120,
        top_p: float = 1.0,
        cfg_scale: float = 3.0,
    ) -> dict:
        """Single-prompt convenience wrapper around ``generate_diagonal_batch``."""
        results = self.generate_diagonal_batch(
            prompts=[prompt], uncond_prompts=[uncond_prompt],
            heights=[height], widths=[width],
            temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
        )
        return results[0]

    def generate_diagonal_batch(
        self,
        prompts: "list[str]",
        uncond_prompts: "list[str]",
        heights: "list[int]",
        widths: "list[int]",
        temperature: float = 1.0,
        top_k: int = 5120,
        top_p: float = 1.0,
        cfg_scale: float = 3.0,
        seeds: "Optional[list[int]]" = None,
    ) -> "list[dict]":
        """Run batched diagonal visual-token decoding.

        Submits N parents up front, drives engine_core.step() until all reach
        all_done. Each step: for every running parent, fetch its
        next_pre_append_tokens, append to BOTH child Requests of that parent.

        Requires InprocClient (VLLM_ENABLE_V1_MULTIPROCESSING=0) because we
        directly mutate Request._all_token_ids in the engine.
        """
        N = len(prompts)
        assert len(uncond_prompts) == N == len(heights) == len(widths), (
            "prompts/uncond_prompts/heights/widths must have the same length")
        if seeds is not None:
            assert len(seeds) == N, "seeds must have the same length as prompts"

        engine_core_client = self.llm.llm_engine.engine_core
        inner_engine = getattr(engine_core_client, "engine_core", None)
        if inner_engine is None:
            raise RuntimeError(
                "Xiaomi-Robotics-U0-FlashAR vLLM requires InprocClient "
                "(set VLLM_ENABLE_V1_MULTIPROCESSING=0)")
        scheduler = inner_engine.scheduler

        # Bypass collective_rpc for the
        # per-step states read. In InprocClient mode the worker is in-process,
        # so we hold a direct reference to model._unis_flashar_states (same dict
        # the worker mutates). Saves a round-trip and a dict copy per step.
        try:
            _direct_worker = inner_engine.model_executor.driver_worker.worker
            _direct_model = _direct_worker.model_runner.model
        except AttributeError:
            _direct_model = None

        visual_token_offset = self.tokenizer.encode("<|image end|>")[0] + 1
        eoi_token_id = self.tokenizer.encode("<|image end|>")[0]

        # ------------------------------------------------------------ submit
        if not hasattr(self, "_diagonal_counter"):
            self._diagonal_counter = 0
        parents = []  # list of dict per request: {parent_id, H, W, n_visual, n_diag}
        for i in range(N):
            self._diagonal_counter += 1
            parent_id = f"unis_flashar_{self._diagonal_counter}"
            H, W = heights[i], widths[i]
            prompt_ids = self.tokenizer.encode(prompts[i], add_special_tokens=False)
            uncond_ids = self.tokenizer.encode(uncond_prompts[i], add_special_tokens=False)
            n_visual = H * W
            n_diag = H + W - 1

            # Caller can pin per-prompt seeds so serial and concurrent runs
            # are byte-comparable. Default uses a counter-derived seed.
            seed = seeds[i] if seeds is not None else (self.cfg.seed + self._diagonal_counter)
            reg = self.llm.llm_engine.collective_rpc(
                worker_register_diagonal_request,
                args=(parent_id, H, W, len(prompt_ids), len(uncond_ids),
                      visual_token_offset, eoi_token_id,
                      cfg_scale, temperature, top_k, top_p,
                      seed),
            )
            if not _all_rpc_ok(reg):
                raise RuntimeError(f"register parent {parent_id} failed: {reg}")

            sampling_params = SamplingParams(
                max_tokens=n_visual + 32,
                temperature=temperature,
                top_k=top_k if top_k > 0 else -1,
                top_p=top_p,
                extra_args={"guidance_scale": cfg_scale},
            )
            self.llm.llm_engine.add_request(
                request_id=parent_id,
                prompt={
                    "prompt_token_ids": prompt_ids,
                    "uncond_prompt_token_ids": uncond_ids,
                },
                params=sampling_params,
            )
            # Register this parent for logits dumping. The dump
            # store is gated by XR_U0_FLASHAR_DUMP_LOGITS=<dir>; when unset,
            # LOGITS_DUMP.is_active() is False and register is a no-op
            # downstream. We always store the manifest so a follow-up
            # XR_U0_FLASHAR_DUMP_LOGITS run on the same generate path produces
            # a self-describing dump.
            if LOGITS_DUMP.is_active():
                LOGITS_DUMP.register(parent_id, {
                    "parent_id": parent_id,
                    "prompt": prompts[i],
                    "uncond_prompt": uncond_prompts[i],
                    "prompt_ids": list(map(int, prompt_ids)),
                    "uncond_ids": list(map(int, uncond_ids)),
                    "height": int(H), "width": int(W),
                    "cfg_scale": float(cfg_scale),
                    "temperature": float(temperature),
                    "top_k": int(top_k),
                    "top_p": float(top_p),
                    "seed": int(seed),
                    "visual_token_offset": int(visual_token_offset),
                    "eoi_token_id": int(eoi_token_id),
                })
            parents.append({
                "parent_id": parent_id, "H": H, "W": W,
                "n_visual": n_visual, "n_diag": n_diag,
            })

        # ------------------------------------------------------------ drive
        # Multi-wave dispatch: max_steps formula scales with the number of waves.
        max_diag = max(p["n_diag"] for p in parents)
        concurrent_parents = max(1, self.cfg.max_num_seqs // 2)
        n_waves = (N + concurrent_parents - 1) // concurrent_parents
        # Tight max_num_batched_tokens forces decode pairs to
        # serialize. At peak diagonal, one pair costs 2 * max(H, W) tokens;
        # if that's > budget, we cap per-step concurrency. Bound max_steps by
        # max_pairs_per_step so the loop survives serialized prefill + decode.
        max_grid_side = max(max(p["H"], p["W"]) for p in parents)
        peak_pair_cost = 2 * max_grid_side
        max_pairs_per_step = max(
            1, self.cfg.max_num_batched_tokens // peak_pair_cost)
        decode_serialization = (
            (concurrent_parents + max_pairs_per_step - 1)
            // max_pairs_per_step)
        max_steps = (n_waves * decode_serialization * (max_diag + 6) + 64)
        n_steps = 0
        done_parents = set()
        # Snapshot a parent's token stream the moment it hits all_done, then
        # immediately finish its child reqs so vLLM can schedule the next wave.
        parent_tokens: dict = {}
        from vllm.v1.request import RequestStatus

        # Publish initial vertical meta (zeroed) so the first
        # build() call (= prefill) sees a populated hook.
        self.llm.llm_engine.collective_rpc(worker_publish_initial_vertical_meta)


        while inner_engine.scheduler.has_requests() and n_steps < max_steps:
            with _nvtx(f"driver_step_{n_steps}"):
                with _nvtx("rpc_get_states"):
                    if _direct_model is not None:
                        # Direct attribute read; same dict the worker mutates.
                        # We snapshot phase/tokens here to avoid re-reading
                        # mid-loop in case the engine_step modifies them.
                        raw_states = getattr(_direct_model, "_unis_flashar_states", {}) or {}
                        states = {
                            pid: {
                                "phase": st.get("phase"),
                                "next_pre_append_tokens":
                                    list(st.get("next_pre_append_tokens", [])),
                            } for pid, st in raw_states.items()
                        }
                    else:
                        states_info = self.llm.llm_engine.collective_rpc(
                            worker_get_diagonal_states)[0]
                        states = states_info.get("states", {})

                with _nvtx("append_tokens"):
                    for parent_id, info in states.items():
                        phase_str = info.get("phase", "")
                        tokens_to_append = info.get("next_pre_append_tokens", [])
                        if not tokens_to_append:
                            continue
                        if not (phase_str.endswith("_pending_forward") or phase_str == "all_done"):
                            continue
                        cond_req = _find_scheduler_request(
                            scheduler, f"0_cfg_{parent_id}")
                        uncond_req = _find_scheduler_request(
                            scheduler, f"1_cfg_{parent_id}")
                        if cond_req is None or uncond_req is None:
                            continue
                        ids_int = [int(t) for t in tokens_to_append]
                        cond_req.append_output_token_ids(ids_int)
                        uncond_req.append_output_token_ids(ids_int)
                        # Clear the worker-side buffer immediately: tokens are
                        # now durable in both child Requests, including waiting
                        # children that will be scheduled in a later step.
                        if os.environ.get("FLASHAR_CLEAR_NPAT", "1") != "0":
                            info["next_pre_append_tokens"] = []
                            if _direct_model is not None:
                                raw_st = getattr(
                                    _direct_model, "_unis_flashar_states", {}).get(
                                        parent_id)
                                if raw_st is not None:
                                    raw_st["next_pre_append_tokens"] = []
                            else:
                                self.llm.llm_engine.collective_rpc(
                                    worker_clear_diagonal_next_tokens,
                                    args=(parent_id,))

                with _nvtx("engine_step"):
                    inner_engine.step()
                n_steps += 1

            for parent_id, info in states.items():
                if info.get("phase") == "all_done" and parent_id not in done_parents:
                    # Snapshot tokens BEFORE finishing the child reqs (the
                    # finish path may reset _output_token_ids).
                    cond_req = _find_cond_request(scheduler, parent_id)
                    if cond_req is not None:
                        parent_tokens[parent_id] = list(cond_req._output_token_ids)
                    # Free the child slots so vLLM can pull the next wave.
                    child_ids = [f"0_cfg_{parent_id}", f"1_cfg_{parent_id}"]
                    try:
                        scheduler.finish_requests(
                            child_ids, RequestStatus.FINISHED_ABORTED)
                    except Exception:
                        pass
                    done_parents.add(parent_id)
            if len(done_parents) == N:
                break

        if len(done_parents) < N:
            print(f"[xr-u0-flashar-vllm] WARNING driver exited at n_steps={n_steps}, "
                  f"done={len(done_parents)}/{N}; raise max_num_seqs or chunk")

        # ----------------------------------------------------------- collect
        try:
            tp_rank_summaries = self.llm.llm_engine.collective_rpc(
                worker_get_tp_rank_state_summary)
        except Exception:
            tp_rank_summaries = []
        results = []
        strict_errors = []
        for p in parents:
            parent_id = p["parent_id"]
            H, W, n_visual, n_diag = p["H"], p["W"], p["n_visual"], p["n_diag"]

            # Tokens were snapshotted at all_done; fall back to cond_req lookup
            # only for the (rare) case where snapshot missed.
            token_ids = parent_tokens.get(parent_id)
            if token_ids is None:
                cond_req = _find_cond_request(scheduler, parent_id)
                token_ids = list(cond_req._output_token_ids) if cond_req else []
            visual_tokens_raw = [
                int(t) for t in token_ids if int(t) >= visual_token_offset
            ]
            n_visual_actual = len(visual_tokens_raw)
            token_min = min(visual_tokens_raw) if visual_tokens_raw else None
            token_max = max(visual_tokens_raw) if visual_tokens_raw else None
            warn = None
            strict_error = None
            if n_visual_actual != n_visual:
                warn = (f"got {n_visual_actual} visual tokens "
                        f"(out of {len(token_ids)} total), wanted {n_visual}")
                if self.cfg.strict_visual_tokens:
                    strict_error = RuntimeError(
                        f"FlashAR generation produced invalid visual-token "
                        f"count for {parent_id}: {warn}")
                pad = (
                    visual_tokens_raw[0]
                    if visual_tokens_raw else visual_token_offset
                )
                if n_visual_actual < n_visual:
                    visual_tokens = (
                        visual_tokens_raw + [pad] * (n_visual - n_visual_actual))
                else:
                    visual_tokens = visual_tokens_raw
            else:
                visual_tokens = visual_tokens_raw
            visual_tokens = visual_tokens[:n_visual]
            grid_flat = torch.zeros(H * W, dtype=torch.long)
            cursor = 0
            for d in range(n_diag):
                for flat_pos in diagonal_list(d, H, W):
                    grid_flat[flat_pos] = visual_tokens[cursor]
                    cursor += 1
                    if cursor >= n_visual:
                        break
                if cursor >= n_visual:
                    break
            grid = grid_flat.view(H, W)

            # Cleanup: child requests + parent state.
            child_ids = [f"0_cfg_{parent_id}", f"1_cfg_{parent_id}"]
            try:
                scheduler.finish_requests(child_ids, RequestStatus.FINISHED_ABORTED)
            except Exception:
                pass
            try:
                self.llm.llm_engine.abort_request([parent_id])
            except Exception:
                pass
            self.llm.llm_engine.collective_rpc(
                worker_unregister_diagonal_request, args=(parent_id,))
            if strict_error is not None:
                strict_errors.append(strict_error)
                continue

            results.append({
                "parent_id": parent_id,
                "grid": grid,
                "output_token_ids": token_ids,
                "visual_tokens": visual_tokens,
                "n_steps": n_steps,
                "height": H,
                "width": W,
                "n_visual_expected": n_visual,
                "n_visual_actual": n_visual_actual,
                "visual_token_offset": visual_token_offset,
                "token_min": token_min,
                "token_max": token_max,
                "warn": warn,
                "tp_rank_summaries": tp_rank_summaries,
            })

        # Flush logits dump to disk. No-op when XR_U0_FLASHAR_DUMP_LOGITS
        # is unset.
        if LOGITS_DUMP.is_active():
            written = LOGITS_DUMP.flush()
            if written:
                print(f"[wrapper] XR_U0_FLASHAR_DUMP_LOGITS wrote {len(written)} parent dumps "
                      f"under {LOGITS_DUMP.dump_dir}")

        if strict_errors:
            raise strict_errors[0]

        return results

"""Packed visual-logit sampling for Xiaomi-Robotics-U0-FlashAR diagonal decoding."""

from __future__ import annotations

from typing import Optional

import torch

from xr_u0_flashar.vllm.debug_dump import LOGITS_DUMP


def sample_diagonal_logits(step_logits, state, dump_ctx: Optional[dict] = None):
    """Apply CFG, visual-token masking, top-k/top-p, then sample one diagonal."""
    cfg_scale = state["cfg_scale"]
    visual_token_offset = state["visual_token_offset"]
    temperature = state["temperature"]
    top_k = state["top_k"]
    top_p = state["top_p"]
    generator = state["generator"]

    cond_logits = step_logits[0]
    uncond_logits = step_logits[1]
    if cfg_scale != 1.0:
        logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
    else:
        logits = cond_logits

    logits[:, :visual_token_offset] = float("-inf")
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        topk_vals, _ = torch.topk(logits, top_k, dim=-1)
        kth = topk_vals[:, -1:].expand_as(logits)
        logits = torch.where(
            logits < kth,
            torch.full_like(logits, float("-inf")),
            logits,
        )
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_mask = cum_probs > top_p
        sorted_mask[:, 0] = False
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1,
            index=sorted_idx,
            src=sorted_mask,
        )
        logits.masked_fill_(mask, float("-inf"))

    if temperature <= 0.0:
        sampled = logits.argmax(dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probs,
            num_samples=1,
            generator=generator,
        ).squeeze(-1)

    if dump_ctx is not None and LOGITS_DUMP.is_active():
        final_probs = torch.softmax(logits, dim=-1)
        LOGITS_DUMP.record_step(
            parent_id=dump_ctx["parent_id"],
            diag_idx=dump_ctx["diag_idx"],
            step_positions=dump_ctx["step_positions"],
            cond_logits=cond_logits,
            uncond_logits=uncond_logits,
            sampled=sampled,
            final_probs=final_probs,
        )
    return sampled


def record_sample_summary(state: dict, parent_id: Optional[str], diag_idx: int, sampled) -> None:
    """Attach a compact last-sample summary to a per-parent runtime state."""
    if isinstance(sampled, torch.Tensor):
        tokens = sampled.detach().to("cpu", non_blocking=False).tolist()
    else:
        tokens = list(sampled)
    state["_last_sample_summary"] = {
        "parent_id": parent_id,
        "diag_idx": int(diag_idx),
        "n_tokens": len(tokens),
        "tokens_head": [int(token) for token in tokens[:8]],
    }

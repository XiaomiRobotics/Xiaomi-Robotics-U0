"""Complete interleaved text/image decoding for Sequence checkpoints."""
from __future__ import annotations

from typing import Any
import torch

IMG_ID = 151851
BOI_ID = 151852
EOI_ID = 151853
EOL_ID = 151846
BSS_ID = 151746
ESS_ID = 151747
EOS_ID = 151850
VISUAL_TOKEN_OFFSET = 151854
VISUAL_TOKEN_END = 282926

def _decode_shape_with_tokenizer(shape_tokens: list[int], tokenizer) -> tuple[int, int]:
    decoded = tokenizer.detokenize(shape_tokens) if hasattr(tokenizer, "detokenize") else tokenizer.decode(shape_tokens)
    text = "".join(str(decoded).split())
    parts = text.split("*")
    if len(parts) == 3:
        parts = parts[1:]
    if len(parts) != 2:
        raw = "".join(chr(tok) for tok in shape_tokens if 48 <= int(tok) <= 57 or int(tok) == 42)
        parts = raw.split("*", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse image shape from token ids: {shape_tokens!r}")
    height, width = (int(part) for part in parts)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape {height}*{width}")
    return height, width

def _filter_top_k_top_p(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits.clone()
    if int(top_k) > 0 and int(top_k) < int(logits.size(-1)):
        threshold = torch.topk(logits, int(top_k), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)
    if float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits.float(), dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter_remove = torch.zeros_like(remove).scatter(dim=-1, index=sorted_indices, src=remove)
        logits = logits.masked_fill(scatter_remove, torch.finfo(logits.dtype).min)
    return logits

def _sample_nonvisual_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
) -> torch.Tensor:
    scoped = logits[:, :VISUAL_TOKEN_OFFSET]
    if greedy:
        return torch.argmax(scoped, dim=-1)
    if float(temperature) <= 0:
        raise ValueError("temperature must be > 0 unless --greedy is used")
    filtered = _filter_top_k_top_p(
        scoped / float(temperature), top_k=int(top_k), top_p=float(top_p)
    )
    return torch.multinomial(torch.softmax(filtered.float(), dim=-1), num_samples=1).squeeze(-1)

def _append_plain_ar_token(
    model,
    *,
    token_id: int,
    position: int,
    past_key_values,
    device: torch.device,
):
    token = torch.tensor([[int(token_id)]], device=device, dtype=torch.long)
    position_ids = torch.tensor([[int(position)]], device=device, dtype=torch.long)
    outputs = model(
        input_ids=token,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    return outputs.past_key_values, outputs.logits[:, -1, :]

@torch.inference_mode()
def generate_sequence(
    model,
    tokenizer,
    *,
    prefix_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
    cfg_scale: float,
    uncond_context_ids: torch.Tensor | None,
    uncond_context_position_ids: torch.Tensor | None,
    max_images: int,
    max_image_tokens: int,
    max_header_tokens: int,
    control_greedy: bool,
    bss_in_prefix: bool = False,
    cfg_scale_end: float | None = None,
    cfg_decay_images: int = 1,
    quick_validation: bool = False,
) -> dict[str, Any]:
    prefix = prefix_ids.view(1, -1).to(device=device, dtype=torch.long)
    if int(prefix.size(1)) == 0:
        raise ValueError("full_span decoding requires a non-empty context before BSS")
    if bool(bss_in_prefix) and int(prefix[0, -1].item()) != BSS_ID:
        raise ValueError(
            f"full_span bss_in_prefix requires prefix to end with BSS={BSS_ID}, "
            f"got {int(prefix[0, -1].item())}"
        )
    prefix_positions = torch.arange(
        int(prefix.size(1)), device=device, dtype=torch.long
    ).view(1, -1)
    outputs = model(
        input_ids=prefix,
        position_ids=prefix_positions,
        use_cache=True,
        return_dict=True,
    )
    cond_past = outputs.past_key_values
    cond_logits = outputs.logits[:, -1, :]

    if cfg_scale_end is None:
        cfg_scale_end = float(cfg_scale)
    if int(cfg_decay_images) <= 0:
        raise ValueError("cfg_decay_images must be positive")
    if not 1.0 <= float(cfg_scale_end) <= float(cfg_scale):
        raise ValueError("cfg_scale_end must be in [1.0, cfg_scale]")
    if float(cfg_scale_end) != float(cfg_scale) and int(cfg_decay_images) < 2:
        raise ValueError("decaying CFG requires cfg_decay_images >= 2")
    use_cfg = max(float(cfg_scale), float(cfg_scale_end)) != 1.0
    uncond_past = None
    uncond_logits = None
    if use_cfg:
        if uncond_context_ids is None or int(uncond_context_ids.numel()) == 0:
            raise ValueError("full_span CFG requires full_span_uncond_context_ids")
        if uncond_context_position_ids is None:
            raise ValueError("full_span CFG requires full_span_uncond_context_position_ids")
        uncond = uncond_context_ids.view(1, -1).to(device=device, dtype=torch.long)
        uncond_positions = uncond_context_position_ids.view(1, -1).to(
            device=device, dtype=torch.long
        )
        if int(uncond.numel()) != int(uncond_positions.numel()):
            raise ValueError("full_span unconditional ids/position ids length mismatch")
        if not bool(torch.all(uncond_positions[:, 1:] > uncond_positions[:, :-1])):
            raise ValueError("full_span unconditional position ids must be strictly increasing")
        uncond_outputs = model(
            input_ids=uncond,
            position_ids=uncond_positions,
            use_cache=True,
            return_dict=True,
        )
        uncond_past = uncond_outputs.past_key_values
        uncond_logits = uncond_outputs.logits[:, -1, :]

    generated: list[int] = []
    image_grids: list[torch.Tensor] = []
    image_shapes: list[list[int]] = []
    special_events: list[dict[str, int]] = []
    state = "outside" if bool(bss_in_prefix) else "need_bss"
    header_ids: list[int] = []
    current_rows: list[list[int]] = []
    height = width = row = col = 0
    sync_uncond_image = False
    seen_bss = bool(bss_in_prefix)
    seen_ess = False
    stop_reason = "max_new_tokens"
    active_cfg_scale = float(cfg_scale)

    for gen_idx in range(int(max_new_tokens)):
        is_visual_step = state == "image" and col < width
        if is_visual_step:
            visual_end = min(int(cond_logits.size(-1)), VISUAL_TOKEN_END)
            cond_visual = cond_logits[:, VISUAL_TOKEN_OFFSET:visual_end]
            if use_cfg:
                if uncond_logits is None:
                    raise RuntimeError("full_span CFG lost unconditional logits inside image payload")
                uncond_visual = uncond_logits[:, VISUAL_TOKEN_OFFSET:visual_end]
                scoped_logits = uncond_visual + float(active_cfg_scale) * (
                    cond_visual - uncond_visual
                )
            else:
                scoped_logits = cond_visual
            if bool(greedy):
                next_id = int(torch.argmax(scoped_logits, dim=-1).item()) + VISUAL_TOKEN_OFFSET
            else:
                filtered = _filter_top_k_top_p(
                    scoped_logits / float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                )
                next_id = int(
                    torch.multinomial(torch.softmax(filtered.float(), dim=-1), 1).item()
                ) + VISUAL_TOKEN_OFFSET
        else:
            next_id = int(
                _sample_nonvisual_logits(
                    cond_logits,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    greedy=bool(control_greedy),
                ).item()
            )

        token_position = int(prefix.size(1)) + gen_idx
        generated.append(next_id)
        if next_id in {BSS_ID, ESS_ID, EOS_ID, BOI_ID, IMG_ID, EOI_ID, EOL_ID}:
            special_events.append(
                {"generated_index": gen_idx, "absolute_position": token_position, "token_id": next_id}
            )

        sync_this_token = next_id == BSS_ID or next_id == BOI_ID or sync_uncond_image

        if state == "need_bss":
            if next_id != BSS_ID:
                stop_reason = f"expected_bss_got_{next_id}"
            else:
                seen_bss = True
                state = "outside"
        elif state == "outside":
            if next_id == BOI_ID:
                if len(image_grids) >= int(max_images):
                    stop_reason = "max_images_exceeded"
                else:
                    header_ids = []
                    sync_uncond_image = True
                    state = "header"
            elif next_id == ESS_ID:
                seen_ess = True
                state = "need_eos"
            elif next_id == EOS_ID:
                stop_reason = "eos_before_ess"
            elif next_id in {BSS_ID, IMG_ID, EOI_ID, EOL_ID}:
                stop_reason = f"unexpected_outside_image_token_{next_id}"
        elif state == "header":
            if next_id == IMG_ID:
                try:
                    height, width = _decode_shape_with_tokenizer(header_ids, tokenizer)
                except Exception as exc:
                    stop_reason = f"invalid_shape:{exc}"
                else:
                    if height * width > int(max_image_tokens):
                        stop_reason = f"image_too_large:{height}x{width}"
                    else:
                        image_shapes.append([height, width])
                        if int(cfg_decay_images) <= 1 or float(cfg_scale_end) == float(cfg_scale):
                            active_cfg_scale = float(cfg_scale)
                        else:
                            image_index = len(image_grids)
                            progress = min(image_index, int(cfg_decay_images) - 1) / float(int(cfg_decay_images) - 1)
                            active_cfg_scale = float(cfg_scale) + (
                                float(cfg_scale_end) - float(cfg_scale)
                            ) * progress
                        current_rows = [[] for _ in range(height)]
                        row = col = 0
                        state = "image"
            elif next_id in {BSS_ID, ESS_ID, EOS_ID, BOI_ID, EOI_ID, EOL_ID}:
                stop_reason = f"invalid_header_token_{next_id}"
            else:
                header_ids.append(next_id)
                if len(header_ids) > int(max_header_tokens):
                    stop_reason = "max_header_tokens_exceeded"
        elif state == "image":
            if col < width:
                current_rows[row].append(next_id)
                col += 1
            else:
                expected = EOI_ID if row == height - 1 else EOL_ID
                if next_id != expected:
                    stop_reason = f"expected_image_control_{expected}_got_{next_id}"
                elif expected == EOL_ID:
                    row += 1
                    col = 0
                else:
                    image_grids.append(torch.tensor(current_rows, dtype=torch.long))
                    state = "outside"
                    sync_uncond_image = False
                    if quick_validation:
                        stop_reason = "quick_image_complete"
        elif state == "need_eos":
            if next_id == EOS_ID:
                stop_reason = "eos"
            else:
                stop_reason = f"expected_eos_got_{next_id}"

        terminal = stop_reason != "max_new_tokens"
        if terminal:
            break

        cond_past, cond_logits = _append_plain_ar_token(
            model,
            token_id=next_id,
            position=token_position,
            past_key_values=cond_past,
            device=device,
        )
        if use_cfg and sync_this_token:
            uncond_past, uncond_logits = _append_plain_ar_token(
                model,
                token_id=next_id,
                position=token_position,
                past_key_values=uncond_past,
                device=device,
            )

    format_valid = bool(
        stop_reason == "eos" and seen_bss and seen_ess and len(image_grids) > 0
    )
    return {
        "generated_ids": generated,
        "image_grids": image_grids,
        "image_shapes": image_shapes,
        "format_valid": format_valid,
        "stop_reason": stop_reason,
        "seen_bss": seen_bss,
        "seen_ess": seen_ess,
        "bss_in_prefix": bool(bss_in_prefix),
        "cfg_scale": float(cfg_scale),
        "cfg_scale_end": float(cfg_scale_end),
        "cfg_decay_images": int(cfg_decay_images),
        "special_events": special_events,
    }

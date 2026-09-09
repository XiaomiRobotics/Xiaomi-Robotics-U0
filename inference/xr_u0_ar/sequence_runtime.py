"""Prompt serialization and minimal outputs for Sequence inference."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from .image_tokens import build_image
from .sequence_generation import generate_sequence, BOI_ID, EOI_ID, BSS_ID
from .video_outputs import decode_video_parts, _save_mp4


def encode_text(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def serialize_prompt(tokenizer, prompt, image_strings):
    marker = "<|VIS_PLH|>"
    if prompt.count(marker) != len(image_strings):
        raise ValueError("Sequence prompt placeholders must match input images")
    for image in image_strings:
        prompt = prompt.replace(marker, image, 1)
    prompt = prompt.rstrip("\r\n")
    if not prompt.endswith("<|extra_100|>"):
        raise ValueError("Sequence prompt must end at the generation boundary")
    ids = encode_text(tokenizer, prompt)
    if ids[0] != tokenizer.bos_token_id:
        ids.insert(0, tokenizer.bos_token_id)
    return ids


def unconditional_context(tokenizer, ids):
    if ids[-1] != BSS_ID:
        raise ValueError("Expected generation boundary at end of prefix")
    boundary = len(ids) - 1
    marker = None
    for text in (" Instruction:", "Instruction:"):
        pattern = encode_text(tokenizer, text)
        for pos in range(boundary - len(pattern), -1, -1):
            if ids[pos:pos + len(pattern)] == pattern:
                marker = pos
                break
        if marker is not None:
            break
    positions = list(range(marker if marker is not None else 0))
    cursor = 0
    while cursor < boundary:
        try:
            start = ids.index(BOI_ID, cursor, boundary)
            stop = ids.index(EOI_ID, start, boundary) + 1
        except ValueError:
            break
        if start >= (marker if marker is not None else 0):
            positions.extend(range(start, stop))
        cursor = stop
    positions.append(boundary)
    positions = sorted(set(positions))
    return [ids[pos] for pos in positions], positions


def generation_budget(tokenizer, shapes, prefix_length, context_length, requested=0):
    if requested > 0:
        budget = int(requested)
    else:
        budget = 2  # ESS and EOS; BSS is already in the prompt.
        for height, width in shapes:
            header = encode_text(tokenizer, f"{height}*{width}")
            budget += 1 + len(header) + 1 + height * width + (height - 1) + 1
    remaining = context_length - prefix_length
    if remaining < 1:
        raise ValueError("Sequence prefix exceeds the model context window")
    return min(budget, remaining)


def save_sequence_parts(parts, output_dir, case_id, task, context_paths, fps):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(part["kind"] == "bad_image" for part in parts):
        raise ValueError("Invalid image block in generated sequence")
    frames = [frame for part in parts if part["kind"] == "image" for frame in part["frames"]]
    if not frames:
        raise ValueError("Sequence generated no complete image")
    if task == "interleave_video":
        context = []
        for path in context_paths:
            with Image.open(path) as image:
                context.append(image.convert("RGB").resize(frames[0].size))
        # A single video stream needs a fixed frame size.
        path = output_dir / f"{case_id}.mp4"
        _save_mp4(context + [f.resize(frames[0].size) for f in frames], path, fps)
    else:
        ordered = []
        image_index = 0
        for part in parts:
            if part["kind"] == "text" and part["text"].strip():
                ordered.append({"kind": "text", "text": part["text"].strip()})
            elif part["kind"] == "image":
                for frame in part["frames"]:
                    name = f"{case_id}_{image_index:03d}.png"
                    frame.save(output_dir / name)
                    ordered.append({"kind": "image", "path": name})
                    image_index += 1
        path = output_dir / f"{case_id}.json"
        path.write_text(json.dumps({"parts": ordered}, indent=2, ensure_ascii=False) + "\n")
    return path, len(frames)


@torch.inference_mode()
def run_sequence_cases(cfg, model, tokenizer, vision_tokenizer):
    device = next(model.parameters()).device
    sampling = cfg.sampling_params
    for case_id, case in cfg.prompts.items():
        torch.manual_seed(int(getattr(cfg, "seed", 42)))
        started = time.monotonic()
        image_strings = []
        for path in case["image_list"]:
            with Image.open(path) as source:
                image = source.convert("RGB")
            width, height = image.size
            if width % 16 or height % 16:
                raise ValueError("Sequence input sizes must be multiples of 16")
            image_strings.append(build_image(image, SimpleNamespace(image_area=width * height), tokenizer, vision_tokenizer))
        ids = serialize_prompt(tokenizer, case["text_prompt"], image_strings)
        uncond, positions = unconditional_context(tokenizer, ids)
        budget = generation_budget(tokenizer, case["target_grid_shapes"], len(ids),
                                   model.config.max_position_embeddings, sampling.get("max_new_tokens", 0))
        print(f"Sequence {case_id}: prefix={len(ids)}, generation_budget={budget}", flush=True)
        result = generate_sequence(
            model, tokenizer, prefix_ids=torch.tensor(ids), device=device,
            max_new_tokens=budget, temperature=sampling["image_temperature"],
            top_k=sampling["image_top_k"], top_p=sampling["image_top_p"], greedy=False,
            cfg_scale=cfg.classifier_free_guidance,
            uncond_context_ids=torch.tensor(uncond), uncond_context_position_ids=torch.tensor(positions),
            max_images=sampling["max_images"], max_image_tokens=sampling["max_image_tokens"],
            max_header_tokens=sampling["max_header_tokens"], control_greedy=sampling["control_greedy"],
            bss_in_prefix=True, cfg_scale_end=sampling["cfg_min"], cfg_decay_images=sampling["cfg_total_images"],
            quick_validation=bool(sampling.get("quick_validation", False)),
        )
        raw = tokenizer.decode(result["generated_ids"], skip_special_tokens=False)
        parts = decode_video_parts(raw, tokenizer, vision_tokenizer)
        path, count = save_sequence_parts(parts, cfg.save_path, case_id, cfg.task_type,
                                          case["image_list"], cfg.output_fps)
        report = {
            "case": case_id, "output": str(path), "generated_images": count,
            "generated_tokens": len(result["generated_ids"]), "generation_budget": budget,
            "stop_reason": result["stop_reason"], "format_valid": result["format_valid"],
            "status": "quick_pass" if result["stop_reason"] == "quick_image_complete" else ("ok" if result["format_valid"] else "incomplete"),
            "seconds": time.monotonic() - started,
        }
        audit = os.environ.get("U0_AUDIT_DIR")
        if audit:
            Path(audit).mkdir(parents=True, exist_ok=True)
            (Path(audit) / f"{case_id}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        if result["stop_reason"] not in {"eos", "max_new_tokens", "quick_image_complete"}:
            raise RuntimeError(f"Sequence format failure: {result['stop_reason']}")

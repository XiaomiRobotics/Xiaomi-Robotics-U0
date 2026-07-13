from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from xr_u0_ar.image_tokens import build_image
from xr_u0_ar.task_prompts import (
    T2I_PROMPT_TEMPLATE,
    T2I_UNCOND_PROMPT,
    X2I_PROMPT_TEMPLATE,
    X2I_UNCOND_PROMPT,
)
from xr_u0_ar.vision_tokenizer import build_vision_tokenizer
from xr_u0_flashar.outputs import decode_grid, write_flashar_audit
from xr_u0_flashar.vllm import LLM


X2I_UNCOND_TEMPLATE = X2I_UNCOND_PROMPT
T2I_UNCOND_TEMPLATE = T2I_UNCOND_PROMPT
IMAGE_ANCHOR_TEMPLATE = "<|image start|>{H}*{W}<|image token|>"


def _read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _ensure_anchor(text: str, *, height: int, width: int) -> str:
    stripped = text.rstrip()
    if stripped.endswith("<|image token|>"):
        return stripped
    return stripped + IMAGE_ANCHOR_TEMPLATE.format(H=height, W=width)


def _case_id(row: dict[str, Any], index: int) -> str:
    raw = str(row.get("id") or row.get("key") or f"{index:05d}")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)


def _render_inputs(
    rows: list[dict[str, Any]],
    *,
    tokenizer,
    vision_tokenizer,
    height: int,
    width: int,
    source_image_area: int,
) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    unconds: list[str] = []
    image_cfg = SimpleNamespace(image_area=source_image_area)
    for index, row in enumerate(rows):
        prompt = str(row.get("prompt") or row.get("text") or "")
        if not prompt:
            raise ValueError(f"row {index}: missing prompt/text")
        image_path = row.get("image") or row.get("reference_image")
        if image_path:
            image = Image.open(image_path).convert("RGB")
            image_str = build_image(image, image_cfg, tokenizer, vision_tokenizer)
            rendered = X2I_PROMPT_TEMPLATE.format(question=prompt).replace(
                "<|IMAGE|>", image_str)
            uncond = X2I_UNCOND_TEMPLATE.replace("<|IMAGE|>", image_str)
            prompts.append(_ensure_anchor(rendered, height=height, width=width))
            unconds.append(_ensure_anchor(uncond, height=height, width=width))
        else:
            rendered = T2I_PROMPT_TEMPLATE.format(text=prompt)
            uncond = T2I_UNCOND_TEMPLATE
            prompts.append(_ensure_anchor(rendered, height=height, width=width))
            unconds.append(_ensure_anchor(uncond, height=height, width=width))
    return prompts, unconds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batched Xiaomi-Robotics-U0-FlashAR vLLM inference from JSONL."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--vision-tokenizer-dir", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-dir", default="outputs/flashar_batch")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--source-image-area", type=int, default=1024 * 1024)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5120)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=20000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.input_jsonl), args.limit)
    if not rows:
        raise ValueError(f"no rows found in {args.input_jsonl}")

    llm = LLM.from_pretrained(
        args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enable_prefix_caching=False,
        strict_visual_tokens=True,
    )
    vision_tokenizer = build_vision_tokenizer(
        "ibq",
        args.vision_tokenizer_dir,
        device=args.vision_device,
    )
    prompts, unconds = _render_inputs(
        rows,
        tokenizer=llm.tokenizer,
        vision_tokenizer=vision_tokenizer,
        height=args.height,
        width=args.width,
        source_image_area=args.source_image_area,
    )
    results = llm.generate(
        prompts,
        height=args.height,
        width=args.width,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        uncond_prompt=unconds,
        prompt_template="{text}",
    )

    out_dir = Path(args.out_dir)
    visual_offset = llm.tokenizer.encode("<|image end|>")[0] + 1
    for index, (row, result) in enumerate(zip(rows, results)):
        out_path = out_dir / f"{_case_id(row, index)}.png"
        decode_grid(result.grid, vision_tokenizer, visual_offset, out_path)
        write_flashar_audit(
            out_path,
            result,
            height=args.height,
            width=args.width,
            metadata={"id": row.get("id") or row.get("key")},
        )
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()

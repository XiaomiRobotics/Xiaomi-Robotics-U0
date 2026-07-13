from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def decode_grid(
    grid: torch.Tensor,
    vision_tokenizer: Any,
    visual_offset: int,
    out_path: str | Path,
    *,
    clamp_min: bool = False,
) -> Path:
    out_path = Path(out_path)
    height, width = grid.shape
    device = next(vision_tokenizer.parameters()).device
    codes = grid.to(device) - visual_offset
    if clamp_min:
        codes = codes.clamp_min(0)
    codes = codes.reshape(-1).long()
    with torch.no_grad():
        image = vision_tokenizer.decode_code(codes, shape=(1, height, width, 256))
    image = ((image.squeeze(0).cpu().float() + 1.0) / 2.0).clamp(0, 1)
    array = (image * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(out_path)
    return out_path


def audit_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_suffix(output_path.suffix + ".flashar_audit.json")


def write_flashar_audit(
    output_path: str | Path,
    result: Any,
    *,
    height: int | None = None,
    width: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path)
    height = int(height if height is not None else getattr(result, "height"))
    width = int(width if width is not None else getattr(result, "width"))
    expected = height * width
    actual = int(getattr(result, "n_visual_actual", expected))
    payload: dict[str, Any] = {
        "backend": "xr_u0_flashar_vllm",
        "output_path": str(output_path),
        "height": height,
        "width": width,
        "n_visual_expected": expected,
        "n_visual_actual": actual,
        "visual_token_complete": actual == expected,
        "token_min": getattr(result, "token_min", None),
        "token_max": getattr(result, "token_max", None),
        "warn": getattr(result, "warn", None),
        "n_steps": int(getattr(result, "n_steps", 0)),
    }
    grid = getattr(result, "grid", None)
    if grid is not None:
        payload["grid_shape"] = [int(x) for x in grid.shape]
    if metadata:
        payload.update(metadata)
    path = audit_path(output_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

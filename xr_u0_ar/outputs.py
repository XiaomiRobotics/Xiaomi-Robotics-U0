from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


@dataclass
class SavedGenerationOutput:
    output_path: Path
    manifest_path: Path
    image_paths: list[Path]
    text_path: Path | None
    parts: list[dict[str, Any]]

    @property
    def primary_path(self) -> Path:
        if self.image_paths:
            return self.image_paths[0]
        if self.text_path is not None:
            return self.text_path
        return self.manifest_path


def _image_path(output_path: Path, image_index: int) -> Path:
    suffix = output_path.suffix or ".png"
    base_path = output_path if output_path.suffix else output_path.with_suffix(suffix)
    if image_index == 0:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{image_index:03d}{suffix}")


def _text_path(output_path: Path) -> Path:
    return output_path.with_suffix(".txt")


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".outputs.json")


def _format_text_parts(text_parts: Iterable[tuple[str, str]]) -> str:
    chunks: list[str] = []
    for kind, text in text_parts:
        stripped = text.strip()
        if not stripped:
            continue
        chunks.append(f"[{kind}]\n{stripped}")
    return "\n\n".join(chunks).rstrip() + "\n"


def _image_headers(raw_text: str | None) -> list[dict[str, int]]:
    if not raw_text:
        return []
    headers: list[dict[str, int]] = []
    pattern = re.compile(
        r"<\|image start\|>\s*(\d+)\s*\*\s*(\d+)\s*<\|image token\|>"
    )
    for match in pattern.finditer(raw_text):
        height_tokens = int(match.group(1))
        width_tokens = int(match.group(2))
        headers.append({
            "generated_height_tokens": height_tokens,
            "generated_width_tokens": width_tokens,
            "expected_pixel_height": height_tokens * 16,
            "expected_pixel_width": width_tokens * 16,
        })
    return headers


def save_multimodal_parts(
    decoded_parts: Iterable[tuple[str, Any]],
    output_path: str | Path,
    *,
    raw_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    require_image: bool = False,
) -> SavedGenerationOutput:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    text_parts: list[tuple[str, str]] = []
    manifest_parts: list[dict[str, Any]] = []
    image_headers = _image_headers(raw_text)

    for index, (kind, payload) in enumerate(decoded_parts):
        if kind == "image" and isinstance(payload, Image.Image):
            image_index = len(image_paths)
            image_path = _image_path(output_path, len(image_paths))
            payload.save(image_path)
            image_paths.append(image_path)
            width_px, height_px = payload.size
            part: dict[str, Any] = {
                "index": index,
                "kind": "image",
                "path": str(image_path),
                "pixel_width": width_px,
                "pixel_height": height_px,
            }
            if image_index < len(image_headers):
                part.update(image_headers[image_index])
            manifest_parts.append(part)
            continue

        text = str(payload)
        if text.strip():
            text_parts.append((kind, text))
            manifest_parts.append({
                "index": index,
                "kind": kind,
                "text": text,
            })

    if not manifest_parts and raw_text and raw_text.strip():
        text_parts.append(("raw_text", raw_text))
        manifest_parts.append({
            "index": 0,
            "kind": "raw_text",
            "text": raw_text,
        })

    text_path: Path | None = None
    if text_parts:
        text_path = _text_path(output_path)
        text_path.write_text(_format_text_parts(text_parts), encoding="utf-8")

    manifest_path = _manifest_path(output_path)
    payload: dict[str, Any] = {
        "output_path": str(output_path),
        "primary_path": str(image_paths[0] if image_paths else text_path or manifest_path),
        "image_paths": [str(path) for path in image_paths],
        "text_path": str(text_path) if text_path is not None else None,
        "image_count": len(image_paths),
        "part_count": len(manifest_parts),
        "parts": manifest_parts,
    }
    if metadata:
        payload["metadata"] = metadata
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    saved = SavedGenerationOutput(
        output_path=output_path,
        manifest_path=manifest_path,
        image_paths=image_paths,
        text_path=text_path,
        parts=manifest_parts,
    )
    if require_image and not image_paths:
        raise RuntimeError(f"no image was decoded from the generated sequence; saved {manifest_path}")
    return saved


def save_generated_sequence(
    tokens: Any,
    tokenizer: Any,
    vision_tokenizer: Any,
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    require_image: bool = False,
) -> SavedGenerationOutput:
    from xr_u0_ar.generation import multimodal_decode

    text = tokenizer.decode(tokens, skip_special_tokens=False)
    decoded = multimodal_decode(text, tokenizer, vision_tokenizer)
    return save_multimodal_parts(
        decoded,
        output_path,
        raw_text=text,
        metadata=metadata,
        require_image=require_image,
    )

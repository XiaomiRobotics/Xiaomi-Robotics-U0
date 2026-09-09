from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


BOI = "<|image start|>"
IMG = "<|image token|>"
EOI = "<|image end|>"
EOL = "<|extra_200|>"


@dataclass
class SavedVideoGenerationOutput:
    output_path: Path
    manifest_path: Path
    frame_paths: list[Path]
    concat_path: Path | None
    text_path: Path | None
    parts: list[dict[str, Any]]

    @property
    def primary_path(self) -> Path:
        if self.output_path.exists():
            return self.output_path
        if self.concat_path is not None:
            return self.concat_path
        if self.frame_paths:
            return self.frame_paths[0]
        if self.text_path is not None:
            return self.text_path
        return self.manifest_path


def _tokenizer_attr(tokenizer: Any, name: str, fallback: str) -> str:
    return str(getattr(tokenizer, name, fallback))


def _special_tokens(tokenizer: Any) -> dict[str, str]:
    return {
        "boi": _tokenizer_attr(tokenizer, "boi_token", BOI),
        "img": _tokenizer_attr(tokenizer, "img_token", IMG),
        "eoi": _tokenizer_attr(tokenizer, "eoi_token", EOI),
        "eol": _tokenizer_attr(tokenizer, "eol_token", EOL),
    }


def _segment_pattern(tokenizer: Any) -> re.Pattern[str]:
    tokens = _special_tokens(tokenizer)
    return re.compile(
        rf"({re.escape(tokens['boi'])}.*?{re.escape(tokens['eoi'])})",
        re.DOTALL,
    )


def _parse_shape(header: str) -> tuple[int | None, int, int]:
    text = header.strip().replace(" ", "")
    match = re.fullmatch(r"(?:(\d+)FPS)?(\d+)\*(\d+)", text)
    if match is None:
        raise ValueError(f"unsupported video image header: {header!r}")
    header_fps = int(match.group(1)) if match.group(1) else None
    height = int(match.group(2))
    width = int(match.group(3))
    if header_fps is not None and header_fps <= 0:
        raise ValueError(f"invalid video image header: {header!r}")
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid video image header: {header!r}")
    return header_fps, height, width


def _vision_device(vision_tokenizer: Any) -> torch.device:
    try:
        return next(iter(vision_tokenizer.parameters())).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor_to_image(frame: torch.Tensor) -> Image.Image:
    array = (
        ((frame.float() + 1.0) * 127.5)
        .clamp(0, 255)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    return Image.fromarray(array)


def _decode_visual_segment(
    segment: str,
    tokenizer: Any,
    vision_tokenizer: Any,
) -> tuple[list[Image.Image], dict[str, int]]:
    tokens = _special_tokens(tokenizer)
    start = segment.index(tokens["boi"]) + len(tokens["boi"])
    img = segment.index(tokens["img"], start)
    end = segment.rindex(tokens["eoi"])
    header = segment[start:img]
    body = segment[img + len(tokens["img"]):end]

    header_fps, height, width = _parse_shape(header)
    visual_tokens = [int(item) for item in re.findall(r"<\|visual token (\d+)\|>", body)]
    tokens_per_frame = height * width
    if len(visual_tokens) == tokens_per_frame:
        frames = 1
    elif header_fps is not None and len(visual_tokens) == header_fps * tokens_per_frame:
        frames = header_fps
    else:
        expected = tokens_per_frame
        if header_fps is not None:
            expected = f"{tokens_per_frame} or {header_fps * tokens_per_frame}"
        raise ValueError(
            f"visual token count mismatch: expected {expected}, got {len(visual_tokens)}"
        )

    codes = torch.tensor(
        visual_tokens,
        dtype=torch.long,
        device=_vision_device(vision_tokenizer),
    ).reshape(frames, height, width)
    decoded = vision_tokenizer.decode_code(codes, shape=(frames, height, width, 256)).float()
    images = [_tensor_to_image(frame) for frame in decoded]
    header_info: dict[str, int] = {
        "frame_count": frames,
        "generated_height_tokens": height,
        "generated_width_tokens": width,
        "expected_pixel_height": height * 16,
        "expected_pixel_width": width * 16,
    }
    if header_fps is not None:
        header_info["header_fps"] = header_fps
    return images, header_info


def decode_video_parts(
    raw_text: str,
    tokenizer: Any,
    vision_tokenizer: Any,
) -> list[dict[str, Any]]:
    cleaned = raw_text.replace("<|extra_101|>", "").replace("<|extra_204|>", "")
    parts: list[dict[str, Any]] = []
    chunks = re.split(_segment_pattern(tokenizer), cleaned)
    for chunk in chunks:
        if not chunk:
            continue
        if BOI in chunk or _special_tokens(tokenizer)["boi"] in chunk:
            tokens = _special_tokens(tokenizer)
            if tokens["img"] not in chunk or tokens["eoi"] not in chunk:
                continue
            try:
                frames, info = _decode_visual_segment(chunk, tokenizer, vision_tokenizer)
            except Exception as exc:
                parts.append({"kind": "bad_image", "text": chunk, "error": repr(exc)})
            else:
                parts.append({"kind": "image", "frames": frames, **info})
            continue
        if chunk.strip():
            parts.append({"kind": "text", "text": chunk})
    return parts


def _as_rgb_array(image: Image.Image, *, size: tuple[int, int] | None = None) -> np.ndarray:
    image = image.convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size)
    return np.asarray(image, dtype=np.uint8)


def _save_mp4(images: list[Image.Image], path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    if not images:
        raise ValueError("cannot save an empty video")
    arrays = [_as_rgb_array(image) for image in images]
    with imageio.get_writer(path, fps=fps, mode="I") as writer:
        for array in arrays:
            writer.append_data(array)


def _save_concat(images: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [_as_rgb_array(image) for image in images]
    max_height = max(array.shape[0] for array in arrays)
    padded: list[np.ndarray] = []
    for array in arrays:
        height, width = array.shape[:2]
        if height < max_height:
            pad = np.zeros((max_height - height, width, 3), dtype=np.uint8)
            array = np.concatenate([array, pad], axis=0)
        padded.append(array)
    Image.fromarray(np.concatenate(padded, axis=1)).save(path)


def _load_context_images(paths: list[str], target_size: tuple[int, int] | None) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if target_size is not None and image.size != target_size:
            image = image.resize(target_size)
        images.append(image)
    return images


def save_video_sequence(
    tokens: Any,
    tokenizer: Any,
    vision_tokenizer: Any,
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    context_image_paths: list[str] | None = None,
    fps: int = 3,
    require_image: bool = False,
) -> SavedVideoGenerationOutput:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_text = tokenizer.decode(tokens, skip_special_tokens=False)
    parts = decode_video_parts(raw_text, tokenizer, vision_tokenizer)

    frame_dir = output_path.with_name(f"{output_path.stem}_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    text_chunks: list[str] = []
    manifest_parts: list[dict[str, Any]] = []
    generated_frames: list[Image.Image] = []

    for part_index, part in enumerate(parts):
        kind = part["kind"]
        if kind == "image":
            frames: list[Image.Image] = part["frames"]
            part_frame_paths: list[str] = []
            for frame in frames:
                frame_path = frame_dir / f"{len(frame_paths):04d}.png"
                frame.save(frame_path)
                frame_paths.append(frame_path)
                part_frame_paths.append(str(frame_path))
                generated_frames.append(frame)
            manifest_parts.append({
                "index": part_index,
                "kind": "image",
                "frame_paths": part_frame_paths,
                "frame_count": len(frames),
                "generated_height_tokens": part["generated_height_tokens"],
                "generated_width_tokens": part["generated_width_tokens"],
                "expected_pixel_height": part["expected_pixel_height"],
                "expected_pixel_width": part["expected_pixel_width"],
                **({"header_fps": part["header_fps"]} if "header_fps" in part else {}),
            })
        elif kind == "text":
            text = str(part["text"])
            text_chunks.append(text)
            manifest_parts.append({"index": part_index, "kind": "text", "text": text})
        else:
            manifest_parts.append({
                "index": part_index,
                "kind": "bad_image",
                "text": str(part.get("text", "")),
                "error": str(part.get("error", "")),
            })

    if require_image and not generated_frames:
        raise RuntimeError(f"no video frame was decoded from the generated sequence: {output_path}")

    text_path: Path | None = None
    if text_chunks:
        text_path = output_path.with_suffix(".txt")
        text_path.write_text("\n\n".join(text.strip() for text in text_chunks if text.strip()) + "\n",
                             encoding="utf-8")

    target_size = generated_frames[0].size if generated_frames else None
    context_images = _load_context_images(context_image_paths or [], target_size)
    video_frames = context_images + generated_frames

    concat_path: Path | None = None
    if video_frames:
        _save_mp4(video_frames, output_path, max(1, int(fps)))
        concat_path = output_path.with_name(f"{output_path.stem}_concat.png")
        _save_concat(video_frames, concat_path)

    manifest_path = output_path.with_suffix(".outputs.json")
    payload: dict[str, Any] = {
        "output_path": str(output_path),
        "primary_path": str(output_path if output_path.exists() else manifest_path),
        "frame_dir": str(frame_dir),
        "frame_paths": [str(path) for path in frame_paths],
        "concat_path": str(concat_path) if concat_path is not None else None,
        "text_path": str(text_path) if text_path is not None else None,
        "generated_frame_count": len(generated_frames),
        "context_frame_count": len(context_images),
        "fps": max(1, int(fps)),
        "parts": manifest_parts,
    }
    if metadata:
        payload["metadata"] = metadata
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return SavedVideoGenerationOutput(
        output_path=output_path,
        manifest_path=manifest_path,
        frame_paths=frame_paths,
        concat_path=concat_path,
        text_path=text_path,
        parts=manifest_parts,
    )

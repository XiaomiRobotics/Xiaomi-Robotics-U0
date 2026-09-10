from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from wm_fsdp.data.sequence import resolve_target_image_segments


@dataclass(frozen=True)
class VisualIdRange:
    minimum: int
    maximum: int
    count: int


def validate_visual_grid(
    grid: Any,
    *,
    id_space: str,
    visual_vocab_size: int,
    visual_token_offset: int,
    context: str,
) -> VisualIdRange:
    if not torch.is_tensor(grid) or grid.ndim != 2:
        raise ValueError(f"{context}: visual token grid must be [H,W], got {getattr(grid, 'shape', None)}")
    if grid.numel() == 0:
        raise ValueError(f"{context}: visual token grid cannot be empty")
    if grid.dtype == torch.bool or grid.dtype.is_floating_point or grid.dtype.is_complex:
        raise ValueError(f"{context}: visual token grid must use an integer dtype, got {grid.dtype}")
    minimum = int(grid.min())
    maximum = int(grid.max())
    if id_space == "raw":
        lower, upper = 0, visual_vocab_size
    elif id_space == "hf":
        lower, upper = visual_token_offset, visual_token_offset + visual_vocab_size
    else:
        raise ValueError(f"{context}: unsupported visual id space {id_space!r}")
    if minimum < lower or maximum >= upper:
        raise ValueError(
            f"{context}: {id_space} visual IDs [{minimum}, {maximum}] are outside [{lower}, {upper})"
        )
    return VisualIdRange(minimum=minimum, maximum=maximum, count=grid.numel())


def validate_multimodal_sample(
    sample: dict[str, Any],
    *,
    visual_vocab_size: int,
    visual_token_offset: int,
) -> list[VisualIdRange]:
    sample_id = str(sample.get("sample_id", "<unknown>"))
    segments = sample.get("input_segments")
    if not isinstance(segments, list):
        raise ValueError(f"{sample_id}: input_segments must be a list")

    ranges: list[VisualIdRange] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"{sample_id}: input segment {index} must be an object")
        segment_type = segment.get("type")
        if segment_type == "text":
            if not isinstance(segment.get("text"), str) or not segment["text"].strip():
                raise ValueError(f"{sample_id}: text segment {index} must be non-empty")
        elif segment_type == "image":
            ranges.append(
                validate_visual_grid(
                    segment.get("tokens"),
                    id_space=str(segment.get("id_space", "raw")),
                    visual_vocab_size=visual_vocab_size,
                    visual_token_offset=visual_token_offset,
                    context=f"{sample_id} input image {index}",
                )
            )
        else:
            raise ValueError(f"{sample_id}: unsupported input segment type {segment_type!r}")

    for index, target in enumerate(resolve_target_image_segments(sample)):
        ranges.append(
            validate_visual_grid(
                target.get("tokens"),
                id_space=str(target.get("id_space", "raw")),
                visual_vocab_size=visual_vocab_size,
                visual_token_offset=visual_token_offset,
                context=f"{sample_id} target image {index}",
            )
        )
    return ranges


__all__ = ["VisualIdRange", "validate_multimodal_sample", "validate_visual_grid"]

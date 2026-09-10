from __future__ import annotations

from typing import Any

import torch

from wm_fsdp.tokenizers.u0_text import encode_u0_text, u0_image_prefix, u0_token_id


def resolve_target_image_segments(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve ordered image targets and fail on conflicting legacy aliases."""
    raw_segments = sample.get("target_segments")
    if raw_segments is None:
        target = sample.get("target_tokens")
        if target is None:
            raise ValueError(f"Sample {sample.get('sample_id')} is missing image targets")
        return [
            {
                "type": "image",
                "tokens": target,
                "id_space": str(sample.get("target_id_space", "raw")),
            }
        ]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"Sample {sample.get('sample_id')} target_segments must be a non-empty list")

    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, dict) or segment.get("type") != "image":
            raise ValueError(
                f"Sample {sample.get('sample_id')} target segment {index} must be an image object"
            )
        if segment.get("tokens") is None:
            raise ValueError(
                f"Sample {sample.get('sample_id')} target segment {index} is missing tokens"
            )
        segments.append(segment)

    legacy_target = sample.get("target_tokens")
    if legacy_target is not None:
        first_target = segments[0]["tokens"]
        if (
            not torch.is_tensor(legacy_target)
            or not torch.is_tensor(first_target)
            or not torch.equal(legacy_target, first_target)
            or str(sample.get("target_id_space", "raw"))
            != str(segments[0].get("id_space", "raw"))
        ):
            raise ValueError(
                f"Sample {sample.get('sample_id')} legacy target conflicts with target_segments[0]"
            )
    return segments


class MultimodalCollator:
    """Build causal sequences from ordered input and target segments."""

    def __init__(
        self,
        model: Any,
        *,
        prompt_template: str = "You are a helpful assistant. USER: {text} ASSISTANT: ",
        max_length: int = 32768,
    ) -> None:
        if "{text}" not in prompt_template:
            raise ValueError("prompt_template must contain {text}")
        self.tokenizer = model.tokenizer
        self.visual_token_offset = int(model.visual_token_offset)
        self.visual_vocab_size = int(model.visual_vocab_size)
        self.prompt_template = prompt_template
        self.max_length = int(max_length)
        self.bos_id = int(self.tokenizer.bos_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.bss_id = u0_token_id(self.tokenizer, self.tokenizer.bss_token)
        self.ess_id = u0_token_id(self.tokenizer, self.tokenizer.ess_token)
        self.eol_id = u0_token_id(self.tokenizer, self.tokenizer.eol_token)
        self.eoi_id = u0_token_id(self.tokenizer, self.tokenizer.eoi_token)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        sequences: list[list[int]] = []
        labels: list[list[int]] = []
        for sample in samples:
            sequence, sample_labels = self._build_sample(sample)
            sequences.append(sequence)
            labels.append(sample_labels)

        max_length = max(len(sequence) for sequence in sequences)
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full((len(samples), max_length), pad_id, dtype=torch.long)
        label_ids = torch.full((len(samples), max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(samples), max_length), dtype=torch.long)
        for index, (sequence, sample_labels) in enumerate(zip(sequences, labels)):
            length = len(sequence)
            input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long)
            label_ids[index, :length] = torch.tensor(sample_labels, dtype=torch.long)
            attention_mask[index, :length] = 1
        # A repeated tensor is required here: DataLoader pinning rejects the
        # zero-stride, overlapping batch dimension produced by ``expand``.
        position_ids = torch.arange(max_length, dtype=torch.long).repeat(len(samples), 1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": label_ids,
            "dataset_names": [str(sample.get("dataset_name", "<unknown>")) for sample in samples],
            "sample_ids": [str(sample.get("sample_id", "<unknown>")) for sample in samples],
        }

    def _build_sample(self, sample: dict[str, Any]) -> tuple[list[int], list[int]]:
        target_segments = resolve_target_image_segments(sample)

        sequence = [self.bos_id]
        for segment in sample.get("input_segments", []):
            segment_type = segment.get("type")
            if segment_type == "text":
                text = str(segment.get("text", ""))
                if segment.get("apply_template", False):
                    text = self.prompt_template.replace("{text}", text)
                sequence.extend(encode_u0_text(self.tokenizer, text))
            elif segment_type == "image":
                sequence.extend(
                    self._image_block(
                        segment.get("tokens"),
                        str(segment.get("id_space", "raw")),
                        sample.get("sample_id"),
                    )
                )
            else:
                raise ValueError(f"Unsupported input segment type: {segment_type!r}")

        sequence.append(self.bss_id)
        target_start = len(sequence)
        for target_segment in target_segments:
            sequence.extend(
                self._image_block(
                    target_segment["tokens"],
                    str(target_segment.get("id_space", "raw")),
                    sample.get("sample_id"),
                )
            )
        sequence.extend([self.ess_id, self.eos_id])
        if self.max_length > 0 and len(sequence) > self.max_length:
            raise ValueError(
                f"Sample {sample.get('sample_id')} length={len(sequence)} exceeds max_length={self.max_length}"
            )
        sample_labels = [-100] * len(sequence)
        # Supervise the complete target span, including BSS, image-format
        # control tokens, ESS, and EOS. Prompt and condition images precede BSS.
        supervision_start = target_start - 1  # include the BSS just appended
        sample_labels[supervision_start:] = sequence[supervision_start:]
        return sequence, sample_labels

    def _image_block(self, grid: Any, id_space: str, sample_id: Any) -> list[int]:
        values = self._normalize_image_tokens(grid, id_space, sample_id)
        height, width = (int(value) for value in values.shape)
        return self._image_prefix(height, width) + self._image_body(values)

    def _normalize_image_tokens(self, grid: Any, id_space: str, sample_id: Any) -> torch.Tensor:
        if not torch.is_tensor(grid) or grid.ndim != 2:
            shape = getattr(grid, "shape", None)
            raise ValueError(f"Sample {sample_id} image grid must be [H,W], got {shape}")
        values = grid.detach().cpu().long().contiguous()
        if values.numel() == 0 or int(values.min()) < 0:
            raise ValueError(f"Sample {sample_id} contains invalid visual token ids")
        if id_space == "raw":
            if int(values.max()) >= self.visual_vocab_size:
                raise ValueError(f"Sample {sample_id} raw visual token exceeds the U0 IBQ vocabulary")
            values = values + self.visual_token_offset
        elif id_space == "hf":
            upper = self.visual_token_offset + self.visual_vocab_size
            if int(values.min()) < self.visual_token_offset or int(values.max()) >= upper:
                raise ValueError(f"Sample {sample_id} HF visual token is outside the model vocabulary")
        else:
            raise ValueError(f"Unsupported visual id space: {id_space!r}")
        return values

    def _image_prefix(self, height: int, width: int) -> list[int]:
        return u0_image_prefix(self.tokenizer, height, width)

    def _image_body(self, values: torch.Tensor) -> list[int]:
        output, _ = self._image_body_with_visual_mask(values)
        return output

    def _image_body_with_visual_mask(self, values: torch.Tensor) -> tuple[list[int], list[bool]]:
        height = int(values.shape[0])
        output: list[int] = []
        visual_mask: list[bool] = []
        for row_index, row in enumerate(values):
            row_values = [int(value) for value in row.tolist()]
            output.extend(row_values)
            visual_mask.extend([True] * len(row_values))
            if row_index + 1 < height:
                output.append(self.eol_id)
                visual_mask.append(False)
        output.append(self.eoi_id)
        visual_mask.append(False)
        return output, visual_mask


__all__ = [
    "MultimodalCollator",
    "resolve_target_image_segments",
]

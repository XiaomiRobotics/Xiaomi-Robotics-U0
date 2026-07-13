from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


LEGACY_JSONL_ENV = "XR_U0_VIDEO_GEN_LEGACY_JSONL"


def ar_image_sampling(max_new_tokens: int) -> dict[str, Any]:
    return {
        "image_top_k": 5120,
        "image_top_p": 1.0,
        "image_temperature": 1.0,
        "max_new_tokens": max_new_tokens,
    }


def flashar_sampling() -> dict[str, Any]:
    return {"temperature": 1.0, "top_k": 5120, "top_p": 1.0}


def active_examples(prefix: str, examples: list[Any], limit: int | None = None) -> dict[str, Any]:
    if not examples:
        raise ValueError(f"{prefix} must define at least one active example")
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("num_samples must be at least 1")
        examples = examples[:limit]
        if not examples:
            raise ValueError(f"{prefix} has no active examples")
    cases: dict[str, Any] = {}
    width = max(3, len(str(len(examples) - 1)))
    for index, item in enumerate(examples):
        case = deepcopy(item)
        if isinstance(case, dict):
            case_id = str(case.get("id") or f"{prefix}_{index:0{width}d}")
        else:
            case_id = f"{prefix}_{index:0{width}d}"
        cases[case_id] = case
    return cases


def repeat_case(prefix: str, case: Any, count: int | None) -> dict[str, Any]:
    repeat = 1 if count is None else int(count)
    if repeat < 1:
        raise ValueError("num_samples must be at least 1")
    width = max(3, len(str(max(repeat - 1, 0))))
    cases: dict[str, Any] = {}
    for idx in range(repeat):
        case_id = f"{prefix}_{idx:0{width}d}"
        cloned = deepcopy(case)
        if isinstance(cloned, dict):
            cloned["id"] = case_id
        cases[case_id] = cloned
    return cases


def set_reference(case: dict[str, Any], reference_images: list[str] | None) -> None:
    if not reference_images:
        return
    case["reference_image"] = reference_images[0] if len(reference_images) == 1 else list(reference_images)


def override_or_examples(
    *,
    prefix: str,
    examples: list[Any],
    num_samples: int | None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
    prompt_field: str = "prompt",
) -> dict[str, Any]:
    if prompt is None and not reference_images:
        return active_examples(prefix, examples, limit=num_samples)

    first = deepcopy(examples[0])
    case = first if isinstance(first, dict) else {prompt_field: str(first)}
    if prompt is not None:
        case[prompt_field] = prompt
    set_reference(case, reference_images)
    return repeat_case(prefix, case, num_samples)


def legacy_output_name(case: dict[str, Any], fallback: str) -> str:
    image_list = case.get("image_list") or case.get("image")
    if isinstance(image_list, list) and image_list:
        first = image_list[0]
        if isinstance(first, str) and first.strip():
            return Path(first).stem
    if isinstance(image_list, str) and image_list.strip():
        return Path(image_list).stem
    return fallback


def load_legacy_video_cases(limit: int, jsonl_path: str | None = None) -> dict[str, Any] | None:
    value = jsonl_path or os.environ.get(LEGACY_JSONL_ENV)
    if not value:
        return None
    if limit < 1:
        raise ValueError("num_samples must be at least 1")

    cases: dict[str, Any] = {}
    with Path(value).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            case = json.loads(line)
            output_name = legacy_output_name(case, f"{index:04d}")
            case["legacy_sample_index"] = index
            case["legacy_output_name"] = output_name
            cases[output_name] = case
    if len(cases) != limit:
        raise ValueError(f"expected {limit} legacy video cases, got {len(cases)}")
    return cases

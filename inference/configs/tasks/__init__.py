from __future__ import annotations

from typing import Any

from . import interleave_subtask, interleave_video, scene_gen, t2i, transfer, x2i


TASK_MODULES = {
    "t2i": t2i,
    "x2i": x2i,
    "scene_gen": scene_gen,
    "transfer": transfer,
    "interleave_subtask": interleave_subtask,
    "interleave_video": interleave_video,
}


def task_config(
    engine: str,
    task: str,
    *,
    num_samples: int | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
    legacy_video_jsonl: str | None = None,
) -> dict[str, Any]:
    if task in {"interleave_subtask", "interleave_video"} and engine != "ar":
        raise ValueError("Sequence tasks support AR eager only")
    try:
        module = TASK_MODULES[task]
    except KeyError as exc:
        raise ValueError(f"unsupported task: {task}") from exc
    return module.task_config(
        engine,
        num_samples=num_samples,
        prompt=prompt,
        reference_images=reference_images,
        legacy_video_jsonl=legacy_video_jsonl,
    )

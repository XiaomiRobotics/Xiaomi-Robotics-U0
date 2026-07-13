from __future__ import annotations

from typing import Any

from . import scene_gen, t2i, transfer, video_gen, x2i


TASK_MODULES = {
    "t2i": t2i,
    "x2i": x2i,
    "scene_gen": scene_gen,
    "transfer": transfer,
    "video_gen": video_gen,
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
    if engine == "flashar" and task == "video_gen":
        raise ValueError("Xiaomi-Robotics-U0-FlashAR does not support Video Gen")
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

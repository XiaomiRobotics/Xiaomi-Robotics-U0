from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from .base import base_config
from .profiles import profile_config
from .runtimes import runtime_config
from .tasks import task_config


ENGINE_ALIASES = {
    "ar": "ar",
    "xr-u0-ar": "ar",
    "flashar": "flashar",
    "xr-u0-flashar": "flashar",
    "flashar+xr-u0": "flashar",
}
BACKEND_ALIASES = {"eager": "eager", "hf": "eager", "vllm": "vllm"}
TASK_ALIASES = {
    "t2i": "t2i",
    "text-to-image": "t2i",
    "text_to_image": "t2i",
    "x2i": "x2i",
    "image-edit": "x2i",
    "image_edit": "x2i",
    "scene-gen": "scene_gen",
    "scene_gen": "scene_gen",
    "scene gen": "scene_gen",
    "transfer": "transfer",
    "video-gen": "video_gen",
    "video_gen": "video_gen",
    "video gen": "video_gen",
}
PROFILE_ALIASES = {
    "single": "single_gpu",
    "single-gpu": "single_gpu",
    "single_gpu": "single_gpu",
    "1gpu": "single_gpu",
    "multi": "multi_gpu",
    "multi-gpu": "multi_gpu",
    "multi_gpu": "multi_gpu",
    "2gpu": "multi_gpu",
}


def canonical_engine(value: str) -> str:
    return _canonicalize(value, ENGINE_ALIASES, "engine")


def canonical_backend(value: str) -> str:
    return _canonicalize(value, BACKEND_ALIASES, "backend")


def canonical_task(value: str) -> str:
    return _canonicalize(value, TASK_ALIASES, "task")


def canonical_profile(value: str) -> str:
    return _canonicalize(value, PROFILE_ALIASES, "profile")


def _canonicalize(value: str, aliases: dict[str, str], label: str) -> str:
    key = value.strip().lower().replace("_", "-")
    key = key.replace(" ", "-") if key not in aliases else key
    if key in aliases:
        return aliases[key]
    spaced = key.replace("-", " ")
    if spaced in aliases:
        return aliases[spaced]
    raise ValueError(f"unsupported {label}: {value}")


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def set_by_path(config: dict[str, Any], key: str, value: Any) -> None:
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        current = target.get(part)
        if not isinstance(current, dict):
            current = {}
            target[part] = current
        target = current
    target[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    return deep_merge(config, overrides or {})


def _save_path(engine: str, backend: str, task: str, profile: str) -> str:
    stem = f"{engine}_{backend}_{task}"
    if profile == "multi_gpu":
        return f"outputs/multigpu/{stem}"
    return f"outputs/{stem}"


def compose_dict(
    *,
    engine: str,
    backend: str,
    task: str,
    profile: str = "single_gpu",
    num_samples: int | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
    legacy_video_jsonl: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    engine = canonical_engine(engine)
    backend = canonical_backend(backend)
    task = canonical_task(task)
    profile = canonical_profile(profile)

    config: dict[str, Any] = {}
    for layer in (
        base_config(engine, task),
        task_config(
            engine,
            task,
            num_samples=num_samples,
            prompt=prompt,
            reference_images=reference_images,
            legacy_video_jsonl=legacy_video_jsonl,
        ),
        runtime_config(engine, backend, task),
        profile_config(profile, engine, backend, task),
        {"save_path": _save_path(engine, backend, task, profile)},
    ):
        config = deep_merge(config, layer)
    config["engine"] = engine
    config["profile"] = profile
    return apply_overrides(config, overrides)


def compose_config(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**compose_dict(**kwargs))


def jsonable(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return jsonable(vars(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

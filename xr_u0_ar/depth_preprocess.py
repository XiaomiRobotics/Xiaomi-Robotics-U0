from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_DA3_MODEL_PATH = "depth-anything/DA3-LARGE-1.1"
DEFAULT_DA3_DEVICE = "cuda:0"
DEFAULT_DA3_MIN_DEPTH = 0.7
DEFAULT_DA3_MAX_DEPTH = 2.0
DEFAULT_DA3_PROCESS_RES = 504
DA3_PROCESS_RES_METHOD = "upper_bound_resize"
DA3_TRANSFER_VIEW_COUNT = 3
DA3_DEPTH_ENCODING = "inverse_depth_dynamic_minmax_bright_near"

_DA3_MODEL_CACHE: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class DepthPreprocessOptions:
    input_image_type: str = "depth"
    da3_model_path: str | None = DEFAULT_DA3_MODEL_PATH
    da3_device: str = DEFAULT_DA3_DEVICE
    da3_min_depth: float = DEFAULT_DA3_MIN_DEPTH
    da3_max_depth: float = DEFAULT_DA3_MAX_DEPTH
    da3_process_res: int = DEFAULT_DA3_PROCESS_RES


def _source_get(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def normalize_input_image_type(value: Any) -> str:
    text = str(value or "depth").strip().lower().replace("-", "_")
    aliases = {
        "depth": "depth",
        "depth_map": "depth",
        "depthmap": "depth",
        "rgb": "rgb",
        "image": "rgb",
        "color": "rgb",
    }
    if text not in aliases:
        raise ValueError("input_image_type must be either 'depth' or 'rgb'")
    return aliases[text]


def is_transfer_task(task_type: Any) -> bool:
    text = str(task_type or "").strip().lower().replace("_", " ").replace("-", " ")
    return text == "transfer"


def depth_options_from_config(config: Any = None, case: Any = None) -> DepthPreprocessOptions:
    input_image_type = _source_get(case, "input_image_type", _source_get(config, "input_image_type", "depth"))
    da3_model_path = _source_get(case, "da3_model_path", _source_get(config, "da3_model_path", None))
    da3_device = _source_get(case, "da3_device", _source_get(config, "da3_device", DEFAULT_DA3_DEVICE))
    da3_min_depth = _source_get(case, "da3_min_depth", _source_get(config, "da3_min_depth", DEFAULT_DA3_MIN_DEPTH))
    da3_max_depth = _source_get(case, "da3_max_depth", _source_get(config, "da3_max_depth", DEFAULT_DA3_MAX_DEPTH))
    da3_process_res = _source_get(
        case,
        "da3_process_res",
        _source_get(config, "da3_process_res", DEFAULT_DA3_PROCESS_RES),
    )
    model_path = str(da3_model_path).strip() if da3_model_path else None
    return DepthPreprocessOptions(
        input_image_type=normalize_input_image_type(input_image_type),
        da3_model_path=model_path or DEFAULT_DA3_MODEL_PATH,
        da3_device=str(da3_device or DEFAULT_DA3_DEVICE),
        da3_min_depth=float(da3_min_depth),
        da3_max_depth=float(da3_max_depth),
        da3_process_res=int(da3_process_res),
    )


def validate_reference_preprocess_options(
    task_type: Any,
    options: DepthPreprocessOptions,
    *,
    context: str = "reference image",
) -> None:
    if options.input_image_type == "depth":
        return
    if not is_transfer_task(task_type):
        raise ValueError(f"{context}: input_image_type='rgb' is only supported for Transfer")
    if not (options.da3_max_depth > options.da3_min_depth > 0):
        raise ValueError(
            f"{context}: require da3_max_depth > da3_min_depth > 0, got "
            f"{options.da3_min_depth} and {options.da3_max_depth}"
        )
    if options.da3_process_res <= 0:
        raise ValueError(f"{context}: da3_process_res must be positive")


def validate_depth_preprocess_config(config: Any) -> None:
    global_task = _source_get(config, "task_type", "")
    prompts = _source_get(config, "prompts", {})
    if isinstance(prompts, dict):
        cases = [(str(key), value) for key, value in prompts.items()]
    elif isinstance(prompts, list):
        cases = [(str(index), value) for index, value in enumerate(prompts)]
    else:
        cases = [("config", None)]
    for case_id, case in cases:
        task = _source_get(case, "task_type", global_task) if isinstance(case, dict) else global_task
        options = depth_options_from_config(config, case if isinstance(case, dict) else None)
        validate_reference_preprocess_options(task, options, context=f"case {case_id}")


def depth_preprocess_metadata(
    task_type: Any,
    options: DepthPreprocessOptions,
    *,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not is_transfer_task(task_type):
        return {}
    payload: dict[str, Any] = {"input_image_type": options.input_image_type}
    if options.input_image_type == "rgb":
        payload.update(
            {
                "da3_model_path": options.da3_model_path,
                "da3_device": options.da3_device,
                "da3_min_depth": options.da3_min_depth,
                "da3_max_depth": options.da3_max_depth,
                "da3_process_res": options.da3_process_res,
                "da3_process_res_method": DA3_PROCESS_RES_METHOD,
                "da3_view_count": DA3_TRANSFER_VIEW_COUNT,
                "da3_depth_encoding": DA3_DEPTH_ENCODING,
            }
        )
        if artifact_dir is not None:
            payload["da3_artifact_dir"] = str(Path(artifact_dir))
    return payload


def load_da3_model(model_path: str | Path, device: str = DEFAULT_DA3_DEVICE):
    model_key = (str(model_path), str(device))
    cached = _DA3_MODEL_CACHE.get(model_key)
    if cached is not None:
        return cached

    os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "Depth Anything 3 is required for Transfer RGB inputs. Install the "
            "optional depth dependencies in the same environment, then install "
            "the depth-anything-3 package."
        ) from exc

    model = DepthAnything3.from_pretrained(str(model_path))
    model = model.to(str(device)).eval()
    _DA3_MODEL_CACHE[model_key] = model
    return model


def normalize_inverse_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    if not (max_depth > min_depth > 0):
        raise ValueError("require max_depth > min_depth > 0")
    depth = depth.astype(np.float32, copy=False)
    clipped = np.clip(depth, float(min_depth), float(max_depth))
    inv_lo = 1.0 / float(max_depth)
    inv_hi = 1.0 / float(min_depth)
    norm = (1.0 / clipped - inv_lo) / (inv_hi - inv_lo)
    return np.clip(norm, 0.0, 1.0)


def inverse_depth_to_rgb_image(depth: np.ndarray, min_depth: float, max_depth: float) -> Image.Image:
    gray = (normalize_inverse_depth(depth, min_depth, max_depth) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB")


def normalize_inverse_depth_dynamic(depth: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    depth = depth.astype(np.float32, copy=False)
    inv_depth = 1.0 / np.maximum(depth, 1e-8)
    inv_min = float(inv_depth.min())
    inv_max = float(inv_depth.max())
    if inv_max - inv_min < 1e-8:
        gray = np.zeros_like(inv_depth, dtype=np.uint8)
    else:
        gray = ((inv_depth - inv_min) / (inv_max - inv_min) * 255.0).astype(np.uint8)
    stats = {
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "inverse_depth_min": inv_min,
        "inverse_depth_max": inv_max,
    }
    return gray, stats


def inverse_depth_dynamic_to_rgb_image(depth: np.ndarray) -> tuple[Image.Image, dict[str, float]]:
    gray, stats = normalize_inverse_depth_dynamic(depth)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), mode="RGB"), stats


def _resize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if depth.shape == (height, width):
        return depth.astype(np.float32, copy=False)
    image = Image.fromarray(depth.astype(np.float32, copy=False))
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(image.resize((width, height), resample=resample), dtype=np.float32)


def _as_depth_map(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    depth = np.asarray(value, dtype=np.float32)
    while depth.ndim > 2:
        if depth.shape[0] == 1:
            depth = depth[0]
        elif depth.shape[-1] == 1:
            depth = depth[..., 0]
        else:
            break
    if depth.ndim != 2:
        raise RuntimeError(f"DA3 returned a non-2D depth map with shape {depth.shape}")
    return depth


def _prediction_depth_maps(prediction: Any, expected_count: int) -> list[np.ndarray]:
    if not hasattr(prediction, "depth"):
        raise RuntimeError("DA3 inference result does not expose a depth attribute")
    raw_depths = prediction.depth
    if hasattr(raw_depths, "detach"):
        raw_depths = raw_depths.detach().cpu().numpy()
    if isinstance(raw_depths, np.ndarray) and raw_depths.ndim >= 3 and raw_depths.shape[0] == expected_count:
        depth_maps = [_as_depth_map(raw_depths[index]) for index in range(expected_count)]
    elif isinstance(raw_depths, (list, tuple)):
        depth_maps = [_as_depth_map(depth) for depth in raw_depths]
    else:
        depth_maps = [_as_depth_map(raw_depths)]
    if len(depth_maps) != expected_count:
        raise RuntimeError(f"DA3 returned {len(depth_maps)} depth maps for {expected_count} RGB views")
    return depth_maps


def split_horizontal_triptych(image: Image.Image, view_count: int = DA3_TRANSFER_VIEW_COUNT) -> list[Image.Image]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < view_count:
        raise ValueError(f"RGB Transfer reference width {width} is too small for {view_count} views")
    base_width = width // view_count
    views: list[Image.Image] = []
    for index in range(view_count):
        left = index * base_width
        right = (index + 1) * base_width if index < view_count - 1 else width
        views.append(rgb.crop((left, 0, right, height)))
    return views


def concat_horizontal(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("cannot concatenate an empty image list")
    rgb_images = [image.convert("RGB") for image in images]
    height = rgb_images[0].height
    width = sum(image.width for image in rgb_images)
    canvas = Image.new("RGB", (width, height))
    offset = 0
    for image in rgb_images:
        if image.height != height:
            image = image.resize((image.width, height), Image.BICUBIC)
        canvas.paste(image, (offset, 0))
        offset += image.width
    return canvas


def _write_da3_artifacts(
    artifact_dir: str | Path | None,
    *,
    source_path: str | Path | None,
    original_image: Image.Image,
    view_images: list[Image.Image],
    depth_images: list[Image.Image],
    output_image: Image.Image,
    depth_stats: list[dict[str, float]],
    options: DepthPreprocessOptions,
) -> Path | None:
    if artifact_dir is None:
        return None
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)

    original_path = root / "reference_rgb.png"
    output_path = root / "depth_triptych.png"
    original_image.convert("RGB").save(original_path)
    output_image.save(output_path)

    view_payload: list[dict[str, Any]] = []
    for index, (view, depth, stats) in enumerate(zip(view_images, depth_images, depth_stats)):
        rgb_path = root / f"view_{index}_rgb.png"
        depth_path = root / f"view_{index}_depth.png"
        view.save(rgb_path)
        depth.save(depth_path)
        view_payload.append(
            {
                "index": index,
                "rgb_path": str(rgb_path.resolve()),
                "depth_path": str(depth_path.resolve()),
                "rgb_size": [int(view.width), int(view.height)],
                "depth_size": [int(depth.width), int(depth.height)],
                **stats,
            }
        )

    metadata_path = root / "metadata.json"
    metadata = {
        "source_path": str(source_path) if source_path is not None else None,
        "original_path": str(original_path.resolve()),
        "output_path": str(output_path.resolve()),
        "original_size": [int(original_image.width), int(original_image.height)],
        "output_size": [int(output_image.width), int(output_image.height)],
        "da3_model_path": options.da3_model_path,
        "da3_device": options.da3_device,
        "da3_process_res": options.da3_process_res,
        "da3_process_res_method": DA3_PROCESS_RES_METHOD,
        "da3_view_count": len(view_images),
        "da3_depth_encoding": DA3_DEPTH_ENCODING,
        "views": view_payload,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def rgb_to_inverse_depth_image(
    image: Image.Image,
    *,
    model: Any,
    min_depth: float,
    max_depth: float,
    process_res: int = DEFAULT_DA3_PROCESS_RES,
    device: str = DEFAULT_DA3_DEVICE,
    artifact_dir: str | Path | None = None,
    source_path: str | Path | None = None,
    options: DepthPreprocessOptions | None = None,
) -> Image.Image:
    del min_depth, max_depth, device
    rgb = image.convert("RGB")
    view_images = split_horizontal_triptych(rgb)
    prediction = model.inference(
        image=view_images,
        process_res=int(process_res),
    )
    depths = _prediction_depth_maps(prediction, len(view_images))

    depth_images: list[Image.Image] = []
    depth_stats: list[dict[str, float]] = []
    for view, depth in zip(view_images, depths):
        resized_depth = _resize_depth(depth, view.size)
        depth_image, stats = inverse_depth_dynamic_to_rgb_image(resized_depth)
        depth_images.append(depth_image)
        depth_stats.append(stats)
    output = concat_horizontal(depth_images)

    artifact_options = options or DepthPreprocessOptions(
        input_image_type="rgb",
        da3_model_path=DEFAULT_DA3_MODEL_PATH,
        da3_process_res=int(process_res),
    )
    _write_da3_artifacts(
        artifact_dir,
        source_path=source_path,
        original_image=rgb,
        view_images=view_images,
        depth_images=depth_images,
        output_image=output,
        depth_stats=depth_stats,
        options=artifact_options,
    )
    return output


def preprocess_reference_image(
    image: Image.Image,
    *,
    task_type: Any,
    options: DepthPreprocessOptions,
    artifact_dir: str | Path | None = None,
    source_path: str | Path | None = None,
) -> Image.Image:
    validate_reference_preprocess_options(task_type, options)
    if options.input_image_type == "depth" or not is_transfer_task(task_type):
        return image.convert("RGB")
    model = load_da3_model(options.da3_model_path or "", options.da3_device)
    return rgb_to_inverse_depth_image(
        image,
        model=model,
        min_depth=options.da3_min_depth,
        max_depth=options.da3_max_depth,
        process_res=options.da3_process_res,
        device=options.da3_device,
        artifact_dir=artifact_dir,
        source_path=source_path,
        options=options,
    )


def open_reference_image(
    path: str | Path,
    *,
    task_type: Any,
    config: Any = None,
    case: Any = None,
    artifact_dir: str | Path | None = None,
) -> Image.Image:
    options = depth_options_from_config(config, case)
    image = Image.open(path).convert("RGB")
    return preprocess_reference_image(
        image,
        task_type=task_type,
        options=options,
        artifact_dir=artifact_dir,
        source_path=path,
    )

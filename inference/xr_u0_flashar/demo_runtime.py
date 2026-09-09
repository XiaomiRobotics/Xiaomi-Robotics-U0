from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from xr_u0_ar.depth_preprocess import (
    DEFAULT_DA3_DEVICE,
    DEFAULT_DA3_MAX_DEPTH,
    DEFAULT_DA3_MIN_DEPTH,
    DEFAULT_DA3_MODEL_PATH,
    DEFAULT_DA3_PROCESS_RES,
    DepthPreprocessOptions,
    depth_preprocess_metadata,
    normalize_input_image_type,
    preprocess_reference_image,
)
from xr_u0_ar.task_prompts import (
    T2I_PROMPT_TEMPLATE,
    T2I_UNCOND_PROMPT,
    X2I_PROMPT_TEMPLATE,
    X2I_UNCOND_PROMPT,
)
from xr_u0_flashar.outputs import decode_grid, write_flashar_audit


BOS = "<|extra_203|>"
BSS = "<|extra_100|>"
IMAGE_ANCHOR_TEMPLATE = "<|image start|>{H}*{W}<|image token|>"

TASK_T2I = "T2I"
TASK_X2I = "X2I"
TASK_SCENE = "Scene Gen"
TASK_TRANSFER = "Transfer"
SUPPORTED_TASKS = (TASK_T2I, TASK_X2I, TASK_SCENE, TASK_TRANSFER)

DEFAULT_T2I_TEMPLATE = T2I_PROMPT_TEMPLATE
DEFAULT_T2I_UNCOND = T2I_UNCOND_PROMPT
DEFAULT_X2I_TEMPLATE = X2I_PROMPT_TEMPLATE
DEFAULT_X2I_UNCOND = X2I_UNCOND_PROMPT

SCENE_SYSTEM_PROMPT = (
    "You are an advanced controllable image generation model specialized in synthesizing "
    "high-quality robotic vision data. You will receive a scene description of the robot "
    "workspace. Your task is to generate the high-quality initial observation containing "
    "3 views with one full size view on top and two half size views side by side at the "
    "bottom. The actual output canvas is one horizontally concatenated multi-view image. "
    "You should keep the consistency across different views. Image Style: {style}."
)

TRANSFER_SYSTEM_PROMPT = (
    "You are an advanced controllable image generation model. You will be provided with "
    "a scene description and a grid of monocular depth maps from multiple viewpoints "
    "(where pixel intensity represents distance from the camera - brighter means closer, "
    "darker means farther). The depth maps are arranged in a grid layout. Your objective "
    "is to generate a corresponding grid of photorealistic images that strictly respect "
    "the 3D geometry defined by each depth map. The output canvas is one horizontally "
    "concatenated multi-view image. You should keep the consistency across different "
    "views in the grid. Image Style: {style}."
)

DEFAULT_HEIGHT_WIDTH = {
    TASK_T2I: (64, 64),
    TASK_SCENE: (32, 128),
}
DEFAULT_CFG_SCALE = {
    TASK_T2I: 3.0,
    TASK_X2I: 3.0,
    TASK_SCENE: 2.0,
    TASK_TRANSFER: 3.0,
}


@dataclass
class RenderedFlashARRequest:
    task_type: str
    prompt: str
    uncond_prompt: str
    height: int
    width: int
    cfg_scale: float
    metadata: dict[str, Any]


def normalize_task_type(task_type: str) -> str:
    cleaned = task_type.strip().replace("_", " ").lower()
    aliases = {
        "t2i": TASK_T2I,
        "text to image": TASK_T2I,
        "x2i": TASK_X2I,
        "image edit": TASK_X2I,
        "image to image": TASK_X2I,
        "scene": TASK_SCENE,
        "scene gen": TASK_SCENE,
        "scene generation": TASK_SCENE,
        "transfer": TASK_TRANSFER,
    }
    if cleaned not in aliases:
        raise ValueError(f"unsupported task_type: {task_type!r}")
    return aliases[cleaned]


def task_defaults(task_type: str) -> dict[str, Any]:
    task = normalize_task_type(task_type)
    height, width = DEFAULT_HEIGHT_WIDTH.get(task, (0, 0))
    return {
        "task_type": task,
        "height": height,
        "width": width,
        "cfg_scale": DEFAULT_CFG_SCALE[task],
    }


def ensure_anchor(text: str, *, height: int, width: int) -> str:
    stripped = text.lstrip()
    if not stripped.startswith(BOS):
        stripped = BOS + stripped
    stripped = stripped.rstrip()
    if stripped.endswith("<|image token|>"):
        return stripped
    return stripped + IMAGE_ANCHOR_TEMPLATE.format(H=height, W=width)


def pil_image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_path_to_base64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def pil_image_from_base64(value: str) -> Image.Image:
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    data = base64.b64decode(payload)
    return Image.open(io.BytesIO(data)).convert("RGB")


def build_reference_tokens(
    images: list[Image.Image],
    *,
    tokenizer: Any,
    vision_tokenizer: Any,
    image_area: int,
) -> list[str]:
    if not images:
        return []
    from xr_u0_ar.image_tokens import build_image

    cfg = SimpleNamespace(image_area=int(image_area))
    return [
        build_image(image.convert("RGB"), cfg, tokenizer, vision_tokenizer)
        for image in images
    ]


def reference_token_shape(image: Image.Image, image_area: int) -> tuple[int, int]:
    from xr_u0_ar.image_tokens import smart_resize

    resized = smart_resize(image.convert("RGB"), int(image_area))
    width_px, height_px = resized.size
    return height_px // 16, width_px // 16


def resolve_shape(
    task_type: str,
    *,
    reference_images: list[Image.Image],
    height: int | None,
    width: int | None,
    source_image_area: int,
) -> tuple[int, int]:
    task = normalize_task_type(task_type)
    if (height is None) != (width is None):
        raise ValueError("height and width must be set together")
    if height is not None and width is not None and height > 0 and width > 0:
        return int(height), int(width)
    if task in DEFAULT_HEIGHT_WIDTH:
        return DEFAULT_HEIGHT_WIDTH[task]
    if not reference_images:
        raise ValueError(f"{task} requires a reference image when height/width are auto")
    shapes = {reference_token_shape(image, source_image_area) for image in reference_images}
    if len(shapes) != 1:
        raise ValueError(f"reference images resolve to mixed token shapes: {sorted(shapes)}")
    return next(iter(shapes))


def render_flashar_request(
    *,
    task_type: str,
    text: str,
    tokenizer: Any,
    vision_tokenizer: Any,
    reference_images: list[Image.Image] | None = None,
    height: int | None = None,
    width: int | None = None,
    cfg_scale: float | None = None,
    source_image_area: int = 1024 * 1024,
    robot_arm_type: str = "AgiBot G1",
    image_style: str = "Real",
    input_image_type: str = "depth",
    da3_model_path: str | None = None,
    da3_device: str = DEFAULT_DA3_DEVICE,
    da3_min_depth: float = DEFAULT_DA3_MIN_DEPTH,
    da3_max_depth: float = DEFAULT_DA3_MAX_DEPTH,
    da3_process_res: int = DEFAULT_DA3_PROCESS_RES,
    da3_artifact_dir: str | Path | None = None,
) -> RenderedFlashARRequest:
    task = normalize_task_type(task_type)
    user_text = text.strip()
    if not user_text:
        raise ValueError("prompt text is required")
    refs = reference_images or []
    depth_options = DepthPreprocessOptions(
        input_image_type=normalize_input_image_type(input_image_type),
        da3_model_path=(str(da3_model_path).strip() if da3_model_path else DEFAULT_DA3_MODEL_PATH),
        da3_device=str(da3_device or DEFAULT_DA3_DEVICE),
        da3_min_depth=float(da3_min_depth),
        da3_max_depth=float(da3_max_depth),
        da3_process_res=int(da3_process_res),
    )
    prepared_refs: list[Image.Image] = []
    for index, image in enumerate(refs):
        reference_artifact_dir = None
        if da3_artifact_dir is not None:
            reference_artifact_dir = Path(da3_artifact_dir)
            if len(refs) > 1:
                reference_artifact_dir = reference_artifact_dir / f"reference_{index}"
        prepared_refs.append(
            preprocess_reference_image(
                image,
                task_type=task,
                options=depth_options,
                artifact_dir=reference_artifact_dir,
            )
        )
    h, w = resolve_shape(
        task,
        reference_images=prepared_refs,
        height=height,
        width=width,
        source_image_area=source_image_area,
    )
    reference_tokens = build_reference_tokens(
        prepared_refs,
        tokenizer=tokenizer,
        vision_tokenizer=vision_tokenizer,
        image_area=source_image_area,
    )
    image_prefix = "".join(reference_tokens)

    if task == TASK_T2I:
        prompt = DEFAULT_T2I_TEMPLATE.format(text=user_text)
        uncond = DEFAULT_T2I_UNCOND
    elif task == TASK_X2I:
        if not refs:
            raise ValueError("X2I requires one reference image")
        prompt = DEFAULT_X2I_TEMPLATE.replace("<|IMAGE|>", image_prefix).format(
            question=user_text,
            text=user_text,
        )
        uncond = DEFAULT_X2I_UNCOND.replace("<|IMAGE|>", image_prefix).format(
            question=user_text,
            text=user_text,
        )
    elif task == TASK_SCENE:
        system = SCENE_SYSTEM_PROMPT.format(style=image_style.strip() or "Real")
        prompt = (
            f"{system} Robot Arm Type: {robot_arm_type.strip() or 'AgiBot G1'}. "
            f"Scene Description: {user_text} {BSS}"
        )
        uncond = f"{system} {BSS}"
    elif task == TASK_TRANSFER:
        if not refs:
            raise ValueError("Transfer requires one reference image")
        system = TRANSFER_SYSTEM_PROMPT.format(style=image_style.strip() or "Simulator")
        prompt = f"{system} Scene Description: {user_text} {image_prefix}{BSS}"
        uncond = f"{BOS}{system} {image_prefix}{BSS}"
    else:
        raise AssertionError(f"unreachable task: {task}")

    cfg = DEFAULT_CFG_SCALE[task] if cfg_scale is None else float(cfg_scale)
    return RenderedFlashARRequest(
        task_type=task,
        prompt=ensure_anchor(prompt, height=h, width=w),
        uncond_prompt=ensure_anchor(uncond, height=h, width=w),
        height=h,
        width=w,
        cfg_scale=cfg,
        metadata={
            "task_type": task,
            "height": h,
            "width": w,
            "cfg_scale": cfg,
            "source_image_area": int(source_image_area),
            "n_reference_images": len(refs),
            "robot_arm_type": robot_arm_type,
            "image_style": image_style,
            **depth_preprocess_metadata(task, depth_options, artifact_dir=da3_artifact_dir),
        },
    )


def save_flashar_result(
    *,
    result: Any,
    vision_tokenizer: Any,
    visual_offset: int,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path)
    decode_grid(result.grid, vision_tokenizer, visual_offset, output_path, clamp_min=True)
    audit_path = write_flashar_audit(
        output_path,
        result,
        height=int(getattr(result, "height", result.grid.shape[0])),
        width=int(getattr(result, "width", result.grid.shape[1])),
        metadata=metadata,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return {
        "output_path": str(output_path.resolve()),
        "audit_path": str(audit_path.resolve()),
        "audit": audit,
    }

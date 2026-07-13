from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xr_u0_ar.task_prompts import (
    T2I_PROMPT_TEMPLATE,
    T2I_UNCOND_PROMPT,
    X2I_PROMPT_TEMPLATE,
    X2I_UNCOND_PROMPT,
)
from xr_u0_flashar.outputs import decode_grid, write_flashar_audit


DEFAULT_T2I_TEMPLATE = T2I_PROMPT_TEMPLATE
DEFAULT_T2I_UNCOND = T2I_UNCOND_PROMPT
DEFAULT_X2I_TEMPLATE = X2I_PROMPT_TEMPLATE
DEFAULT_X2I_UNCOND = X2I_UNCOND_PROMPT
IMAGE_ANCHOR_TEMPLATE = "<|image start|>{H}*{W}<|image token|>"
SPECIAL_TOKENS = {
    "BOS": "<|extra_203|>",
    "BSS": "<|extra_100|>",
}


def load_config(path: str):
    cfg_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(cfg_path.stem, cfg_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cfg_get(cfg, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import torch

        if isinstance(value, torch.device):
            return str(value)
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def path_get(cfg, *names: str) -> str:
    for name in names:
        value = getattr(cfg, name, None)
        if value:
            return str(value)
    raise AttributeError(f"config must define one of: {', '.join(names)}")


def normalize_cases(prompts: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(prompts, dict):
        raw = [(str(k), v) for k, v in prompts.items()]
    elif isinstance(prompts, list):
        raw = [(f"{idx:03d}", item) for idx, item in enumerate(prompts)]
    else:
        raise TypeError("config.prompts must be a dict or a list")

    cases: list[tuple[str, dict[str, Any]]] = []
    for case_id, item in raw:
        if isinstance(item, dict):
            case = dict(item)
        else:
            case = {"prompt": str(item)}
        if not case.get("id"):
            case["id"] = case_id
        cases.append((case_id, case))
    return cases


def case_reference_images(case: dict[str, Any]) -> list[str]:
    value = case.get("reference_image") or case.get("image")
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def task_type(cfg, case: dict[str, Any]) -> str:
    return str(case.get("task_type") or cfg_get(cfg, "task_type", "T2I"))


def _case_get(case: dict[str, Any], name: str, default: Any = None) -> Any:
    value = case.get(name, None)
    return default if value is None else value


def case_raw_prompt(case: dict[str, Any]) -> str | None:
    value = case.get("text_prompt")
    if value is None:
        return None
    return str(value)


def case_uncond_prompt(case: dict[str, Any]) -> str | None:
    value = case.get("uncond_prompt")
    if value is None:
        return None
    return str(value)


def case_visual_placeholder(case: dict[str, Any]) -> str:
    return str(case.get("visual_placeholder") or "<|VIS_PLH|>")


def case_supervised_start(case: dict[str, Any]) -> str:
    return str(case.get("supervised_start") or SPECIAL_TOKENS["BSS"])


def _ensure_bos(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith(SPECIAL_TOKENS["BOS"]):
        return stripped
    return SPECIAL_TOKENS["BOS"] + stripped


def ensure_anchor(text: str, *, height: int, width: int) -> str:
    stripped = _ensure_bos(text).rstrip()
    if stripped.endswith("<|image token|>"):
        return stripped
    return stripped + IMAGE_ANCHOR_TEMPLATE.format(H=height, W=width)


def _reference_token_shape(path: str, image_area: int) -> tuple[int, int]:
    from PIL import Image
    from xr_u0_ar.image_tokens import smart_resize

    resized = smart_resize(Image.open(path).convert("RGB"), image_area)
    width_px, height_px = resized.size
    return height_px // 16, width_px // 16


def resolve_case_shape(cfg, case: dict[str, Any]) -> tuple[int, int]:
    height = _case_get(case, "height", cfg_get(cfg, "height", None))
    width = _case_get(case, "width", cfg_get(cfg, "width", None))
    if height is None:
        height = _case_get(case, "target_height", cfg_get(cfg, "target_height", None))
    if width is None:
        width = _case_get(case, "target_width", cfg_get(cfg, "target_width", None))
    if (height is None) != (width is None):
        raise ValueError("FlashAR height and width must be set together")
    if height is not None:
        return int(height), int(width)

    match_reference_shape = bool(
        _case_get(case, "match_reference_shape", cfg_get(cfg, "match_reference_shape", False))
    )
    if match_reference_shape:
        references = case_reference_images(case)
        if not references:
            raise ValueError(
                f"case {case.get('id')}: match_reference_shape=True requires reference_image"
            )
        image_area = int(cfg_get(cfg, "source_image_area", cfg_get(cfg, "image_area", 1024 * 1024)))
        shapes = {_reference_token_shape(path, image_area) for path in references}
        if len(shapes) != 1:
            raise ValueError(f"case {case.get('id')}: reference images resolve to mixed shapes {shapes}")
        return next(iter(shapes))

    raise ValueError(
        f"case {case.get('id')}: FlashAR requires explicit height/width, "
        "or match_reference_shape=True for image-edit cases"
    )


def template_pair(cfg, case: dict[str, Any]) -> tuple[str, str]:
    if hasattr(cfg, "template") and hasattr(cfg, "unc_prompt"):
        return str(cfg.template), str(cfg.unc_prompt)
    task = task_type(cfg, case).lower().replace("_", " ").strip()
    if task in {"x2i", "transfer"} or case_reference_images(case):
        return DEFAULT_X2I_TEMPLATE, DEFAULT_X2I_UNCOND
    return DEFAULT_T2I_TEMPLATE, DEFAULT_T2I_UNCOND


def _replace_visual_placeholders(
    text: str,
    image_strings: list[str],
    marker: str,
    *,
    append_unmatched: bool = True,
    remove_unresolved: bool = False,
) -> str:
    rendered = text
    for image_string in image_strings:
        if marker in rendered:
            rendered = rendered.replace(marker, image_string, 1)
        elif append_unmatched:
            rendered += image_string
    if marker in rendered:
        if remove_unresolved:
            return rendered.replace(marker, "")
        raise ValueError(f"raw prompt has unresolved visual placeholder: {marker}")
    return rendered


def _raw_uncond_prompt(prefix: str, image_strings: list[str], supervised_start: str) -> str:
    match = re.search(
        r"(.*?)\s*(Scene Description:|Instruction:|Robot Arm Type:|Image Style:)",
        prefix,
        re.DOTALL,
    )
    uncond_text = match.group(1).strip() if match else prefix.strip().split("\n")[0]
    uncond_text = uncond_text.removeprefix(SPECIAL_TOKENS["BOS"]).strip()

    robot_match = re.search(r"(Robot Arm Type:[^.]*\.)\s*", prefix)
    if robot_match:
        uncond_text = f"{uncond_text} {robot_match.group(1).strip()}".strip()

    has_view_chunk = "<|extra_10|>" in prefix
    if has_view_chunk:
        uncond_text = f"{uncond_text} <|extra_10|>".strip()
    view_chunk_suffix = "<|extra_11|>" if has_view_chunk else ""
    return (
        SPECIAL_TOKENS["BOS"]
        + uncond_text
        + "".join(image_strings)
        + view_chunk_suffix
        + supervised_start
    )


def _render_raw_case(
    case: dict[str, Any],
    raw_prompt: str,
    image_strings: list[str],
) -> tuple[str, str]:
    marker = case_visual_placeholder(case)
    supervised_start = case_supervised_start(case)
    prefix = raw_prompt.split(supervised_start, 1)[0] if supervised_start in raw_prompt else raw_prompt
    prompt = _replace_visual_placeholders(prefix, image_strings, marker) + supervised_start
    explicit_uncond = case_uncond_prompt(case)
    if explicit_uncond is not None:
        uncond = _replace_visual_placeholders(
            explicit_uncond,
            image_strings,
            marker,
            append_unmatched=False,
            remove_unresolved=True,
        )
    else:
        uncond = _raw_uncond_prompt(prefix, image_strings, supervised_start)
    return prompt, uncond


def render_case(
    cfg,
    case: dict[str, Any],
    *,
    tokenizer,
    vision_tokenizer,
    height: int,
    width: int,
    da3_artifact_dir: str | Path | None = None,
) -> tuple[str, str]:
    image_strings: list[str] = []
    references = case_reference_images(case)
    if references:
        from xr_u0_ar.depth_preprocess import open_reference_image
        from xr_u0_ar.image_tokens import build_image

        image_cfg = SimpleNamespace(image_area=int(cfg_get(cfg, "source_image_area", cfg_get(cfg, "image_area", 1024 * 1024))))
        task = task_type(cfg, case)
        image_strings = []
        for index, path in enumerate(references):
            artifact_dir = None
            if da3_artifact_dir is not None:
                artifact_dir = Path(da3_artifact_dir)
                if len(references) > 1:
                    artifact_dir = artifact_dir / f"reference_{index}"
            image_strings.append(
                build_image(
                    open_reference_image(path, task_type=task, config=cfg, case=case, artifact_dir=artifact_dir),
                    image_cfg,
                    tokenizer,
                    vision_tokenizer,
                )
            )
    image_prefix = "".join(image_strings)

    raw_prompt = case_raw_prompt(case)
    if raw_prompt is not None:
        prompt, uncond = _render_raw_case(case, raw_prompt, image_strings)
        return (
            ensure_anchor(prompt, height=height, width=width),
            ensure_anchor(uncond, height=height, width=width),
        )

    prompt_text = str(case.get("prompt") or case.get("text") or "")
    if not prompt_text:
        raise ValueError(f"case {case.get('id')}: missing prompt/text")
    template, uncond_template = template_pair(cfg, case)
    prompt = template.replace("<|IMAGE|>", image_prefix).format(
        task=task_type(cfg, case).lower(),
        task_type=task_type(cfg, case),
        image=image_prefix,
        question=prompt_text,
        text=prompt_text,
    )
    uncond = uncond_template.replace("<|IMAGE|>", image_prefix).format(
        task=task_type(cfg, case).lower(),
        task_type=task_type(cfg, case),
        image=image_prefix,
        question=prompt_text,
        text=prompt_text,
    )
    return (
        ensure_anchor(prompt, height=height, width=width),
        ensure_anchor(uncond, height=height, width=width),
    )


def _reference_preprocess_metadata(
    cfg,
    case: dict[str, Any],
    *,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    from xr_u0_ar.depth_preprocess import depth_options_from_config, depth_preprocess_metadata

    options = depth_options_from_config(cfg, case)
    return depth_preprocess_metadata(task_type(cfg, case), options, artifact_dir=artifact_dir)


def _length_summary(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "max": 0, "avg": 0.0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "avg": sum(lengths) / len(lengths),
    }


def _write_timing(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state(path: str) -> dict:
    state_path = Path(path)
    if state_path.name.endswith(".index.json"):
        from safetensors.torch import load_file as load_safetensors

        index = json.loads(state_path.read_text(encoding="utf-8"))
        state = {}
        for shard_name in sorted(set(index.get("weight_map", {}).values())):
            state.update(load_safetensors(state_path.parent / shard_name, device="cpu"))
        return state
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file as load_safetensors

        return load_safetensors(path, device="cpu")
    import torch

    return torch.load(path, map_location="cpu")


def is_integrated_flashar_state(path: str) -> bool:
    state_path = Path(path)
    if state_path.name.endswith(".index.json"):
        index = json.loads(state_path.read_text(encoding="utf-8"))
        return any(key.startswith("backbone.") for key in index.get("weight_map", {}))
    if state_path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(state_path, framework="pt", device="cpu") as handle:
            return any(key.startswith("backbone.") for key in handle.keys())
    return False


def build_empty_backbone(config, torch_dtype):
    import torch
    from transformers.modeling_utils import no_init_weights
    from xr_u0_ar.modeling_unis import UNISForCausalLM

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        with no_init_weights():
            return UNISForCausalLM(config)
    finally:
        torch.set_default_dtype(default_dtype)


def _module_device(module) -> Any:
    try:
        return next(module.parameters()).device
    except Exception:
        return "cpu"


def _normal_device_name(value: Any) -> str:
    text = str(value)
    if text.isdigit():
        return f"cuda:{text}"
    return text


def _cuda_devices_from_max_memory(max_memory: Any) -> list[str]:
    if not isinstance(max_memory, dict):
        return []
    devices: list[str] = []
    for key in max_memory:
        name = _normal_device_name(key)
        if name.startswith("cuda"):
            devices.append(name)
    return devices


def _visible_cuda_devices(max_memory: Any = None) -> list[str]:
    devices = _cuda_devices_from_max_memory(max_memory)
    if devices:
        return devices
    import torch

    if not torch.cuda.is_available():
        return []
    return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]


def _balanced_flashar_device_map(
    config,
    *,
    flashar_device: str,
    vertical_layers: int,
    max_memory: Any = None,
) -> dict[str, str]:
    devices = _visible_cuda_devices(max_memory)
    if not devices:
        devices = [flashar_device]
    num_layers = int(config.num_hidden_layers)
    device_map: dict[str, str] = {
        "backbone.embed_tokens": devices[0],
        "backbone.norm": devices[-1],
        "horizontal_head": flashar_device,
        "vertical_head": flashar_device,
        "hv_gate_mlp": flashar_device,
        "hv_gate_corner": flashar_device,
    }
    for layer_idx in range(num_layers):
        device_idx = min((layer_idx * len(devices)) // max(num_layers, 1), len(devices) - 1)
        device_map[f"backbone.layers.{layer_idx}"] = devices[device_idx]
    for layer_idx in range(int(vertical_layers)):
        device_map[f"vertical_block.{layer_idx}"] = flashar_device
    device_map["vertical_norm"] = flashar_device
    return device_map


def _resolve_flashar_device_map(
    cfg,
    config,
    *,
    flashar_device: str,
    vertical_layers: int,
) -> Any:
    device_map = cfg_get(cfg, "device_map", None)
    if device_map in (None, "", "none", "None"):
        return None
    max_memory = cfg_get(cfg, "max_memory", cfg_get(cfg, "model_max_memory", None))
    if isinstance(device_map, str) and device_map in {"auto", "balanced", "balanced_low_0"}:
        return _balanced_flashar_device_map(
            config,
            flashar_device=flashar_device,
            vertical_layers=vertical_layers,
            max_memory=max_memory,
        )
    return device_map


def _flashar_device_metadata(model, cfg, resolved_device_map: Any) -> dict[str, Any]:
    metadata = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(cfg_get(cfg, "device", "cuda:0")),
        "flashar_device": str(cfg_get(cfg, "flashar_device", cfg_get(cfg, "device", "cuda:0"))),
        "device_map": _jsonable(cfg_get(cfg, "device_map", None)),
        "resolved_device_map": _jsonable(resolved_device_map),
        "max_memory": _jsonable(cfg_get(cfg, "max_memory", cfg_get(cfg, "model_max_memory", None))),
        "input_device": str(_module_device(model.backbone.embed_tokens)),
    }
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        metadata["hf_device_map"] = _jsonable(hf_device_map)
    return metadata


def _write_flashar_eager_sidecar(
    out_path: Path,
    *,
    grid,
    height: int,
    width: int,
    metadata: dict[str, Any],
) -> Path:
    token_min = int(grid.min().item()) if grid.numel() else None
    token_max = int(grid.max().item()) if grid.numel() else None
    payload = {
        "backend": "xr_u0_flashar_eager",
        "output_path": str(out_path),
        "height": int(height),
        "width": int(width),
        "grid_shape": [int(x) for x in grid.shape],
        "n_visual_expected": int(height * width),
        "n_visual_actual": int(grid.numel()),
        "visual_token_complete": int(grid.numel()) == int(height * width),
        "token_min": token_min,
        "token_max": token_max,
        **metadata,
    }
    sidecar = out_path.with_suffix(out_path.suffix + ".flashar_eager.json")
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar


def discover_flashar_state(cfg, model_dir: str | Path | None = None) -> str:
    from xr_u0_ar.hub_paths import resolve_local_file, resolve_model_path, hub_kwargs_from_config

    if model_dir is None:
        model_dir = resolve_model_path(path_get(cfg, "model_path", "model_dir"), **hub_kwargs_from_config(cfg))
    model_dir = Path(model_dir)
    explicit = cfg_get(cfg, "flashar_state", None)
    if explicit:
        return resolve_local_file(explicit, kind="FlashAR state", base_dir=model_dir)
    for name in (
        "flashar.safetensors",
        "flashar.pt",
        "pytorch_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
    ):
        candidate = model_dir / name
        if candidate.exists():
            return str(candidate)
    raise AttributeError(
        "config must define flashar_state, or the FlashAR model directory must "
        "contain flashar.safetensors/flashar.pt"
    )


def build_flashar_eager_model(
    cfg,
    config,
    *,
    model_dir: str,
    torch_dtype,
    visual_offset: int,
    vertical_layers: int,
    vertical_start_layer: int,
    flashar_state: str,
):
    import torch
    from xr_u0_ar.modeling_unis import UNISForCausalLM
    from xr_u0_flashar.model import UNISFlashAR

    flashar_device = str(cfg_get(cfg, "flashar_device", cfg_get(cfg, "device", "cuda:0")))
    resolved_device_map = _resolve_flashar_device_map(
        cfg,
        config,
        flashar_device=flashar_device,
        vertical_layers=vertical_layers,
    )
    if resolved_device_map is None:
        device = torch.device(str(cfg_get(cfg, "device", "cuda:0")))
        if is_integrated_flashar_state(flashar_state):
            backbone = build_empty_backbone(config, torch_dtype)
        else:
            backbone = UNISForCausalLM.from_pretrained(
                model_dir,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation=str(cfg_get(cfg, "attn_implementation", "eager")),
            )
        backbone = backbone.to(device)
        model = UNISFlashAR(
            pretrained_backbone=backbone.model,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=-100,
            mask_token_id=config.pad_token_id,
            visual_token_offset=visual_offset,
            use_vertical_block=True,
            vertical_layers=vertical_layers,
            vertical_start_layer=vertical_start_layer,
        ).to(device=device, dtype=torch_dtype)
        model.load_state_dict(load_state(flashar_state), strict=True)
        model.eval()
        return model, _flashar_device_metadata(model, cfg, resolved_device_map)

    if not is_integrated_flashar_state(flashar_state):
        raise ValueError(
            "FlashAR eager device_map currently requires an integrated checkpoint "
            "with backbone.* keys."
        )

    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

    with init_empty_weights():
        backbone = UNISForCausalLM(config)
        model = UNISFlashAR(
            pretrained_backbone=backbone.model,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=-100,
            mask_token_id=config.pad_token_id,
            visual_token_offset=visual_offset,
            use_vertical_block=True,
            vertical_layers=vertical_layers,
            vertical_start_layer=vertical_start_layer,
        )
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=flashar_state,
        device_map=resolved_device_map,
        max_memory=cfg_get(cfg, "max_memory", cfg_get(cfg, "model_max_memory", None)),
        dtype=torch_dtype,
        no_split_module_classes=["UNISDecoderLayer"],
    )
    model.eval()
    return model, _flashar_device_metadata(model, cfg, resolved_device_map)


def run_eager(cfg) -> None:
    import torch
    from xr_u0_ar.configuration_unis import UNISConfig
    from xr_u0_ar.hub_paths import (
        hub_kwargs_from_config,
        resolve_model_path,
    )
    from xr_u0_ar.utils import build_text_tokenizer, register_transformers
    from xr_u0_ar.vision_tokenizer import build_vision_tokenizer

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[str(cfg_get(cfg, "dtype", "bf16"))]

    register_transformers()
    hub_kwargs = hub_kwargs_from_config(cfg)
    model_dir = resolve_model_path(path_get(cfg, "model_path", "model_dir"), **hub_kwargs)
    config = UNISConfig.from_pretrained(model_dir, trust_remote_code=True)
    config._attn_implementation = str(cfg_get(cfg, "attn_implementation", "eager"))
    flashar_state = discover_flashar_state(cfg, model_dir=model_dir)
    visual_offset = int(config.eoi_token_id) + 1
    vertical_layers = int(cfg_get(cfg, "vertical_layers", 4))
    vertical_start_layer = int(cfg_get(cfg, "vertical_start_layer", -1))
    if vertical_start_layer < 0:
        vertical_start_layer = int(config.num_hidden_layers) - vertical_layers
    model, device_metadata = build_flashar_eager_model(
        cfg,
        config,
        model_dir=model_dir,
        torch_dtype=torch_dtype,
        visual_offset=visual_offset,
        vertical_layers=vertical_layers,
        vertical_start_layer=vertical_start_layer,
        flashar_state=flashar_state,
    )
    input_device = _module_device(model.backbone.embed_tokens)

    tokenizer = build_text_tokenizer(path_get(cfg, "tokenizer_path", "tokenizer_dir"), **hub_kwargs)
    vision_tokenizer = build_vision_tokenizer(
        str(cfg_get(cfg, "vq_type", "ibq")),
        path_get(cfg, "vq_path", "vision_tokenizer_path", "vision_tokenizer_dir"),
        device=str(cfg_get(cfg, "vq_device", cfg_get(cfg, "vision_device", "cuda:0"))),
        **hub_kwargs,
    )
    out_dir = Path(str(cfg_get(cfg, "save_path", "outputs/flashar_eager")))
    timing_path = out_dir / "timing.json"
    started = time.perf_counter()
    saved_paths: list[str] = []
    errors: list[dict[str, str]] = []
    for case_id, case in normalize_cases(cfg.prompts):
        height, width = resolve_case_shape(cfg, case)
        da3_artifact_dir = out_dir / "da3_preprocess" / case_id
        prompt, uncond = render_case(
            cfg, case, tokenizer=tokenizer, vision_tokenizer=vision_tokenizer,
            height=height, width=width, da3_artifact_dir=da3_artifact_dir
        )
        out_path = out_dir / f"{case_id}.png"
        try:
            text_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(input_device)
            uncond_ids = tokenizer.encode(uncond, return_tensors="pt", add_special_tokens=False).to(input_device)
            grid = model.generate(
                height=height,
                width=width,
                device=input_device,
                text_input_ids=text_ids,
                unconditional_text_input_ids=uncond_ids,
                cfg_scale=float(cfg_get(cfg, "classifier_free_guidance", cfg_get(cfg, "cfg_scale", 3.0))),
                temperature=float(cfg_get(cfg, "temperature", 1.0)),
                top_k=int(cfg_get(cfg, "top_k", 5120)),
                top_p=float(cfg_get(cfg, "top_p", 1.0)),
            )
            decode_grid(grid, vision_tokenizer, visual_offset, out_path, clamp_min=True)
            _write_flashar_eager_sidecar(
                out_path,
                grid=grid,
                height=height,
                width=width,
                metadata={
                    "id": case.get("id"),
                    "task_type": task_type(cfg, case),
                    **device_metadata,
                    **_reference_preprocess_metadata(cfg, case, artifact_dir=da3_artifact_dir),
                },
            )
            saved_paths.append(str(out_path.resolve()))
            print(f"saved {out_path.resolve()}")
        except Exception as exc:
            errors.append({"id": case_id, "error": repr(exc)})
            raise
    timing = {
        "backend": "xr_u0_flashar_eager",
        "task_type": str(cfg_get(cfg, "task_type", "")),
        "status": "ok" if not errors else "partial",
        "n_inputs": len(normalize_cases(cfg.prompts)),
        "n_saved": len(saved_paths),
        "n_failed": len(errors),
        "saved_paths": saved_paths,
        "errors": errors,
        "total_seconds": time.perf_counter() - started,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **device_metadata,
    }
    _write_timing(timing_path, timing)
    print(f"saved {timing_path}")


def run_vllm(cfg) -> None:
    import os

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from xr_u0_ar.hub_paths import hub_kwargs_from_config
    from xr_u0_ar.vision_tokenizer import build_vision_tokenizer
    from xr_u0_flashar.vllm import LLM

    hub_kwargs = hub_kwargs_from_config(cfg)
    out_dir = Path(str(cfg_get(cfg, "save_path", "outputs/flashar_vllm")))
    cases = normalize_cases(cfg.prompts)
    if not cases:
        raise ValueError("config.prompts must contain at least one case")
    case_groups: dict[tuple[int, int], list[tuple[str, dict[str, Any]]]] = {}
    for case_id, case in cases:
        shape = resolve_case_shape(cfg, case)
        case_groups.setdefault(shape, []).append((case_id, case))
    unique_shapes = list(case_groups)

    llm = LLM.from_pretrained(
        path_get(cfg, "model_path", "model_dir"),
        tokenizer_dir=path_get(cfg, "tokenizer_path", "tokenizer_dir"),
        hf_revision=hub_kwargs.get("revision"),
        hf_cache_dir=hub_kwargs.get("cache_dir"),
        local_files_only=hub_kwargs.get("local_files_only"),
        tensor_parallel_size=int(cfg_get(cfg, "tensor_parallel_size", 1)),
        max_model_len=int(cfg_get(cfg, "max_model_len", 8192)),
        max_num_seqs=int(cfg_get(cfg, "max_num_seqs", 128)),
        max_num_batched_tokens=int(cfg_get(cfg, "max_num_batched_tokens", 20000)),
        gpu_memory_utilization=float(cfg_get(cfg, "gpu_memory_utilization", 0.85)),
        seed=int(cfg_get(cfg, "seed", 42)),
        enable_prefix_caching=bool(cfg_get(cfg, "enable_prefix_caching", False)),
        strict_visual_tokens=bool(cfg_get(cfg, "strict_visual_tokens", True)),
    )
    vision_tokenizer = build_vision_tokenizer(
        str(cfg_get(cfg, "vq_type", "ibq")),
        path_get(cfg, "vq_path", "vision_tokenizer_path", "vision_tokenizer_dir"),
        device=str(cfg_get(cfg, "vq_device", cfg_get(cfg, "vision_device", "cuda:0"))),
        **hub_kwargs,
    )
    render_groups: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    uncond_lengths: list[int] = []
    render_started = time.perf_counter()
    for (height, width), group_cases in case_groups.items():
        prompts: list[str] = []
        unconds: list[str] = []
        for case_id, case in group_cases:
            da3_artifact_dir = out_dir / "da3_preprocess" / case_id
            prompt, uncond = render_case(
                cfg, case, tokenizer=llm.tokenizer, vision_tokenizer=vision_tokenizer,
                height=height, width=width, da3_artifact_dir=da3_artifact_dir
            )
            prompts.append(prompt)
            unconds.append(uncond)
            prompt_lengths.append(len(llm.tokenizer.encode(prompt, add_special_tokens=False)))
            uncond_lengths.append(len(llm.tokenizer.encode(uncond, add_special_tokens=False)))
        render_groups.append(
            {
                "height": height,
                "width": width,
                "cases": group_cases,
                "prompts": prompts,
                "unconds": unconds,
            }
        )
    render_seconds = time.perf_counter() - render_started

    timing_path = out_dir / "timing.json"
    first_height, first_width = unique_shapes[0]
    run_metadata = {
        "max_model_len": int(cfg_get(cfg, "max_model_len", 8192)),
        "max_num_seqs": int(cfg_get(cfg, "max_num_seqs", 128)),
        "max_num_batched_tokens": int(cfg_get(cfg, "max_num_batched_tokens", 20000)),
        "tensor_parallel_size": int(cfg_get(cfg, "tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(cfg_get(cfg, "gpu_memory_utilization", 0.85)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "enable_prefix_caching": bool(cfg_get(cfg, "enable_prefix_caching", False)),
        "strict_visual_tokens": bool(cfg_get(cfg, "strict_visual_tokens", True)),
    }
    timing: dict[str, Any] = {
        "backend": "xr_u0_flashar_vllm",
        "task_type": str(cfg_get(cfg, "task_type", "")),
        "n_inputs": len(cases),
        "height": first_height if len(unique_shapes) == 1 else None,
        "width": first_width if len(unique_shapes) == 1 else None,
        "unique_shapes": [f"{h}x{w}" for h, w in unique_shapes],
        "shape_groups": [
            {
                "height": int(group["height"]),
                "width": int(group["width"]),
                "n_inputs": len(group["cases"]),
                "case_ids": [case_id for case_id, _ in group["cases"]],
            }
            for group in render_groups
        ],
        **run_metadata,
        "classifier_free_guidance": float(cfg_get(cfg, "classifier_free_guidance", cfg_get(cfg, "cfg_scale", 3.0))),
        "prompt_token_lengths": _length_summary(prompt_lengths),
        "uncond_token_lengths": _length_summary(uncond_lengths),
        "render_seconds": render_seconds,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    visual_offset = llm.tokenizer.encode("<|image end|>")[0] + 1
    batch_started = time.perf_counter()
    saved_paths: list[str] = []
    errors: list[dict[str, str]] = []
    generation_seconds = 0.0
    save_seconds = 0.0
    case_indices = {case_id: index for index, (case_id, _) in enumerate(cases)}
    generation_groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(render_groups):
        height = int(group["height"])
        width = int(group["width"])
        group_cases = group["cases"]
        prompts = group["prompts"]
        unconds = group["unconds"]
        try:
            generation_started = time.perf_counter()
            results = llm.generate(
                prompts,
                height=height,
                width=width,
                cfg_scale=float(cfg_get(cfg, "classifier_free_guidance", cfg_get(cfg, "cfg_scale", 3.0))),
                temperature=float(cfg_get(cfg, "temperature", 1.0)),
                top_k=int(cfg_get(cfg, "top_k", 5120)),
                top_p=float(cfg_get(cfg, "top_p", 1.0)),
                uncond_prompt=unconds,
                prompt_template="{text}",
            )
            group_generation_seconds = time.perf_counter() - generation_started
            generation_seconds += group_generation_seconds
        except Exception as exc:
            timing.update({
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "status": "failed",
                "error": repr(exc),
                "failed_group": {
                    "height": height,
                    "width": width,
                    "case_ids": [case_id for case_id, _ in group_cases],
                },
                "generation_seconds": generation_seconds,
                "total_seconds": time.perf_counter() - batch_started,
                "n_saved": len(saved_paths),
                "n_failed": len(cases) - len(saved_paths),
                "saved_paths": saved_paths,
                "errors": errors,
            })
            _write_timing(timing_path, timing)
            raise

        generation_groups.append(
            {
                "height": height,
                "width": width,
                "n_inputs": len(group_cases),
                "case_ids": [case_id for case_id, _ in group_cases],
                "generation_seconds": group_generation_seconds,
            }
        )
        save_started = time.perf_counter()
        for case_id, case in group_cases:
            result = results.pop(0)
            out_path = out_dir / f"{case_id}.png"
            da3_artifact_dir = out_dir / "da3_preprocess" / case_id
            try:
                decode_grid(result.grid, vision_tokenizer, visual_offset, out_path, clamp_min=True)
                write_flashar_audit(
                    out_path,
                    result,
                    height=height,
                    width=width,
                    metadata={
                        "id": case.get("id"),
                        "task_type": task_type(cfg, case),
                        "batch_index": case_indices[case_id],
                        "shape_group_index": group_index,
                        **run_metadata,
                        **_reference_preprocess_metadata(cfg, case, artifact_dir=da3_artifact_dir),
                    },
                )
                saved_paths.append(str(out_path.resolve()))
                print(f"saved {out_path.resolve()}")
            except Exception as exc:
                errors.append({"id": case_id, "error": repr(exc)})
        save_seconds += time.perf_counter() - save_started

    timing.update({
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "ok" if not errors else "partial",
        "generation_seconds": generation_seconds,
        "generation_groups": generation_groups,
        "save_seconds": save_seconds,
        "total_seconds": time.perf_counter() - batch_started,
        "n_saved": len(saved_paths),
        "n_failed": len(errors),
        "saved_paths": saved_paths,
        "errors": errors,
    })
    _write_timing(timing_path, timing)
    if errors:
        raise RuntimeError(f"{len(errors)} FlashAR vLLM outputs failed to save; see {timing_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Xiaomi-Robotics-U0-FlashAR inference from a Python config.")
    parser.add_argument("--cfg", required=True, help="Path to a Python config file.")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    backend = str(cfg_get(cfg, "backend", "vllm")).lower()
    if backend == "eager":
        run_eager(cfg)
    elif backend == "vllm":
        run_vllm(cfg)
    else:
        raise ValueError(f"unsupported backend for Xiaomi-Robotics-U0-FlashAR: {backend}")


if __name__ == "__main__":
    main()

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


SPECIAL_TOKENS = {
    "BOS": "<|extra_203|>",
    "EOS": "<|extra_204|>",
    "PAD": "<|endoftext|>",
    "EOL": "<|extra_200|>",
    "EOF": "<|extra_201|>",
    "TMS": "<|extra_202|>",
    "IMG": "<|image token|>",
    "BOI": "<|image start|>",
    "EOI": "<|image end|>",
    "BSS": "<|extra_100|>",
    "ESS": "<|extra_101|>",
    "BOG": "<|extra_60|>",
    "EOG": "<|extra_61|>",
    "BOC": "<|extra_50|>",
    "EOC": "<|extra_51|>",
}


DEFAULT_SAMPLING_PARAMS = {
    "use_cache": True,
    "text_top_k": 1024,
    "text_top_p": 0.9,
    "text_temperature": 1.0,
    "image_top_k": 5120,
    "image_top_p": 1.0,
    "image_temperature": 1.0,
    "top_k": 131072,
    "top_p": 1.0,
    "temperature": 1.0,
    "num_beams_per_group": 1,
    "num_beam_groups": 1,
    "diversity_penalty": 0.0,
    "max_new_tokens": 5120,
    "guidance_scale": 1.0,
    "use_differential_sampling": True,
    "do_sample": True,
    "num_beams": 1,
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


def _model_input_device(model) -> Any:
    try:
        embeddings = model.get_input_embeddings()
        return next(embeddings.parameters()).device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return getattr(model, "device", "cpu")


def _hf_device_metadata(model, cfg) -> dict[str, Any]:
    configured_device_map = cfg_get(cfg, "device_map", cfg_get(cfg, "model_device", "auto"))
    configured_max_memory = cfg_get(cfg, "max_memory", cfg_get(cfg, "model_max_memory", None))
    metadata = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_map": _jsonable(configured_device_map),
        "max_memory": _jsonable(configured_max_memory),
        "input_device": str(_model_input_device(model)),
    }
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        metadata["hf_device_map"] = _jsonable(hf_device_map)
    return metadata


def _ar_vllm_run_metadata(cfg) -> dict[str, Any]:
    return {
        "tensor_parallel_size": int(cfg_get(cfg, "tensor_parallel_size", 1)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "max_num_seqs": int(cfg_get(cfg, "max_num_seqs", 2)),
        "max_num_batched_tokens": int(cfg_get(cfg, "max_num_batched_tokens", 26000)),
        "gpu_memory_utilization": float(cfg_get(cfg, "gpu_memory_utilization", 0.85)),
        "enable_prefix_caching": bool(cfg_get(cfg, "enable_prefix_caching", False)),
        "enable_chunked_prefill": bool(cfg_get(cfg, "enable_chunked_prefill", False)),
    }


def path_get(cfg, *names: str) -> str:
    for name in names:
        value = getattr(cfg, name, None)
        if value:
            return str(value)
    raise AttributeError(f"config must define one of: {', '.join(names)}")


def normalize_cases(prompts: Any) -> list[tuple[str, Any]]:
    if isinstance(prompts, dict):
        return [(str(k), v) for k, v in prompts.items()]
    if isinstance(prompts, list):
        return [(f"{idx:03d}", item) for idx, item in enumerate(prompts)]
    raise TypeError("config.prompts must be a dict or a list")


def case_prompt(case: Any) -> str:
    if isinstance(case, dict):
        value = case.get("prompt") or case.get("text")
        if not value:
            raise ValueError("case is missing prompt/text")
        return str(value)
    return str(case)


def case_reference_images(case: Any) -> list[str]:
    if not isinstance(case, dict):
        return []
    value = case.get("reference_image") or case.get("image") or case.get("image_list")
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def case_raw_prompt(case: Any) -> str | None:
    if not isinstance(case, dict):
        return None
    value = case.get("text_prompt")
    if value is None:
        return None
    return str(value)


def case_uncond_prompt(case: Any) -> str | None:
    if not isinstance(case, dict):
        return None
    value = case.get("uncond_prompt")
    if value is None:
        return None
    return str(value)


def case_visual_placeholder(case: Any) -> str:
    if isinstance(case, dict) and case.get("visual_placeholder"):
        return str(case["visual_placeholder"])
    return "<|VIS_PLH|>"


def case_supervised_start(case: Any) -> str:
    if isinstance(case, dict) and case.get("supervised_start"):
        return str(case["supervised_start"])
    return SPECIAL_TOKENS["BSS"]


def case_task_type(cfg, case: Any) -> str:
    if isinstance(case, dict) and case.get("task_type"):
        return str(case["task_type"])
    return str(cfg_get(cfg, "task_type", "T2I"))


def special_token_ids(tokenizer) -> dict[str, int]:
    return {name: tokenizer.encode(value)[0] for name, value in SPECIAL_TOKENS.items()}


def ensure_bos(input_ids, bos_token_id: int):
    if input_ids.numel() == 0 or input_ids[0, 0] != bos_token_id:
        import torch

        bos = torch.tensor(
            [[bos_token_id]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([bos, input_ids], dim=1)
    return input_ids


def _parse_aspect_ratio(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("target_aspect_ratio tuple/list must be (width, height)")
        width, height = float(value[0]), float(value[1])
        if width <= 0 or height <= 0:
            raise ValueError("target_aspect_ratio values must be positive")
        return width / height
    if isinstance(value, (int, float)):
        ratio = float(value)
        if ratio <= 0:
            raise ValueError("target_aspect_ratio must be positive")
        return ratio

    text = str(value).strip().lower().replace(" ", "")
    for sep in (":", "/", "*", "x"):
        if sep in text:
            left, right = text.split(sep, 1)
            width, height = float(left), float(right)
            if width <= 0 or height <= 0:
                raise ValueError("target_aspect_ratio values must be positive")
            return width / height
    ratio = float(text)
    if ratio <= 0:
        raise ValueError("target_aspect_ratio must be positive")
    return ratio


def _round_to_factor(value: float, factor: int) -> int:
    rounded = int((value + factor / 2) // factor) * factor
    return max(factor, rounded)


def _target_shape_from_aspect_ratio(
    aspect_ratio: Any,
    *,
    image_area: int,
    ds_factor: int = 16,
) -> tuple[int, int]:
    ratio = _parse_aspect_ratio(aspect_ratio)
    if ratio is None:
        raise ValueError("target_aspect_ratio must not be None")
    height_px = _round_to_factor((image_area / ratio) ** 0.5, ds_factor)
    width_px = _round_to_factor(height_px * ratio, ds_factor)
    return height_px // ds_factor, width_px // ds_factor


def _target_shape(cfg, image_area: int) -> tuple[int | None, int | None]:
    target_height = cfg_get(cfg, "target_height", None)
    target_width = cfg_get(cfg, "target_width", None)
    if (target_height is None) != (target_width is None):
        raise ValueError("target_height and target_width must be set together")
    if target_height is not None:
        return int(target_height), int(target_width)

    target_aspect_ratio = cfg_get(cfg, "target_aspect_ratio", None)
    if target_aspect_ratio is None:
        return None, None
    return _target_shape_from_aspect_ratio(target_aspect_ratio, image_area=image_area)


def runtime_config(cfg) -> SimpleNamespace:
    sampling = DEFAULT_SAMPLING_PARAMS.copy()
    sampling.update(cfg_get(cfg, "sampling_params", {}))
    image_area = int(cfg_get(cfg, "image_area", 1024 * 1024))
    target_height, target_width = _target_shape(cfg, image_area)
    return SimpleNamespace(
        streaming=bool(cfg_get(cfg, "streaming", False)),
        task_type=str(cfg_get(cfg, "task_type", "T2I")).lower(),
        template=str(cfg.template),
        unc_prompt=str(cfg.unc_prompt),
        classifier_free_guidance=float(cfg_get(cfg, "classifier_free_guidance", 3.0)),
        unconditional_type=str(cfg_get(cfg, "unconditional_type", "no_text")),
        image_cfg_scale=float(cfg_get(cfg, "image_cfg_scale", 1.0)),
        image_area=image_area,
        target_height=target_height,
        target_width=target_width,
        force_same_image_size=bool(cfg_get(cfg, "force_same_image_size", True)),
        stop_on_image_end=bool(cfg_get(cfg, "stop_on_image_end", False)),
        max_hw_digits=int(cfg_get(cfg, "max_hw_digits", 3)),
        sampling_params=sampling,
        input_image_type=str(cfg_get(cfg, "input_image_type", "depth")),
        da3_model_path=cfg_get(cfg, "da3_model_path", None),
        da3_device=str(cfg_get(cfg, "da3_device", "cuda:0")),
        da3_min_depth=float(cfg_get(cfg, "da3_min_depth", 0.7)),
        da3_max_depth=float(cfg_get(cfg, "da3_max_depth", 2.0)),
        da3_process_res=int(cfg_get(cfg, "da3_process_res", 504)),
    )


def _build_reference_strings(
    cfg_obj: SimpleNamespace,
    case: Any,
    tokenizer,
    vision_tokenizer,
    *,
    artifact_dir: str | Path | None = None,
) -> list[str]:
    references = case_reference_images(case)
    if not references:
        return []

    from xr_u0_ar.depth_preprocess import open_reference_image
    from xr_u0_ar.image_tokens import build_image

    image_cfg = SimpleNamespace(image_area=cfg_obj.image_area)
    task = case_task_type(cfg_obj, case)
    image_strings: list[str] = []
    for index, path in enumerate(references):
        reference_artifact_dir = None
        if artifact_dir is not None:
            reference_artifact_dir = Path(artifact_dir)
            if len(references) > 1:
                reference_artifact_dir = reference_artifact_dir / f"reference_{index}"
        image_strings.append(
            build_image(
                open_reference_image(
                    path,
                    task_type=task,
                    config=cfg_obj,
                    case=case,
                    artifact_dir=reference_artifact_dir,
                ),
                image_cfg,
                tokenizer,
                vision_tokenizer,
            )
        )
    return image_strings


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
    case: Any,
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
    cfg_obj: SimpleNamespace,
    case: Any,
    tokenizer,
    vision_tokenizer,
    *,
    da3_artifact_dir: str | Path | None = None,
) -> tuple[str, str]:
    image_strings = _build_reference_strings(
        cfg_obj,
        case,
        tokenizer,
        vision_tokenizer,
        artifact_dir=da3_artifact_dir,
    )
    raw_prompt = case_raw_prompt(case)
    if raw_prompt is not None:
        return _render_raw_case(case, raw_prompt, image_strings)

    prompt_text = case_prompt(case)
    image_prefix = "".join(image_strings)
    task_type = case_task_type(cfg_obj, case)
    template = cfg_obj.template.replace("<|IMAGE|>", image_prefix)
    uncond_template = cfg_obj.unc_prompt.replace("<|IMAGE|>", image_prefix)
    prompt = template.format(
        task=task_type.lower(),
        task_type=task_type,
        image=image_prefix,
        question=prompt_text,
        text=prompt_text,
    )
    uncond = uncond_template.format(
        task=task_type.lower(),
        task_type=task_type,
        image=image_prefix,
        question=prompt_text,
        text=prompt_text,
    )
    return prompt, uncond


def _reference_preprocess_metadata(
    cfg_obj: SimpleNamespace,
    case: Any,
    *,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    from xr_u0_ar.depth_preprocess import depth_options_from_config, depth_preprocess_metadata

    case_cfg = case if isinstance(case, dict) else None
    options = depth_options_from_config(cfg_obj, case_cfg)
    return depth_preprocess_metadata(case_task_type(cfg_obj, case), options, artifact_dir=artifact_dir)


def build_vllm_model(cfg, tokenizer):
    from vllm import LLM
    from xr_u0_ar.hub_paths import hub_kwargs_from_config, resolve_model_path, resolve_tokenizer_path

    hub_kwargs = hub_kwargs_from_config(cfg)
    model_path = resolve_model_path(path_get(cfg, "model_path", "model_dir"), **hub_kwargs)
    tokenizer_path = resolve_tokenizer_path(path_get(cfg, "tokenizer_path", "tokenizer_dir"), **hub_kwargs)
    resolution_map = {
        tokenizer.encode(ch)[0]: ch
        for ch in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*"]
    }
    model = LLM(
        model_path,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        dtype=cfg_get(cfg, "dtype", "auto"),
        tensor_parallel_size=int(cfg_get(cfg, "tensor_parallel_size", 1)),
        gpu_memory_utilization=float(cfg_get(cfg, "gpu_memory_utilization", 0.85)),
        disable_log_stats=not bool(cfg_get(cfg, "enable_log_stats", False)),
        enable_chunked_prefill=bool(cfg_get(cfg, "enable_chunked_prefill", False)),
        enable_prefix_caching=bool(cfg_get(cfg, "enable_prefix_caching", False)),
        max_num_batched_tokens=int(cfg_get(cfg, "max_num_batched_tokens", 26000)),
        max_num_seqs=int(cfg_get(cfg, "max_num_seqs", 2)),
        seed=int(cfg_get(cfg, "seed", 42)),
        generation_config="vllm",
        scheduler_cls="vllm.v1.core.sched.batch_scheduler.Scheduler",
        compilation_config=cfg_get(
            cfg,
            "compilation_config",
            {
                "full_cuda_graph": True,
                "backend": "cudagraph",
            },
        ),
        additional_config={
            "boi_token_id": tokenizer.encode("<|image start|>")[0],
            "soi_token_id": tokenizer.encode("<|image token|>")[0],
            "eol_token_id": tokenizer.encode("<|extra_200|>")[0],
            "eoi_token_id": tokenizer.encode("<|image end|>")[0],
            "resolution_map": resolution_map,
        },
    )
    model.set_tokenizer(tokenizer)
    return model


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


def _is_video_task(cfg_obj: SimpleNamespace) -> bool:
    return cfg_obj.task_type.replace("_", " ").replace("-", " ") == "video gen"


def _save_generation_output(
    cfg,
    cfg_obj: SimpleNamespace,
    tokens: Any,
    tokenizer,
    vision_tokenizer,
    output_path: Path,
    *,
    case: Any,
    metadata: dict[str, Any],
    require_image: bool = False,
):
    if _is_video_task(cfg_obj):
        from xr_u0_ar.video_outputs import save_video_sequence

        video_fps = int(cfg_get(cfg, "output_fps", 3))
        video_metadata = {
            **metadata,
            "cfg_decay": cfg_obj.sampling_params.get("cfg_decay", "none"),
            "cfg_min": cfg_obj.sampling_params.get("cfg_min", 1.0),
            "cfg_total_images": cfg_obj.sampling_params.get("cfg_total_images", 1),
            "max_new_tokens": cfg_obj.sampling_params.get("max_new_tokens"),
            "video_save_fps": video_fps,
        }
        if isinstance(case, dict):
            for key in ("legacy_sample_index", "legacy_output_name"):
                if key in case:
                    video_metadata[key] = case[key]
        return save_video_sequence(
            tokens,
            tokenizer,
            vision_tokenizer,
            output_path.with_suffix(".mp4"),
            metadata=video_metadata,
            context_image_paths=case_reference_images(case),
            fps=video_fps,
            require_image=require_image,
        )

    from xr_u0_ar.outputs import save_generated_sequence

    return save_generated_sequence(
        tokens,
        tokenizer,
        vision_tokenizer,
        output_path,
        metadata=metadata,
        require_image=require_image,
    )


def run_eager(cfg) -> None:
    import torch
    from xr_u0_ar.hub_paths import hub_kwargs_from_config
    from xr_u0_ar.generation import generate
    from xr_u0_ar.utils import build_unis_ar

    cfg_obj = runtime_config(cfg)
    hub_kwargs = hub_kwargs_from_config(cfg)
    model, tokenizer, vision_tokenizer = build_unis_ar(
        path_get(cfg, "model_path", "model_dir"),
        path_get(cfg, "tokenizer_path", "tokenizer_dir"),
        path_get(cfg, "vq_path", "vision_tokenizer_path", "vision_tokenizer_dir"),
        model_device=cfg_get(cfg, "model_device", "auto"),
        device_map=cfg_get(cfg, "device_map", None),
        max_memory=cfg_get(cfg, "max_memory", cfg_get(cfg, "model_max_memory", None)),
        vision_device=str(cfg_get(cfg, "vq_device", cfg_get(cfg, "vision_device", "cuda:0"))),
        vision_type=str(cfg_get(cfg, "vq_type", "ibq")),
        torch_dtype=cfg_get(cfg, "torch_dtype", "auto"),
        attn_implementation=str(cfg_get(cfg, "attn_implementation", "eager")),
        **hub_kwargs,
    )
    cfg_obj.special_token_ids = special_token_ids(tokenizer)
    input_device = _model_input_device(model)
    device_metadata = _hf_device_metadata(model, cfg)

    out_dir = Path(str(cfg_get(cfg, "save_path", "outputs/ar_eager")))
    for case_id, case in normalize_cases(cfg.prompts):
        da3_artifact_dir = out_dir / "da3_preprocess" / case_id
        prompt, uncond = render_case(
            cfg_obj,
            case,
            tokenizer,
            vision_tokenizer,
            da3_artifact_dir=da3_artifact_dir,
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(input_device)
        uncond_ids = tokenizer.encode(uncond, return_tensors="pt", add_special_tokens=False).to(input_device)
        input_ids = ensure_bos(input_ids, cfg_obj.special_token_ids["BOS"])
        uncond_ids = ensure_bos(uncond_ids, cfg_obj.special_token_ids["BOS"])
        for result_tokens in generate(
            cfg_obj,
            model,
            tokenizer,
            input_ids,
            uncond_ids,
            force_same_image_size=cfg_obj.force_same_image_size,
        ):
            saved = _save_generation_output(
                cfg,
                cfg_obj,
                result_tokens,
                tokenizer,
                vision_tokenizer,
                out_dir / f"{case_id}.png",
                case=case,
                metadata={
                    "id": case_id,
                    "backend": "ar_eager",
                    **device_metadata,
                    **_reference_preprocess_metadata(cfg_obj, case, artifact_dir=da3_artifact_dir),
                },
                require_image=_is_video_task(cfg_obj),
            )
            print(f"saved {saved.primary_path}")
            break


def run_vllm_batch(
    cfg,
    cfg_obj: SimpleNamespace,
    tokenizer,
    vision_tokenizer,
    model,
    out_dir: Path,
) -> None:
    from xr_u0_ar.vllm_generation import generate_batch as vllm_generate_batch

    cases = normalize_cases(cfg.prompts)
    if not cases:
        raise ValueError("config.prompts must contain at least one case")
    input_ids_batch = []
    uncond_ids_batch = []
    prompt_lengths: list[int] = []
    uncond_lengths: list[int] = []
    render_started = time.perf_counter()
    for case_id, case in cases:
        da3_artifact_dir = out_dir / "da3_preprocess" / case_id
        prompt, uncond = render_case(
            cfg_obj,
            case,
            tokenizer,
            vision_tokenizer,
            da3_artifact_dir=da3_artifact_dir,
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        uncond_ids = tokenizer.encode(uncond, return_tensors="pt", add_special_tokens=False)
        input_ids = ensure_bos(input_ids, cfg_obj.special_token_ids["BOS"])
        uncond_ids = ensure_bos(uncond_ids, cfg_obj.special_token_ids["BOS"])
        prompt_lengths.append(int(input_ids.shape[-1]))
        uncond_lengths.append(int(uncond_ids.shape[-1]))
        input_ids_batch.append(input_ids)
        uncond_ids_batch.append(uncond_ids)
    render_seconds = time.perf_counter() - render_started

    timing_path = out_dir / "timing.json"
    run_metadata = _ar_vllm_run_metadata(cfg)
    timing: dict[str, Any] = {
        "backend": "ar_vllm",
        "task_type": cfg_obj.task_type,
        "n_inputs": len(cases),
        **run_metadata,
        "classifier_free_guidance": cfg_obj.classifier_free_guidance,
        "target_height": cfg_obj.target_height,
        "target_width": cfg_obj.target_width,
        "image_area": cfg_obj.image_area,
        "prompt_token_lengths": _length_summary(prompt_lengths),
        "uncond_token_lengths": _length_summary(uncond_lengths),
        "render_seconds": render_seconds,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    batch_started = time.perf_counter()
    try:
        generation_started = time.perf_counter()
        result_batches = vllm_generate_batch(
            cfg_obj,
            model,
            tokenizer,
            input_ids_batch,
            uncond_ids_batch,
        )
        generation_seconds = time.perf_counter() - generation_started
    except Exception as exc:
        total_seconds = time.perf_counter() - batch_started
        timing.update({
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "failed",
            "error": repr(exc),
            "total_seconds": total_seconds,
            "n_saved": 0,
            "n_failed": len(cases),
        })
        _write_timing(timing_path, timing)
        raise

    saved_paths: list[str] = []
    errors: list[dict[str, str]] = []
    first_saved_seconds: float | None = None
    save_started = time.perf_counter()
    for batch_index, ((case_id, _), result_tokens) in enumerate(zip(cases, result_batches)):
        case = cases[batch_index][1]
        da3_artifact_dir = out_dir / "da3_preprocess" / case_id
        try:
            saved = _save_generation_output(
                cfg,
                cfg_obj,
                result_tokens,
                tokenizer,
                vision_tokenizer,
                out_dir / f"{case_id}.png",
                case=case,
                metadata={
                    "id": case_id,
                    "backend": "ar_vllm",
                    "batch_index": batch_index,
                    "batch_n": len(cases),
                    **run_metadata,
                    **_reference_preprocess_metadata(cfg_obj, case, artifact_dir=da3_artifact_dir),
                },
                require_image=True,
            )
        except Exception as exc:
            errors.append({"id": case_id, "error": repr(exc)})
            continue
        if first_saved_seconds is None:
            first_saved_seconds = time.perf_counter() - batch_started
        saved_paths.append(str(saved.primary_path))
        print(f"saved {saved.primary_path}")
    save_seconds = time.perf_counter() - save_started
    total_seconds = time.perf_counter() - batch_started
    n_saved = len(saved_paths)
    n_failed = len(cases) - n_saved
    timing.update({
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "ok" if n_failed == 0 else "partial",
        "generation_seconds": generation_seconds,
        "save_seconds": save_seconds,
        "total_seconds": total_seconds,
        "first_image_saved_seconds": first_saved_seconds,
        "n_saved": n_saved,
        "n_failed": n_failed,
        "generation_img_per_second": n_saved / generation_seconds if generation_seconds else 0.0,
        "total_img_per_second": n_saved / total_seconds if total_seconds else 0.0,
        "avg_generation_seconds_per_image": generation_seconds / len(cases),
        "avg_total_seconds_per_image": total_seconds / len(cases),
        "sample_outputs": saved_paths[:8],
        "errors": errors,
    })
    _write_timing(timing_path, timing)
    print(f"saved {timing_path}")
    if errors:
        raise RuntimeError(f"{len(errors)} batch output(s) failed to save; see {timing_path}")


def run_vllm(cfg) -> None:
    import torch
    from xr_u0_ar.hub_paths import hub_kwargs_from_config
    from xr_u0_ar.utils import build_text_tokenizer
    from xr_u0_ar.vision_tokenizer import build_vision_tokenizer

    cfg_obj = runtime_config(cfg)
    hub_kwargs = hub_kwargs_from_config(cfg)
    tokenizer = build_text_tokenizer(path_get(cfg, "tokenizer_path", "tokenizer_dir"), **hub_kwargs)
    vision_tokenizer = build_vision_tokenizer(
        str(cfg_get(cfg, "vq_type", "ibq")),
        path_get(cfg, "vq_path", "vision_tokenizer_path", "vision_tokenizer_dir"),
        device=str(cfg_get(cfg, "vq_device", cfg_get(cfg, "vision_device", "cuda:0"))),
        **hub_kwargs,
    )
    cfg_obj.special_token_ids = special_token_ids(tokenizer)
    model = build_vllm_model(cfg, tokenizer)

    from xr_u0_ar.vllm_generation import generate as vllm_generate

    out_dir = Path(str(cfg_get(cfg, "save_path", "outputs/ar_vllm")))
    cases = normalize_cases(cfg.prompts)
    if len(cases) > 1:
        run_vllm_batch(
            cfg,
            cfg_obj,
            tokenizer,
            vision_tokenizer,
            model,
            out_dir,
        )
        return
    if not cases:
        raise ValueError("config.prompts must contain at least one case")

    run_metadata = _ar_vllm_run_metadata(cfg)
    timing_path = out_dir / "timing.json"
    case_id, case = cases[0]
    da3_artifact_dir = out_dir / "da3_preprocess" / case_id
    render_started = time.perf_counter()
    prompt, uncond = render_case(
        cfg_obj,
        case,
        tokenizer,
        vision_tokenizer,
        da3_artifact_dir=da3_artifact_dir,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    uncond_ids = tokenizer.encode(uncond, return_tensors="pt", add_special_tokens=False)
    input_ids = ensure_bos(input_ids, cfg_obj.special_token_ids["BOS"])
    uncond_ids = ensure_bos(uncond_ids, cfg_obj.special_token_ids["BOS"])
    render_seconds = time.perf_counter() - render_started

    timing: dict[str, Any] = {
        "backend": "ar_vllm",
        "task_type": cfg_obj.task_type,
        "n_inputs": 1,
        **run_metadata,
        "classifier_free_guidance": cfg_obj.classifier_free_guidance,
        "target_height": cfg_obj.target_height,
        "target_width": cfg_obj.target_width,
        "image_area": cfg_obj.image_area,
        "prompt_token_lengths": _length_summary([int(input_ids.shape[-1])]),
        "uncond_token_lengths": _length_summary([int(uncond_ids.shape[-1])]),
        "render_seconds": render_seconds,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    run_started = time.perf_counter()
    try:
        generation_started = time.perf_counter()
        saved = None
        for result_tokens in vllm_generate(cfg_obj, model, tokenizer, input_ids, uncond_ids):
            saved = _save_generation_output(
                cfg,
                cfg_obj,
                result_tokens,
                tokenizer,
                vision_tokenizer,
                out_dir / f"{case_id}.png",
                case=case,
                metadata={
                    "id": case_id,
                    "backend": "ar_vllm",
                    **run_metadata,
                    **_reference_preprocess_metadata(cfg_obj, case, artifact_dir=da3_artifact_dir),
                },
                require_image=_is_video_task(cfg_obj),
            )
            print(f"saved {saved.primary_path}")
            break
        generation_seconds = time.perf_counter() - generation_started
        if saved is None:
            raise RuntimeError("vLLM generation yielded no output")
    except Exception as exc:
        timing.update({
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "failed",
            "error": repr(exc),
            "total_seconds": time.perf_counter() - run_started,
            "n_saved": 0,
            "n_failed": 1,
        })
        _write_timing(timing_path, timing)
        raise

    total_seconds = time.perf_counter() - run_started
    timing.update({
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "ok",
        "generation_seconds": generation_seconds,
        "total_seconds": total_seconds,
        "n_saved": 1,
        "n_failed": 0,
        "generation_img_per_second": 1 / generation_seconds if generation_seconds else 0.0,
        "total_img_per_second": 1 / total_seconds if total_seconds else 0.0,
        "avg_generation_seconds_per_image": generation_seconds,
        "avg_total_seconds_per_image": total_seconds,
        "sample_outputs": [str(saved.primary_path)],
        "errors": [],
    })
    _write_timing(timing_path, timing)
    print(f"saved {timing_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Xiaomi-Robotics-U0 AR inference from a Python config.")
    parser.add_argument("--cfg", required=True, help="Path to a Python config file.")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    backend = str(cfg_get(cfg, "backend", "eager")).lower()
    os.makedirs(str(cfg_get(cfg, "save_path", "outputs")), exist_ok=True)
    if backend == "eager":
        run_eager(cfg)
    elif backend == "vllm":
        run_vllm(cfg)
    else:
        raise ValueError(f"unsupported backend for AR: {backend}")


if __name__ == "__main__":
    main()

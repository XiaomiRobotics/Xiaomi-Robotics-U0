from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.compose import compose_config, jsonable, set_by_path


def _parse_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--override must use key=value form: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--override has an empty key: {item}")
        set_by_path(overrides, key, _parse_value(raw_value))
    return overrides


def _set_if_present(overrides: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        set_by_path(overrides, key, value)


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides = _parse_overrides(args.override)
    for key in ("model_path", "tokenizer_path", "vq_path", "save_path"):
        _set_if_present(overrides, key, getattr(args, key))
    for key in ("hf_revision", "hf_cache_dir", "hf_local_files_only"):
        _set_if_present(overrides, key, getattr(args, key))
    for key in ("input_image_type", "da3_model_path", "da3_device"):
        _set_if_present(overrides, key, getattr(args, key))
    for key in ("da3_min_depth", "da3_max_depth", "da3_process_res"):
        _set_if_present(overrides, key, getattr(args, key))
    for key in (
        "tensor_parallel_size",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
    ):
        _set_if_present(overrides, key, getattr(args, key))

    if args.height is not None:
        set_by_path(overrides, "target_height" if args.engine == "ar" else "height", args.height)
    if args.width is not None:
        set_by_path(overrides, "target_width" if args.engine == "ar" else "width", args.width)
    _set_if_present(overrides, "target_height", args.target_height)
    _set_if_present(overrides, "target_width", args.target_width)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Xiaomi-Robotics-U0 inference from composable config layers.")
    parser.add_argument("--engine", required=True, choices=["ar", "flashar"])
    parser.add_argument("--backend", required=True, choices=["eager", "vllm"])
    parser.add_argument(
        "--task",
        required=True,
        choices=["t2i", "x2i", "scene-gen", "scene_gen", "transfer", "video-gen", "video_gen"],
    )
    parser.add_argument("--profile", default="single-gpu", choices=["single-gpu", "single_gpu", "multi-gpu", "multi_gpu"])
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Limit default task examples, or repeat CLI-provided prompt/reference overrides.",
    )
    parser.add_argument("--prompt", help="Override the default example prompt or raw task text.")
    parser.add_argument("--reference-image", action="append", default=[], help="Reference image for X2I/Transfer/Video Gen.")
    parser.add_argument("--legacy-video-jsonl", help="Optional legacy Video Gen JSONL input file.")
    parser.add_argument(
        "--input-image-type",
        choices=["depth", "rgb"],
        help="Transfer reference type. Use rgb to run DA3 RGB-to-depth preprocessing.",
    )
    parser.add_argument("--da3-model-path", help="Optional Depth Anything 3 model path or HuggingFace repo for Transfer RGB input.")
    parser.add_argument("--da3-device", help="Device used for DA3 preprocessing, e.g. cuda:0 or cpu.")
    parser.add_argument("--da3-min-depth", type=float, help="DA3 raw depth clip lower bound for RGB Transfer input.")
    parser.add_argument("--da3-max-depth", type=float, help="DA3 raw depth clip upper bound for RGB Transfer input.")
    parser.add_argument("--da3-process-res", type=int, help="DA3 processing resolution for RGB Transfer input.")
    parser.add_argument("--model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--vq-path")
    parser.add_argument("--hf-revision", help="Optional HuggingFace Hub revision for model/tokenizer/VisionTokenizer IDs.")
    parser.add_argument("--hf-cache-dir", help="Optional HuggingFace cache directory for automatic downloads.")
    parser.add_argument(
        "--hf-local-files-only",
        action="store_true",
        default=None,
        help="Resolve HuggingFace IDs from local cache only.",
    )
    parser.add_argument("--save-path")
    parser.add_argument("--height", type=int, help="Output token height. Maps to target_height for AR.")
    parser.add_argument("--width", type=int, help="Output token width. Maps to target_width for AR.")
    parser.add_argument("--target-height", type=int, help="Advanced AR-only target token height override.")
    parser.add_argument("--target-width", type=int, help="Advanced AR-only target token width override.")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Advanced override in key=value form. Dot paths such as sampling_params.image_top_k are supported.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config and exit.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be set together")
    if (args.target_height is None) != (args.target_width is None):
        parser.error("--target-height and --target-width must be set together")
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num-samples must be at least 1")

    try:
        cfg = compose_config(
            engine=args.engine,
            backend=args.backend,
            task=args.task,
            profile=args.profile,
            num_samples=args.num_samples,
            prompt=args.prompt,
            reference_images=args.reference_image,
            legacy_video_jsonl=args.legacy_video_jsonl,
            overrides=_cli_overrides(args),
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        from xr_u0_ar.depth_preprocess import validate_depth_preprocess_config

        validate_depth_preprocess_config(cfg)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(json.dumps(jsonable(cfg), indent=2, ensure_ascii=False))
        return

    if cfg.backend == "vllm":
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    if cfg.engine == "ar":
        from scripts import inference_ar

        if cfg.backend == "eager":
            inference_ar.run_eager(cfg)
        else:
            inference_ar.run_vllm(cfg)
        return

    if cfg.engine == "flashar":
        from scripts import inference_flashar

        if cfg.backend == "eager":
            inference_flashar.run_eager(cfg)
        else:
            inference_flashar.run_vllm(cfg)
        return

    raise ValueError(f"unsupported engine: {cfg.engine}")


if __name__ == "__main__":
    main()

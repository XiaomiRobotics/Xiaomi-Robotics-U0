from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

from xr_u0_ar.configuration_unis import UNISConfig
from xr_u0_ar.hub_paths import (
    resolve_local_file,
    resolve_model_path,
)
from xr_u0_ar.modeling_unis import UNISForCausalLM
from xr_u0_ar.task_prompts import T2I_PROMPT_TEMPLATE, T2I_UNCOND_PROMPT
from xr_u0_ar.utils import build_text_tokenizer, register_transformers
from xr_u0_ar.vision_tokenizer import build_vision_tokenizer
from xr_u0_flashar.model import UNISFlashAR
from xr_u0_flashar.outputs import decode_grid


def load_state(path: str) -> dict:
    state_path = Path(path)
    if state_path.name.endswith(".index.json"):
        index = json.loads(state_path.read_text(encoding="utf-8"))
        state = {}
        for shard_name in sorted(set(index.get("weight_map", {}).values())):
            state.update(load_safetensors(state_path.parent / shard_name, device="cpu"))
        return state
    if path.endswith(".safetensors"):
        return load_safetensors(path, device="cpu")
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


def discover_flashar_state(model_dir: str, explicit: str | None = None) -> str:
    if explicit:
        return resolve_local_file(explicit, kind="FlashAR state", base_dir=model_dir)
    root = Path(model_dir)
    for name in (
        "flashar.safetensors",
        "flashar.pt",
        "pytorch_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
    ):
        candidate = root / name
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(
        "FlashAR state not found. Pass --flashar-state or use a model directory "
        "containing model.safetensors.index.json, model.safetensors, flashar.safetensors, or flashar.pt."
    )


def build_empty_backbone(config, torch_dtype):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        return UNISForCausalLM(config)
    finally:
        torch.set_default_dtype(default_dtype)


def render_t2i_prompts(text: str, *, height: int, width: int) -> tuple[str, str]:
    image_anchor = f"<|image start|>{height}*{width}<|image token|>"
    return (
        T2I_PROMPT_TEMPLATE.format(text=text) + image_anchor,
        T2I_UNCOND_PROMPT + image_anchor,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Xiaomi-Robotics-U0-FlashAR eager text-to-image inference.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--flashar-state")
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--vision-tokenizer-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="outputs/xr_u0_flashar_eager.png")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5120)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--vertical-layers", type=int, default=4)
    parser.add_argument("--vertical-start-layer", type=int, default=-1)
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    device = torch.device(args.device)
    torch_dtype = dtype_map[args.dtype]

    register_transformers()
    model_dir = resolve_model_path(args.model_dir)
    flashar_state = discover_flashar_state(model_dir, args.flashar_state)
    config = UNISConfig.from_pretrained(model_dir, trust_remote_code=True)
    if is_integrated_flashar_state(flashar_state):
        backbone = build_empty_backbone(config, torch_dtype)
    else:
        backbone = UNISForCausalLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
    backbone = backbone.to(device)
    visual_offset = int(config.eoi_token_id) + 1
    vertical_start_layer = (
        int(args.vertical_start_layer)
        if args.vertical_start_layer >= 0
        else int(config.num_hidden_layers) - int(args.vertical_layers)
    )
    model = UNISFlashAR(
        pretrained_backbone=backbone.model,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        pad_token_id=-100,
        mask_token_id=config.pad_token_id,
        visual_token_offset=visual_offset,
        use_vertical_block=True,
        vertical_layers=args.vertical_layers,
        vertical_start_layer=vertical_start_layer,
    ).to(device=device, dtype=torch_dtype)
    model.load_state_dict(load_state(flashar_state), strict=True)
    model.eval()

    tokenizer = build_text_tokenizer(args.tokenizer_dir)
    prompt, uncond = render_t2i_prompts(
        args.prompt,
        height=args.height,
        width=args.width,
    )
    text_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    uncond_ids = tokenizer.encode(uncond, return_tensors="pt", add_special_tokens=False).to(device)
    grid = model.generate(
        height=args.height,
        width=args.width,
        device=device,
        text_input_ids=text_ids,
        unconditional_text_input_ids=uncond_ids,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    vision_tokenizer = build_vision_tokenizer("ibq", args.vision_tokenizer_dir, device=device)
    decode_grid(grid, vision_tokenizer, visual_offset, args.out, clamp_min=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

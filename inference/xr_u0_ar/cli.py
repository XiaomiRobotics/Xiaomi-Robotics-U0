from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from .generation import generate
from .image_tokens import build_image
from .outputs import save_generated_sequence
from .utils import build_text_tokenizer, build_unis_ar


def _special_token_ids(tokenizer) -> dict[str, int]:
    tokens = {
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
    return {name: tokenizer.encode(value)[0] for name, value in tokens.items()}


def _runtime_config(args: argparse.Namespace):
    return SimpleNamespace(
        streaming=False,
        task_type="x2i" if args.reference_image else "t2i",
        template=(
            "<|extra_203|>You are a helpful assistant for {task} task. "
            "USER: {image}{question} ASSISTANT: <|extra_100|>"
        ),
        unc_prompt=(
            "<|extra_203|>You are a helpful assistant. USER: {image} "
            "ASSISTANT: <|extra_100|>"
        ),
        classifier_free_guidance=args.cfg_scale,
        unconditional_type="no_text",
        image_cfg_scale=1.0,
        image_area=args.image_area,
        target_height=args.height,
        target_width=args.width,
        sampling_params={
            "use_cache": True,
            "text_top_k": 1024,
            "text_top_p": 0.9,
            "text_temperature": 1.0,
            "image_top_k": args.top_k,
            "image_top_p": args.top_p,
            "image_temperature": args.temperature,
            "top_k": 131072,
            "top_p": 1.0,
            "temperature": 1.0,
            "num_beams_per_group": 1,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "guidance_scale": 1.0,
            "use_differential_sampling": True,
            "do_sample": True,
            "num_beams": 1,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Xiaomi-Robotics-U0-AR eager inference.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--vision-tokenizer-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference-image", action="append", default=[])
    parser.add_argument("--out", default="outputs/xr_u0_ar.png")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--image-area", type=int, default=1024 * 1024)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5120)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=5120)
    parser.add_argument("--model-device", default="auto")
    parser.add_argument("--vision-device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(42)
    cfg = _runtime_config(args)
    model, tokenizer, vision_tokenizer = build_unis_ar(
        args.model_dir,
        args.tokenizer_dir,
        args.vision_tokenizer_dir,
        model_device=args.model_device,
        vision_device=args.vision_device,
    )
    cfg.special_token_ids = _special_token_ids(tokenizer)

    image_prefix = ""
    if args.reference_image:
        from PIL import Image

        image_prefix = "".join(
            build_image(Image.open(path).convert("RGB"), cfg, tokenizer, vision_tokenizer)
            for path in args.reference_image
        )

    task = "x2i" if args.reference_image else "t2i"
    prompt = cfg.template.format(task=task, image=image_prefix, question=args.prompt)
    uncond = cfg.unc_prompt.format(image=image_prefix)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    uncond_ids = tokenizer.encode(uncond, return_tensors="pt", add_special_tokens=False).to(model.device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device)
        input_ids = torch.cat([bos, input_ids], dim=1)

    output_dir = Path(args.out).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for result_tokens in generate(cfg, model, tokenizer, input_ids, uncond_ids):
        saved = save_generated_sequence(
            result_tokens,
            tokenizer,
            vision_tokenizer,
            args.out,
            metadata={"backend": "ar_eager_cli", "task_type": task},
        )
        print(f"saved {saved.primary_path}")
        return


if __name__ == "__main__":
    main()

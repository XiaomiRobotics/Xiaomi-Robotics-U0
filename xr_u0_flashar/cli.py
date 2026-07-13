from __future__ import annotations

import argparse

from xr_u0_ar.vision_tokenizer import build_vision_tokenizer
from xr_u0_flashar.outputs import decode_grid, write_flashar_audit
from xr_u0_flashar.vllm import LLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Xiaomi-Robotics-U0-FlashAR vLLM text-to-image inference.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--vision-tokenizer-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="outputs/xr_u0_flashar.png")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5120)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    args = parser.parse_args()

    llm = LLM.from_pretrained(
        args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        strict_visual_tokens=True,
    )
    result = llm.generate(
        args.prompt,
        height=args.height,
        width=args.width,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )[0]
    vision_tokenizer = build_vision_tokenizer("ibq", args.vision_tokenizer_dir, device=args.vision_device)
    visual_offset = llm.tokenizer.encode("<|image end|>")[0] + 1
    decode_grid(result.grid, vision_tokenizer, visual_offset, args.out)
    write_flashar_audit(args.out, result)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

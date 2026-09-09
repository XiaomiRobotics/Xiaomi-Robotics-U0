"""FlashAR vLLM high-throughput inference path.

Public API:
    from xr_u0_flashar.vllm import LLM
    llm = LLM.from_pretrained(
        "checkpoints/Xiaomi-Robotics-U0-FlashAR",
        tokenizer_dir="checkpoints/Xiaomi-Robotics-U0-FlashAR",
        max_num_seqs=16,
    )
    images = llm.generate(["a red apple"], height=32, width=32, cfg_scale=3.0)

This namespace is intentionally separate from the original FlashAR HuggingFace
inference code, which lives at flashar.inference.* and operates on plain
HuggingFace transformers — that path is unchanged.

The vLLM path requires:
  1. vLLM 0.11.0 installed: ``pip install vllm==0.11.0``
  2. The vLLM patches under third_party/vllm/ applied:
     ``python -m xr_u0_flashar.apply_vllm_patches``
"""

from xr_u0_flashar.vllm.api import GenerateResult, LLM, LLMConfig

__all__ = ["GenerateResult", "LLM", "LLMConfig"]

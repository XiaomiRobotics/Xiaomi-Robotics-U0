from __future__ import annotations

import os.path as osp

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .configuration_unis import UNISConfig
from .hub_paths import (
    resolve_model_path,
    resolve_tokenizer_path,
    resolve_vision_tokenizer_path,
)
from .modeling_unis import UNISForCausalLM
from .vision_tokenizer import build_vision_tokenizer


def register_transformers() -> None:
    AutoConfig.register(UNISConfig.model_type, UNISConfig, exist_ok=True)
    AutoModelForCausalLM.register(UNISConfig, UNISForCausalLM, exist_ok=True)


def build_text_tokenizer(
    tokenizer_dir: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool | None = None,
):
    tokenizer_dir = resolve_tokenizer_path(
        tokenizer_dir,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        special_tokens_file=osp.join(tokenizer_dir, "unis_vision_tokens.txt"),
        trust_remote_code=True,
    )
    tokenizer.bos_token = "<|extra_203|>"
    tokenizer.eos_token = "<|extra_204|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.eof_token = "<|extra_201|>"
    tokenizer.tms_token = "<|extra_202|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"
    tokenizer.bss_token = "<|extra_100|>"
    tokenizer.ess_token = "<|extra_101|>"
    tokenizer.bog_token = "<|extra_60|>"
    tokenizer.eog_token = "<|extra_61|>"
    tokenizer.boc_token = "<|extra_50|>"
    tokenizer.eoc_token = "<|extra_51|>"
    return tokenizer


def build_unis_ar(
    model_dir: str,
    tokenizer_dir: str,
    vision_tokenizer_dir: str,
    *,
    model_device: str = "auto",
    device_map=None,
    max_memory=None,
    vision_device: str = "cuda:0",
    vision_type: str = "ibq",
    torch_dtype="auto",
    attn_implementation: str = "eager",
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool | None = None,
    **vision_kwargs,
):
    hub_kwargs = {
        "revision": revision,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    model_dir = resolve_model_path(model_dir, **hub_kwargs)
    tokenizer_dir = resolve_tokenizer_path(tokenizer_dir, **hub_kwargs)
    vision_tokenizer_dir = resolve_vision_tokenizer_path(vision_tokenizer_dir, **hub_kwargs)
    register_transformers()
    config = UNISConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = UNISForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        device_map=model_device if device_map is None else device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    tokenizer = build_text_tokenizer(tokenizer_dir, **hub_kwargs)
    vision_tokenizer = build_vision_tokenizer(
        vision_type,
        vision_tokenizer_dir,
        device=vision_device,
        **hub_kwargs,
        **vision_kwargs,
    )
    return model, tokenizer, vision_tokenizer

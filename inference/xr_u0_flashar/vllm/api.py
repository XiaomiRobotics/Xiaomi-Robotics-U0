"""User-facing FlashAR vLLM API.

Provides ``LLM`` (modeled after ``vllm.LLM``) — a single class that loads a
FlashAR-vLLM checkpoint and generates images. All low-level details (worker
hooks, driver loop, CFG split, FlashAR head fusion) are hidden inside.

Example:
    from xr_u0_flashar.vllm import LLM
    llm = LLM.from_pretrained(
        "checkpoints/Xiaomi-Robotics-U0-FlashAR",
        tokenizer_dir="checkpoints/Xiaomi-Robotics-U0-FlashAR",
        max_num_seqs=16,
    )
    grids = llm.generate(
        ["a red apple on a wooden table"],
        height=32, width=32, cfg_scale=3.0,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, Union

import torch

from xr_u0_ar.task_prompts import T2I_PROMPT_TEMPLATE, T2I_UNCOND_PROMPT

_IMAGE_ANCHOR_TEMPLATE = "<|image start|>{H}*{W}<|image token|>"
_DEFAULT_UNCOND_TEMPLATE = T2I_UNCOND_PROMPT + _IMAGE_ANCHOR_TEMPLATE
_DEFAULT_PROMPT_TEMPLATE = T2I_PROMPT_TEMPLATE + _IMAGE_ANCHOR_TEMPLATE


@dataclass
class LLMConfig:
    """Inference-time configuration for FlashAR-vLLM.

    ``model_dir`` is a HuggingFace-style Xiaomi-Robotics-U0-FlashAR directory containing the
    backbone weights and FlashAR vertical/head weights. Its ``config.json``
    must report ``architectures=["UNISFlashARForCausalLM"]``.
    """
    model_dir: Optional[str] = None
    tokenizer_dir: str = ""
    backbone_dir: Optional[str] = None
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    max_num_seqs: int = 16
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 4096
    enable_log_stats: bool = False
    strict_visual_tokens: bool = True
    seed: int = 42
    hf_revision: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    local_files_only: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.model_dir is None:
            if self.backbone_dir is None:
                raise TypeError("model_dir is required")
            self.model_dir = str(self.backbone_dir)
        if self.backbone_dir is None:
            self.backbone_dir = str(self.model_dir)
        if not self.tokenizer_dir:
            raise TypeError("tokenizer_dir is required")


@dataclass
class GenerateResult:
    """Output of a single FlashAR generation."""
    grid: torch.Tensor          # (H, W) long tensor of vision-token IDs
    visual_tokens: list         # flattened visual tokens in raster order
    n_steps: int                # vLLM forward steps consumed
    output_token_ids: list = field(default_factory=list)  # raw vLLM output IDs
    height: int = 0
    width: int = 0
    n_visual_expected: int = 0
    n_visual_actual: int = 0
    visual_token_offset: int = 0
    token_min: Optional[int] = None
    token_max: Optional[int] = None
    warn: Optional[str] = None  # any non-fatal warning
    parent_id: Optional[str] = None
    tp_rank_summaries: Optional[list] = None  # optional per-rank state summary


class LLM:
    """FlashAR vLLM high-throughput image generator.

    Construct with :py:meth:`from_pretrained`, then call :py:meth:`generate`
    on a list of text prompts. The class hides vLLM request-pair scheduling,
    diagonal decode, anchor capture, and packed visual-logit sampling.
    """

    def __init__(self, config: LLMConfig):
        from xr_u0_ar.hub_paths import resolve_model_path, resolve_tokenizer_path

        hub_kwargs = {
            "revision": config.hf_revision,
            "cache_dir": config.hf_cache_dir,
            "local_files_only": config.local_files_only,
        }
        config.model_dir = resolve_model_path(config.model_dir or "", **hub_kwargs)
        config.backbone_dir = resolve_model_path(config.backbone_dir or config.model_dir, **hub_kwargs)
        config.tokenizer_dir = resolve_tokenizer_path(config.tokenizer_dir, **hub_kwargs)
        self._config = config
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        # Lazy import because the runtime pulls in vLLM and CUDA.
        from xr_u0_flashar.vllm.runtime import UNISFlashARVLLMConfig, UNISFlashARVLLMWrapper
        runtime_cfg = UNISFlashARVLLMConfig(
            backbone_dir=config.model_dir,
            tokenizer_dir=config.tokenizer_dir,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            seed=config.seed,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_chunked_prefill=config.enable_chunked_prefill,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_log_stats=config.enable_log_stats,
            strict_visual_tokens=config.strict_visual_tokens,
            max_num_seqs=config.max_num_seqs,
        )
        self._wrapper = UNISFlashARVLLMWrapper(runtime_cfg)
        self._tokenizer = self._wrapper.tokenizer

    # ---------------------------------------------------------- constructors
    @classmethod
    def from_pretrained(cls, model_dir: Optional[Union[str, Path]] = None,
                        tokenizer_dir: Optional[Union[str, Path]] = None,
                        **kwargs) -> "LLM":
        """Load an FlashAR-vLLM checkpoint.

        ``model_dir`` is the public Xiaomi-Robotics-U0-FlashAR model directory. For
        compatibility, callers may still pass ``backbone_dir=...`` in
        ``kwargs`` instead of the first positional argument.

        Any extra keyword arguments (``max_num_seqs``,
        ``gpu_memory_utilization``, etc.) are forwarded to :class:`LLMConfig`.
        """
        if model_dir is None and "backbone_dir" in kwargs:
            model_dir = kwargs.pop("backbone_dir")
        if model_dir is None:
            raise TypeError("model_dir is required")
        if tokenizer_dir is None:
            raise TypeError("tokenizer_dir is required")
        if "revision" in kwargs and "hf_revision" not in kwargs:
            kwargs["hf_revision"] = kwargs.pop("revision")
        if "cache_dir" in kwargs and "hf_cache_dir" not in kwargs:
            kwargs["hf_cache_dir"] = kwargs.pop("cache_dir")
        config = LLMConfig(
            model_dir=str(model_dir),
            tokenizer_dir=str(tokenizer_dir),
            **kwargs,
        )
        return cls(config)

    # -------------------------------------------------------------- generate
    def generate(
        self,
        prompts: Union[str, list[str]],
        *,
        height: int = 32,
        width: int = 32,
        cfg_scale: float = 3.0,
        temperature: float = 1.0,
        top_k: int = 5120,
        top_p: float = 1.0,
        uncond_prompt: Optional[Union[str, list[str]]] = None,
        seeds: Optional[list[int]] = None,
        prompt_template: Optional[str] = None,
        uncond_template: Optional[str] = None,
    ) -> list[GenerateResult]:
        """Generate visual-token grids for one or more text prompts.

        Args:
            prompts: text prompt(s). A bare string is wrapped into a 1-elem
                list. Each prompt is rendered through ``prompt_template``
                (defaults to FlashAR's training-time T2I template) before
                being tokenized.
            height, width: output resolution in vision-token units. FlashAR
                does not infer the image header itself; callers should pass the
                task-specific grid shape used in the rendered prompt.
            cfg_scale: classifier-free guidance scale; 1.0 disables CFG.
            temperature, top_k, top_p: standard sampling params.
            uncond_prompt: text used for the CFG unconditional branch. By
                default the T2I task template with empty user text is used.
                Pass a string to override for all prompts; pass a list to
                override per-prompt.
            seeds: per-prompt RNG seeds. Default = config.seed + counter.
            prompt_template / uncond_template: format strings; ``{text}``
                gets the user prompt, ``{H}``/``{W}`` get the resolution.

        Returns:
            list of :class:`GenerateResult`, one per input prompt.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        n = len(prompts)

        # Render prompts through the templates
        ptpl = prompt_template or _DEFAULT_PROMPT_TEMPLATE
        utpl = uncond_template or _DEFAULT_UNCOND_TEMPLATE
        rendered_prompts = [
            ptpl.format(text=t, H=height, W=width) for t in prompts
        ]
        if uncond_prompt is None:
            uncs = [utpl.format(H=height, W=width) for _ in prompts]
        elif isinstance(uncond_prompt, str):
            uncs = [uncond_prompt] * n
        else:
            assert len(uncond_prompt) == n, "uncond_prompt list size mismatch"
            uncs = list(uncond_prompt)

        runtime_results = []
        max_parent_batch = max(1, int(self._config.max_num_seqs) // 2)
        for start in range(0, n, max_parent_batch):
            end = min(start + max_parent_batch, n)
            runtime_results.extend(
                self._wrapper.generate_diagonal_batch(
                    prompts=rendered_prompts[start:end],
                    uncond_prompts=uncs[start:end],
                    heights=[height] * (end - start),
                    widths=[width] * (end - start),
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seeds=None if seeds is None else seeds[start:end],
                )
            )
        return [
            GenerateResult(
                grid=r["grid"],
                visual_tokens=r["visual_tokens"],
                output_token_ids=r.get("output_token_ids", []),
                n_steps=r["n_steps"],
                height=int(r.get("height", r["grid"].shape[0])),
                width=int(r.get("width", r["grid"].shape[1])),
                n_visual_expected=int(r.get("n_visual_expected", r["grid"].numel())),
                n_visual_actual=int(r.get("n_visual_actual", len(r["visual_tokens"]))),
                visual_token_offset=int(r.get("visual_token_offset", 0)),
                token_min=r.get("token_min"),
                token_max=r.get("token_max"),
                warn=r.get("warn"),
                parent_id=r.get("parent_id"),
                tp_rank_summaries=r.get("tp_rank_summaries"),
            )
            for r in runtime_results
        ]

    # --------------------------------------------------- low-level passthrough
    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self) -> LLMConfig:
        return self._config

    @property
    def vllm_engine(self):
        """Underlying ``vllm.LLM`` instance (escape hatch for advanced use)."""
        return self._wrapper.llm

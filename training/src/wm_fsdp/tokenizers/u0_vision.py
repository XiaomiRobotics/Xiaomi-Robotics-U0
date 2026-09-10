from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
from torch import nn


def _resolve_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    name = str(value or "float32").removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported vision tokenizer dtype: {value!r}")
    return dtype


class U0IBQTokenizer(nn.Module):
    """Frozen U0 IBQ encoder returning raw ``[batch, height, width]`` ids."""

    def __init__(self, checkpoint: str, device: str = "auto", dtype: Any = "float32"):
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("vision tokenizer device is CUDA, but CUDA is unavailable")
        self._dtype = _resolve_dtype(dtype)
        # U0 inference uses the native IBQ implementation. HF AutoModel vision
        # checkpoints are deliberately rejected:
        # their preprocessing/codebook can produce different image ids.
        checkpoint_path = Path(checkpoint).expanduser()
        config_path = checkpoint_path / "config.yaml"
        ckpt_path = checkpoint_path / "model.ckpt"
        if not config_path.is_file() or not ckpt_path.is_file():
            raise FileNotFoundError(
                "native U0 IBQ tokenizer requires config.yaml and model.ckpt: "
                f"{checkpoint_path}"
            )
        from .native_vision_tokenizer import build_vision_tokenizer

        self._native_ibq = build_vision_tokenizer("ibq", str(checkpoint_path), device=str(self._device))
        self._native_ibq.eval().requires_grad_(False)
        self.visual_vocab_size = int(getattr(self._native_ibq.quantize, "n_e", 131072))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.LongTensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images [B,3,H,W], got {tuple(images.shape)}")
        images = images.to(device=self.device, dtype=self.dtype, non_blocking=self.device.type == "cuda")
        quantized, _loss, info = self._native_ibq.encode(images)
        token_ids = info[-1]
        if quantized is None:
            raise ValueError("U0 IBQ encode output does not contain quantized features")
        height, width = (int(value) for value in quantized.shape[-2:])
        token_ids = token_ids.reshape(images.shape[0], height, width)
        token_ids = token_ids.long().cpu().contiguous()
        if token_ids.numel() and (int(token_ids.min()) < 0 or int(token_ids.max()) >= self.visual_vocab_size):
            raise ValueError("U0 IBQ encoder returned ids outside its codebook")
        return token_ids

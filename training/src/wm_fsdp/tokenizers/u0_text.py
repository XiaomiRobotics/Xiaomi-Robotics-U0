from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from typing import Any


def load_u0_tokenizer(tokenizer_path: str, special_tokens_file: str | None = None) -> Any:
    path = Path(os.path.expanduser(os.path.expandvars(tokenizer_path))).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"U0 tokenizer directory does not exist: {path}")
    special_path = (
        Path(special_tokens_file).expanduser().resolve()
        if special_tokens_file
        else path / "unis_vision_tokens.txt"
    )
    if not special_path.is_file():
        raise FileNotFoundError(f"U0 vision token list does not exist: {special_path}")

    # Prefer the repository-local implementation.  This avoids importing a
    # package-specific remote-code wrapper (for example ``xr_u0_ar``) when a
    # tokenizer directory only contains a tiktoken vocabulary and token list.
    vocab_path = path / "unis.tiktoken"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"U0 tokenizer vocabulary does not exist: {vocab_path}")
    from .tokenization_unis import UNISTokenizer

    tokenizer = UNISTokenizer(
        vocab_file=str(vocab_path),
        special_tokens_file=str(special_path),
    )
    aliases = {
        "bos_token": "<|extra_203|>",
        "eos_token": "<|extra_204|>",
        "pad_token": "<|endoftext|>",
        "eol_token": "<|extra_200|>",
        "eof_token": "<|extra_201|>",
        "tms_token": "<|extra_202|>",
        "img_token": "<|image token|>",
        "boi_token": "<|image start|>",
        "eoi_token": "<|image end|>",
        "bss_token": "<|extra_100|>",
        "ess_token": "<|extra_101|>",
        "bog_token": "<|extra_60|>",
        "eog_token": "<|extra_61|>",
        "boc_token": "<|extra_50|>",
        "eoc_token": "<|extra_51|>",
    }
    for attribute, token in aliases.items():
        setattr(tokenizer, attribute, token)
    return tokenizer


def encode_u0_text(tokenizer: Any, text: str) -> list[int]:
    text = unicodedata.normalize("NFC", str(text))
    raw_tokenizer = getattr(tokenizer, "tokenizer", None)
    if raw_tokenizer is not None and hasattr(raw_tokenizer, "encode"):
        try:
            return [
                int(token_id)
                for token_id in raw_tokenizer.encode(
                    text,
                    allowed_special="all",
                    disallowed_special=(),
                )
            ]
        except TypeError:
            pass
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def u0_token_id(tokenizer: Any, token: str) -> int:
    ids = encode_u0_text(tokenizer, token)
    if len(ids) != 1:
        raise ValueError(f"Expected one token for {token!r}, got {ids}")
    return ids[0]


def u0_image_prefix(tokenizer: Any, height: int, width: int) -> list[int]:
    return [
        u0_token_id(tokenizer, tokenizer.boi_token),
        *encode_u0_text(tokenizer, f"{height}*{width}"),
        u0_token_id(tokenizer, tokenizer.img_token),
    ]

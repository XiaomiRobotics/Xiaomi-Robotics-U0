"""Native Xiaomi-Robotics-U0 IBQ vision tokenizer used by the release tools.

The checkpoint is intentionally supplied by the caller.  Only the tokenizer
implementation is part of this repository.
"""

import os
from os import path as osp

import torch
from omegaconf import OmegaConf
from .ibq import IBQ


def build_vision_tokenizer(
    tokenizer_type: str,
    model_path: str,
    device: str = "cuda:0",
    config_file: str = "config.yaml",
    ckpt_file: str = "model.ckpt",
    **kwargs,
):
    """Build the U0 IBQ tokenizer from a local native checkpoint."""
    if tokenizer_type != "ibq":
        raise NotImplementedError(f"Unsupported vision tokenizer type: {tokenizer_type}")
    model_path = os.path.expanduser(os.path.expandvars(str(model_path)))
    cfg_path = osp.join(model_path, config_file)
    ckpt_path = osp.join(model_path, ckpt_file)
    if not osp.exists(model_path):
        raise FileNotFoundError(f"native IBQ tokenizer directory does not exist: {model_path}")
    if not (osp.exists(cfg_path) and osp.exists(ckpt_path)):
        raise FileNotFoundError(
            f"native IBQ tokenizer requires {config_file} and {ckpt_file} under {model_path}"
        )
    cfg = OmegaConf.load(cfg_path)
    tokenizer = IBQ(**cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    tokenizer.load_state_dict(checkpoint)
    return tokenizer.eval().to(device)

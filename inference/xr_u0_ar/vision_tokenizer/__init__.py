import os.path as osp

from omegaconf import OmegaConf
import torch
from transformers import AutoModel

from xr_u0_ar.hub_paths import resolve_vision_tokenizer_path

from .ibq import IBQ


def build_vision_tokenizer(
    type,
    model_path,
    device="cuda:0",
    config_file="config.yaml",
    ckpt_file="model.ckpt",
    revision=None,
    cache_dir=None,
    local_files_only=None,
    **kwargs,
):
    match type:
        case "ibq":
            model_path = resolve_vision_tokenizer_path(
                model_path,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            if osp.exists(model_path):
                cfg_path = osp.join(model_path, config_file)
                ckpt_path = osp.join(model_path, ckpt_file)
                if osp.exists(cfg_path) or osp.exists(ckpt_path):
                    if not (osp.exists(cfg_path) and osp.exists(ckpt_path)):
                        raise FileNotFoundError(
                            f"config/ckpt not found under {model_path}: expected {config_file}, {ckpt_file}"
                        )
                    cfg = OmegaConf.load(cfg_path)
                    tokenizer = IBQ(**cfg)
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        ckpt = ckpt["state_dict"]
                    tokenizer.load_state_dict(ckpt)
                    tokenizer.eval().to(device)
                    return tokenizer

            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            model = model.to(device)
            model.eval()
            return model
        case _:
            raise NotImplementedError(f"Unsupported vision tokenizer type: {type}")

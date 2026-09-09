from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _argument_value(name: str) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} requires a value")
    return sys.argv[index + 1]


def _configure_hf_modules_cache() -> None:
    config_path = _argument_value("--config")
    if config_path is None:
        raise ValueError("--config is required")
    results_dir = _argument_value("--results-dir")
    if results_dir is None:
        config = OmegaConf.load(config_path)
        results_dir = str(config.training.results_dir)

    rank = int(os.environ.get("RANK", "0"))
    cache_root = os.environ.get("WM_FSDP_HF_MODULES_CACHE_ROOT")
    if cache_root is None:
        cache_root = str(Path(results_dir) / "hf_modules")
    os.environ["HF_MODULES_CACHE"] = str(
        Path(cache_root).expanduser() / f"rank_{rank:05d}"
    )


def main() -> None:
    # transformers reads HF_MODULES_CACHE into a module constant at import
    # time. This must run before importing the training entry and model code.
    _configure_hf_modules_cache()
    from wm_fsdp.train.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()

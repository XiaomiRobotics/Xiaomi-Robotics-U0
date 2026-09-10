from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _expand_paths(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    return value


def load_data_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix.lower() != ".json":
        raise ValueError(f"Data config must be JSON: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Data config must contain a JSON object: {config_path}")
    return _expand_paths(config)


__all__ = ["load_data_config"]

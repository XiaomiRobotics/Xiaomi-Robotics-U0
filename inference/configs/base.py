from __future__ import annotations

from copy import deepcopy


COMMON_VISION_TOKENIZER = {
    "vq_path": "checkpoints/VisionTokenizer",
    "vq_type": "ibq",
    "vq_device": "cuda:0",
}

MODEL_PATHS = {
    "ar": {
        "default": {
            "model_path": "checkpoints/Xiaomi-Robotics-U0",
            "tokenizer_path": "checkpoints/Xiaomi-Robotics-U0",
        },
        "4b": {
            "model_path": "../ckpt/Xiaomi-Robotics-U0-4B",
            "tokenizer_path": "../ckpt/Xiaomi-Robotics-U0-4B",
        },
    },
    "flashar": {
        "default": {
            "model_path": "checkpoints/Xiaomi-Robotics-U0-FlashAR",
            "tokenizer_path": "checkpoints/Xiaomi-Robotics-U0-FlashAR",
        },
    },
}


def base_config(engine: str, task: str, model_size: str | None = None) -> dict:
    if model_size is None:
        model_size = "38b" if engine == "flashar" else "34b"
    model_size = model_size.strip().lower()
    if model_size not in {"4b", "34b", "38b"}:
        raise ValueError(f"unsupported model size: {model_size}")
    if engine == "flashar" and model_size != "38b":
        raise ValueError("FlashAR supports the 38B checkpoint only; the 4B checkpoint is AR-only")
    if engine == "ar" and model_size == "38b":
        raise ValueError("AR supports the 34B and 4B checkpoints; 38B is the FlashAR checkpoint")
    config = deepcopy(COMMON_VISION_TOKENIZER)
    engine_paths = MODEL_PATHS[engine]
    if engine == "ar" and task in {"interleave_subtask", "interleave_video"}:
        config.update({"model_path": "../ckpt/Xiaomi-Robotics-U0-4B-Sequence" if model_size == "4b" else "../ckpt/Xiaomi-Robotics-U0-Sequence", "tokenizer_path": "../ckpt/Xiaomi-Robotics-U0-4B-Sequence" if model_size == "4b" else "../ckpt/Xiaomi-Robotics-U0-Sequence"})
    elif engine == "ar" and model_size == "4b":
        config.update(deepcopy(engine_paths["4b"]))
    else:
        config.update(deepcopy(engine_paths.get(task, engine_paths["default"])))
    config["model_size"] = model_size
    return config

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
        "video_gen": {
            "model_path": "checkpoints/Xiaomi-Robotics-U0-Video",
            "tokenizer_path": "checkpoints/Xiaomi-Robotics-U0-Video",
        },
    },
    "flashar": {
        "default": {
            "model_path": "checkpoints/Xiaomi-Robotics-U0-FlashAR",
            "tokenizer_path": "checkpoints/Xiaomi-Robotics-U0-FlashAR",
        },
    },
}


def base_config(engine: str, task: str) -> dict:
    config = deepcopy(COMMON_VISION_TOKENIZER)
    engine_paths = MODEL_PATHS[engine]
    config.update(deepcopy(engine_paths.get(task, engine_paths["default"])))
    return config

from __future__ import annotations


def profile_config(profile: str, engine: str, backend: str, task: str) -> dict:
    if profile == "single_gpu":
        return {}
    if profile != "multi_gpu":
        raise ValueError(f"unsupported profile: {profile}")

    if backend == "eager":
        if engine == "ar":
            return {"device_map": "balanced", "model_device": "auto"}
        return {"device_map": "balanced", "flashar_device": "cuda:0"}

    if backend == "vllm":
        return {
            "tensor_parallel_size": 2,
            "max_num_seqs": 2 if task == "video_gen" else 4,
        }

    raise ValueError(f"unsupported backend: {backend}")

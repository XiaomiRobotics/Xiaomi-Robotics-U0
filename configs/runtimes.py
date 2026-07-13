from __future__ import annotations


def runtime_config(engine: str, backend: str, task: str) -> dict:
    if backend == "eager":
        if engine == "ar":
            return {"backend": "eager", "attn_implementation": "eager"}
        return {
            "backend": "eager",
            "vertical_layers": 4,
            "vertical_start_layer": -1,
        }

    if backend != "vllm":
        raise ValueError(f"unsupported backend: {backend}")

    if engine == "ar":
        if task == "video_gen":
            return {
                "backend": "vllm",
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.8,
                "max_num_seqs": 2,
                "max_num_batched_tokens": 26000,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "enable_log_stats": True,
                "seed": 0,
            }
        return {
            "backend": "vllm",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 16 if task in {"x2i", "transfer"} else 32,
            "max_num_batched_tokens": 65536,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
        }

    if engine == "flashar":
        return {
            "backend": "vllm",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "max_model_len": 16384,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 32768,
            "enable_prefix_caching": False,
            "strict_visual_tokens": True,
        }

    raise ValueError(f"unsupported engine: {engine}")

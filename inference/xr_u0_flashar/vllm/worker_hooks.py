"""Worker-side vLLM hooks used by the Xiaomi-Robotics-U0-FlashAR runtime.

These symbols are intentionally grouped here for readers who need to understand
how the vLLM patch is driven. The implementation lives in ``runtime`` to avoid
splitting the execution path across multiple driver implementations.
"""

from xr_u0_flashar.vllm.runtime import (
    worker_clear_diagonal_next_tokens,
    worker_get_diagonal_states,
    worker_get_tp_rank_state_summary,
    worker_get_vertical_intro_log,
    worker_install_diagonal_decode,
    worker_install_hooks,
    worker_install_non_causal_hook,
    worker_install_query_meta_capture,
    worker_install_sample_skip,
    worker_publish_initial_vertical_meta,
    worker_register_diagonal_request,
    worker_reset_vertical_intro_log,
    worker_unregister_diagonal_request,
)

__all__ = [
    "worker_install_hooks",
    "worker_install_query_meta_capture",
    "worker_install_sample_skip",
    "worker_install_non_causal_hook",
    "worker_install_diagonal_decode",
    "worker_register_diagonal_request",
    "worker_unregister_diagonal_request",
    "worker_publish_initial_vertical_meta",
    "worker_get_vertical_intro_log",
    "worker_reset_vertical_intro_log",
    "worker_get_diagonal_states",
    "worker_clear_diagonal_next_tokens",
    "worker_get_tp_rank_state_summary",
]

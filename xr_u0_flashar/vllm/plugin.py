"""vLLM plugin entry point for FlashAR custom model registration."""

from __future__ import annotations


def register_unis_flashar_model() -> None:
    """Register FlashAR's vLLM model class in every vLLM process.

    vLLM TP uses spawned worker processes. Dynamic registration in the parent
    process is not visible there, so workers must discover this function via
    the `vllm.general_plugins` entry point group and run it at startup.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "UNISFlashARForCausalLM",
        "xr_u0_flashar.vllm.model:UNISFlashARForCausalLM",
    )

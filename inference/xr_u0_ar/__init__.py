"""Xiaomi-Robotics-U0-AR inference package."""

__all__ = ["UNISConfig", "UNISForCausalLM", "UNISModel"]


def __getattr__(name: str):
    if name == "UNISConfig":
        from .configuration_unis import UNISConfig

        return UNISConfig
    if name in {"UNISForCausalLM", "UNISModel"}:
        from .modeling_unis import UNISForCausalLM, UNISModel

        return {"UNISForCausalLM": UNISForCausalLM, "UNISModel": UNISModel}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

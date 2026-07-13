"""Xiaomi-Robotics-U0-FlashAR inference package."""

__all__ = ["UNISFlashAR"]


def __getattr__(name: str):
    if name == "UNISFlashAR":
        from .model import UNISFlashAR

        return UNISFlashAR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""CFG request-pair helpers for Xiaomi-Robotics-U0-FlashAR vLLM.

The vLLM patch expands one public image-generation request into two child
requests: a conditional branch and an unconditional branch. The diagonal
runtime keeps both branches in lockstep once visual-token decoding starts, then
combines their logits with classifier-free guidance.
"""

from __future__ import annotations

from typing import Optional


COND_PREFIX = "0_cfg_"
UNCOND_PREFIX = "1_cfg_"


def parent_id_from_child(req_id: str) -> Optional[str]:
    """Return the public parent id for a CFG child request id."""
    if "_cfg_" not in req_id:
        return None
    head, _, parent = req_id.partition("_cfg_")
    if head not in ("0", "1"):
        return None
    return parent


def role_of_child(req_id: str) -> Optional[str]:
    """Return ``cond`` or ``uncond`` for a CFG child request id."""
    if req_id.startswith(COND_PREFIX):
        return "cond"
    if req_id.startswith(UNCOND_PREFIX):
        return "uncond"
    return None


def child_request_id(parent_id: str, role: str) -> str:
    """Build the vLLM child request id for a parent and branch role."""
    if role == "cond":
        return f"{COND_PREFIX}{parent_id}"
    if role == "uncond":
        return f"{UNCOND_PREFIX}{parent_id}"
    raise ValueError(f"unknown CFG role: {role!r}")

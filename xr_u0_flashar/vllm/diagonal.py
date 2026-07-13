"""Anti-diagonal visual-token order used by Xiaomi-Robotics-U0-FlashAR decoding."""

from __future__ import annotations

import torch


def diagonal_positions(diag_idx: int, height: int, width: int) -> list[int]:
    """Return raster-order flat indices for one anti-diagonal.

    A visual position ``(r, c)`` belongs to diagonal ``r + c``. Generated
    visual tokens are sampled diagonal by diagonal, then written back into the
    final grid by these flat indices.
    """
    positions: list[int] = []
    r_lo = max(0, diag_idx - width + 1)
    r_hi = min(height, diag_idx + 1)
    for row in range(r_lo, r_hi):
        col = diag_idx - row
        positions.append(row * width + col)
    return positions


_DIAG_LIST_CACHE: dict[tuple[int, int], list[list[int]]] = {}
_DIAG_PREFIX_CACHE: dict[tuple[int, int], list[int]] = {}
_DIAG_TENSOR_CACHE: dict[tuple[int, int, str], list[torch.Tensor]] = {}


def diagonal_list(diag_idx: int, height: int, width: int) -> list[int]:
    """Cached ``diagonal_positions`` for one grid shape."""
    cache = _DIAG_LIST_CACHE.get((height, width))
    if cache is None:
        cache = [
            diagonal_positions(d, height, width)
            for d in range(height + width - 1)
        ]
        _DIAG_LIST_CACHE[(height, width)] = cache
    return cache[diag_idx]


def diagonal_prefix_sum(height: int, width: int) -> list[int]:
    """Cumulative token count before each diagonal."""
    cache = _DIAG_PREFIX_CACHE.get((height, width))
    if cache is None:
        list_cache = _DIAG_LIST_CACHE.get((height, width))
        if list_cache is None:
            list_cache = [
                diagonal_positions(d, height, width)
                for d in range(height + width - 1)
            ]
            _DIAG_LIST_CACHE[(height, width)] = list_cache
        prefix = [0]
        for positions in list_cache:
            prefix.append(prefix[-1] + len(positions))
        _DIAG_PREFIX_CACHE[(height, width)] = prefix
        cache = prefix
    return cache


def diagonal_tensor(
    diag_idx: int,
    height: int,
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Cached device tensor for one anti-diagonal's flat indices."""
    dev_key = str(device)
    cache = _DIAG_TENSOR_CACHE.get((height, width, dev_key))
    if cache is None:
        list_cache = _DIAG_LIST_CACHE.get((height, width))
        if list_cache is None:
            list_cache = [
                diagonal_positions(d, height, width)
                for d in range(height + width - 1)
            ]
            _DIAG_LIST_CACHE[(height, width)] = list_cache
        cache = [
            torch.tensor(pos, dtype=torch.long, device=device)
            for pos in list_cache
        ]
        _DIAG_TENSOR_CACHE[(height, width, dev_key)] = cache
    return cache[diag_idx]

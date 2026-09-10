from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


class _DifferentiableAllToAllSingle(torch.autograd.Function):
    """Equal-split all-to-all with an explicit symmetric backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        tensor_contiguous = tensor.contiguous()
        output = torch.empty_like(
            tensor_contiguous,
            memory_format=torch.contiguous_format,
        )
        dist.all_to_all_single(output, tensor_contiguous, group=group)
        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        grad_output_contiguous = grad_output.contiguous()
        grad_input = torch.empty_like(
            grad_output_contiguous,
            memory_format=torch.contiguous_format,
        )
        dist.all_to_all_single(
            grad_input,
            grad_output_contiguous,
            group=ctx.group,
        )
        return grad_input, None


class _DifferentiableSequenceAllGather(torch.autograd.Function):
    """Gather sequence shards and reduce their replicated output gradients."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.local_length = int(tensor.size(1))
        parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        return torch.cat(parts, dim=1)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        # Every SP lane evaluates the same logical output. Summing the matching
        # shard is the reduce-scatter backward of all-gather. Move sequence to
        # dimension zero because reduce_scatter_tensor splits only that axis.
        sequence_first = grad_output.movedim(1, 0).contiguous()
        local_sequence_first = torch.empty(
            (ctx.local_length, *sequence_first.shape[1:]),
            dtype=sequence_first.dtype,
            device=sequence_first.device,
        )
        dist.reduce_scatter_tensor(
            local_sequence_first,
            sequence_first,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
        )
        return local_sequence_first.movedim(0, 1).contiguous(), None


@dataclass(frozen=True)
class UlyssesState:
    """Process groups and ranks for a two-dimensional DP x SP topology."""

    size: int = 1
    rank: int = 0
    group: Optional[dist.ProcessGroup] = None
    dp_group: Optional[dist.ProcessGroup] = None
    dp_rank: int = 0
    data_rank: int = 0
    data_world_size: int = 1
    fsdp_uses_dp_group: bool = False

    @property
    def enabled(self) -> bool:
        return self.size > 1 and self.group is not None and dist.is_initialized()

    @property
    def sp_rank(self) -> int:
        return self.rank

    @property
    def sp_group(self) -> Optional[dist.ProcessGroup]:
        return self.group

    @property
    def dp_size(self) -> int:
        return self.data_world_size


_STATE = UlyssesState()


def get_ulysses_state() -> UlyssesState:
    return _STATE


def reset_ulysses() -> None:
    """Reset module state without destroying the default process group."""

    global _STATE
    _STATE = UlyssesState()


def build_ulysses_rank_groups(
    world_size: int,
    ulysses_size: int,
    local_world_size: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return all SP groups followed by all DP groups in creation order.

    With ``local_world_size`` set, groups follow the node-local topology: SP
    groups are node-local and each DP group connects the same SP lane and
    local group index across nodes. Without it, retain the historical
    row-major mapping for standalone topology tests and callers.
    """

    world_size = int(world_size)
    ulysses_size = int(ulysses_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if ulysses_size <= 0:
        raise ValueError(f"ulysses_size must be positive, got {ulysses_size}")
    if world_size % ulysses_size != 0:
        raise ValueError(
            f"ulysses_size={ulysses_size} must divide world_size={world_size}"
        )

    if local_world_size is None:
        dp_size = world_size // ulysses_size
        sp_groups = [
            list(range(dp_rank * ulysses_size, (dp_rank + 1) * ulysses_size))
            for dp_rank in range(dp_size)
        ]
        dp_groups = [
            [dp_rank * ulysses_size + sp_rank for dp_rank in range(dp_size)]
            for sp_rank in range(ulysses_size)
        ]
        return sp_groups, dp_groups

    local_world_size = int(local_world_size)
    if local_world_size <= 0 or world_size % local_world_size:
        raise ValueError(
            f"local_world_size={local_world_size} must be positive and divide "
            f"world_size={world_size}"
        )
    if local_world_size % ulysses_size:
        raise ValueError(
            f"ulysses_size={ulysses_size} must divide local_world_size={local_world_size}"
        )
    nodes = world_size // local_world_size
    groups_per_node = local_world_size // ulysses_size
    sp_groups = [
        list(range(node * local_world_size + group * ulysses_size,
                   node * local_world_size + (group + 1) * ulysses_size))
        for node in range(nodes)
        for group in range(groups_per_node)
    ]
    dp_groups = [
        [node * local_world_size + group * ulysses_size + sp_rank for node in range(nodes)]
        for group in range(groups_per_node)
        for sp_rank in range(ulysses_size)
    ]
    return sp_groups, dp_groups


def _resolve_distributed_topology(
    rank: int | None,
    world_size: int | None,
) -> tuple[int, int]:
    initialized = dist.is_available() and dist.is_initialized()
    actual_rank = dist.get_rank() if initialized else None
    actual_world_size = dist.get_world_size() if initialized else None

    if rank is None:
        rank = actual_rank if actual_rank is not None else 0
    if world_size is None:
        world_size = actual_world_size if actual_world_size is not None else 1
    rank = int(rank)
    world_size = int(world_size)

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if actual_rank is not None and rank != actual_rank:
        raise ValueError(f"rank={rank} does not match distributed rank={actual_rank}")
    if actual_world_size is not None and world_size != actual_world_size:
        raise ValueError(
            f"world_size={world_size} does not match distributed world_size={actual_world_size}"
        )
    return rank, world_size


def _validate_local_topology(
    *,
    rank: int,
    world_size: int,
    local_rank: int | None,
    local_world_size: int | None,
) -> None:
    if local_rank is None and local_world_size is None:
        return
    if local_rank is None or local_world_size is None:
        raise ValueError("local_rank and local_world_size must be provided together")
    local_rank = int(local_rank)
    local_world_size = int(local_world_size)
    if local_world_size <= 0 or world_size % local_world_size != 0:
        raise ValueError(
            f"local_world_size={local_world_size} must be positive and divide world_size={world_size}"
        )
    if local_rank < 0 or local_rank >= local_world_size:
        raise ValueError(
            f"local_rank must be in [0, {local_world_size}), got {local_rank}"
        )
    expected_local_rank = rank % local_world_size
    if local_rank != expected_local_rank:
        raise ValueError(
            f"local_rank={local_rank} does not match rank % local_world_size={expected_local_rank}"
        )


def configure_ulysses(
    *,
    ulysses_size: int,
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    local_world_size: int | None = None,
    use_fsdp_dp_group: bool = True,
    process_group_timeout_seconds: int = 600,
) -> UlyssesState:
    """Configure global DP x SP process groups.

    Every process creates every group in the same deterministic order. SP=1
    deliberately creates no groups and all tensor primitives become no-ops.
    """

    global _STATE
    ulysses_size = int(ulysses_size)
    process_group_timeout_seconds = int(process_group_timeout_seconds)
    if process_group_timeout_seconds <= 0:
        raise ValueError("Ulysses process-group timeout must be positive")
    rank, world_size = _resolve_distributed_topology(rank, world_size)
    _validate_local_topology(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
    )
    if local_world_size is None:
        local_world_size = world_size
    build_ulysses_rank_groups(world_size, ulysses_size)
    if local_world_size is not None and int(local_world_size) % ulysses_size != 0:
        raise ValueError(
            f"ulysses_size={ulysses_size} must divide local_world_size={int(local_world_size)}"
        )

    if ulysses_size == 1:
        _STATE = UlyssesState(
            dp_rank=rank,
            data_rank=rank,
            data_world_size=world_size,
        )
        return _STATE
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("ulysses_size > 1 requires torch.distributed to be initialized")

    sp_groups, dp_groups = build_ulysses_rank_groups(
        world_size,
        ulysses_size,
        local_world_size=local_world_size,
    )
    group_timeout = timedelta(seconds=process_group_timeout_seconds)
    group_backend = dist.get_backend()
    sp_group = None
    dp_group = None
    for group_index, ranks in enumerate(sp_groups):
        group = dist.new_group(
            ranks=ranks,
            timeout=group_timeout,
            backend=group_backend,
            group_desc=f"ulysses_sp_{group_index}",
        )
        if rank in ranks:
            sp_group = group
    for group_index, ranks in enumerate(dp_groups):
        group = dist.new_group(
            ranks=ranks,
            timeout=group_timeout,
            backend=group_backend,
            group_desc=f"ulysses_dp_{group_index}",
        )
        if rank in ranks:
            dp_group = group
    if sp_group is None or dp_group is None:
        raise RuntimeError(f"Failed to assign Ulysses groups for global rank {rank}")

    # ``dp_rank`` is the rank within the cross-node DP group.  This is the
    # source-rank coordinate used by FSDP sync_module_states and must not be
    # confused with the global row-major index.
    dp_rank = dist.get_rank(dp_group) if dp_group is not None else rank
    sp_rank = int(local_rank) % ulysses_size
    groups_per_node = int(local_world_size) // ulysses_size
    data_rank = (int(rank) // int(local_world_size)) * groups_per_node + (
        int(local_rank) // ulysses_size
    )
    _STATE = UlyssesState(
        size=ulysses_size,
        rank=sp_rank,
        group=sp_group,
        dp_group=dp_group if use_fsdp_dp_group else None,
        dp_rank=dp_rank,
        data_rank=data_rank,
        data_world_size=world_size // ulysses_size,
        fsdp_uses_dp_group=bool(use_fsdp_dp_group),
    )
    return _STATE


def synchronize_ulysses_mesh(
    state: UlyssesState | None = None,
    *,
    device: torch.device | None = None,
) -> None:
    """Synchronize DP before SP so overlapping NCCL groups use one order."""

    state = get_ulysses_state() if state is None else state
    if not state.enabled:
        return
    if state.dp_group is None or state.sp_group is None:
        raise RuntimeError("Ulysses mesh synchronization requires both DP and SP groups")
    barrier_kwargs: dict[str, object] = {}
    if device is not None and device.type == "cuda":
        barrier_kwargs["device_ids"] = [device.index if device.index is not None else 0]
    dist.barrier(group=state.dp_group, **barrier_kwargs)
    dist.barrier(group=state.sp_group, **barrier_kwargs)


def _state_for_tensor_primitive() -> UlyssesState | None:
    state = get_ulysses_state()
    if state.size == 1:
        return None
    if not state.enabled:
        raise RuntimeError("Ulysses sequence parallelism is not fully configured")
    return state


def pad_sequence_for_ulysses(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
    """Pad sequence dimension 1 to a multiple of the SP size."""

    state = _state_for_tensor_primitive()
    seq_len = int(hidden_states.size(1))
    if state is None:
        return hidden_states, position_ids, seq_len, seq_len
    if position_ids is not None and position_ids.size(1) != seq_len:
        raise ValueError(
            f"position_ids sequence length={position_ids.size(1)} does not match hidden_states={seq_len}"
        )

    padded_len = ((seq_len + state.size - 1) // state.size) * state.size
    if padded_len == seq_len:
        return hidden_states, position_ids, seq_len, padded_len
    pad_len = padded_len - seq_len
    hidden_shape = list(hidden_states.shape)
    hidden_shape[1] = pad_len
    hidden_states = torch.cat([hidden_states, hidden_states.new_zeros(hidden_shape)], dim=1)
    if position_ids is not None:
        position_shape = list(position_ids.shape)
        position_shape[1] = pad_len
        position_ids = torch.cat([position_ids, position_ids.new_zeros(position_shape)], dim=1)
    return hidden_states, position_ids, seq_len, padded_len


def shard_sequence(tensor: torch.Tensor) -> torch.Tensor:
    state = _state_for_tensor_primitive()
    if state is None:
        return tensor
    seq_len = int(tensor.size(1))
    if seq_len % state.size != 0:
        raise ValueError(
            f"sequence length={seq_len} must be divisible by ulysses_size={state.size}"
        )
    local_len = seq_len // state.size
    return tensor.narrow(1, state.rank * local_len, local_len).contiguous()


def shard_position_ids(
    position_ids: torch.Tensor | None,
    padded_len: int,
) -> torch.Tensor | None:
    if position_ids is None:
        return None
    state = _state_for_tensor_primitive()
    if state is None:
        return position_ids
    padded_len = int(padded_len)
    if padded_len % state.size != 0:
        raise ValueError(
            f"padded_len={padded_len} must be divisible by ulysses_size={state.size}"
        )
    if position_ids.size(1) != padded_len:
        raise ValueError(
            f"position_ids sequence length={position_ids.size(1)} does not match padded_len={padded_len}"
        )
    local_len = padded_len // state.size
    return position_ids.narrow(1, state.rank * local_len, local_len).contiguous()


def sequence_to_head(tensor: torch.Tensor) -> torch.Tensor:
    """Exchange ``[B, H, S_local, D]`` for ``[B, H/SP, S, D]``."""

    state = _state_for_tensor_primitive()
    if state is None:
        return tensor
    if tensor.ndim != 4:
        raise ValueError(f"sequence_to_head expects a 4-D tensor, got shape={tuple(tensor.shape)}")
    batch_size, num_heads, local_len, head_dim = tensor.shape
    if num_heads % state.size != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by ulysses_size={state.size}"
        )
    heads_per_rank = num_heads // state.size
    send = tensor.reshape(batch_size, state.size, heads_per_rank, local_len, head_dim)
    send = send.permute(1, 0, 2, 3, 4).contiguous()
    recv = _DifferentiableAllToAllSingle.apply(send, state.group)
    return recv.permute(1, 2, 0, 3, 4).reshape(
        batch_size,
        heads_per_rank,
        state.size * local_len,
        head_dim,
    )


def head_to_sequence(tensor: torch.Tensor) -> torch.Tensor:
    """Exchange ``[B, H/SP, S, D]`` for ``[B, H, S_local, D]``."""

    state = _state_for_tensor_primitive()
    if state is None:
        return tensor
    if tensor.ndim != 4:
        raise ValueError(f"head_to_sequence expects a 4-D tensor, got shape={tuple(tensor.shape)}")
    batch_size, heads_per_rank, seq_len, head_dim = tensor.shape
    if seq_len % state.size != 0:
        raise ValueError(
            f"sequence length={seq_len} must be divisible by ulysses_size={state.size}"
        )
    local_len = seq_len // state.size
    send = tensor.reshape(batch_size, heads_per_rank, state.size, local_len, head_dim)
    send = send.permute(2, 0, 1, 3, 4).contiguous()
    recv = _DifferentiableAllToAllSingle.apply(send, state.group)
    return recv.permute(1, 0, 2, 3, 4).reshape(
        batch_size,
        state.size * heads_per_rank,
        local_len,
        head_dim,
    )


def gather_sequence(tensor: torch.Tensor, original_len: int) -> torch.Tensor:
    state = _state_for_tensor_primitive()
    if state is None:
        return tensor
    gathered = _DifferentiableSequenceAllGather.apply(tensor.contiguous(), state.group)
    original_len = int(original_len)
    if original_len < 0 or original_len > gathered.size(1):
        raise ValueError(
            f"original_len must be in [0, {gathered.size(1)}], got {original_len}"
        )
    return gathered[:, :original_len].contiguous()


__all__ = [
    "UlyssesState",
    "build_ulysses_rank_groups",
    "configure_ulysses",
    "gather_sequence",
    "get_ulysses_state",
    "head_to_sequence",
    "pad_sequence_for_ulysses",
    "reset_ulysses",
    "sequence_to_head",
    "shard_position_ids",
    "shard_sequence",
    "synchronize_ulysses_mesh",
]

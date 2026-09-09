from __future__ import annotations

import functools
import json
import random
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


@dataclass(frozen=True)
class FSDPConfig:
    sharding_strategy: str = "NO_SHARD"
    backward_prefetch: str = "BACKWARD_PRE"
    cpu_offload: bool = False
    num_replicate: int = 1
    num_shard: int = 1
    sync_module_states: bool = False
    process_group_timeout_seconds: int = 600


def apply_transformer_checkpointing(
    model: torch.nn.Module,
    layer_classes: Iterable[type],
    *,
    enabled: bool = True,
    every_n_layers: int = 1,
    use_reentrant: bool = False,
) -> int:
    classes = tuple(layer_classes)
    if not enabled or not classes:
        return 0
    if every_n_layers <= 0:
        raise ValueError("activation_checkpoint.every_n_layers must be positive")

    layers = [module for module in model.modules() if isinstance(module, classes)]
    selected_layer_ids = {
        id(module) for index, module in enumerate(layers) if index % every_n_layers == 0
    }
    checkpoint_impl = (
        CheckpointImpl.REENTRANT if use_reentrant else CheckpointImpl.NO_REENTRANT
    )
    wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=checkpoint_impl)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: id(module) in selected_layer_ids,
    )
    return len(selected_layer_ids)


def scale_gradients(model: torch.nn.Module, scale: float) -> None:
    """Scale local or sharded gradients in place before gradient clipping."""
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def _all_ranks_have_batch(local_has_batch: bool, device: torch.device) -> bool:
    """Return true only when every data-parallel rank fetched a batch."""
    availability = torch.tensor(int(local_has_batch), dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(availability, op=dist.ReduceOp.MIN)
    return bool(availability.item())


def _build_hybrid_shard_process_groups(
    num_replicate: int,
    num_shard: int,
    timeout_seconds: int,
    ulysses_size: int = 1,
) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """Build deterministic ``DP-replicate x DP-shard x SP`` process groups."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    ulysses_size = int(ulysses_size)
    if ulysses_size <= 0:
        raise ValueError("ulysses_size must be positive")
    expected = num_replicate * num_shard * ulysses_size
    if expected != world_size:
        raise ValueError(f"HYBRID_SHARD expects world_size={expected}, got {world_size}")
    if timeout_seconds <= 0:
        raise ValueError("FSDP process-group timeout must be positive")

    timeout = timedelta(seconds=timeout_seconds)
    backend = dist.get_backend()
    shard_group = None
    replicate_group = None

    # Global ranks are laid out as (logical_dp_rank, sp_rank). Keep each HSDP
    # group on one SP lane so the two lanes own identically-shaped parameter
    # shards and can average those shard gradients before the optimizer step.
    for sp_rank in range(ulysses_size):
        for replicate_index in range(num_replicate):
            ranks = [
                ((replicate_index * num_shard + shard_index) * ulysses_size)
                + sp_rank
                for shard_index in range(num_shard)
            ]
            group = dist.new_group(
                ranks=ranks,
                timeout=timeout,
                backend=backend,
                group_desc=f"hsdp_sp{sp_rank}_shard_{replicate_index}",
            )
            if rank in ranks:
                shard_group = group

        for shard_index in range(num_shard):
            ranks = [
                ((replicate_index * num_shard + shard_index) * ulysses_size)
                + sp_rank
                for replicate_index in range(num_replicate)
            ]
            group = dist.new_group(
                ranks=ranks,
                timeout=timeout,
                backend=backend,
                group_desc=f"hsdp_sp{sp_rank}_replicate_{shard_index}",
            )
            if rank in ranks:
                replicate_group = group

    if shard_group is None or replicate_group is None:
        raise RuntimeError(f"Rank {rank} was not assigned to both HSDP process groups")
    return shard_group, replicate_group


def wrap_fsdp(
    model: torch.nn.Module,
    config: FSDPConfig,
    *,
    layer_classes: Iterable[type],
    device: torch.device,
    process_group: dist.ProcessGroup | None = None,
    ulysses_size: int = 1,
) -> FSDP:
    classes = tuple(layer_classes)
    if not classes:
        raise ValueError("FSDP requires at least one transformer layer class")
    strategy_name = config.sharding_strategy.upper()
    strategy = ShardingStrategy[strategy_name]
    fsdp_process_group = process_group
    if strategy_name == "HYBRID_SHARD":
        fsdp_process_group = _build_hybrid_shard_process_groups(
            config.num_replicate,
            config.num_shard,
            config.process_group_timeout_seconds,
            ulysses_size=ulysses_size,
        )
    elif strategy_name == "FULL_SHARD":
        # FULL_SHARD follows the supplied process group.  With Ulysses this is
        # the cross-node DP group, not the logical data world including SP.
        # ``num_shard`` remains metadata for HSDP and is not a second source of
        # truth for an explicit FULL_SHARD process group.
        pass
    param_init_fn = None
    if config.sync_module_states:
        def materialize_meta_module(module: torch.nn.Module) -> None:
            direct_tensors = [
                *module.parameters(recurse=False),
                *module.buffers(recurse=False),
            ]
            if any(tensor.is_meta for tensor in direct_tensors):
                module.to_empty(device=device, recurse=False)

        param_init_fn = materialize_meta_module
    wrapped = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(classes),
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=device if device.type == "cuda" else None,
        sharding_strategy=strategy,
        backward_prefetch=BackwardPrefetch[config.backward_prefetch.upper()],
        cpu_offload=CPUOffload(offload_params=config.cpu_offload),
        process_group=fsdp_process_group,
        use_orig_params=True,
        sync_module_states=config.sync_module_states,
        param_init_fn=param_init_fn,
    )
    return wrapped


def sync_gradients_across_group(
    model: torch.nn.Module,
    group: dist.ProcessGroup | None,
) -> None:
    if group is None:
        return
    group_size = dist.get_world_size(group)
    if group_size <= 1:
        return
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
        parameter.grad.div_(float(group_size))


def _distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    state_path = checkpoint_path / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Checkpoint is missing trainer_state.json: {checkpoint_path}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Checkpoint has invalid trainer_state.json: {checkpoint_path}") from exc
    if not isinstance(state, dict) or "step" not in state or "world_size" not in state:
        raise RuntimeError(
            f"Checkpoint trainer_state.json must contain step and world_size: {checkpoint_path}"
        )
    return state


def _validate_checkpoint_files(
    checkpoint_path: Path,
    *,
    fsdp_enabled: bool,
    require_complete: bool,
) -> dict[str, Any]:
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
    if require_complete and not (checkpoint_path / "COMPLETE").is_file():
        raise RuntimeError(f"Checkpoint is incomplete: {checkpoint_path}")

    state = _checkpoint_state(checkpoint_path)
    try:
        world_size = int(state["world_size"])
        int(state["step"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Checkpoint has invalid step/world_size metadata: {checkpoint_path}") from exc
    if world_size <= 0:
        raise RuntimeError(f"Checkpoint has invalid world_size={world_size}: {checkpoint_path}")
    if require_complete:
        marker_text = (checkpoint_path / "COMPLETE").read_text(encoding="utf-8").strip()
        # Version-1 checkpoints used a plain "ok" marker. Version 2 records
        # the step so a copied or partially assembled directory fails closed.
        if marker_text != "ok":
            try:
                marker = json.loads(marker_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Checkpoint has invalid COMPLETE marker: {checkpoint_path}") from exc
            try:
                marker_is_consistent = (
                    isinstance(marker, dict)
                    and marker.get("format_version") == 2
                    and int(marker.get("step", -1)) == int(state["step"])
                )
            except (TypeError, ValueError):
                marker_is_consistent = False
            if not marker_is_consistent:
                raise RuntimeError(
                    f"Checkpoint COMPLETE marker disagrees with trainer state: {checkpoint_path}"
                )

    if fsdp_enabled:
        if not (checkpoint_path / "state" / ".metadata").is_file():
            raise RuntimeError(f"Checkpoint is missing distributed state metadata: {checkpoint_path}")
    else:
        for filename in ("model.pt", "optimizer.pt"):
            if not (checkpoint_path / filename).is_file():
                raise RuntimeError(f"Checkpoint is missing {filename}: {checkpoint_path}")

    missing_rng = [
        rank
        for rank in range(world_size)
        if not (checkpoint_path / f"rng_state_rank_{rank:05d}.pt").is_file()
    ]
    if missing_rng:
        preview = ", ".join(str(rank) for rank in missing_rng[:8])
        suffix = "..." if len(missing_rng) > 8 else ""
        raise RuntimeError(
            f"Checkpoint is missing RNG state for ranks {preview}{suffix}: {checkpoint_path}"
        )
    return state


def _broadcast_checkpoint_control(
    control: dict[str, Any],
    rank: int,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return control
    payload: list[dict[str, Any] | None] = [control if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0, group=process_group)
    assert payload[0] is not None
    return payload[0]


def reject_future_checkpoints(checkpoint_root: str | Path, resume_step: int) -> None:
    """Reject complete checkpoints that would belong to a discarded future trajectory."""
    root = Path(checkpoint_root)
    if not root.exists():
        return
    future: list[tuple[int, Path]] = []
    for candidate in root.iterdir():
        if not candidate.is_dir() or not candidate.name.isdecimal():
            continue
        if not (candidate / "COMPLETE").is_file():
            continue
        state = _checkpoint_state(candidate)
        saved_step = int(state["step"])
        if saved_step != int(candidate.name):
            raise RuntimeError(
                f"Checkpoint directory {candidate} encodes step {candidate.name}, "
                f"but trainer_state.json records step {saved_step}"
            )
        if saved_step > int(resume_step):
            future.append((saved_step, candidate))
    if future:
        future.sort()
        preview = ", ".join(str(path) for _, path in future[:5])
        suffix = " ..." if len(future) > 5 else ""
        raise RuntimeError(
            f"Refusing to resume from step {resume_step}: checkpoint root contains "
            f"{len(future)} complete future checkpoint(s): {preview}{suffix}. "
            "Use a new results directory or explicitly archive the discarded trajectory."
        )


def save_checkpoint(
    checkpoint_root: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    fsdp_enabled: bool,
    rank: int,
    trainer_state: dict[str, Any] | None = None,
    fsdp_process_group: dist.ProcessGroup | None = None,
    control_process_group: dist.ProcessGroup | None = None,
    write_distributed_state: bool = True,
) -> Path:
    checkpoint_root = Path(checkpoint_root)
    output = checkpoint_root / f"{step:07d}"
    world_size = _distributed_world_size()
    if not (dist.is_available() and dist.is_initialized()) and rank != 0:
        raise ValueError("A non-distributed checkpoint save requires rank=0")

    control: dict[str, Any] = {}
    if rank == 0:
        try:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            if output.exists():
                try:
                    existing_state = _validate_checkpoint_files(
                        output,
                        fsdp_enabled=fsdp_enabled,
                        require_complete=True,
                    )
                except (FileNotFoundError, RuntimeError):
                    quarantine = checkpoint_root / f".{output.name}.incomplete-{uuid.uuid4().hex}"
                    output.rename(quarantine)
                else:
                    if int(existing_state["step"]) != int(step):
                        raise RuntimeError(
                            f"Existing checkpoint metadata step={existing_state['step']} "
                            f"does not match requested step={step}: {output}"
                        )
                    requested_trajectory = (trainer_state or {}).get("trajectory_id")
                    existing_trajectory = existing_state.get("trajectory_id")
                    if (
                        requested_trajectory is not None
                        and existing_trajectory != requested_trajectory
                    ):
                        raise RuntimeError(
                            "Existing checkpoint belongs to a different training trajectory: "
                            f"requested={requested_trajectory!r} "
                            f"existing={existing_trajectory!r} path={output}"
                        )
                    control = {"action": "reuse"}
            if not control:
                attempt = checkpoint_root / f".{output.name}.tmp-{uuid.uuid4().hex}"
                attempt.mkdir()
                control = {"action": "write", "attempt": str(attempt)}
        except Exception as exc:
            control = {"action": "error", "message": f"{type(exc).__name__}: {exc}"}
    control = _broadcast_checkpoint_control(control, rank, control_process_group)
    if control["action"] == "error":
        raise RuntimeError(f"Checkpoint preparation failed: {control['message']}")
    if control["action"] == "reuse":
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=control_process_group)
        return output

    attempt = Path(control["attempt"])
    local_error = ""
    try:
        if fsdp_enabled:
            assert isinstance(model, FSDP)
            options = StateDictOptions(full_state_dict=False, cpu_offload=True)
            model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer, options=options)
            if write_distributed_state:
                save(
                    {"model": model_state_dict, "optimizer": optimizer_state_dict},
                    storage_writer=FileSystemWriter(str(attempt / "state")),
                    process_group=fsdp_process_group,
                )
        elif rank == 0:
            torch.save(model.state_dict(), attempt / "model.pt")
            torch.save(optimizer.state_dict(), attempt / "optimizer.pt")
        rng_state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()
        torch.save(rng_state, attempt / f"rng_state_rank_{rank:05d}.pt")
    except Exception as exc:
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"

    if dist.is_available() and dist.is_initialized():
        control_world_size = dist.get_world_size(control_process_group)
        write_errors: list[str | None] = [None] * control_world_size
        dist.all_gather_object(
            write_errors,
            local_error,
            group=control_process_group,
        )
    else:
        write_errors = [local_error]
    write_errors = [error for error in write_errors if error]
    if write_errors:
        raise RuntimeError("Checkpoint write failed: " + "; ".join(write_errors))

    if rank == 0:
        publish_error = ""
        try:
            state = dict(trainer_state or {})
            state["step"] = int(step)
            state["world_size"] = world_size
            (attempt / "trainer_state.json").write_text(
                json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _validate_checkpoint_files(
                attempt,
                fsdp_enabled=fsdp_enabled,
                require_complete=False,
            )
            (attempt / "COMPLETE").write_text(
                json.dumps({"format_version": 2, "step": int(step)}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            attempt.rename(output)
        except Exception as exc:
            publish_error = f"{type(exc).__name__}: {exc}"
        publish_control = {"error": publish_error}
    else:
        publish_control = {}
    publish_control = _broadcast_checkpoint_control(
        publish_control,
        rank,
        control_process_group,
    )
    if publish_control["error"]:
        raise RuntimeError(f"Checkpoint publish failed: {publish_control['error']}")
    return output


def load_checkpoint(
    checkpoint: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    fsdp_enabled: bool,
    rank: int,
    fsdp_process_group: dist.ProcessGroup | None = None,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    state = _validate_checkpoint_files(
        checkpoint_path,
        fsdp_enabled=fsdp_enabled,
        require_complete=True,
    )
    if checkpoint_root is not None:
        reject_future_checkpoints(checkpoint_root, int(state["step"]))
    current_world_size = _distributed_world_size()
    saved_world_size = int(state["world_size"])
    if saved_world_size != current_world_size:
        raise ValueError(
            f"Exact resume requires the saved world_size={saved_world_size}, got {current_world_size}"
        )
    if fsdp_enabled:
        assert isinstance(model, FSDP)
        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer, options=options)
        load(
            {"model": model_state_dict, "optimizer": optimizer_state_dict},
            storage_reader=FileSystemReader(str(checkpoint_path / "state")),
            process_group=fsdp_process_group,
        )
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optimizer_state_dict,
            options=options,
        )
    else:
        model.load_state_dict(torch.load(checkpoint_path / "model.pt", map_location="cpu", weights_only=True))
        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt", map_location="cpu", weights_only=True))
    rng_state_path = checkpoint_path / f"rng_state_rank_{rank:05d}.pt"
    rng_state = torch.load(rng_state_path, map_location="cpu", weights_only=False)
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and "cuda" in rng_state:
        torch.cuda.set_rng_state(rng_state["cuda"])
    return state


__all__ = [
    "FSDPConfig",
    "_all_ranks_have_batch",
    "apply_transformer_checkpointing",
    "load_checkpoint",
    "reject_future_checkpoints",
    "scale_gradients",
    "save_checkpoint",
    "sync_gradients_across_group",
    "wrap_fsdp",
]

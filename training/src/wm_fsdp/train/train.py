from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader

from wm_fsdp.data import (
    DIAGNOSTIC_KEYS,
    MEGATRON_PACKED_SAMPLER_VERSION,
    MegatronPackedCollator,
    MegatronPackedDataset,
    MultimodalCollator,
    MultimodalWebDataset,
    load_data_config,
)
from wm_fsdp.train.fsdp_utils import (
    FSDPConfig,
    apply_transformer_checkpointing,
    load_checkpoint,
    scale_gradients,
    save_checkpoint,
    sync_gradients_across_group,
    wrap_fsdp,
)
from wm_fsdp.train.ulysses import (
    configure_ulysses,
    reset_ulysses,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multimodal causal language model with PyTorch FSDP")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-config")
    parser.add_argument("--model-checkpoint")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--special-tokens-file")
    parser.add_argument("--results-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--ulysses-size", type=int)
    parser.add_argument("--sharding-strategy")
    parser.add_argument("--num-replicate", type=int)
    parser.add_argument("--num-shard", type=int)
    parser.add_argument("--sync-each-micro-batch", type=int, choices=(0, 1))
    parser.add_argument("--activation-checkpoint", type=int, choices=(0, 1))
    parser.add_argument("--activation-checkpoint-every-n-layers", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument(
        "--resume-allow-step-extension",
        action="store_true",
        help="Explicitly continue beyond a checkpoint's saved stop step while preserving its scheduler contract.",
    )
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--grad-accumulation-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--resume-from")
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-final", dest="save_final", action="store_true")
    save_group.add_argument("--no-save-final", dest="save_final", action="store_false")
    parser.set_defaults(save_final=None)
    parser.add_argument("--no-fsdp", action="store_true")
    return parser.parse_args()


def _init_distributed() -> tuple[int, int, torch.device]:
    use_cuda = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        # Bind the process before NCCL initialization so barriers and object
        # collectives cannot guess the wrong rank-to-device mapping.
        torch.cuda.set_device(device)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        timeout_seconds = int(os.environ.get("WM_FSDP_DIST_TIMEOUT_SECONDS", "600"))
        if timeout_seconds <= 0:
            raise ValueError("WM_FSDP_DIST_TIMEOUT_SECONDS must be positive")
        init_kwargs: dict[str, Any] = {
            "backend": "nccl" if use_cuda else "gloo",
            "timeout": timedelta(seconds=timeout_seconds),
        }
        if use_cuda:
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size, device


def _is_sharded_init_source_rank(
    rank: int,
    sharding_strategy: str,
    num_shard: int,
    ulysses_size: int = 1,
) -> bool:
    """Return whether this rank seeds its FSDP shard group from pretrained weights."""
    strategy = sharding_strategy.upper()
    if num_shard <= 0:
        raise ValueError("fsdp.num_shard must be positive")
    if int(ulysses_size) > 1:
        # The initialized Ulysses FSDP group has the authoritative mapping;
        # callers that have not built the group retain the conservative
        # legacy answer.  train() uses the group-local dp_rank directly.
        return True
    if strategy == "HYBRID_SHARD":
        # The (replicate, shard) device mesh is row-major, so each replica's
        # shard-group rank 0 is global rank replica_index * num_shard.
        return rank % num_shard == 0
    if strategy == "FULL_SHARD":
        return rank == 0
    if strategy == "SHARD_GRAD_OP":
        return rank == 0
    raise ValueError(f"fsdp.sharded_init does not support {sharding_strategy}")


def _should_sync_sharded_init_states(
    *,
    sharded_init: bool,
    ulysses_size: int,
) -> bool:
    """Use FSDP1 state broadcast only outside the Ulysses local-load path."""

    return bool(sharded_init) and int(ulysses_size) == 1


def _write_sharded_init_evidence(
    results_dir: Path,
    model: torch.nn.Module,
    *,
    rank: int,
    local_rank: int,
    sharding_strategy: str,
    num_shard: int,
    source: bool,
) -> None:
    parameters = list(model.parameters())
    meta_tensors = sum(parameter.is_meta for parameter in parameters)
    if source and meta_tensors:
        raise RuntimeError(f"Sharded-init source rank {rank} unexpectedly has meta parameters")
    if not source and meta_tensors != len(parameters):
        raise RuntimeError(
            f"Sharded-init meta rank {rank} materialized {len(parameters) - meta_tensors} parameters early"
        )
    evidence_dir = results_dir / "sharded_init"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output = evidence_dir / f"rank_{rank:05d}.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            {
                "rank": rank,
                "local_rank": local_rank,
                "strategy": sharding_strategy,
                "num_shard": num_shard,
                "source": source,
                "parameter_tensors": len(parameters),
                "meta_parameter_tensors": meta_tensors,
                "parameters": sum(parameter.numel() for parameter in parameters),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _logger(rank: int, results_dir: Path) -> logging.Logger:
    logger = logging.getLogger("wm_fsdp.train")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if rank == 0:
        file_handler = logging.FileHandler(results_dir / "train.log", mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _synchronized_next(
    data_iterator: Iterator[Any],
    device: torch.device,
) -> tuple[Any | None, bool, float, float]:
    """Fetch one batch and propagate DataLoader failures to every rank."""
    fetch_started = time.perf_counter()
    batch: Any | None = None
    local_error: Exception | None = None
    try:
        batch = next(data_iterator)
        local_status = 1
    except StopIteration:
        local_status = 0
    except Exception as exc:
        local_status = -1
        local_error = exc
    loader_time = time.perf_counter() - fetch_started

    if not (dist.is_available() and dist.is_initialized()):
        if local_error is not None:
            raise local_error
        return batch, local_status == 1, loader_time, 0.0

    sync_started = time.perf_counter()
    global_status = torch.tensor(local_status, dtype=torch.int32, device=device)
    dist.all_reduce(global_status, op=dist.ReduceOp.MIN)
    sync_time = time.perf_counter() - sync_started
    if int(global_status.item()) < 0:
        rank = dist.get_rank()
        local_message = ""
        if local_error is not None:
            local_message = f"rank {rank}: {type(local_error).__name__}: {local_error}"
        messages: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(messages, local_message)
        failures = [message for message in messages if message]
        detail = "; ".join(failures) if failures else "error details unavailable"
        synchronized_error = RuntimeError(f"Distributed DataLoader failure: {detail}")
        if local_error is not None:
            raise synchronized_error from local_error
        raise synchronized_error
    return batch, int(global_status.item()) == 1, loader_time, sync_time


def _learning_rate(
    config: Any,
    step: int,
    *,
    total_steps_override: int | None = None,
) -> float:
    base_lr = float(config.training.lr)
    warmup = int(config.training.get("warmup_steps", 0))
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if str(config.training.get("lr_scheduler", "constant")) != "cosine":
        return base_lr
    schedule_total_steps = (
        int(config.training.total_steps)
        if total_steps_override is None
        else int(total_steps_override)
    )
    decay_steps = max(1, schedule_total_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
    minimum = float(config.training.get("min_lr", 0.0))
    return minimum + 0.5 * (base_lr - minimum) * (1.0 + math.cos(math.pi * progress))


def _should_save_checkpoint(step: int, save_every: int, first_checkpoint_step: int = 0) -> bool:
    return step > 0 and (step % save_every == 0 or step == first_checkpoint_step)


def _estimate_transformer_training_flops(
    llm_config: Any,
    padded_tokens: int,
    squared_sequence_lengths: int,
    target_tokens: int,
) -> float:
    """Estimate training FLOPs with a dense backbone and selective LM head."""
    hidden_size = int(llm_config.hidden_size)
    num_layers = int(llm_config.num_hidden_layers)
    ffn_hidden_size = int(llm_config.intermediate_size)
    num_heads = int(llm_config.num_attention_heads)
    num_query_groups = int(getattr(llm_config, "num_key_value_heads", num_heads))
    head_dim = int(getattr(llm_config, "head_dim", hidden_size // num_heads))
    vocab_size = int(llm_config.vocab_size)
    query_projection_ratio = head_dim * num_heads / hidden_size
    gated_multiplier = (
        1.5 if str(getattr(llm_config, "hidden_act", "")).lower() in {"silu", "swish"} else 1.0
    )
    expansion_factor = 3 * 2 * 2
    backbone_per_token_flops = (
        expansion_factor * num_layers * hidden_size * ffn_hidden_size * gated_multiplier
        + expansion_factor
        * num_layers
        * hidden_size**2
        * (1.0 + num_query_groups / num_heads)
        * query_projection_ratio
    )
    lm_head_per_target_token_flops = 3 * 2 * hidden_size * vocab_size
    attention_sequence_flops = (
        expansion_factor
        * num_layers
        * hidden_size
        * query_projection_ratio
        * squared_sequence_lengths
        / 2.0
    )
    return (
        float(padded_tokens) * backbone_per_token_flops
        + float(target_tokens) * lm_head_per_target_token_flops
        + attention_sequence_flops
    )


def _format_training_log(
    *,
    iteration: int,
    total_steps: int,
    consumed_samples: int,
    step_time_sec: float,
    throughput_tflops_per_gpu: float,
    learning_rate: float,
    global_batch_size: int,
    loss: float,
    image_ce: float,
    text_ce: float,
    grad_norm: float,
    skipped_iterations: int = 0,
    nan_iterations: int = 0,
) -> str:
    return (
        f"iteration {iteration:8d}/{total_steps:8d} | "
        f"consumed samples: {consumed_samples:12d} | "
        f"elapsed time per iteration (ms): {step_time_sec * 1000.0:.1f} | "
        f"throughput per GPU (TFLOP/s/GPU): {throughput_tflops_per_gpu:.1f} | "
        f"learning rate: {learning_rate:.6E} | "
        f"global batch size: {global_batch_size:5d} | "
        f"total loss: {loss:.6E} | "
        f"image ce: {image_ce:.6E} | "
        f"text ce: {text_ce:.6E} | "
        f"grad norm: {grad_norm:.3f} | "
        f"number of skipped iterations: {skipped_iterations:3d} | "
        f"number of nan iterations: {nan_iterations:3d} |"
    )


def _set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _distributed_sum(values: list[float], device: torch.device) -> list[float]:
    result = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result.cpu().tolist()


def _distributed_max(values: list[float], device: torch.device) -> list[float]:
    result = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.cpu().tolist()


def _distributed_sum_tensors(values: list[torch.Tensor]) -> list[torch.Tensor]:
    """Sum equally ordered statistic tensors with one distributed collective."""
    if not values:
        return []
    sizes = [value.numel() for value in values]
    result = torch.cat(
        [value.detach().reshape(-1).to(dtype=torch.float64) for value in values]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    chunks = result.cpu().split(sizes)
    return [chunk.reshape(value.shape) for chunk, value in zip(chunks, values)]


def _distributed_sum_int_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    result = value.detach().to(device=device, dtype=torch.int64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result.cpu()


def _accumulate_group_loss(
    names: list[str],
    indices: dict[str, int],
    sample_losses: torch.Tensor,
    numerators: torch.Tensor,
    denominators: torch.Tensor,
) -> None:
    """Attribute per-sample multimodal objectives to their data source."""
    detached_losses = sample_losses.detach().float().reshape(-1)
    if len(names) != int(detached_losses.numel()):
        raise ValueError("Source metadata does not match per-sample loss count")
    for name, sample_loss in zip(names, detached_losses):
        index = indices[name]
        numerators[index] += sample_loss
        denominators[index] += 1.0


def _optimizer_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    *,
    no_decay_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Match Megatron: do not decay bias or one-dimensional norm parameters."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        canonical_name = name.replace("_fsdp_wrapped_module.", "")
        exclude = (
            canonical_name in no_decay_names
            if no_decay_names is not None
            else name.endswith(".bias") or parameter.ndim <= 1
        )
        if exclude:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _optimizer_no_decay_names(model: torch.nn.Module) -> set[str]:
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (name.endswith(".bias") or parameter.ndim <= 1)
    }


def _write_metrics(handle: Any, metrics: dict[str, Any]) -> None:
    handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def _truncate_metrics_after_step(metrics_path: Path, resume_step: int) -> int:
    """Atomically discard metrics from a trajectory after its resume checkpoint."""
    if not metrics_path.exists():
        return 0
    kept: list[str] = []
    removed = 0
    previous_step = -1
    for line_number, raw_line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
            step = int(row["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid metric record at {metrics_path}:{line_number}"
            ) from exc
        if step <= int(resume_step):
            if step <= previous_step:
                raise ValueError(
                    f"Metric steps through resume step {resume_step} must be strictly increasing; "
                    f"found {step} after {previous_step} at {metrics_path}:{line_number}"
                )
            kept.append(raw_line + "\n")
            previous_step = step
        else:
            removed += 1
    if removed:
        temporary = metrics_path.with_name(
            f".{metrics_path.name}.resume-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text("".join(kept), encoding="utf-8")
            os.replace(temporary, metrics_path)
        finally:
            temporary.unlink(missing_ok=True)
    return removed


def _apply_overrides(config: Any, args: argparse.Namespace) -> None:
    if args.dataset_config:
        config.data.dataset_config_file = args.dataset_config
    if args.model_checkpoint:
        config.model.llm.checkpoint = args.model_checkpoint
    if args.tokenizer_path:
        config.model.llm.tokenizer_path = args.tokenizer_path
    if args.special_tokens_file:
        config.model.llm.special_tokens_file = args.special_tokens_file
    if args.results_dir:
        config.training.results_dir = args.results_dir
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.prefetch_factor is not None:
        config.data.prefetch_factor = args.prefetch_factor
    if args.ulysses_size is not None:
        config.parallel.ulysses_size = args.ulysses_size
    if args.sharding_strategy is not None:
        config.fsdp.sharding_strategy = args.sharding_strategy
    if args.num_replicate is not None:
        config.fsdp.num_replicate = args.num_replicate
    if args.num_shard is not None:
        config.fsdp.num_shard = args.num_shard
    if args.sync_each_micro_batch is not None:
        config.fsdp.sync_each_micro_batch = args.sync_each_micro_batch
    if getattr(args, "activation_checkpoint", None) is not None:
        config.activation_checkpoint.enabled = bool(args.activation_checkpoint)
    if getattr(args, "activation_checkpoint_every_n_layers", None) is not None:
        config.activation_checkpoint.every_n_layers = args.activation_checkpoint_every_n_layers
    if args.total_steps is not None:
        config.training.total_steps = args.total_steps
    if getattr(args, "resume_allow_step_extension", False):
        config.training.resume_allow_step_extension = True
    if getattr(args, "global_batch_size", None) is not None:
        config.training.global_batch_size = args.global_batch_size
    if args.grad_accumulation_steps is not None:
        config.training.grad_accumulation_steps = args.grad_accumulation_steps
    if args.warmup_steps is not None:
        config.training.warmup_steps = args.warmup_steps
    if args.lr is not None:
        config.training.lr = args.lr
    if args.min_lr is not None:
        config.training.min_lr = args.min_lr
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.max_grad_norm is not None:
        config.training.max_grad_norm = args.max_grad_norm
    if args.save_every is not None:
        config.training.save_every = args.save_every
    if args.log_every is not None:
        config.training.log_every = args.log_every
    if args.resume_from is not None:
        config.training.resume_from = args.resume_from
    if getattr(args, "save_final", None) is not None:
        config.training.save_final = bool(args.save_final)


def _save_resolved_config(config: Any, path: Path) -> None:
    """Persist the effective configuration instead of OmegaConf interpolations."""
    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.save(resolved, path)


def _resolve_runtime_parallel_config(
    config: Any,
    world_size: int,
    local_world_size: int | None = None,
) -> dict[str, int | str]:
    """Resolve topology-dependent FSDP and global-batch values into the config."""

    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    ulysses_size = int(config.get("parallel", {}).get("ulysses_size", 1))
    if ulysses_size <= 0 or world_size % ulysses_size:
        raise ValueError(
            f"parallel.ulysses_size={ulysses_size} must be positive and divide "
            f"world_size={world_size}"
        )
    data_parallel_size = world_size // ulysses_size
    strategy = str(config.fsdp.sharding_strategy).upper()
    config.fsdp.sharding_strategy = strategy

    # With node-local Ulysses SP, each FSDP process group connects the same
    # local SP lane/group across nodes. Its shard size is therefore the node
    # count, while data_world_size remains world_size / ulysses_size.
    fsdp_group_size = (
        world_size // int(local_world_size)
        if ulysses_size > 1 and local_world_size is not None
        else data_parallel_size
    )

    if strategy == "HYBRID_SHARD":
        if "num_shard" not in config.fsdp:
            raise ValueError("HYBRID_SHARD requires fsdp.num_shard in the YAML config")
        num_shard = int(config.fsdp.num_shard)
        if num_shard <= 0 or data_parallel_size % num_shard:
            raise ValueError(
                f"HYBRID_SHARD fsdp.num_shard={num_shard} must be positive and divide "
                f"logical data parallel size={data_parallel_size}"
            )
        expected_num_replicate = data_parallel_size // num_shard
    elif strategy in {"FULL_SHARD", "SHARD_GRAD_OP"}:
        num_shard = fsdp_group_size
        expected_num_replicate = 1
    elif strategy == "NO_SHARD":
        num_shard = 1
        expected_num_replicate = 1
    else:
        raise ValueError(
            "fsdp.sharding_strategy must be one of NO_SHARD, SHARD_GRAD_OP, "
            f"FULL_SHARD, HYBRID_SHARD; got {strategy}"
        )

    if "num_shard" in config.fsdp and strategy != "HYBRID_SHARD":
        configured_num_shard = int(config.fsdp.num_shard)
        if configured_num_shard != num_shard:
            raise ValueError(
                f"{strategy} requires fsdp.num_shard={num_shard} for logical data "
                f"parallel size={data_parallel_size}, got {configured_num_shard}"
            )
    if "num_replicate" in config.fsdp:
        configured_num_replicate = int(config.fsdp.num_replicate)
        if configured_num_replicate != expected_num_replicate:
            raise ValueError(
                f"{strategy} requires fsdp.num_replicate={expected_num_replicate} for "
                f"logical data parallel size={data_parallel_size}, got "
                f"{configured_num_replicate}"
            )
    config.fsdp.num_shard = num_shard
    config.fsdp.num_replicate = expected_num_replicate

    batch_size = int(config.data.batch_size)
    if batch_size <= 0:
        raise ValueError(f"data.batch_size must be positive, got {batch_size}")
    samples_per_micro_batch = data_parallel_size * batch_size
    configured_global_batch = config.training.get("global_batch_size", None)
    configured_accumulation = config.training.get("grad_accumulation_steps", None)
    if configured_global_batch is None:
        if configured_accumulation is None:
            raise ValueError(
                "training.global_batch_size or training.grad_accumulation_steps is required"
            )
        grad_accumulation_steps = int(configured_accumulation)
        if grad_accumulation_steps <= 0:
            raise ValueError("training.grad_accumulation_steps must be positive")
        global_batch_size = samples_per_micro_batch * grad_accumulation_steps
        config.training.global_batch_size = global_batch_size
    else:
        global_batch_size = int(configured_global_batch)
        if global_batch_size <= 0:
            raise ValueError("training.global_batch_size must be positive")
        if global_batch_size % samples_per_micro_batch:
            raise ValueError(
                f"training.global_batch_size={global_batch_size} must be divisible by "
                f"logical_dp={data_parallel_size} * data.batch_size={batch_size}"
            )
        grad_accumulation_steps = global_batch_size // samples_per_micro_batch
        if configured_accumulation is not None and int(configured_accumulation) != grad_accumulation_steps:
            raise ValueError(
                f"training.grad_accumulation_steps={int(configured_accumulation)} conflicts "
                f"with derived value={grad_accumulation_steps} for global_batch_size="
                f"{global_batch_size}"
            )
    config.training.grad_accumulation_steps = grad_accumulation_steps
    return {
        "world_size": world_size,
        "ulysses_size": ulysses_size,
        "data_parallel_size": data_parallel_size,
        "sharding_strategy": strategy,
        "num_replicate": expected_num_replicate,
        "num_shard": num_shard,
        "batch_size": batch_size,
        "global_batch_size": global_batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
    }


def main() -> None:
    args = _parse_args()
    rank, world_size, device = _init_distributed()
    config = OmegaConf.load(args.config)
    _apply_overrides(config, args)
    results_dir = Path(config.training.results_dir)
    runtime_parallel = _resolve_runtime_parallel_config(
        config,
        world_size,
        local_world_size=int(
            os.environ.get(
                "LOCAL_WORLD_SIZE",
                1 if world_size == 1 else torch.cuda.device_count(),
            )
        ),
    )
    save_final = bool(config.training.get("save_final", False))
    resume_from = config.training.get("resume_from")
    # The output directory is shared by all ranks, so this check can be local.
    # Avoid a default-group object collective before Ulysses groups are built.
    if (results_dir / "metrics.jsonl").is_file() and not resume_from:
        raise RuntimeError(
            "Refusing to append to existing metrics without training.resume_from: "
            f"{results_dir}"
        )
    ulysses_size = int(config.get("parallel", {}).get("ulysses_size", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(
        os.environ.get(
            "LOCAL_WORLD_SIZE",
            1 if world_size == 1 else torch.cuda.device_count(),
        )
    )
    ulysses = configure_ulysses(
        ulysses_size=ulysses_size,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        use_fsdp_dp_group=ulysses_size > 1,
        process_group_timeout_seconds=int(
            os.environ.get("WM_FSDP_DIST_TIMEOUT_SECONDS", "600")
        ),
    )
    # Construct process groups before loading the HF checkpoint and wrapping
    # with FSDP. Startup barriers are avoided because the first real
    # collectives establish communicator ordering.
    checkpoint_dir = results_dir / "checkpoints"
    if rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _save_resolved_config(config, results_dir / "resolved_config.yaml")
    logger = _logger(rank, results_dir)
    metrics_handle = None
    seed = int(config.training.get("seed", 42))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    if world_size > 1 and args.no_fsdp:
        raise ValueError("--no-fsdp can only be used with a single process")
    fsdp_enabled = world_size > 1

    sharding_strategy = str(config.fsdp.sharding_strategy).upper()
    sharded_init = bool(config.fsdp.get("sharded_init", False))
    if sharded_init and not fsdp_enabled:
        raise ValueError("fsdp.sharded_init requires distributed FSDP")
    num_shard = int(config.fsdp.get("num_shard", ulysses.data_world_size))
    if sharded_init and ulysses.enabled:
        # Each Ulysses SP lane owns its own FSDP DP group.  Only that group's
        # rank zero reads the full checkpoint; all other ranks construct meta
        # parameters and receive the state through FSDP's group-local
        # sync_module_states broadcast.  This reduces a 64-rank 34B startup
        # from 64 full reads to one read per FSDP group.
        # Pick a different node as source for each local SP/FSDP group.  The
        # FSDP DP group is ordered by node, so this spreads the one full-file
        # read per group across the cluster instead of concentrating all
        # source reads on node zero.
        nodes = max(1, world_size // local_world_size)
        local_group_index = int(local_rank) // int(ulysses.size)
        source_dp_rank = (local_group_index * int(ulysses.size) + int(ulysses.sp_rank)) % nodes
        sharded_init_source = int(ulysses.dp_rank) == source_dp_rank
        sync_sharded_init_states = True
    else:
        sharded_init_source = (
            _is_sharded_init_source_rank(
                rank, sharding_strategy, num_shard, ulysses_size=ulysses.size
            )
            if sharded_init
            else False
        )
        sync_sharded_init_states = _should_sync_sharded_init_states(
            sharded_init=sharded_init,
            ulysses_size=ulysses.size,
        )
    from wm_fsdp.models import CausalLMModel

    model_config = config.model
    if sharded_init and not sharded_init_source:
        model_config = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
        model_config.llm.init_on_meta = True
    model = CausalLMModel(model_config)
    if fsdp_enabled and not sharded_init:
        # FSDP mixed precision casts full parameters for compute while retaining
        # FP32 original shards and Adam states, matching Megatron main-param
        # semantics.
        model = model.float().to(device)
    elif not fsdp_enabled:
        # Keep the checkpoint's configured dtype for an explicit single-process
        # --no-fsdp run. A fixed 16K 4B smoke test otherwise keeps a complete
        # FP32 model, gradients, and Adam states on one GPU and cannot execute
        # more than one optimizer step on a 140 GiB device. This path is for
        # local data/model validation; distributed FSDP semantics are unchanged.
        model = model.to(device)
    else:
        # Non-Ulysses sharded initialization uses one pretrained CPU source per
        # shard group and materializes the remaining ranks from meta tensors.
        # Ulysses uses one local HF source per FSDP DP group and synchronizes
        # the source state into the group's meta ranks while wrapping.
        model = (
            model.to(dtype=torch.bfloat16)
            if ulysses.enabled
            else model.float()
        )
    if sharded_init:
        _write_sharded_init_evidence(
            results_dir,
            model,
            rank=rank,
            local_rank=local_rank,
            sharding_strategy=sharding_strategy,
            num_shard=num_shard,
            source=sharded_init_source,
        )
    llm_flops_config = model.llm_config
    image_loss_weight = model.image_loss_weight
    text_loss_weight = model.text_loss_weight
    model.enable_ulysses()
    model.train()
    activation_checkpoint_config = config.get("activation_checkpoint", {})
    activation_checkpoint_enabled = bool(
        activation_checkpoint_config.get("enabled", True)
    )
    activation_checkpoint_every_n = int(
        activation_checkpoint_config.get("every_n_layers", 1)
    )
    activation_checkpoint_use_reentrant = bool(
        activation_checkpoint_config.get("use_reentrant", False)
    )
    data_config = load_data_config(config.data.dataset_config_file)
    data_config_sha256 = hashlib.sha256(
        json.dumps(data_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    data_type = str(data_config.get("type", ""))
    if data_type == "webdataset":
        dataset = MultimodalWebDataset(
            data_config,
            rank=ulysses.data_rank,
            world_size=ulysses.data_world_size,
            batch_size_per_rank=int(config.data.batch_size),
        )
        collator = MultimodalCollator(
            model,
            prompt_template=str(config.data.prompt_template),
            max_length=int(config.data.max_length),
        )
    elif data_type == "megatron_packed":
        dataset = MegatronPackedDataset(
            data_config,
            rank=ulysses.data_rank,
            world_size=ulysses.data_world_size,
            batch_size_per_rank=int(config.data.batch_size),
        )
        collator = MegatronPackedCollator(
            data_config,
            max_length=int(config.data.max_length),
        )
    else:
        raise ValueError(
            "Unsupported data config type. Expected 'webdataset' or 'megatron_packed', "
            f"got {data_type!r}"
        )
    dataset_names = [spec.name for spec in dataset.specs]
    dataset_indices = {name: index for index, name in enumerate(dataset_names)}
    data_diagnostic_keys = tuple(getattr(dataset, "diagnostic_keys", DIAGNOSTIC_KEYS))
    num_workers = int(config.data.num_workers)
    loader_timeout_seconds = int(config.data.get("loader_timeout_seconds", 0))
    if loader_timeout_seconds < 0:
        raise ValueError("data.loader_timeout_seconds must be non-negative")
    # PyTorch requires timeout=0 for a single-process DataLoader. Keeping the
    # configured timeout for worker-backed loaders preserves the production
    # hang detector while allowing deterministic single-GPU/no-worker smoke
    # tests and debugging.
    effective_loader_timeout_seconds = loader_timeout_seconds if num_workers > 0 else 0
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.data.batch_size),
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        prefetch_factor=int(config.data.get("prefetch_factor", 2)) if num_workers > 0 else None,
        drop_last=True,
        generator=torch.Generator(),
        timeout=effective_loader_timeout_seconds,
    )

    fsdp_config = FSDPConfig(
        sharding_strategy=str(config.fsdp.sharding_strategy),
        backward_prefetch=str(config.fsdp.backward_prefetch),
        cpu_offload=bool(config.fsdp.cpu_offload),
        num_replicate=int(config.fsdp.get("num_replicate", 1)),
        num_shard=int(config.fsdp.get("num_shard", ulysses.data_world_size)),
        sync_module_states=sync_sharded_init_states,
        process_group_timeout_seconds=int(
            os.environ.get("WM_FSDP_DIST_TIMEOUT_SECONDS", "600")
        ),
    )
    sync_each_micro_batch_value = int(
        config.fsdp.get("sync_each_micro_batch", 0)
    )
    if sync_each_micro_batch_value not in (0, 1):
        raise ValueError("fsdp.sync_each_micro_batch must be 0 or 1")
    sync_each_micro_batch = bool(sync_each_micro_batch_value)
    # Record classification by name before FSDP can expose original parameters
    # as one-dimensional shards or replace meta Parameters during materialization.
    no_decay_names = _optimizer_no_decay_names(model)
    if fsdp_enabled:
        model = wrap_fsdp(
            model,
            fsdp_config,
            layer_classes=model.fsdp_transformer_layer_cls,
            device=device,
            process_group=ulysses.dp_group,
            ulysses_size=ulysses.size,
        )
    apply_transformer_checkpointing(
        model,
        model.fsdp_transformer_layer_cls,
        enabled=activation_checkpoint_enabled,
        every_n_layers=activation_checkpoint_every_n,
        use_reentrant=activation_checkpoint_use_reentrant,
    )
    optimizer_param_groups = _optimizer_param_groups(
        model,
        float(config.training.weight_decay),
        no_decay_names=no_decay_names,
    )

    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=float(config.training.lr),
        betas=(float(config.training.beta1), float(config.training.beta2)),
        eps=float(config.training.eps),
    )
    trainer_state: dict[str, Any] = {"step": 0, "epoch": 0, "batches_in_epoch": 0}
    if resume_from:
        trainer_state.update(
            load_checkpoint(
                resume_from,
                model,
                optimizer,
                fsdp_enabled=fsdp_enabled,
                rank=rank,
                fsdp_process_group=ulysses.dp_group,
                checkpoint_root=checkpoint_dir,
            )
        )

    resume_extension = bool(config.training.get("resume_allow_step_extension", False))
    saved_schedule_total_steps: int | None = None
    if resume_from and resume_extension:
        saved_optimizer_contract = trainer_state.get("optimizer_contract")
        if not isinstance(saved_optimizer_contract, dict) or "total_steps" not in saved_optimizer_contract:
            raise ValueError(
                "Step-extension resume requires optimizer_contract.total_steps in the checkpoint"
            )
        saved_schedule_total_steps = int(saved_optimizer_contract["total_steps"])
        requested_total_steps = int(config.training.total_steps)
        saved_step = int(trainer_state["step"])
        if requested_total_steps <= saved_step:
            raise ValueError(
                "Step-extension resume requires training.total_steps to be greater than "
                f"the checkpoint step={saved_step}, got {requested_total_steps}"
            )
        if requested_total_steps < saved_schedule_total_steps:
            raise ValueError(
                "Step-extension resume cannot shorten the saved scheduler horizon: "
                f"saved={saved_schedule_total_steps}, requested={requested_total_steps}"
            )

    resume_contract = {
        "batch_size_per_data_rank": int(config.data.batch_size),
        "num_workers": num_workers,
        "grad_accumulation_steps": int(config.training.grad_accumulation_steps),
        "seed": seed,
        "dataset_config_file": str(Path(config.data.dataset_config_file).resolve()),
        "dataset_config_sha256": data_config_sha256,
        "data_world_size": ulysses.data_world_size,
        "data_sampler_version": (
            MEGATRON_PACKED_SAMPLER_VERSION
            if data_type == "megatron_packed"
            else "webdataset_stateful_v1"
        ),
        "ulysses_size": ulysses.size,
        "fsdp_contract": {
            "sharding_strategy": fsdp_config.sharding_strategy.upper(),
            "num_replicate": fsdp_config.num_replicate,
            "num_shard": fsdp_config.num_shard,
            "sync_each_micro_batch": sync_each_micro_batch,
        },
        "activation_checkpoint_contract": {
            "enabled": activation_checkpoint_enabled,
            "every_n_layers": activation_checkpoint_every_n,
            "use_reentrant": activation_checkpoint_use_reentrant,
        },
        "loss_objective": {
            "name": "grouped_multimodal_pooled_v2",
            "image_weight": image_loss_weight,
            "text_weight": text_loss_weight,
        },
        "optimizer_contract": {
            "lr_scheduler": str(config.training.get("lr_scheduler", "constant")),
            "lr": float(config.training.lr),
            "min_lr": float(config.training.get("min_lr", 0.0)),
            "warmup_steps": int(config.training.get("warmup_steps", 0)),
            # An explicit step-extension run keeps the checkpoint's scheduler
            # horizon immutable; only the termination target is extended.
            "total_steps": (
                saved_schedule_total_steps
                if saved_schedule_total_steps is not None
                else int(config.training.total_steps)
            ),
            "beta1": float(config.training.beta1),
            "beta2": float(config.training.beta2),
            "eps": float(config.training.eps),
            "weight_decay": float(config.training.weight_decay),
            "max_grad_norm": float(config.training.max_grad_norm),
        },
    }
    if sharded_init:
        resume_contract["fsdp_contract"]["sharded_init"] = True
        resume_contract["fsdp_contract"]["sharded_init_source_policy"] = (
            "every_rank_local_hf"
            if ulysses.enabled
            else "per_shard_group_rank0"
        )
    if resume_from:
        missing_contract = [
            key
            for key in (
                "loss_objective",
                "optimizer_contract",
                "ulysses_size",
                "data_world_size",
                "data_sampler_version",
            )
            if key not in trainer_state
        ]
        if missing_contract:
            raise ValueError(
                "Checkpoint predates the current loss/parallel contract and cannot be "
                f"resumed exactly; missing {missing_contract}. Start a new run from the HF model checkpoint."
            )
    for key, current_value in resume_contract.items():
        if key in trainer_state and trainer_state[key] != current_value:
            raise ValueError(
                f"Exact resume requires saved {key}={trainer_state[key]}, got {current_value}"
            )

    total_steps = int(config.training.total_steps)
    grad_accumulation = int(config.training.grad_accumulation_steps)
    if grad_accumulation <= 0:
        raise ValueError("grad_accumulation_steps must be positive")
    log_every = int(config.training.log_every)
    save_every = int(config.training.save_every)
    first_checkpoint_step = int(config.training.get("first_checkpoint_step", 0))
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if save_every <= 0:
        raise ValueError("save_every must be positive")
    if first_checkpoint_step < 0:
        raise ValueError("first_checkpoint_step must be non-negative")
    train_step = int(trainer_state["step"])
    trajectory_id = str(trainer_state.get("trajectory_id", ""))
    if not trajectory_id:
        if resume_from:
            trajectory_id = hashlib.sha256(
                str(Path(resume_from).resolve()).encode("utf-8")
            ).hexdigest()
        elif rank == 0:
            trajectory_id = uuid.uuid4().hex
        if dist.is_available() and dist.is_initialized():
            trajectory_payload = [trajectory_id if rank == 0 else None]
            dist.broadcast_object_list(trajectory_payload, src=0)
            trajectory_id = str(trajectory_payload[0])
    resume_contract["trajectory_id"] = trajectory_id

    metrics_control: dict[str, Any] = {"error": "", "removed": 0}
    if rank == 0 and resume_from:
        try:
            metrics_control["removed"] = _truncate_metrics_after_step(
                results_dir / "metrics.jsonl", train_step
            )
        except Exception as exc:
            metrics_control["error"] = f"{type(exc).__name__}: {exc}"
    if dist.is_available() and dist.is_initialized():
        metrics_payload = [metrics_control if rank == 0 else None]
        dist.broadcast_object_list(metrics_payload, src=0)
        metrics_control = metrics_payload[0]
    if metrics_control["error"]:
        raise RuntimeError(f"Metrics resume preparation failed: {metrics_control['error']}")
    if rank == 0:
        if metrics_control["removed"]:
            logger.info(
                "Removed %d metric record(s) after resume step %d",
                metrics_control["removed"],
                train_step,
            )
        metrics_handle = (results_dir / "metrics.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
    logical_global_batch = (
        ulysses.data_world_size * int(config.data.batch_size) * grad_accumulation
    )
    consumed_samples = int(
        trainer_state.get("consumed_samples", train_step * logical_global_batch)
    )
    epoch = int(trainer_state.get("epoch", 0))
    resume_batches_in_epoch = int(trainer_state.get("batches_in_epoch", 0))
    optimizer.zero_grad(set_to_none=True)
    dataset.set_epoch(epoch)
    direct_data_resume = callable(getattr(dataset, "set_start_batch", None))
    if direct_data_resume:
        dataset.set_start_batch(resume_batches_in_epoch)
    assert dataloader.generator is not None
    dataloader.generator.manual_seed(
        seed + ulysses.data_rank + epoch * ulysses.data_world_size
    )
    data_iterator = iter(dataloader)
    batches_in_epoch = 0
    if resume_batches_in_epoch and direct_data_resume:
        logger.info(
            "Restored deterministic data position directly: epoch=%d batches_in_epoch=%d",
            epoch,
            resume_batches_in_epoch,
        )
        batches_in_epoch = resume_batches_in_epoch
    elif resume_batches_in_epoch:
        logger.info("Restoring data position: epoch=%d batches_in_epoch=%d", epoch, resume_batches_in_epoch)
        for _ in range(resume_batches_in_epoch):
            _, all_ranks_have_batch, _, _ = _synchronized_next(data_iterator, device)
            if not all_ranks_have_batch:
                raise RuntimeError(
                    "Checkpoint data position exceeds the resumed epoch; the dataset or worker topology changed"
                )
            batches_in_epoch += 1
    accumulated_micro_batches = 0
    step_loss_is_finite = torch.ones((), dtype=torch.bool, device=device)
    step_loss_numerator = torch.zeros((), dtype=torch.float32, device=device)
    step_image_ce_numerator = torch.zeros((), dtype=torch.float32, device=device)
    step_text_ce_numerator = torch.zeros((), dtype=torch.float32, device=device)
    step_image_tokens = torch.zeros((), dtype=torch.float32, device=device)
    step_text_tokens = torch.zeros((), dtype=torch.float32, device=device)
    step_target_tokens = torch.zeros((), dtype=torch.float32, device=device)
    step_input_tokens = torch.zeros((), dtype=torch.float32, device=device)
    step_samples = 0
    step_padded_tokens = 0
    step_squared_sequence_lengths = 0
    step_sample_ids: list[str] = []
    step_dataset_samples = torch.zeros(len(dataset_names), dtype=torch.float32, device=device)
    step_dataset_target_tokens = torch.zeros(len(dataset_names), dtype=torch.float32, device=device)
    step_dataset_loss_numerator = torch.zeros(len(dataset_names), dtype=torch.float32, device=device)
    step_dataset_loss_samples = torch.zeros(len(dataset_names), dtype=torch.float32, device=device)
    step_data_time = 0.0
    step_loader_time = 0.0
    step_data_sync_time = 0.0
    step_start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        while train_step < total_steps:
            batch, all_ranks_have_batch, loader_time, data_sync_time = _synchronized_next(
                data_iterator, device
            )
            step_loader_time += loader_time
            step_data_sync_time += data_sync_time
            step_data_time += loader_time + data_sync_time
            if not all_ranks_have_batch:
                any_rank_consumed_batch = torch.tensor(int(batches_in_epoch > 0), dtype=torch.int32, device=device)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(any_rank_consumed_batch, op=dist.ReduceOp.MAX)
                if not bool(any_rank_consumed_batch.item()):
                    raise RuntimeError(
                        "WebDataset cannot provide one full batch on every rank; "
                        "check shard count, rank/worker partitioning, and batch_size"
                    )
                epoch += 1
                dataset.set_epoch(epoch)
                if direct_data_resume:
                    dataset.set_start_batch(0)
                dataloader.generator.manual_seed(
                    seed + ulysses.data_rank + epoch * ulysses.data_world_size
                )
                data_iterator = iter(dataloader)
                batches_in_epoch = 0
                continue
            assert batch is not None
            batches_in_epoch += 1
            batch_dataset_names = [str(value) for value in batch.pop("dataset_names")]
            batch_sample_ids = [str(value) for value in batch.pop("sample_ids")]
            batch = _move_to_device(batch, device)
            target_tokens_per_sample = (batch["labels"][:, 1:] != -100).sum(dim=1)
            target_tokens = target_tokens_per_sample.sum()
            input_tokens = batch["attention_mask"].sum()
            local_samples = int(batch["input_ids"].shape[0])
            padded_sequence_length = int(batch["input_ids"].shape[1])
            step_padded_tokens += local_samples * padded_sequence_length
            step_squared_sequence_lengths += local_samples * padded_sequence_length**2
            if len(batch_dataset_names) != local_samples:
                raise RuntimeError("Batch metadata does not match the tensor batch size")
            if len(batch_sample_ids) != local_samples:
                raise RuntimeError("Batch sample IDs do not match the tensor batch size")
            step_sample_ids.extend(batch_sample_ids)
            for sample_index, dataset_name in enumerate(batch_dataset_names):
                if dataset_name not in dataset_indices:
                    raise RuntimeError(f"Unknown batch source metadata: dataset={dataset_name!r}")
                dataset_index = dataset_indices[dataset_name]
                sample_target_tokens = target_tokens_per_sample[sample_index].detach().float()
                step_dataset_samples[dataset_index] += 1
                step_dataset_target_tokens[dataset_index] += sample_target_tokens

            autocast_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                loss_output = model(**batch)
            sample_losses = loss_output["sample_loss"]
            loss_sum = loss_output["loss_sum"]
            sample_image_ce = loss_output["sample_image_ce"]
            sample_text_ce = loss_output["sample_text_ce"]
            finite_loss = (
                torch.isfinite(sample_losses.detach()).all()
                & torch.isfinite(sample_image_ce.detach()).all()
                & torch.isfinite(sample_text_ce.detach()).all()
            )
            step_loss_is_finite.logical_and_(finite_loss)
            if int(target_tokens.item()) <= 0:
                raise RuntimeError("Micro-batch contains no supervised target tokens")
            if int(sample_losses.numel()) != local_samples:
                raise RuntimeError("Model per-sample losses do not match the micro-batch size")
            is_last_micro_batch = accumulated_micro_batches + 1 == grad_accumulation
            sync_context = (
                contextlib.nullcontext()
                if (
                    sync_each_micro_batch
                    or is_last_micro_batch
                    or not isinstance(model, FSDP)
                )
                else model.no_sync()
            )
            with sync_context:
                loss_sum.backward()

            _accumulate_group_loss(
                batch_dataset_names,
                dataset_indices,
                sample_losses,
                step_dataset_loss_numerator,
                step_dataset_loss_samples,
            )
            step_loss_numerator += loss_sum.detach().float()
            step_image_ce_numerator += (
                loss_output["image_ce"].detach().float() * local_samples
            )
            step_text_ce_numerator += (
                loss_output["text_ce"].detach().float() * local_samples
            )
            step_image_tokens += loss_output["image_token_count"].detach().float()
            step_text_tokens += loss_output["text_token_count"].detach().float()
            step_target_tokens += target_tokens.detach().float()
            step_input_tokens += input_tokens.detach().float()
            step_samples += local_samples
            accumulated_micro_batches += 1
            if accumulated_micro_batches < grad_accumulation:
                continue
            globally_finite = step_loss_is_finite.to(dtype=torch.int32)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(globally_finite, op=dist.ReduceOp.MIN)
            if not bool(globally_finite.item()):
                raise FloatingPointError(f"Non-finite loss at optimizer step {train_step + 1}")

            global_samples_for_backward = torch.tensor(
                float(step_samples), dtype=torch.float32, device=device
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(
                    global_samples_for_backward,
                    op=dist.ReduceOp.SUM,
                    group=ulysses.dp_group,
                )
            if float(global_samples_for_backward.item()) <= 0:
                raise RuntimeError("Training step contains no samples")
            sync_gradients_across_group(
                model,
                ulysses.sp_group if ulysses.enabled else None,
            )
            # FSDP averages over logical DP ranks. Undo that average while dividing
            # the accumulated sample-objective gradient by the logical global batch.
            gradient_scale = ulysses.data_world_size / float(
                global_samples_for_backward.item()
            )
            scale_gradients(model, gradient_scale)

            learning_rate = _learning_rate(
                config,
                train_step,
                total_steps_override=saved_schedule_total_steps,
            )
            _set_learning_rate(optimizer, learning_rate)
            max_norm = float(config.training.max_grad_norm)
            if isinstance(model, FSDP):
                grad_norm = model.clip_grad_norm_(max_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            grad_norm_value = float(grad_norm.detach().float().item() if torch.is_tensor(grad_norm) else grad_norm)
            if not math.isfinite(grad_norm_value):
                raise FloatingPointError(f"Non-finite gradient norm at optimizer step {train_step + 1}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_step += 1

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_time = time.perf_counter() - step_start_time
            compute_time = max(0.0, step_time - step_data_time)
            peak_allocated_gb = 0.0
            peak_reserved_gb = 0.0
            if device.type == "cuda":
                peak_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)

            totals = _distributed_sum(
                [
                    float(step_loss_numerator.item()),
                    float(step_image_ce_numerator.item()),
                    float(step_text_ce_numerator.item()),
                    float(step_image_tokens.item()),
                    float(step_text_tokens.item()),
                    float(step_target_tokens.item()),
                    float(step_input_tokens.item()),
                    float(step_samples),
                    float(step_padded_tokens),
                    float(step_squared_sequence_lengths),
                ],
                device,
            )
            maxima = _distributed_max(
                [
                    step_time,
                    step_data_time,
                    step_loader_time,
                    step_data_sync_time,
                    compute_time,
                    grad_norm_value,
                    peak_allocated_gb,
                    peak_reserved_gb,
                ],
                device,
            )
            (
                global_loss_numerator,
                global_image_ce_numerator,
                global_text_ce_numerator,
                global_image_tokens,
                global_text_tokens,
                global_target_tokens,
                global_input_tokens,
                global_samples,
                global_padded_tokens,
                global_squared_sequence_lengths,
            ) = totals
            (
                global_dataset_samples,
                global_dataset_target_tokens,
                global_dataset_loss_numerator,
                global_dataset_loss_samples,
            ) = _distributed_sum_tensors(
                [
                    step_dataset_samples,
                    step_dataset_target_tokens,
                    step_dataset_loss_numerator,
                    step_dataset_loss_samples,
                ]
            )
            global_data_diagnostics = _distributed_sum_int_tensor(
                dataset.diagnostics_tensor(), device
            )
            if ulysses.size > 1:
                duplicate_factor = float(ulysses.size)
                global_loss_numerator /= duplicate_factor
                global_image_ce_numerator /= duplicate_factor
                global_text_ce_numerator /= duplicate_factor
                global_image_tokens /= duplicate_factor
                global_text_tokens /= duplicate_factor
                global_target_tokens /= duplicate_factor
                global_input_tokens /= duplicate_factor
                global_samples /= duplicate_factor
                global_padded_tokens /= duplicate_factor
                global_squared_sequence_lengths /= duplicate_factor
                for value in (
                    global_dataset_samples,
                    global_dataset_target_tokens,
                    global_dataset_loss_numerator,
                    global_dataset_loss_samples,
                ):
                    value.div_(duplicate_factor)
                global_data_diagnostics.div_(int(duplicate_factor), rounding_mode="floor")
            (
                max_step_time,
                max_data_time,
                max_loader_time,
                max_data_sync_time,
                max_compute_time,
                max_grad_norm,
                max_allocated_gb,
                max_reserved_gb,
            ) = maxima
            if global_target_tokens <= 0:
                raise RuntimeError("Training step contains no supervised target tokens")
            if global_samples <= 0:
                raise RuntimeError("Training step contains no samples")
            mean_loss = global_loss_numerator / global_samples
            mean_image_ce = global_image_ce_numerator / global_samples
            mean_text_ce = global_text_ce_numerator / global_samples
            dataset_sample_metrics = {
                name: int(global_dataset_samples[index].item())
                for index, name in enumerate(dataset_names)
                if global_dataset_samples[index].item() > 0
            }
            dataset_target_token_metrics = {
                name: int(global_dataset_target_tokens[index].item())
                for index, name in enumerate(dataset_names)
                if global_dataset_target_tokens[index].item() > 0
            }
            dataset_loss_metrics = {
                name: float(
                    global_dataset_loss_numerator[index].item()
                    / global_dataset_loss_samples[index].item()
                )
                for index, name in enumerate(dataset_names)
                if global_dataset_loss_samples[index].item() > 0
            }
            dataset_loss_sample_metrics = {
                name: int(global_dataset_loss_samples[index].item())
                for index, name in enumerate(dataset_names)
                if global_dataset_loss_samples[index].item() > 0
            }
            data_diagnostic_metrics = {
                name: {
                    key: int(global_data_diagnostics[dataset_index, diagnostic_index].item())
                    for diagnostic_index, key in enumerate(data_diagnostic_keys)
                    if global_data_diagnostics[dataset_index, diagnostic_index].item() > 0
                }
                for dataset_index, name in enumerate(dataset_names)
            }
            data_diagnostic_metrics = {
                name: values for name, values in data_diagnostic_metrics.items() if values
            }
            samples_per_second = global_samples / max(max_step_time, 1.0e-12)
            input_tokens_per_second = global_input_tokens / max(max_step_time, 1.0e-12)
            target_tokens_per_second = global_target_tokens / max(max_step_time, 1.0e-12)
            global_batch_size = int(global_samples)
            consumed_samples += global_batch_size
            training_flops = _estimate_transformer_training_flops(
                llm_flops_config,
                int(global_padded_tokens),
                int(global_squared_sequence_lengths),
                int(global_target_tokens),
            )
            throughput_tflops_per_gpu = training_flops / (
                max(max_step_time, 1.0e-12) * 1.0e12 * world_size
            )
            metrics = {
                "step": train_step,
                "epoch": epoch,
                "loss": mean_loss,
                "image_ce": mean_image_ce,
                "text_ce": mean_text_ce,
                "image_loss_weight": image_loss_weight,
                "text_loss_weight": text_loss_weight,
                "learning_rate": learning_rate,
                "grad_norm": max_grad_norm,
                "step_time_sec": max_step_time,
                "data_time_sec": max_data_time,
                "loader_time_sec": max_loader_time,
                "data_sync_time_sec": max_data_sync_time,
                "compute_time_sec": max_compute_time,
                "data_time_ratio": max_data_time / max(max_step_time, 1.0e-12),
                "samples": int(global_samples),
                "consumed_samples": consumed_samples,
                "input_tokens": int(global_input_tokens),
                "target_tokens": int(global_target_tokens),
                "image_target_tokens": int(global_image_tokens),
                "text_target_tokens": int(global_text_tokens),
                "samples_per_sec": samples_per_second,
                "input_tokens_per_sec": input_tokens_per_second,
                "target_tokens_per_sec": target_tokens_per_second,
                "throughput_tflops_per_gpu": throughput_tflops_per_gpu,
                "peak_memory_allocated_gb": max_allocated_gb,
                "peak_memory_reserved_gb": max_reserved_gb,
                "world_size": world_size,
                "data_parallel_size": ulysses.data_world_size,
                "ulysses_size": ulysses.size,
                "micro_batches_per_rank": accumulated_micro_batches,
                "dataset_samples": dataset_sample_metrics,
                "dataset_target_tokens": dataset_target_token_metrics,
                "dataset_loss": dataset_loss_metrics,
                "dataset_loss_samples": dataset_loss_sample_metrics,
                "data_diagnostics_cumulative": data_diagnostic_metrics,
                "timestamp_unix": time.time(),
            }
            if bool(config.training.get("log_sample_ids", False)):
                gathered_sample_ids: list[list[str]] = [step_sample_ids]
                if dist.is_available() and dist.is_initialized():
                    gathered_sample_ids = [[] for _ in range(world_size)]
                    dist.all_gather_object(gathered_sample_ids, step_sample_ids)
                if ulysses.size > 1:
                    gathered_sample_ids = gathered_sample_ids[:: ulysses.size]
                sample_id_payload = json.dumps(
                    gathered_sample_ids,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                metrics["sample_ids_sha256"] = hashlib.sha256(sample_id_payload).hexdigest()
            if rank == 0:
                assert metrics_handle is not None
                _write_metrics(metrics_handle, metrics)
                if train_step % log_every == 0:
                    logger.info(
                        _format_training_log(
                            iteration=train_step,
                            total_steps=total_steps,
                            consumed_samples=consumed_samples,
                            step_time_sec=max_step_time,
                            throughput_tflops_per_gpu=throughput_tflops_per_gpu,
                            learning_rate=learning_rate,
                            global_batch_size=global_batch_size,
                            loss=mean_loss,
                            image_ce=mean_image_ce,
                            text_ce=mean_text_ce,
                            grad_norm=max_grad_norm,
                        )
                    )

            if _should_save_checkpoint(train_step, save_every, first_checkpoint_step):
                # Match the reference Ulysses checkpoint topology: all ranks in
                # the global-rank-0 FSDP subgroup participate in the DCP write,
                # while the equivalent model replicas do not write the same
                # directory concurrently.
                write_distributed_state = (
                    not ulysses.enabled
                    or (
                        ulysses.sp_rank == 0
                        and local_rank // ulysses.size == 0
                    )
                )
                save_checkpoint(
                    checkpoint_dir,
                    train_step,
                    model,
                    optimizer,
                    fsdp_enabled=fsdp_enabled,
                    rank=rank,
                    fsdp_process_group=ulysses.dp_group,
                    control_process_group=None,
                    write_distributed_state=write_distributed_state,
                    trainer_state={
                        "epoch": epoch,
                        "batches_in_epoch": batches_in_epoch,
                        "consumed_samples": consumed_samples,
                        **resume_contract,
                    },
                )

            accumulated_micro_batches = 0
            step_loss_is_finite.fill_(True)
            step_loss_numerator.zero_()
            step_image_ce_numerator.zero_()
            step_text_ce_numerator.zero_()
            step_image_tokens.zero_()
            step_text_tokens.zero_()
            step_target_tokens.zero_()
            step_input_tokens.zero_()
            step_samples = 0
            step_padded_tokens = 0
            step_squared_sequence_lengths = 0
            step_sample_ids.clear()
            step_dataset_samples.zero_()
            step_dataset_target_tokens.zero_()
            step_dataset_loss_numerator.zero_()
            step_dataset_loss_samples.zero_()
            step_data_time = 0.0
            step_loader_time = 0.0
            step_data_sync_time = 0.0
            step_start_time = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        if save_final:
            write_distributed_state = (
                not ulysses.enabled
                or (
                    ulysses.sp_rank == 0
                    and local_rank // ulysses.size == 0
                )
            )
            save_checkpoint(
                checkpoint_dir,
                train_step,
                model,
                optimizer,
                fsdp_enabled=fsdp_enabled,
                rank=rank,
                fsdp_process_group=ulysses.dp_group,
                control_process_group=None,
                write_distributed_state=write_distributed_state,
                trainer_state={
                    "epoch": epoch,
                    "batches_in_epoch": batches_in_epoch,
                    "consumed_samples": consumed_samples,
                    **resume_contract,
                },
            )
    finally:
        if metrics_handle is not None:
            metrics_handle.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        reset_ulysses()


if __name__ == "__main__":
    main()

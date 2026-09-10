from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import torch

from wm_fsdp.data import (
    MultimodalWebDataset,
    load_data_config,
    resolve_target_image_segments,
    validate_multimodal_sample,
)


def _iter_limited(dataset: MultimodalWebDataset, samples: int) -> Iterable[dict[str, Any]]:
    if samples < 0:
        raise ValueError("--samples cannot be negative")
    return dataset if samples == 0 else islice(dataset, samples)


def _update_id_range(stats: dict[str, dict[str, int]], id_space: str, grid: torch.Tensor) -> None:
    current = stats.setdefault(id_space, {"minimum": int(grid.min()), "maximum": int(grid.max()), "tokens": 0})
    current["minimum"] = min(current["minimum"], int(grid.min()))
    current["maximum"] = max(current["maximum"], int(grid.max()))
    current["tokens"] += grid.numel()


def _expected_weights(dataset: MultimodalWebDataset) -> dict[str, float]:
    positive = {spec.name: spec.weight for spec in dataset.specs if spec.weight > 0}
    total = sum(positive.values())
    return {name: value / total for name, value in positive.items()} if total else {}


def _mixture_report(dataset: MultimodalWebDataset, counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    observed = {name: count / total for name, count in sorted(counts.items())} if total else {}
    expected = _expected_weights(dataset)
    return {
        "expected": expected,
        "observed": observed,
        "absolute_error": {
            name: abs(observed.get(name, 0.0) - expected_value)
            for name, expected_value in expected.items()
        },
        "note": "Weighted sampling is stochastic; use a large --samples value before enforcing tolerance.",
    }


def _partition_report(config: dict[str, Any], *, world_size: int, num_workers: int) -> dict[str, Any]:
    if world_size <= 0 or num_workers <= 0:
        raise ValueError("partition world size and workers must be positive")
    unweighted = copy.deepcopy(config)
    unweighted["weighted_sampling"] = False
    assigned: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    collisions: list[dict[str, str]] = []
    expected: set[str] = set()
    for rank in range(world_size):
        rank_dataset = MultimodalWebDataset(unweighted, rank=rank, world_size=world_size)
        expected.update(f"{ref.dataset.name}:{ref.path}" for ref in rank_dataset.shards)
        for worker_id in range(num_workers):
            partition = f"rank={rank},worker={worker_id}"
            refs = rank_dataset.shards_for_worker(worker_id=worker_id, num_workers=num_workers)
            identifiers = [f"{ref.dataset.name}:{ref.path}" for ref in refs]
            assigned[partition] = identifiers
            for identifier in identifiers:
                if identifier in seen:
                    collisions.append({"shard": identifier, "first": seen[identifier], "second": partition})
                else:
                    seen[identifier] = partition
    missing = sorted(expected - seen.keys())
    return {
        "passed": not collisions and not missing,
        "world_size": world_size,
        "workers_per_rank": num_workers,
        "total_unique_shards": len(expected),
        "assigned_shards": {partition: len(values) for partition, values in assigned.items()},
        "collisions": collisions,
        "missing": missing,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_data_config(args.config)
    dataset = MultimodalWebDataset(config, rank=0, world_size=1)
    datasets: Counter[str] = Counter()
    target_shapes: Counter[str] = Counter()
    id_spaces: Counter[str] = Counter()
    id_ranges: dict[str, dict[str, int]] = {}
    condition_images = 0
    multi_output_manifests = 0
    target_images = 0
    decoded = 0

    for sample in _iter_limited(dataset, args.samples):
        validate_multimodal_sample(
            sample,
            visual_vocab_size=args.visual_vocab_size,
            visual_token_offset=args.visual_token_offset,
        )
        decoded += 1
        datasets[str(sample["dataset_name"])] += 1
        output_count = int(sample.get("manifest_output_count", 1))
        if output_count > 1:
            multi_output_manifests += 1
        targets = resolve_target_image_segments(sample)
        if len(targets) != output_count:
            raise ValueError(
                f"{sample['sample_id']}: decoded {len(targets)} targets but manifest declares {output_count}"
            )
        target_images += len(targets)
        for target in targets:
            target_shapes["x".join(str(value) for value in target["tokens"].shape)] += 1
            target_space = str(target.get("id_space", "raw"))
            id_spaces[target_space] += 1
            _update_id_range(id_ranges, target_space, target["tokens"])
        for segment in sample["input_segments"]:
            if segment.get("type") == "image":
                condition_images += 1
                _update_id_range(id_ranges, str(segment.get("id_space", "raw")), segment["tokens"])

    if decoded == 0:
        raise RuntimeError("WebDataset yielded no samples")
    mixture = _mixture_report(dataset, datasets)
    if args.weight_tolerance is not None:
        violations = {
            name: error
            for name, error in mixture["absolute_error"].items()
            if error > args.weight_tolerance
        }
        if violations:
            raise ValueError(
                f"Observed dataset mixture exceeds --weight-tolerance={args.weight_tolerance}: {violations}"
            )

    report: dict[str, Any] = {
        "status": "passed",
        "config": str(Path(args.config).resolve()),
        "resolved_shards": len(dataset.shards),
        "decoded_samples": decoded,
        "datasets": dict(datasets),
        "target_shapes": dict(target_shapes),
        "target_id_spaces": dict(id_spaces),
        "visual_id_ranges": id_ranges,
        "condition_images": condition_images,
        "reader_diagnostics": dataset.diagnostics_snapshot(),
        "sampling_plan_cycle_0": dataset.sampling_plan_snapshot(cycle=0),
        "multi_target_contract": {
            "multi_output_manifests": multi_output_manifests,
            "decoded_target_images": target_images,
            "ignored_output_images": 0,
            "semantics": "All manifest outputs are decoded and supervised in manifest order.",
        },
        "mixture": mixture,
    }
    if args.check_partitions:
        report["partition_check"] = _partition_report(
            config,
            world_size=args.partition_world_size,
            num_workers=args.partition_workers,
        )
        if not report["partition_check"]["passed"]:
            raise RuntimeError(f"Rank/worker partition check failed: {report['partition_check']}")
    return report


def _emit(report: dict[str, Any], report_path: str | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate U0 IBQ WebDataset shards")
    parser.add_argument("--config", required=True, help="WebDataset JSON config")
    parser.add_argument("--samples", type=int, default=16, help="Samples to decode; 0 scans the full epoch")
    parser.add_argument("--visual-vocab-size", type=int, default=131072)
    parser.add_argument("--visual-token-offset", type=int, default=151854)
    parser.add_argument("--weight-tolerance", type=float, default=None)
    parser.add_argument("--check-partitions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partition-world-size", type=int, default=2)
    parser.add_argument("--partition-workers", type=int, default=2)
    parser.add_argument("--report", default=None, help="Optional JSON report output path")
    args = parser.parse_args()

    try:
        report = validate(args)
    except Exception as exc:
        report = {
            "status": "failed",
            "config": str(Path(args.config).resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _emit(report, args.report)
        raise SystemExit(1) from exc
    _emit(report, args.report)


if __name__ == "__main__":
    main()

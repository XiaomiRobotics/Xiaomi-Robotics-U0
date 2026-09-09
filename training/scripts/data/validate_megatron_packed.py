#!/usr/bin/env python3
"""Validate that every configured packed source yields a valid 16K batch."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

from wm_fsdp.data.megatron_packed import (
    MegatronPackedCollator,
    MegatronPackedDataset,
    build_prefix_inventory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight a wm-fsdp megatron_packed data config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.samples_per_dataset <= 0:
        raise ValueError("samples-per-dataset must be positive")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("type") != "megatron_packed":
        raise ValueError("validate_megatron_packed.py requires type=megatron_packed")
    collator = MegatronPackedCollator(config, max_length=int(config["sequence_length"]))
    checked = 0
    target_tokens = 0
    image_tokens = 0
    source_rows = list(config["datasets"])
    prefix_count = 0
    record_count = 0
    for row in source_rows:
        expected_inventory = row.get("prefixes")
        if not isinstance(expected_inventory, list) or not expected_inventory:
            raise ValueError(
                f"Packed source {row.get('name')!r} has no embedded prefix inventory; "
                "regenerate the config before formal training"
            )
        actual_inventory = build_prefix_inventory(
            str(row["path"]), sequence_length=int(config["sequence_length"])
        )
        if actual_inventory != expected_inventory:
            raise ValueError(
                f"Packed inventory changed for source {row.get('name')!r}; "
                "regenerate and review the data config"
            )
        prefix_count += len(actual_inventory)
        record_count += sum(int(item["sequence_count"]) for item in actual_inventory)
        one_source = copy.deepcopy(config)
        one_source["datasets"] = [copy.deepcopy(row)]
        one_source["samples_per_epoch"] = args.samples_per_dataset
        dataset = MegatronPackedDataset(one_source, rank=0, world_size=1, batch_size_per_rank=1)
        iterator = iter(dataset)
        for _ in range(args.samples_per_dataset):
            batch = collator([next(iterator)])
            labels = batch["labels"][:, 1:]
            target_tokens += int(labels.ne(-100).sum().item())
            image_tokens += int(labels.ge(151854).sum().item())
            checked += 1

    output = {
        "status": "passed",
        "config": str(Path(args.config).resolve()),
        "sources": len(source_rows),
        "prefixes": prefix_count,
        "records": record_count,
        "samples": checked,
        "mean_target_tokens": target_tokens / checked,
        "mean_image_tokens": image_tokens / checked,
    }
    print(
        "Megatron packed preflight PASS: "
        f"sources={output['sources']} prefixes={prefix_count} records={record_count} "
        f"samples={checked} mean_target_tokens={output['mean_target_tokens']:.1f} "
        f"mean_image_tokens={output['mean_image_tokens']:.1f}"
    )
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{report.name}.tmp-", dir=report.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, report)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise


if __name__ == "__main__":
    main()

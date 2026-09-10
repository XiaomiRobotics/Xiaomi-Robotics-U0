#!/usr/bin/env python3
"""Convert a weighted train_data_path list into wm-fsdp JSON metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from wm_fsdp.data.megatron_packed import build_prefix_inventory


def _extract_train_data_pairs(path: Path) -> list[tuple[float, str]]:
    text = path.read_text(encoding="utf-8")
    marker = "  train_data_path: ["
    if marker not in text or "\n  tokenizer:" not in text:
        raise ValueError(f"Could not locate data.train_data_path in {path}")
    block = text.split(marker, 1)[1].split("\n  tokenizer:", 1)[0]
    pairs: list[tuple[float, str]] = []
    pattern = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)?),\s*["\']?(.+?)["\']?,?\s*$')
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "]":
            continue
        match = pattern.match(line)
        if match is None:
            raise ValueError(f"Unsupported train_data_path line: {raw_line}")
        weight = float(match.group(1))
        dataset_path = match.group(2)
        if weight <= 0.0 or not dataset_path:
            raise ValueError(f"Invalid train_data_path entry: {raw_line}")
        pairs.append((weight, dataset_path))
    if not pairs:
        raise ValueError(f"No train_data_path entries found in {path}")
    return pairs


def _dataset_name(dataset_path: str) -> str:
    path = Path(dataset_path)
    # The initial-scene sources end in .../<dataset>/ibq_token/<metas>.
    # The depth sources need two ancestor names because names such as
    # ``v01`` and ``qwen_desc_refine`` recur across different corpora.
    parent = path.parent
    if parent.name == "ibq_token":
        return parent.parent.name
    if parent.name == "depth_train":
        return "__".join(path.parts[-4:-2])
    return parent.name


def build_config(
    source_yaml: Path,
    *,
    seed: int,
    sequence_length: int,
    max_open_files: int,
    prefix_block_size: int,
) -> dict[str, Any]:
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than one")
    if max_open_files <= 0:
        raise ValueError("max_open_files must be positive")
    if prefix_block_size <= 0:
        raise ValueError("prefix_block_size must be positive")
    pairs = _extract_train_data_pairs(source_yaml)
    grouped: OrderedDict[str, float] = OrderedDict()
    for weight, dataset_path in pairs:
        grouped[dataset_path] = grouped.get(dataset_path, 0.0) + weight

    names: set[str] = set()
    datasets: list[dict[str, Any]] = []
    for dataset_path, weight in grouped.items():
        name = _dataset_name(dataset_path)
        if not name:
            raise ValueError(
                "Cannot derive a dataset name from train_data_path: "
                f"{dataset_path!r} -> {name!r}"
            )
        if name in names:
            name = f"{name}__{hashlib.sha256(dataset_path.encode('utf-8')).hexdigest()[:8]}"
        if name in names:
            raise ValueError(f"Derived duplicate dataset name for {dataset_path!r}: {name!r}")
        names.add(name)
        datasets.append(
            {
                "name": name,
                "path": dataset_path,
                "weight": weight,
                "prefixes": build_prefix_inventory(
                    dataset_path, sequence_length=sequence_length
                ),
            }
        )
    return {
        "type": "megatron_packed",
        "format": "multimodal_sft_v1",
        "seed": int(seed),
        "weighted_sampling": True,
        "sampling_policy": "weighted_with_replacement",
        "sequence_length": int(sequence_length),
        "pad_token_id": 151643,
        "cls_token_id": 151849,
        "eod_token_id": 151850,
        "multimodal_visual_start_end_tokens": [151851, 151852, 151853],
        "supervision_tokens": [151746, 151747],
        "vocab_size": 282926,
        "max_open_files": int(max_open_files),
        "prefix_block_size": int(prefix_block_size),
        "source": {
            "format": "weighted_train_data_path",
            "yaml_path": str(source_yaml.resolve()),
            "weight_rows": len(pairs),
            "unique_dataset_paths": len(datasets),
            "weight_sum": sum(weight for weight, _ in pairs),
            "prefix_count": sum(len(row["prefixes"]) for row in datasets),
            "record_count": sum(
                int(prefix["sequence_count"])
                for row in datasets
                for prefix in row["prefixes"]
            ),
        },
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a wm-fsdp megatron_packed data config from a training YAML"
    )
    parser.add_argument("--source-yaml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=16384)
    parser.add_argument("--max-open-files", type=int, default=16)
    parser.add_argument("--prefix-block-size", type=int, default=256)
    args = parser.parse_args()

    config = build_config(
        Path(args.source_yaml),
        seed=args.seed,
        sequence_length=args.sequence_length,
        max_open_files=args.max_open_files,
        prefix_block_size=args.prefix_block_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "Wrote "
        f"{output}: source_rows={config['source']['weight_rows']} "
        f"unique_datasets={config['source']['unique_dataset_paths']} "
        f"weight_sum={config['source']['weight_sum']:.8f}"
    )


if __name__ == "__main__":
    main()

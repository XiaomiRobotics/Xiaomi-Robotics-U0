from __future__ import annotations

import glob
import hashlib
import random
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


MEGATRON_PACKED_DIAGNOSTIC_KEYS = (
    "yielded_samples",
    "opened_index_files",
    "bad_sequences",
    "read_errors",
)
MEGATRON_PACKED_SAMPLER_VERSION = "stateless_weighted_v2"

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_DTYPES: dict[int, type[np.number]] = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


@dataclass(frozen=True)
class MegatronPackedDatasetSpec:
    """One weighted source of Megatron indexed, fixed-window records."""

    name: str
    path: str
    weight: float
    sample_count: int
    prefixes: tuple[str, ...]
    prefix_sample_counts: tuple[int, ...]
    prefix_idx_sizes: tuple[int, ...]
    prefix_bin_sizes: tuple[int, ...]


class _IndexedSequenceFile:
    """Minimal, dependency-free reader for Megatron's MMIDIDX .idx/.bin pair."""

    def __init__(
        self,
        prefix: str,
        *,
        expected_sequence_length: int | None = None,
        expected_sequence_count: int | None = None,
        expected_idx_size: int | None = None,
        expected_bin_size: int | None = None,
    ) -> None:
        self.prefix = str(prefix)
        idx_path = Path(f"{self.prefix}.idx")
        bin_path = Path(f"{self.prefix}.bin")
        if not idx_path.is_file() or not bin_path.is_file():
            raise FileNotFoundError(f"Missing Megatron indexed pair for prefix: {self.prefix}")
        if expected_idx_size is not None and idx_path.stat().st_size != expected_idx_size:
            raise ValueError(
                f"Megatron index size changed: {idx_path} "
                f"expected={expected_idx_size} actual={idx_path.stat().st_size}"
            )
        if expected_bin_size is not None and bin_path.stat().st_size != expected_bin_size:
            raise ValueError(
                f"Megatron bin size changed: {bin_path} "
                f"expected={expected_bin_size} actual={bin_path.stat().st_size}"
            )

        with idx_path.open("rb") as stream:
            if stream.read(len(_INDEX_HEADER)) != _INDEX_HEADER:
                raise ValueError(f"Unsupported Megatron index header: {idx_path}")
            version = struct.unpack("<Q", stream.read(8))[0]
            if version != 1:
                raise ValueError(f"Unsupported Megatron index version={version}: {idx_path}")
            dtype_code = struct.unpack("<B", stream.read(1))[0]
            try:
                self.dtype = np.dtype(_DTYPES[dtype_code])
            except KeyError as exc:
                raise ValueError(f"Unsupported Megatron index dtype code={dtype_code}: {idx_path}") from exc
            sequence_count = struct.unpack("<Q", stream.read(8))[0]
            document_count = struct.unpack("<Q", stream.read(8))[0]
            if sequence_count <= 0 or document_count <= 0:
                raise ValueError(f"Empty Megatron indexed dataset: {idx_path}")
            if expected_sequence_count is not None and sequence_count != expected_sequence_count:
                raise ValueError(
                    f"Megatron sequence count changed: {idx_path} "
                    f"expected={expected_sequence_count} actual={sequence_count}"
                )
            self.sequence_lengths = np.fromfile(stream, dtype=np.int32, count=sequence_count)
            self.sequence_pointers = np.fromfile(stream, dtype=np.int64, count=sequence_count)
            document_indices = np.fromfile(stream, dtype=np.int64, count=document_count)

        if len(self.sequence_lengths) != sequence_count or len(self.sequence_pointers) != sequence_count:
            raise ValueError(f"Truncated Megatron index metadata: {idx_path}")
        if len(document_indices) != document_count or int(document_indices[-1]) != sequence_count:
            raise ValueError(f"Invalid Megatron document indices: {idx_path}")
        if np.any(self.sequence_lengths <= 0) or np.any(self.sequence_pointers < 0):
            raise ValueError(f"Invalid Megatron sequence lengths or pointers: {idx_path}")
        if expected_sequence_length is not None and np.any(
            self.sequence_lengths != expected_sequence_length
        ):
            invalid = np.flatnonzero(self.sequence_lengths != expected_sequence_length)
            first = int(invalid[0])
            raise ValueError(
                f"Megatron sequence length mismatch: {idx_path} index={first} "
                f"expected={expected_sequence_length} actual={int(self.sequence_lengths[first])}"
            )
        if np.any(self.sequence_pointers % self.dtype.itemsize):
            raise ValueError(f"Unaligned Megatron sequence pointer: {idx_path}")

        self.tokens = np.memmap(bin_path, mode="r", dtype=self.dtype)
        sequence_ends = (
            self.sequence_pointers // self.dtype.itemsize
            + self.sequence_lengths.astype(np.int64, copy=False)
        )
        if np.any(sequence_ends > int(self.tokens.shape[0])):
            first = int(np.flatnonzero(sequence_ends > int(self.tokens.shape[0]))[0])
            raise ValueError(
                f"Megatron sequence exceeds .bin size: prefix={self.prefix} index={first}"
            )

    def __len__(self) -> int:
        return int(self.sequence_lengths.shape[0])

    def get(self, index: int) -> np.ndarray:
        if not 0 <= int(index) < len(self):
            raise IndexError(f"Indexed sequence out of range: {index} not in [0, {len(self)})")
        pointer = int(self.sequence_pointers[int(index)])
        length = int(self.sequence_lengths[int(index)])
        start = pointer // self.dtype.itemsize
        end = start + length
        if end > int(self.tokens.shape[0]):
            raise ValueError(f"Indexed sequence exceeds .bin size: prefix={self.prefix} index={index}")
        return np.array(self.tokens[start:end], dtype=np.int64, copy=True)


def _read_index_sequence_count(prefix: str) -> int:
    idx_path = Path(f"{prefix}.idx")
    with idx_path.open("rb") as stream:
        if stream.read(len(_INDEX_HEADER)) != _INDEX_HEADER:
            raise ValueError(f"Unsupported Megatron index header: {idx_path}")
        version = struct.unpack("<Q", stream.read(8))[0]
        if version != 1:
            raise ValueError(f"Unsupported Megatron index version={version}: {idx_path}")
        dtype_code = struct.unpack("<B", stream.read(1))[0]
        if dtype_code not in _DTYPES:
            raise ValueError(f"Unsupported Megatron index dtype code={dtype_code}: {idx_path}")
        sequence_count = struct.unpack("<Q", stream.read(8))[0]
    if sequence_count <= 0:
        raise ValueError(f"Megatron indexed dataset contains no sequences: {idx_path}")
    return int(sequence_count)


def _resolve_prefixes(path: str) -> tuple[str, ...]:
    raw = str(path).strip()
    if not raw:
        raise ValueError("Megatron packed dataset path cannot be empty")
    candidate = Path(raw)
    if candidate.is_dir():
        idx_prefixes = {str(item.with_suffix("")) for item in candidate.glob("*.idx")}
        bin_prefixes = {str(item.with_suffix("")) for item in candidate.glob("*.bin")}
        if idx_prefixes != bin_prefixes:
            missing_bin = sorted(idx_prefixes - bin_prefixes)
            missing_idx = sorted(bin_prefixes - idx_prefixes)
            raise ValueError(
                f"Incomplete Megatron indexed pairs in {candidate}: "
                f"missing_bin={missing_bin[:5]} missing_idx={missing_idx[:5]}"
            )
        prefixes = tuple(sorted(idx_prefixes))
    elif raw.endswith(".idx"):
        prefixes = (raw[:-4],)
    elif Path(f"{raw}.idx").is_file():
        prefixes = (raw,)
    else:
        prefixes = tuple(str(Path(item).with_suffix("")) for item in sorted(glob.glob(raw)))
    missing = [
        prefix
        for prefix in prefixes
        if not Path(f"{prefix}.idx").is_file() or not Path(f"{prefix}.bin").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete Megatron indexed pairs: {missing[:5]}")
    if not prefixes:
        raise FileNotFoundError(f"No Megatron .idx/.bin pairs found at {path!r}")
    return prefixes


def build_prefix_inventory(path: str, *, sequence_length: int) -> list[dict[str, Any]]:
    """Resolve and fully validate immutable index metadata for one source."""
    inventory: list[dict[str, Any]] = []
    for prefix in _resolve_prefixes(path):
        reader = _IndexedSequenceFile(prefix, expected_sequence_length=sequence_length)
        idx_path = Path(f"{prefix}.idx")
        bin_path = Path(f"{prefix}.bin")
        inventory.append(
            {
                "path": prefix,
                "sequence_count": len(reader),
                "idx_size": idx_path.stat().st_size,
                "bin_size": bin_path.stat().st_size,
                "idx_sha256": hashlib.sha256(idx_path.read_bytes()).hexdigest(),
            }
        )
    return inventory


def _trim_outer_visual_context(
    values: list[int],
    *,
    pad_token_id: int,
    eod_token_id: int,
    cls_token_id: int,
    visual_start_token_id: int,
    visual_end_token_id: int,
) -> list[int]:
    """Match the source dataset's fixed-window visual-boundary normalization."""
    if not values:
        raise ValueError("Megatron packed record is empty")
    original_length = len(values)
    if values.count(visual_start_token_id) <= 1:
        result = list(values)
        if result[-1] == cls_token_id:
            result[-1] = pad_token_id
        return result

    result = list(values)
    try:
        first_start = result.index(visual_start_token_id)
        first_end = result.index(visual_end_token_id)
    except ValueError as exc:
        raise ValueError("Malformed visual boundary tokens in packed record") from exc
    if first_start > first_end:
        result = result[first_end + 1 :]

    reversed_result = result[::-1]
    try:
        last_start = reversed_result.index(visual_start_token_id)
        last_end = reversed_result.index(visual_end_token_id)
    except ValueError as exc:
        raise ValueError("Malformed visual boundary tokens in packed record") from exc
    if last_start < last_end:
        result = reversed_result[last_start + 1 :][::-1]
    if result and result[0] == eod_token_id:
        result = result[1:]
    if len(result) > original_length:
        raise ValueError("Packed visual-context normalization unexpectedly increased sequence length")
    return result + [pad_token_id] * (original_length - len(result))


class MegatronPackedCollator:
    """Compile fixed packed windows into the wm-fsdp causal-LM batch contract."""

    def __init__(self, config: dict[str, Any], *, max_length: int) -> None:
        if config.get("type") != "megatron_packed":
            raise ValueError("MegatronPackedCollator requires type=megatron_packed")
        if config.get("format") != "multimodal_sft_v1":
            raise ValueError("Only format=multimodal_sft_v1 is supported")
        self.sequence_length = int(config["sequence_length"])
        self.max_length = int(max_length)
        if self.sequence_length <= 1:
            raise ValueError("megatron_packed sequence_length must be greater than one")
        if self.max_length != self.sequence_length:
            raise ValueError(
                "data.max_length must equal megatron_packed sequence_length to preserve "
                f"the fixed packed window: max_length={self.max_length}, "
                f"sequence_length={self.sequence_length}"
            )
        self.pad_token_id = int(config["pad_token_id"])
        self.eod_token_id = int(config["eod_token_id"])
        self.cls_token_id = int(config["cls_token_id"])
        supervision = tuple(int(value) for value in config["supervision_tokens"])
        visual_tokens = tuple(int(value) for value in config["multimodal_visual_start_end_tokens"])
        if len(supervision) != 2 or len(visual_tokens) != 3:
            raise ValueError(
                "megatron_packed requires exactly two supervision_tokens and three "
                "multimodal_visual_start_end_tokens"
            )
        self.supervision_start_token_id, self.supervision_end_token_id = supervision
        _, self.visual_start_token_id, self.visual_end_token_id = visual_tokens
        self.vocab_size = int(config.get("vocab_size", 0))
        if self.vocab_size <= 0:
            raise ValueError("megatron_packed vocab_size must be positive")

    def _compile_labels(self, tokens: torch.Tensor) -> torch.Tensor:
        starts = tokens.eq(self.supervision_start_token_id)
        ends = tokens.eq(self.supervision_end_token_id)
        if int(starts.sum().item()) != int(ends.sum().item()):
            raise ValueError(
                "Packed record has mismatched supervision markers: "
                f"start={int(starts.sum().item())} end={int(ends.sum().item())}"
            )
        mask = starts.cumsum(dim=0).gt(ends.cumsum(dim=0))
        mask[tokens.eq(self.supervision_start_token_id)] = True
        mask[tokens.eq(self.supervision_end_token_id)] = True
        mask[tokens.eq(self.eod_token_id)] = True
        mask[tokens.eq(self.cls_token_id)] = True
        mask[tokens.eq(self.pad_token_id)] = False
        return tokens.masked_fill(~mask, -100)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty Megatron packed batch")
        input_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        attention_rows: list[torch.Tensor] = []
        names: list[str] = []
        sample_ids: list[str] = []
        for sample in samples:
            raw = sample.get("tokens")
            if not torch.is_tensor(raw) or raw.ndim != 1:
                raise ValueError("Megatron packed sample must provide one-dimensional tokens")
            if int(raw.numel()) != self.sequence_length:
                raise ValueError(
                    f"Expected fixed packed sequence length={self.sequence_length}, "
                    f"got {int(raw.numel())}"
                )
            values = [int(value) for value in raw.tolist()]
            values = _trim_outer_visual_context(
                values,
                pad_token_id=self.pad_token_id,
                eod_token_id=self.eod_token_id,
                cls_token_id=self.cls_token_id,
                visual_start_token_id=self.visual_start_token_id,
                visual_end_token_id=self.visual_end_token_id,
            )
            tokens = torch.tensor(values, dtype=torch.long)
            if int(tokens.min().item()) < 0 or int(tokens.max().item()) >= self.vocab_size:
                raise ValueError(
                    "Packed record contains an input token outside the configured vocabulary: "
                    f"min={int(tokens.min().item())} max={int(tokens.max().item())} "
                    f"vocab_size={self.vocab_size}"
                )
            input_rows.append(tokens)
            label_rows.append(self._compile_labels(tokens))
            attention_rows.append(tokens.ne(self.pad_token_id).long())
            names.append(str(sample.get("dataset_name", "<unknown>")))
            sample_ids.append(str(sample.get("sample_id", "<unknown>")))

        input_ids = torch.stack(input_rows)
        return {
            "input_ids": input_ids,
            "labels": torch.stack(label_rows),
            "attention_mask": torch.stack(attention_rows),
            "position_ids": torch.arange(self.sequence_length, dtype=torch.long).repeat(len(samples), 1),
            "dataset_names": names,
            "sample_ids": sample_ids,
        }


class MegatronPackedDataset(IterableDataset):
    """Distributed weighted sampling over Megatron packed records."""

    diagnostic_keys = MEGATRON_PACKED_DIAGNOSTIC_KEYS

    def __init__(
        self,
        config: dict[str, Any],
        *,
        rank: int = 0,
        world_size: int = 1,
        batch_size_per_rank: int = 1,
    ) -> None:
        super().__init__()
        if config.get("type") != "megatron_packed":
            raise ValueError("MegatronPackedDataset requires type=megatron_packed")
        if config.get("format") != "multimodal_sft_v1":
            raise ValueError("Only format=multimodal_sft_v1 is supported")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_size_per_rank = int(batch_size_per_rank)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"Invalid rank topology: rank={self.rank} world_size={self.world_size}")
        if self.batch_size_per_rank <= 0:
            raise ValueError("batch_size_per_rank must be positive")
        self.seed = int(config.get("seed", 42))
        if config.get("weighted_sampling") is not True:
            raise ValueError("megatron_packed requires weighted_sampling=true")
        if config.get("sampling_policy") != "weighted_with_replacement":
            raise ValueError(
                "megatron_packed requires sampling_policy=weighted_with_replacement"
            )
        self.sequence_length = int(config["sequence_length"])
        if self.sequence_length <= 1:
            raise ValueError("megatron_packed sequence_length must be greater than one")
        self.samples_per_epoch = config.get("samples_per_epoch")
        if self.samples_per_epoch is not None:
            self.samples_per_epoch = int(self.samples_per_epoch)
            if self.samples_per_epoch <= 0:
                raise ValueError("samples_per_epoch must be positive")
            global_batch_size = self.world_size * self.batch_size_per_rank
            if self.samples_per_epoch % global_batch_size:
                raise ValueError(
                    "samples_per_epoch must be divisible by world_size * batch_size_per_rank; "
                    f"got {self.samples_per_epoch} and {global_batch_size}"
                )
        self.max_open_files = int(config.get("max_open_files", 16))
        if self.max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        self.prefix_block_size = int(config.get("prefix_block_size", 1))
        if self.prefix_block_size <= 0:
            raise ValueError("prefix_block_size must be positive")
        self.epoch = 0
        self.start_batch = 0
        self.specs = self._parse_specs(config.get("datasets"))
        self._dataset_indices = {spec.name: index for index, spec in enumerate(self.specs)}
        self._diagnostic_indices = {
            name: index for index, name in enumerate(self.diagnostic_keys)
        }
        self._diagnostic_worker_slots = int(config.get("diagnostic_worker_slots", 128))
        if self._diagnostic_worker_slots <= 0:
            raise ValueError("diagnostic_worker_slots must be positive")
        self._diagnostics = torch.zeros(
            (len(self.specs), len(self.diagnostic_keys), self._diagnostic_worker_slots),
            dtype=torch.int64,
        ).share_memory_()
        self._readers: OrderedDict[str, _IndexedSequenceFile] = OrderedDict()
        self._weights = tuple(spec.weight for spec in self.specs)

    @staticmethod
    def _parse_specs(raw_specs: Any) -> tuple[MegatronPackedDatasetSpec, ...]:
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError("megatron_packed datasets must be a non-empty list")
        names: set[str] = set()
        specs: list[MegatronPackedDatasetSpec] = []
        for raw in raw_specs:
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid megatron_packed dataset entry: {raw!r}")
            name = str(raw.get("name", "")).strip()
            if not name or name in names:
                raise ValueError(f"Megatron packed dataset names must be unique and non-empty: {name!r}")
            path = str(raw.get("path", "")).strip()
            weight = float(raw.get("weight", 1.0))
            if weight <= 0.0:
                raise ValueError(f"Megatron packed dataset weight must be positive: {name}={weight}")
            raw_inventory = raw.get("prefixes")
            if raw_inventory is None:
                prefixes = _resolve_prefixes(path)
                prefix_counts = tuple(_read_index_sequence_count(prefix) for prefix in prefixes)
                prefix_idx_sizes = tuple(Path(f"{prefix}.idx").stat().st_size for prefix in prefixes)
                prefix_bin_sizes = tuple(Path(f"{prefix}.bin").stat().st_size for prefix in prefixes)
            else:
                if not isinstance(raw_inventory, list) or not raw_inventory:
                    raise ValueError(f"Megatron packed prefix inventory must be non-empty: {name}")
                prefixes = tuple(str(item["path"]) for item in raw_inventory)
                prefix_counts = tuple(int(item["sequence_count"]) for item in raw_inventory)
                prefix_idx_sizes = tuple(int(item["idx_size"]) for item in raw_inventory)
                prefix_bin_sizes = tuple(int(item["bin_size"]) for item in raw_inventory)
                if len(set(prefixes)) != len(prefixes):
                    raise ValueError(f"Megatron packed prefix inventory contains duplicates: {name}")
                if any(count <= 0 for count in prefix_counts):
                    raise ValueError(f"Megatron packed prefix inventory has empty prefix: {name}")
            specs.append(
                MegatronPackedDatasetSpec(
                    name=name,
                    path=path,
                    weight=weight,
                    sample_count=sum(prefix_counts),
                    prefixes=prefixes,
                    prefix_sample_counts=prefix_counts,
                    prefix_idx_sizes=prefix_idx_sizes,
                    prefix_bin_sizes=prefix_bin_sizes,
                )
            )
            names.add(name)
        return tuple(specs)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = OrderedDict()
        return state

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_batch(self, batch_index: int) -> None:
        self.start_batch = int(batch_index)
        if self.start_batch < 0:
            raise ValueError("Megatron packed start batch must be non-negative")

    def samples_for_worker(self, *, worker_id: int, num_workers: int) -> int | None:
        if not 0 <= worker_id < num_workers:
            raise ValueError(f"worker_id={worker_id} must be in [0, {num_workers})")
        if self.samples_per_epoch is None:
            return None
        batches_per_rank = self.samples_per_epoch // (
            self.world_size * self.batch_size_per_rank
        )
        remaining_batches = max(0, batches_per_rank - self.start_batch)
        base_batches, extra_batches = divmod(remaining_batches, num_workers)
        return (base_batches + int(worker_id < extra_batches)) * self.batch_size_per_rank

    def diagnostics_tensor(self) -> torch.Tensor:
        return self._diagnostics.sum(dim=2)

    def _increment(self, dataset_name: str, key: str, worker_id: int) -> None:
        self._diagnostics[
            self._dataset_indices[dataset_name],
            self._diagnostic_indices[key],
            worker_id,
        ] += 1

    def _reader(
        self,
        spec: MegatronPackedDatasetSpec,
        prefix_index: int,
        *,
        worker_id: int,
    ) -> _IndexedSequenceFile:
        prefix = spec.prefixes[prefix_index]
        reader = self._readers.pop(prefix, None)
        if reader is None:
            reader = _IndexedSequenceFile(
                prefix,
                expected_sequence_length=self.sequence_length,
                expected_sequence_count=spec.prefix_sample_counts[prefix_index],
                expected_idx_size=spec.prefix_idx_sizes[prefix_index],
                expected_bin_size=spec.prefix_bin_sizes[prefix_index],
            )
            self._increment(spec.name, "opened_index_files", worker_id)
        self._readers[prefix] = reader
        while len(self._readers) > self.max_open_files:
            self._readers.popitem(last=False)
        return reader

    @staticmethod
    def _source_prefix_index(spec: MegatronPackedDatasetSpec, rng: random.Random) -> int:
        return rng.choices(
            population=range(len(spec.prefixes)),
            weights=spec.prefix_sample_counts,
            k=1,
        )[0]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        if num_workers > self._diagnostic_worker_slots:
            raise ValueError(
                f"num_workers={num_workers} exceeds diagnostic_worker_slots={self._diagnostic_worker_slots}"
            )
        sample_budget = self.samples_for_worker(worker_id=worker_id, num_workers=num_workers)
        emitted = 0
        while sample_budget is None or emitted < sample_budget:
            worker_sample_index = emitted
            worker_batch_index, sample_offset = divmod(
                worker_sample_index, self.batch_size_per_rank
            )
            # A freshly constructed DataLoader always consumes worker 0 first.
            # Map those output slots onto the unconsumed batch suffix so resume
            # remains exact even when start_batch is not worker-aligned.
            batch_index = self.start_batch + worker_id + worker_batch_index * num_workers
            rank_sample_index = batch_index * self.batch_size_per_rank + sample_offset
            global_sample_index = (
                (batch_index * self.world_size + self.rank) * self.batch_size_per_rank
                + sample_offset
            )
            seed_payload = f"{self.seed}:{self.epoch}:{global_sample_index}".encode("ascii")
            sample_seed = int.from_bytes(
                hashlib.blake2b(seed_payload, digest_size=8).digest(), "little"
            )
            rng = random.Random(sample_seed)
            spec = self.specs[rng.choices(range(len(self.specs)), weights=self._weights, k=1)[0]]
            # Keep source selection independent per sample. Only prefix choice is
            # stable within a short rank-local block, reducing repeated .idx
            # opens without changing the configured source-weight expectation.
            prefix_block = rank_sample_index // self.prefix_block_size
            prefix_seed_payload = (
                f"{self.seed}:{self.epoch}:{self.rank}:{spec.name}:{prefix_block}"
            ).encode("utf-8")
            prefix_seed = int.from_bytes(
                hashlib.blake2b(prefix_seed_payload, digest_size=8).digest(), "little"
            )
            prefix_index = self._source_prefix_index(spec, random.Random(prefix_seed))
            prefix = spec.prefixes[prefix_index]
            sequence_index = rng.randrange(spec.prefix_sample_counts[prefix_index])
            try:
                raw = self._reader(spec, prefix_index, worker_id=worker_id).get(sequence_index)
                if raw.ndim != 1 or int(raw.shape[0]) != self.sequence_length:
                    self._increment(spec.name, "bad_sequences", worker_id)
                    raise ValueError(
                        "Megatron packed sequence must match the configured fixed length: "
                        f"got shape={raw.shape}, expected=({self.sequence_length},)"
                    )
            except Exception:
                self._increment(spec.name, "read_errors", worker_id)
                raise
            self._increment(spec.name, "yielded_samples", worker_id)
            emitted += 1
            yield {
                "tokens": torch.from_numpy(raw).long(),
                "dataset_name": spec.name,
                "sample_id": f"{spec.name}/{Path(prefix).name}:{sequence_index}",
            }


__all__ = [
    "build_prefix_inventory",
    "MEGATRON_PACKED_DIAGNOSTIC_KEYS",
    "MEGATRON_PACKED_SAMPLER_VERSION",
    "MegatronPackedCollator",
    "MegatronPackedDataset",
    "MegatronPackedDatasetSpec",
]

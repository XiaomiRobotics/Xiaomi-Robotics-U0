from __future__ import annotations

import glob
import io
import json
import os.path as osp
import posixpath
import random
import tarfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info


DIAGNOSTIC_KEYS = (
    "yielded_samples",
    "opened_shards",
    "bad_shards",
    "bad_manifests",
    "bad_simple_samples",
    "missing_members",
    "invalid_shapes",
    "invalid_dtypes",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    paths: tuple[str, ...]
    weight: float = 1.0
    sample_count: int | None = None
    visual_id_space: str = "raw"


@dataclass(frozen=True)
class ShardRef:
    dataset: DatasetSpec
    path: str


def _load_tensor(
    payload: bytes,
    filename: str,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    try:
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(io.BytesIO(payload), map_location="cpu")
    if not torch.is_tensor(value):
        raise ValueError(f"WebDataset member must contain a tensor: {filename}")
    if value.dtype == torch.bool or value.dtype.is_floating_point or value.dtype.is_complex:
        raise ValueError(f"Visual token grid must use an integer dtype, got {value.dtype} in {filename}")
    if value.ndim != 2:
        raise ValueError(f"Visual token grid must be [H,W], got {tuple(value.shape)} in {filename}")
    if expected_shape is not None and tuple(value.shape) != expected_shape:
        raise ValueError(
            f"Visual token grid shape {tuple(value.shape)} does not match manifest shape "
            f"{expected_shape} in {filename}"
        )
    return value.long().contiguous()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _entry_file(entry: dict[str, Any]) -> str:
    for key in ("file", "path", "pt", "token_file"):
        value = entry.get(key)
        if value:
            return str(value)
    raise ValueError(f"Image entry is missing a token file: {entry}")


def _entry_shape(entry: dict[str, Any]) -> tuple[int, int] | None:
    height = entry.get("height")
    width = entry.get("width")
    if height is None and width is None:
        return None
    if height is None or width is None:
        raise ValueError(f"Image entry must declare both height and width: {entry}")
    shape = (int(height), int(width))
    if min(shape) <= 0:
        raise ValueError(f"Image entry has a non-positive shape: {entry}")
    return shape


def _normalize_tar_path(path: str, *, context: str) -> str:
    """Return a safe, canonical POSIX path without discarding its hierarchy."""
    value = str(path).strip()
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"unsafe {context} path: {path!r}")
    raw_parts = value.split("/")
    if any(part == ".." for part in raw_parts):
        raise ValueError(f"unsafe {context} path traversal: {path!r}")
    parts = [part for part in raw_parts if part not in {"", "."}]
    if not parts:
        raise ValueError(f"unsafe empty {context} path: {path!r}")
    return "/".join(parts)


class _TarMemberIndex:
    """Index tar metadata and load only members needed by the current sample."""

    def __init__(self, archive: tarfile.TarFile, *, max_member_bytes: int) -> None:
        self.archive = archive
        self.max_member_bytes = max_member_bytes
        self.members: dict[str, tarfile.TarInfo] = {}
        for member in archive:
            if not member.isfile():
                continue
            name = _normalize_tar_path(member.name, context="tar member")
            if name in self.members:
                raise ValueError(f"duplicate canonical tar member path: {name}")
            self.members[name] = member

    def names(self) -> tuple[str, ...]:
        return tuple(self.members)

    def read_exact(self, name: str) -> bytes:
        canonical = _normalize_tar_path(name, context="tar member")
        member = self.members.get(canonical)
        if member is None:
            raise KeyError(f"missing WebDataset member {canonical}")
        if member.size > self.max_member_bytes:
            raise ValueError(
                f"WebDataset member {canonical} is {member.size} bytes, exceeding "
                f"max_member_bytes={self.max_member_bytes}"
            )
        extracted = self.archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"could not read WebDataset member {canonical}")
        return extracted.read()

    def resolve_reference(self, reference: str, *, manifest_name: str) -> str:
        relative = _normalize_tar_path(reference, context="manifest reference")
        manifest_dir = posixpath.dirname(manifest_name)
        sibling = posixpath.join(manifest_dir, relative) if manifest_dir else relative
        candidates = tuple(dict.fromkeys((sibling, relative)))
        matches = [candidate for candidate in candidates if candidate in self.members]
        if not matches:
            raise KeyError(
                f"missing WebDataset member {reference!r} referenced by {manifest_name}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous WebDataset member reference {reference!r} in {manifest_name}: "
                f"matches {matches}"
            )
        return matches[0]

    def read_reference(self, reference: str, *, manifest_name: str) -> tuple[str, bytes]:
        name = self.resolve_reference(reference, manifest_name=manifest_name)
        return name, self.read_exact(name)


class MultimodalWebDataset(IterableDataset):
    """Stream pretokenized multimodal tar shards for causal training."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        rank: int = 0,
        world_size: int = 1,
        batch_size_per_rank: int = 1,
    ):
        super().__init__()
        if config.get("type") != "webdataset":
            raise ValueError("MultimodalWebDataset requires type=webdataset")
        if config.get("format", "tokenized_tar") != "tokenized_tar":
            raise ValueError("Only format=tokenized_tar is supported")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_size_per_rank = int(batch_size_per_rank)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"Invalid rank topology: rank={self.rank} world_size={self.world_size}")
        if self.batch_size_per_rank <= 0:
            raise ValueError("batch_size_per_rank must be positive")
        self.seed = int(config.get("seed", 42))
        self.shuffle = bool(config.get("shuffle", True))
        self.weighted_sampling = bool(config.get("weighted_sampling", True))
        self.samples_per_epoch = config.get("samples_per_epoch")
        if self.samples_per_epoch is not None:
            self.samples_per_epoch = int(self.samples_per_epoch)
            if self.samples_per_epoch <= 0:
                raise ValueError("samples_per_epoch must be positive")
        self.shards_per_epoch = config.get("shards_per_epoch")
        if self.shards_per_epoch is not None:
            self.shards_per_epoch = int(self.shards_per_epoch)
            if self.shards_per_epoch <= 0:
                raise ValueError("shards_per_epoch must be positive")
        self.max_member_bytes = int(config.get("max_member_bytes", 256 * 1024 * 1024))
        if self.max_member_bytes <= 0:
            raise ValueError("max_member_bytes must be positive")
        self.skip_bad_samples = bool(config.get("skip_bad_samples", False))
        self.epoch = 0
        self.specs = self._parse_specs(config.get("datasets"))
        self.shards = self._resolve_shards(self.specs)
        self._dataset_indices = {spec.name: index for index, spec in enumerate(self.specs)}
        self._diagnostic_indices = {name: index for index, name in enumerate(DIAGNOSTIC_KEYS)}
        self._diagnostic_worker_slots = int(config.get("diagnostic_worker_slots", 128))
        if self._diagnostic_worker_slots <= 0:
            raise ValueError("diagnostic_worker_slots must be positive")
        self._diagnostics = torch.zeros(
            (len(self.specs), len(DIAGNOSTIC_KEYS), self._diagnostic_worker_slots),
            dtype=torch.int64,
        ).share_memory_()

        if self.samples_per_epoch is not None:
            global_batch_size = self.world_size * self.batch_size_per_rank
            if self.samples_per_epoch % global_batch_size:
                raise ValueError(
                    "samples_per_epoch must be divisible by the logical global batch size "
                    f"world_size * batch_size_per_rank = {self.world_size} * "
                    f"{self.batch_size_per_rank} = {global_batch_size}; got {self.samples_per_epoch}"
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        if num_workers > self._diagnostic_worker_slots:
            raise ValueError(
                f"num_workers={num_workers} exceeds diagnostic_worker_slots={self._diagnostic_worker_slots}"
            )
        sample_budget = self.samples_for_worker(worker_id=worker_id, num_workers=num_workers)
        if sample_budget == 0:
            return

        emitted = 0
        cycle = 0
        while True:
            shard_plan = self.shards_for_worker(
                worker_id=worker_id,
                num_workers=num_workers,
                cycle=cycle,
            )
            if not shard_plan:
                raise RuntimeError(
                    "WebDataset shard plan assigns no shards to "
                    f"rank={self.rank} worker={worker_id}; increase the shard plan size"
                )
            cycle_emitted = 0
            for shard in shard_plan:
                for sample in self._iter_shard(shard):
                    emitted += 1
                    cycle_emitted += 1
                    self._increment_diagnostic(shard.dataset.name, "yielded_samples", worker_id)
                    yield sample
                    if sample_budget is not None and emitted >= sample_budget:
                        return

            if sample_budget is None:
                return
            if cycle_emitted == 0:
                raise RuntimeError(
                    "WebDataset worker read no samples from an entire shard-plan cycle: "
                    f"rank={self.rank} worker={worker_id} cycle={cycle}"
                )
            cycle += 1

    def samples_for_worker(self, *, worker_id: int, num_workers: int) -> int | None:
        """Return an exact, full-batch-aligned sample budget for one loader worker."""
        if not 0 <= worker_id < num_workers:
            raise ValueError(f"worker_id={worker_id} must be in [0, {num_workers})")
        if self.samples_per_epoch is None:
            return None
        batches_per_rank = self.samples_per_epoch // (
            self.world_size * self.batch_size_per_rank
        )
        base_batches, extra_batches = divmod(batches_per_rank, num_workers)
        worker_batches = base_batches + int(worker_id < extra_batches)
        return worker_batches * self.batch_size_per_rank

    def diagnostics_snapshot(self) -> dict[str, dict[str, int]]:
        """Return cumulative reader counters, including DataLoader worker activity."""
        totals = self._diagnostics.sum(dim=2)
        return {
            spec.name: {
                key: int(totals[dataset_index, diagnostic_index].item())
                for diagnostic_index, key in enumerate(DIAGNOSTIC_KEYS)
            }
            for dataset_index, spec in enumerate(self.specs)
        }

    def diagnostics_tensor(self) -> torch.Tensor:
        """Return cumulative counters as [dataset, diagnostic] for distributed reduction."""
        return self._diagnostics.sum(dim=2)

    def sampling_plan_snapshot(self, *, cycle: int = 0) -> dict[str, dict[str, float | int]]:
        """Describe configured weights and the deterministic shard plan for one cycle."""
        shards_by_name: dict[str, list[ShardRef]] = {}
        for shard in self.shards:
            shards_by_name.setdefault(shard.dataset.name, []).append(shard)
        selected_counts = {spec.name: 0 for spec in self.specs}
        for shard in self._shard_plan(cycle=cycle):
            selected_counts[shard.dataset.name] += 1
        return {
            spec.name: {
                "configured_weight": spec.weight,
                "resolved_shards": len(shards_by_name[spec.name]),
                "sample_count": spec.sample_count or 0,
                "average_samples_per_shard": self._average_samples_per_shard(
                    spec, shards_by_name[spec.name]
                ),
                "effective_shard_weight": spec.weight
                / self._average_samples_per_shard(spec, shards_by_name[spec.name]),
                "selected_shards": selected_counts[spec.name],
            }
            for spec in self.specs
        }

    def _increment_diagnostic(self, dataset_name: str, key: str, worker_id: int) -> None:
        self._diagnostics[
            self._dataset_indices[dataset_name],
            self._diagnostic_indices[key],
            worker_id,
        ] += 1

    def _record_error(self, dataset_name: str, scope: str, exc: Exception, worker_id: int) -> None:
        self._increment_diagnostic(dataset_name, scope, worker_id)
        message = str(exc).lower()
        if isinstance(exc, KeyError) or "missing" in message:
            self._increment_diagnostic(dataset_name, "missing_members", worker_id)
        if "shape" in message or "[h,w]" in message:
            self._increment_diagnostic(dataset_name, "invalid_shapes", worker_id)
        if "dtype" in message or "integer" in message:
            self._increment_diagnostic(dataset_name, "invalid_dtypes", worker_id)

    def _shard_plan(self, *, cycle: int = 0) -> list[ShardRef]:
        if cycle < 0:
            raise ValueError("cycle must be non-negative")
        # Preserve the original first-cycle seed while giving repeats independent,
        # deterministic plans within the same logical epoch.
        cycle_seed = self.seed + self.epoch + cycle * 0x9E3779B97F4A7C15
        rng = random.Random(cycle_seed)
        if not self.weighted_sampling or len(self.specs) == 1:
            plan = list(self.shards)
            if self.shuffle:
                rng.shuffle(plan)
            return plan

        shards_by_name: dict[str, list[ShardRef]] = {}
        for shard in self.shards:
            shards_by_name.setdefault(shard.dataset.name, []).append(shard)
        active = [spec for spec in self.specs if spec.weight > 0 and shards_by_name.get(spec.name)]
        if not active:
            raise ValueError("No WebDataset source has a positive weight")
        weights = [spec.weight / self._average_samples_per_shard(spec, shards_by_name[spec.name]) for spec in active]
        count = self.shards_per_epoch or len(self.shards)
        selected = rng.choices(active, weights=weights, k=count)
        return [rng.choice(shards_by_name[spec.name]) for spec in selected]

    def shards_for_worker(
        self,
        *,
        worker_id: int = 0,
        num_workers: int = 1,
        cycle: int = 0,
    ) -> list[ShardRef]:
        """Return the deterministic shard slice assigned to one rank/worker."""
        if not 0 <= worker_id < num_workers:
            raise ValueError(f"worker_id={worker_id} must be in [0, {num_workers})")
        global_worker = self.rank * num_workers + worker_id
        global_workers = self.world_size * num_workers
        return self._shard_plan(cycle=cycle)[global_worker::global_workers]

    @staticmethod
    def _average_samples_per_shard(spec: DatasetSpec, shards: list[ShardRef]) -> float:
        if spec.sample_count is None:
            return 1.0
        return max(1.0, float(spec.sample_count) / len(shards))

    def _iter_shard(self, shard: ShardRef) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        try:
            self._increment_diagnostic(shard.dataset.name, "opened_shards", worker_id)
            with tarfile.open(shard.path, "r:*") as archive:
                members = _TarMemberIndex(archive, max_member_bytes=self.max_member_bytes)
                yield from self._manifest_samples(shard, members)
                yield from self._simple_paired_samples(shard, members)
        except Exception as exc:
            if not self.skip_bad_samples:
                raise
            self._record_error(shard.dataset.name, "bad_shards", exc, worker_id)
            warnings.warn(f"Skipping WebDataset shard {shard.path}: {exc}", RuntimeWarning, stacklevel=2)

    def _manifest_samples(
        self,
        shard: ShardRef,
        members: _TarMemberIndex,
    ) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        for filename in sorted(name for name in members.names() if name.lower().endswith(".json")):
            try:
                manifest = json.loads(members.read_exact(filename).decode("utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("manifest must be a JSON object")
                sample_id = str(manifest.get("sample_id") or osp.splitext(filename)[0])
                input_segments: list[dict[str, Any]] = []
                manifest_segments = _as_list(manifest.get("input_segments"))
                if any(not isinstance(segment, dict) for segment in manifest_segments):
                    raise ValueError("manifest input_segments entries must be objects")
                has_segment_text = any(
                    str(segment.get("type", "")).lower() == "text" and segment.get("text")
                    for segment in manifest_segments
                )
                prompt = self._manifest_prompt(manifest)
                if prompt and not has_segment_text:
                    input_segments.append({"type": "text", "text": prompt, "apply_template": True})
                for segment in manifest_segments:
                    segment_type = str(segment.get("type", "")).lower()
                    if segment_type == "text" and segment.get("text"):
                        input_segments.append({"type": "text", "text": str(segment["text"])})
                    elif segment_type == "image":
                        member_name, payload = members.read_reference(
                            _entry_file(segment), manifest_name=filename
                        )
                        input_segments.append(
                            {
                                "type": "image",
                                "tokens": _load_tensor(
                                    payload,
                                    member_name,
                                    expected_shape=_entry_shape(segment),
                                ),
                                "id_space": str(segment.get("id_space", shard.dataset.visual_id_space)),
                            }
                        )
                    else:
                        raise ValueError(f"unsupported manifest input segment: {segment}")
                outputs = self._manifest_outputs(manifest)
                if not outputs:
                    raise ValueError("manifest has no output image")
                target_segments: list[dict[str, Any]] = []
                for output_index, target_entry in enumerate(outputs):
                    target_name, payload = members.read_reference(
                        _entry_file(target_entry), manifest_name=filename
                    )
                    target_segments.append(
                        {
                            "type": "image",
                            "tokens": _load_tensor(
                                payload,
                                target_name,
                                expected_shape=_entry_shape(target_entry),
                            ),
                            "id_space": str(
                                target_entry.get("id_space", shard.dataset.visual_id_space)
                            ),
                            "output_index": output_index,
                            "member_name": target_name,
                        }
                    )
                first_target = target_segments[0]
                yield {
                    "sample_id": f"{shard.dataset.name}:{sample_id}",
                    "dataset_name": shard.dataset.name,
                    "input_segments": input_segments,
                    "target_segments": target_segments,
                    # Compatibility aliases for consumers which only inspect the first target.
                    # The collator always consumes target_segments when it is present.
                    "target_tokens": first_target["tokens"],
                    "target_id_space": first_target["id_space"],
                    "manifest_output_count": len(outputs),
                }
            except Exception as exc:
                if not self.skip_bad_samples:
                    raise ValueError(f"Invalid manifest {filename} in {shard.path}: {exc}") from exc
                self._record_error(shard.dataset.name, "bad_manifests", exc, worker_id)
                warnings.warn(
                    f"Skipping manifest {filename} in {shard.path}: {exc}", RuntimeWarning, stacklevel=2
                )

    def _simple_paired_samples(
        self,
        shard: ShardRef,
        members: _TarMemberIndex,
    ) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        text_by_stem = {
            osp.splitext(name)[0]: name
            for name in members.names()
            if name.lower().endswith(".txt")
        }
        tensor_by_stem = {
            osp.splitext(name)[0]: name
            for name in members.names()
            if name.lower().endswith(".pt") and ".in." not in name and ".out." not in name
        }
        for stem in sorted(text_by_stem.keys() | tensor_by_stem.keys()):
            try:
                if stem not in text_by_stem or stem not in tensor_by_stem:
                    missing = f"{stem}.txt" if stem not in text_by_stem else f"{stem}.pt"
                    raise KeyError(f"missing simple paired member {missing}")
                text_name = text_by_stem[stem]
                tensor_name = tensor_by_stem[stem]
                target_tokens = _load_tensor(members.read_exact(tensor_name), tensor_name)
                target_segment = {
                    "type": "image",
                    "tokens": target_tokens,
                    "id_space": shard.dataset.visual_id_space,
                    "output_index": 0,
                    "member_name": tensor_name,
                }
                yield {
                    "sample_id": f"{shard.dataset.name}:{stem}",
                    "dataset_name": shard.dataset.name,
                    "input_segments": [
                        {
                            "type": "text",
                            "text": members.read_exact(text_name).decode("utf-8", errors="replace"),
                            "apply_template": True,
                        }
                    ],
                    "target_segments": [target_segment],
                    "target_tokens": target_tokens,
                    "target_id_space": shard.dataset.visual_id_space,
                }
            except Exception as exc:
                if not self.skip_bad_samples:
                    raise ValueError(f"Invalid simple paired sample {stem} in {shard.path}: {exc}") from exc
                self._record_error(shard.dataset.name, "bad_simple_samples", exc, worker_id)
                warnings.warn(
                    f"Skipping simple paired sample {stem} in {shard.path}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    @staticmethod
    def _manifest_prompt(manifest: dict[str, Any]) -> str:
        for key in ("prompt", "input_prompt", "text", "question", "instruction"):
            if manifest.get(key):
                return str(manifest[key])
        return ""

    @staticmethod
    def _manifest_outputs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("output_sequence", "output_images", "outputs"):
            if key not in manifest or manifest[key] is None:
                continue
            outputs = _as_list(manifest[key])
            if any(not isinstance(item, dict) for item in outputs):
                raise ValueError(f"manifest {key} entries must be objects")
            if outputs:
                return outputs
        return []

    @staticmethod
    def _parse_specs(raw_specs: Any) -> tuple[DatasetSpec, ...]:
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError("WebDataset config requires a non-empty datasets list")
        specs: list[DatasetSpec] = []
        names: set[str] = set()
        for raw in raw_specs:
            if not isinstance(raw, dict):
                raise ValueError("Each WebDataset entry must be an object")
            name = str(raw.get("name", "")).strip()
            paths = raw.get("paths")
            if not name or name in names:
                raise ValueError(f"Dataset names must be non-empty and unique: {name!r}")
            if not isinstance(paths, list) or not paths:
                raise ValueError(f"Dataset {name} requires a non-empty paths list")
            weight = float(raw.get("weight", 1.0))
            if weight < 0:
                raise ValueError(f"Dataset {name} weight cannot be negative")
            sample_count = raw.get("sample_count")
            if sample_count is not None:
                sample_count = int(sample_count)
                if sample_count <= 0:
                    raise ValueError(f"Dataset {name} sample_count must be positive")
            visual_id_space = str(raw.get("visual_id_space", raw.get("id_space", "raw"))).lower()
            if visual_id_space not in {"raw", "hf"}:
                raise ValueError(f"Dataset {name} has invalid visual_id_space={visual_id_space!r}")
            specs.append(
                DatasetSpec(
                    name=name,
                    paths=tuple(str(path) for path in paths),
                    weight=weight,
                    sample_count=sample_count,
                    visual_id_space=visual_id_space,
                )
            )
            names.add(name)
        return tuple(specs)

    @staticmethod
    def _resolve_shards(specs: tuple[DatasetSpec, ...]) -> tuple[ShardRef, ...]:
        shards: list[ShardRef] = []
        for spec in specs:
            resolved: set[str] = set()
            for raw_path in spec.paths:
                path = Path(raw_path).expanduser()
                if path.is_file() and ".tar" in path.suffixes:
                    resolved.add(str(path.resolve()))
                elif path.is_dir():
                    resolved.update(str(item.resolve()) for item in path.rglob("*.tar"))
                    resolved.update(str(item.resolve()) for item in path.rglob("*.tar.gz"))
                else:
                    resolved.update(str(Path(item).resolve()) for item in glob.glob(raw_path, recursive=True))
            if not resolved:
                raise FileNotFoundError(f"Dataset {spec.name} has no tar shards: {list(spec.paths)}")
            shards.extend(ShardRef(spec, path) for path in sorted(resolved))
        return tuple(shards)


__all__ = ["DIAGNOSTIC_KEYS", "MultimodalWebDataset"]

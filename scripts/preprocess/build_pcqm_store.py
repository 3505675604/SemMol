"""Build the final four-modality PCQM LMDB store and nested subset views.

Candidates are consumed in ``(gap_bin, selection_rank)`` order.  Feature
failures are recorded and the next ranked molecule in the same gap bin is
used, so every requested target remains exact and smaller targets are strict
subsets of the largest target.  Published output is staged in a sibling
directory and exposed only after every shard, view, and checksum is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import pyarrow.parquet as parquet

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.datasets.feature_building import (  # noqa: E402
    FeatureBuildConfig,
    FeatureBuildError,
    MultimodalFeatureBuilder,
    tokenizer_artifact_sha256,
)
from src.datasets.storage import (  # noqa: E402
    LmdbShardWriter,
    RecordCodec,
    StoreMetadata,
    recover_published_shard_sidecar,
    write_store_metadata,
)
from src.molecular.espf_tokenizer import (  # noqa: E402
    ARTIFACT_MANIFEST_NAME,
    ESPFTokenizer,
    SCHEMA_VERSION as TOKENIZER_SCHEMA_VERSION,
)
from src.molecular.geometry import GeometryRecord  # noqa: E402
from src.utils.io import (  # noqa: E402
    atomic_output_path,
    atomic_write_json,
    sha256_file,
)


BUILD_SCHEMA = "semmol.pcqm_store_build.v1"
BUILD_INDEX_SCHEMA = "semmol.pcqm_store_shard_index.v1"
VIEW_SCHEMA = "semmol.pcqm_view.v1"
REQUIRED_SELECTION_COLUMNS = (
    "source_index",
    "canonical_smiles",
    "gap",
    "gap_bin",
    "selection_rank",
)
GEOMETRY_SCHEMA = "semmol.geometry.v1"
GEOMETRY_RUN_STATE_SCHEMA = "semmol.geometry_run_state.v1"
GEOMETRY_FINGERPRINT_INPUT_FIELDS = frozenset(
    {"input", "manifest", "official_sdf"}
)
GEOMETRY_FINGERPRINT_PARAMETER_FIELDS = frozenset(
    {
        "schema",
        "smiles_col",
        "source_index_col",
        "sdf_ordinal_col",
        "sdf_energy_property",
        "num_conformers",
        "prune_rms_thresh",
        "seed",
        "optimize",
        "num_workers",
        "worker_chunksize",
        "shard_size",
        "table_chunk_size",
        "verify_checksums",
    }
)


def _fsync_directory(path: Path) -> None:
    """Durably persist a directory entry on the Linux training server."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_tree(root: Path) -> None:
    """Persist every regular file and directory entry below a private tree."""

    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"cannot fsync redirected directory tree: {root}")
    directories = [root]
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            children = list(entries)
        for entry in children:
            candidate = Path(entry.path)
            is_junction = getattr(candidate, "is_junction", None)
            if entry.is_symlink() or (
                callable(is_junction) and is_junction()
            ):
                raise RuntimeError(
                    f"private snapshot contains a redirected entry: "
                    f"{candidate}"
                )
            if entry.is_dir(follow_symlinks=False):
                directories.append(candidate)
                pending.append(candidate)
            elif entry.is_file(follow_symlinks=False):
                with candidate.open("rb") as stream:
                    os.fsync(stream.fileno())
            else:
                raise RuntimeError(
                    f"private snapshot contains a non-regular entry: "
                    f"{candidate}"
                )
    for directory in reversed(directories):
        _fsync_directory(directory)


@dataclass(frozen=True)
class Candidate:
    source_index: int
    canonical_smiles: str
    gap: float
    gap_bin: int
    selection_rank: int


@dataclass(frozen=True)
class AcceptedCandidate:
    record_index: int
    source_index: int
    gap_bin: int
    selection_rank: int
    success_rank: int


@dataclass(frozen=True)
class GeometryShardDescriptor:
    shard_id: int
    artifact_name: str
    artifact_sha256: str
    record_count: int
    sidecar_path: Path


@dataclass(frozen=True)
class TokenizerSnapshot:
    root: Path
    load_path: Path
    artifact_sha256: str
    vocab_size: int


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _read_pinned_json(
    path: Path,
    *,
    label: str,
    expected_sha256: Optional[str] = None,
) -> tuple[dict[str, Any], str]:
    """Hash and decode the same bytes so pin validation has no reopen gap."""
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(f"{label} changed after contract pinning")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload, actual_sha256


def _validate_geometry_run_contract(
    manifest: Mapping[str, Any],
    run_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the self-hashed producer fingerprint and persisted run state."""
    if manifest.get("schema") != GEOMETRY_SCHEMA:
        raise RuntimeError("geometry manifest schema is invalid")
    if (
        set(run_state) != {"schema", "fingerprint"}
        or run_state.get("schema") != GEOMETRY_RUN_STATE_SCHEMA
    ):
        raise RuntimeError("geometry run-state contract is invalid")
    fingerprint = manifest.get("run_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise RuntimeError("geometry manifest has no run fingerprint")
    run_state_fingerprint = run_state.get("fingerprint")
    if not isinstance(run_state_fingerprint, Mapping):
        raise RuntimeError("geometry run state has no fingerprint")
    manifest_fingerprint_json = json.dumps(
        fingerprint,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    run_state_fingerprint_json = json.dumps(
        run_state_fingerprint,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if run_state_fingerprint_json != manifest_fingerprint_json:
        raise RuntimeError("geometry manifest/run-state fingerprint differs")
    if set(fingerprint) != {"sha256", "inputs", "parameters"}:
        raise RuntimeError("geometry run fingerprint fields are invalid")

    inputs = fingerprint.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or set(inputs) != GEOMETRY_FINGERPRINT_INPUT_FIELDS
    ):
        raise RuntimeError("geometry run fingerprint inputs are invalid")
    for name in sorted(GEOMETRY_FINGERPRINT_INPUT_FIELDS):
        descriptor = inputs[name]
        if descriptor is None and name != "input":
            continue
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"name", "size", "sha256"}
        ):
            raise RuntimeError(
                f"geometry fingerprint input descriptor is invalid: {name}"
            )
        filename = descriptor.get("name")
        size = descriptor.get("size")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not _is_sha256(descriptor.get("sha256"))
        ):
            raise RuntimeError(
                f"geometry fingerprint input descriptor is invalid: {name}"
            )

    parameters = fingerprint.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or set(parameters) != GEOMETRY_FINGERPRINT_PARAMETER_FIELDS
        or parameters.get("schema") != GEOMETRY_SCHEMA
    ):
        raise RuntimeError("geometry run fingerprint parameters are invalid")
    string_or_none_fields = (
        "source_index_col",
        "sdf_ordinal_col",
        "sdf_energy_property",
    )
    if (
        not isinstance(parameters.get("smiles_col"), str)
        or not parameters["smiles_col"]
        or any(
            parameters.get(field) is not None
            and (
                not isinstance(parameters[field], str)
                or not parameters[field]
            )
            for field in string_or_none_fields
        )
    ):
        raise RuntimeError("geometry fingerprint column parameters are invalid")
    positive_integer_fields = (
        "num_conformers",
        "num_workers",
        "worker_chunksize",
        "shard_size",
        "table_chunk_size",
    )
    if any(
        not isinstance(parameters.get(field), int)
        or isinstance(parameters[field], bool)
        or parameters[field] <= 0
        for field in positive_integer_fields
    ):
        raise RuntimeError("geometry fingerprint integer parameters are invalid")
    prune_rms_thresh = parameters.get("prune_rms_thresh")
    if (
        isinstance(prune_rms_thresh, bool)
        or not isinstance(prune_rms_thresh, (int, float))
        or not math.isfinite(float(prune_rms_thresh))
        or float(prune_rms_thresh) < 0
        or not isinstance(parameters.get("seed"), int)
        or isinstance(parameters["seed"], bool)
        or not isinstance(parameters.get("optimize"), bool)
        or not isinstance(parameters.get("verify_checksums"), bool)
    ):
        raise RuntimeError("geometry fingerprint generation parameters are invalid")

    payload = {
        "inputs": dict(inputs),
        "parameters": dict(parameters),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if (
        not _is_sha256(fingerprint.get("sha256"))
        or fingerprint["sha256"] != expected_sha256
    ):
        raise RuntimeError("geometry run fingerprint checksum is invalid")
    return dict(fingerprint)


def _verify_generation_directory(
    path: Path,
    expected_sha256: str,
) -> tuple[str, ...]:
    manifest_path = path / ARTIFACT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"tokenizer generation manifest is missing: {manifest_path}"
        )
    payload, _ = _read_pinned_json(
        manifest_path,
        label="tokenizer generation manifest",
        expected_sha256=expected_sha256,
    )
    if (
        set(payload) != {"schema_version", "artifacts"}
        or payload.get("schema_version") != TOKENIZER_SCHEMA_VERSION
        or isinstance(payload.get("schema_version"), bool)
    ):
        raise RuntimeError("tokenizer generation manifest schema is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RuntimeError("tokenizer generation has no artifact inventory")
    expected_names = set(artifacts) | {ARTIFACT_MANIFEST_NAME}
    actual_names = {entry.name for entry in path.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError("tokenizer generation file inventory mismatch")
    for name, descriptor in artifacts.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"size", "sha256"}
            or not isinstance(descriptor.get("size"), int)
            or isinstance(descriptor["size"], bool)
            or descriptor["size"] < 0
            or not _is_sha256(descriptor.get("sha256"))
        ):
            raise RuntimeError("invalid tokenizer artifact descriptor")
        artifact = path / name
        if (
            not artifact.is_file()
            or descriptor["size"] != artifact.stat().st_size
            or descriptor["sha256"] != sha256_file(artifact)
        ):
            raise RuntimeError(f"tokenizer artifact integrity failure: {artifact}")
    return tuple(sorted(expected_names))


def resolve_tokenizer_snapshot(
    tokenizer_dir: os.PathLike[str] | str,
) -> TokenizerSnapshot:
    root = Path(tokenizer_dir).resolve()
    tokenizer = ESPFTokenizer.from_pretrained(root)
    artifact_sha256 = tokenizer_artifact_sha256(root)
    if tokenizer.generation_id is not None:
        if tokenizer.generation_id != artifact_sha256:
            raise RuntimeError("active tokenizer generation hash is inconsistent")
        load_path = root / "generations" / artifact_sha256
        _verify_generation_directory(load_path, artifact_sha256)
    else:
        load_path = root
    return TokenizerSnapshot(
        root=root,
        load_path=load_path,
        artifact_sha256=artifact_sha256,
        vocab_size=int(tokenizer.vocab_size),
    )


def verify_tokenizer_snapshot(snapshot: TokenizerSnapshot) -> None:
    """Recheck the pinned tokenizer bytes immediately before publication."""

    generation_manifest = snapshot.load_path / ARTIFACT_MANIFEST_NAME
    if generation_manifest.is_file():
        _verify_generation_directory(
            snapshot.load_path,
            snapshot.artifact_sha256,
        )
        return
    if tokenizer_artifact_sha256(snapshot.load_path) != snapshot.artifact_sha256:
        raise RuntimeError(
            "legacy tokenizer changed during store construction"
        )


def _load_pinned_tokenizer(
    snapshot_path: os.PathLike[str] | str,
    *,
    expected_sha256: str,
    expected_vocab_size: int,
) -> ESPFTokenizer:
    path = Path(snapshot_path).resolve()
    generation_manifest = path / ARTIFACT_MANIFEST_NAME
    if generation_manifest.is_file():
        _verify_generation_directory(path, expected_sha256)
        tokenizer = ESPFTokenizer.from_pretrained(path)
        _verify_generation_directory(path, expected_sha256)
    else:
        before = tokenizer_artifact_sha256(path)
        if before != expected_sha256:
            raise RuntimeError("legacy tokenizer changed before worker initialization")
        tokenizer = ESPFTokenizer.from_pretrained(path)
        after = tokenizer_artifact_sha256(path)
        if after != expected_sha256:
            raise RuntimeError("legacy tokenizer changed during worker initialization")
    if int(tokenizer.vocab_size) != int(expected_vocab_size):
        raise RuntimeError(
            "tokenizer vocabulary size differs from the pinned build contract"
        )
    return tokenizer


def target_quotas(target_size: int, n_bins: int) -> tuple[int, ...]:
    if not isinstance(target_size, int) or isinstance(target_size, bool):
        raise TypeError("target_size must be an integer")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool):
        raise TypeError("n_bins must be an integer")
    if target_size <= 0 or n_bins <= 0:
        raise ValueError("target_size and n_bins must be positive")
    base, remainder = divmod(target_size, n_bins)
    return tuple(
        base + (1 if gap_bin < remainder else 0)
        for gap_bin in range(n_bins)
    )


def validate_resume_indices(
    accepted: Sequence[AcceptedCandidate],
) -> dict[str, np.ndarray]:
    ordered = sorted(accepted, key=lambda item: item.record_index)
    record_indices = np.asarray(
        [item.record_index for item in ordered],
        dtype=np.int64,
    )
    if not np.array_equal(
        record_indices,
        np.arange(len(ordered), dtype=np.int64),
    ):
        raise ValueError("resume record_index values must be contiguous from zero")
    source_indices = np.asarray(
        [item.source_index for item in ordered],
        dtype=np.int64,
    )
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("resume source_index values must be unique")
    return {
        "record_index": record_indices,
        "source_index": source_indices,
        "gap_bin": np.asarray(
            [item.gap_bin for item in ordered],
            dtype=np.int16,
        ),
        "selection_rank": np.asarray(
            [item.selection_rank for item in ordered],
            dtype=np.int64,
        ),
        "success_rank": np.asarray(
            [item.success_rank for item in ordered],
            dtype=np.int64,
        ),
    }


class SelectionAccumulator:
    """Track successful per-bin candidates and derive every nested view."""

    def __init__(self, target_sizes: Sequence[int], n_bins: int = 10) -> None:
        targets = tuple(sorted({int(size) for size in target_sizes}))
        if not targets or any(size <= 0 for size in targets):
            raise ValueError("target_sizes must contain positive integers")
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        self.target_sizes = targets
        self.n_bins = int(n_bins)
        self._quotas = {
            target: target_quotas(target, self.n_bins)
            for target in targets
        }
        self._record_indices_by_bin = [
            array("q") for _ in range(self.n_bins)
        ]
        self._source_indices_by_bin = [
            array("q") for _ in range(self.n_bins)
        ]
        self._selection_ranks_by_bin = [
            array("q") for _ in range(self.n_bins)
        ]
        self._source_indices: set[int] = set()
        self._accepted_count = 0

    @property
    def max_target(self) -> int:
        return self.target_sizes[-1]

    @property
    def accepted_count(self) -> int:
        return self._accepted_count

    def bin_is_full(self, gap_bin: int) -> bool:
        self._validate_gap_bin(gap_bin)
        return (
            len(self._record_indices_by_bin[gap_bin])
            >= self._quotas[self.max_target][gap_bin]
        )

    def accepted_in_bin(self, gap_bin: int) -> int:
        self._validate_gap_bin(gap_bin)
        return len(self._record_indices_by_bin[gap_bin])

    def _validate_gap_bin(self, gap_bin: int) -> None:
        if not 0 <= int(gap_bin) < self.n_bins:
            raise ValueError(
                f"gap_bin={gap_bin} outside [0, {self.n_bins})"
            )

    def accept(
        self,
        *,
        source_index: int,
        gap_bin: int,
        selection_rank: int,
        record_index: int,
    ) -> AcceptedCandidate:
        self._validate_gap_bin(gap_bin)
        if self.bin_is_full(gap_bin):
            raise ValueError(f"gap_bin={gap_bin} already has its maximum quota")
        if record_index != self.accepted_count:
            raise ValueError(
                f"record_index must be contiguous: expected "
                f"{self.accepted_count}, got {record_index}"
            )
        if source_index in self._source_indices:
            raise ValueError(f"duplicate accepted source_index={source_index}")
        selection_ranks = self._selection_ranks_by_bin[gap_bin]
        if selection_ranks and selection_rank <= selection_ranks[-1]:
            raise ValueError(
                "selection_rank must increase strictly within each gap bin"
            )
        success_rank = len(selection_ranks)
        accepted = AcceptedCandidate(
            record_index=int(record_index),
            source_index=int(source_index),
            gap_bin=int(gap_bin),
            selection_rank=int(selection_rank),
            success_rank=success_rank,
        )
        self._record_indices_by_bin[gap_bin].append(accepted.record_index)
        self._source_indices_by_bin[gap_bin].append(accepted.source_index)
        selection_ranks.append(accepted.selection_rank)
        self._source_indices.add(accepted.source_index)
        self._accepted_count += 1
        return accepted

    def restore(self, accepted: Sequence[AcceptedCandidate]) -> None:
        validate_resume_indices(accepted)
        for item in sorted(accepted, key=lambda row: row.record_index):
            restored = self.accept(
                source_index=item.source_index,
                gap_bin=item.gap_bin,
                selection_rank=item.selection_rank,
                record_index=item.record_index,
            )
            if restored.success_rank != item.success_rank:
                raise ValueError(
                    "resume success_rank is inconsistent with prior records"
                )

    def last_selection_rank(self, gap_bin: int) -> int:
        self._validate_gap_bin(gap_bin)
        ranks = self._selection_ranks_by_bin[gap_bin]
        return -1 if not ranks else int(ranks[-1])

    def validate_complete(self) -> None:
        for gap_bin, required in enumerate(
            self._quotas[self.max_target]
        ):
            actual = len(self._record_indices_by_bin[gap_bin])
            if actual != required:
                raise RuntimeError(
                    f"gap_bin={gap_bin} has {actual} successful records, "
                    f"requires {required}"
                )
        if self.accepted_count != self.max_target:
            raise RuntimeError(
                f"accepted_count={self.accepted_count} != {self.max_target}"
            )

    def all_accepted(self) -> list[AcceptedCandidate]:
        accepted = []
        for gap_bin in range(self.n_bins):
            records = self._record_indices_by_bin[gap_bin]
            sources = self._source_indices_by_bin[gap_bin]
            ranks = self._selection_ranks_by_bin[gap_bin]
            accepted.extend(
                AcceptedCandidate(
                    record_index=int(record_index),
                    source_index=int(source_index),
                    gap_bin=gap_bin,
                    selection_rank=int(selection_rank),
                    success_rank=success_rank,
                )
                for success_rank, (
                    record_index,
                    source_index,
                    selection_rank,
                ) in enumerate(zip(records, sources, ranks))
            )
        accepted.sort(key=lambda item: item.record_index)
        return accepted

    def views(self) -> dict[int, dict[str, np.ndarray]]:
        result: dict[int, dict[str, np.ndarray]] = {}
        previous_sources: set[int] = set()
        for target in self.target_sizes:
            record_parts = []
            source_parts = []
            bin_parts = []
            rank_parts = []
            success_parts = []
            for gap_bin in range(self.n_bins):
                count = self._quotas[target][gap_bin]
                record_parts.append(
                    np.frombuffer(
                        self._record_indices_by_bin[gap_bin],
                        dtype=np.int64,
                        count=count,
                    ).copy()
                )
                source_parts.append(
                    np.frombuffer(
                        self._source_indices_by_bin[gap_bin],
                        dtype=np.int64,
                        count=count,
                    ).copy()
                )
                rank_parts.append(
                    np.frombuffer(
                        self._selection_ranks_by_bin[gap_bin],
                        dtype=np.int64,
                        count=count,
                    ).copy()
                )
                bin_parts.append(
                    np.full(count, gap_bin, dtype=np.int16)
                )
                success_parts.append(np.arange(count, dtype=np.int64))
            record_indices = np.concatenate(record_parts)
            source_indices = np.concatenate(source_parts)
            gap_bins = np.concatenate(bin_parts)
            selection_ranks = np.concatenate(rank_parts)
            success_ranks = np.concatenate(success_parts)
            order = np.argsort(record_indices, kind="stable")
            arrays = {
                "record_index": record_indices[order],
                "source_index": source_indices[order],
                "gap_bin": gap_bins[order],
                "selection_rank": selection_ranks[order],
                "success_rank": success_ranks[order],
            }
            if len(arrays["record_index"]) != target:
                raise RuntimeError(
                    f"target={target} view has "
                    f"{len(arrays['record_index'])} records"
                )
            if len(np.unique(arrays["record_index"])) != target:
                raise RuntimeError(f"target={target} contains duplicate records")
            current_sources = set(arrays["source_index"].tolist())
            if not previous_sources.issubset(current_sources):
                raise RuntimeError("target views are not nested")
            previous_sources = current_sources
            result[target] = arrays
        return result


class GeometryRepository:
    """Compact random access over checksummed geometry NPZ shards.

    Only source/shard/record-ordinal arrays remain resident.  Row/SDF identity
    arrays are retained only for the coordinator audit, while shard record
    inventories are parsed lazily into a bounded LRU in workers.
    """

    def __init__(
        self,
        geometry_dir: os.PathLike[str] | str,
        *,
        verify_checksums: bool,
        validate_index_inventory: bool = True,
        expected_manifest_sha256: Optional[str] = None,
        expected_run_state_sha256: Optional[str] = None,
        expected_index_metadata_sha256: Optional[str] = None,
        expected_index_artifact_sha256: Optional[str] = None,
        max_open_shards: int = 8,
    ) -> None:
        if max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        self.directory = Path(geometry_dir).resolve()
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"geometry manifest is missing: {manifest_path}"
            )
        manifest, _ = _read_pinned_json(
            manifest_path,
            label="geometry manifest",
            expected_sha256=expected_manifest_sha256,
        )
        if manifest.get("schema") != "semmol.geometry.v1":
            raise RuntimeError(
                f"unsupported geometry manifest schema: "
                f"{manifest.get('schema')!r}"
            )
        run_state_path = self.directory / "run_state.json"
        if not run_state_path.is_file():
            raise FileNotFoundError(
                f"geometry run state is missing: {run_state_path}"
            )
        run_state, _ = _read_pinned_json(
            run_state_path,
            label="geometry run state",
            expected_sha256=expected_run_state_sha256,
        )
        _validate_geometry_run_contract(manifest, run_state)
        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise RuntimeError("geometry manifest has no shard inventory")
        self._descriptors: dict[int, GeometryShardDescriptor] = {}
        for raw_descriptor in raw_shards:
            if not isinstance(raw_descriptor, Mapping):
                raise RuntimeError("geometry shard descriptor must be an object")
            shard_id = raw_descriptor.get("shard_id")
            artifact_name = str(raw_descriptor.get("filename", ""))
            artifact_sha256 = str(raw_descriptor.get("sha256", ""))
            record_count = raw_descriptor.get("record_count")
            if (
                not isinstance(shard_id, int)
                or isinstance(shard_id, bool)
                or shard_id < 0
                or shard_id in self._descriptors
                or not artifact_name
                or Path(artifact_name).name != artifact_name
                or artifact_name != f"shard_{shard_id:06d}.npz"
                or len(artifact_sha256) != 64
                or set(artifact_sha256) - set("0123456789abcdef")
                or not isinstance(record_count, int)
                or isinstance(record_count, bool)
                or record_count <= 0
            ):
                raise RuntimeError("geometry manifest shard descriptor is invalid")
            artifact_path = self.directory / artifact_name
            sidecar_path = self.directory / f"shard_{shard_id:06d}.json"
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"geometry shard artifact is missing: {artifact_path}"
                )
            if not sidecar_path.is_file():
                raise FileNotFoundError(
                    f"geometry shard sidecar is missing: {sidecar_path}"
                )
            self._descriptors[shard_id] = GeometryShardDescriptor(
                shard_id=shard_id,
                artifact_name=artifact_name,
                artifact_sha256=artifact_sha256,
                record_count=record_count,
                sidecar_path=sidecar_path,
            )
        manifest_record_count = sum(
            descriptor.record_count
            for descriptor in self._descriptors.values()
        )
        if int(manifest.get("successful_records", -1)) != manifest_record_count:
            raise RuntimeError(
                "geometry manifest successful_records differs from its shards"
            )

        metadata_path = self.directory / "geometry_index.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"geometry index metadata missing: {metadata_path}")
        metadata, _ = _read_pinned_json(
            metadata_path,
            label="geometry index metadata",
            expected_sha256=expected_index_metadata_sha256,
        )
        if (
            set(metadata)
            != {
                "schema",
                "filename",
                "sha256",
                "record_count",
                "sorted_by",
                "lookup",
            }
            or metadata.get("schema") != "semmol.geometry_index.v2"
            or metadata.get("sorted_by") != ["source_index", "row_index"]
            or metadata.get("lookup")
            != "numpy.searchsorted(source_index, requested_source_index)"
        ):
            raise RuntimeError(
                f"unsupported geometry index schema: {metadata.get('schema')!r}"
            )
        index_name = str(metadata.get("filename", ""))
        index_artifact_sha256 = str(metadata.get("sha256", ""))
        index_record_count = metadata.get("record_count")
        if (
            not index_name
            or Path(index_name).name != index_name
            or not _is_sha256(index_artifact_sha256)
            or not isinstance(index_record_count, int)
            or isinstance(index_record_count, bool)
            or index_record_count < 0
            or (
                expected_index_artifact_sha256 is not None
                and index_artifact_sha256
                != expected_index_artifact_sha256
            )
        ):
            raise RuntimeError("geometry index filename is invalid")
        index_path = self.directory / index_name
        source_index_contract = manifest.get("source_index")
        if (
            not isinstance(source_index_contract, Mapping)
            or source_index_contract.get("artifact") != index_name
            or source_index_contract.get("metadata") != metadata_path.name
            or source_index_contract.get("sha256") != metadata.get("sha256")
            or source_index_contract.get("record_count")
            != metadata.get("record_count")
        ):
            raise RuntimeError(
                "geometry manifest and index metadata contracts differ"
            )
        if not index_path.is_file():
            raise FileNotFoundError(
                f"geometry index artifact is missing: {index_path}"
            )
        self.index_name = index_name
        if verify_checksums:
            actual = sha256_file(index_path)
            if actual != str(metadata.get("sha256", "")):
                raise RuntimeError(
                    f"geometry index checksum mismatch: {index_path}"
                )
        with np.load(index_path, allow_pickle=False) as archive:
            required_index_fields = {
                "source_index",
                "row_index",
                "sdf_ordinal",
                "shard_id",
                "record_ordinal",
            }
            if (
                len(archive.files) != len(required_index_fields)
                or set(archive.files) != required_index_fields
            ):
                raise RuntimeError(
                    "geometry index array inventory differs from v2 schema"
                )

            def load_integer_array(
                name: str,
                *,
                dtype: Any,
                minimum: int,
                maximum: int,
            ) -> np.ndarray:
                raw = np.asarray(archive[name])
                if (
                    raw.ndim != 1
                    or raw.size != index_record_count
                    or raw.dtype != np.dtype(dtype)
                ):
                    raise RuntimeError(
                        f"geometry index {name} has an invalid shape or dtype"
                    )
                if raw.size and (
                    int(raw.min()) < minimum
                    or int(raw.max()) > maximum
                ):
                    raise RuntimeError(
                        f"geometry index {name} values are out of range"
                    )
                return raw.astype(dtype, copy=True)

            def validate_integer_header(
                name: str,
                *,
                dtype: Any,
            ) -> None:
                with archive.zip.open(f"{name}.npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran_order, stored_dtype = (
                            np.lib.format.read_array_header_1_0(stream)
                        )
                    elif version == (2, 0):
                        shape, fortran_order, stored_dtype = (
                            np.lib.format.read_array_header_2_0(stream)
                        )
                    else:
                        raise RuntimeError(
                            f"geometry index {name} has an unsupported "
                            "NPY header"
                        )
                if (
                    shape != (index_record_count,)
                    or fortran_order
                    or stored_dtype != np.dtype(dtype)
                ):
                    raise RuntimeError(
                        f"geometry index {name} has an invalid shape or dtype"
                    )

            self.source_indices = load_integer_array(
                "source_index",
                dtype=np.int64,
                minimum=0,
                maximum=np.iinfo(np.int64).max,
            )
            row_indices: Optional[np.ndarray] = None
            sdf_ordinals: Optional[np.ndarray] = None
            if validate_index_inventory:
                row_indices = load_integer_array(
                    "row_index",
                    dtype=np.int64,
                    minimum=0,
                    maximum=np.iinfo(np.int64).max,
                )
                sdf_ordinals = load_integer_array(
                    "sdf_ordinal",
                    dtype=np.int64,
                    minimum=-1,
                    maximum=np.iinfo(np.int64).max,
                )
            else:
                validate_integer_header(
                    "row_index",
                    dtype=np.int64,
                )
                validate_integer_header(
                    "sdf_ordinal",
                    dtype=np.int64,
                )
            self.shard_ids = load_integer_array(
                "shard_id",
                dtype=np.int32,
                minimum=0,
                maximum=np.iinfo(np.int32).max,
            )
            self.record_ordinals = load_integer_array(
                "record_ordinal",
                dtype=np.int32,
                minimum=0,
                maximum=np.iinfo(np.int32).max,
            )
        if (
            self.source_indices.size != manifest_record_count
            or self.shard_ids.shape != self.source_indices.shape
            or self.record_ordinals.shape != self.source_indices.shape
            or (
                row_indices is not None
                and row_indices.shape != self.source_indices.shape
            )
            or (
                sdf_ordinals is not None
                and sdf_ordinals.shape != self.source_indices.shape
            )
        ):
            raise RuntimeError(
                "geometry index record count differs from the shard inventory"
            )
        if self.source_indices.size and np.any(
            self.source_indices[1:] <= self.source_indices[:-1]
        ):
            raise RuntimeError(
                "geometry source_index must be strictly increasing and unique"
            )
        if row_indices is not None and row_indices.size:
            sorted_rows = np.sort(row_indices)
            if np.any(sorted_rows[1:] == sorted_rows[:-1]):
                raise RuntimeError(
                    "geometry row_index must be unique"
                )
            del sorted_rows
        if sdf_ordinals is not None:
            nonnegative_ordinals = sdf_ordinals[sdf_ordinals >= 0]
            if nonnegative_ordinals.size:
                nonnegative_ordinals.sort()
                if np.any(
                    nonnegative_ordinals[1:] == nonnegative_ordinals[:-1]
                ):
                    raise RuntimeError(
                        "geometry sdf_ordinal must be unique when present"
                    )
            del nonnegative_ordinals
        indexed_shards, indexed_counts = np.unique(
            self.shard_ids,
            return_counts=True,
        )
        actual_counts = {
            int(shard_id): int(count)
            for shard_id, count in zip(indexed_shards, indexed_counts)
        }
        expected_counts = {
            shard_id: descriptor.record_count
            for shard_id, descriptor in self._descriptors.items()
        }
        if actual_counts != expected_counts:
            raise RuntimeError(
                "geometry index shard counts differ from the manifest"
            )

        self.verify_checksums = bool(verify_checksums)
        self.max_open_shards = int(max_open_shards)
        self._verified_shards: set[int] = set()
        self._open: OrderedDict[int, Any] = OrderedDict()
        self._records: OrderedDict[int, tuple[Mapping[str, Any], ...]] = (
            OrderedDict()
        )
        if validate_index_inventory:
            if row_indices is None or sdf_ordinals is None:
                raise RuntimeError(
                    "geometry index identity arrays were not retained"
                )
            self._validate_sidecar_locator_inventory(
                row_indices=row_indices,
                sdf_ordinals=sdf_ordinals,
            )

    def _load_sidecar_records(
        self,
        shard_id: int,
        *,
        cache: bool,
    ) -> tuple[Mapping[str, Any], ...]:
        cached = self._records.pop(shard_id, None)
        if cached is not None:
            self._records[shard_id] = cached
            return cached
        descriptor = self._descriptors[shard_id]
        sidecar = json.loads(
            descriptor.sidecar_path.read_text(encoding="utf-8")
        )
        records = sidecar.get("records")
        if (
            sidecar.get("schema") != "semmol.geometry.v1"
            or sidecar.get("shard_id") != descriptor.shard_id
            or sidecar.get("filename") != descriptor.artifact_name
            or sidecar.get("sha256") != descriptor.artifact_sha256
            or sidecar.get("record_count") != descriptor.record_count
            or not isinstance(records, list)
            or len(records) != descriptor.record_count
        ):
            raise RuntimeError(
                f"geometry sidecar differs from manifest: "
                f"{descriptor.sidecar_path}"
            )
        normalized: list[Mapping[str, Any]] = []
        for ordinal, entry in enumerate(records):
            source_index = (
                entry.get("source_index")
                if isinstance(entry, Mapping)
                else None
            )
            row_index = (
                entry.get("row_index")
                if isinstance(entry, Mapping)
                else None
            )
            sdf_ordinal = (
                entry.get("sdf_ordinal")
                if isinstance(entry, Mapping)
                else None
            )
            train_ordinal = (
                entry.get("train_ordinal")
                if isinstance(entry, Mapping)
                else None
            )
            if (
                not isinstance(entry, Mapping)
                or not {
                    "key",
                    "source_index",
                    "row_index",
                    "sdf_ordinal",
                    "train_ordinal",
                }.issubset(entry)
                or entry.get("key") != f"r{ordinal:06d}"
                or not isinstance(source_index, int)
                or isinstance(source_index, bool)
                or not 0 <= source_index <= np.iinfo(np.int64).max
                or not isinstance(row_index, int)
                or isinstance(row_index, bool)
                or not 0 <= row_index <= np.iinfo(np.int64).max
                or (
                    sdf_ordinal is not None
                    and (
                        not isinstance(sdf_ordinal, int)
                        or isinstance(sdf_ordinal, bool)
                        or not 0 <= sdf_ordinal <= np.iinfo(np.int64).max
                    )
                )
                or train_ordinal != sdf_ordinal
            ):
                raise RuntimeError(
                    f"geometry sidecar has invalid record ordinal={ordinal}: "
                    f"{descriptor.sidecar_path}"
                )
            normalized.append(entry)
        result = tuple(normalized)
        if cache:
            self._records[shard_id] = result
            while len(self._records) > self.max_open_shards:
                self._records.popitem(last=False)
        return result

    def _validate_sidecar_locator_inventory(
        self,
        *,
        row_indices: np.ndarray,
        sdf_ordinals: np.ndarray,
    ) -> None:
        offsets: dict[int, int] = {}
        total = 0
        sidecar_sources = np.empty(
            self.source_indices.shape,
            dtype=np.int64,
        )
        sidecar_rows = np.empty(
            self.source_indices.shape,
            dtype=np.int64,
        )
        sidecar_sdf_ordinals = np.empty(
            self.source_indices.shape,
            dtype=np.int64,
        )
        for shard_id in sorted(self._descriptors):
            descriptor = self._descriptors[shard_id]
            offsets[shard_id] = total
            records = self._load_sidecar_records(shard_id, cache=False)
            for ordinal, entry in enumerate(records):
                sidecar_sources[total + ordinal] = int(entry["source_index"])
                sidecar_rows[total + ordinal] = int(entry["row_index"])
                sidecar_sdf_ordinals[total + ordinal] = (
                    -1
                    if entry["sdf_ordinal"] is None
                    else int(entry["sdf_ordinal"])
                )
            total += descriptor.record_count
        if total != self.source_indices.size:
            raise RuntimeError(
                "geometry sidecar inventory size differs from the index"
            )
        seen_locators = np.zeros(total, dtype=np.bool_)
        for start in range(0, total, 65_536):
            stop = min(total, start + 65_536)
            record_limits = np.fromiter(
                (
                    self._descriptors[int(shard_id)].record_count
                    for shard_id in self.shard_ids[start:stop]
                ),
                dtype=np.int64,
                count=stop - start,
            )
            if np.any(
                self.record_ordinals[start:stop].astype(
                    np.int64,
                    copy=False,
                )
                >= record_limits
            ):
                raise RuntimeError(
                    "geometry index record_ordinal exceeds its shard"
                )
            positions = np.fromiter(
                (
                    offsets[int(shard_id)] + int(record_ordinal)
                    for shard_id, record_ordinal in zip(
                        self.shard_ids[start:stop],
                        self.record_ordinals[start:stop],
                    )
                ),
                dtype=np.int64,
                count=stop - start,
            )
            if (
                np.any(positions < 0)
                or np.any(positions >= total)
                or np.unique(positions).size != positions.size
                or np.any(seen_locators[positions])
            ):
                raise RuntimeError(
                    "geometry index shard/record locators are not a "
                    "permutation of the sidecar inventory"
                )
            seen_locators[positions] = True
            if (
                not np.array_equal(
                    sidecar_sources[positions],
                    self.source_indices[start:stop],
                )
                or not np.array_equal(
                    sidecar_rows[positions],
                    row_indices[start:stop],
                )
                or not np.array_equal(
                    sidecar_sdf_ordinals[positions],
                    sdf_ordinals[start:stop],
                )
            ):
                raise RuntimeError(
                    "geometry index numeric locators differ from sidecars"
                )
        if total and not bool(np.all(seen_locators)):
            raise RuntimeError(
                "geometry index does not cover the sidecar inventory"
            )

    def _archive(self, shard_id: int):
        archive = self._open.pop(shard_id, None)
        if archive is not None:
            self._open[shard_id] = archive
            return archive
        descriptor = self._descriptors[shard_id]
        artifact = self.directory / descriptor.artifact_name
        if self.verify_checksums and shard_id not in self._verified_shards:
            actual = sha256_file(artifact)
            if actual != descriptor.artifact_sha256:
                raise RuntimeError(
                    f"geometry shard checksum mismatch: {artifact}"
                )
            self._verified_shards.add(shard_id)
        archive = np.load(artifact, allow_pickle=False)
        self._open[shard_id] = archive
        while len(self._open) > self.max_open_shards:
            _, old_archive = self._open.popitem(last=False)
            old_archive.close()
        return archive

    def get(self, source_index: int) -> GeometryRecord:
        position = int(
            np.searchsorted(
                self.source_indices,
                int(source_index),
                side="left",
            )
        )
        if (
            position >= len(self.source_indices)
            or int(self.source_indices[position]) != int(source_index)
        ):
            raise KeyError(f"geometry source_index={source_index} not found")
        shard_id = int(self.shard_ids[position])
        ordinal = int(self.record_ordinals[position])
        records = self._load_sidecar_records(shard_id, cache=True)
        if (
            ordinal < 0
            or ordinal >= len(records)
            or int(records[ordinal]["source_index"]) != int(source_index)
        ):
            raise RuntimeError(
                "geometry index source/key locator differs from its sidecar"
            )
        key = f"r{ordinal:06d}"
        return GeometryRecord.from_storage_dict(
            self._archive(shard_id),
            prefix=f"{key}__",
        )

    def verify_all_artifacts(self) -> None:
        """Read and verify every referenced geometry artifact exactly once."""

        for shard_id in sorted(self._descriptors):
            if shard_id in self._verified_shards:
                continue
            descriptor = self._descriptors[shard_id]
            artifact = self.directory / descriptor.artifact_name
            actual = sha256_file(artifact)
            if actual != descriptor.artifact_sha256:
                raise RuntimeError(
                    f"geometry shard checksum mismatch: {artifact}"
                )
            self._verified_shards.add(shard_id)

    def close(self) -> None:
        for archive in self._open.values():
            archive.close()
        self._open.clear()
        self._records.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return


def _selection_batches(
    path: os.PathLike[str] | str,
    *,
    batch_size: int,
    n_bins: int,
) -> Iterator[tuple[int, list[Candidate]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = Path(path)
    parquet_file = parquet.ParquetFile(source)
    available = set(parquet_file.schema.names)
    missing = set(REQUIRED_SELECTION_COLUMNS) - available
    if missing:
        raise ValueError(
            f"selection manifest is missing columns: {sorted(missing)}"
        )
    expected_rank = [0] * n_bins
    previous_pair: Optional[tuple[int, int]] = None
    pending_bin: Optional[int] = None
    pending: list[Candidate] = []

    for record_batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=list(REQUIRED_SELECTION_COLUMNS),
    ):
        columns = record_batch.to_pydict()
        for values in zip(
            *(columns[name] for name in REQUIRED_SELECTION_COLUMNS)
        ):
            source_index, smiles, gap, gap_bin, rank = values
            normalized_bin = int(gap_bin)
            normalized_rank = int(rank)
            if not 0 <= normalized_bin < n_bins:
                raise ValueError(
                    f"selection gap_bin={normalized_bin} outside [0, {n_bins})"
                )
            if normalized_rank != expected_rank[normalized_bin]:
                raise ValueError(
                    f"gap_bin={normalized_bin} selection ranks must be "
                    f"contiguous: expected {expected_rank[normalized_bin]}, "
                    f"got {normalized_rank}"
                )
            expected_rank[normalized_bin] += 1
            pair = (normalized_bin, normalized_rank)
            if previous_pair is not None and pair <= previous_pair:
                raise ValueError(
                    "selection manifest must be strictly sorted by "
                    "(gap_bin, selection_rank)"
                )
            previous_pair = pair
            candidate = Candidate(
                source_index=int(source_index),
                canonical_smiles=str(smiles),
                gap=float(gap),
                gap_bin=normalized_bin,
                selection_rank=normalized_rank,
            )
            if not math.isfinite(candidate.gap):
                raise ValueError(
                    f"source_index={candidate.source_index} has non-finite gap"
                )
            if pending_bin is None:
                pending_bin = candidate.gap_bin
            if (
                candidate.gap_bin != pending_bin
                or len(pending) >= batch_size
            ):
                yield int(pending_bin), pending
                pending = []
                pending_bin = candidate.gap_bin
            pending.append(candidate)
    if pending:
        if pending_bin is None:
            raise RuntimeError("internal selection batching invariant failed")
        yield int(pending_bin), pending


def _density_extent_fits(
    coords: np.ndarray,
    conformer_mask: np.ndarray,
    *,
    conformer_index: Optional[int],
    grid_size: int,
    spacing: float,
    padding: float,
) -> bool:
    coordinates = np.asarray(coords, dtype=np.float64)
    mask = np.asarray(conformer_mask, dtype=np.bool_)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise ValueError("geometry coords must have shape (C, A, 3)")
    if mask.shape != (coordinates.shape[0],):
        raise ValueError("geometry conformer_mask must have shape (C,)")
    if conformer_index is None:
        selected = coordinates[mask]
    else:
        index = int(conformer_index)
        if index < 0 or index >= coordinates.shape[0] or not mask[index]:
            return False
        selected = coordinates[index : index + 1]
    if selected.size == 0 or not np.all(np.isfinite(selected)):
        return False
    required_extent = (
        selected.max(axis=(0, 1))
        - selected.min(axis=(0, 1))
        + 2.0 * float(padding)
    )
    available_extent = (int(grid_size) - 1) * float(spacing)
    return bool(np.all(required_extent <= available_extent + 1e-10))


def _density_conformer_selection_valid(
    coords: np.ndarray,
    conformer_mask: np.ndarray,
    conformer_index: Optional[int],
) -> bool:
    coordinates = np.asarray(coords, dtype=np.float64)
    mask = np.asarray(conformer_mask, dtype=np.bool_)
    if (
        coordinates.ndim != 3
        or coordinates.shape[2] != 3
        or mask.shape != (coordinates.shape[0],)
        or not np.all(np.isfinite(coordinates))
    ):
        return False
    if conformer_index is None:
        return bool(np.any(mask))
    index = int(conformer_index)
    return bool(
        0 <= index < coordinates.shape[0]
        and mask[index]
    )


def _preflight_density_extent(
    *,
    selection_path: Path,
    repository: GeometryRepository,
    accumulator: SelectionAccumulator,
    feature_config: FeatureBuildConfig,
    batch_size: int,
    prior_failure_count: int,
    max_failure_rate: float,
) -> dict[str, Any]:
    max_quotas = target_quotas(
        accumulator.max_target,
        accumulator.n_bins,
    )
    remaining = [
        max_quotas[gap_bin] - accumulator.accepted_in_bin(gap_bin)
        for gap_bin in range(accumulator.n_bins)
    ]
    passing = [0] * accumulator.n_bins
    scanned = [0] * accumulator.n_bins
    overflow = [0] * accumulator.n_bins
    unavailable = [0] * accumulator.n_bins
    initial_window_passing = [0] * accumulator.n_bins
    initial_window_scanned = [0] * accumulator.n_bins

    if any(value < 0 for value in remaining):
        raise RuntimeError("resume state exceeds the configured density quotas")
    if not any(remaining):
        return {
            "available_extent": (
                (feature_config.grid_size - 1) * feature_config.grid_spacing
            ),
            "remaining_required_by_bin": remaining,
            "scanned_by_bin": scanned,
            "passing_by_bin": passing,
            "overflow_by_bin": overflow,
            "unavailable_by_bin": unavailable,
            "initial_window_scanned_by_bin": initial_window_scanned,
            "initial_window_passing_by_bin": initial_window_passing,
            "minimum_projected_failure_rate": (
                0.0
                if accumulator.accepted_count + prior_failure_count == 0
                else prior_failure_count
                / (accumulator.accepted_count + prior_failure_count)
            ),
        }

    for gap_bin, candidates in _selection_batches(
        selection_path,
        batch_size=batch_size,
        n_bins=accumulator.n_bins,
    ):
        if passing[gap_bin] >= remaining[gap_bin]:
            continue
        last_completed_rank = accumulator.last_selection_rank(gap_bin)
        for candidate in candidates:
            if candidate.selection_rank <= last_completed_rank:
                continue
            if passing[gap_bin] >= remaining[gap_bin]:
                break
            scanned[gap_bin] += 1
            in_initial_window = candidate.selection_rank < max_quotas[gap_bin]
            if in_initial_window:
                initial_window_scanned[gap_bin] += 1
            try:
                geometry = repository.get(candidate.source_index)
            except KeyError:
                unavailable[gap_bin] += 1
                continue
            fits = (
                _density_conformer_selection_valid(
                    geometry.coords,
                    geometry.conformer_mask,
                    feature_config.density_conformer_index,
                )
                if not feature_config.strict_density_bounds
                else _density_extent_fits(
                    geometry.coords,
                    geometry.conformer_mask,
                    conformer_index=feature_config.density_conformer_index,
                    grid_size=feature_config.grid_size,
                    spacing=feature_config.grid_spacing,
                    padding=feature_config.grid_padding,
                )
            )
            if not fits:
                overflow[gap_bin] += 1
                continue
            passing[gap_bin] += 1
            if in_initial_window:
                initial_window_passing[gap_bin] += 1
        if all(
            passing[index] >= remaining[index]
            for index in range(accumulator.n_bins)
        ):
            break

    deficits = {
        str(gap_bin): remaining[gap_bin] - passing[gap_bin]
        for gap_bin in range(accumulator.n_bins)
        if passing[gap_bin] < remaining[gap_bin]
    }
    projected_failures = (
        int(prior_failure_count) + sum(overflow) + sum(unavailable)
    )
    projected_attempts = (
        accumulator.accepted_count
        + int(prior_failure_count)
        + sum(scanned)
    )
    projected_failure_rate = (
        0.0
        if projected_attempts == 0
        else projected_failures / projected_attempts
    )
    summary = {
        "available_extent": (
            (feature_config.grid_size - 1) * feature_config.grid_spacing
        ),
        "remaining_required_by_bin": remaining,
        "scanned_by_bin": scanned,
        "passing_by_bin": passing,
        "overflow_by_bin": overflow,
        "unavailable_by_bin": unavailable,
        "initial_window_scanned_by_bin": initial_window_scanned,
        "initial_window_passing_by_bin": initial_window_passing,
        "minimum_projected_failure_rate": projected_failure_rate,
    }
    if deficits:
        raise RuntimeError(
            "fixed-grid density preflight cannot satisfy exact per-bin quotas; "
            f"deficits={deficits}, summary={summary}"
        )
    if projected_failure_rate > max_failure_rate:
        raise RuntimeError(
            "fixed-grid density preflight already exceeds the configured "
            "feature failure-rate limit; "
            f"minimum_rate={projected_failure_rate:.6f}, "
            f"limit={max_failure_rate:.6f}, summary={summary}"
        )
    return summary


_WORKER_BUILDER: Optional[MultimodalFeatureBuilder] = None
_WORKER_GEOMETRY: Optional[GeometryRepository] = None


def _release_local_worker_state() -> None:
    global _WORKER_BUILDER, _WORKER_GEOMETRY
    if _WORKER_GEOMETRY is not None:
        _WORKER_GEOMETRY.close()
    _WORKER_GEOMETRY = None
    _WORKER_BUILDER = None


def _initialize_worker(
    tokenizer_snapshot: str,
    expected_tokenizer_sha256: str,
    expected_tokenizer_vocab_size: int,
    geometry_dir: str,
    expected_geometry_manifest_sha256: str,
    expected_geometry_run_state_sha256: str,
    expected_geometry_index_metadata_sha256: str,
    expected_geometry_index_artifact_sha256: str,
    feature_config: Mapping[str, Any],
    verify_checksums: bool,
    max_open_geometry_shards: int,
) -> None:
    global _WORKER_BUILDER, _WORKER_GEOMETRY
    _release_local_worker_state()
    tokenizer = _load_pinned_tokenizer(
        tokenizer_snapshot,
        expected_sha256=expected_tokenizer_sha256,
        expected_vocab_size=expected_tokenizer_vocab_size,
    )
    _WORKER_BUILDER = MultimodalFeatureBuilder(
        tokenizer,
        FeatureBuildConfig(**dict(feature_config)),
    )
    _WORKER_GEOMETRY = GeometryRepository(
        geometry_dir,
        verify_checksums=verify_checksums,
        validate_index_inventory=False,
        expected_manifest_sha256=expected_geometry_manifest_sha256,
        expected_run_state_sha256=expected_geometry_run_state_sha256,
        expected_index_metadata_sha256=(
            expected_geometry_index_metadata_sha256
        ),
        expected_index_artifact_sha256=(
            expected_geometry_index_artifact_sha256
        ),
        max_open_shards=max_open_geometry_shards,
    )


def _build_candidate(
    candidate: Candidate,
) -> tuple[Candidate, Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    if _WORKER_BUILDER is None or _WORKER_GEOMETRY is None:
        raise RuntimeError("PCQM build worker was not initialized")
    try:
        try:
            geometry = _WORKER_GEOMETRY.get(candidate.source_index)
        except KeyError as exc:
            raise FeatureBuildError(
                "geometry_lookup",
                str(exc),
                source_index=candidate.source_index,
            ) from exc
        record = _WORKER_BUILDER.build_record(
            smiles=candidate.canonical_smiles,
            source_index=candidate.source_index,
            sample_namespace="pcqm",
            geometry=geometry,
            metadata={
                "gap": candidate.gap,
                "gap_bin": candidate.gap_bin,
                "selection_rank": candidate.selection_rank,
            },
        )
        return candidate, record, None
    except FeatureBuildError as exc:
        return candidate, None, exc.to_dict(candidate.canonical_smiles)


def _atomic_savez(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool,
) -> None:
    with atomic_output_path(path, overwrite=overwrite) as temporary:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())


def _write_build_index(
    staging_dir: Path,
    shard_id: int,
    accepted: Sequence[AcceptedCandidate],
) -> Path:
    if not accepted:
        raise ValueError("cannot write an empty build index")
    ordered = sorted(accepted, key=lambda item: item.record_index)
    expected = np.arange(
        ordered[0].record_index,
        ordered[0].record_index + len(ordered),
        dtype=np.int64,
    )
    actual = np.asarray(
        [item.record_index for item in ordered],
        dtype=np.int64,
    )
    if not np.array_equal(actual, expected):
        raise ValueError(
            "record indices inside one shard must be contiguous"
        )
    arrays = {
        "record_index": actual,
        "source_index": np.asarray(
            [item.source_index for item in ordered],
            dtype=np.int64,
        ),
        "gap_bin": np.asarray(
            [item.gap_bin for item in ordered],
            dtype=np.int16,
        ),
        "selection_rank": np.asarray(
            [item.selection_rank for item in ordered],
            dtype=np.int64,
        ),
        "success_rank": np.asarray(
            [item.success_rank for item in ordered],
            dtype=np.int64,
        ),
    }
    metadata = {
        "schema": BUILD_INDEX_SCHEMA,
        "shard_id": int(shard_id),
        "start_index": int(ordered[0].record_index),
        "record_count": len(ordered),
    }
    arrays = {
        **arrays,
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
    }
    destination = staging_dir / f"build-index-{shard_id:06d}.npz"
    _atomic_savez(destination, arrays, overwrite=False)
    return destination


def _artifact_shard_ids(directory: Path, pattern: str) -> set[int]:
    expression = re.compile(pattern)
    result: set[int] = set()
    for path in directory.iterdir():
        match = expression.fullmatch(path.name)
        if match is not None:
            result.add(int(match.group(1)))
    return result


def _converge_staged_shards(staging_dir: Path) -> None:
    lmdb_ids = _artifact_shard_ids(
        staging_dir,
        r"shard-(\d{6})\.lmdb",
    )
    sidecar_ids = _artifact_shard_ids(
        staging_dir,
        r"shard-(\d{6})\.json",
    )
    index_ids = _artifact_shard_ids(
        staging_dir,
        r"build-index-(\d{6})\.npz",
    )
    sidecars_without_lmdb = sidecar_ids - lmdb_ids
    if sidecars_without_lmdb:
        raise RuntimeError(
            "staged shard sidecars have no LMDB directories: "
            f"{sorted(sidecars_without_lmdb)}"
        )
    for shard_id in sorted(index_ids - lmdb_ids):
        orphan = staging_dir / f"build-index-{shard_id:06d}.npz"
        if orphan.parent != staging_dir:
            raise RuntimeError("refusing to remove an out-of-scope build index")
        orphan.unlink()
    for shard_id in sorted(lmdb_ids - sidecar_ids):
        index_path = staging_dir / f"build-index-{shard_id:06d}.npz"
        if not index_path.is_file():
            raise RuntimeError(
                "published LMDB has neither sidecar nor recoverable build index: "
                f"shard_id={shard_id}"
            )
        with np.load(index_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            records = np.asarray(archive["record_index"], dtype=np.int64)
        if (
            metadata.get("schema") != BUILD_INDEX_SCHEMA
            or int(metadata.get("shard_id", -1)) != shard_id
            or records.ndim != 1
            or records.size == 0
            or int(metadata.get("record_count", -1)) != records.size
            or int(metadata.get("start_index", -1)) != int(records[0])
            or not np.array_equal(
                records,
                np.arange(
                    int(records[0]),
                    int(records[0]) + records.size,
                    dtype=np.int64,
                ),
            )
        ):
            raise RuntimeError(
                f"cannot recover shard from invalid build index: {index_path}"
            )
        recover_published_shard_sidecar(
            staging_dir / f"shard-{shard_id:06d}.lmdb",
            staging_dir / f"shard-{shard_id:06d}.json",
            shard_id=shard_id,
            start_index=int(records[0]),
            expected_records=int(records.size),
        )


def _restore_completed_indices(
    staging_dir: Path,
    *,
    accumulator: SelectionAccumulator,
    verify_checksums: bool,
) -> int:
    _converge_staged_shards(staging_dir)
    sidecars = sorted(staging_dir.glob("shard-*.json"))
    shard_dirs = sorted(staging_dir.glob("shard-*.lmdb"))
    expected_names = {
        f"shard-{index:06d}.lmdb" for index in range(len(sidecars))
    }
    if {path.name for path in shard_dirs} != expected_names:
        raise RuntimeError(
            "staged LMDB shards are not a contiguous, sidecar-complete prefix"
        )
    for shard_id, sidecar_path in enumerate(sidecars):
        if sidecar_path.name != f"shard-{shard_id:06d}.json":
            raise RuntimeError("staged shard sidecars are not contiguous")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        shard_dir = staging_dir / f"shard-{shard_id:06d}.lmdb"
        expected_start = accumulator.accepted_count
        sidecar_count = sidecar.get("record_count")
        if (
            sidecar.get("schema_version") != 1
            or sidecar.get("shard_id") != shard_id
            or sidecar.get("start_index") != expected_start
            or not isinstance(sidecar_count, int)
            or isinstance(sidecar_count, bool)
            or sidecar_count <= 0
            or sidecar.get("end_index_exclusive")
            != expected_start + sidecar_count
            or sidecar.get("codec") != "msgpack+zstd+sha256"
        ):
            raise RuntimeError(
                f"invalid staged shard sidecar: {sidecar_path}"
            )
        if verify_checksums:
            actual = sha256_file(shard_dir / "data.mdb")
            if actual != str(sidecar.get("sha256", "")):
                raise RuntimeError(
                    f"staged shard checksum mismatch: {shard_dir}"
                )
        index_path = staging_dir / f"build-index-{shard_id:06d}.npz"
        if not index_path.is_file():
            raise RuntimeError(
                f"staged shard build index missing: {index_path}"
            )
        with np.load(index_path, allow_pickle=False) as archive:
            expected_fields = {
                "record_index",
                "source_index",
                "gap_bin",
                "selection_rank",
                "success_rank",
                "metadata_json",
            }
            if set(archive.files) != expected_fields:
                raise RuntimeError(
                    f"invalid staged build-index inventory: {index_path}"
                )
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (
                metadata.get("schema") != BUILD_INDEX_SCHEMA
                or int(metadata.get("shard_id", -1)) != shard_id
                or int(metadata.get("start_index", -1)) != expected_start
                or int(metadata.get("record_count", -1)) != sidecar_count
            ):
                raise RuntimeError(f"invalid staged build index: {index_path}")
            record_index = np.asarray(archive["record_index"], dtype=np.int64)
            source_index = np.asarray(archive["source_index"], dtype=np.int64)
            gap_bin = np.asarray(archive["gap_bin"], dtype=np.int64)
            selection_rank = np.asarray(
                archive["selection_rank"],
                dtype=np.int64,
            )
            success_rank = np.asarray(archive["success_rank"], dtype=np.int64)
        lengths = {
            len(record_index),
            len(source_index),
            len(gap_bin),
            len(selection_rank),
            len(success_rank),
        }
        if lengths != {sidecar_count}:
            raise RuntimeError(
                f"staged build index count differs from shard: {index_path}"
            )
        if not np.array_equal(
            record_index,
            np.arange(
                expected_start,
                expected_start + sidecar_count,
                dtype=np.int64,
            ),
        ):
            raise RuntimeError(
                f"staged build index is not contiguous: {index_path}"
            )
        for values in zip(
            record_index,
            source_index,
            gap_bin,
            selection_rank,
            success_rank,
        ):
            restored = accumulator.accept(
                record_index=int(values[0]),
                source_index=int(values[1]),
                gap_bin=int(values[2]),
                selection_rank=int(values[3]),
            )
            if restored.success_rank != int(values[4]):
                raise RuntimeError(
                    "staged success_rank differs from its per-bin position"
                )
    return accumulator.accepted_count


def _contract_payload(
    *,
    selection_manifest: Path,
    selection_manifest_sha256: str,
    selection_metadata: Path,
    selection_metadata_sha256: str,
    tokenizer_snapshot: TokenizerSnapshot,
    geometry_dir: Path,
    target_sizes: Sequence[int],
    n_bins: int,
    records_per_shard: int,
    map_size: int,
    compression_level: int,
    feature_config: FeatureBuildConfig,
) -> dict[str, Any]:
    geometry_index_metadata = geometry_dir / "geometry_index.json"
    if not geometry_index_metadata.is_file():
        raise FileNotFoundError(
            f"geometry index metadata missing: {geometry_index_metadata}"
        )
    geometry_manifest = geometry_dir / "manifest.json"
    if not geometry_manifest.is_file():
        raise FileNotFoundError(
            f"geometry manifest missing: {geometry_manifest}"
        )
    geometry_run_state = geometry_dir / "run_state.json"
    if not geometry_run_state.is_file():
        raise FileNotFoundError(
            f"geometry run state missing: {geometry_run_state}"
        )
    geometry_manifest_payload, geometry_manifest_sha256 = _read_pinned_json(
        geometry_manifest,
        label="geometry manifest",
    )
    geometry_run_state_payload, geometry_run_state_sha256 = _read_pinned_json(
        geometry_run_state,
        label="geometry run state",
    )
    _validate_geometry_run_contract(
        geometry_manifest_payload,
        geometry_run_state_payload,
    )
    if (
        not _is_sha256(selection_manifest_sha256)
        or not _is_sha256(selection_metadata_sha256)
    ):
        raise ValueError("selection artifact SHA256 is invalid")
    geometry_index_contract, geometry_index_metadata_sha256 = (
        _read_pinned_json(
            geometry_index_metadata,
            label="geometry index metadata",
        )
    )
    geometry_index_artifact = geometry_dir / str(
        geometry_index_contract.get("filename", "")
    )
    geometry_index_artifact_sha256 = str(
        geometry_index_contract.get("sha256", "")
    )
    if (
        set(geometry_index_contract)
        != {
            "schema",
            "filename",
            "sha256",
            "record_count",
            "sorted_by",
            "lookup",
        }
        or geometry_index_contract.get("schema")
        != "semmol.geometry_index.v2"
        or geometry_index_contract.get("sorted_by")
        != ["source_index", "row_index"]
        or geometry_index_contract.get("lookup")
        != "numpy.searchsorted(source_index, requested_source_index)"
        or not isinstance(
            geometry_index_contract.get("record_count"),
            int,
        )
        or isinstance(geometry_index_contract["record_count"], bool)
        or geometry_index_contract["record_count"] < 0
        or not geometry_index_artifact.is_file()
        or not _is_sha256(geometry_index_artifact_sha256)
    ):
        raise RuntimeError("geometry index metadata contract is invalid")
    return {
        "schema": BUILD_SCHEMA,
        "selection_manifest": {
            "path": str(selection_manifest.resolve()),
            "sha256": selection_manifest_sha256,
        },
        "selection_metadata": {
            "path": str(selection_metadata.resolve()),
            "sha256": selection_metadata_sha256,
        },
        "tokenizer": {
            "path": str(tokenizer_snapshot.root),
            "snapshot_path": str(tokenizer_snapshot.load_path),
            "artifact_sha256": tokenizer_snapshot.artifact_sha256,
            "vocab_size": tokenizer_snapshot.vocab_size,
        },
        "geometry": {
            "path": str(geometry_dir.resolve()),
            "manifest_sha256": geometry_manifest_sha256,
            "run_state_sha256": geometry_run_state_sha256,
            "index_metadata_sha256": geometry_index_metadata_sha256,
            "index_artifact_sha256": geometry_index_artifact_sha256,
        },
        "target_sizes": list(map(int, target_sizes)),
        "n_bins": int(n_bins),
        "storage": {
            "records_per_shard": int(records_per_shard),
            "map_size": int(map_size),
            "compression_level": int(compression_level),
        },
        "features": feature_config.to_dict(),
    }


def _prepare_selection_snapshot(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> Path:
    """Create or verify the private immutable selection input for this build."""
    poison_path = destination.parent / ".selection-snapshot.poisoned.json"
    if poison_path.exists():
        raise RuntimeError(
            "selection snapshot staging is poisoned and cannot be resumed; "
            f"remove the staged build and restart: {poison_path}"
        )
    if not _is_sha256(expected_sha256):
        raise ValueError("selection snapshot expected SHA256 is invalid")
    if destination.exists() or destination.is_symlink():
        try:
            if _path_is_redirected(destination) or not destination.is_file():
                raise RuntimeError(
                    f"selection snapshot is not a private file: {destination}"
                )
            if sha256_file(destination) != expected_sha256:
                raise RuntimeError("selection snapshot hash mismatch")
        except BaseException as exc:
            atomic_write_json(
                poison_path,
                {
                    "schema": "semmol.pcqm_selection_snapshot_poison.v1",
                    "reason": "snapshot_hash_mismatch",
                    "expected_sha256": expected_sha256,
                },
                overwrite=True,
            )
            raise RuntimeError(
                "staged selection snapshot differs from the build contract"
            ) from exc
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(destination, overwrite=False) as temporary:
        with source.open("rb") as source_stream, temporary.open(
            "wb"
        ) as snapshot_stream:
            shutil.copyfileobj(
                source_stream,
                snapshot_stream,
                length=1024 * 1024,
            )
            snapshot_stream.flush()
            os.fsync(snapshot_stream.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise RuntimeError(
                "selection manifest changed while creating the build snapshot"
            )
    return destination


_PRIVATE_SNAPSHOT_TEMP_PATTERNS = (
    "..selection-manifest.snapshot.parquet.tmp-*",
    "..geometry-input.snapshot.tmp-*",
    "..tokenizer-generation.snapshot.tmp-*",
    "..selection-snapshot.poisoned.json.tmp-*",
    "..geometry-snapshot.poisoned.json.tmp-*",
    "..tokenizer-snapshot.poisoned.json.tmp-*",
    "..geometry-input.snapshot.delete-*",
    "..tokenizer-generation.snapshot.delete-*",
)


def _path_is_redirected(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or (callable(is_junction) and is_junction())
        or path.resolve() != path.absolute()
    )


def _assert_staging_tree_not_redirected(staging_path: Path) -> None:
    """Reject links and junctions before reading or mutating staged state."""

    if not staging_path.is_dir() or _path_is_redirected(staging_path):
        raise RuntimeError(
            f"PCQM staging root is missing or redirected: {staging_path}"
        )
    pending = [staging_path]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                children = list(entries)
        except OSError as exc:
            raise RuntimeError(
                f"cannot audit PCQM staging directory: {directory}"
            ) from exc
        for entry in children:
            candidate = Path(entry.path)
            is_junction = getattr(candidate, "is_junction", None)
            if entry.is_symlink() or (
                callable(is_junction) and is_junction()
            ):
                raise RuntimeError(
                    f"PCQM staging entry is redirected: {candidate}"
                )
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(
                    f"cannot inspect PCQM staging entry: {candidate}"
                ) from exc
            if is_directory:
                pending.append(candidate)


def _cleanup_private_snapshot_temps(staging_path: Path) -> None:
    """Converge only known private-input temporary artifacts after a crash."""
    root = staging_path.resolve()
    removed = False
    for pattern in _PRIVATE_SNAPSHOT_TEMP_PATTERNS:
        for candidate in root.glob(pattern):
            if candidate.parent.resolve() != root:
                raise RuntimeError(
                    f"private snapshot temporary escaped staging: {candidate}"
                )
            if _path_is_redirected(candidate):
                raise RuntimeError(
                    f"private snapshot temporary is redirected: {candidate}"
                )
            if candidate.is_file():
                candidate.unlink()
            elif candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                raise RuntimeError(
                    f"unsupported private snapshot temporary: {candidate}"
                )
            removed = True
    if removed:
        _fsync_directory(root)


def _cleanup_known_build_temporaries(staging_path: Path) -> None:
    """Remove only crash leftovers whose names are owned by this builder."""

    root = staging_path.resolve()
    root_file_patterns = (
        re.compile(
            r"\.(?:build-contract\.json|build-manifest\.json|"
            r"feature-failures\.jsonl|build-index-\d{6}\.npz)"
            r"\.tmp-.+"
        ),
        re.compile(r"\.store\.json\.tmp-.+\.json"),
        re.compile(r"\.shard-\d{6}\.json\.tmp-.+\.json"),
    )
    shard_directory_pattern = re.compile(r"\.shard-\d{6}\.tmp-.+")
    removed_root = False
    for candidate in list(root.iterdir()):
        if shard_directory_pattern.fullmatch(candidate.name):
            if (
                _path_is_redirected(candidate)
                or not candidate.is_dir()
                or candidate.resolve().parent != root
            ):
                raise RuntimeError(
                    f"unsafe staged LMDB temporary: {candidate}"
                )
            shutil.rmtree(candidate)
            removed_root = True
            continue
        if any(pattern.fullmatch(candidate.name) for pattern in root_file_patterns):
            if candidate.is_dir() and not candidate.is_symlink():
                raise RuntimeError(
                    f"atomic build temporary is a directory: {candidate}"
                )
            if candidate.is_symlink():
                raise RuntimeError(
                    f"atomic build temporary is redirected: {candidate}"
                )
            candidate.unlink(missing_ok=True)
            removed_root = True
    if removed_root:
        _fsync_directory(root)

    views_dir = root / "views"
    if views_dir.is_symlink():
        raise RuntimeError(f"views directory is redirected: {views_dir}")
    if views_dir.is_dir():
        view_temporary = re.compile(
            r"\.pcqm_\d+(?:m)?\.npz\.tmp-.+"
        )
        removed_views = False
        for candidate in list(views_dir.iterdir()):
            if not view_temporary.fullmatch(candidate.name):
                continue
            if candidate.is_dir() and not candidate.is_symlink():
                raise RuntimeError(
                    f"view atomic temporary is a directory: {candidate}"
                )
            if candidate.is_symlink():
                raise RuntimeError(
                    f"view atomic temporary is redirected: {candidate}"
                )
            candidate.unlink(missing_ok=True)
            removed_views = True
        if removed_views:
            _fsync_directory(views_dir)


def _assert_no_private_staging_entries(staging_path: Path) -> None:
    hidden = sorted(
        path.name for path in staging_path.iterdir()
        if path.name.startswith(".")
    )
    if hidden:
        raise RuntimeError(
            f"private PCQM staging artifacts remain before publication: "
            f"{hidden}"
        )


def _assert_final_staging_inventory(
    staging_path: Path,
    *,
    metadata: StoreMetadata,
    view_paths: Sequence[str],
) -> None:
    """Prove the published directory is exact and self-contained."""

    shard_count = len(metadata.shards)
    expected_root = {
        "build-contract.json",
        "build-manifest.json",
        "feature-failures.jsonl",
        "store.json",
        "views",
        *metadata.shards,
        *(
            f"shard-{shard_id:06d}.json"
            for shard_id in range(shard_count)
        ),
        *(
            f"build-index-{shard_id:06d}.npz"
            for shard_id in range(shard_count)
        ),
    }
    actual_root = {entry.name for entry in staging_path.iterdir()}
    if actual_root != expected_root:
        raise RuntimeError(
            "final PCQM staging inventory is not exact: "
            f"missing={sorted(expected_root - actual_root)}, "
            f"unknown={sorted(actual_root - expected_root)}"
        )
    expected_views = set()
    for relative_name in view_paths:
        relative = Path(relative_name)
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != "views"
            or relative.name != relative.parts[1]
        ):
            raise RuntimeError(
                f"invalid final PCQM view path: {relative_name!r}"
            )
        expected_views.add(relative.name)
    views_dir = staging_path / "views"
    actual_views = {entry.name for entry in views_dir.iterdir()}
    if actual_views != expected_views:
        raise RuntimeError(
            "final PCQM view inventory is not exact: "
            f"missing={sorted(expected_views - actual_views)}, "
            f"unknown={sorted(actual_views - expected_views)}"
        )
    for shard_name in metadata.shards:
        shard_dir = staging_path / shard_name
        shard_inventory = {entry.name for entry in shard_dir.iterdir()}
        if shard_inventory != {"data.mdb", "lock.mdb"}:
            raise RuntimeError(
                f"final LMDB inventory is not exact for {shard_name}: "
                f"{sorted(shard_inventory)}"
            )


def _expected_pcqm_view_paths(
    target_sizes: Sequence[int],
) -> tuple[str, ...]:
    paths = []
    for target in target_sizes:
        short_name = (
            f"{target // 1_000_000}m"
            if target % 1_000_000 == 0
            else str(target)
        )
        paths.append(f"views/pcqm_{short_name}.npz")
    return tuple(paths)


def _legacy_tokenizer_inventory(directory: Path) -> tuple[Path, ...]:
    excluded_names = {"failures.jsonl", "statistics.json"}
    inventory = tuple(
        sorted(
            (
                path.relative_to(directory)
                for path in directory.rglob("*")
                if path.is_file()
                and path.name not in excluded_names
                and ".tmp" not in path.name
            ),
            key=lambda path: path.as_posix(),
        )
    )
    if not inventory:
        raise FileNotFoundError(
            f"legacy tokenizer has no immutable artifacts: {directory}"
        )
    return inventory


def _prepare_tokenizer_snapshot(
    source_snapshot: TokenizerSnapshot,
    destination: Path,
) -> TokenizerSnapshot:
    """Create or verify the private tokenizer inventory consumed by workers."""
    poison_path = destination.parent / ".tokenizer-snapshot.poisoned.json"
    if poison_path.exists():
        raise RuntimeError(
            "tokenizer snapshot staging is poisoned and cannot be resumed; "
            f"remove the staged build and restart: {poison_path}"
        )
    private_snapshot = TokenizerSnapshot(
        root=source_snapshot.root,
        load_path=destination,
        artifact_sha256=source_snapshot.artifact_sha256,
        vocab_size=source_snapshot.vocab_size,
    )
    if destination.exists() or destination.is_symlink():
        try:
            if _path_is_redirected(destination) or not destination.is_dir():
                raise RuntimeError(
                    f"tokenizer snapshot is not a private directory: "
                    f"{destination}"
                )
            _load_pinned_tokenizer(
                destination,
                expected_sha256=source_snapshot.artifact_sha256,
                expected_vocab_size=source_snapshot.vocab_size,
            )
        except BaseException as exc:
            atomic_write_json(
                poison_path,
                {
                    "schema": "semmol.pcqm_tokenizer_snapshot_poison.v1",
                    "reason": "snapshot_contract_mismatch",
                    "expected_sha256": source_snapshot.artifact_sha256,
                },
                overwrite=True,
            )
            raise RuntimeError(
                "staged tokenizer snapshot differs from the build contract"
            ) from exc
        return private_snapshot

    source_root = source_snapshot.load_path.resolve()
    generation_manifest = source_root / ARTIFACT_MANIFEST_NAME
    if generation_manifest.is_file():
        inventory = tuple(
            Path(name)
            for name in _verify_generation_directory(
                source_root,
                source_snapshot.artifact_sha256,
            )
        )
    else:
        if (
            tokenizer_artifact_sha256(source_root)
            != source_snapshot.artifact_sha256
        ):
            raise RuntimeError(
                "legacy tokenizer changed before private snapshot creation"
            )
        inventory = _legacy_tokenizer_inventory(source_root)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        for relative in inventory:
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(
                    f"tokenizer snapshot path is invalid: {relative}"
                )
            source = (source_root / relative).resolve()
            try:
                source.relative_to(source_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"tokenizer snapshot path escapes its root: {relative}"
                ) from exc
            if not source.is_file():
                raise FileNotFoundError(
                    f"tokenizer snapshot artifact is missing: {source}"
                )
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as source_stream, target.open(
                "wb"
            ) as target_stream:
                shutil.copyfileobj(
                    source_stream,
                    target_stream,
                    length=1024 * 1024,
                )
                target_stream.flush()
                os.fsync(target_stream.fileno())
        _fsync_regular_tree(temporary)
        _load_pinned_tokenizer(
            temporary,
            expected_sha256=source_snapshot.artifact_sha256,
            expected_vocab_size=source_snapshot.vocab_size,
        )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return private_snapshot


def _audit_pinned_geometry(
    geometry_dir: Path,
    geometry_contract: Mapping[str, Any],
    *,
    max_open_shards: int,
) -> frozenset[str]:
    """Coordinator-only full provenance audit without per-worker rehashing."""
    repository = GeometryRepository(
        geometry_dir,
        verify_checksums=True,
        validate_index_inventory=True,
        expected_manifest_sha256=str(
            geometry_contract["manifest_sha256"]
        ),
        expected_run_state_sha256=str(
            geometry_contract["run_state_sha256"]
        ),
        expected_index_metadata_sha256=str(
            geometry_contract["index_metadata_sha256"]
        ),
        expected_index_artifact_sha256=str(
            geometry_contract["index_artifact_sha256"]
        ),
        max_open_shards=max_open_shards,
    )
    try:
        repository.verify_all_artifacts()
        inventory = {
            "manifest.json",
            "run_state.json",
            "geometry_index.json",
            repository.index_name,
            *(
                descriptor.artifact_name
                for descriptor in repository._descriptors.values()
            ),
            *(
                descriptor.sidecar_path.name
                for descriptor in repository._descriptors.values()
            ),
        }
    finally:
        repository.close()
    return frozenset(inventory)


def _prepare_geometry_snapshot(
    source_dir: Path,
    destination: Path,
    geometry_contract: Mapping[str, Any],
    *,
    max_open_shards: int,
) -> Path:
    """Create or verify the exact private geometry inventory consumed."""
    poison_path = destination.parent / ".geometry-snapshot.poisoned.json"
    if poison_path.exists():
        raise RuntimeError(
            "geometry snapshot staging is poisoned and cannot be resumed; "
            f"remove the staged build and restart: {poison_path}"
        )
    if destination.exists() or destination.is_symlink():
        try:
            if _path_is_redirected(destination) or not destination.is_dir():
                raise RuntimeError(
                    f"geometry snapshot is not a private directory: "
                    f"{destination}"
                )
            inventory = _audit_pinned_geometry(
                destination,
                geometry_contract,
                max_open_shards=max_open_shards,
            )
            if {entry.name for entry in destination.iterdir()} != set(
                inventory
            ):
                raise RuntimeError(
                    "geometry snapshot file inventory differs from contract"
                )
        except BaseException as exc:
            atomic_write_json(
                poison_path,
                {
                    "schema": "semmol.pcqm_geometry_snapshot_poison.v1",
                    "reason": "snapshot_contract_mismatch",
                },
                overwrite=True,
            )
            raise RuntimeError(
                "staged geometry snapshot differs from the build contract"
            ) from exc
        return destination

    inventory = _audit_pinned_geometry(
        source_dir,
        geometry_contract,
        max_open_shards=max_open_shards,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        source_root = source_dir.resolve()
        for filename in sorted(inventory):
            source = (source_root / filename).resolve()
            if (
                not source.is_file()
                or source.parent != source_root
                or Path(filename).name != filename
            ):
                raise RuntimeError(
                    f"geometry snapshot inventory path is invalid: {filename}"
                )
            target = temporary / filename
            with source.open("rb") as source_stream, target.open(
                "wb"
            ) as target_stream:
                shutil.copyfileobj(
                    source_stream,
                    target_stream,
                    length=1024 * 1024,
                )
                target_stream.flush()
                os.fsync(target_stream.fileno())
        _fsync_regular_tree(temporary)
        _audit_pinned_geometry(
            temporary,
            geometry_contract,
            max_open_shards=max_open_shards,
        )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _remove_private_snapshots_before_publish(
    staging_path: Path,
    *,
    selection_snapshot: Path,
    geometry_snapshot: Path,
    tokenizer_snapshot: Path,
) -> None:
    """Remove private build inputs and prove none can enter final output."""
    if selection_snapshot.exists():
        if (
            _path_is_redirected(selection_snapshot)
            or not selection_snapshot.is_file()
        ):
            raise RuntimeError(
                f"invalid private selection snapshot: {selection_snapshot}"
            )
        selection_snapshot.unlink()
        _fsync_directory(staging_path)
    for snapshot in (geometry_snapshot, tokenizer_snapshot):
        if snapshot.exists():
            if _path_is_redirected(snapshot) or not snapshot.is_dir():
                raise RuntimeError(f"invalid private snapshot: {snapshot}")
            deletion_path = Path(
                tempfile.mkdtemp(
                    prefix=f".{snapshot.name}.delete-",
                    dir=staging_path,
                )
            )
            deletion_path.rmdir()
            os.replace(snapshot, deletion_path)
            _fsync_directory(staging_path)
            shutil.rmtree(deletion_path)
            _fsync_directory(staging_path)
    _cleanup_private_snapshot_temps(staging_path)

    forbidden = {
        selection_snapshot,
        geometry_snapshot,
        tokenizer_snapshot,
        staging_path / ".selection-snapshot.poisoned.json",
        staging_path / ".geometry-snapshot.poisoned.json",
        staging_path / ".tokenizer-snapshot.poisoned.json",
    }
    forbidden.update(
        candidate
        for pattern in _PRIVATE_SNAPSHOT_TEMP_PATTERNS
        for candidate in staging_path.glob(pattern)
    )
    remaining = sorted(
        str(path)
        for path in forbidden
        if path.exists() or path.is_symlink()
    )
    if remaining:
        raise RuntimeError(
            "private snapshot artifacts remain before publication: "
            f"{remaining}"
        )


def _validate_selection_provenance(
    metadata_path: Path,
    *,
    allow_unverified_split: bool,
    expected_generation_id: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> str:
    payload, metadata_sha256 = _read_pinned_json(
        metadata_path,
        label="selection metadata",
        expected_sha256=expected_sha256,
    )
    schema = str(payload.get("schema", ""))
    if schema != "semmol.pcqm_selection.v1":
        raise ValueError(
            f"unsupported selection metadata schema: {schema!r}"
        )
    generation_id = payload.get("generation_id")
    if expected_generation_id is not None and generation_id != expected_generation_id:
        raise ValueError(
            "selection metadata generation does not match the committed CURRENT "
            f"generation: expected={expected_generation_id!r}, "
            f"actual={generation_id!r}"
        )
    source = payload.get("input")
    if not isinstance(source, Mapping):
        raise ValueError("selection metadata has no input provenance")
    source_integrity = source.get("integrity")
    if (
        not isinstance(source_integrity, Mapping)
        or set(source_integrity) != {"size_bytes", "sha256"}
        or not isinstance(source_integrity.get("size_bytes"), int)
        or isinstance(source_integrity["size_bytes"], bool)
        or source_integrity["size_bytes"] < 0
        or not _is_sha256(source_integrity.get("sha256"))
    ):
        raise ValueError("selection metadata has invalid source integrity")
    split_column = source.get("official_split_column")
    if split_column is None and not allow_unverified_split:
        raise ValueError(
            "selection metadata does not prove official train-only filtering; "
            "use --allow-unverified-split only for an intentionally prefiltered input"
        )
    split_counts = source.get("official_split_counts", {})
    if split_column is not None and (
        not isinstance(split_counts, Mapping)
        or int(split_counts.get("train", 0)) <= 0
    ):
        raise ValueError(
            "selection metadata has no positive official train count"
        )
    return metadata_sha256


def _resolve_selection_artifacts(
    selection_input: os.PathLike[str] | str,
    selection_metadata: Optional[os.PathLike[str] | str],
    *,
    verify_integrity: bool,
) -> tuple[
    Path,
    Path,
    Optional[Path],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    """Resolve either a committed CURRENT pointer or explicit artifacts."""

    input_path = Path(selection_input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.name == "pcqm_selection_CURRENT.json":
        if selection_metadata is not None:
            raise ValueError(
                "selection_metadata must be omitted when using a CURRENT pointer"
            )
        from scripts.preprocess.filter_pcqm import resolve_pcqm_generation

        resolved = resolve_pcqm_generation(
            input_path,
            verify_integrity=verify_integrity,
        )
        return (
            resolved["manifest"].resolve(),
            resolved["metadata"].resolve(),
            resolved["current"].resolve(),
            str(resolved["current_sha256"]),
            str(resolved["manifest_sha256"]),
            str(resolved["metadata_sha256"]),
        )

    if input_path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError(
            "selection input must be a committed "
            "pcqm_selection_CURRENT.json or a Parquet manifest"
        )
    if selection_metadata is None:
        inferred = input_path.with_name("pcqm_selection_metadata.json")
        if not inferred.is_file():
            raise FileNotFoundError(
                "explicit selection Parquet requires --selection-metadata; "
                f"inferred path does not exist: {inferred}"
            )
        metadata_path = inferred
    else:
        metadata_path = Path(selection_metadata).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return input_path, metadata_path, None, None, None, None


def _failure_key(row: Mapping[str, Any]) -> tuple[int, str]:
    source_index = row.get("source_index")
    return (
        -1 if source_index is None else int(source_index),
        str(row.get("stage", "")),
    )


def _load_existing_failures(path: Path) -> tuple[set[tuple[int, str]], int]:
    if not path.is_file():
        return set(), 0
    records: dict[tuple[int, str], Mapping[str, Any]] = {}
    line_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            line_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid staged failure JSONL at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    f"failure JSONL line {line_number} is not an object"
                )
            records.setdefault(_failure_key(row), dict(row))
    if line_count != len(records):
        with atomic_output_path(path, overwrite=True) as temporary:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for row in records.values():
                    stream.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                stream.flush()
                os.fsync(stream.fileno())
    return set(records), len(records)


def _publish_views(
    staging_dir: Path,
    accumulator: SelectionAccumulator,
) -> dict[str, Mapping[str, Any]]:
    views_dir = staging_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Mapping[str, Any]] = {}
    for target, arrays in accumulator.views().items():
        short_name = (
            f"{target // 1_000_000}m"
            if target % 1_000_000 == 0
            else str(target)
        )
        filename = f"pcqm_{short_name}.npz"
        metadata = {
            "schema": VIEW_SCHEMA,
            "target_size": target,
            "nested_selection": True,
            "gap_bins": accumulator.n_bins,
        }
        payload = {
            **arrays,
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                dtype=np.str_,
            ),
        }
        destination = views_dir / filename
        _atomic_savez(destination, payload, overwrite=True)
        summaries[str(target)] = {
            "path": f"views/{filename}",
            "sha256": sha256_file(destination),
            "record_count": target,
        }
    return summaries


def _expected_store_metadata(
    *,
    record_count: int,
    records_per_shard: int,
    tokenizer_sha256: str,
    tokenizer_vocab_size: int,
) -> StoreMetadata:
    shard_count = math.ceil(record_count / records_per_shard)
    return StoreMetadata(
        schema_version=1,
        record_count=record_count,
        records_per_shard=records_per_shard,
        modalities=("1d", "2d", "3d", "qm"),
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_vocab_size=tokenizer_vocab_size,
        shards=tuple(
            f"shard-{shard_id:06d}.lmdb"
            for shard_id in range(shard_count)
        ),
    )


def _write_or_validate_store_metadata(
    staging_dir: Path,
    metadata: StoreMetadata,
) -> None:
    path = staging_dir / "store.json"
    expected = metadata.to_dict()
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(
                "existing staged store.json differs from final build metadata"
            )
        return
    write_store_metadata(staging_dir, metadata)


def _validate_ready_pcqm_staging(
    staging_dir: Path,
    *,
    contract: Mapping[str, Any],
    accumulator: SelectionAccumulator,
    metadata: StoreMetadata,
    failure_path: Path,
    failure_count: int,
) -> bool:
    manifest_path = staging_dir / "build-manifest.json"
    if not manifest_path.exists():
        return False
    if not manifest_path.is_file():
        raise RuntimeError("staged build-manifest.json is not a file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in contract.items()):
        raise RuntimeError(
            "existing complete PCQM manifest differs from the build contract"
        )
    expected_manifest_fields = set(contract) | {
        "status",
        "record_count",
        "shard_count",
        "attempted_candidates",
        "failed_candidates",
        "failure_rate",
        "density_extent_preflight",
        "failure_log",
        "views",
    }
    if set(manifest) != expected_manifest_fields:
        raise RuntimeError(
            "existing staged PCQM manifest inventory is not exact"
        )
    shard_count = len(metadata.shards)
    expected_attempts = accumulator.max_target + failure_count
    expected_failure_rate = (
        0.0
        if expected_attempts == 0
        else failure_count / expected_attempts
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("record_count") != accumulator.max_target
        or manifest.get("shard_count") != shard_count
        or manifest.get("attempted_candidates") != expected_attempts
        or manifest.get("failed_candidates") != failure_count
        or manifest.get("failure_rate") != expected_failure_rate
        or not isinstance(
            manifest.get("density_extent_preflight"),
            Mapping,
        )
    ):
        raise RuntimeError("existing staged PCQM manifest is incomplete")
    _write_or_validate_store_metadata(staging_dir, metadata)
    failure_descriptor = manifest.get("failure_log")
    if (
        not isinstance(failure_descriptor, Mapping)
        or set(failure_descriptor) != {
            "path",
            "sha256",
            "record_count",
        }
        or failure_descriptor.get("path") != failure_path.name
        or failure_descriptor.get("record_count") != failure_count
        or not failure_path.is_file()
        or failure_descriptor.get("sha256") != sha256_file(failure_path)
    ):
        raise RuntimeError("existing staged PCQM failure log is inconsistent")

    expected_views = accumulator.views()
    raw_views = manifest.get("views")
    if (
        not isinstance(raw_views, Mapping)
        or set(raw_views) != {str(target) for target in expected_views}
    ):
        raise RuntimeError("existing staged PCQM view inventory is invalid")
    for target, expected_arrays in expected_views.items():
        descriptor = raw_views[str(target)]
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "sha256", "record_count"}
        ):
            raise RuntimeError(f"invalid staged PCQM view target={target}")
        short_name = (
            f"{target // 1_000_000}m"
            if target % 1_000_000 == 0
            else str(target)
        )
        expected_relative = f"views/pcqm_{short_name}.npz"
        path = (staging_dir / expected_relative).resolve()
        if (
            descriptor.get("path") != expected_relative
            or descriptor.get("record_count") != target
            or not path.is_file()
            or descriptor.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(
                f"existing staged PCQM view target={target} is inconsistent"
            )
        with np.load(path, allow_pickle=False) as archive:
            expected_fields = set(expected_arrays) | {"metadata_json"}
            if set(archive.files) != expected_fields:
                raise RuntimeError(
                    f"existing staged PCQM view target={target} has bad schema"
                )
            for field, expected in expected_arrays.items():
                if not np.array_equal(np.asarray(archive[field]), expected):
                    raise RuntimeError(
                        f"existing staged PCQM view target={target} "
                        f"differs at {field}"
                    )
            view_metadata = json.loads(str(archive["metadata_json"].item()))
        if view_metadata != {
            "schema": VIEW_SCHEMA,
            "target_size": target,
            "nested_selection": True,
            "gap_bins": accumulator.n_bins,
        }:
            raise RuntimeError(
                f"existing staged PCQM view target={target} metadata differs"
            )
    return True


def build_pcqm_store(
    *,
    selection_manifest: os.PathLike[str] | str,
    selection_metadata: Optional[os.PathLike[str] | str],
    geometry_dir: os.PathLike[str] | str,
    tokenizer_dir: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    target_sizes: Sequence[int],
    n_bins: int,
    records_per_shard: int,
    lmdb_map_size: int,
    compression_level: int,
    commit_interval: int,
    feature_config: FeatureBuildConfig,
    workers: int,
    work_batch_size: int,
    max_open_geometry_shards: int,
    max_failure_rate: float,
    resume: bool,
    verify_checksums: bool,
    allow_unverified_split: bool,
) -> Path:
    """Build, resume at shard boundaries, validate, then atomically publish."""

    (
        selection_path,
        selection_metadata_path,
        selection_pointer_path,
        selection_pointer_sha256,
        committed_selection_manifest_sha256,
        committed_selection_metadata_sha256,
    ) = _resolve_selection_artifacts(
        selection_manifest,
        selection_metadata,
        verify_integrity=verify_checksums,
    )
    geometry_path = Path(geometry_dir).resolve()
    tokenizer_path = Path(tokenizer_dir).resolve()
    final_path = Path(output_dir).resolve()
    for required_dir in (geometry_path, tokenizer_path):
        if not required_dir.is_dir():
            raise FileNotFoundError(required_dir)
    if final_path.exists():
        raise FileExistsError(
            f"final store already exists and will not be overwritten: {final_path}"
        )
    if workers <= 0 or work_batch_size <= 0:
        raise ValueError("workers and work_batch_size must be positive")
    if records_per_shard <= 0 or commit_interval <= 0:
        raise ValueError("storage shard and commit sizes must be positive")
    if lmdb_map_size < 1024 * 1024:
        raise ValueError("lmdb_map_size must be at least 1 MiB")
    if not 0.0 <= max_failure_rate < 1.0:
        raise ValueError("max_failure_rate must be in [0, 1)")

    targets = tuple(sorted({int(size) for size in target_sizes}))
    accumulator = SelectionAccumulator(targets, n_bins=n_bins)
    tokenizer_snapshot = resolve_tokenizer_snapshot(tokenizer_path)
    selection_manifest_sha256 = (
        sha256_file(selection_path)
        if committed_selection_manifest_sha256 is None
        else committed_selection_manifest_sha256
    )
    if not _is_sha256(selection_manifest_sha256):
        raise RuntimeError("selection manifest SHA256 is invalid")
    selection_metadata_sha256 = _validate_selection_provenance(
        selection_metadata_path,
        allow_unverified_split=allow_unverified_split,
        expected_generation_id=(
            selection_path.parent.name
            if selection_pointer_path is not None
            else None
        ),
        expected_sha256=committed_selection_metadata_sha256,
    )
    contract = _contract_payload(
        selection_manifest=selection_path,
        selection_manifest_sha256=selection_manifest_sha256,
        selection_metadata=selection_metadata_path,
        selection_metadata_sha256=selection_metadata_sha256,
        tokenizer_snapshot=tokenizer_snapshot,
        geometry_dir=geometry_path,
        target_sizes=targets,
        n_bins=n_bins,
        records_per_shard=records_per_shard,
        map_size=lmdb_map_size,
        compression_level=compression_level,
        feature_config=feature_config,
    )
    if selection_pointer_path is not None:
        if not _is_sha256(selection_pointer_sha256):
            raise RuntimeError("selection CURRENT SHA256 is invalid")
        contract["selection_current"] = {
            "path": str(selection_pointer_path),
            "sha256": selection_pointer_sha256,
        }

    staging_path = final_path.parent / f".{final_path.name}.building"
    contract_path = staging_path / "build-contract.json"
    selection_snapshot_path = (
        staging_path / ".selection-manifest.snapshot.parquet"
    )
    selection_snapshot_poison_path = (
        staging_path / ".selection-snapshot.poisoned.json"
    )
    geometry_snapshot_path = staging_path / ".geometry-input.snapshot"
    geometry_snapshot_poison_path = (
        staging_path / ".geometry-snapshot.poisoned.json"
    )
    tokenizer_private_snapshot_path = (
        staging_path / ".tokenizer-generation.snapshot"
    )
    tokenizer_snapshot_poison_path = (
        staging_path / ".tokenizer-snapshot.poisoned.json"
    )
    if resume:
        if not staging_path.exists() and not staging_path.is_symlink():
            raise FileNotFoundError(
                f"no resumable staged build exists at {staging_path}"
            )
        _assert_staging_tree_not_redirected(staging_path)
        if not contract_path.is_file():
            raise FileNotFoundError(
                f"staged build contract is missing at {contract_path}"
            )
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != contract:
            raise RuntimeError(
                "resume contract differs from the staged input/configuration"
            )
        poison_paths = (
            selection_snapshot_poison_path,
            geometry_snapshot_poison_path,
            tokenizer_snapshot_poison_path,
        )
        if any(path.exists() for path in poison_paths):
            raise RuntimeError(
                "private input snapshot staging is poisoned and cannot be "
                "resumed; remove the staged build and restart"
            )
    else:
        if staging_path.exists():
            raise FileExistsError(
                f"staged build already exists; use --resume or inspect it: "
                f"{staging_path}"
            )
        staging_path.mkdir(parents=True)
        _fsync_directory(final_path.parent)
        atomic_write_json(contract_path, contract)

    _assert_staging_tree_not_redirected(staging_path)
    _cleanup_private_snapshot_temps(staging_path)
    _cleanup_known_build_temporaries(staging_path)
    selection_consumption_path = _prepare_selection_snapshot(
        selection_path,
        selection_snapshot_path,
        expected_sha256=contract["selection_manifest"]["sha256"],
    )
    geometry_consumption_path = _prepare_geometry_snapshot(
        geometry_path,
        geometry_snapshot_path,
        contract["geometry"],
        max_open_shards=max_open_geometry_shards,
    )
    tokenizer_consumption_snapshot = _prepare_tokenizer_snapshot(
        tokenizer_snapshot,
        tokenizer_private_snapshot_path,
    )
    if resume:
        _restore_completed_indices(
            staging_path,
            accumulator=accumulator,
            verify_checksums=verify_checksums,
        )
    failure_path = staging_path / "feature-failures.jsonl"
    failure_keys, failure_count = _load_existing_failures(failure_path)
    failure_mode = "a" if failure_path.exists() else "w"
    final_store_metadata = _expected_store_metadata(
        record_count=accumulator.max_target,
        records_per_shard=records_per_shard,
        tokenizer_sha256=tokenizer_snapshot.artifact_sha256,
        tokenizer_vocab_size=tokenizer_snapshot.vocab_size,
    )
    if (
        (staging_path / "build-manifest.json").exists()
        and accumulator.accepted_count != accumulator.max_target
    ):
        raise RuntimeError(
            "staged complete manifest exists without a complete shard prefix"
        )
    if accumulator.accepted_count == accumulator.max_target:
        accumulator.validate_complete()
        if _validate_ready_pcqm_staging(
            staging_path,
            contract=contract,
            accumulator=accumulator,
            metadata=final_store_metadata,
            failure_path=failure_path,
            failure_count=failure_count,
        ):
            _prepare_geometry_snapshot(
                geometry_path,
                geometry_snapshot_path,
                contract["geometry"],
                max_open_shards=max_open_geometry_shards,
            )
            _prepare_tokenizer_snapshot(
                tokenizer_snapshot,
                tokenizer_private_snapshot_path,
            )
            _assert_staging_tree_not_redirected(staging_path)
            _remove_private_snapshots_before_publish(
                staging_path,
                selection_snapshot=selection_snapshot_path,
                geometry_snapshot=geometry_snapshot_path,
                tokenizer_snapshot=tokenizer_private_snapshot_path,
            )
            _cleanup_known_build_temporaries(staging_path)
            _assert_no_private_staging_entries(staging_path)
            _assert_staging_tree_not_redirected(staging_path)
            _assert_final_staging_inventory(
                staging_path,
                metadata=final_store_metadata,
                view_paths=_expected_pcqm_view_paths(
                    accumulator.target_sizes
                ),
            )
            os.replace(staging_path, final_path)
            _fsync_directory(final_path.parent)
            return final_path
    preflight_repository = GeometryRepository(
        geometry_consumption_path,
        verify_checksums=True,
        validate_index_inventory=True,
        expected_manifest_sha256=contract["geometry"]["manifest_sha256"],
        expected_run_state_sha256=contract["geometry"]["run_state_sha256"],
        expected_index_metadata_sha256=(
            contract["geometry"]["index_metadata_sha256"]
        ),
        expected_index_artifact_sha256=(
            contract["geometry"]["index_artifact_sha256"]
        ),
        max_open_shards=max_open_geometry_shards,
    )
    try:
        preflight_repository.verify_all_artifacts()
        density_extent_preflight = _preflight_density_extent(
            selection_path=selection_consumption_path,
            repository=preflight_repository,
            accumulator=accumulator,
            feature_config=feature_config,
            batch_size=work_batch_size,
            prior_failure_count=failure_count,
            max_failure_rate=max_failure_rate,
        )
    finally:
        preflight_repository.close()
    del preflight_repository

    codec = RecordCodec(compression_level=compression_level)
    next_shard_id = math.ceil(
        accumulator.accepted_count / records_per_shard
    )
    if (
        accumulator.accepted_count % records_per_shard
        and accumulator.accepted_count != accumulator.max_target
    ):
        raise RuntimeError(
            "resume prefix ends with a partial shard; staged build is inconsistent"
        )
    current_writer: Optional[LmdbShardWriter] = None
    current_shard_accepted: list[AcceptedCandidate] = []
    attempted_count = accumulator.accepted_count + failure_count

    pool: Optional[Pool] = None
    # The preflight owns the configured checksum pass. Workers still validate
    # the v2 numeric index schema without rereading each large artifact once
    # per process.
    worker_verify_checksums = False
    if workers == 1:
        _initialize_worker(
            str(tokenizer_consumption_snapshot.load_path),
            tokenizer_consumption_snapshot.artifact_sha256,
            tokenizer_consumption_snapshot.vocab_size,
            str(geometry_consumption_path),
            contract["geometry"]["manifest_sha256"],
            contract["geometry"]["run_state_sha256"],
            contract["geometry"]["index_metadata_sha256"],
            contract["geometry"]["index_artifact_sha256"],
            feature_config.to_dict(),
            worker_verify_checksums,
            max_open_geometry_shards,
        )
    else:
        pool = Pool(
            processes=workers,
            initializer=_initialize_worker,
            initargs=(
                str(tokenizer_consumption_snapshot.load_path),
                tokenizer_consumption_snapshot.artifact_sha256,
                tokenizer_consumption_snapshot.vocab_size,
                str(geometry_consumption_path),
                contract["geometry"]["manifest_sha256"],
                contract["geometry"]["run_state_sha256"],
                contract["geometry"]["index_metadata_sha256"],
                contract["geometry"]["index_artifact_sha256"],
                feature_config.to_dict(),
                worker_verify_checksums,
                max_open_geometry_shards,
            ),
        )

    try:
        with failure_path.open(
            failure_mode,
            encoding="utf-8",
            newline="\n",
        ) as failure_stream:
            for gap_bin, raw_batch in _selection_batches(
                selection_consumption_path,
                batch_size=work_batch_size,
                n_bins=n_bins,
            ):
                if accumulator.bin_is_full(gap_bin):
                    continue
                last_completed_rank = accumulator.last_selection_rank(gap_bin)
                batch = [
                    candidate
                    for candidate in raw_batch
                    if candidate.selection_rank > last_completed_rank
                ]
                if not batch:
                    continue
                results = (
                    list(map(_build_candidate, batch))
                    if pool is None
                    else pool.map(_build_candidate, batch)
                )
                for candidate, record, failure in results:
                    if accumulator.bin_is_full(gap_bin):
                        break
                    if failure is not None:
                        key = _failure_key(failure)
                        if key not in failure_keys:
                            attempted_count += 1
                            failure_stream.write(
                                json.dumps(
                                    failure,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            failure_keys.add(key)
                            failure_count += 1
                        continue
                    if record is None:
                        raise RuntimeError(
                            "worker returned neither a record nor a failure"
                        )
                    attempted_count += 1
                    record_index = accumulator.accepted_count
                    accepted = accumulator.accept(
                        source_index=candidate.source_index,
                        gap_bin=candidate.gap_bin,
                        selection_rank=candidate.selection_rank,
                        record_index=record_index,
                    )
                    record["success_rank"] = accepted.success_rank
                    if current_writer is None:
                        remaining = accumulator.max_target - record_index
                        expected = min(records_per_shard, remaining)
                        current_writer = LmdbShardWriter(
                            store_dir=staging_path,
                            shard_id=next_shard_id,
                            start_index=record_index,
                            expected_records=expected,
                            map_size=lmdb_map_size,
                            codec=codec,
                            commit_interval=commit_interval,
                        )
                        current_shard_accepted = []
                    current_writer.put(record_index, record)
                    current_shard_accepted.append(accepted)
                    if (
                        current_writer.record_count
                        == current_writer.expected_records
                    ):
                        index_path = _write_build_index(
                            staging_path,
                            next_shard_id,
                            current_shard_accepted,
                        )
                        try:
                            current_writer.finalize()
                        except BaseException:
                            if (
                                index_path.exists()
                                and index_path.parent == staging_path
                            ):
                                index_path.unlink()
                            current_writer.abort()
                            raise
                        current_writer = None
                        current_shard_accepted = []
                        next_shard_id += 1
                        failure_stream.flush()
                        os.fsync(failure_stream.fileno())

            accumulator.validate_complete()
            if current_writer is not None:
                raise RuntimeError(
                    "completed target left an unfinalized shard"
                )
            failure_stream.flush()
            os.fsync(failure_stream.fileno())
    except BaseException:
        if current_writer is not None:
            current_writer.abort()
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _release_local_worker_state()

    if (
        sha256_file(selection_consumption_path)
        != contract["selection_manifest"]["sha256"]
    ):
        atomic_write_json(
            staging_path / ".selection-snapshot.poisoned.json",
            {
                "schema": "semmol.pcqm_selection_snapshot_poison.v1",
                "reason": "snapshot_changed_during_feature_construction",
                "expected_sha256": contract["selection_manifest"]["sha256"],
            },
            overwrite=True,
        )
        raise RuntimeError(
            "private selection snapshot changed during feature construction"
        )
    effective_failure_rate = (
        0.0 if attempted_count == 0 else failure_count / attempted_count
    )
    if effective_failure_rate > max_failure_rate:
        raise RuntimeError(
            f"feature failure rate {effective_failure_rate:.6f} exceeds "
            f"limit {max_failure_rate:.6f}; staged build is preserved"
        )

    _prepare_tokenizer_snapshot(
        tokenizer_snapshot,
        tokenizer_private_snapshot_path,
    )
    shard_count = len(final_store_metadata.shards)
    _write_or_validate_store_metadata(
        staging_path,
        final_store_metadata,
    )
    views = _publish_views(staging_path, accumulator)
    _prepare_geometry_snapshot(
        geometry_path,
        geometry_snapshot_path,
        contract["geometry"],
        max_open_shards=max_open_geometry_shards,
    )
    _prepare_tokenizer_snapshot(
        tokenizer_snapshot,
        tokenizer_private_snapshot_path,
    )
    build_manifest = {
        **contract,
        "status": "complete",
        "record_count": accumulator.max_target,
        "shard_count": shard_count,
        "attempted_candidates": attempted_count,
        "failed_candidates": failure_count,
        "failure_rate": effective_failure_rate,
        "density_extent_preflight": density_extent_preflight,
        "failure_log": {
            "path": failure_path.name,
            "sha256": sha256_file(failure_path),
            "record_count": failure_count,
        },
        "views": views,
    }
    atomic_write_json(
        staging_path / "build-manifest.json",
        build_manifest,
    )
    if not _validate_ready_pcqm_staging(
        staging_path,
        contract=contract,
        accumulator=accumulator,
        metadata=final_store_metadata,
        failure_path=failure_path,
        failure_count=failure_count,
    ):
        raise RuntimeError(
            "completed PCQM staging did not pass final validation"
        )
    _assert_staging_tree_not_redirected(staging_path)
    _remove_private_snapshots_before_publish(
        staging_path,
        selection_snapshot=selection_snapshot_path,
        geometry_snapshot=geometry_snapshot_path,
        tokenizer_snapshot=tokenizer_private_snapshot_path,
    )
    _cleanup_known_build_temporaries(staging_path)
    _assert_no_private_staging_entries(staging_path)
    _assert_staging_tree_not_redirected(staging_path)
    _assert_final_staging_inventory(
        staging_path,
        metadata=final_store_metadata,
        view_paths=_expected_pcqm_view_paths(accumulator.target_sizes),
    )
    os.replace(staging_path, final_path)
    _fsync_directory(final_path.parent)
    return final_path


def _parse_target_sizes(value: str) -> tuple[int, ...]:
    try:
        targets = tuple(
            sorted(
                {
                    int(part.strip())
                    for part in value.split(",")
                    if part.strip()
                }
            )
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target sizes must be comma-separated integers"
        ) from exc
    if not targets or any(target <= 0 for target in targets):
        raise argparse.ArgumentTypeError(
            "target sizes must be positive comma-separated integers"
        )
    return targets


def _parse_optional_float(value: str) -> Optional[float]:
    normalized = value.strip().lower()
    if normalized in {"element", "none"}:
        return None
    try:
        parsed = float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "atomic sigma must be a positive float or 'element'"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "atomic sigma must be a positive finite float"
        )
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a safe sharded PCQM multimodal store with exact nested "
            "failure-backfilled views."
        )
    )
    parser.add_argument(
        "--selection-manifest",
        default=(
            "data/processed/pcqm/manifests/"
            "pcqm_selection_CURRENT.json"
        ),
    )
    parser.add_argument(
        "--selection-metadata",
        default=None,
        help="required only when --selection-manifest is a direct Parquet file",
    )
    parser.add_argument(
        "--geometry-dir",
        default="data/processed/pcqm/geometry",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default="data/processed/pcqm/tokenizer",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/pcqm/store",
    )
    parser.add_argument(
        "--target-sizes",
        type=_parse_target_sizes,
        default=(1_000_000, 3_000_000),
    )
    parser.add_argument("--gap-bins", type=int, default=10)
    parser.add_argument("--records-per-shard", type=int, default=8192)
    parser.add_argument(
        "--lmdb-map-size-gib",
        type=float,
        default=4.0,
    )
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--commit-interval", type=int, default=256)
    parser.add_argument("--max-smiles-length", type=int, default=256)
    parser.add_argument("--generated-conformers", type=int, default=3)
    parser.add_argument("--prune-rms-threshold", type=float, default=0.5)
    parser.add_argument("--geometry-seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--grid-spacing", type=float, default=0.75)
    parser.add_argument("--grid-padding", type=float, default=4.0)
    parser.add_argument(
        "--atomic-sigma",
        type=_parse_optional_float,
        default=None,
    )
    parser.add_argument(
        "--density-conformer-index",
        default="0",
        help="non-negative conformer index or 'mean' after heavy-atom alignment",
    )
    parser.add_argument(
        "--density-storage-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--work-batch-size", type=int, default=2048)
    parser.add_argument("--max-open-geometry-shards", type=int, default=8)
    parser.add_argument("--max-failure-rate", type=float, default=0.005)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-verify-checksums", action="store_true")
    parser.add_argument("--allow-unverified-split", action="store_true")
    parser.add_argument("--no-optimize-geometry", action="store_true")
    parser.add_argument("--allow-density-overflow", action="store_true")
    parser.add_argument(
        "--continuous-density-normalization",
        action="store_true",
    )
    args = parser.parse_args(argv)
    if args.density_conformer_index.strip().lower() == "mean":
        args.density_conformer_index = None
    else:
        try:
            args.density_conformer_index = int(
                args.density_conformer_index
            )
        except ValueError:
            parser.error(
                "--density-conformer-index must be non-negative or 'mean'"
            )
        if args.density_conformer_index < 0:
            parser.error(
                "--density-conformer-index must be non-negative or 'mean'"
            )
    if not math.isfinite(args.lmdb_map_size_gib) or args.lmdb_map_size_gib <= 0:
        parser.error("--lmdb-map-size-gib must be positive and finite")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    feature_config = FeatureBuildConfig(
        max_smiles_length=args.max_smiles_length,
        generated_conformers=args.generated_conformers,
        prune_rms_threshold=args.prune_rms_threshold,
        geometry_seed=args.geometry_seed,
        optimize_geometry=not args.no_optimize_geometry,
        grid_size=args.grid_size,
        grid_spacing=args.grid_spacing,
        grid_padding=args.grid_padding,
        atomic_sigma=args.atomic_sigma,
        density_conformer_index=args.density_conformer_index,
        strict_density_bounds=not args.allow_density_overflow,
        discrete_density_normalization=(
            not args.continuous_density_normalization
        ),
        density_storage_dtype=args.density_storage_dtype,
    )
    output = build_pcqm_store(
        selection_manifest=args.selection_manifest,
        selection_metadata=args.selection_metadata,
        geometry_dir=args.geometry_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        target_sizes=args.target_sizes,
        n_bins=args.gap_bins,
        records_per_shard=args.records_per_shard,
        lmdb_map_size=int(args.lmdb_map_size_gib * 1024**3),
        compression_level=args.compression_level,
        commit_interval=args.commit_interval,
        feature_config=feature_config,
        workers=args.workers,
        work_batch_size=args.work_batch_size,
        max_open_geometry_shards=args.max_open_geometry_shards,
        max_failure_rate=args.max_failure_rate,
        resume=args.resume,
        verify_checksums=not args.no_verify_checksums,
        allow_unverified_split=args.allow_unverified_split,
    )
    print(json.dumps({"store": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

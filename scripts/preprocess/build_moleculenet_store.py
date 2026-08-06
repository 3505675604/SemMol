"""Build safe four-modality stores for the nine MoleculeNet benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.datasets.feature_building import (  # noqa: E402
    FeatureBuildConfig,
    FeatureBuildError,
    MultimodalFeatureBuilder,
    tokenizer_artifact_sha256,
)
from src.datasets.moleculenet_dataset import (  # noqa: E402
    MOLECULENET_REGISTRY,
    MoleculeNetRows,
    extract_moleculenet_rows,
    get_moleculenet_spec,
    resolve_moleculenet_csv,
)
from src.datasets.scaffold_split import (  # noqa: E402
    generate_scaffold,
    load_scaffold_split,
)
from src.datasets.storage import (  # noqa: E402
    LmdbShardWriter,
    RecordCodec,
    ShardedRecordStore,
    StoreMetadata,
    recover_published_shard_sidecar,
    write_store_metadata,
)
from src.molecular.espf_tokenizer import (  # noqa: E402
    ARTIFACT_MANIFEST_NAME,
    ESPFTokenizer,
)
from src.utils.io import (  # noqa: E402
    atomic_output_path,
    atomic_write_json,
    sha256_file,
)


BUILD_SCHEMA = "semmol.moleculenet_store_build.v1"
BUILD_INDEX_SCHEMA = "semmol.moleculenet_store_shard_index.v1"
VIEW_SCHEMA = "semmol.moleculenet_view.v1"
SPLIT_NAMES = ("train", "valid", "test")
TOKENIZER_SNAPSHOT_DELETE_PREFIX = ".tokenizer-snapshot.delete-"


def _fsync_directory(path: Path) -> None:
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


def _path_is_redirected(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or (callable(is_junction) and is_junction())
        or path.resolve() != path.absolute()
    )


def _fsync_regular_tree(root: Path) -> None:
    """Persist every regular file and directory entry in a private snapshot."""

    if not root.is_dir() or _path_is_redirected(root):
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


def _assert_staging_tree_not_redirected(staging_dir: Path) -> None:
    """Reject links and junctions before staged state is read or mutated."""

    if not staging_dir.is_dir() or _path_is_redirected(staging_dir):
        raise RuntimeError(
            f"MoleculeNet staging root is missing or redirected: "
            f"{staging_dir}"
        )
    pending = [staging_dir]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                children = list(entries)
        except OSError as exc:
            raise RuntimeError(
                f"cannot audit MoleculeNet staging directory: {directory}"
            ) from exc
        for entry in children:
            candidate = Path(entry.path)
            is_junction = getattr(candidate, "is_junction", None)
            if entry.is_symlink() or (
                callable(is_junction) and is_junction()
            ):
                raise RuntimeError(
                    f"MoleculeNet staging entry is redirected: {candidate}"
                )
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(
                    f"cannot inspect MoleculeNet staging entry: {candidate}"
                ) from exc
            if is_directory:
                pending.append(candidate)


@dataclass(frozen=True)
class MoleculeNetTask:
    dataset_name: str
    source_index: int
    smiles: str
    labels: np.ndarray
    label_mask: np.ndarray
    task_type: str
    label_columns: tuple[str, ...]


@dataclass(frozen=True)
class TokenizerSnapshot:
    root: Path
    load_path: Path
    artifact_sha256: str
    vocab_size: int


def _verify_generation_directory(path: Path, expected_sha256: str) -> None:
    manifest_path = path / ARTIFACT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"tokenizer generation manifest is missing: {manifest_path}"
        )
    if sha256_file(manifest_path) != expected_sha256:
        raise RuntimeError("tokenizer generation ID/hash mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        ):
            raise RuntimeError("invalid tokenizer artifact descriptor")
        artifact = path / name
        if (
            not artifact.is_file()
            or int(descriptor.get("size", -1)) != artifact.stat().st_size
            or str(descriptor.get("sha256", "")) != sha256_file(artifact)
        ):
            raise RuntimeError(f"tokenizer artifact integrity failure: {artifact}")


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


def _cleanup_private_tokenizer_transients(staging_dir: Path) -> None:
    if not staging_dir.is_dir():
        raise RuntimeError(
            f"tokenizer snapshot staging directory is missing: {staging_dir}"
        )
    candidates: list[Path] = list(
        staging_dir.glob(".tokenizer-snapshot.tmp-*")
    )
    candidates.extend(
        staging_dir.glob(f"{TOKENIZER_SNAPSHOT_DELETE_PREFIX}*")
    )
    removed = False
    resolved_staging = staging_dir.resolve()
    for candidate in candidates:
        if candidate.parent != staging_dir:
            raise RuntimeError(
                f"tokenizer snapshot transient escaped staging: {candidate}"
            )
        if candidate.name.startswith(TOKENIZER_SNAPSHOT_DELETE_PREFIX):
            suffix = candidate.name.removeprefix(
                TOKENIZER_SNAPSHOT_DELETE_PREFIX
            )
            if (
                len(suffix) != 32
                or set(suffix) - set("0123456789abcdef")
            ):
                raise RuntimeError(
                    f"invalid tokenizer snapshot tombstone: {candidate}"
                )
        if candidate.is_symlink():
            raise RuntimeError(
                f"tokenizer snapshot transient is redirected: {candidate}"
            )
        if candidate.is_dir():
            if candidate.resolve().parent != resolved_staging:
                raise RuntimeError(
                    f"tokenizer snapshot transient escaped staging: {candidate}"
                )
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()
        removed = True
    for temporary in staging_dir.glob(
        "..tokenizer-snapshot.poisoned.json.tmp-*"
    ):
        if temporary.is_dir() and not temporary.is_symlink():
            raise RuntimeError(
                f"tokenizer poison temporary is a directory: {temporary}"
            )
        temporary.unlink(missing_ok=True)
        removed = True
    if removed:
        _fsync_directory(staging_dir)


def _prepare_private_tokenizer_snapshot(
    source: TokenizerSnapshot,
    destination: Path,
) -> TokenizerSnapshot:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_private_tokenizer_transients(destination.parent)
    poison_path = destination.parent / ".tokenizer-snapshot.poisoned.json"
    if poison_path.exists():
        raise RuntimeError(
            "tokenizer snapshot staging is poisoned and cannot be resumed"
        )
    if destination.exists() or destination.is_symlink():
        try:
            if _path_is_redirected(destination) or not destination.is_dir():
                raise RuntimeError(
                    f"private tokenizer snapshot is redirected: {destination}"
                )
            tokenizer = _load_pinned_tokenizer(
                destination,
                expected_sha256=source.artifact_sha256,
                expected_vocab_size=source.vocab_size,
            )
        except BaseException:
            atomic_write_json(
                poison_path,
                {
                    "schema": "semmol.tokenizer_snapshot_poison.v1",
                    "reason": "snapshot_integrity_failure",
                    "expected_sha256": source.artifact_sha256,
                },
                overwrite=True,
            )
            raise
        if int(tokenizer.vocab_size) != source.vocab_size:
            raise RuntimeError("private tokenizer snapshot vocabulary mismatch")
        return TokenizerSnapshot(
            root=source.root,
            load_path=destination,
            artifact_sha256=source.artifact_sha256,
            vocab_size=source.vocab_size,
        )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=".tokenizer-snapshot.tmp-",
            dir=destination.parent,
        )
    )
    temporary_snapshot = temporary_root / "payload"
    try:
        shutil.copytree(source.load_path, temporary_snapshot)
        _fsync_regular_tree(temporary_snapshot)
        tokenizer = _load_pinned_tokenizer(
            temporary_snapshot,
            expected_sha256=source.artifact_sha256,
            expected_vocab_size=source.vocab_size,
        )
        if int(tokenizer.vocab_size) != source.vocab_size:
            raise RuntimeError("copied tokenizer snapshot vocabulary mismatch")
        os.replace(temporary_snapshot, destination)
        _fsync_directory(destination.parent)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return TokenizerSnapshot(
        root=source.root,
        load_path=destination,
        artifact_sha256=source.artifact_sha256,
        vocab_size=source.vocab_size,
    )


def _remove_private_tokenizer_snapshot(path: Path) -> None:
    staging_dir = path.parent
    expected = staging_dir / ".tokenizer.snapshot"
    if path != expected:
        raise RuntimeError(
            f"refusing to remove unexpected tokenizer snapshot: {path}"
        )
    _cleanup_private_tokenizer_transients(staging_dir)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"invalid private tokenizer snapshot: {path}")
    tombstone: Optional[Path] = None
    for _attempt in range(16):
        candidate = staging_dir / (
            f"{TOKENIZER_SNAPSHOT_DELETE_PREFIX}{uuid.uuid4().hex}"
        )
        if not candidate.exists() and not candidate.is_symlink():
            tombstone = candidate
            break
    if tombstone is None:
        raise RuntimeError("cannot allocate tokenizer snapshot tombstone")
    os.replace(path, tombstone)
    _fsync_directory(staging_dir)
    try:
        shutil.rmtree(tombstone)
        _fsync_directory(staging_dir)
    except OSError as exc:
        raise RuntimeError(
            "retired tokenizer snapshot cleanup did not complete; resume "
            "will converge the tombstone"
        ) from exc


def _cleanup_known_build_temporaries(staging_dir: Path) -> None:
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
    for candidate in list(staging_dir.iterdir()):
        if shard_directory_pattern.fullmatch(candidate.name):
            if (
                candidate.is_symlink()
                or not candidate.is_dir()
                or candidate.resolve().parent != staging_dir.resolve()
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
            candidate.unlink(missing_ok=True)
            removed_root = True
    if removed_root:
        _fsync_directory(staging_dir)

    views_dir = staging_dir / "views"
    if views_dir.is_dir() and not views_dir.is_symlink():
        removed_views = False
        view_temporary = re.compile(
            r"\.(?:train|valid|test)\.npz\.tmp-.+"
        )
        for candidate in list(views_dir.iterdir()):
            if not view_temporary.fullmatch(candidate.name):
                continue
            if candidate.is_dir() and not candidate.is_symlink():
                raise RuntimeError(
                    f"view atomic temporary is a directory: {candidate}"
                )
            candidate.unlink(missing_ok=True)
            removed_views = True
        if removed_views:
            _fsync_directory(views_dir)


def _assert_no_private_staging_entries(staging_dir: Path) -> None:
    hidden = sorted(
        path.name for path in staging_dir.iterdir() if path.name.startswith(".")
    )
    if hidden:
        raise RuntimeError(
            f"private MoleculeNet staging artifacts remain: {hidden}"
        )


def _assert_final_staging_inventory(
    staging_dir: Path,
    *,
    metadata: StoreMetadata,
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
    actual_root = {entry.name for entry in staging_dir.iterdir()}
    if actual_root != expected_root:
        raise RuntimeError(
            "final MoleculeNet staging inventory is not exact: "
            f"missing={sorted(expected_root - actual_root)}, "
            f"unknown={sorted(actual_root - expected_root)}"
        )
    views_dir = staging_dir / "views"
    expected_views = {f"{split_name}.npz" for split_name in SPLIT_NAMES}
    actual_views = {entry.name for entry in views_dir.iterdir()}
    if actual_views != expected_views:
        raise RuntimeError(
            "final MoleculeNet view inventory is not exact: "
            f"missing={sorted(expected_views - actual_views)}, "
            f"unknown={sorted(actual_views - expected_views)}"
        )
    for shard_name in metadata.shards:
        shard_dir = staging_dir / shard_name
        shard_inventory = {entry.name for entry in shard_dir.iterdir()}
        if shard_inventory != {"data.mdb", "lock.mdb"}:
            raise RuntimeError(
                f"final LMDB inventory is not exact for {shard_name}: "
                f"{sorted(shard_inventory)}"
            )


def _load_pinned_tokenizer(
    snapshot_path: os.PathLike[str] | str,
    *,
    expected_sha256: str,
    expected_vocab_size: int,
) -> ESPFTokenizer:
    path = Path(snapshot_path).resolve()
    if (path / ARTIFACT_MANIFEST_NAME).is_file():
        _verify_generation_directory(path, expected_sha256)
        tokenizer = ESPFTokenizer.from_pretrained(path)
        _verify_generation_directory(path, expected_sha256)
    else:
        before = tokenizer_artifact_sha256(path)
        if before != expected_sha256:
            raise RuntimeError("legacy tokenizer changed before worker initialization")
        tokenizer = ESPFTokenizer.from_pretrained(path)
        if tokenizer_artifact_sha256(path) != expected_sha256:
            raise RuntimeError("legacy tokenizer changed during worker initialization")
    if int(tokenizer.vocab_size) != int(expected_vocab_size):
        raise RuntimeError(
            "tokenizer vocabulary size differs from the pinned build contract"
        )
    return tokenizer


def validate_split_partition(
    split_indices: Mapping[str, Sequence[int]],
    *,
    available_source_indices: set[int],
) -> None:
    if set(split_indices) != set(SPLIT_NAMES):
        raise ValueError(
            f"split mapping must contain exactly {list(SPLIT_NAMES)}"
        )
    normalized: dict[str, set[int]] = {}
    for split_name in SPLIT_NAMES:
        values = [int(value) for value in split_indices[split_name]]
        if len(values) != len(set(values)):
            raise ValueError(f"{split_name} split contains duplicate indices")
        normalized[split_name] = set(values)
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            overlap = normalized[left_name] & normalized[right_name]
            if overlap:
                raise ValueError(
                    f"split overlap between {left_name}/{right_name}: "
                    f"source_index={min(overlap)}"
                )
    assigned = set().union(*(normalized[name] for name in SPLIT_NAMES))
    unknown = assigned - available_source_indices
    if unknown:
        raise ValueError(
            f"split source_index={min(unknown)} is not present in the raw CSV"
        )
    if not assigned:
        raise ValueError("scaffold split does not contain any valid molecule")


def ordered_valid_source_indices(
    split_indices: Mapping[str, Sequence[int]],
) -> list[int]:
    assigned = [
        int(value)
        for split_name in SPLIT_NAMES
        for value in split_indices[split_name]
    ]
    if len(assigned) != len(set(assigned)):
        raise ValueError("split indices overlap or contain duplicates")
    return sorted(assigned)


def _resolve_split_path(
    split_root: os.PathLike[str] | str,
    dataset_name: str,
) -> Path:
    root = Path(split_root)
    candidates = (
        root / f"{dataset_name}_scaffold.json",
        root / f"{dataset_name}_scaffold.npz",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected exactly one safe scaffold split for {dataset_name}; "
            f"candidates={[str(path) for path in candidates]}"
        )
    return existing[0]


def _read_split_metadata(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            payload = json.loads(str(archive["metadata_json"].item()))
    else:
        raise ValueError(f"unsupported scaffold split suffix: {path.suffix}")
    if not isinstance(payload, Mapping):
        raise ValueError("scaffold split metadata must be an object")
    return payload


def validate_split_contract(
    *,
    metadata: Mapping[str, Any],
    expected_dataset_name: str,
    split_indices: Mapping[str, Sequence[int]],
    available_source_indices: set[int],
) -> None:
    if metadata.get("dataset_name") != expected_dataset_name:
        raise ValueError(
            "scaffold split dataset_name mismatch: "
            f"expected={expected_dataset_name!r}, "
            f"actual={metadata.get('dataset_name')!r}"
        )
    invalid_rows = metadata.get("invalid")
    if not isinstance(invalid_rows, list):
        raise ValueError("scaffold split invalid-row report is missing")
    invalid_indices: list[int] = []
    for row in invalid_rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("source_index"), int)
            or isinstance(row.get("source_index"), bool)
            or not isinstance(row.get("reason"), str)
        ):
            raise ValueError("scaffold split invalid-row report is malformed")
        invalid_indices.append(int(row["source_index"]))
    if len(invalid_indices) != len(set(invalid_indices)):
        raise ValueError("scaffold split invalid rows contain duplicate indices")
    invalid_set = set(invalid_indices)
    assigned = {
        int(value)
        for split_name in SPLIT_NAMES
        for value in split_indices[split_name]
    }
    if assigned & invalid_set:
        raise ValueError("assigned split rows overlap invalid rows")
    unknown_invalid = invalid_set - available_source_indices
    if unknown_invalid:
        raise ValueError(
            f"invalid source_index={min(unknown_invalid)} is not in the raw CSV"
        )
    covered = assigned | invalid_set
    if covered != available_source_indices:
        missing = sorted(available_source_indices - covered)[:10]
        extra = sorted(covered - available_source_indices)[:10]
        raise ValueError(
            "assigned and invalid scaffold rows must cover the raw CSV exactly; "
            f"missing={missing}, extra={extra}"
        )


def validate_split_content_contract(
    *,
    metadata: Mapping[str, Any],
    split_indices: Mapping[str, Sequence[int]],
    smiles_by_source: Mapping[int, str],
    include_chirality: bool = False,
) -> None:
    scaffold_hashes = metadata.get("scaffold_hashes")
    scaffold_members = metadata.get("scaffold_members")
    if (
        not isinstance(scaffold_hashes, Mapping)
        or not isinstance(scaffold_members, Mapping)
        or set(scaffold_members) != set(scaffold_hashes)
    ):
        raise ValueError(
            "scaffold split must contain a complete scaffold membership audit"
        )
    split_by_source: dict[int, str] = {}
    for split_name in SPLIT_NAMES:
        for raw_source_index in split_indices[split_name]:
            source_index = int(raw_source_index)
            if source_index in split_by_source:
                raise ValueError("scaffold split source indices are not disjoint")
            split_by_source[source_index] = split_name

    current_members: dict[str, list[int]] = {}
    current_splits: dict[str, str] = {}
    for source_index, split_name in split_by_source.items():
        if source_index not in smiles_by_source:
            raise ValueError(
                f"split source_index={source_index} is absent from current CSV"
            )
        scaffold = generate_scaffold(
            smiles_by_source[source_index],
            include_chirality=include_chirality,
        )
        if scaffold is None:
            raise ValueError(
                f"assigned source_index={source_index} is invalid in current CSV"
            )
        scaffold_hash = hashlib.sha256(
            scaffold.encode("utf-8")
        ).hexdigest()
        current_members.setdefault(scaffold_hash, []).append(source_index)
        previous_split = current_splits.setdefault(scaffold_hash, split_name)
        if previous_split != split_name:
            raise ValueError(
                "current CSV scaffold crosses persisted splits: "
                f"scaffold_hash={scaffold_hash}"
            )
    for members in current_members.values():
        members.sort()

    expected_hashes = {
        str(scaffold_hash): str(split_name)
        for scaffold_hash, split_name in scaffold_hashes.items()
    }
    expected_members = {
        str(scaffold_hash): sorted(map(int, members))
        for scaffold_hash, members in scaffold_members.items()
    }
    if expected_hashes != current_splits:
        raise ValueError(
            "current CSV scaffold hashes/split assignments differ from "
            "the persisted split audit"
        )
    if expected_members != current_members:
        raise ValueError(
            "current CSV scaffold memberships differ from the persisted "
            "split audit"
        )

    invalid_rows = metadata.get("invalid")
    if not isinstance(invalid_rows, list):
        raise ValueError("scaffold split invalid-row report is missing")
    for row in invalid_rows:
        if not isinstance(row, Mapping):
            raise ValueError("scaffold split invalid-row report is malformed")
        source_index = int(row["source_index"])
        if source_index not in smiles_by_source:
            raise ValueError(
                f"invalid source_index={source_index} is absent from current CSV"
            )
        smiles = smiles_by_source[source_index]
        if generate_scaffold(
            smiles,
            include_chirality=include_chirality,
        ) is not None:
            raise ValueError(
                f"invalid source_index={source_index} became valid in current CSV"
            )
        expected_reason = (
            "missing_smiles"
            if not str(smiles).strip()
            else "invalid_smiles"
        )
        if row.get("reason") != expected_reason:
            raise ValueError(
                f"invalid source_index={source_index} reason differs from "
                "the current CSV"
            )


def _split_mapping(
    path: Path,
    *,
    expected_dataset_name: str,
    available_source_indices: set[int],
    smiles_by_source: Mapping[int, str],
) -> dict[str, list[int]]:
    train, valid, test = load_scaffold_split(path)
    result = {
        "train": list(map(int, train)),
        "valid": list(map(int, valid)),
        "test": list(map(int, test)),
    }
    validate_split_partition(
        result,
        available_source_indices=available_source_indices,
    )
    metadata = _read_split_metadata(path)
    validate_split_contract(
        metadata=metadata,
        expected_dataset_name=expected_dataset_name,
        split_indices=result,
        available_source_indices=available_source_indices,
    )
    validate_split_content_contract(
        metadata=metadata,
        split_indices=result,
        smiles_by_source=smiles_by_source,
        include_chirality=False,
    )
    return result


def _rows_by_source(rows: MoleculeNetRows) -> dict[int, int]:
    result = {
        int(source_index): row_position
        for row_position, source_index in enumerate(rows.row_indices.tolist())
    }
    if len(result) != len(rows.row_indices):
        raise ValueError("MoleculeNet rows contain duplicate source indices")
    return result


_WORKER_BUILDER: Optional[MultimodalFeatureBuilder] = None


def _initialize_worker(
    tokenizer_snapshot: str,
    expected_tokenizer_sha256: str,
    expected_tokenizer_vocab_size: int,
    feature_config: Mapping[str, Any],
) -> None:
    global _WORKER_BUILDER
    tokenizer = _load_pinned_tokenizer(
        tokenizer_snapshot,
        expected_sha256=expected_tokenizer_sha256,
        expected_vocab_size=expected_tokenizer_vocab_size,
    )
    _WORKER_BUILDER = MultimodalFeatureBuilder(
        tokenizer,
        FeatureBuildConfig(**dict(feature_config)),
    )


def _build_task(
    task: MoleculeNetTask,
) -> tuple[
    MoleculeNetTask,
    Optional[dict[str, Any]],
    Optional[dict[str, Any]],
]:
    if _WORKER_BUILDER is None:
        raise RuntimeError("MoleculeNet build worker was not initialized")
    try:
        record = _WORKER_BUILDER.build_record(
            smiles=task.smiles,
            source_index=task.source_index,
            sample_namespace=task.dataset_name,
            labels=task.labels,
            label_mask=task.label_mask,
            metadata={
                "dataset_name": task.dataset_name,
                "task_type": task.task_type,
                "label_columns": list(task.label_columns),
            },
        )
        return task, record, None
    except FeatureBuildError as exc:
        return task, None, exc.to_dict(task.smiles)


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


def _write_shard_index(
    staging_dir: Path,
    shard_id: int,
    record_indices: Sequence[int],
    source_indices: Sequence[int],
) -> Path:
    records = np.asarray(record_indices, dtype=np.int64)
    sources = np.asarray(source_indices, dtype=np.int64)
    if records.ndim != 1 or sources.shape != records.shape or records.size == 0:
        raise ValueError("shard record/source index arrays are invalid")
    expected = np.arange(
        int(records[0]),
        int(records[0]) + len(records),
        dtype=np.int64,
    )
    if not np.array_equal(records, expected):
        raise ValueError("shard record indices must be contiguous")
    metadata = {
        "schema": BUILD_INDEX_SCHEMA,
        "shard_id": int(shard_id),
        "start_index": int(records[0]),
        "record_count": len(records),
    }
    destination = staging_dir / f"build-index-{shard_id:06d}.npz"
    _atomic_savez(
        destination,
        {
            "record_index": records,
            "source_index": sources,
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                dtype=np.str_,
            ),
        },
        overwrite=False,
    )
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


def _load_completed_prefix(
    staging_dir: Path,
    expected_sources: Sequence[int],
    *,
    verify_checksums: bool,
) -> int:
    _converge_staged_shards(staging_dir)
    sidecars = sorted(staging_dir.glob("shard-*.json"))
    shard_dirs = sorted(staging_dir.glob("shard-*.lmdb"))
    if len(sidecars) != len(shard_dirs):
        raise RuntimeError("staged shard directories and sidecars differ")
    expected_record_index = 0
    for shard_id, sidecar_path in enumerate(sidecars):
        if sidecar_path.name != f"shard-{shard_id:06d}.json":
            raise RuntimeError("staged shard sidecars are not contiguous")
        shard_dir = staging_dir / f"shard-{shard_id:06d}.lmdb"
        if not shard_dir.is_dir():
            raise RuntimeError(f"staged shard is missing: {shard_dir}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar_count = sidecar.get("record_count")
        if (
            sidecar.get("schema_version") != 1
            or sidecar.get("shard_id") != shard_id
            or sidecar.get("start_index") != expected_record_index
            or not isinstance(sidecar_count, int)
            or isinstance(sidecar_count, bool)
            or sidecar_count <= 0
            or sidecar.get("end_index_exclusive")
            != expected_record_index + sidecar_count
            or sidecar.get("codec") != "msgpack+zstd+sha256"
        ):
            raise RuntimeError(
                f"invalid staged shard sidecar: {sidecar_path}"
            )
        if verify_checksums and sha256_file(
            shard_dir / "data.mdb"
        ) != str(sidecar.get("sha256", "")):
            raise RuntimeError(f"staged shard checksum mismatch: {shard_dir}")
        index_path = staging_dir / f"build-index-{shard_id:06d}.npz"
        if not index_path.is_file():
            raise RuntimeError(f"staged build index is missing: {index_path}")
        with np.load(index_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "record_index",
                "source_index",
                "metadata_json",
            }:
                raise RuntimeError(
                    f"invalid staged build-index inventory: {index_path}"
                )
            metadata = json.loads(str(archive["metadata_json"].item()))
            records = np.asarray(archive["record_index"], dtype=np.int64)
            sources = np.asarray(archive["source_index"], dtype=np.int64)
        if (
            metadata.get("schema") != BUILD_INDEX_SCHEMA
            or int(metadata.get("shard_id", -1)) != shard_id
            or int(metadata.get("start_index", -1))
            != expected_record_index
            or int(metadata.get("record_count", -1)) != sidecar_count
            or len(records) != sidecar_count
            or sources.shape != records.shape
        ):
            raise RuntimeError(f"invalid staged build index: {index_path}")
        expected_records = np.arange(
            expected_record_index,
            expected_record_index + len(records),
            dtype=np.int64,
        )
        if not np.array_equal(records, expected_records):
            raise RuntimeError("staged build indices are not contiguous")
        expected_source_slice = np.asarray(
            expected_sources[
                expected_record_index
                : expected_record_index + len(sources)
            ],
            dtype=np.int64,
        )
        if not np.array_equal(sources, expected_source_slice):
            raise RuntimeError(
                "staged source indices are not the expected deterministic prefix"
            )
        expected_record_index += len(records)
    return expected_record_index


def _publish_split_views(
    staging_dir: Path,
    split_indices: Mapping[str, Sequence[int]],
    record_by_source: Mapping[int, int],
) -> dict[str, Mapping[str, Any]]:
    views_dir = staging_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Mapping[str, Any]] = {}
    seen_records: set[int] = set()
    for split_name in SPLIT_NAMES:
        sources = np.asarray(
            [int(value) for value in split_indices[split_name]],
            dtype=np.int64,
        )
        records = np.asarray(
            [record_by_source[int(value)] for value in sources],
            dtype=np.int64,
        )
        if len(np.unique(records)) != len(records):
            raise RuntimeError(f"{split_name} view contains duplicate records")
        overlap = seen_records.intersection(records.tolist())
        if overlap:
            raise RuntimeError(
                f"split views overlap at record_index={min(overlap)}"
            )
        seen_records.update(records.tolist())
        metadata = {
            "schema": VIEW_SCHEMA,
            "split": split_name,
            "record_count": len(records),
        }
        destination = views_dir / f"{split_name}.npz"
        _atomic_savez(
            destination,
            {
                "record_index": records,
                "source_index": sources,
                "metadata_json": np.asarray(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    dtype=np.str_,
                ),
            },
            overwrite=True,
        )
        result[split_name] = {
            "path": f"views/{destination.name}",
            "sha256": sha256_file(destination),
            "record_count": len(records),
        }
    return result


def _expected_store_metadata(
    *,
    record_count: int,
    records_per_shard: int,
    tokenizer_sha256: str,
    tokenizer_vocab_size: int,
) -> StoreMetadata:
    return StoreMetadata(
        schema_version=1,
        record_count=record_count,
        records_per_shard=records_per_shard,
        modalities=("1d", "2d", "3d", "qm"),
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_vocab_size=tokenizer_vocab_size,
        shards=tuple(
            f"shard-{shard_id:06d}.lmdb"
            for shard_id in range(
                math.ceil(record_count / records_per_shard)
            )
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


def _validate_ready_dataset_staging(
    staging_dir: Path,
    *,
    contract: Mapping[str, Any],
    dataset_name: str,
    expected_record_count: int,
    failure_path: Path,
    failure_count: int,
    verify_checksums: bool,
) -> bool:
    manifest_path = staging_dir / "build-manifest.json"
    if not manifest_path.exists():
        return False
    if not manifest_path.is_file():
        raise RuntimeError("staged build-manifest.json is not a file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in contract.items()):
        raise RuntimeError(
            "existing complete MoleculeNet manifest differs from build contract"
        )
    expected_manifest_fields = set(contract) | {
        "status",
        "record_count",
        "shard_count",
        "views",
        "failure_log",
    }
    if set(manifest) != expected_manifest_fields:
        raise RuntimeError(
            "existing staged MoleculeNet manifest inventory is not exact"
        )
    failure_descriptor = manifest.get("failure_log")
    if (
        manifest.get("status") != "complete"
        or manifest.get("record_count") != expected_record_count
        or manifest.get("shard_count")
        != math.ceil(
            expected_record_count
            / int(contract["storage"]["records_per_shard"])
        )
        or not isinstance(failure_descriptor, Mapping)
        or set(failure_descriptor)
        != {
            "path",
            "sha256",
            "record_count",
            "resolved_record_count",
        }
        or failure_descriptor.get("path") != failure_path.name
        or failure_descriptor.get("record_count") != failure_count
        or failure_descriptor.get("resolved_record_count") != failure_count
        or not failure_path.is_file()
        or failure_descriptor.get("sha256") != sha256_file(failure_path)
    ):
        raise RuntimeError(
            "existing staged MoleculeNet completion metadata is inconsistent"
        )
    validate_completed_dataset_store(
        staging_dir,
        expected_dataset_name=dataset_name,
        verify_checksums=verify_checksums,
    )
    return True


def _contract(
    *,
    dataset_name: str,
    csv_path: Path,
    csv_sha256: str,
    split_path: Path,
    split_sha256: str,
    tokenizer_snapshot: TokenizerSnapshot,
    feature_config: FeatureBuildConfig,
    records_per_shard: int,
    lmdb_map_size: int,
    compression_level: int,
    label_columns: Sequence[str],
) -> dict[str, Any]:
    spec = get_moleculenet_spec(dataset_name)
    return {
        "schema": BUILD_SCHEMA,
        "dataset_name": dataset_name,
        "source": {
            "path": str(csv_path.resolve()),
            "sha256": csv_sha256,
            "smiles_column": spec.smiles_column,
            "label_columns": list(label_columns),
        },
        "split": {
            "path": str(split_path.resolve()),
            "sha256": split_sha256,
            "include_chirality": False,
        },
        "tokenizer": {
            "path": str(tokenizer_snapshot.root),
            "snapshot_path": str(tokenizer_snapshot.load_path),
            "artifact_sha256": tokenizer_snapshot.artifact_sha256,
            "vocab_size": tokenizer_snapshot.vocab_size,
        },
        "features": feature_config.to_dict(),
        "storage": {
            "records_per_shard": int(records_per_shard),
            "lmdb_map_size": int(lmdb_map_size),
            "compression_level": int(compression_level),
        },
        "task": {
            "type": spec.task_type,
            "num_tasks": spec.num_tasks,
            "main_metric": spec.main_metric,
        },
    }


def _read_pinned_file_bytes(path: Path, *, label: str) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                chunks.append(chunk)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as exc:
        raise RuntimeError(f"cannot pin {label}: {path}") from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field, None) != getattr(after, field, None)
        or getattr(after, field, None) != getattr(current, field, None)
        for field in stable_fields
    ):
        raise RuntimeError(f"{label} changed while its bytes were pinned")
    payload = b"".join(chunks)
    if len(payload) != int(after.st_size):
        raise RuntimeError(f"{label} byte count differs from its file size")
    return payload, digest.hexdigest()


def _load_split_from_pinned_bytes(
    payload: bytes,
    *,
    original_path: Path,
    expected_dataset_name: str,
    available_source_indices: set[int],
    smiles_by_source: Mapping[int, str],
) -> dict[str, list[int]]:
    suffix = original_path.suffix.lower()
    if suffix not in {".json", ".npz"}:
        raise ValueError(f"unsupported scaffold split suffix: {suffix}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{original_path.stem}.pinned-",
        suffix=suffix,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return _split_mapping(
            temporary,
            expected_dataset_name=expected_dataset_name,
            available_source_indices=available_source_indices,
            smiles_by_source=smiles_by_source,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _failure_key(row: Mapping[str, Any]) -> tuple[int, str]:
    source_index = row.get("source_index")
    return (
        -1 if source_index is None else int(source_index),
        str(row.get("stage", "")),
    )


def _load_existing_failures(path: Path) -> set[tuple[int, str]]:
    if not path.is_file():
        return set()
    records: dict[tuple[int, str], Mapping[str, Any]] = {}
    line_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
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
    return set(records)


def build_dataset_store(
    *,
    dataset_name: str,
    raw_root: os.PathLike[str] | str,
    split_root: os.PathLike[str] | str,
    tokenizer_dir: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
    feature_config: FeatureBuildConfig,
    records_per_shard: int,
    lmdb_map_size: int,
    compression_level: int,
    commit_interval: int,
    workers: int,
    work_batch_size: int,
    resume: bool,
    verify_checksums: bool,
) -> Path:
    """Build and atomically publish one registered MoleculeNet dataset."""

    spec = get_moleculenet_spec(dataset_name)
    csv_path = resolve_moleculenet_csv(raw_root, spec.name).resolve()
    split_path = _resolve_split_path(split_root, spec.name).resolve()
    tokenizer_path = Path(tokenizer_dir).resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(tokenizer_path)
    csv_bytes, csv_sha256 = _read_pinned_file_bytes(
        csv_path,
        label=f"{spec.name} raw CSV",
    )
    split_bytes, split_sha256 = _read_pinned_file_bytes(
        split_path,
        label=f"{spec.name} scaffold split",
    )
    csv_compression = (
        "gzip" if csv_path.suffix.lower() == ".gz" else None
    )
    frame = pd.read_csv(
        io.BytesIO(csv_bytes),
        compression=csv_compression,
    )
    frame.index = np.arange(len(frame), dtype=np.int64)
    rows = extract_moleculenet_rows(frame, spec)
    source_to_row = _rows_by_source(rows)
    available_source_indices = set(source_to_row)
    smiles_by_source = {
        source_index: rows.smiles[row_position]
        for source_index, row_position in source_to_row.items()
    }
    split_indices = _load_split_from_pinned_bytes(
        split_bytes,
        original_path=split_path,
        expected_dataset_name=spec.name,
        available_source_indices=available_source_indices,
        smiles_by_source=smiles_by_source,
    )
    ordered_sources = ordered_valid_source_indices(split_indices)
    tokenizer_snapshot = resolve_tokenizer_snapshot(tokenizer_path)
    contract = _contract(
        dataset_name=spec.name,
        csv_path=csv_path,
        csv_sha256=csv_sha256,
        split_path=split_path,
        split_sha256=split_sha256,
        tokenizer_snapshot=tokenizer_snapshot,
        feature_config=feature_config,
        records_per_shard=records_per_shard,
        lmdb_map_size=lmdb_map_size,
        compression_level=compression_level,
        label_columns=rows.label_columns,
    )

    output_parent = Path(output_root).resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    final_path = output_parent / spec.name
    staging_path = output_parent / f".{spec.name}.building"
    contract_path = staging_path / "build-contract.json"
    tokenizer_snapshot_path = staging_path / ".tokenizer.snapshot"
    if final_path.exists():
        raise FileExistsError(
            f"final dataset store already exists: {final_path}"
        )
    if workers <= 0 or work_batch_size <= 0:
        raise ValueError("workers and work_batch_size must be positive")
    if records_per_shard <= 0 or commit_interval <= 0:
        raise ValueError("storage sizes must be positive")
    if resume:
        if not staging_path.exists() and not staging_path.is_symlink():
            raise FileNotFoundError(
                f"no resumable staged build exists: {staging_path}"
            )
        _assert_staging_tree_not_redirected(staging_path)
        if not contract_path.is_file():
            raise FileNotFoundError(
                f"staged build contract is missing: {contract_path}"
            )
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError(
                "resume contract differs from staged input/configuration"
            )
    else:
        if staging_path.exists():
            raise FileExistsError(
                f"staged build exists; use --resume or inspect {staging_path}"
            )
        staging_path.mkdir(parents=True)
        _fsync_directory(output_parent)
        atomic_write_json(contract_path, contract)
    _assert_staging_tree_not_redirected(staging_path)
    _cleanup_known_build_temporaries(staging_path)
    worker_tokenizer_snapshot = _prepare_private_tokenizer_snapshot(
        tokenizer_snapshot,
        tokenizer_snapshot_path,
    )
    completed_count = (
        _load_completed_prefix(
            staging_path,
            ordered_sources,
            verify_checksums=verify_checksums,
        )
        if resume
        else 0
    )

    if (
        completed_count % records_per_shard
        and completed_count != len(ordered_sources)
    ):
        raise RuntimeError("resume prefix ends in an unexpected partial shard")
    next_shard_id = math.ceil(completed_count / records_per_shard)
    codec = RecordCodec(compression_level=compression_level)
    current_writer: Optional[LmdbShardWriter] = None
    current_records: list[int] = []
    current_sources: list[int] = []
    failure_path = staging_path / "feature-failures.jsonl"
    failure_keys = _load_existing_failures(failure_path)
    failure_mode = "a" if failure_path.exists() else "w"
    final_store_metadata = _expected_store_metadata(
        record_count=len(ordered_sources),
        records_per_shard=records_per_shard,
        tokenizer_sha256=tokenizer_snapshot.artifact_sha256,
        tokenizer_vocab_size=tokenizer_snapshot.vocab_size,
    )
    if (
        (staging_path / "build-manifest.json").exists()
        and completed_count != len(ordered_sources)
    ):
        raise RuntimeError(
            "staged complete manifest exists without a complete shard prefix"
        )
    if completed_count == len(ordered_sources):
        if _validate_ready_dataset_staging(
            staging_path,
            contract=contract,
            dataset_name=spec.name,
            expected_record_count=len(ordered_sources),
            failure_path=failure_path,
            failure_count=len(failure_keys),
            verify_checksums=verify_checksums,
        ):
            verify_tokenizer_snapshot(worker_tokenizer_snapshot)
            _assert_staging_tree_not_redirected(staging_path)
            _remove_private_tokenizer_snapshot(tokenizer_snapshot_path)
            _cleanup_known_build_temporaries(staging_path)
            _assert_no_private_staging_entries(staging_path)
            _assert_staging_tree_not_redirected(staging_path)
            _assert_final_staging_inventory(
                staging_path,
                metadata=final_store_metadata,
            )
            os.replace(staging_path, final_path)
            _fsync_directory(output_parent)
            return final_path
    pool: Optional[Pool] = None
    if workers == 1:
        _initialize_worker(
            str(worker_tokenizer_snapshot.load_path),
            tokenizer_snapshot.artifact_sha256,
            tokenizer_snapshot.vocab_size,
            feature_config.to_dict(),
        )
    else:
        pool = Pool(
            processes=workers,
            initializer=_initialize_worker,
            initargs=(
                str(worker_tokenizer_snapshot.load_path),
                tokenizer_snapshot.artifact_sha256,
                tokenizer_snapshot.vocab_size,
                feature_config.to_dict(),
            ),
        )

    try:
        with failure_path.open(
            failure_mode,
            encoding="utf-8",
            newline="\n",
        ) as failure_stream:
            for start in range(
                completed_count,
                len(ordered_sources),
                work_batch_size,
            ):
                source_batch = ordered_sources[start : start + work_batch_size]
                tasks = []
                for source_index in source_batch:
                    row_position = source_to_row[source_index]
                    tasks.append(
                        MoleculeNetTask(
                            dataset_name=spec.name,
                            source_index=source_index,
                            smiles=rows.smiles[row_position],
                            labels=rows.labels[row_position].copy(),
                            label_mask=rows.label_mask[row_position].copy(),
                            task_type=spec.task_type,
                            label_columns=rows.label_columns,
                        )
                    )
                results = (
                    list(map(_build_task, tasks))
                    if pool is None
                    else pool.map(_build_task, tasks)
                )
                for offset, (task, record, failure) in enumerate(results):
                    if failure is not None:
                        key = _failure_key(failure)
                        if key not in failure_keys:
                            failure_stream.write(
                                json.dumps(
                                    failure,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            failure_keys.add(key)
                        failure_stream.flush()
                        os.fsync(failure_stream.fileno())
                        raise RuntimeError(
                            f"{spec.name} feature construction failed for "
                            f"source_index={task.source_index}; staged output "
                            f"and failure report are preserved"
                        )
                    if record is None:
                        raise RuntimeError(
                            "worker returned neither a record nor a failure"
                        )
                    # Pool.map preserves task order, so the source prefix is
                    # also the contiguous global record order.
                    record_index = start + offset
                    if ordered_sources[record_index] != task.source_index:
                        raise RuntimeError(
                            "worker result order differs from task order"
                        )
                    if current_writer is None:
                        remaining = len(ordered_sources) - record_index
                        current_writer = LmdbShardWriter(
                            store_dir=staging_path,
                            shard_id=next_shard_id,
                            start_index=record_index,
                            expected_records=min(
                                records_per_shard,
                                remaining,
                            ),
                            map_size=lmdb_map_size,
                            codec=codec,
                            commit_interval=commit_interval,
                        )
                        current_records = []
                        current_sources = []
                    current_writer.put(record_index, record)
                    current_records.append(record_index)
                    current_sources.append(task.source_index)
                    if (
                        current_writer.record_count
                        == current_writer.expected_records
                    ):
                        index_path = _write_shard_index(
                            staging_path,
                            next_shard_id,
                            current_records,
                            current_sources,
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
                        current_records = []
                        current_sources = []
                        next_shard_id += 1
            if current_writer is not None:
                raise RuntimeError("dataset build left an unfinalized shard")
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

    verify_tokenizer_snapshot(worker_tokenizer_snapshot)
    record_by_source = {
        source_index: record_index
        for record_index, source_index in enumerate(ordered_sources)
    }
    shard_count = len(final_store_metadata.shards)
    _write_or_validate_store_metadata(
        staging_path,
        final_store_metadata,
    )
    views = _publish_split_views(
        staging_path,
        split_indices,
        record_by_source,
    )
    verify_tokenizer_snapshot(worker_tokenizer_snapshot)
    manifest = {
        **contract,
        "status": "complete",
        "record_count": len(ordered_sources),
        "shard_count": shard_count,
        "views": views,
        "failure_log": {
            "path": failure_path.name,
            "sha256": sha256_file(failure_path),
            "record_count": len(failure_keys),
            "resolved_record_count": len(failure_keys),
        },
    }
    atomic_write_json(
        staging_path / "build-manifest.json",
        manifest,
    )
    if not _validate_ready_dataset_staging(
        staging_path,
        contract=contract,
        dataset_name=spec.name,
        expected_record_count=len(ordered_sources),
        failure_path=failure_path,
        failure_count=len(failure_keys),
        verify_checksums=verify_checksums,
    ):
        raise RuntimeError(
            "completed MoleculeNet staging did not pass final validation"
        )
    _assert_staging_tree_not_redirected(staging_path)
    _remove_private_tokenizer_snapshot(tokenizer_snapshot_path)
    _cleanup_known_build_temporaries(staging_path)
    _assert_no_private_staging_entries(staging_path)
    _assert_staging_tree_not_redirected(staging_path)
    _assert_final_staging_inventory(
        staging_path,
        metadata=final_store_metadata,
    )
    os.replace(staging_path, final_path)
    _fsync_directory(output_parent)
    return final_path


def validate_completed_dataset_store(
    store_dir: os.PathLike[str] | str,
    *,
    expected_dataset_name: str,
    verify_checksums: bool,
) -> None:
    directory = Path(store_dir).resolve()
    manifest_path = directory / "build-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"completed store has no build manifest: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != BUILD_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("dataset_name") != expected_dataset_name
    ):
        raise RuntimeError(
            f"completed store manifest is invalid for {expected_dataset_name}"
        )
    store = ShardedRecordStore(directory)
    try:
        if len(store) != int(manifest.get("record_count", -1)):
            raise RuntimeError("completed store record count differs from manifest")
        if len(store.metadata.shards) != int(
            manifest.get("shard_count", -1)
        ):
            raise RuntimeError("completed store shard count differs from manifest")
        tokenizer_contract = manifest.get("tokenizer")
        if (
            not isinstance(tokenizer_contract, Mapping)
            or store.metadata.tokenizer_sha256
            != str(tokenizer_contract.get("artifact_sha256", ""))
            or store.metadata.tokenizer_vocab_size
            != int(tokenizer_contract.get("vocab_size", -1))
        ):
            raise RuntimeError(
                "completed store tokenizer metadata differs from build manifest"
            )
        if verify_checksums:
            store.verify_shard_checksums()
    finally:
        store.close()

    views = manifest.get("views")
    if not isinstance(views, Mapping) or set(views) != set(SPLIT_NAMES):
        raise RuntimeError("completed MoleculeNet store has invalid split views")
    seen: set[int] = set()
    for split_name in SPLIT_NAMES:
        descriptor = views[split_name]
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "sha256", "record_count"}
        ):
            raise RuntimeError(f"invalid {split_name} view descriptor")
        relative = Path(str(descriptor.get("path", "")))
        path = (directory / relative).resolve()
        if relative.is_absolute() or directory not in path.parents:
            raise RuntimeError(f"{split_name} view escapes its store directory")
        if not path.is_file() or sha256_file(path) != str(
            descriptor.get("sha256", "")
        ):
            raise RuntimeError(f"{split_name} view integrity check failed")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "record_index",
                "source_index",
                "metadata_json",
            }:
                raise RuntimeError(
                    f"{split_name} view inventory is not exact"
                )
            records = np.asarray(archive["record_index"], dtype=np.int64)
            sources = np.asarray(archive["source_index"], dtype=np.int64)
            view_metadata = json.loads(
                str(archive["metadata_json"].item())
            )
        if (
            records.ndim != 1
            or sources.shape != records.shape
            or len(records) != int(descriptor.get("record_count", -1))
            or len(np.unique(records)) != len(records)
            or len(np.unique(sources)) != len(sources)
            or view_metadata
            != {
                "schema": VIEW_SCHEMA,
                "split": split_name,
                "record_count": len(records),
            }
        ):
            raise RuntimeError(f"{split_name} view schema/count is invalid")
        overlap = seen.intersection(records.tolist())
        if overlap:
            raise RuntimeError(f"{split_name} view overlaps another split")
        seen.update(records.tolist())
    if seen != set(range(int(manifest["record_count"]))):
        raise RuntimeError("completed split views do not cover the whole store")


def build_all_moleculenet_stores(
    *,
    datasets: Sequence[str],
    raw_root: os.PathLike[str] | str,
    split_root: os.PathLike[str] | str,
    tokenizer_dir: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
    feature_config: FeatureBuildConfig,
    records_per_shard: int,
    lmdb_map_size: int,
    compression_level: int,
    commit_interval: int,
    workers: int,
    work_batch_size: int,
    resume: bool,
    verify_checksums: bool,
    skip_completed: bool,
) -> dict[str, str]:
    normalized = [str(name).strip().lower() for name in datasets]
    unknown = sorted(set(normalized) - set(MOLECULENET_REGISTRY))
    if unknown:
        raise ValueError(f"unknown MoleculeNet datasets: {unknown}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("datasets must not contain duplicates")
    output_path = Path(output_root).resolve()
    result: dict[str, str] = {}
    for name in normalized:
        final_path = output_path / name
        if final_path.is_dir() and skip_completed:
            validate_completed_dataset_store(
                final_path,
                expected_dataset_name=name,
                verify_checksums=verify_checksums,
            )
            result[name] = str(final_path)
            continue
        built = build_dataset_store(
            dataset_name=name,
            raw_root=raw_root,
            split_root=split_root,
            tokenizer_dir=tokenizer_dir,
            output_root=output_root,
            feature_config=feature_config,
            records_per_shard=records_per_shard,
            lmdb_map_size=lmdb_map_size,
            compression_level=compression_level,
            commit_interval=commit_interval,
            workers=workers,
            work_batch_size=work_batch_size,
            resume=resume,
            verify_checksums=verify_checksums,
        )
        result[name] = str(built)
    return result


def _optional_sigma(value: str) -> Optional[float]:
    normalized = value.strip().lower()
    if normalized in {"element", "none"}:
        return None
    try:
        result = float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "atomic sigma must be positive or 'element'"
        ) from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError(
            "atomic sigma must be positive and finite"
        )
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build four-modality LMDB stores for MoleculeNet benchmarks"
    )
    parser.add_argument("--raw-root", default="data/raw/MoleculeNet")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument(
        "--tokenizer-dir",
        default="data/processed/pcqm/tokenizer",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/moleculenet",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(MOLECULENET_REGISTRY),
        default=sorted(MOLECULENET_REGISTRY),
    )
    parser.add_argument("--max-smiles-length", type=int, default=256)
    parser.add_argument("--num-conformers", type=int, default=3)
    parser.add_argument("--prune-rms-threshold", type=float, default=0.5)
    parser.add_argument("--geometry-seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--grid-spacing", type=float, default=0.75)
    parser.add_argument("--grid-padding", type=float, default=4.0)
    parser.add_argument(
        "--atomic-sigma",
        type=_optional_sigma,
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
    parser.add_argument("--records-per-shard", type=int, default=4096)
    parser.add_argument("--lmdb-map-size-gib", type=float, default=2.0)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--commit-interval", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--work-batch-size", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--no-verify-checksums", action="store_true")
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
        generated_conformers=args.num_conformers,
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
    outputs = build_all_moleculenet_stores(
        datasets=args.datasets,
        raw_root=args.raw_root,
        split_root=args.split_root,
        tokenizer_dir=args.tokenizer_dir,
        output_root=args.output_root,
        feature_config=feature_config,
        records_per_shard=args.records_per_shard,
        lmdb_map_size=int(args.lmdb_map_size_gib * 1024**3),
        compression_level=args.compression_level,
        commit_interval=args.commit_interval,
        workers=args.workers,
        work_batch_size=args.work_batch_size,
        resume=args.resume,
        verify_checksums=not args.no_verify_checksums,
        skip_completed=args.skip_completed,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create checksummed, resumable shards of full-atom molecular geometries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.molecular.geometry import (  # noqa: E402
    GeometryRecord,
    generate_conformers,
    geometry_from_sdf_or_fallback,
    iter_sdf_molecules,
)
from src.molecular.rdkit_utils import smiles_hash  # noqa: E402

GEOMETRY_SCHEMA = "semmol.geometry.v1"
GEOMETRY_INDEX_SCHEMA = "semmol.geometry_index.v2"
RUN_STATE_SCHEMA = "semmol.geometry_run_state.v1"
WORK_DATABASE_SCHEMA = "semmol.geometry_work.v4"
EMBEDDED_SHARD_METADATA_KEY = "__metadata_json__"
DEFAULT_TABLE_CHUNK_SIZE = 100_000
OFFICIAL_SDF_SNAPSHOT_SCHEMA = "semmol.official_sdf_snapshot.v1"


class ArtifactIntegrityError(RuntimeError):
    """Raised when a shard and its sidecar are not a complete unique pair."""


class ResumeStateError(RuntimeError):
    """Raised when resume inputs or generation parameters do not match."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    _fsync_directory(path.parent)


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


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ResumeStateError(
            "run fingerprint contains a non-canonical JSON value"
        ) from exc


def _validate_run_fingerprint(fingerprint: Any) -> str:
    if (
        not isinstance(fingerprint, dict)
        or set(fingerprint) != {"sha256", "inputs", "parameters"}
        or not isinstance(fingerprint.get("inputs"), dict)
        or not isinstance(fingerprint.get("parameters"), dict)
    ):
        raise ResumeStateError("run fingerprint structure is invalid")
    checksum = fingerprint.get("sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or set(checksum) - set("0123456789abcdef")
    ):
        raise ResumeStateError("run fingerprint checksum is invalid")
    payload_serialized = _canonical_json(
        {
            "inputs": fingerprint["inputs"],
            "parameters": fingerprint["parameters"],
        }
    )
    expected_checksum = hashlib.sha256(
        payload_serialized.encode("utf-8")
    ).hexdigest()
    if checksum != expected_checksum:
        raise ResumeStateError("run fingerprint checksum is not self-consistent")
    return _canonical_json(fingerprint)


def compute_run_fingerprint(
    inputs: dict[str, Optional[os.PathLike[str] | str]],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Fingerprint input bytes and every output-affecting parameter."""
    if (
        not isinstance(inputs, dict)
        or not inputs
        or any(not isinstance(name, str) or not name for name in inputs)
        or not isinstance(parameters, dict)
    ):
        raise ResumeStateError(
            "fingerprint inputs and parameters must be JSON object mappings"
        )
    input_metadata: dict[str, Optional[dict[str, Any]]] = {}
    for name, raw_path in sorted(inputs.items()):
        if raw_path is None:
            input_metadata[name] = None
            continue
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"fingerprint input missing: {path}")
        input_metadata[name] = {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    payload = {
        "inputs": input_metadata,
        "parameters": parameters,
    }
    serialized = _canonical_json(payload)
    normalized_payload = json.loads(serialized)
    fingerprint = {
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        **normalized_payload,
    }
    _validate_run_fingerprint(fingerprint)
    return fingerprint


def ensure_run_state(
    output_dir: os.PathLike[str] | str,
    fingerprint: dict[str, Any],
    resume: bool,
    state_schema: str = RUN_STATE_SCHEMA,
) -> None:
    """Persist a new run fingerprint or strictly validate a resumed run."""
    expected_serialized = _validate_run_fingerprint(fingerprint)
    if not isinstance(state_schema, str) or not state_schema:
        raise ResumeStateError("run-state schema must be a non-empty string")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "run_state.json"
    if resume:
        if not state_path.is_file():
            raise ResumeStateError(f"resume state missing: {state_path}")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResumeStateError(f"invalid resume state: {state_path}") from exc
        if (
            not isinstance(state, dict)
            or set(state) != {"schema", "fingerprint"}
            or not isinstance(state.get("schema"), str)
            or state["schema"] != state_schema
        ):
            raise ResumeStateError(
                "resume fingerprint differs from the persisted inputs/parameters"
            )
        try:
            persisted_serialized = _validate_run_fingerprint(
                state["fingerprint"]
            )
        except ResumeStateError as exc:
            raise ResumeStateError(
                "persisted resume fingerprint is invalid"
            ) from exc
        if persisted_serialized != expected_serialized:
            raise ResumeStateError(
                "resume fingerprint differs from the persisted inputs/parameters"
            )
        return
    if state_path.exists():
        raise ResumeStateError(
            f"run state already exists at {state_path}; use --resume"
        )
    _atomic_write_text(
        state_path,
        json.dumps(
            {"schema": state_schema, "fingerprint": fingerprint},
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def verify_run_fingerprint(
    inputs: dict[str, Optional[os.PathLike[str] | str]],
    parameters: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Reject input mutation between initial pinning and final publication."""
    expected_serialized = _validate_run_fingerprint(expected)
    observed = compute_run_fingerprint(inputs, parameters)
    if _validate_run_fingerprint(observed) != expected_serialized:
        raise ResumeStateError(
            "input files or generation parameters changed during generation"
        )


def _strict_identity_integer(value: Any, *, field: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise TypeError(f"{field} must be an integer")
    normalized = int(value)
    if normalized < 0 or normalized > 2**63 - 1:
        raise ValueError(f"{field} must be within non-negative int64 range")
    return normalized


def write_geometry_shard(
    records: list[
        tuple[int, str, int, GeometryRecord]
        | tuple[int, str, int, Optional[int], GeometryRecord]
    ],
    output_dir: os.PathLike[str] | str,
    shard_id: int,
) -> dict[str, Any]:
    """Atomically write one safe NPZ shard and its checksummed JSON sidecar."""
    if not records:
        raise ValueError("cannot write an empty geometry shard")
    normalized_shard_id = _strict_identity_integer(
        shard_id,
        field="shard_id",
    )
    directory = Path(output_dir)
    filename = f"shard_{normalized_shard_id:06d}.npz"
    shard_path = directory / filename
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    seen_rows: set[int] = set()
    seen_sources: set[int] = set()
    seen_ordinals: set[int] = set()
    for record_index, item in enumerate(records):
        if len(item) == 4:
            row_index, smiles, source_index, record = item
            sdf_ordinal = None
        elif len(item) == 5:
            row_index, smiles, source_index, sdf_ordinal, record = item
        else:
            raise ValueError("geometry shard records must contain 4 or 5 values")
        normalized_row = _strict_identity_integer(
            row_index,
            field="row_index",
        )
        normalized_source = _strict_identity_integer(
            source_index,
            field="source_index",
        )
        normalized_ordinal = (
            None
            if sdf_ordinal is None
            else _strict_identity_integer(
                sdf_ordinal,
                field="sdf_ordinal",
            )
        )
        if normalized_row in seen_rows:
            raise ValueError(f"duplicate row_index={normalized_row} in shard")
        if normalized_source in seen_sources:
            raise ValueError(
                f"duplicate source_index={normalized_source} in shard"
            )
        if (
            normalized_ordinal is not None
            and normalized_ordinal in seen_ordinals
        ):
            raise ValueError(
                f"duplicate sdf_ordinal={normalized_ordinal} in shard"
            )
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("smiles must be a non-empty string")
        if not isinstance(record, GeometryRecord):
            raise TypeError("record must be a GeometryRecord")
        seen_rows.add(normalized_row)
        seen_sources.add(normalized_source)
        if normalized_ordinal is not None:
            seen_ordinals.add(normalized_ordinal)
        key = f"r{record_index:06d}"
        for field, value in record.to_storage_dict().items():
            arrays[f"{key}__{field}"] = value
        entries.append(
            {
                "key": key,
                "row_index": normalized_row,
                "source_index": normalized_source,
                "sdf_ordinal": normalized_ordinal,
                "train_ordinal": normalized_ordinal,
                "smiles": smiles,
                "smiles_hash": smiles_hash(smiles),
                "canonical_smiles": record.canonical_smiles,
                "num_atoms": int(record.atomic_numbers.size),
                "num_conformers": int(record.coords.shape[0]),
                "reason": record.reason,
            }
        )

    core_metadata: dict[str, Any] = {
        "schema": GEOMETRY_SCHEMA,
        "shard_id": normalized_shard_id,
        "filename": filename,
        "record_count": len(entries),
        "records": entries,
    }
    arrays[EMBEDDED_SHARD_METADATA_KEY] = np.asarray(
        json.dumps(
            core_metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        dtype=np.str_,
    )
    _atomic_savez(shard_path, arrays)
    metadata: dict[str, Any] = {
        **core_metadata,
        "sha256": _sha256_file(shard_path),
    }
    _atomic_write_text(
        directory / f"shard_{normalized_shard_id:06d}.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    return metadata


def _recover_shard_sidecar(artifact: Path, sidecar: Path) -> None:
    try:
        with np.load(artifact, allow_pickle=False) as arrays:
            if EMBEDDED_SHARD_METADATA_KEY not in arrays.files:
                raise ArtifactIntegrityError(
                    f"orphan shard lacks embedded recovery metadata: {artifact}"
                )
            embedded = json.loads(
                str(arrays[EMBEDDED_SHARD_METADATA_KEY].item())
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"cannot recover orphan geometry shard: {artifact}"
        ) from exc
    expected_stem = artifact.stem
    expected_id_text = expected_stem.rsplit("_", 1)[-1]
    if (
        embedded.get("schema") != GEOMETRY_SCHEMA
        or embedded.get("filename") != artifact.name
        or not expected_id_text.isdigit()
        or int(embedded.get("shard_id", -1)) != int(expected_id_text)
        or not isinstance(embedded.get("records"), list)
        or int(embedded.get("record_count", -1))
        != len(embedded.get("records", ()))
        or int(embedded.get("record_count", 0)) <= 0
    ):
        raise ArtifactIntegrityError(
            f"embedded recovery metadata is invalid: {artifact}"
        )
    metadata = {
        **embedded,
        "sha256": _sha256_file(artifact),
    }
    _atomic_write_text(
        sidecar,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )


def reconcile_geometry_artifacts(
    output_dir: os.PathLike[str] | str,
    verify_checksums: bool,
) -> None:
    """Converge interrupted shard/index publication to a verified state."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for temporary in (
        directory / ".geometry_index_work.sqlite3",
        directory / ".geometry_index_work.sqlite3-journal",
        directory / ".geometry_index_work.sqlite3-wal",
        directory / ".geometry_index_work.sqlite3-shm",
    ):
        if not temporary.exists() and not temporary.is_symlink():
            continue
        if temporary.is_dir() and not temporary.is_symlink():
            raise ArtifactIntegrityError(
                f"geometry index scratch is an unexpected directory: {temporary}"
            )
        temporary.unlink(missing_ok=True)
    for temporary in directory.glob(".shard_*.tmp"):
        temporary.unlink()
    for temporary in directory.glob(".geometry_index*.tmp"):
        temporary.unlink()
    for pattern in (
        ".manifest.json.*.tmp",
        ".failures.jsonl.*.tmp",
        ".run_state.json.*.tmp",
    ):
        for temporary in directory.glob(pattern):
            temporary.unlink()
    for temporary in directory.glob(
        ".geometry_index_validate-*.sqlite3*"
    ):
        if temporary.is_file():
            temporary.unlink()

    artifacts = {path.stem: path for path in directory.glob("shard_*.npz")}
    sidecars = {path.stem: path for path in directory.glob("shard_*.json")}
    for stem in sorted(set(artifacts) - set(sidecars)):
        _recover_shard_sidecar(
            artifacts[stem],
            directory / f"{stem}.json",
        )
    for stem in sorted(set(sidecars) - set(artifacts)):
        sidecar_path = sidecars[stem]
        try:
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                f"invalid orphan geometry sidecar: {sidecar_path}"
            ) from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema") != GEOMETRY_SCHEMA
            or metadata.get("filename") != f"{stem}.npz"
        ):
            raise ArtifactIntegrityError(
                f"orphan sidecar cannot be safely discarded: {sidecar_path}"
            )
        sidecar_path.unlink()
        _fsync_directory(directory)

    index_path = directory / "geometry_index.npz"
    index_sidecar = directory / "geometry_index.json"
    if index_path.exists() and not index_sidecar.exists():
        index_path.unlink()
        _fsync_directory(directory)
    elif index_sidecar.exists() and not index_path.exists():
        index_sidecar.unlink()
        _fsync_directory(directory)
    elif index_path.exists():
        try:
            _validate_geometry_index_pair(
                directory,
                verify_checksums=verify_checksums,
            )
        except ArtifactIntegrityError:
            index_path.unlink(missing_ok=True)
            index_sidecar.unlink(missing_ok=True)
            _fsync_directory(directory)


def _complete_geometry_sidecars(
    output_dir: os.PathLike[str] | str,
    verify_checksums: bool,
) -> list[dict[str, Any]]:
    """Compatibility materializer over the streaming strict verifier."""

    return list(
        _iter_verified_geometry_sidecars(
            output_dir,
            verify_checksums=verify_checksums,
        )
    )


def load_completed_rows(
    output_dir: os.PathLike[str] | str,
    verify_checksums: bool = True,
) -> set[int]:
    """Return rows represented by complete sidecar+shard pairs."""
    completed: set[int] = set()
    directory = Path(output_dir)
    if not directory.exists():
        return completed
    reconcile_geometry_artifacts(
        directory,
        verify_checksums=verify_checksums,
    )
    index_path = directory / "geometry_index.npz"
    index_sidecar = directory / "geometry_index.json"
    if index_path.exists() != index_sidecar.exists():
        raise ArtifactIntegrityError(
            "geometry index artifact and sidecar must either both exist or both be absent"
        )
    if index_path.exists() and verify_checksums:
        metadata = json.loads(index_sidecar.read_text(encoding="utf-8"))
        if (
            metadata.get("schema") != GEOMETRY_INDEX_SCHEMA
            or metadata.get("filename") != index_path.name
            or metadata.get("sha256") != _sha256_file(index_path)
        ):
            raise ArtifactIntegrityError("geometry index metadata/checksum mismatch")
    for metadata in _complete_geometry_sidecars(
        directory,
        verify_checksums=verify_checksums,
    ):
        completed.update(int(entry["row_index"]) for entry in metadata["records"])
    return completed


def write_geometry_index(
    output_dir: os.PathLike[str] | str,
    sidecars: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Build a sorted safe index with an external SQLite sort and memmaps."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    reconcile_geometry_artifacts(directory, verify_checksums=True)
    index_database = directory / ".geometry_index_work.sqlite3"
    scratch_directory = directory / ".geometry_index_arrays"
    for scratch in (
        directory / ".geometry_index_work.sqlite3",
        directory / ".geometry_index_work.sqlite3-journal",
        directory / ".geometry_index_work.sqlite3-wal",
        directory / ".geometry_index_work.sqlite3-shm",
    ):
        if not scratch.exists() and not scratch.is_symlink():
            continue
        if scratch.is_dir() and not scratch.is_symlink():
            raise ArtifactIntegrityError(
                f"geometry index scratch is an unexpected directory: {scratch}"
            )
        scratch.unlink(missing_ok=True)
    if scratch_directory.exists():
        shutil.rmtree(scratch_directory)
    scratch_directory.mkdir()
    connection = sqlite3.connect(index_database)
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE locators ("
        "source_index INTEGER PRIMARY KEY,"
        "row_index INTEGER NOT NULL UNIQUE,"
        "sdf_ordinal INTEGER NOT NULL,"
        "shard_id INTEGER NOT NULL,"
        "record_ordinal INTEGER NOT NULL,"
        "UNIQUE(shard_id,record_ordinal)"
        ")"
    )
    connection.execute(
        "CREATE UNIQUE INDEX unique_sdf_ordinal "
        "ON locators(sdf_ordinal) WHERE sdf_ordinal >= 0"
    )
    metadata_iterator: Iterable[dict[str, Any]]
    if sidecars is None:
        metadata_iterator = _iter_verified_geometry_sidecars(
            directory,
            verify_checksums=True,
            _reconcile=False,
        )
    else:
        metadata_iterator = iter(sidecars)
    arrays: dict[str, np.ndarray] = {}
    try:
        for metadata in metadata_iterator:
            rows = []
            for record_ordinal, entry in enumerate(metadata["records"]):
                if entry["source_index"] is None:
                    raise ValueError(
                        f"{metadata['filename']} contains a record without source_index"
                    )
                if entry["key"] != f"r{record_ordinal:06d}":
                    raise ArtifactIntegrityError(
                        f"{metadata['filename']} contains a non-canonical record key"
                    )
                rows.append(
                    (
                        int(entry["source_index"]),
                        int(entry["row_index"]),
                        (
                            -1
                            if entry.get("sdf_ordinal") is None
                            else int(entry["sdf_ordinal"])
                        ),
                        int(metadata["shard_id"]),
                        record_ordinal,
                    )
                )
            try:
                with connection:
                    connection.executemany(
                        "INSERT INTO locators VALUES (?,?,?,?,?)",
                        rows,
                    )
            except sqlite3.IntegrityError as exc:
                raise ArtifactIntegrityError(
                    "duplicate source, row, SDF ordinal, or shard record "
                    "across geometry shards"
                ) from exc

        record_count = int(
            connection.execute("SELECT COUNT(*) FROM locators").fetchone()[0]
        )

        def allocate(name: str, dtype: Any) -> np.ndarray:
            if record_count == 0:
                return np.empty((0,), dtype=dtype)
            return np.lib.format.open_memmap(
                scratch_directory / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(record_count,),
            )

        arrays = {
            "source_index": allocate(
                "source_index",
                np.int64,
            ),
            "row_index": allocate(
                "row_index",
                np.int64,
            ),
            "sdf_ordinal": allocate(
                "sdf_ordinal",
                np.int64,
            ),
            "shard_id": allocate(
                "shard_id",
                np.int32,
            ),
            "record_ordinal": allocate(
                "record_ordinal",
                np.int32,
            ),
        }
        cursor = connection.execute(
            "SELECT source_index,row_index,sdf_ordinal,shard_id,"
            "record_ordinal FROM locators "
            "ORDER BY source_index,row_index"
        )
        offset = 0
        while True:
            rows = cursor.fetchmany(65_536)
            if not rows:
                break
            end = offset + len(rows)
            columns = tuple(zip(*rows))
            arrays["source_index"][offset:end] = columns[0]
            arrays["row_index"][offset:end] = columns[1]
            arrays["sdf_ordinal"][offset:end] = columns[2]
            arrays["shard_id"][offset:end] = columns[3]
            arrays["record_ordinal"][offset:end] = columns[4]
            offset = end
        if offset != record_count:
            raise ArtifactIntegrityError(
                f"geometry index rows {offset} != expected {record_count}"
            )
        for array in arrays.values():
            if isinstance(array, np.memmap):
                array.flush()

        index_path = directory / "geometry_index.npz"
        _atomic_savez(index_path, arrays)
        metadata: dict[str, Any] = {
            "schema": GEOMETRY_INDEX_SCHEMA,
            "filename": index_path.name,
            "sha256": _sha256_file(index_path),
            "record_count": record_count,
            "sorted_by": ["source_index", "row_index"],
            "lookup": "numpy.searchsorted(source_index, requested_source_index)",
        }
        _atomic_write_text(
            directory / "geometry_index.json",
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )
        return metadata
    finally:
        connection.close()
        for array in arrays.values():
            if isinstance(array, np.memmap) and array._mmap is not None:
                array._mmap.close()
        for scratch in (
            directory / ".geometry_index_work.sqlite3",
            directory / ".geometry_index_work.sqlite3-journal",
            directory / ".geometry_index_work.sqlite3-wal",
            directory / ".geometry_index_work.sqlite3-shm",
        ):
            if not scratch.exists() and not scratch.is_symlink():
                continue
            if scratch.is_dir() and not scratch.is_symlink():
                raise ArtifactIntegrityError(
                    f"geometry index scratch is an unexpected directory: {scratch}"
                )
            scratch.unlink(missing_ok=True)
        if scratch_directory.exists():
            shutil.rmtree(scratch_directory)


def _iter_verified_geometry_sidecars(
    output_dir: os.PathLike[str] | str,
    *,
    verify_checksums: bool,
    _reconcile: bool = True,
) -> Iterator[dict[str, Any]]:
    directory = Path(output_dir)
    if _reconcile:
        reconcile_geometry_artifacts(
            directory,
            verify_checksums=verify_checksums,
        )
    artifacts = {path.stem: path for path in directory.glob("shard_*.npz")}
    sidecar_paths = {
        path.stem: path for path in directory.glob("shard_*.json")
    }
    if set(artifacts) != set(sidecar_paths):
        raise ArtifactIntegrityError(
            "incomplete geometry shard pairs after reconciliation; "
            f"missing_artifacts={sorted(set(sidecar_paths) - set(artifacts))}, "
            f"missing_sidecars={sorted(set(artifacts) - set(sidecar_paths))}"
        )
    metadata_fields = {
        "schema",
        "shard_id",
        "filename",
        "record_count",
        "records",
        "sha256",
    }
    record_metadata_fields = {
        "key",
        "row_index",
        "source_index",
        "sdf_ordinal",
        "train_ordinal",
        "smiles",
        "smiles_hash",
        "canonical_smiles",
        "num_atoms",
        "num_conformers",
        "reason",
    }
    array_fields = {
        "atomic_numbers",
        "coords",
        "energies",
        "energy_mask",
        "conformer_mask",
        "sources",
        "conformer_source",
        "heavy_atom_indices",
        "heavy_atom_mapping",
        "canonical_smiles",
        "reason",
    }
    for stem in sorted(artifacts):
        sidecar_path = sidecar_paths[stem]
        shard_id_text = stem.removeprefix("shard_")
        if (
            not shard_id_text.isdigit()
            or len(shard_id_text) != 6
        ):
            raise ArtifactIntegrityError(
                f"non-canonical geometry shard name: {stem}"
            )
        expected_shard_id = int(shard_id_text)
        try:
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                f"invalid geometry sidecar: {sidecar_path}"
            ) from exc
        if not isinstance(metadata, dict) or set(metadata) != metadata_fields:
            raise ArtifactIntegrityError(
                f"geometry sidecar fields differ from schema: {sidecar_path}"
            )
        artifact = artifacts[stem]
        records = metadata.get("records")
        raw_shard_id = metadata.get("shard_id")
        raw_record_count = metadata.get("record_count")
        checksum = metadata.get("sha256")
        if (
            metadata.get("schema") != GEOMETRY_SCHEMA
            or not isinstance(raw_shard_id, int)
            or isinstance(raw_shard_id, bool)
            or raw_shard_id != expected_shard_id
            or metadata.get("filename") != artifact.name
            or not isinstance(records, list)
            or not records
            or not isinstance(raw_record_count, int)
            or isinstance(raw_record_count, bool)
            or raw_record_count != len(records)
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(
                character not in "0123456789abcdef"
                for character in checksum
            )
        ):
            raise ArtifactIntegrityError(
                f"invalid geometry sidecar contract: {sidecar_path}"
            )
        expected_stem = f"shard_{expected_shard_id:06d}"
        if sidecar_path.stem != expected_stem or artifact.stem != expected_stem:
            raise ArtifactIntegrityError(
                f"geometry shard naming mismatch: {sidecar_path}"
            )
        if verify_checksums and checksum != _sha256_file(artifact):
            raise ArtifactIntegrityError(
                f"checksum mismatch for geometry shard: {artifact}"
            )
        seen_rows: set[int] = set()
        seen_sources: set[int] = set()
        seen_ordinals: set[int] = set()
        for record_ordinal, entry in enumerate(records):
            if (
                not isinstance(entry, dict)
                or set(entry) != record_metadata_fields
                or entry.get("key") != f"r{record_ordinal:06d}"
            ):
                raise ArtifactIntegrityError(
                    f"invalid record metadata in {sidecar_path}"
                )
            try:
                row_index = _strict_identity_integer(
                    entry["row_index"],
                    field="row_index",
                )
                source_index = _strict_identity_integer(
                    entry["source_index"],
                    field="source_index",
                )
                sdf_ordinal = (
                    None
                    if entry["sdf_ordinal"] is None
                    else _strict_identity_integer(
                        entry["sdf_ordinal"],
                        field="sdf_ordinal",
                    )
                )
                train_ordinal = (
                    None
                    if entry["train_ordinal"] is None
                    else _strict_identity_integer(
                        entry["train_ordinal"],
                        field="train_ordinal",
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    f"invalid record identity in {sidecar_path}"
                ) from exc
            smiles = entry.get("smiles")
            canonical_smiles = entry.get("canonical_smiles")
            num_atoms = entry.get("num_atoms")
            num_conformers = entry.get("num_conformers")
            if (
                row_index in seen_rows
                or source_index in seen_sources
                or sdf_ordinal != train_ordinal
                or (
                    sdf_ordinal is not None
                    and sdf_ordinal in seen_ordinals
                )
                or not isinstance(smiles, str)
                or not smiles
                or entry.get("smiles_hash") != smiles_hash(smiles)
                or not isinstance(canonical_smiles, str)
                or not canonical_smiles
                or not isinstance(num_atoms, int)
                or isinstance(num_atoms, bool)
                or num_atoms <= 0
                or not isinstance(num_conformers, int)
                or isinstance(num_conformers, bool)
                or num_conformers <= 0
                or not isinstance(entry.get("reason"), str)
            ):
                raise ArtifactIntegrityError(
                    f"invalid record metadata in {sidecar_path}"
                )
            seen_rows.add(row_index)
            seen_sources.add(source_index)
            if sdf_ordinal is not None:
                seen_ordinals.add(sdf_ordinal)
        try:
            with np.load(artifact, allow_pickle=False) as arrays:
                available = set(arrays.files)
                expected_arrays = {
                    f"{entry['key']}__{field}"
                    for entry in records
                    for field in array_fields
                }
                expected_arrays.add(EMBEDDED_SHARD_METADATA_KEY)
                if available != expected_arrays:
                    raise ArtifactIntegrityError(
                        f"geometry artifact inventory differs: {artifact}; "
                        f"missing={sorted(expected_arrays - available)}, "
                        f"unexpected={sorted(available - expected_arrays)}"
                    )
                embedded_array = arrays[EMBEDDED_SHARD_METADATA_KEY]
                if (
                    embedded_array.shape != ()
                    or embedded_array.dtype.kind != "U"
                ):
                    raise ArtifactIntegrityError(
                        f"invalid embedded geometry metadata: {artifact}"
                    )
                embedded = json.loads(str(embedded_array.item()))
                core_metadata = {
                    key: value
                    for key, value in metadata.items()
                    if key != "sha256"
                }
                if embedded != core_metadata:
                    raise ArtifactIntegrityError(
                        f"embedded geometry metadata mismatch: {artifact}"
                    )
        except ArtifactIntegrityError:
            raise
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"cannot inspect geometry shard: {artifact}"
            ) from exc
        yield metadata


def _validate_geometry_index_pair(
    directory: Path,
    *,
    verify_checksums: bool,
) -> tuple[list[dict[str, Any]], int]:
    index_path = directory / "geometry_index.npz"
    metadata_path = directory / "geometry_index.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"invalid geometry index metadata: {metadata_path}"
        ) from exc
    metadata_fields = {
        "schema",
        "filename",
        "sha256",
        "record_count",
        "sorted_by",
        "lookup",
    }
    checksum = metadata.get("sha256") if isinstance(metadata, dict) else None
    record_count = (
        metadata.get("record_count") if isinstance(metadata, dict) else None
    )
    if (
        not isinstance(metadata, dict)
        or set(metadata) != metadata_fields
        or metadata.get("schema") != GEOMETRY_INDEX_SCHEMA
        or metadata.get("filename") != index_path.name
        or not isinstance(checksum, str)
        or len(checksum) != 64
        or set(checksum) - set("0123456789abcdef")
        or not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count < 0
        or metadata.get("sorted_by") != ["source_index", "row_index"]
        or metadata.get("lookup")
        != "numpy.searchsorted(source_index, requested_source_index)"
        or (
            verify_checksums
            and _sha256_file(index_path) != checksum
        )
    ):
        raise ArtifactIntegrityError("geometry index metadata contract is invalid")

    scratch_fd, scratch_name = tempfile.mkstemp(
        prefix=".geometry_index_validate-",
        suffix=".sqlite3",
        dir=directory,
    )
    os.close(scratch_fd)
    scratch_path = Path(scratch_name)
    connection = sqlite3.connect(scratch_path)
    loaded: dict[str, np.ndarray] = {}
    verified_summaries: list[dict[str, Any]] = []
    expected_count = 0
    try:
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE expected_locators ("
            "source_index INTEGER PRIMARY KEY,"
            "row_index INTEGER NOT NULL UNIQUE,"
            "sdf_ordinal INTEGER NOT NULL,"
            "shard_id INTEGER NOT NULL,"
            "record_ordinal INTEGER NOT NULL,"
            "UNIQUE(shard_id,record_ordinal)"
            ")"
        )
        connection.execute(
            "CREATE UNIQUE INDEX expected_unique_sdf_ordinal "
            "ON expected_locators(sdf_ordinal) WHERE sdf_ordinal >= 0"
        )
        for sidecar in _iter_verified_geometry_sidecars(
            directory,
            verify_checksums=verify_checksums,
            _reconcile=False,
        ):
            shard_id = int(sidecar["shard_id"])
            rows = [
                (
                    int(entry["source_index"]),
                    int(entry["row_index"]),
                    (
                        -1
                        if entry["sdf_ordinal"] is None
                        else int(entry["sdf_ordinal"])
                    ),
                    shard_id,
                    record_ordinal,
                )
                for record_ordinal, entry in enumerate(sidecar["records"])
            ]
            try:
                with connection:
                    connection.executemany(
                        "INSERT INTO expected_locators VALUES (?,?,?,?,?)",
                        rows,
                    )
            except sqlite3.IntegrityError as exc:
                raise ArtifactIntegrityError(
                    "geometry sidecars contain duplicate index identities"
                ) from exc
            expected_count += len(rows)
            verified_summaries.append(
                {
                    "shard_id": sidecar["shard_id"],
                    "filename": sidecar["filename"],
                    "sha256": sidecar["sha256"],
                    "record_count": sidecar["record_count"],
                }
            )
        if record_count != expected_count:
            raise ArtifactIntegrityError(
                "geometry index count differs from verified shard inventory"
            )

        required_arrays = {
            "source_index",
            "row_index",
            "sdf_ordinal",
            "shard_id",
            "record_ordinal",
        }
        try:
            with np.load(index_path, allow_pickle=False) as arrays:
                if set(arrays.files) != required_arrays:
                    raise ArtifactIntegrityError(
                        "geometry index array inventory differs from schema"
                    )
                loaded = {
                    name: np.asarray(arrays[name])
                    for name in required_arrays
                }
        except ArtifactIntegrityError:
            raise
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"cannot inspect geometry index: {index_path}"
            ) from exc
        if any(array.ndim != 1 for array in loaded.values()):
            raise ArtifactIntegrityError(
                "geometry index arrays must be one-dimensional"
            )
        if any(array.size != record_count for array in loaded.values()):
            raise ArtifactIntegrityError(
                "geometry index arrays have unequal lengths"
            )
        expected_dtypes = {
            "source_index": np.dtype(np.int64),
            "row_index": np.dtype(np.int64),
            "sdf_ordinal": np.dtype(np.int64),
            "shard_id": np.dtype(np.int32),
            "record_ordinal": np.dtype(np.int32),
        }
        for name, expected_dtype in expected_dtypes.items():
            if loaded[name].dtype != expected_dtype:
                raise ArtifactIntegrityError(
                    f"geometry index {name} dtype differs from schema"
                )
        if (
            np.any(loaded["source_index"] < 0)
            or np.any(loaded["row_index"] < 0)
            or np.any(loaded["sdf_ordinal"] < -1)
            or np.any(loaded["shard_id"] < 0)
            or np.any(loaded["record_ordinal"] < 0)
        ):
            raise ArtifactIntegrityError(
                "geometry index contains an out-of-range identity"
            )
        if record_count > 1 and np.any(
            loaded["source_index"][1:] <= loaded["source_index"][:-1]
        ):
            raise ArtifactIntegrityError(
                "geometry index source_index must be strictly sorted and unique"
            )

        cursor = connection.execute(
            "SELECT source_index,row_index,sdf_ordinal,shard_id,"
            "record_ordinal FROM expected_locators "
            "ORDER BY source_index,row_index"
        )
        offset = 0
        while True:
            rows = cursor.fetchmany(65_536)
            if not rows:
                break
            end = offset + len(rows)
            columns = tuple(zip(*rows))
            for column_index, name in enumerate(
                (
                    "source_index",
                    "row_index",
                    "sdf_ordinal",
                    "shard_id",
                    "record_ordinal",
                )
            ):
                expected_values = np.asarray(
                    columns[column_index],
                    dtype=expected_dtypes[name],
                )
                if not np.array_equal(
                    loaded[name][offset:end],
                    expected_values,
                ):
                    raise ArtifactIntegrityError(
                        "geometry index locator differs from sidecars at "
                        f"position {offset}"
                    )
            offset = end
        if offset != record_count:
            raise ArtifactIntegrityError(
                "geometry index does not cover the complete shard inventory"
            )
    finally:
        connection.close()
        for candidate in (
            scratch_path,
            Path(f"{scratch_path}-journal"),
            Path(f"{scratch_path}-wal"),
            Path(f"{scratch_path}-shm"),
        ):
            candidate.unlink(missing_ok=True)
    return verified_summaries, expected_count


def load_geometry_by_source_index(
    output_dir: os.PathLike[str] | str,
    source_index: int,
    occurrence: int = 0,
    verify_checksums: bool = True,
) -> GeometryRecord:
    """Random-read one geometry through the sorted ``geometry_index.npz``."""
    if occurrence < 0:
        raise ValueError("occurrence must be non-negative")
    directory = Path(output_dir)
    index_metadata_path = directory / "geometry_index.json"
    if not index_metadata_path.is_file():
        raise FileNotFoundError(f"geometry index missing: {index_metadata_path}")
    index_metadata = json.loads(index_metadata_path.read_text(encoding="utf-8"))
    index_checksum = (
        index_metadata.get("sha256")
        if isinstance(index_metadata, dict)
        else None
    )
    record_count = (
        index_metadata.get("record_count")
        if isinstance(index_metadata, dict)
        else None
    )
    if (
        not isinstance(index_metadata, dict)
        or set(index_metadata)
        != {
            "schema",
            "filename",
            "sha256",
            "record_count",
            "sorted_by",
            "lookup",
        }
        or index_metadata.get("schema") != GEOMETRY_INDEX_SCHEMA
        or index_metadata.get("filename") != "geometry_index.npz"
        or not isinstance(index_checksum, str)
        or len(index_checksum) != 64
        or set(index_checksum) - set("0123456789abcdef")
        or not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count < 0
        or index_metadata.get("sorted_by") != ["source_index", "row_index"]
        or index_metadata.get("lookup")
        != "numpy.searchsorted(source_index, requested_source_index)"
    ):
        raise RuntimeError("unsupported or invalid geometry index schema")
    index_path = directory / "geometry_index.npz"
    if verify_checksums and _sha256_file(index_path) != index_metadata["sha256"]:
        raise RuntimeError(f"checksum mismatch for geometry index {index_path}")

    with np.load(index_path, allow_pickle=False) as index:
        required_arrays = {
            "source_index",
            "row_index",
            "sdf_ordinal",
            "shard_id",
            "record_ordinal",
        }
        if set(index.files) != required_arrays:
            raise RuntimeError("geometry index array inventory differs from schema")
        indexed_arrays = {
            name: np.asarray(index[name])
            for name in required_arrays
        }
        expected_dtypes = {
            "source_index": np.dtype(np.int64),
            "row_index": np.dtype(np.int64),
            "sdf_ordinal": np.dtype(np.int64),
            "shard_id": np.dtype(np.int32),
            "record_ordinal": np.dtype(np.int32),
        }
        if any(
            indexed_arrays[name].dtype != expected_dtypes[name]
            or indexed_arrays[name].ndim != 1
            or indexed_arrays[name].size != record_count
            for name in required_arrays
        ):
            raise RuntimeError("geometry index array shape/dtype differs from schema")
        source_indices = indexed_arrays["source_index"]
        if record_count > 1 and np.any(
            source_indices[1:] <= source_indices[:-1]
        ):
            raise RuntimeError(
                "geometry index source_index is not strictly sorted"
            )
        left = int(np.searchsorted(source_indices, int(source_index), side="left"))
        right = int(np.searchsorted(source_indices, int(source_index), side="right"))
        position = left + occurrence
        if left == right or position >= right:
            raise KeyError(
                f"source_index={source_index}, occurrence={occurrence} not found"
            )
        shard_id = int(indexed_arrays["shard_id"][position])
        record_ordinal = int(indexed_arrays["record_ordinal"][position])
        indexed_row = int(indexed_arrays["row_index"][position])
        indexed_ordinal = int(indexed_arrays["sdf_ordinal"][position])
    if (
        shard_id < 0
        or record_ordinal < 0
        or indexed_row < 0
        or indexed_ordinal < -1
    ):
        raise RuntimeError("geometry index contains an invalid shard locator")
    artifact = f"shard_{shard_id:06d}.npz"
    key = f"r{record_ordinal:06d}"
    sidecar_path = directory / f"shard_{shard_id:06d}.json"
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"cannot resolve geometry sidecar locator: {sidecar_path}"
        ) from exc
    records = sidecar.get("records") if isinstance(sidecar, dict) else None
    if (
        not isinstance(records, list)
        or record_ordinal >= len(records)
        or not isinstance(records[record_ordinal], dict)
    ):
        raise RuntimeError(
            f"cannot resolve geometry sidecar locator: {sidecar_path}"
        )
    entry = records[record_ordinal]
    expected_sdf_ordinal = (
        None if indexed_ordinal == -1 else indexed_ordinal
    )
    if (
        set(sidecar)
        != {
            "schema",
            "shard_id",
            "filename",
            "record_count",
            "records",
            "sha256",
        }
        or set(entry)
        != {
            "key",
            "row_index",
            "source_index",
            "sdf_ordinal",
            "train_ordinal",
            "smiles",
            "smiles_hash",
            "canonical_smiles",
            "num_atoms",
            "num_conformers",
            "reason",
        }
        or sidecar.get("schema") != GEOMETRY_SCHEMA
        or sidecar.get("shard_id") != shard_id
        or sidecar.get("filename") != artifact
        or sidecar.get("record_count") != len(records)
        or entry.get("key") != key
        or entry.get("source_index") != int(source_index)
        or entry.get("row_index") != indexed_row
        or entry.get("sdf_ordinal") != expected_sdf_ordinal
        or entry.get("train_ordinal") != expected_sdf_ordinal
    ):
        raise RuntimeError("geometry index locator differs from its shard sidecar")
    expected_artifact_sha = sidecar.get("sha256")
    if (
        not isinstance(expected_artifact_sha, str)
        or len(expected_artifact_sha) != 64
        or set(expected_artifact_sha) - set("0123456789abcdef")
    ):
        raise RuntimeError("geometry shard sidecar checksum is invalid")
    artifact_path = directory / artifact
    if verify_checksums and _sha256_file(artifact_path) != expected_artifact_sha:
        raise RuntimeError(f"checksum mismatch for geometry artifact {artifact_path}")
    with np.load(artifact_path, allow_pickle=False) as arrays:
        return GeometryRecord.from_storage_dict(arrays, prefix=f"{key}__")


def _read_table(path: os.PathLike[str] | str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported table format: {path}")


def _iter_table_chunks(
    path: os.PathLike[str] | str,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    source = Path(path)
    suffix = source.suffix.lower()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if suffix == ".csv":
        yield from pd.read_csv(source, chunksize=chunk_size)
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("streaming Parquet input requires pyarrow") from exc
        parquet_file = parquet.ParquetFile(source, memory_map=True)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()
        return
    if suffix in {".jsonl", ".ndjson"}:
        yield from pd.read_json(source, lines=True, chunksize=chunk_size)
        return
    if suffix == ".json":
        raise ValueError(
            "streaming CLI does not accept JSON arrays; convert to JSONL or Parquet"
        )
    raise ValueError(f"unsupported table format: {source}")


def _optional_integer(
    value: Any,
    *,
    field: str,
    row_number: int,
) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be an integer at row {row_number}")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} must be an integer at row {row_number}: {value!r}"
        ) from exc
    if isinstance(value, (float, np.floating)) and float(value) != normalized:
        raise ValueError(
            f"{field} must be integral at row {row_number}: {value!r}"
        )
    if not isinstance(value, (str, bytes, float, np.floating)):
        try:
            is_integral = bool(value == normalized)
        except (TypeError, ValueError):
            is_integral = False
        if not is_integral:
            raise ValueError(
                f"{field} must be integral at row {row_number}: {value!r}"
            )
    if normalized < -(2**63) or normalized > 2**63 - 1:
        raise ValueError(
            f"{field} exceeds SQLite int64 at row {row_number}: {normalized}"
        )
    return normalized


def _ordinal_columns(
    columns: Iterable[Any],
    explicit_column: Optional[str],
) -> list[str]:
    available = {str(column) for column in columns}
    candidates = [
        name
        for name in (explicit_column, "sdf_ordinal", "train_ordinal")
        if name and name in available
    ]
    return list(dict.fromkeys(candidates))


def _chunk_ordinal(
    row: tuple[Any, ...],
    candidates: list[str],
    *,
    column_positions: dict[str, int],
    row_number: int,
) -> Optional[int]:
    values = [
        _optional_integer(
            row[column_positions[column]],
            field=column,
            row_number=row_number,
        )
        for column in candidates
    ]
    if any(value is not None and value < -1 for value in values):
        raise ValueError(
            f"SDF ordinals must be -1 or non-negative at row {row_number}"
        )
    present = [
        value
        for value in values
        if value is not None and value != -1
    ]
    if present and any(value != present[0] for value in present[1:]):
        raise ValueError(
            f"conflicting SDF ordinal columns {candidates} at row {row_number}"
        )
    return present[0] if present else None


def _open_work_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _insert_input_chunks(
    connection: sqlite3.Connection,
    *,
    input_path: os.PathLike[str] | str,
    smiles_col: str,
    source_index_col: Optional[str],
    sdf_ordinal_col: Optional[str],
    chunk_size: int,
) -> None:
    sequence = 0
    for frame in _iter_table_chunks(input_path, chunk_size):
        if smiles_col not in frame.columns:
            raise ValueError(f"missing SMILES column {smiles_col!r}")
        if source_index_col and source_index_col not in frame.columns:
            raise ValueError(
                f"missing source-index column {source_index_col!r}"
            )
        ordinal_columns = _ordinal_columns(frame.columns, sdf_ordinal_col)
        column_positions = {
            str(column): position
            for position, column in enumerate(frame.columns)
        }
        if len(column_positions) != len(frame.columns):
            raise ValueError("input table contains duplicate column names")
        rows: list[tuple[int, int, int, str, Optional[int]]] = []
        for row in frame.itertuples(index=False, name=None):
            source_value = (
                row[column_positions[source_index_col]]
                if source_index_col
                else row[column_positions["source_index"]]
                if "source_index" in frame.columns
                else sequence
            )
            source_index = _optional_integer(
                source_value,
                field="source_index",
                row_number=sequence,
            )
            if source_index is None or source_index < 0:
                raise ValueError(
                    f"source_index must be non-negative at row {sequence}"
                )
            row_value = (
                row[column_positions["row_index"]]
                if "row_index" in frame.columns
                else source_index
            )
            row_index = _optional_integer(
                row_value,
                field="row_index",
                row_number=sequence,
            )
            if row_index is None or row_index < 0:
                raise ValueError(
                    f"row_index must be non-negative at row {sequence}"
                )
            smiles_value = row[column_positions[smiles_col]]
            if pd.isna(smiles_value) or not str(smiles_value).strip():
                raise ValueError(f"SMILES is missing at row {sequence}")
            rows.append(
                (
                    sequence,
                    row_index,
                    source_index,
                    str(smiles_value),
                    _chunk_ordinal(
                        row,
                        ordinal_columns,
                        column_positions=column_positions,
                        row_number=sequence,
                    ),
                )
            )
            sequence += 1
        try:
            with connection:
                connection.executemany(
                    "INSERT INTO input_rows "
                    "(sequence,row_index,source_index,smiles,sdf_ordinal) "
                    "VALUES (?,?,?,?,?)",
                    rows,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "input source_index/row_index values must be unique"
            ) from exc


def _insert_manifest_chunks(
    connection: sqlite3.Connection,
    *,
    manifest_path: os.PathLike[str] | str,
    smiles_col: str,
    sdf_ordinal_col: Optional[str],
    chunk_size: int,
) -> None:
    sequence = 0
    for frame in _iter_table_chunks(manifest_path, chunk_size):
        if "source_index" not in frame.columns:
            raise ValueError("manifest must contain source_index")
        ordinal_columns = _ordinal_columns(frame.columns, sdf_ordinal_col)
        has_smiles = smiles_col in frame.columns
        column_positions = {
            str(column): position
            for position, column in enumerate(frame.columns)
        }
        if len(column_positions) != len(frame.columns):
            raise ValueError("manifest contains duplicate column names")
        rows: list[tuple[int, int, Optional[str], Optional[int]]] = []
        for row in frame.itertuples(index=False, name=None):
            source_index = _optional_integer(
                row[column_positions["source_index"]],
                field="manifest.source_index",
                row_number=sequence,
            )
            if source_index is None or source_index < 0:
                raise ValueError(
                    "manifest source_index must be non-negative at "
                    f"row {sequence}"
                )
            manifest_smiles: Optional[str] = None
            if has_smiles:
                value = row[column_positions[smiles_col]]
                if not pd.isna(value) and str(value).strip():
                    manifest_smiles = str(value)
            rows.append(
                (
                    sequence,
                    source_index,
                    manifest_smiles,
                    _chunk_ordinal(
                        row,
                        ordinal_columns,
                        column_positions=column_positions,
                        row_number=sequence,
                    ),
                )
            )
            sequence += 1
        try:
            with connection:
                connection.executemany(
                    "INSERT INTO selected "
                    "(sequence,source_index,smiles,sdf_ordinal) VALUES (?,?,?,?)",
                    rows,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("manifest source_index values must be unique") from exc


def _initialize_work_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE input_rows (
            sequence INTEGER PRIMARY KEY,
            row_index INTEGER NOT NULL UNIQUE,
            source_index INTEGER NOT NULL UNIQUE,
            smiles TEXT NOT NULL,
            sdf_ordinal INTEGER
        );
        CREATE TABLE selected (
            sequence INTEGER PRIMARY KEY,
            source_index INTEGER NOT NULL UNIQUE,
            smiles TEXT,
            sdf_ordinal INTEGER
        );
        CREATE TABLE work_items (
            sequence INTEGER PRIMARY KEY,
            row_index INTEGER NOT NULL UNIQUE,
            source_index INTEGER NOT NULL UNIQUE,
            smiles TEXT NOT NULL,
            sdf_ordinal INTEGER
        );
        CREATE TABLE completed (
            row_index INTEGER PRIMARY KEY
                REFERENCES work_items(row_index) ON DELETE CASCADE
        );
        CREATE TABLE failures (
            row_index INTEGER PRIMARY KEY
                REFERENCES work_items(row_index) ON DELETE CASCADE,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX work_sdf_order
            ON work_items(sdf_ordinal, sequence);
        CREATE TRIGGER completed_must_not_be_failed
        BEFORE INSERT ON completed
        WHEN EXISTS (
            SELECT 1 FROM failures WHERE row_index = NEW.row_index
        )
        BEGIN
            SELECT RAISE(ABORT, 'work item already has a failure outcome');
        END;
        CREATE TRIGGER failure_must_not_be_completed
        BEFORE INSERT ON failures
        WHEN EXISTS (
            SELECT 1 FROM completed WHERE row_index = NEW.row_index
        )
        BEGIN
            SELECT RAISE(ABORT, 'work item already has a completed outcome');
        END;
        """
    )


def _work_queue_identity(
    connection: sqlite3.Connection,
) -> tuple[int, str]:
    """Return a canonical streaming identity for every immutable work item."""

    digest = hashlib.sha256()
    count = 0
    cursor = connection.execute(
        "SELECT sequence,row_index,source_index,smiles,sdf_ordinal "
        "FROM work_items ORDER BY sequence"
    )
    while True:
        rows = cursor.fetchmany(4096)
        if not rows:
            break
        for sequence, row_index, source_index, smiles, sdf_ordinal in rows:
            if int(sequence) != count:
                raise ResumeStateError(
                    "work item sequence is not contiguous and deterministic"
                )
            payload = [
                int(sequence),
                int(row_index),
                int(source_index),
                str(smiles),
                None if sdf_ordinal is None else int(sdf_ordinal),
            ]
            digest.update(_canonical_json(payload).encode("utf-8"))
            digest.update(b"\n")
            count += 1
    return count, digest.hexdigest()


def _validate_work_database(
    connection: sqlite3.Connection,
    *,
    fingerprint_sha256: str,
) -> None:
    integrity_rows = connection.execute("PRAGMA quick_check").fetchall()
    if integrity_rows != [("ok",)]:
        raise ResumeStateError("work database failed SQLite integrity checks")
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    except sqlite3.DatabaseError as exc:
        raise ResumeStateError("work database metadata cannot be read") from exc
    expected_metadata_fields = {
        "schema",
        "fingerprint_sha256",
        "status",
        "work_item_count",
        "work_items_sha256",
    }
    if (
        set(metadata) != expected_metadata_fields
        or metadata.get("schema") != WORK_DATABASE_SCHEMA
        or metadata.get("fingerprint_sha256") != fingerprint_sha256
        or metadata.get("status") != "complete"
        or not str(metadata.get("work_item_count", "")).isdigit()
        or not isinstance(metadata.get("work_items_sha256"), str)
        or len(metadata["work_items_sha256"]) != 64
        or set(metadata["work_items_sha256"])
        - set("0123456789abcdef")
    ):
        raise ResumeStateError("work database metadata/fingerprint mismatch")
    count, checksum = _work_queue_identity(connection)
    if (
        metadata["work_item_count"] != str(count)
        or metadata["work_items_sha256"] != checksum
    ):
        raise ResumeStateError(
            "work database queue differs from its persisted identity"
        )


def _work_staging_candidates(path: Path) -> Iterator[Path]:
    for pattern in (
        f".{path.name}.tmp-*",
        f".{path.name}.inputs-*",
    ):
        yield from path.parent.glob(pattern)


def _remove_stale_work_staging(path: Path) -> None:
    parent = path.parent.resolve()
    for candidate in _work_staging_candidates(path):
        resolved = candidate.resolve()
        if resolved.parent != parent:
            raise ResumeStateError(
                f"refusing to remove work staging outside {parent}: {resolved}"
            )
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _assert_no_work_staging(path: Path) -> None:
    leftovers = sorted(
        candidate.name
        for candidate in _work_staging_candidates(path)
    )
    if leftovers:
        raise ResumeStateError(
            f"work database staging artifacts remain: {leftovers}"
        )


def _validate_persisted_failures(
    connection: sqlite3.Connection,
) -> None:
    """Validate the SQLite failure ledger without trusting derived JSONL."""

    for persisted_row_index, payload_json in connection.execute(
        "SELECT row_index,payload_json FROM failures ORDER BY row_index"
    ):
        try:
            failure = json.loads(str(payload_json))
            row_index, canonical_payload = _validated_failure_payload(
                connection,
                failure,
                context=f"work database failure row {persisted_row_index}",
            )
        except (
            ArtifactIntegrityError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            raise ResumeStateError(
                "work database contains an invalid failure payload for "
                f"row_index={persisted_row_index}"
            ) from exc
        if (
            row_index != int(persisted_row_index)
            or canonical_payload != str(payload_json)
        ):
            raise ResumeStateError(
                "work database failure payload is not canonical for "
                f"row_index={persisted_row_index}"
            )


def _validated_failure_payload(
    connection: sqlite3.Connection,
    failure: Any,
    *,
    context: str,
) -> tuple[int, str]:
    fields = {
        "row_index",
        "source_index",
        "sdf_ordinal",
        "smiles",
        "stage",
        "error_type",
        "message",
    }
    if not isinstance(failure, dict) or set(failure) != fields:
        raise ArtifactIntegrityError(
            f"failure payload fields differ from schema: {context}"
        )
    row_index = _strict_identity_integer(
        failure["row_index"],
        field=f"{context}.row_index",
    )
    source_index = _strict_identity_integer(
        failure["source_index"],
        field=f"{context}.source_index",
    )
    raw_ordinal = failure["sdf_ordinal"]
    sdf_ordinal = (
        None
        if raw_ordinal is None
        else _strict_identity_integer(
            raw_ordinal,
            field=f"{context}.sdf_ordinal",
        )
    )
    smiles = failure["smiles"]
    if (
        not isinstance(smiles, str)
        or not smiles
        or not isinstance(failure["stage"], str)
        or not failure["stage"]
        or not isinstance(failure["error_type"], str)
        or not failure["error_type"]
        or not isinstance(failure["message"], str)
    ):
        raise ArtifactIntegrityError(
            f"failure payload strings are invalid: {context}"
        )
    work_identity = connection.execute(
        "SELECT source_index,smiles,sdf_ordinal FROM work_items "
        "WHERE row_index=?",
        (row_index,),
    ).fetchone()
    if work_identity is None:
        raise ArtifactIntegrityError(
            f"failure row is absent from work items: {context}"
        )
    expected_ordinal = (
        None if work_identity[2] is None else int(work_identity[2])
    )
    if (
        int(work_identity[0]) != source_index
        or str(work_identity[1]) != smiles
        or expected_ordinal != sdf_ordinal
    ):
        raise ArtifactIntegrityError(
            f"failure identity differs from work item: {context}"
        )
    if connection.execute(
        "SELECT 1 FROM completed WHERE row_index=?",
        (row_index,),
    ).fetchone() is not None:
        raise ArtifactIntegrityError(
            f"failure row already has a completed outcome: {context}"
        )
    normalized = {
        **failure,
        "row_index": row_index,
        "source_index": source_index,
        "sdf_ordinal": sdf_ordinal,
    }
    return row_index, _canonical_json(normalized)


def _copy_verified_input_snapshot(
    source_path: os.PathLike[str] | str,
    destination: Path,
    *,
    expected_descriptor: Optional[Mapping[str, Any]],
    label: str,
) -> None:
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"{label} input is missing: {source}")
    if expected_descriptor is not None:
        if (
            not isinstance(expected_descriptor, Mapping)
            or set(expected_descriptor) != {"name", "size", "sha256"}
            or expected_descriptor.get("name") != source.name
            or not isinstance(expected_descriptor.get("size"), int)
            or isinstance(expected_descriptor["size"], bool)
            or expected_descriptor["size"] < 0
            or not isinstance(expected_descriptor.get("sha256"), str)
            or len(expected_descriptor["sha256"]) != 64
            or set(expected_descriptor["sha256"])
            - set("0123456789abcdef")
        ):
            raise ResumeStateError(
                f"{label} run-fingerprint descriptor is invalid"
            )
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as source_handle, destination.open("xb") as target:
        while True:
            chunk = source_handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        target.flush()
        os.fsync(target.fileno())
    observed_descriptor = {
        "name": source.name,
        "size": size,
        "sha256": digest.hexdigest(),
    }
    if (
        expected_descriptor is not None
        and _canonical_json(observed_descriptor)
        != _canonical_json(dict(expected_descriptor))
    ):
        raise ResumeStateError(
            f"{label} changed before its work-database snapshot completed"
        )


def _official_sdf_snapshot_paths(
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    return (
        output_dir / ".official_sdf.snapshot.sdf",
        output_dir / ".official_sdf.snapshot.json",
        output_dir / ".official_sdf.snapshot.poison.json",
    )


def _remove_official_sdf_temporaries(output_dir: Path) -> None:
    for pattern in (
        ".official_sdf.snapshot.copy-*.tmp",
        "..official_sdf.snapshot.sdf.*.tmp",
        "..official_sdf.snapshot.json.*.tmp",
        "..official_sdf.snapshot.poison.json.*.tmp",
    ):
        for candidate in output_dir.glob(pattern):
            if candidate.resolve().parent != output_dir.resolve():
                raise ResumeStateError(
                    f"official SDF temporary escaped output directory: {candidate}"
                )
            candidate.unlink(missing_ok=True)
    _fsync_directory(output_dir)


def _verify_official_sdf_snapshot(
    output_dir: Path,
    expected_descriptor: Mapping[str, Any],
) -> Path:
    snapshot_path, metadata_path, poison_path = (
        _official_sdf_snapshot_paths(output_dir)
    )
    if poison_path.exists():
        raise ResumeStateError(
            f"official SDF snapshot is poisoned: {poison_path}"
        )
    if not snapshot_path.is_file() or not metadata_path.is_file():
        raise ResumeStateError("official SDF snapshot pair is incomplete")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeStateError(
            f"official SDF snapshot metadata is invalid: {metadata_path}"
        ) from exc
    expected_metadata = {
        "schema": OFFICIAL_SDF_SNAPSHOT_SCHEMA,
        "source": dict(expected_descriptor),
    }
    if _canonical_json(metadata) != _canonical_json(expected_metadata):
        raise ResumeStateError(
            "official SDF snapshot metadata differs from the run fingerprint"
        )
    _verify_official_sdf_snapshot_bytes(
        snapshot_path,
        expected_descriptor,
    )
    return snapshot_path


def _verify_official_sdf_snapshot_bytes(
    snapshot_path: Path,
    expected_descriptor: Mapping[str, Any],
) -> None:
    digest = hashlib.sha256()
    try:
        before = snapshot_path.stat()
        with snapshot_path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
            opened_after = os.fstat(handle.fileno())
        after = snapshot_path.stat()
    except OSError as exc:
        raise ResumeStateError(
            f"cannot inspect official SDF snapshot: {snapshot_path}"
        ) from exc
    if (
        not os.path.samestat(before, opened_before)
        or not os.path.samestat(opened_before, opened_after)
        or not os.path.samestat(opened_after, after)
        or opened_before.st_size != opened_after.st_size
        or opened_before.st_mtime_ns != opened_after.st_mtime_ns
        or opened_after.st_size != expected_descriptor.get("size")
        or digest.hexdigest() != expected_descriptor.get("sha256")
    ):
        raise ResumeStateError(
            "official SDF snapshot bytes or file identity changed"
        )


def _poison_official_sdf_snapshot(
    output_dir: Path,
    reason: str,
) -> None:
    _, _, poison_path = _official_sdf_snapshot_paths(output_dir)
    _atomic_write_text(
        poison_path,
        json.dumps(
            {
                "schema": OFFICIAL_SDF_SNAPSHOT_SCHEMA,
                "reason": str(reason),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )


def _has_geometry_derivatives(output_dir: Path) -> bool:
    if (
        next(output_dir.glob("shard_*.npz"), None) is not None
        or next(output_dir.glob("shard_*.json"), None) is not None
        or (output_dir / "geometry_index.npz").exists()
        or (output_dir / "geometry_index.json").exists()
        or (output_dir / "manifest.json").exists()
    ):
        return True
    failure_path = output_dir / "failures.jsonl"
    try:
        if failure_path.is_file() and failure_path.stat().st_size > 0:
            return True
    except OSError:
        return True
    database_path = output_dir / "geometry_work.sqlite3"
    if not database_path.exists():
        return False
    if not database_path.is_file():
        return True
    try:
        connection = sqlite3.connect(database_path)
        try:
            outcome_count = int(
                connection.execute(
                    "SELECT "
                    "(SELECT COUNT(*) FROM completed) + "
                    "(SELECT COUNT(*) FROM failures)"
                ).fetchone()[0]
            )
        finally:
            connection.close()
    except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
        return True
    return outcome_count > 0


def _prepare_official_sdf_snapshot(
    output_dir: Path,
    source_path: os.PathLike[str] | str,
    expected_descriptor: Mapping[str, Any],
) -> Path:
    if (
        not isinstance(expected_descriptor, Mapping)
        or set(expected_descriptor) != {"name", "size", "sha256"}
    ):
        raise ResumeStateError(
            "official SDF run-fingerprint descriptor is invalid"
        )
    snapshot_path, metadata_path, poison_path = (
        _official_sdf_snapshot_paths(output_dir)
    )
    _remove_official_sdf_temporaries(output_dir)
    if poison_path.exists():
        raise ResumeStateError(
            f"official SDF snapshot is poisoned: {poison_path}"
        )
    pair_exists = (snapshot_path.exists(), metadata_path.exists())
    if pair_exists == (True, True):
        try:
            return _verify_official_sdf_snapshot(
                output_dir,
                expected_descriptor,
            )
        except ResumeStateError as exc:
            if _has_geometry_derivatives(output_dir):
                _poison_official_sdf_snapshot(output_dir, str(exc))
                raise
            snapshot_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            _fsync_directory(output_dir)
    elif pair_exists == (True, False):
        try:
            _verify_official_sdf_snapshot_bytes(
                snapshot_path,
                expected_descriptor,
            )
        except ResumeStateError as exc:
            if _has_geometry_derivatives(output_dir):
                _poison_official_sdf_snapshot(output_dir, str(exc))
                raise
            snapshot_path.unlink(missing_ok=True)
            _fsync_directory(output_dir)
        else:
            _atomic_write_text(
                metadata_path,
                json.dumps(
                    {
                        "schema": OFFICIAL_SDF_SNAPSHOT_SCHEMA,
                        "source": dict(expected_descriptor),
                    },
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
            )
            return _verify_official_sdf_snapshot(
                output_dir,
                expected_descriptor,
            )
    elif pair_exists == (False, True):
        if _has_geometry_derivatives(output_dir):
            reason = "official SDF snapshot payload is missing"
            _poison_official_sdf_snapshot(output_dir, reason)
            raise ResumeStateError(reason)
        metadata_path.unlink(missing_ok=True)
        _fsync_directory(output_dir)

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=".official_sdf.snapshot.copy-",
        suffix=".tmp",
        dir=output_dir,
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _copy_verified_input_snapshot(
            source_path,
            temporary,
            expected_descriptor=expected_descriptor,
            label="official SDF",
        )
        os.replace(temporary, snapshot_path)
        _fsync_directory(output_dir)
        _atomic_write_text(
            metadata_path,
            json.dumps(
                {
                    "schema": OFFICIAL_SDF_SNAPSHOT_SCHEMA,
                    "source": dict(expected_descriptor),
                },
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
        return _verify_official_sdf_snapshot(
            output_dir,
            expected_descriptor,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        if snapshot_path.exists() != metadata_path.exists():
            snapshot_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            _fsync_directory(output_dir)
        raise


def _remove_official_sdf_snapshot(output_dir: Path) -> None:
    snapshot_path, metadata_path, poison_path = (
        _official_sdf_snapshot_paths(output_dir)
    )
    if poison_path.exists():
        raise ResumeStateError(
            f"cannot publish a poisoned SDF snapshot: {poison_path}"
        )
    metadata_path.unlink(missing_ok=True)
    _fsync_directory(output_dir)
    snapshot_path.unlink(missing_ok=True)
    _remove_official_sdf_temporaries(output_dir)
    leftovers = [
        path.name
        for path in (snapshot_path, metadata_path, poison_path)
        if path.exists()
    ]
    if leftovers:
        raise ResumeStateError(
            f"official SDF private artifacts remain: {leftovers}"
        )
    _fsync_directory(output_dir)


def _assert_no_geometry_staging(output_dir: Path) -> None:
    candidates = [
        output_dir / ".geometry_index_work.sqlite3",
        output_dir / ".geometry_index_work.sqlite3-journal",
        output_dir / ".geometry_index_work.sqlite3-wal",
        output_dir / ".geometry_index_work.sqlite3-shm",
        output_dir / ".geometry_index_arrays",
    ]
    for pattern in (
        ".geometry_index_validate-*.sqlite3*",
        ".geometry_index*.tmp",
        ".shard_*.tmp",
        ".manifest.json.*.tmp",
        ".failures.jsonl.*.tmp",
        ".run_state.json.*.tmp",
        ".official_sdf.snapshot*",
        "..official_sdf.snapshot*",
    ):
        candidates.extend(output_dir.glob(pattern))
    leftovers = sorted(
        {candidate.name for candidate in candidates if candidate.exists()}
    )
    if leftovers:
        raise ResumeStateError(
            f"geometry staging artifacts remain before publication: {leftovers}"
        )


def prepare_work_database(
    database_path: os.PathLike[str] | str,
    *,
    input_path: os.PathLike[str] | str,
    smiles_col: str,
    source_index_col: Optional[str],
    sdf_ordinal_col: Optional[str],
    manifest_path: Optional[os.PathLike[str] | str],
    fingerprint_sha256: str,
    expected_run_fingerprint: Optional[dict[str, Any]] = None,
    chunk_size: int = DEFAULT_TABLE_CHUNK_SIZE,
) -> sqlite3.Connection:
    """Build or reopen the external-memory deterministic work queue."""

    if (
        not isinstance(fingerprint_sha256, str)
        or len(fingerprint_sha256) != 64
        or set(fingerprint_sha256) - set("0123456789abcdef")
    ):
        raise ResumeStateError("work database fingerprint checksum is invalid")
    expected_inputs: Optional[dict[str, Any]] = None
    if expected_run_fingerprint is not None:
        _validate_run_fingerprint(expected_run_fingerprint)
        if expected_run_fingerprint["sha256"] != fingerprint_sha256:
            raise ResumeStateError(
                "work database checksum differs from the run fingerprint"
            )
        expected_inputs = expected_run_fingerprint["inputs"]
        if (
            set(expected_inputs) != {"input", "manifest", "official_sdf"}
            or expected_inputs.get("input") is None
            or (
                manifest_path is None
                and expected_inputs.get("manifest") is not None
            )
            or (
                manifest_path is not None
                and expected_inputs.get("manifest") is None
            )
        ):
            raise ResumeStateError(
                "work database inputs differ from the run fingerprint"
            )
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove_stale_work_staging(path)
    if path.is_file():
        connection = _open_work_database(path)
        try:
            _validate_work_database(
                connection,
                fingerprint_sha256=fingerprint_sha256,
            )
            _validate_persisted_failures(connection)
        except BaseException:
            connection.close()
            raise
        return connection

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    snapshot_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{path.name}.inputs-",
            dir=path.parent,
        )
    )
    input_source = Path(input_path)
    input_snapshot = snapshot_directory / (
        f"input{input_source.suffix.lower()}"
    )
    manifest_snapshot: Optional[Path] = None
    if manifest_path is not None:
        manifest_source = Path(manifest_path)
        manifest_snapshot = snapshot_directory / (
            f"manifest{manifest_source.suffix.lower()}"
        )
    connection = _open_work_database(temporary)
    try:
        _copy_verified_input_snapshot(
            input_source,
            input_snapshot,
            expected_descriptor=(
                None if expected_inputs is None else expected_inputs["input"]
            ),
            label="geometry input",
        )
        if manifest_path is not None and manifest_snapshot is not None:
            _copy_verified_input_snapshot(
                manifest_path,
                manifest_snapshot,
                expected_descriptor=(
                    None
                    if expected_inputs is None
                    else expected_inputs["manifest"]
                ),
                label="geometry manifest",
            )
        _fsync_directory(snapshot_directory)
        _initialize_work_schema(connection)
        _insert_input_chunks(
            connection,
            input_path=input_snapshot,
            smiles_col=smiles_col,
            source_index_col=source_index_col,
            sdf_ordinal_col=sdf_ordinal_col,
            chunk_size=chunk_size,
        )
        if manifest_path is not None:
            if manifest_snapshot is None:
                raise RuntimeError("manifest snapshot was not created")
            _insert_manifest_chunks(
                connection,
                manifest_path=manifest_snapshot,
                smiles_col=smiles_col,
                sdf_ordinal_col=sdf_ordinal_col,
                chunk_size=chunk_size,
            )
            missing = connection.execute(
                "SELECT s.source_index FROM selected AS s "
                "LEFT JOIN input_rows AS i USING(source_index) "
                "WHERE i.source_index IS NULL ORDER BY s.sequence LIMIT 10"
            ).fetchall()
            if missing:
                raise ValueError(
                    "manifest source_index values are absent from input: "
                    f"{[int(row[0]) for row in missing]}"
                )
            smiles_conflicts = connection.execute(
                "SELECT s.source_index FROM selected AS s "
                "JOIN input_rows AS i USING(source_index) "
                "WHERE s.smiles IS NOT NULL AND s.smiles != i.smiles "
                "ORDER BY s.sequence LIMIT 10"
            ).fetchall()
            if smiles_conflicts:
                raise ValueError(
                    "manifest SMILES conflict with input for source_index "
                    "values: "
                    f"{[int(row[0]) for row in smiles_conflicts]}"
                )
            ordinal_conflicts = connection.execute(
                "SELECT s.source_index FROM selected AS s "
                "JOIN input_rows AS i USING(source_index) "
                "WHERE s.sdf_ordinal IS NOT NULL "
                "AND i.sdf_ordinal IS NOT NULL "
                "AND s.sdf_ordinal != i.sdf_ordinal "
                "ORDER BY s.sequence LIMIT 10"
            ).fetchall()
            if ordinal_conflicts:
                raise ValueError(
                    "manifest SDF ordinal conflict with input for "
                    "source_index values: "
                    f"{[int(row[0]) for row in ordinal_conflicts]}"
                )
            with connection:
                connection.execute(
                    "INSERT INTO work_items "
                    "(sequence,row_index,source_index,smiles,sdf_ordinal) "
                    "SELECT s.sequence,i.row_index,i.source_index,"
                    "COALESCE(s.smiles,i.smiles),"
                    "COALESCE(s.sdf_ordinal,i.sdf_ordinal) "
                    "FROM selected AS s JOIN input_rows AS i USING(source_index) "
                    "ORDER BY s.sequence"
                )
        else:
            with connection:
                connection.execute(
                    "INSERT INTO work_items "
                    "(sequence,row_index,source_index,smiles,sdf_ordinal) "
                    "SELECT sequence,row_index,source_index,smiles,sdf_ordinal "
                    "FROM input_rows ORDER BY sequence"
                )
        work_item_count, work_items_sha256 = _work_queue_identity(connection)
        with connection:
            connection.execute("DROP TABLE input_rows")
            connection.execute("DROP TABLE selected")
            connection.executemany(
                "INSERT INTO metadata(key,value) VALUES (?,?)",
                (
                    ("schema", WORK_DATABASE_SCHEMA),
                    ("fingerprint_sha256", fingerprint_sha256),
                    ("work_item_count", str(work_item_count)),
                    ("work_items_sha256", work_items_sha256),
                    ("status", "complete"),
                ),
            )
        connection.execute("VACUUM")
        connection.close()
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(snapshot_directory, ignore_errors=True)
    _assert_no_work_staging(path)
    connection = _open_work_database(path)
    try:
        _validate_work_database(
            connection,
            fingerprint_sha256=fingerprint_sha256,
        )
        _validate_persisted_failures(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def validate_official_sdf_work_contract(
    connection: sqlite3.Connection,
) -> None:
    """Validate absolute SDF positions; sparse selection ordinals are valid."""
    invalid = connection.execute(
        "SELECT source_index,sdf_ordinal FROM work_items "
        "WHERE sdf_ordinal IS NULL OR sdf_ordinal < 0 "
        "ORDER BY sequence LIMIT 10"
    ).fetchall()
    if invalid:
        examples = [
            {
                "source_index": int(source_index),
                "sdf_ordinal": (
                    None if sdf_ordinal is None else int(sdf_ordinal)
                ),
            }
            for source_index, sdf_ordinal in invalid
        ]
        raise ValueError(
            "official SDF ordinals must be non-negative for every work item; "
            f"invalid examples={examples}"
        )
    duplicates = connection.execute(
        "SELECT sdf_ordinal,COUNT(*) FROM work_items "
        "GROUP BY sdf_ordinal HAVING COUNT(*) > 1 "
        "ORDER BY sdf_ordinal LIMIT 10"
    ).fetchall()
    if duplicates:
        examples = [
            {"sdf_ordinal": int(ordinal), "count": int(count)}
            for ordinal, count in duplicates
        ]
        raise ValueError(
            "official SDF ordinals must be unique within work_items; "
            f"duplicate examples={examples}"
        )


def iter_pending_work_items(
    connection: sqlite3.Connection,
    *,
    order_by: str = "sequence",
) -> Iterator[dict[str, Any]]:
    if order_by not in {"sequence", "sdf_ordinal"}:
        raise ValueError("order_by must be 'sequence' or 'sdf_ordinal'")
    last_sequence = -1
    last_ordinal: Optional[int] = None
    while True:
        if order_by == "sequence":
            rows = connection.execute(
                "SELECT w.sequence,w.row_index,w.smiles,w.source_index,"
                "w.sdf_ordinal FROM work_items AS w "
                "LEFT JOIN completed AS c USING(row_index) "
                "LEFT JOIN failures AS f USING(row_index) "
                "WHERE c.row_index IS NULL AND f.row_index IS NULL "
                "AND w.sequence > ? ORDER BY w.sequence LIMIT 4096",
                (last_sequence,),
            ).fetchall()
        elif last_ordinal is None:
            rows = connection.execute(
                "SELECT w.sequence,w.row_index,w.smiles,w.source_index,"
                "w.sdf_ordinal FROM work_items AS w "
                "LEFT JOIN completed AS c USING(row_index) "
                "LEFT JOIN failures AS f USING(row_index) "
                "WHERE c.row_index IS NULL AND f.row_index IS NULL "
                "AND w.sdf_ordinal IS NOT NULL "
                "ORDER BY w.sdf_ordinal,w.sequence LIMIT 4096"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT w.sequence,w.row_index,w.smiles,w.source_index,"
                "w.sdf_ordinal FROM work_items AS w "
                "LEFT JOIN completed AS c USING(row_index) "
                "LEFT JOIN failures AS f USING(row_index) "
                "WHERE c.row_index IS NULL AND f.row_index IS NULL "
                "AND w.sdf_ordinal IS NOT NULL "
                "AND (w.sdf_ordinal > ? OR "
                "(w.sdf_ordinal = ? AND w.sequence > ?)) "
                "ORDER BY w.sdf_ordinal,w.sequence LIMIT 4096",
                (last_ordinal, last_ordinal, last_sequence),
            ).fetchall()
        if not rows:
            return
        for sequence, row_index, smiles, source_index, sdf_ordinal in rows:
            last_sequence = int(sequence)
            if sdf_ordinal is not None:
                last_ordinal = int(sdf_ordinal)
            yield {
                "row_index": int(row_index),
                "smiles": str(smiles),
                "source_index": int(source_index),
                "sdf_ordinal": (
                    None if sdf_ordinal is None else int(sdf_ordinal)
                ),
            }


class FailureJournal:
    """Durably deduplicated, streaming failure log backed by the work DB."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        path: os.PathLike[str] | str,
    ) -> None:
        self.connection = connection
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            try:
                for (payload,) in connection.execute(
                    "SELECT payload_json FROM failures ORDER BY row_index"
                ):
                    row = json.loads(payload)
                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, self.path)
        _fsync_directory(self.path.parent)
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")

    def record(self, failure: dict[str, Any]) -> bool:
        row_index, payload = _validated_failure_payload(
            self.connection,
            failure,
            context="generated failure",
        )
        with self.connection:
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO failures(row_index,payload_json) "
                "VALUES (?,?)",
                (row_index, payload),
            )
        if cursor.rowcount == 0:
            existing = self.connection.execute(
                "SELECT payload_json FROM failures WHERE row_index=?",
                (row_index,),
            ).fetchone()
            if existing is None or str(existing[0]) != payload:
                raise ArtifactIntegrityError(
                    f"conflicting generated failure for row_index={row_index}"
                )
            return False
        self._handle.write(
            json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return True

    @property
    def count(self) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        )

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "FailureJournal":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def load_work_items(
    input_path: os.PathLike[str] | str,
    smiles_col: str,
    source_index_col: Optional[str],
    sdf_ordinal_col: Optional[str],
    manifest_path: Optional[os.PathLike[str] | str],
) -> list[dict[str, Any]]:
    """Join selection manifests by global source id, never by row position."""
    frame = _read_table(input_path)
    if smiles_col not in frame.columns:
        raise ValueError(f"missing SMILES column {smiles_col!r}")
    frame = frame.copy()
    if source_index_col:
        if source_index_col not in frame.columns:
            raise ValueError(f"missing source-index column {source_index_col!r}")
        frame["source_index"] = frame[source_index_col]
    elif "source_index" not in frame.columns:
        frame["source_index"] = np.arange(len(frame), dtype=np.int64)

    def normalize_identity_column(
        table: pd.DataFrame,
        column: str,
        *,
        context: str,
    ) -> None:
        normalized: list[int] = []
        for row_number, value in enumerate(table[column].tolist()):
            parsed = _optional_integer(
                value,
                field=context,
                row_number=row_number,
            )
            if parsed is None or parsed < 0:
                raise ValueError(
                    f"{context} must contain non-negative integers; "
                    f"invalid row {row_number}"
                )
            normalized.append(parsed)
        table[column] = normalized

    normalize_identity_column(
        frame,
        "source_index",
        context="input.source_index",
    )
    if frame["source_index"].duplicated().any():
        raise ValueError("input source_index values must be unique")
    if "row_index" not in frame.columns:
        frame["row_index"] = frame["source_index"]
    normalize_identity_column(
        frame,
        "row_index",
        context="input.row_index",
    )

    def add_sdf_ordinal(
        table: pd.DataFrame,
        explicit_column: Optional[str],
    ) -> pd.DataFrame:
        table = table.copy()
        candidates = [
            name
            for name in (explicit_column, "sdf_ordinal", "train_ordinal")
            if name and name in table.columns
        ]
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            table["sdf_ordinal"] = None
            return table
        normalized_candidates: list[pd.Series] = []
        for candidate in candidates:
            normalized_values: list[Optional[int]] = []
            for row_number, value in enumerate(table[candidate].tolist()):
                parsed = _optional_integer(
                    value,
                    field=candidate,
                    row_number=row_number,
                )
                if parsed is not None and parsed < -1:
                    raise ValueError(
                        f"{candidate} must be -1 or non-negative at "
                        f"row {row_number}"
                    )
                normalized_values.append(
                    None if parsed in (None, -1) else parsed
                )
            normalized_candidates.append(
                pd.Series(
                    normalized_values,
                    index=table.index,
                    dtype=object,
                )
            )
        reference = normalized_candidates[0]
        for candidate_values in normalized_candidates[1:]:
            mismatch = (
                reference.notna()
                & candidate_values.notna()
                & (reference != candidate_values)
            )
            if mismatch.any():
                raise ValueError(
                    f"conflicting SDF ordinal columns: {candidates}"
                )
            reference = reference.combine_first(candidate_values)
        table["sdf_ordinal"] = reference
        return table

    frame = add_sdf_ordinal(frame, sdf_ordinal_col)
    if manifest_path:
        manifest = _read_table(manifest_path)
        if "source_index" not in manifest.columns:
            raise ValueError("manifest must contain source_index")
        manifest = add_sdf_ordinal(manifest, sdf_ordinal_col)
        manifest = manifest.rename(
            columns={"sdf_ordinal": "_selected_sdf_ordinal"}
        )
        normalize_identity_column(
            manifest,
            "source_index",
            context="manifest.source_index",
        )
        if manifest["source_index"].duplicated().any():
            raise ValueError("manifest source_index values must be unique")
        frame = manifest.merge(
            frame,
            on="source_index",
            how="left",
            validate="one_to_one",
            suffixes=("_manifest", ""),
        )
        if frame[smiles_col].isna().any():
            missing = frame.loc[frame[smiles_col].isna(), "source_index"].tolist()
            raise ValueError(
                f"manifest source_index values are absent from input: {missing[:10]}"
            )
        manifest_smiles = f"{smiles_col}_manifest"
        if manifest_smiles in frame.columns:
            manifest_values = frame[manifest_smiles]
            manifest_provided = (
                manifest_values.notna()
                & manifest_values.astype(str).str.strip().ne("")
            )
            smiles_conflict = (
                manifest_provided
                & frame[smiles_col].notna()
                & manifest_values.ne(frame[smiles_col])
            )
            if smiles_conflict.any():
                conflicts = frame.loc[
                    smiles_conflict,
                    "source_index",
                ].astype(int)
                raise ValueError(
                    "manifest SMILES conflict with input for source_index "
                    f"values: {conflicts.tolist()[:10]}"
                )
            frame.loc[manifest_provided, smiles_col] = manifest_values.loc[
                manifest_provided
            ]
        selected_ordinals = frame.pop("_selected_sdf_ordinal")
        ordinal_conflict = (
            selected_ordinals.notna()
            & frame["sdf_ordinal"].notna()
            & selected_ordinals.ne(frame["sdf_ordinal"])
        )
        if ordinal_conflict.any():
            conflicts = frame.loc[
                ordinal_conflict,
                "source_index",
            ].astype(int)
            raise ValueError(
                "manifest SDF ordinal conflict with input for source_index "
                f"values: {conflicts.tolist()[:10]}"
            )
        frame["sdf_ordinal"] = selected_ordinals.combine_first(
            frame["sdf_ordinal"]
        )

    invalid_smiles = [
        int(row_index)
        for row_index, smiles in frame[
            ["row_index", smiles_col]
        ].itertuples(index=False, name=None)
        if not isinstance(smiles, str) or not smiles.strip()
    ]
    if invalid_smiles:
        raise ValueError(
            f"SMILES is invalid for row indices {invalid_smiles[:10]}"
        )

    items: list[dict[str, Any]] = []
    for row_number, (
        row_index,
        smiles,
        source_index,
        sdf_ordinal,
    ) in enumerate(
        frame[
            ["row_index", smiles_col, "source_index", "sdf_ordinal"]
        ].itertuples(index=False, name=None)
    ):
        normalized_ordinal = _optional_integer(
            sdf_ordinal,
            field="sdf_ordinal",
            row_number=row_number,
        )
        if normalized_ordinal is not None and normalized_ordinal < -1:
            raise ValueError(
                f"sdf_ordinal must be -1 or non-negative at row {row_number}"
            )
        items.append(
            {
                "row_index": _strict_identity_integer(
                    row_index,
                    field="row_index",
                ),
                "smiles": smiles,
                "source_index": _strict_identity_integer(
                    source_index,
                    field="source_index",
                ),
                "sdf_ordinal": (
                    None
                    if normalized_ordinal in (None, -1)
                    else normalized_ordinal
                ),
            }
        )
    duplicate_rows = len(items) != len({item["row_index"] for item in items})
    if duplicate_rows:
        raise ValueError("row_index values must be unique")
    return items


def _deterministic_seed(smiles: str, base_seed: int) -> int:
    molecule_bits = int(hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:8], 16)
    return int((molecule_bits ^ int(base_seed)) & 0x7FFFFFFF)


def _generate_task(
    task: tuple[dict[str, Any], int, float, int, bool],
) -> tuple[dict[str, Any], Optional[GeometryRecord], Optional[dict[str, Any]]]:
    item, num_conformers, prune_rms_thresh, base_seed, optimize = task
    try:
        record = generate_conformers(
            item["smiles"],
            num_conformers=num_conformers,
            prune_rms_thresh=prune_rms_thresh,
            seed=_deterministic_seed(item["smiles"], base_seed),
            optimize=optimize,
            on_invalid="raise",
        )
        return item, record, None
    except (RuntimeError, ValueError) as exc:
        return item, None, {
            "row_index": item["row_index"],
            "source_index": item["source_index"],
            "sdf_ordinal": item.get("sdf_ordinal"),
            "smiles": item["smiles"],
            "stage": "etkdg_fallback",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _next_shard_id(output_dir: Path) -> int:
    ids = [
        int(path.stem.split("_")[-1])
        for path in output_dir.glob("shard_*.json")
        if path.stem.split("_")[-1].isdigit()
    ]
    return max(ids, default=-1) + 1


def _sync_completed_rows(
    connection: sqlite3.Connection,
    output_dir: Path,
    *,
    verify_checksums: bool,
) -> int:
    with connection:
        connection.execute("DELETE FROM completed")
    count = 0
    for metadata in _iter_verified_geometry_sidecars(
        output_dir,
        verify_checksums=verify_checksums,
    ):
        entries = metadata["records"]
        row_indices = [int(entry["row_index"]) for entry in entries]
        work_identities: dict[int, tuple[int, str, Optional[int]]] = {}
        for start in range(0, len(row_indices), 900):
            chunk = row_indices[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            for row in connection.execute(
                "SELECT row_index,source_index,smiles,sdf_ordinal "
                f"FROM work_items WHERE row_index IN ({placeholders})",
                chunk,
            ):
                work_identities[int(row[0])] = (
                    int(row[1]),
                    str(row[2]),
                    None if row[3] is None else int(row[3]),
                )
        for entry in entries:
            row_index = int(entry["row_index"])
            expected = work_identities.get(row_index)
            observed = (
                int(entry["source_index"]),
                str(entry["smiles"]),
                (
                    None
                    if entry["sdf_ordinal"] is None
                    else int(entry["sdf_ordinal"])
                ),
            )
            if expected is None or observed != expected:
                raise ArtifactIntegrityError(
                    "completed geometry identity differs from the current "
                    f"work item: row_index={row_index}"
                )
        rows = [(row_index,) for row_index in row_indices]
        try:
            with connection:
                connection.executemany(
                    "INSERT INTO completed(row_index) VALUES (?)",
                    rows,
                )
        except sqlite3.IntegrityError as exc:
            raise ArtifactIntegrityError(
                "completed geometry row_index is duplicated or absent "
                "from the current work database"
            ) from exc
        count += len(rows)
    return count


def _write_run_metadata(
    output_dir: Path,
    failure_count: int,
    arguments: argparse.Namespace,
    *,
    connection: sqlite3.Connection,
    requested_count: int,
    official_sdf_snapshot: Optional[Path],
    run_fingerprint: dict[str, Any],
    fingerprint_inputs: dict[
        str,
        Optional[os.PathLike[str] | str],
    ],
    fingerprint_parameters: dict[str, Any],
) -> None:
    official_sdf_descriptor = run_fingerprint["inputs"]["official_sdf"]
    if (official_sdf_snapshot is None) != (official_sdf_descriptor is None):
        raise ResumeStateError(
            "official SDF snapshot presence differs from the run fingerprint"
        )
    if official_sdf_snapshot is not None:
        try:
            verified_snapshot = _verify_official_sdf_snapshot(
                output_dir,
                official_sdf_descriptor,
            )
        except ResumeStateError as exc:
            _poison_official_sdf_snapshot(output_dir, str(exc))
            raise
        if verified_snapshot != official_sdf_snapshot:
            raise ResumeStateError("official SDF snapshot path changed")
    index_metadata = write_geometry_index(output_dir)
    verify_run_fingerprint(
        fingerprint_inputs,
        fingerprint_parameters,
        run_fingerprint,
    )
    (
        final_shard_summaries,
        final_successful_records,
    ) = _validate_geometry_index_pair(
        output_dir,
        verify_checksums=True,
    )
    _validate_work_database(
        connection,
        fingerprint_sha256=str(run_fingerprint["sha256"]),
    )
    _validate_persisted_failures(connection)
    completed_count = int(
        connection.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
    )
    persisted_failure_count = int(
        connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
    )
    overlap_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM completed AS c "
            "JOIN failures AS f USING(row_index)"
        ).fetchone()[0]
    )
    missing = connection.execute(
        "SELECT w.row_index FROM work_items AS w "
        "LEFT JOIN completed AS c USING(row_index) "
        "LEFT JOIN failures AS f USING(row_index) "
        "WHERE c.row_index IS NULL AND f.row_index IS NULL "
        "ORDER BY w.sequence LIMIT 10"
    ).fetchall()
    queue_count, queue_sha256 = _work_queue_identity(connection)
    database_metadata = dict(
        connection.execute("SELECT key,value FROM metadata")
    )
    if (
        requested_count != queue_count
        or database_metadata.get("work_item_count") != str(queue_count)
        or database_metadata.get("work_items_sha256") != queue_sha256
        or overlap_count != 0
        or missing
        or completed_count + persisted_failure_count != requested_count
        or persisted_failure_count != int(failure_count)
        or completed_count != final_successful_records
    ):
        raise ArtifactIntegrityError(
            "geometry success/failure outcomes are not a disjoint, complete "
            "cover of the immutable work queue"
        )
    _assert_no_work_staging(output_dir / "geometry_work.sqlite3")
    verify_run_fingerprint(
        fingerprint_inputs,
        fingerprint_parameters,
        run_fingerprint,
    )
    ensure_run_state(output_dir, run_fingerprint, resume=True)
    manifest = {
        "schema": GEOMETRY_SCHEMA,
        "shards": final_shard_summaries,
        "successful_records": final_successful_records,
        "failed_records": int(failure_count),
        "run_fingerprint": run_fingerprint,
        "source_index": {
            "artifact": index_metadata["filename"],
            "metadata": "geometry_index.json",
            "sha256": index_metadata["sha256"],
            "record_count": index_metadata["record_count"],
        },
        "parameters": {
            "num_conformers": arguments.num_conformers,
            "prune_rms_thresh": arguments.prune_rms_thresh,
            "seed": arguments.seed,
            "optimize": not arguments.no_optimize,
            "official_sdf": arguments.official_sdf,
        },
    }
    if official_sdf_snapshot is not None:
        try:
            verified_snapshot = _verify_official_sdf_snapshot(
                output_dir,
                official_sdf_descriptor,
            )
        except ResumeStateError as exc:
            _poison_official_sdf_snapshot(output_dir, str(exc))
            raise
        if verified_snapshot != official_sdf_snapshot:
            raise ResumeStateError("official SDF snapshot path changed")
        _remove_official_sdf_snapshot(output_dir)
    _assert_no_geometry_staging(output_dir)
    _atomic_write_text(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )


def _consume_results(
    results: Iterable[
        tuple[dict[str, Any], Optional[GeometryRecord], Optional[dict[str, Any]]]
    ],
    output_dir: Path,
    shard_size: int,
    first_shard_id: int,
    connection: sqlite3.Connection,
    failure_journal: FailureJournal,
) -> tuple[int, int, int]:
    pending: list[
        tuple[int, str, int, Optional[int], GeometryRecord]
    ] = []
    shard_id = first_shard_id
    successful = 0
    new_failures = 0
    for item, record, failure in results:
        if failure is not None:
            if failure_journal.record(failure):
                new_failures += 1
            continue
        if record is None:
            raise RuntimeError("worker returned neither a geometry nor an error")
        pending.append(
            (
                item["row_index"],
                item["smiles"],
                item["source_index"],
                item.get("sdf_ordinal"),
                record,
            )
        )
        if len(pending) >= shard_size:
            write_geometry_shard(pending, output_dir, shard_id)
            with connection:
                connection.executemany(
                    "INSERT INTO completed(row_index) VALUES (?)",
                    [(int(item[0]),) for item in pending],
                )
            successful += len(pending)
            shard_id += 1
            pending = []
    if pending:
        write_geometry_shard(pending, output_dir, shard_id)
        with connection:
            connection.executemany(
                "INSERT INTO completed(row_index) VALUES (?)",
                [(int(item[0]),) for item in pending],
            )
        successful += len(pending)
        shard_id += 1
    return shard_id, successful, new_failures


def _sdf_results(
    items: list[dict[str, Any]],
    sdf_path: os.PathLike[str] | str,
    num_conformers: int,
    prune_rms_thresh: float,
    seed: int,
    energy_property: Optional[str],
    optimize: bool,
) -> Iterator[
    tuple[dict[str, Any], Optional[GeometryRecord], Optional[dict[str, Any]]]
]:
    by_ordinal: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        if item.get("sdf_ordinal") is None:
            raise ValueError(
                "official SDF extraction requires an explicit "
                "train_ordinal/sdf_ordinal column "
                f"for source_index={item['source_index']}"
            )
        ordinal = int(item["sdf_ordinal"])
        if ordinal < 0:
            raise ValueError(
                "official SDF ordinals must be non-negative for every work "
                f"item; source_index={item['source_index']}, "
                f"sdf_ordinal={ordinal}"
            )
        if ordinal in by_ordinal:
            raise ValueError(
                "official SDF ordinals must be unique within work items; "
                f"sdf_ordinal={ordinal}"
            )
        by_ordinal[ordinal] = [item]
    seen: set[int] = set()
    max_ordinal = max(by_ordinal, default=-1)
    for sdf_ordinal, sdf_mol in iter_sdf_molecules(sdf_path):
        if sdf_ordinal > max_ordinal:
            break
        selected = by_ordinal.get(sdf_ordinal)
        if not selected:
            continue
        seen.add(sdf_ordinal)
        for item in selected:
            try:
                record = geometry_from_sdf_or_fallback(
                    sdf_mol,
                    item["smiles"],
                    num_conformers=num_conformers,
                    seed=_deterministic_seed(item["smiles"], seed),
                    energy_property=energy_property,
                    prune_rms_thresh=prune_rms_thresh,
                    optimize=optimize,
                    sdf_unavailable_reason=(
                        "sdf_record_unreadable"
                        if sdf_mol is None
                        else "sdf_record_missing"
                    ),
                )
                yield item, record, None
            except (RuntimeError, ValueError) as exc:
                yield item, None, {
                    "row_index": item["row_index"],
                    "source_index": item["source_index"],
                    "sdf_ordinal": sdf_ordinal,
                    "smiles": item["smiles"],
                    "stage": "sdf_and_etkdg_fallback",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
    for sdf_ordinal in sorted(set(by_ordinal) - seen):
        for item in by_ordinal[sdf_ordinal]:
            try:
                record = geometry_from_sdf_or_fallback(
                    None,
                    item["smiles"],
                    num_conformers=num_conformers,
                    seed=_deterministic_seed(item["smiles"], seed),
                    energy_property=energy_property,
                    prune_rms_thresh=prune_rms_thresh,
                    optimize=optimize,
                )
                yield item, record, None
            except (RuntimeError, ValueError) as exc:
                yield item, None, {
                    "row_index": item["row_index"],
                    "source_index": item["source_index"],
                    "sdf_ordinal": sdf_ordinal,
                    "smiles": item["smiles"],
                    "stage": "missing_sdf_etkdg_fallback",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }


def _sdf_result_for_item(
    item: dict[str, Any],
    sdf_mol: Any,
    *,
    sdf_unavailable_reason: str,
    failure_stage: str,
    num_conformers: int,
    prune_rms_thresh: float,
    seed: int,
    energy_property: Optional[str],
    optimize: bool,
) -> tuple[dict[str, Any], Optional[GeometryRecord], Optional[dict[str, Any]]]:
    try:
        record = geometry_from_sdf_or_fallback(
            sdf_mol,
            item["smiles"],
            num_conformers=num_conformers,
            seed=_deterministic_seed(item["smiles"], seed),
            energy_property=energy_property,
            prune_rms_thresh=prune_rms_thresh,
            optimize=optimize,
            sdf_unavailable_reason=sdf_unavailable_reason,
        )
        return item, record, None
    except (RuntimeError, ValueError) as exc:
        return item, None, {
            "row_index": item["row_index"],
            "source_index": item["source_index"],
            "sdf_ordinal": item["sdf_ordinal"],
            "smiles": item["smiles"],
            "stage": failure_stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _sdf_results_from_database(
    connection: sqlite3.Connection,
    sdf_path: os.PathLike[str] | str,
    num_conformers: int,
    prune_rms_thresh: float,
    seed: int,
    energy_property: Optional[str],
    optimize: bool,
) -> Iterator[
    tuple[dict[str, Any], Optional[GeometryRecord], Optional[dict[str, Any]]]
]:
    """Validate then merge absolute work-item ordinals with streamed SDF."""
    validate_official_sdf_work_contract(connection)
    return _iter_sdf_results_from_database(
        connection,
        sdf_path,
        num_conformers,
        prune_rms_thresh,
        seed,
        energy_property,
        optimize,
    )


def _iter_sdf_results_from_database(
    connection: sqlite3.Connection,
    sdf_path: os.PathLike[str] | str,
    num_conformers: int,
    prune_rms_thresh: float,
    seed: int,
    energy_property: Optional[str],
    optimize: bool,
) -> Iterator[
    tuple[dict[str, Any], Optional[GeometryRecord], Optional[dict[str, Any]]]
]:
    """External-memory merge of sparse sorted ordinals with streamed SDF."""
    train_items = (
        item
        for item in iter_pending_work_items(
            connection,
            order_by="sdf_ordinal",
        )
    )
    current = next(train_items, None)
    for sdf_ordinal, sdf_mol in iter_sdf_molecules(sdf_path):
        while (
            current is not None
            and int(current["sdf_ordinal"]) < sdf_ordinal
        ):
            yield _sdf_result_for_item(
                current,
                None,
                sdf_unavailable_reason="sdf_record_missing",
                failure_stage="missing_sdf_etkdg_fallback",
                num_conformers=num_conformers,
                prune_rms_thresh=prune_rms_thresh,
                seed=seed,
                energy_property=energy_property,
                optimize=optimize,
            )
            current = next(train_items, None)
        while (
            current is not None
            and int(current["sdf_ordinal"]) == sdf_ordinal
        ):
            yield _sdf_result_for_item(
                current,
                sdf_mol,
                sdf_unavailable_reason=(
                    "sdf_record_unreadable"
                    if sdf_mol is None
                    else "sdf_record_missing"
                ),
                failure_stage="sdf_and_etkdg_fallback",
                num_conformers=num_conformers,
                prune_rms_thresh=prune_rms_thresh,
                seed=seed,
                energy_property=energy_property,
                optimize=optimize,
            )
            current = next(train_items, None)
        if current is None:
            break
    while current is not None:
        yield _sdf_result_for_item(
            current,
            None,
            sdf_unavailable_reason="sdf_record_missing",
            failure_stage="missing_sdf_etkdg_fallback",
            num_conformers=num_conformers,
            prune_rms_thresh=prune_rms_thresh,
            seed=seed,
            energy_property=energy_property,
            optimize=optimize,
        )
        current = next(train_items, None)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full-atom 3D geometry shards from official PCQM SDF "
            "records with deterministic ETKDGv2 fallbacks."
        )
    )
    parser.add_argument(
        "--input",
        "--raw-csv",
        "--raw_csv",
        dest="input_path",
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--source-index-col")
    parser.add_argument(
        "--sdf-ordinal-col",
        "--train-ordinal-col",
        dest="sdf_ordinal_col",
    )
    parser.add_argument(
        "--manifest",
        help=(
            "Optional CSV/Parquet/JSONL joined by source_index; official PCQM "
            "SDF use also requires train_ordinal or sdf_ordinal."
        ),
    )
    parser.add_argument("--official-sdf")
    parser.add_argument("--sdf-energy-property")
    parser.add_argument("--num-conformers", type=int, default=3)
    parser.add_argument(
        "--prune-rms-threshold",
        "--prune-rms-thresh",
        dest="prune_rms_thresh",
        type=float,
        default=0.5,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument(
        "--table-chunk-size",
        type=int,
        default=DEFAULT_TABLE_CHUNK_SIZE,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-verify-checksums", action="store_true")
    args = parser.parse_args(argv)
    if args.num_conformers <= 0 or args.num_workers <= 0:
        parser.error("--num-conformers and --num-workers must be positive")
    if (
        args.shard_size <= 0
        or args.worker_chunksize <= 0
        or args.table_chunk_size <= 0
    ):
        parser.error(
            "--shard-size, --worker-chunksize and --table-chunk-size "
            "must be positive"
        )
    if (
        not np.isfinite(args.prune_rms_thresh)
        or args.prune_rms_thresh < 0.0
    ):
        parser.error("--prune-rms-threshold must be finite and non-negative")
    if not args.smiles_col:
        parser.error("--smiles-col must be non-empty")
    for option, value in (
        ("--source-index-col", args.source_index_col),
        ("--sdf-ordinal-col", args.sdf_ordinal_col),
        ("--sdf-energy-property", args.sdf_energy_property),
    ):
        if value is not None and not value:
            parser.error(f"{option} must be non-empty when provided")
    if args.official_sdf and args.num_workers != 1:
        parser.error(
            "official SDF streaming currently requires --num-workers 1; "
            "multiple workers are never silently ignored"
        )
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    has_existing_run_artifacts = bool(
        next(output_dir.glob("shard_*.npz"), None)
        or next(output_dir.glob("shard_*.json"), None)
        or (output_dir / "geometry_index.npz").exists()
        or (output_dir / "geometry_index.json").exists()
        or (output_dir / "geometry_work.sqlite3").exists()
        or (output_dir / "failures.jsonl").exists()
        or (output_dir / "manifest.json").exists()
        or next(output_dir.glob(".official_sdf.snapshot*"), None)
        or next(output_dir.glob("..official_sdf.snapshot*"), None)
    )
    if has_existing_run_artifacts and not args.resume:
        raise RuntimeError(
            f"{output_dir} already contains generation artifacts; use --resume "
            "or choose an empty output directory"
        )
    fingerprint_inputs = {
        "input": args.input_path,
        "manifest": args.manifest,
        "official_sdf": args.official_sdf,
    }
    fingerprint_parameters = {
        "schema": GEOMETRY_SCHEMA,
        "smiles_col": args.smiles_col,
        "source_index_col": args.source_index_col,
        "sdf_ordinal_col": args.sdf_ordinal_col,
        "sdf_energy_property": args.sdf_energy_property,
        "num_conformers": args.num_conformers,
        "prune_rms_thresh": args.prune_rms_thresh,
        "seed": args.seed,
        "optimize": not args.no_optimize,
        "num_workers": args.num_workers,
        "worker_chunksize": args.worker_chunksize,
        "shard_size": args.shard_size,
        "table_chunk_size": args.table_chunk_size,
        "verify_checksums": not args.no_verify_checksums,
    }
    fingerprint = compute_run_fingerprint(
        fingerprint_inputs,
        fingerprint_parameters,
    )
    ensure_run_state(output_dir, fingerprint, resume=args.resume)
    official_sdf_snapshot: Optional[Path] = None
    if args.official_sdf:
        official_sdf_descriptor = fingerprint["inputs"]["official_sdf"]
        if not isinstance(official_sdf_descriptor, Mapping):
            raise ResumeStateError(
                "official SDF fingerprint descriptor is missing"
            )
        official_sdf_snapshot = _prepare_official_sdf_snapshot(
            output_dir,
            args.official_sdf,
            official_sdf_descriptor,
        )
    connection = prepare_work_database(
        output_dir / "geometry_work.sqlite3",
        input_path=args.input_path,
        smiles_col=args.smiles_col,
        source_index_col=args.source_index_col,
        sdf_ordinal_col=args.sdf_ordinal_col,
        manifest_path=args.manifest,
        fingerprint_sha256=str(fingerprint["sha256"]),
        expected_run_fingerprint=fingerprint,
        chunk_size=args.table_chunk_size,
    )
    try:
        invalidated_publications = False
        for publication_path in (
            output_dir / "manifest.json",
            output_dir / "geometry_index.npz",
            output_dir / "geometry_index.json",
        ):
            if publication_path.exists():
                publication_path.unlink()
                invalidated_publications = True
        if invalidated_publications:
            _fsync_directory(output_dir)
        reconcile_geometry_artifacts(
            output_dir,
            verify_checksums=not args.no_verify_checksums,
        )
        completed_before = _sync_completed_rows(
            connection,
            output_dir,
            verify_checksums=not args.no_verify_checksums,
        )
        requested = int(
            connection.execute("SELECT COUNT(*) FROM work_items").fetchone()[0]
        )
        first_shard_id = _next_shard_id(output_dir)
        with FailureJournal(
            connection,
            output_dir / "failures.jsonl",
        ) as failure_journal:
            if args.official_sdf:
                if official_sdf_snapshot is None:
                    raise ResumeStateError(
                        "official SDF snapshot was not prepared"
                    )
                results = _sdf_results_from_database(
                    connection,
                    official_sdf_snapshot,
                    num_conformers=args.num_conformers,
                    prune_rms_thresh=args.prune_rms_thresh,
                    seed=args.seed,
                    energy_property=args.sdf_energy_property,
                    optimize=not args.no_optimize,
                )
                _, generated_successes, _ = _consume_results(
                    results,
                    output_dir,
                    shard_size=args.shard_size,
                    first_shard_id=first_shard_id,
                    connection=connection,
                    failure_journal=failure_journal,
                )
            else:
                items = iter_pending_work_items(connection)
                tasks = (
                    (
                        item,
                        args.num_conformers,
                        args.prune_rms_thresh,
                        args.seed,
                        not args.no_optimize,
                    )
                    for item in items
                )
                if args.num_workers == 1:
                    generated: Iterable[
                        tuple[
                            dict[str, Any],
                            Optional[GeometryRecord],
                            Optional[dict[str, Any]],
                        ]
                    ] = map(_generate_task, tasks)
                    _, generated_successes, _ = _consume_results(
                        generated,
                        output_dir,
                        shard_size=args.shard_size,
                        first_shard_id=first_shard_id,
                        connection=connection,
                        failure_journal=failure_journal,
                    )
                else:
                    with Pool(processes=args.num_workers) as pool:
                        generated = pool.imap(
                            _generate_task,
                            tasks,
                            chunksize=args.worker_chunksize,
                        )
                        _, generated_successes, _ = _consume_results(
                            generated,
                            output_dir,
                            shard_size=args.shard_size,
                            first_shard_id=first_shard_id,
                            connection=connection,
                            failure_journal=failure_journal,
                        )
            failure_count = failure_journal.count
        verify_run_fingerprint(
            fingerprint_inputs,
            fingerprint_parameters,
            fingerprint,
        )
        ensure_run_state(output_dir, fingerprint, resume=True)
        _write_run_metadata(
            output_dir,
            failure_count,
            args,
            connection=connection,
            requested_count=requested,
            official_sdf_snapshot=official_sdf_snapshot,
            run_fingerprint=fingerprint,
            fingerprint_inputs=fingerprint_inputs,
            fingerprint_parameters=fingerprint_parameters,
        )
    finally:
        connection.close()
    print(
        json.dumps(
            {
                "requested": requested,
                "skipped_by_resume": completed_before,
                "generated": generated_successes,
                "failed": failure_count,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

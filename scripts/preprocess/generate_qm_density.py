"""Generate strict promolecular-density shards from geometry artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import uuid
from multiprocessing import Pool
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.preprocess.generate_3d_conformer import (  # noqa: E402
    ArtifactIntegrityError,
    GEOMETRY_INDEX_SCHEMA,
    GEOMETRY_SCHEMA,
    RUN_STATE_SCHEMA as GEOMETRY_RUN_STATE_SCHEMA,
    _atomic_savez,
    _atomic_write_text,
    _fsync_directory,
    _iter_verified_geometry_sidecars,
    _sha256_file,
    compute_run_fingerprint,
    ensure_run_state,
)
from src.molecular.electron_density import (  # noqa: E402
    DensityConfigError,
    DensityGridResult,
    build_promolecular_density,
    validate_density_config,
)
from src.molecular.geometry import GeometryRecord  # noqa: E402
from src.molecular.rdkit_utils import smiles_hash  # noqa: E402

DENSITY_SCHEMA = "semmol.promolecular_density.v2"
DENSITY_RUN_STATE_SCHEMA = "semmol.density_run_state.v1"
EMBEDDED_DENSITY_METADATA_KEY = "__metadata_json__"
GEOMETRY_SNAPSHOT_SCHEMA = "semmol.geometry_snapshot.v1"
GEOMETRY_SNAPSHOT_DIRNAME = ".geometry_snapshot"
GEOMETRY_SNAPSHOT_BUILDING_DIRNAME = ".geometry_snapshot.building"
GEOMETRY_SNAPSHOT_DELETE_PREFIX = ".geometry_snapshot.delete-"
GEOMETRY_SNAPSHOT_STATE_FILENAME = "snapshot.json"
GEOMETRY_SNAPSHOT_POISON_FILENAME = ".geometry_snapshot.poison.json"
STATISTICS_READY_FILENAME = ".statistics.ready.json"
STATISTICS_COMMIT_FILENAME = ".statistics.commit.json"
STATISTICS_COMMIT_SCHEMA = "semmol.density_statistics_commit.v1"
WRITER_LOCK_FILENAME = ".density_writer.lock"
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
DENSITY_FINGERPRINT_INPUT_FIELDS = frozenset(
    {
        "geometry_manifest",
        "geometry_index",
        "geometry_run_state",
        "density_config",
    }
)
DENSITY_REQUIRED_FIELDS = frozenset(
    {
        "grid",
        "origin",
        "spacing",
        "electron_count",
        "integrated_electrons",
        "prequantization_integrated_electrons",
        "overflow",
        "overflow_axes",
        "atomic_sigmas",
        "method",
        "box_padding",
        "conformers_used",
        "conformer_reduction",
        "conformer_alignment",
        "normalization_requested",
        "normalization_applied",
    }
)
DENSITY_SIDECAR_FIELDS = frozenset(
    {
        "schema",
        "shard_id",
        "filename",
        "record_count",
        "records",
        "sha256",
    }
)
DENSITY_RECORD_METADATA_FIELDS = frozenset(
    {
        "key",
        "row_index",
        "source_index",
        "sdf_ordinal",
        "train_ordinal",
        "smiles",
        "smiles_hash",
        "geometry_artifact",
        "geometry_artifact_sha256",
        "geometry_key",
        "geometry_payload_sha256",
        "grid_shape",
        "electron_count",
        "integrated_electrons",
        "prequantization_integrated_electrons",
        "integration_error",
        "prequantization_integration_error",
        "overflow",
        "conformers_used",
        "normalization_requested",
        "normalization_applied",
    }
)
FAILURE_RECORD_FIELDS = frozenset(
    {
        "row_index",
        "source_index",
        "sdf_ordinal",
        "train_ordinal",
        "smiles",
        "smiles_hash",
        "geometry_artifact",
        "geometry_artifact_sha256",
        "geometry_key",
        "geometry_payload_sha256",
        "stage",
        "error_type",
        "message",
    }
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _identity_integer(
    value: Any,
    *,
    field: str,
    allow_none: bool = False,
) -> Optional[int]:
    if value is None and allow_none:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{field} must be an integer")
    normalized = int(value)
    if normalized < -(2**63) or normalized > 2**63 - 1:
        raise ValueError(f"{field} exceeds int64")
    return normalized


def _normalize_ordinal(value: Any, *, field: str) -> Optional[int]:
    normalized = _identity_integer(
        value,
        field=field,
        allow_none=True,
    )
    if normalized == -1:
        return None
    if normalized is not None and normalized < -1:
        raise ValueError(f"{field} must be -1, null, or non-negative")
    return normalized


def _geometry_provenance(
    artifact: Any,
    artifact_sha256: Any,
    key: Any,
    payload_sha256: Any,
    *,
    context: str,
) -> tuple[str, str, str, str]:
    if (
        not isinstance(artifact, str)
        or len(artifact) != len("shard_000000.npz")
        or not artifact.startswith("shard_")
        or not artifact.endswith(".npz")
        or not artifact[6:12].isdigit()
        or Path(artifact).name != artifact
    ):
        raise ArtifactIntegrityError(
            f"invalid geometry artifact provenance in {context}"
        )
    if not _is_sha256(artifact_sha256):
        raise ArtifactIntegrityError(
            f"invalid geometry artifact checksum provenance in {context}"
        )
    if (
        not isinstance(key, str)
        or len(key) != 7
        or not key.startswith("r")
        or not key[1:].isdigit()
    ):
        raise ArtifactIntegrityError(
            f"invalid geometry record key provenance in {context}"
        )
    if not _is_sha256(payload_sha256):
        raise ArtifactIntegrityError(
            f"invalid geometry payload provenance in {context}"
        )
    return artifact, artifact_sha256, key, payload_sha256


def _geometry_payload_sha256(record: GeometryRecord) -> str:
    if not isinstance(record, GeometryRecord):
        raise TypeError("record must be a GeometryRecord")
    digest = hashlib.sha256()
    storage = record.to_storage_dict()
    for field in sorted(storage):
        array = np.ascontiguousarray(np.asarray(storage[field]))
        if array.dtype.hasobject:
            raise ArtifactIntegrityError(
                f"geometry payload field {field!r} has object dtype"
            )
        header = json.dumps(
            {
                "field": field,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": int(array.nbytes),
            },
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_density_shard(
    records: list[
        tuple[
            int,
            str,
            int,
            Optional[int],
            str,
            str,
            str,
            str,
            DensityGridResult,
        ]
    ],
    output_dir: os.PathLike[str] | str,
    shard_id: int,
) -> dict[str, Any]:
    """Atomically write density grids and complete physical metadata."""
    if not records:
        raise ValueError("cannot write an empty density shard")
    normalized_shard_id = _identity_integer(shard_id, field="shard_id")
    if normalized_shard_id is None or normalized_shard_id < 0:
        raise ValueError("shard_id must be non-negative")
    directory = Path(output_dir)
    filename = f"density_{normalized_shard_id:06d}.npz"
    shard_path = directory / filename
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    for record_index, item in enumerate(records):
        if len(item) != 9:
            raise ValueError(
                "density shard records must contain identity, geometry "
                "provenance, and density result"
            )
        (
            row_index,
            smiles,
            source_index,
            sdf_ordinal,
            geometry_artifact,
            geometry_artifact_sha256,
            geometry_key,
            geometry_payload_sha256,
            result,
        ) = item
        normalized_row = _identity_integer(row_index, field="row_index")
        normalized_source = _identity_integer(
            source_index,
            field="source_index",
        )
        normalized_ordinal = _normalize_ordinal(
            sdf_ordinal,
            field="sdf_ordinal",
        )
        if normalized_row is None or normalized_row < 0:
            raise ValueError("row_index must be non-negative")
        if normalized_source is None or normalized_source < 0:
            raise ValueError("source_index must be non-negative")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("smiles must be a non-empty string")
        if not isinstance(result, DensityGridResult):
            raise TypeError("density result must be DensityGridResult")
        (
            normalized_geometry_artifact,
            normalized_geometry_artifact_sha256,
            normalized_geometry_key,
            normalized_geometry_payload_sha256,
        ) = _geometry_provenance(
            geometry_artifact,
            geometry_artifact_sha256,
            geometry_key,
            geometry_payload_sha256,
            context=f"density record {record_index}",
        )
        key = f"r{record_index:06d}"
        stored_result = result.to_storage_dict()
        for field, value in stored_result.items():
            arrays[f"{key}__{field}"] = value
        stored_integral = float(stored_result["integrated_electrons"])
        prequantization_integral = float(
            stored_result["prequantization_integrated_electrons"]
        )
        entries.append(
            {
                "key": key,
                "row_index": normalized_row,
                "source_index": normalized_source,
                "sdf_ordinal": normalized_ordinal,
                "train_ordinal": normalized_ordinal,
                "smiles": smiles,
                "smiles_hash": smiles_hash(smiles),
                "geometry_artifact": normalized_geometry_artifact,
                "geometry_artifact_sha256": (
                    normalized_geometry_artifact_sha256
                ),
                "geometry_key": normalized_geometry_key,
                "geometry_payload_sha256": (
                    normalized_geometry_payload_sha256
                ),
                "grid_shape": list(result.grid.shape),
                "electron_count": result.electron_count,
                "integrated_electrons": stored_integral,
                "prequantization_integrated_electrons": (
                    prequantization_integral
                ),
                "integration_error": (
                    stored_integral - result.electron_count
                ),
                "prequantization_integration_error": (
                    prequantization_integral - result.electron_count
                ),
                "overflow": result.overflow,
                "conformers_used": result.conformers_used.tolist(),
                "normalization_requested": result.normalization_requested,
                "normalization_applied": result.normalization_applied,
            }
        )
    for entry in entries:
        _validate_density_record_arrays(
            arrays,
            entry,
            artifact=shard_path,
        )
    core_metadata: dict[str, Any] = {
        "schema": DENSITY_SCHEMA,
        "shard_id": normalized_shard_id,
        "filename": filename,
        "record_count": len(entries),
        "records": entries,
    }
    arrays[EMBEDDED_DENSITY_METADATA_KEY] = np.asarray(
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
        directory / f"density_{normalized_shard_id:06d}.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    return metadata


def _recover_density_sidecar(artifact: Path, sidecar: Path) -> None:
    try:
        with np.load(artifact, allow_pickle=False) as arrays:
            if EMBEDDED_DENSITY_METADATA_KEY not in arrays.files:
                raise ArtifactIntegrityError(
                    "orphan density artifact lacks embedded recovery "
                    f"metadata: {artifact}"
                )
            raw_metadata = arrays[EMBEDDED_DENSITY_METADATA_KEY]
            if raw_metadata.shape != ():
                raise ArtifactIntegrityError(
                    f"invalid embedded density metadata shape: {artifact}"
                )
            embedded = json.loads(str(raw_metadata.item()))
    except ArtifactIntegrityError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"cannot recover orphan density artifact: {artifact}"
        ) from exc
    records = embedded.get("records")
    expected_id_text = artifact.stem.removeprefix("density_")
    try:
        embedded_shard_id = _identity_integer(
            embedded["shard_id"],
            field=f"{artifact}.shard_id",
        )
        record_count = _identity_integer(
            embedded["record_count"],
            field=f"{artifact}.record_count",
        )
    except (KeyError, ValueError) as exc:
        raise ArtifactIntegrityError(
            f"invalid embedded density metadata: {artifact}"
        ) from exc
    if (
        embedded.get("schema") != DENSITY_SCHEMA
        or embedded.get("filename") != artifact.name
        or not expected_id_text.isdigit()
        or embedded_shard_id is None
        or embedded_shard_id < 0
        or embedded_shard_id != int(expected_id_text)
        or record_count is None
        or not isinstance(records, list)
        or not records
        or record_count != len(records)
    ):
        raise ArtifactIntegrityError(
            f"embedded density metadata contract mismatch: {artifact}"
        )
    metadata = {**embedded, "sha256": _sha256_file(artifact)}
    _atomic_write_text(
        sidecar,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )


def reconcile_density_artifacts(
    output_dir: os.PathLike[str] | str,
) -> None:
    """Converge interrupted density shard publication without data loss."""

    directory = Path(output_dir)
    if not directory.exists():
        return
    removed_temporary = False
    for pattern in (
        ".density_*.tmp",
        ".statistics.json.*.tmp",
        "..statistics.ready.json.*.tmp",
        ".run_state.json.*.tmp",
    ):
        for temporary in directory.glob(pattern):
            temporary.unlink()
            removed_temporary = True
    if removed_temporary:
        _fsync_directory(directory)
    artifacts = {path.stem: path for path in directory.glob("density_*.npz")}
    sidecars = {path.stem: path for path in directory.glob("density_*.json")}
    orphan_sidecars = sorted(set(sidecars) - set(artifacts))
    if orphan_sidecars:
        raise ArtifactIntegrityError(
            "density sidecars reference missing artifacts: "
            f"{orphan_sidecars}"
        )
    for stem in sorted(set(artifacts) - set(sidecars)):
        _recover_density_sidecar(
            artifacts[stem],
            directory / f"{stem}.json",
        )


def _record_ordinal(
    entry: dict[str, Any],
    *,
    context: str,
) -> Optional[int]:
    has_sdf_ordinal = "sdf_ordinal" in entry
    has_train_ordinal = "train_ordinal" in entry
    sdf_ordinal = _normalize_ordinal(
        entry.get("sdf_ordinal"),
        field=f"{context}.sdf_ordinal",
    )
    train_ordinal = _normalize_ordinal(
        entry.get("train_ordinal"),
        field=f"{context}.train_ordinal",
    )
    if (
        has_sdf_ordinal
        and has_train_ordinal
        and sdf_ordinal != train_ordinal
    ):
        raise ArtifactIntegrityError(
            f"conflicting sdf_ordinal/train_ordinal in {context}"
        )
    return (
        sdf_ordinal
        if has_sdf_ordinal
        else train_ordinal
    )


def _validate_density_record_arrays(
    arrays: Any,
    entry: dict[str, Any],
    *,
    artifact: Path,
) -> None:
    key = str(entry["key"])

    def value(field: str) -> np.ndarray:
        return np.asarray(arrays[f"{key}__{field}"])

    grid = value("grid")
    origin = value("origin")
    spacing = value("spacing")
    electron_count = value("electron_count")
    integrated = value("integrated_electrons")
    prequantization_integrated = value(
        "prequantization_integrated_electrons"
    )
    overflow = value("overflow")
    overflow_axes = value("overflow_axes")
    atomic_sigmas = value("atomic_sigmas")
    method = value("method")
    box_padding = value("box_padding")
    conformers_used = value("conformers_used")
    reduction = value("conformer_reduction")
    alignment = value("conformer_alignment")
    normalization_requested = value("normalization_requested")
    normalization_applied = value("normalization_applied")

    raw_grid_shape = entry.get("grid_shape")
    if (
        not isinstance(raw_grid_shape, list)
        or len(raw_grid_shape) != 3
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 2
            for dimension in raw_grid_shape
        )
    ):
        raise ArtifactIntegrityError(
            f"invalid grid_shape for {key} in {artifact}"
        )
    expected_grid_shape = tuple(int(value) for value in raw_grid_shape)
    if len(set(expected_grid_shape)) != 1:
        raise ArtifactIntegrityError(
            f"density grid must be cubic for {key} in {artifact}"
        )
    scalar_arrays = {
        "spacing": spacing,
        "electron_count": electron_count,
        "integrated_electrons": integrated,
        "prequantization_integrated_electrons": prequantization_integrated,
        "overflow": overflow,
        "method": method,
        "box_padding": box_padding,
        "conformer_reduction": reduction,
        "conformer_alignment": alignment,
        "normalization_requested": normalization_requested,
        "normalization_applied": normalization_applied,
    }
    invalid_scalar = [
        name for name, array in scalar_arrays.items() if array.shape != ()
    ]
    if invalid_scalar:
        raise ArtifactIntegrityError(
            f"non-scalar density fields {invalid_scalar} for {key}"
        )
    if (
        grid.shape != expected_grid_shape
        or grid.dtype != np.dtype(np.float32)
        or not np.all(np.isfinite(grid))
        or np.any(grid < 0)
        or origin.shape != (3,)
        or origin.dtype != np.dtype(np.float32)
        or not np.all(np.isfinite(origin))
        or spacing.dtype != np.dtype(np.float32)
        or not np.isfinite(float(spacing))
        or float(spacing) <= 0
        or electron_count.dtype != np.dtype(np.float64)
        or not np.isfinite(float(electron_count))
        or float(electron_count) <= 0
        or not float(electron_count).is_integer()
        or integrated.dtype != np.dtype(np.float64)
        or not np.isfinite(float(integrated))
        or float(integrated) <= 0
        or prequantization_integrated.dtype != np.dtype(np.float64)
        or not np.isfinite(float(prequantization_integrated))
        or float(prequantization_integrated) <= 0
        or overflow.dtype != np.dtype(np.bool_)
        or overflow_axes.shape != (3,)
        or overflow_axes.dtype != np.dtype(np.bool_)
        or atomic_sigmas.ndim != 1
        or atomic_sigmas.size == 0
        or atomic_sigmas.dtype != np.dtype(np.float32)
        or not np.all(np.isfinite(atomic_sigmas))
        or np.any(atomic_sigmas <= 0)
        or method.dtype.kind != "U"
        or str(method.item()) != "promolecular_gaussian"
        or box_padding.dtype != np.dtype(np.float32)
        or not np.isfinite(float(box_padding))
        or float(box_padding) < 0
        or conformers_used.ndim != 1
        or conformers_used.size == 0
        or conformers_used.dtype != np.dtype(np.int64)
        or np.any(conformers_used < 0)
        or np.unique(conformers_used).size != conformers_used.size
        or reduction.dtype.kind != "U"
        or str(reduction.item()) not in {"single", "mean"}
        or alignment.dtype.kind != "U"
        or str(alignment.item()) not in {"none", "heavy_atom_kabsch"}
        or normalization_requested.dtype.kind != "U"
        or str(normalization_requested.item())
        not in {"discrete_electron_count", "continuous_gaussian"}
        or normalization_applied.dtype.kind != "U"
        or str(normalization_applied.item())
        not in {"discrete_electron_count", "continuous_gaussian"}
    ):
        raise ArtifactIntegrityError(
            f"invalid density array contract for {key} in {artifact}"
        )
    expected_conformers = entry.get("conformers_used")
    if (
        not isinstance(expected_conformers, list)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in expected_conformers
        )
        or len(set(expected_conformers)) != len(expected_conformers)
        or conformers_used.tolist() != expected_conformers
    ):
        raise ArtifactIntegrityError(
            f"conformers_used metadata mismatch for {key} in {artifact}"
        )
    numeric_metadata = {
        "electron_count": float(electron_count),
        "integrated_electrons": float(integrated),
        "prequantization_integrated_electrons": float(
            prequantization_integrated
        ),
    }
    for name, actual in numeric_metadata.items():
        raw = entry.get(name)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not np.isfinite(float(raw))
            or not np.isclose(float(raw), actual, rtol=1e-7, atol=1e-7)
        ):
            raise ArtifactIntegrityError(
                f"{name} metadata mismatch for {key} in {artifact}"
            )
    if entry.get("overflow") is not bool(overflow.item()):
        raise ArtifactIntegrityError(
            f"overflow metadata mismatch for {key} in {artifact}"
        )
    requested = str(normalization_requested.item())
    applied = str(normalization_applied.item())
    overflow_value = bool(overflow.item())
    reduction_value = str(reduction.item())
    alignment_value = str(alignment.item())
    if (
        overflow_value != bool(np.any(overflow_axes))
        or (
            reduction_value == "single"
            and (
                conformers_used.size != 1
                or alignment_value != "none"
            )
        )
        or (
            reduction_value == "mean"
            and (
                (
                    conformers_used.size == 1
                    and alignment_value != "none"
                )
                or (
                    conformers_used.size > 1
                    and alignment_value != "heavy_atom_kabsch"
                )
            )
        )
    ):
        raise ArtifactIntegrityError(
            f"density physical metadata mismatch for {key} in {artifact}"
        )
    integrated_from_grid = float(
        grid.sum(dtype=np.float64) * float(spacing) ** 3
    )
    if not np.isclose(
        integrated_from_grid,
        float(integrated),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ArtifactIntegrityError(
            f"integrated density mismatch for {key} in {artifact}"
        )
    expected_applied = (
        "discrete_electron_count"
        if requested == "discrete_electron_count"
        and not overflow_value
        else "continuous_gaussian"
    )
    if (
        applied != expected_applied
        or entry.get("normalization_requested") != requested
        or entry.get("normalization_applied") != applied
        or (
            applied == "discrete_electron_count"
            and not np.isclose(
                float(prequantization_integrated),
                float(electron_count),
                rtol=1e-9,
                atol=1e-7,
            )
        )
    ):
        raise ArtifactIntegrityError(
            f"normalization provenance mismatch for {key} in {artifact}"
        )
    integration_error = entry.get("integration_error")
    expected_error = float(integrated) - float(electron_count)
    if (
        isinstance(integration_error, bool)
        or not isinstance(integration_error, (int, float))
        or not np.isfinite(float(integration_error))
        or not np.isclose(
            float(integration_error),
            expected_error,
            rtol=1e-7,
            atol=1e-7,
        )
    ):
        raise ArtifactIntegrityError(
            f"integration_error metadata mismatch for {key} in {artifact}"
        )
    prequantization_error = entry.get(
        "prequantization_integration_error"
    )
    expected_prequantization_error = (
        float(prequantization_integrated) - float(electron_count)
    )
    if (
        isinstance(prequantization_error, bool)
        or not isinstance(prequantization_error, (int, float))
        or not np.isfinite(float(prequantization_error))
        or not np.isclose(
            float(prequantization_error),
            expected_prequantization_error,
            rtol=1e-7,
            atol=1e-7,
        )
    ):
        raise ArtifactIntegrityError(
            "prequantization_integration_error metadata mismatch "
            f"for {key} in {artifact}"
        )


def _iter_verified_density_sidecars(
    output_dir: os.PathLike[str] | str,
    *,
    verify_checksums: bool,
    validate_payloads: bool = True,
    reconcile: bool = True,
) -> Iterator[dict[str, Any]]:
    directory = Path(output_dir)
    if reconcile:
        reconcile_density_artifacts(directory)
    artifacts = {path.stem: path for path in directory.glob("density_*.npz")}
    sidecar_paths = {
        path.stem: path for path in directory.glob("density_*.json")
    }
    if set(artifacts) != set(sidecar_paths):
        raise ArtifactIntegrityError(
            "incomplete density shard pairs; "
            f"missing_artifacts={sorted(set(sidecar_paths) - set(artifacts))}, "
            f"missing_sidecars={sorted(set(artifacts) - set(sidecar_paths))}"
        )
    for stem in sorted(artifacts):
        sidecar_path = sidecar_paths[stem]
        try:
            metadata = json.loads(
                sidecar_path.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_json_object,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"invalid density sidecar: {sidecar_path}"
            ) from exc
        if (
            not isinstance(metadata, dict)
            or set(metadata) != DENSITY_SIDECAR_FIELDS
            or metadata.get("schema") != DENSITY_SCHEMA
        ):
            raise ArtifactIntegrityError(
                f"invalid density sidecar contract: {sidecar_path}"
            )
        artifact = artifacts[stem]
        if metadata.get("filename") != artifact.name:
            raise ArtifactIntegrityError(
                f"sidecar artifact mismatch for {sidecar_path}"
            )
        try:
            shard_id = _identity_integer(
                metadata["shard_id"],
                field=f"{sidecar_path}.shard_id",
            )
            record_count = _identity_integer(
                metadata["record_count"],
                field=f"{sidecar_path}.record_count",
            )
        except (KeyError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"invalid shard_id in {sidecar_path}"
            ) from exc
        if shard_id is None or record_count is None:
            raise ArtifactIntegrityError(
                f"invalid shard metadata in {sidecar_path}"
            )
        expected_stem = f"density_{shard_id:06d}"
        if shard_id < 0 or stem != expected_stem:
            raise ArtifactIntegrityError(
                f"density shard naming mismatch: {sidecar_path}"
            )
        records = metadata.get("records")
        if (
            not isinstance(records, list)
            or not records
            or record_count != len(records)
        ):
            raise ArtifactIntegrityError(
                f"invalid record_count in {sidecar_path}"
            )
        record_keys: set[str] = set()
        row_indices: set[int] = set()
        source_indices: set[int] = set()
        for ordinal, entry in enumerate(records):
            if not isinstance(entry, dict):
                raise ArtifactIntegrityError(
                    f"record metadata must be an object in {sidecar_path}"
                )
            if set(entry) != DENSITY_RECORD_METADATA_FIELDS:
                raise ArtifactIntegrityError(
                    "density record metadata fields differ from schema in "
                    f"{sidecar_path}"
                )
            key = entry.get("key")
            if key != f"r{ordinal:06d}":
                raise ArtifactIntegrityError(
                    f"non-canonical record key in {sidecar_path}: {key!r}"
                )
            if key in record_keys:
                raise ArtifactIntegrityError(
                    f"duplicate record key {key!r} in {sidecar_path}"
                )
            record_keys.add(key)
            try:
                row_index = _identity_integer(
                    entry["row_index"],
                    field=f"{sidecar_path}.row_index",
                )
                source_index = _identity_integer(
                    entry["source_index"],
                    field=f"{sidecar_path}.source_index",
                )
                _record_ordinal(
                    entry,
                    context=f"{sidecar_path}:{key}",
                )
                _geometry_provenance(
                    entry["geometry_artifact"],
                    entry["geometry_artifact_sha256"],
                    entry["geometry_key"],
                    entry["geometry_payload_sha256"],
                    context=f"{sidecar_path}:{key}",
                )
            except (KeyError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    f"invalid density identity in {sidecar_path}"
                ) from exc
            smiles = entry.get("smiles")
            if (
                row_index is None
                or row_index < 0
                or source_index is None
                or source_index < 0
                or not isinstance(smiles, str)
                or not smiles
            ):
                raise ArtifactIntegrityError(
                    f"invalid density record identity in {sidecar_path}"
                )
            if row_index in row_indices:
                raise ArtifactIntegrityError(
                    f"duplicate density row_index={row_index}"
                )
            if source_index in source_indices:
                raise ArtifactIntegrityError(
                    f"duplicate density source_index={source_index}"
                )
            row_indices.add(row_index)
            source_indices.add(source_index)
            if entry.get("smiles_hash") != smiles_hash(smiles):
                raise ArtifactIntegrityError(
                    f"SMILES hash mismatch for row_index={row_index}"
                )
        expected_checksum = metadata.get("sha256")
        if not _is_sha256(expected_checksum):
            raise ArtifactIntegrityError(
                f"invalid checksum metadata for density shard {artifact}"
            )
        try:
            with artifact.open("rb") as artifact_handle:
                before = os.fstat(artifact_handle.fileno())
                if verify_checksums:
                    actual_checksum = _hash_open_file(artifact_handle)
                    if actual_checksum != expected_checksum:
                        raise ArtifactIntegrityError(
                            f"checksum mismatch for density shard {artifact}"
                        )
                after_hash = os.fstat(artifact_handle.fileno())
                path_after_hash = artifact.stat()
                if (
                    _file_identity(before) != _file_identity(after_hash)
                    or _file_identity(after_hash)
                    != _file_identity(path_after_hash)
                ):
                    raise ArtifactIntegrityError(
                        "density shard changed before verification: "
                        f"{artifact}"
                    )
                artifact_handle.seek(0)
                with np.load(artifact_handle, allow_pickle=False) as arrays:
                    available = set(arrays.files)
                    expected_arrays = {
                        f"{entry['key']}__{field}"
                        for entry in records
                        for field in DENSITY_REQUIRED_FIELDS
                    }
                    allowed_arrays = expected_arrays | {
                        EMBEDDED_DENSITY_METADATA_KEY
                    }
                    if (
                        expected_arrays - available
                        or available - allowed_arrays
                    ):
                        raise ArtifactIntegrityError(
                            "density artifact array inventory differs for "
                            f"{artifact}; "
                            f"missing={sorted(expected_arrays - available)}, "
                            f"unexpected={sorted(available - allowed_arrays)}"
                        )
                    if EMBEDDED_DENSITY_METADATA_KEY in available:
                        embedded_array = arrays[
                            EMBEDDED_DENSITY_METADATA_KEY
                        ]
                        if embedded_array.shape != ():
                            raise ArtifactIntegrityError(
                                f"invalid embedded metadata in {artifact}"
                            )
                        embedded = json.loads(str(embedded_array.item()))
                        core_metadata = {
                            key: value
                            for key, value in metadata.items()
                            if key != "sha256"
                        }
                        if embedded != core_metadata:
                            raise ArtifactIntegrityError(
                                f"embedded metadata mismatch in {artifact}"
                            )
                    if (
                        validate_payloads
                        or EMBEDDED_DENSITY_METADATA_KEY not in available
                    ):
                        for entry in records:
                            _validate_density_record_arrays(
                                arrays,
                                entry,
                                artifact=artifact,
                            )
                after_load = os.fstat(artifact_handle.fileno())
                path_after_load = artifact.stat()
                if (
                    _file_identity(after_hash) != _file_identity(after_load)
                    or _file_identity(after_load)
                    != _file_identity(path_after_load)
                ):
                    raise ArtifactIntegrityError(
                        f"density shard changed during verification: {artifact}"
                    )
        except ArtifactIntegrityError:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                f"cannot inspect density shard: {artifact}"
            ) from exc
        yield metadata


def _complete_density_sidecars(
    output_dir: os.PathLike[str] | str,
    verify_checksums: bool,
) -> list[dict[str, Any]]:
    """Compatibility helper for callers that explicitly need materialization."""

    return list(
        _iter_verified_density_sidecars(
            output_dir,
            verify_checksums=verify_checksums,
        )
    )


def load_completed_rows(
    output_dir: os.PathLike[str] | str,
    verify_checksums: bool = True,
) -> set[int]:
    """Return row indices present in verified density shards."""
    completed: set[int] = set()
    if not Path(output_dir).exists():
        return completed
    for metadata in _iter_verified_density_sidecars(
        output_dir,
        verify_checksums=verify_checksums,
    ):
        for entry in metadata["records"]:
            row_index = int(entry["row_index"])
            if row_index in completed:
                raise ArtifactIntegrityError(
                    f"duplicate density row_index={row_index}"
                )
            completed.add(row_index)
    return completed


class _CompletedRowIndex:
    """Disk-backed uniqueness and membership index for resumable 3M runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for candidate in (
            self.path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
            self.path.with_name(f"{self.path.name}-journal"),
        ):
            candidate.unlink(missing_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE completed_rows ("
            "row_index INTEGER PRIMARY KEY,"
            "source_index INTEGER NOT NULL UNIQUE,"
            "train_ordinal INTEGER,"
            "smiles_hash TEXT NOT NULL,"
            "geometry_artifact TEXT NOT NULL,"
            "geometry_artifact_sha256 TEXT NOT NULL,"
            "geometry_key TEXT NOT NULL,"
            "geometry_payload_sha256 TEXT NOT NULL"
            ")"
        )
        self.connection.execute(
            "CREATE TABLE completed_shards ("
            "shard_id INTEGER PRIMARY KEY,"
            "filename TEXT NOT NULL UNIQUE"
            ")"
        )
        self.connection.execute(
            "CREATE TABLE geometry_rows ("
            "row_index INTEGER PRIMARY KEY,"
            "source_index INTEGER NOT NULL UNIQUE,"
            "train_ordinal INTEGER,"
            "smiles_hash TEXT NOT NULL,"
            "geometry_artifact TEXT NOT NULL,"
            "geometry_artifact_sha256 TEXT NOT NULL,"
            "geometry_key TEXT NOT NULL,"
            "geometry_payload_sha256 TEXT NOT NULL"
            ")"
        )
        self.connection.execute(
            "CREATE TABLE failure_rows ("
            "row_index INTEGER PRIMARY KEY,"
            "source_index INTEGER NOT NULL UNIQUE,"
            "train_ordinal INTEGER,"
            "smiles_hash TEXT NOT NULL,"
            "geometry_artifact TEXT NOT NULL,"
            "geometry_artifact_sha256 TEXT NOT NULL,"
            "geometry_key TEXT NOT NULL,"
            "geometry_payload_sha256 TEXT NOT NULL"
            ")"
        )

    def populate(
        self,
        output_dir: Path,
        *,
        verify_checksums: bool,
    ) -> list[dict[str, Any]]:
        shard_inventory: list[dict[str, Any]] = []
        for metadata in _iter_verified_density_sidecars(
            output_dir,
            verify_checksums=verify_checksums,
            validate_payloads=True,
        ):
            shard_inventory.append(
                {
                    "shard_id": int(metadata["shard_id"]),
                    "filename": str(metadata["filename"]),
                    "sha256": str(metadata["sha256"]),
                    "record_count": int(metadata["record_count"]),
                }
            )
            try:
                with self.connection:
                    self.connection.execute(
                        "INSERT INTO completed_shards(shard_id,filename) "
                        "VALUES (?,?)",
                        (
                            int(metadata["shard_id"]),
                            str(metadata["filename"]),
                        ),
                    )
                    self.connection.executemany(
                        "INSERT INTO completed_rows "
                        "(row_index,source_index,train_ordinal,"
                        "smiles_hash,"
                        "geometry_artifact,geometry_artifact_sha256,"
                        "geometry_key,geometry_payload_sha256) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (
                            (
                                int(entry["row_index"]),
                                int(entry["source_index"]),
                                _record_ordinal(
                                    entry,
                                    context=(
                                        f"{metadata['filename']}:"
                                        f"{entry['key']}"
                                    ),
                                ),
                                str(entry["smiles_hash"]),
                                str(entry["geometry_artifact"]),
                                str(entry["geometry_artifact_sha256"]),
                                str(entry["geometry_key"]),
                                str(entry["geometry_payload_sha256"]),
                            )
                            for entry in metadata["records"]
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                raise ArtifactIntegrityError(
                    "duplicate density shard_id, filename, row_index, "
                    "or source_index"
                ) from exc
        return shard_inventory

    def register_geometry_rows(
        self,
        rows: Iterable[
            tuple[
                int,
                int,
                Optional[int],
                str,
                str,
                str,
                str,
                str,
            ]
        ],
    ) -> None:
        normalized = []
        for (
            row_index,
            source_index,
            train_ordinal,
            expected_smiles_hash,
            geometry_artifact,
            geometry_artifact_sha256,
            geometry_key,
            geometry_payload_sha256,
        ) in rows:
            normalized_row = _identity_integer(
                row_index,
                field="geometry.row_index",
            )
            normalized_source = _identity_integer(
                source_index,
                field="geometry.source_index",
            )
            normalized_ordinal = _normalize_ordinal(
                train_ordinal,
                field="geometry.sdf_ordinal",
            )
            if not _is_sha256(expected_smiles_hash):
                raise ArtifactIntegrityError(
                    "geometry smiles_hash must be a canonical SHA-256"
                )
            provenance = _geometry_provenance(
                geometry_artifact,
                geometry_artifact_sha256,
                geometry_key,
                geometry_payload_sha256,
                context=f"geometry row_index={normalized_row}",
            )
            if (
                normalized_row is None
                or normalized_row < 0
                or normalized_source is None
                or normalized_source < 0
            ):
                raise ArtifactIntegrityError(
                    "geometry row/source identity must be non-negative"
                )
            normalized.append(
                (
                    normalized_row,
                    normalized_source,
                    normalized_ordinal,
                    expected_smiles_hash,
                    *provenance,
                )
            )
        try:
            with self.connection:
                self.connection.executemany(
                    "INSERT INTO geometry_rows "
                    "(row_index,source_index,train_ordinal,"
                    "smiles_hash,"
                    "geometry_artifact,geometry_artifact_sha256,"
                    "geometry_key,geometry_payload_sha256) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    normalized,
                )
        except sqlite3.IntegrityError as exc:
            raise ArtifactIntegrityError(
                "duplicate geometry row_index or source_index"
            ) from exc

    def register_failure_rows(
        self,
        rows: Iterable[
            tuple[
                int,
                int,
                Optional[int],
                str,
                str,
                str,
                str,
                str,
            ]
        ],
    ) -> None:
        try:
            with self.connection:
                self.connection.executemany(
                    "INSERT INTO failure_rows "
                    "(row_index,source_index,train_ordinal,smiles_hash,"
                    "geometry_artifact,geometry_artifact_sha256,"
                    "geometry_key,geometry_payload_sha256) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    rows,
                )
        except sqlite3.IntegrityError as exc:
            raise ArtifactIntegrityError(
                "duplicate failure row_index or source_index"
            ) from exc

    @property
    def count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM completed_rows"
        ).fetchone()
        return int(row[0])

    @property
    def geometry_count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM geometry_rows"
        ).fetchone()
        return int(row[0])

    @property
    def failure_count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM failure_rows"
        ).fetchone()
        return int(row[0])

    def existing(self, row_indices: Iterable[int]) -> set[int]:
        values = [int(value) for value in row_indices]
        found: set[int] = set()
        for start in range(0, len(values), 900):
            chunk = values[start : start + 900]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                "SELECT row_index FROM completed_rows "
                f"WHERE row_index IN ({placeholders})",
                chunk,
            )
            found.update(int(row[0]) for row in rows)
        return found

    def validate_completed_coverage(self) -> None:
        mismatched = self.connection.execute(
            "SELECT c.row_index,c.source_index,g.source_index "
            "FROM completed_rows AS c JOIN geometry_rows AS g "
            "USING(row_index) "
            "WHERE c.source_index != g.source_index "
            "OR c.train_ordinal IS NOT g.train_ordinal "
            "OR c.smiles_hash != g.smiles_hash "
            "OR c.geometry_artifact != g.geometry_artifact "
            "OR c.geometry_artifact_sha256 != g.geometry_artifact_sha256 "
            "OR c.geometry_key != g.geometry_key "
            "OR c.geometry_payload_sha256 != g.geometry_payload_sha256 "
            "LIMIT 1"
        ).fetchone()
        if mismatched is not None:
            raise ArtifactIntegrityError(
                "completed density identity or provenance differs from "
                "geometry for "
                f"row_index={int(mismatched[0])}"
            )
        orphan = self.connection.execute(
            "SELECT c.row_index FROM completed_rows AS c "
            "LEFT JOIN geometry_rows AS g USING(row_index) "
            "WHERE g.row_index IS NULL LIMIT 1"
        ).fetchone()
        if orphan is not None:
            raise ArtifactIntegrityError(
                "completed density row is absent from current geometry: "
                f"row_index={int(orphan[0])}"
            )

    def validate_final_outcomes(self, expected_records: int) -> None:
        self.validate_completed_coverage()
        mismatched_failure = self.connection.execute(
            "SELECT f.row_index FROM failure_rows AS f "
            "JOIN geometry_rows AS g USING(row_index) "
            "WHERE f.source_index != g.source_index "
            "OR f.train_ordinal IS NOT g.train_ordinal "
            "OR f.smiles_hash != g.smiles_hash "
            "OR f.geometry_artifact != g.geometry_artifact "
            "OR f.geometry_artifact_sha256 != g.geometry_artifact_sha256 "
            "OR f.geometry_key != g.geometry_key "
            "OR f.geometry_payload_sha256 != g.geometry_payload_sha256 "
            "LIMIT 1"
        ).fetchone()
        if mismatched_failure is not None:
            raise ArtifactIntegrityError(
                "failed density identity or provenance differs from geometry "
                f"for row_index={int(mismatched_failure[0])}"
            )
        orphan_failure = self.connection.execute(
            "SELECT f.row_index FROM failure_rows AS f "
            "LEFT JOIN geometry_rows AS g USING(row_index) "
            "WHERE g.row_index IS NULL LIMIT 1"
        ).fetchone()
        if orphan_failure is not None:
            raise ArtifactIntegrityError(
                "failed density row is absent from current geometry: "
                f"row_index={int(orphan_failure[0])}"
            )
        overlap = self.connection.execute(
            "SELECT c.row_index FROM completed_rows AS c "
            "JOIN failure_rows AS f "
            "ON c.row_index=f.row_index "
            "OR c.source_index=f.source_index LIMIT 1"
        ).fetchone()
        if overlap is not None:
            raise ArtifactIntegrityError(
                "density success and failure outcomes overlap for "
                f"row_index={int(overlap[0])}"
            )
        missing = self.connection.execute(
            "SELECT g.row_index FROM geometry_rows AS g "
            "LEFT JOIN ("
            "SELECT row_index FROM completed_rows "
            "UNION ALL SELECT row_index FROM failure_rows"
            ") AS outcomes USING(row_index) "
            "WHERE outcomes.row_index IS NULL LIMIT 1"
        ).fetchone()
        if missing is not None:
            raise ArtifactIntegrityError(
                "geometry row has no density success or failure outcome: "
                f"row_index={int(missing[0])}"
            )
        if (
            self.geometry_count != expected_records
            or self.count + self.failure_count != expected_records
        ):
            raise ArtifactIntegrityError(
                "density success/failure outcomes do not exactly cover "
                "the geometry snapshot"
            )

    def close(self) -> None:
        self.connection.close()
        for candidate in (
            self.path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
            self.path.with_name(f"{self.path.name}-journal"),
        ):
            candidate.unlink(missing_ok=True)

    def __enter__(self) -> "_CompletedRowIndex":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _file_identity(stat_result: os.stat_result) -> tuple[int, ...]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _hash_open_file(
    handle: BinaryIO,
    *,
    block_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for block in iter(lambda: handle.read(block_size), b""):
        digest.update(block)
    return digest.hexdigest()


def _pinned_file_descriptor(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            checksum = _hash_open_file(handle)
            after = os.fstat(handle.fileno())
        path_after = path.stat()
    except OSError as exc:
        raise ArtifactIntegrityError(
            f"cannot read pinned snapshot file: {path}"
        ) from exc
    if (
        _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(path_after)
    ):
        raise ArtifactIntegrityError(
            f"snapshot file changed while it was hashed: {path}"
        )
    return {
        "name": path.name,
        "size": int(after.st_size),
        "sha256": checksum,
    }


def _read_pinned_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    digest = hashlib.sha256()
    content = bytearray()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                content.extend(block)
            after = os.fstat(handle.fileno())
        path_after = path.stat()
    except OSError as exc:
        raise ArtifactIntegrityError(
            f"cannot read pinned input file: {path}"
        ) from exc
    if (
        _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(path_after)
    ):
        raise ArtifactIntegrityError(
            f"input file changed while it was read: {path}"
        )
    return bytes(content), {
        "name": path.name,
        "size": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


class _OutputRunLock:
    """Hold a crash-released, process-wide lock for one density output."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._descriptor: Optional[int] = None

    def __enter__(self) -> "_OutputRunLock":
        if os.name == "posix":
            import fcntl

            flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            descriptor = os.open(self.output_dir, flags)
            try:
                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except OSError as exc:
                os.close(descriptor)
                raise RuntimeError(
                    "another density writer already owns the output directory"
                ) from exc
            self._descriptor = descriptor
            return self
        if os.name == "nt":
            import msvcrt

            lock_path = self.output_dir / WRITER_LOCK_FILENAME
            descriptor = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT,
                0o600,
            )
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                os.close(descriptor)
                raise RuntimeError(
                    "another density writer already owns the output directory"
                ) from exc
            self._descriptor = descriptor
            return self
        raise RuntimeError(
            f"unsupported platform for density writer locking: {os.name}"
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is None:
            return
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(descriptor)


def _copy_pinned_file(
    source: Path,
    destination: Path,
    *,
    expected_size: Optional[int] = None,
    expected_sha256: Optional[str] = None,
) -> dict[str, Any]:
    if destination.exists():
        raise ArtifactIntegrityError(
            f"snapshot destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_handle:
            source_before = os.fstat(source_handle.fileno())
            with destination.open("xb") as destination_handle:
                for block in iter(
                    lambda: source_handle.read(1024 * 1024),
                    b"",
                ):
                    digest.update(block)
                    destination_handle.write(block)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            source_after = os.fstat(source_handle.fileno())
        source_path_after = source.stat()
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    checksum = digest.hexdigest()
    if (
        _file_identity(source_before) != _file_identity(source_after)
        or _file_identity(source_after) != _file_identity(source_path_after)
    ):
        destination.unlink(missing_ok=True)
        raise ArtifactIntegrityError(
            f"geometry source changed while snapshotting: {source}"
        )
    if expected_size is not None and int(source_after.st_size) != expected_size:
        destination.unlink(missing_ok=True)
        raise ArtifactIntegrityError(
            f"geometry source size differs from pinned contract: {source}"
        )
    if expected_sha256 is not None and checksum != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ArtifactIntegrityError(
            f"geometry source checksum differs from pinned contract: {source}"
        )
    return {
        "name": destination.name,
        "size": int(source_after.st_size),
        "sha256": checksum,
    }


def _snapshot_input_descriptor(
    fingerprint: dict[str, Any],
    name: str,
    expected_filename: str,
) -> dict[str, Any]:
    inputs = fingerprint.get("inputs")
    descriptor = inputs.get(name) if isinstance(inputs, dict) else None
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"name", "size", "sha256"}
        or descriptor.get("name") != expected_filename
        or not isinstance(descriptor.get("size"), int)
        or isinstance(descriptor["size"], bool)
        or descriptor["size"] < 0
        or not _is_sha256(descriptor.get("sha256"))
    ):
        raise ArtifactIntegrityError(
            f"density fingerprint lacks pinned {name} bytes"
        )
    return descriptor


def _write_geometry_snapshot_poison(
    output_dir: Path,
    reason: str,
) -> None:
    poison_path = output_dir / GEOMETRY_SNAPSHOT_POISON_FILENAME
    if poison_path.exists():
        return
    _atomic_write_text(
        poison_path,
        json.dumps(
            {
                "schema": GEOMETRY_SNAPSHOT_SCHEMA,
                "reason": str(reason),
            },
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _has_density_derivatives(output_dir: Path) -> bool:
    if (
        next(output_dir.glob("density_*.npz"), None) is not None
        or next(output_dir.glob("density_*.json"), None) is not None
    ):
        return True
    failure_paths = [output_dir / "failures.jsonl"]
    failure_paths.extend(output_dir.glob(".failures.jsonl.*.tmp"))
    for failure_path in failure_paths:
        try:
            if failure_path.is_file() and failure_path.stat().st_size > 0:
                return True
        except OSError:
            return True
    for ledger_name in (
        ".density_completed.sqlite3",
        ".density_provenance_audit.sqlite3",
    ):
        ledger_path = output_dir / ledger_name
        if not ledger_path.exists():
            continue
        connection: Optional[sqlite3.Connection] = None
        try:
            ledger_uri = ledger_path.resolve(strict=True).as_uri()
            connection = sqlite3.connect(
                f"{ledger_uri}?mode=ro",
                uri=True,
            )
            row = connection.execute(
                "SELECT COUNT(*) FROM completed_rows"
            ).fetchone()
            if row is None or int(row[0]) > 0:
                return True
        except (OSError, sqlite3.Error, TypeError, ValueError):
            continue
        finally:
            if connection is not None:
                connection.close()
    return False


def _cleanup_geometry_snapshot_transients(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise ArtifactIntegrityError(
            f"density output directory is missing: {output_dir}"
        )
    building_path = output_dir / GEOMETRY_SNAPSHOT_BUILDING_DIRNAME
    candidates: list[Path] = []
    if building_path.exists() or building_path.is_symlink():
        candidates.append(building_path)
    for candidate in sorted(
        output_dir.iterdir(),
        key=lambda path: path.name,
    ):
        if not candidate.name.startswith(GEOMETRY_SNAPSHOT_DELETE_PREFIX):
            continue
        suffix = candidate.name.removeprefix(
            GEOMETRY_SNAPSHOT_DELETE_PREFIX
        )
        if (
            len(suffix) != 32
            or set(suffix) - set("0123456789abcdef")
        ):
            reason = (
                "malformed geometry snapshot deletion tombstone: "
                f"{candidate.name}"
            )
            _write_geometry_snapshot_poison(output_dir, reason)
            raise ArtifactIntegrityError(reason)
        candidates.append(candidate)
    resolved_output = output_dir.resolve(strict=True)
    for candidate in candidates:
        try:
            safe_directory = (
                candidate.parent == output_dir
                and not candidate.is_symlink()
                and candidate.is_dir()
                and not os.path.ismount(candidate)
                and candidate.resolve(strict=True).parent
                == resolved_output
            )
        except OSError as exc:
            raise ArtifactIntegrityError(
                f"cannot inspect private snapshot transient: {candidate}"
            ) from exc
        if not safe_directory:
            reason = (
                "refusing to remove unsafe private snapshot transient: "
                f"{candidate}"
            )
            _write_geometry_snapshot_poison(output_dir, reason)
            raise ArtifactIntegrityError(reason)
        retired: Optional[Path] = None
        for _attempt in range(16):
            tombstone = output_dir / (
                f"{GEOMETRY_SNAPSHOT_DELETE_PREFIX}{uuid.uuid4().hex}"
            )
            if not tombstone.exists() and not tombstone.is_symlink():
                retired = tombstone
                break
        if retired is None:
            raise ArtifactIntegrityError(
                "cannot allocate a transient cleanup tombstone"
            )
        try:
            os.replace(candidate, retired)
            _fsync_directory(output_dir)
        except OSError as exc:
            raise ArtifactIntegrityError(
                "private snapshot transient could not be atomically retired: "
                f"{candidate}"
            ) from exc
        try:
            retired_is_safe = (
                not retired.is_symlink()
                and retired.is_dir()
                and not os.path.ismount(retired)
                and retired.resolve(strict=True).parent == resolved_output
            )
        except OSError as exc:
            raise ArtifactIntegrityError(
                f"cannot inspect retired snapshot transient: {retired}"
            ) from exc
        if not retired_is_safe:
            reason = (
                "refusing to remove unsafe retired snapshot transient: "
                f"{retired}"
            )
            _write_geometry_snapshot_poison(output_dir, reason)
            raise ArtifactIntegrityError(reason)
        try:
            shutil.rmtree(retired)
            _fsync_directory(output_dir)
        except OSError as exc:
            raise ArtifactIntegrityError(
                "retired private snapshot cleanup did not complete: "
                f"{retired}"
            ) from exc


def _validate_geometry_snapshot(
    snapshot_dir: Path,
    fingerprint: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], int, str]:
    if not snapshot_dir.is_dir() or snapshot_dir.is_symlink():
        raise ArtifactIntegrityError(
            f"geometry snapshot is not a private directory: {snapshot_dir}"
        )
    state_path = snapshot_dir / GEOMETRY_SNAPSHOT_STATE_FILENAME
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"invalid geometry snapshot state: {state_path}"
        ) from exc
    if (
        not isinstance(state, dict)
        or set(state)
        != {
            "schema",
            "density_run_fingerprint_sha256",
            "geometry_run_fingerprint_sha256",
            "record_count",
            "inventory",
        }
        or state.get("schema") != GEOMETRY_SNAPSHOT_SCHEMA
        or state.get("density_run_fingerprint_sha256")
        != fingerprint.get("sha256")
        or not _is_sha256(
            state.get("geometry_run_fingerprint_sha256")
        )
        or not isinstance(state.get("record_count"), int)
        or isinstance(state["record_count"], bool)
        or state["record_count"] < 0
        or not isinstance(state.get("inventory"), list)
    ):
        raise ArtifactIntegrityError(
            f"geometry snapshot state contract mismatch: {state_path}"
        )
    inventory: dict[str, dict[str, Any]] = {}
    for entry in state["inventory"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"name", "size", "sha256"}
            or not isinstance(entry.get("name"), str)
            or not entry["name"]
            or Path(entry["name"]).name != entry["name"]
            or entry["name"] == GEOMETRY_SNAPSHOT_STATE_FILENAME
            or not isinstance(entry.get("size"), int)
            or isinstance(entry["size"], bool)
            or entry["size"] < 0
            or not _is_sha256(entry.get("sha256"))
            or entry["name"] in inventory
        ):
            raise ArtifactIntegrityError(
                f"invalid geometry snapshot inventory: {state_path}"
            )
        inventory[entry["name"]] = entry
    actual_children = list(snapshot_dir.iterdir())
    if any(not child.is_file() or child.is_symlink() for child in actual_children):
        raise ArtifactIntegrityError(
            f"geometry snapshot contains a non-file entry: {snapshot_dir}"
        )
    actual_names = {child.name for child in actual_children}
    expected_names_with_state = set(inventory) | {
        GEOMETRY_SNAPSHOT_STATE_FILENAME
    }
    if actual_names != expected_names_with_state:
        raise ArtifactIntegrityError(
            "geometry snapshot file inventory differs; "
            f"missing={sorted(expected_names_with_state - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names_with_state)}"
        )
    for name, expected in inventory.items():
        if _pinned_file_descriptor(snapshot_dir / name) != expected:
            raise ArtifactIntegrityError(
                f"geometry snapshot file differs from state: {name}"
            )

    inputs = fingerprint.get("inputs")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != DENSITY_FINGERPRINT_INPUT_FIELDS
    ):
        raise ArtifactIntegrityError(
            "density run fingerprint input contract is invalid"
        )
    pinned_metadata = {
        "manifest.json": _snapshot_input_descriptor(
            fingerprint,
            "geometry_manifest",
            "manifest.json",
        ),
        "geometry_index.json": _snapshot_input_descriptor(
            fingerprint,
            "geometry_index",
            "geometry_index.json",
        ),
        "run_state.json": _snapshot_input_descriptor(
            fingerprint,
            "geometry_run_state",
            "run_state.json",
        ),
    }
    for filename, expected in pinned_metadata.items():
        if inventory.get(filename) != expected:
            raise ArtifactIntegrityError(
                f"geometry snapshot metadata differs from run state: {filename}"
            )

    expected_shards, expected_records = _load_geometry_publication_contract(
        snapshot_dir,
        verify_checksums=True,
    )
    observed_shards = {
        str(metadata["filename"]): {
            "shard_id": int(metadata["shard_id"]),
            "filename": str(metadata["filename"]),
            "sha256": str(metadata["sha256"]),
            "record_count": int(metadata["record_count"]),
        }
        for metadata in _iter_verified_geometry_sidecars(
            snapshot_dir,
            verify_checksums=True,
            _reconcile=False,
        )
    }
    if observed_shards != expected_shards:
        raise ArtifactIntegrityError(
            "geometry snapshot shards differ from its manifest"
        )
    expected_inventory_names = {
        "manifest.json",
        "run_state.json",
        "geometry_index.json",
        "geometry_index.npz",
    }
    for filename in expected_shards:
        expected_inventory_names.add(filename)
        expected_inventory_names.add(
            f"{Path(filename).stem}.json"
        )
    if set(inventory) != expected_inventory_names:
        raise ArtifactIntegrityError(
            "geometry snapshot does not contain the exact publication "
            "inventory"
        )
    try:
        geometry_run_state = json.loads(
            (snapshot_dir / "run_state.json").read_text(encoding="utf-8")
        )
        geometry_fingerprint_sha256 = geometry_run_state["fingerprint"][
            "sha256"
        ]
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as exc:
        raise ArtifactIntegrityError(
            "cannot read geometry fingerprint from snapshot"
        ) from exc
    if (
        geometry_fingerprint_sha256
        != state["geometry_run_fingerprint_sha256"]
        or expected_records != state["record_count"]
    ):
        raise ArtifactIntegrityError(
            "geometry snapshot state differs from publication metadata"
        )
    return (
        expected_shards,
        expected_records,
        str(geometry_fingerprint_sha256),
    )


def _prepare_geometry_snapshot(
    geometry_dir: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    fingerprint: dict[str, Any],
) -> tuple[Path, dict[str, dict[str, Any]], int]:
    source_dir = Path(geometry_dir)
    destination_root = Path(output_dir)
    snapshot_dir = destination_root / GEOMETRY_SNAPSHOT_DIRNAME
    building_dir = destination_root / GEOMETRY_SNAPSHOT_BUILDING_DIRNAME
    poison_path = destination_root / GEOMETRY_SNAPSHOT_POISON_FILENAME
    _cleanup_geometry_snapshot_transients(destination_root)
    if poison_path.exists():
        raise ArtifactIntegrityError(
            f"geometry snapshot is poisoned: {poison_path}"
        )
    poison_temporaries = sorted(
        destination_root.glob(
            f".{GEOMETRY_SNAPSHOT_POISON_FILENAME}.*.tmp"
        )
    )
    if poison_temporaries:
        reason = (
            "interrupted geometry snapshot poison publication: "
            f"{[path.name for path in poison_temporaries]}"
        )
        _write_geometry_snapshot_poison(destination_root, reason)
        raise ArtifactIntegrityError(reason)
    if snapshot_dir.exists():
        try:
            expected_shards, expected_records, _geometry_fingerprint = (
                _validate_geometry_snapshot(snapshot_dir, fingerprint)
            )
        except Exception as exc:
            if _has_density_derivatives(destination_root):
                _write_geometry_snapshot_poison(
                    destination_root,
                    str(exc),
                )
                raise ArtifactIntegrityError(
                    "existing geometry snapshot is incomplete or mismatched "
                    "after density derivation"
                ) from exc
            _remove_geometry_snapshot(snapshot_dir)
        else:
            return snapshot_dir, expected_shards, expected_records

    try:
        building_dir.mkdir(parents=False, exist_ok=False)
        metadata_sources = {
            "manifest.json": _snapshot_input_descriptor(
                fingerprint,
                "geometry_manifest",
                "manifest.json",
            ),
            "geometry_index.json": _snapshot_input_descriptor(
                fingerprint,
                "geometry_index",
                "geometry_index.json",
            ),
            "run_state.json": _snapshot_input_descriptor(
                fingerprint,
                "geometry_run_state",
                "run_state.json",
            ),
        }
        for filename, descriptor in metadata_sources.items():
            _copy_pinned_file(
                source_dir / filename,
                building_dir / filename,
                expected_size=int(descriptor["size"]),
                expected_sha256=str(descriptor["sha256"]),
            )
        try:
            index_metadata = json.loads(
                (building_dir / "geometry_index.json").read_text(
                    encoding="utf-8"
                )
            )
            index_filename = index_metadata["filename"]
            index_sha256 = index_metadata["sha256"]
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as exc:
            raise ArtifactIntegrityError(
                "cannot read pinned geometry index metadata"
            ) from exc
        if (
            index_filename != "geometry_index.npz"
            or not _is_sha256(index_sha256)
        ):
            raise ArtifactIntegrityError(
                "pinned geometry index metadata is invalid"
            )
        _copy_pinned_file(
            source_dir / index_filename,
            building_dir / index_filename,
            expected_sha256=index_sha256,
        )
        expected_shards, expected_records = (
            _load_geometry_publication_contract(
                building_dir,
                verify_checksums=True,
            )
        )
        for filename in sorted(expected_shards):
            shard = expected_shards[filename]
            _copy_pinned_file(
                source_dir / filename,
                building_dir / filename,
                expected_sha256=str(shard["sha256"]),
            )
            sidecar_filename = f"{Path(filename).stem}.json"
            _copy_pinned_file(
                source_dir / sidecar_filename,
                building_dir / sidecar_filename,
            )
        observed_shards = {
            str(metadata["filename"]): {
                "shard_id": int(metadata["shard_id"]),
                "filename": str(metadata["filename"]),
                "sha256": str(metadata["sha256"]),
                "record_count": int(metadata["record_count"]),
            }
            for metadata in _iter_verified_geometry_sidecars(
                building_dir,
                verify_checksums=True,
                _reconcile=False,
            )
        }
        if observed_shards != expected_shards:
            raise ArtifactIntegrityError(
                "copied geometry shards differ from pinned manifest"
            )
        geometry_run_state = json.loads(
            (building_dir / "run_state.json").read_text(encoding="utf-8")
        )
        geometry_fingerprint_sha256 = geometry_run_state["fingerprint"][
            "sha256"
        ]
        if not _is_sha256(geometry_fingerprint_sha256):
            raise ArtifactIntegrityError(
                "copied geometry run fingerprint is invalid"
            )
        inventory = [
            _pinned_file_descriptor(path)
            for path in sorted(
                building_dir.iterdir(),
                key=lambda candidate: candidate.name,
            )
        ]
        _atomic_write_text(
            building_dir / GEOMETRY_SNAPSHOT_STATE_FILENAME,
            json.dumps(
                {
                    "schema": GEOMETRY_SNAPSHOT_SCHEMA,
                    "density_run_fingerprint_sha256": fingerprint["sha256"],
                    "geometry_run_fingerprint_sha256": (
                        geometry_fingerprint_sha256
                    ),
                    "record_count": expected_records,
                    "inventory": inventory,
                },
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        (
            validated_shards,
            validated_records,
            _validated_geometry_fingerprint,
        ) = _validate_geometry_snapshot(building_dir, fingerprint)
        _fsync_directory(building_dir)
        os.replace(building_dir, snapshot_dir)
        _fsync_directory(destination_root)
        return snapshot_dir, validated_shards, validated_records
    except BaseException as exc:
        cleanup_error: Optional[BaseException] = None
        if building_dir.exists() or building_dir.is_symlink():
            try:
                _cleanup_geometry_snapshot_transients(destination_root)
            except BaseException as cleanup_exc:
                cleanup_error = cleanup_exc
        if cleanup_error is not None:
            raise ArtifactIntegrityError(
                "geometry snapshot preparation and private staging cleanup "
                "both failed"
            ) from cleanup_error
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, ArtifactIntegrityError):
            raise
        raise ArtifactIntegrityError(
            "failed to create immutable geometry snapshot"
        ) from exc


def _invalidate_statistics_marker(output_dir: Path) -> None:
    commit_path = output_dir / STATISTICS_COMMIT_FILENAME
    if commit_path.exists() or commit_path.is_symlink():
        raise ArtifactIntegrityError(
            "refusing to invalidate statistics with a permanent commit"
        )
    removed = False
    for path in (
        output_dir / "statistics.json",
        output_dir / STATISTICS_READY_FILENAME,
    ):
        if path.exists():
            path.unlink()
            removed = True
    for pattern in (
        ".statistics.json.*.tmp",
        "..statistics.ready.json.*.tmp",
        "..statistics.commit.json.*.tmp",
    ):
        for path in output_dir.glob(pattern):
            path.unlink()
            removed = True
    if removed:
        _fsync_directory(output_dir)


def _remove_geometry_snapshot(snapshot_dir: Path) -> None:
    output_dir = snapshot_dir.parent
    expected = output_dir / GEOMETRY_SNAPSHOT_DIRNAME
    if (
        snapshot_dir != expected
        or snapshot_dir.is_symlink()
        or not snapshot_dir.is_dir()
    ):
        _write_geometry_snapshot_poison(
            output_dir,
            f"unexpected snapshot cleanup path: {snapshot_dir}",
        )
        raise ArtifactIntegrityError(
            f"refusing to remove unexpected snapshot path: {snapshot_dir}"
        )
    tombstone: Optional[Path] = None
    for _attempt in range(16):
        candidate = output_dir / (
            f"{GEOMETRY_SNAPSHOT_DELETE_PREFIX}{uuid.uuid4().hex}"
        )
        if not candidate.exists() and not candidate.is_symlink():
            tombstone = candidate
            break
    if tombstone is None:
        raise ArtifactIntegrityError(
            "cannot allocate a private snapshot deletion tombstone"
        )
    try:
        os.replace(snapshot_dir, tombstone)
        _fsync_directory(output_dir)
    except OSError as exc:
        raise ArtifactIntegrityError(
            "geometry snapshot could not be atomically retired"
        ) from exc
    try:
        shutil.rmtree(tombstone)
        _fsync_directory(output_dir)
    except OSError as exc:
        raise ArtifactIntegrityError(
            "retired geometry snapshot cleanup did not complete; resume "
            "will converge the deletion tombstone"
        ) from exc


def _assert_no_private_geometry_residue(output_dir: Path) -> None:
    residue = sorted(
        path.name
        for path in output_dir.iterdir()
        if (
            path.name.startswith(".geometry_snapshot")
            or path.name.startswith("..geometry_snapshot")
        )
    )
    if residue:
        raise ArtifactIntegrityError(
            f"private geometry snapshot residue remains: {residue}"
        )


def _load_geometry_publication_contract(
    geometry_dir: Path,
    *,
    verify_checksums: bool,
) -> tuple[dict[str, dict[str, Any]], int]:
    manifest_path = geometry_dir / "manifest.json"
    index_metadata_path = geometry_dir / "geometry_index.json"
    run_state_path = geometry_dir / "run_state.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_metadata = json.loads(
            index_metadata_path.read_text(encoding="utf-8")
        )
        run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"invalid finalized geometry metadata in {geometry_dir}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != GEOMETRY_SCHEMA:
        raise ArtifactIntegrityError("invalid geometry manifest schema")
    if (
        not isinstance(run_state, dict)
        or set(run_state) != {"schema", "fingerprint"}
        or run_state.get("schema") != GEOMETRY_RUN_STATE_SCHEMA
        or manifest.get("run_fingerprint") != run_state.get("fingerprint")
    ):
        raise ArtifactIntegrityError(
            "geometry manifest/run-state fingerprint contract mismatch"
        )
    fingerprint = run_state["fingerprint"]
    if (
        not isinstance(fingerprint, dict)
        or set(fingerprint) != {"sha256", "inputs", "parameters"}
        or not isinstance(fingerprint.get("inputs"), dict)
        or set(fingerprint["inputs"]) != GEOMETRY_FINGERPRINT_INPUT_FIELDS
    ):
        raise ArtifactIntegrityError("invalid geometry run fingerprint")
    for name, descriptor in fingerprint["inputs"].items():
        if descriptor is None and name != "input":
            continue
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"name", "size", "sha256"}
            or not isinstance(descriptor.get("name"), str)
            or not descriptor["name"]
            or Path(descriptor["name"]).name != descriptor["name"]
            or not isinstance(descriptor.get("size"), int)
            or isinstance(descriptor["size"], bool)
            or descriptor["size"] < 0
            or not _is_sha256(descriptor.get("sha256"))
        ):
            raise ArtifactIntegrityError(
                f"invalid geometry fingerprint input: {name}"
            )
    parameters = fingerprint.get("parameters")
    if (
        not isinstance(parameters, dict)
        or set(parameters) != GEOMETRY_FINGERPRINT_PARAMETER_FIELDS
        or parameters.get("schema") != GEOMETRY_SCHEMA
    ):
        raise ArtifactIntegrityError(
            "invalid geometry run fingerprint parameters"
        )
    optional_string_fields = (
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
            for field in optional_string_fields
        )
    ):
        raise ArtifactIntegrityError(
            "invalid geometry fingerprint column parameters"
        )
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
        raise ArtifactIntegrityError(
            "invalid geometry fingerprint integer parameters"
        )
    prune_rms_thresh = parameters.get("prune_rms_thresh")
    if (
        isinstance(prune_rms_thresh, bool)
        or not isinstance(prune_rms_thresh, (int, float))
        or not math.isfinite(float(prune_rms_thresh))
        or float(prune_rms_thresh) < 0.0
        or not isinstance(parameters.get("seed"), int)
        or isinstance(parameters["seed"], bool)
        or not isinstance(parameters.get("optimize"), bool)
        or not isinstance(parameters.get("verify_checksums"), bool)
    ):
        raise ArtifactIntegrityError(
            "invalid geometry fingerprint generation parameters"
        )
    fingerprint_payload = {
        "inputs": fingerprint["inputs"],
        "parameters": fingerprint["parameters"],
    }
    serialized_fingerprint = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_fingerprint_sha256 = hashlib.sha256(
        serialized_fingerprint.encode("utf-8")
    ).hexdigest()
    if (
        not _is_sha256(fingerprint.get("sha256"))
        or fingerprint["sha256"] != expected_fingerprint_sha256
    ):
        raise ArtifactIntegrityError(
            "geometry run fingerprint checksum mismatch"
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
        or index_metadata.get("sorted_by")
        != ["source_index", "row_index"]
        or index_metadata.get("lookup")
        != "numpy.searchsorted(source_index, requested_source_index)"
    ):
        raise ArtifactIntegrityError("invalid geometry index metadata")
    try:
        index_record_count = _identity_integer(
            index_metadata["record_count"],
            field="geometry_index.record_count",
        )
    except (KeyError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "invalid geometry index record_count"
        ) from exc
    if index_record_count is None or index_record_count < 0:
        raise ArtifactIntegrityError("invalid geometry index record_count")
    index_path = geometry_dir / str(index_metadata["filename"])
    if not index_path.is_file():
        raise ArtifactIntegrityError(f"geometry index missing: {index_path}")
    index_sha256 = index_metadata.get("sha256")
    if (
        not _is_sha256(index_sha256)
        or (
            verify_checksums
            and _sha256_file(index_path) != index_sha256
        )
    ):
        raise ArtifactIntegrityError("geometry index checksum mismatch")

    source_contract = manifest.get("source_index")
    if (
        not isinstance(source_contract, dict)
        or source_contract.get("artifact") != index_path.name
        or source_contract.get("metadata") != index_metadata_path.name
        or source_contract.get("sha256") != index_sha256
        or source_contract.get("record_count") != index_record_count
    ):
        raise ArtifactIntegrityError(
            "geometry manifest source-index contract mismatch"
        )
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list):
        raise ArtifactIntegrityError("geometry manifest shards must be a list")
    expected_shards: dict[str, dict[str, Any]] = {}
    record_count_sum = 0
    shard_ids: set[int] = set()
    for raw in raw_shards:
        if not isinstance(raw, dict):
            raise ArtifactIntegrityError(
                "geometry manifest shard entries must be objects"
            )
        try:
            shard_id = _identity_integer(
                raw["shard_id"],
                field="geometry_manifest.shard_id",
            )
            record_count = _identity_integer(
                raw["record_count"],
                field="geometry_manifest.record_count",
            )
        except (KeyError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "invalid geometry manifest shard identity"
            ) from exc
        filename = raw.get("filename")
        checksum = raw.get("sha256")
        if (
            shard_id is None
            or shard_id < 0
            or record_count is None
            or record_count <= 0
            or not isinstance(filename, str)
            or filename != f"shard_{shard_id:06d}.npz"
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(
                character not in "0123456789abcdef"
                for character in checksum
            )
            or filename in expected_shards
            or shard_id in shard_ids
        ):
            raise ArtifactIntegrityError(
                "invalid or duplicate geometry manifest shard"
            )
        expected_shards[filename] = {
            "shard_id": shard_id,
            "filename": filename,
            "sha256": checksum,
            "record_count": record_count,
        }
        shard_ids.add(shard_id)
        record_count_sum += record_count
    try:
        successful_records = _identity_integer(
            manifest["successful_records"],
            field="geometry_manifest.successful_records",
        )
    except (KeyError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "invalid geometry successful_records"
        ) from exc
    if (
        successful_records is None
        or successful_records < 0
        or successful_records != record_count_sum
        or successful_records != index_record_count
    ):
        raise ArtifactIntegrityError(
            "geometry manifest/index record counts disagree"
        )
    return expected_shards, successful_records


def iter_geometry_records(
    geometry_dir: os.PathLike[str] | str,
    verify_checksums: bool = True,
    completed_rows: Optional[_CompletedRowIndex] = None,
    expected_shards: Optional[dict[str, dict[str, Any]]] = None,
    *,
    reconcile: bool = True,
) -> Iterator[
    tuple[
        int,
        str,
        int,
        Optional[int],
        str,
        str,
        str,
        str,
        GeometryRecord,
    ]
]:
    """Read the geometry product contract one shard at a time."""
    directory = Path(geometry_dir)
    found_sidecar = False
    unseen_shards = (
        None if expected_shards is None else set(expected_shards)
    )
    for metadata in _iter_verified_geometry_sidecars(
        directory,
        verify_checksums=verify_checksums,
        _reconcile=reconcile,
    ):
        found_sidecar = True
        if expected_shards is not None:
            filename = str(metadata.get("filename", ""))
            expected = expected_shards.get(filename)
            actual = {
                "shard_id": int(metadata["shard_id"]),
                "filename": filename,
                "sha256": str(metadata["sha256"]),
                "record_count": int(metadata["record_count"]),
            }
            if expected is None or actual != expected:
                raise ArtifactIntegrityError(
                    f"geometry shard differs from manifest: {filename}"
                )
            if unseen_shards is not None:
                unseen_shards.discard(filename)
        identities: list[tuple[int, int, Optional[int], str, dict[str, Any]]] = []
        for entry in metadata["records"]:
            try:
                row_index = _identity_integer(
                    entry["row_index"],
                    field=f"{metadata['filename']}.row_index",
                )
                source_index = _identity_integer(
                    entry["source_index"],
                    field=f"{metadata['filename']}.source_index",
                )
                train_ordinal = _record_ordinal(
                    entry,
                    context=f"{metadata['filename']}:{entry['key']}",
                )
            except (KeyError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    f"invalid geometry identity in {metadata['filename']}"
                ) from exc
            smiles = entry.get("smiles")
            if (
                row_index is None
                or row_index < 0
                or source_index is None
                or source_index < 0
                or not isinstance(smiles, str)
                or not smiles
                or entry.get("smiles_hash") != smiles_hash(smiles)
            ):
                raise ArtifactIntegrityError(
                    f"invalid geometry identity in {metadata['filename']}"
                )
            identities.append(
                (
                    row_index,
                    source_index,
                    train_ordinal,
                    smiles,
                    entry,
                )
            )
        existing = (
            set()
            if completed_rows is None
            else completed_rows.existing(
                row_index
                for row_index, _, _, _, _ in identities
            )
        )
        shard_path = directory / metadata["filename"]
        try:
            with shard_path.open("rb") as shard_handle:
                before = os.fstat(shard_handle.fileno())
                actual_sha256 = _hash_open_file(shard_handle)
                after_hash = os.fstat(shard_handle.fileno())
                path_after_hash = shard_path.stat()
                if (
                    _file_identity(before) != _file_identity(after_hash)
                    or _file_identity(after_hash)
                    != _file_identity(path_after_hash)
                    or actual_sha256 != str(metadata["sha256"])
                ):
                    raise ArtifactIntegrityError(
                        "geometry snapshot shard changed before consumption: "
                        f"{shard_path}"
                    )
                shard_handle.seek(0)
                with np.load(shard_handle, allow_pickle=False) as arrays:
                    for chunk_start in range(0, len(identities), 256):
                        loaded_records: list[
                            tuple[
                                int,
                                str,
                                int,
                                Optional[int],
                                str,
                                str,
                                str,
                                str,
                                GeometryRecord,
                            ]
                        ] = []
                        geometry_rows: list[
                            tuple[
                                int,
                                int,
                                Optional[int],
                                str,
                                str,
                                str,
                                str,
                                str,
                            ]
                        ] = []
                        for (
                            row_index,
                            source_index,
                            train_ordinal,
                            smiles,
                            entry,
                        ) in identities[chunk_start : chunk_start + 256]:
                            geometry_key = str(entry["key"])
                            prefix = f"{geometry_key}__"
                            record_arrays = {
                                field: np.asarray(arrays[f"{prefix}{field}"])
                                for field in (
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
                                )
                            }
                            record = GeometryRecord.from_storage_dict(
                                record_arrays,
                            )
                            payload_sha256 = _geometry_payload_sha256(record)
                            provenance = (
                                str(metadata["filename"]),
                                str(metadata["sha256"]),
                                geometry_key,
                                payload_sha256,
                            )
                            geometry_rows.append(
                                (
                                    row_index,
                                    source_index,
                                    train_ordinal,
                                    smiles_hash(smiles),
                                    *provenance,
                                )
                            )
                            if row_index not in existing:
                                loaded_records.append(
                                    (
                                        row_index,
                                        smiles,
                                        source_index,
                                        train_ordinal,
                                        *provenance,
                                        record,
                                    )
                                )
                        if completed_rows is not None:
                            completed_rows.register_geometry_rows(
                                geometry_rows
                            )
                        yield from loaded_records
                after_load = os.fstat(shard_handle.fileno())
                path_after_load = shard_path.stat()
                if (
                    _file_identity(after_hash) != _file_identity(after_load)
                    or _file_identity(after_load)
                    != _file_identity(path_after_load)
                ):
                    raise ArtifactIntegrityError(
                        "geometry snapshot shard changed during consumption: "
                        f"{shard_path}"
                    )
        except ArtifactIntegrityError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ArtifactIntegrityError(
                f"cannot consume geometry snapshot shard: {shard_path}"
            ) from exc
    if not found_sidecar and expected_shards is None:
        raise FileNotFoundError(f"no geometry shard sidecars found in {directory}")
    if unseen_shards:
        raise ArtifactIntegrityError(
            "geometry manifest references missing shards: "
            f"{sorted(unseen_shards)[:10]}"
        )
    if completed_rows is not None:
        completed_rows.validate_completed_coverage()


def _iter_private_geometry_snapshot_records(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    completed_rows: Optional[_CompletedRowIndex],
    expected_shards: dict[str, dict[str, Any]],
) -> Iterator[
    tuple[
        int,
        str,
        int,
        Optional[int],
        str,
        str,
        str,
        str,
        GeometryRecord,
    ]
]:
    try:
        yield from iter_geometry_records(
            snapshot_dir,
            verify_checksums=True,
            completed_rows=completed_rows,
            expected_shards=expected_shards,
            reconcile=False,
        )
    except ArtifactIntegrityError as exc:
        if _has_density_derivatives(output_dir):
            try:
                _write_geometry_snapshot_poison(output_dir, str(exc))
            except BaseException as poison_exc:
                raise ArtifactIntegrityError(
                    "private geometry consumption failed after density "
                    "derivation and the poison marker could not be "
                    f"persisted; original_error={exc!r}"
                ) from poison_exc
        raise


def _revalidate_geometry_snapshot_or_poison(
    snapshot_dir: Path,
    output_dir: Path,
    fingerprint: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], int, str]:
    try:
        return _validate_geometry_snapshot(snapshot_dir, fingerprint)
    except Exception as exc:
        if _has_density_derivatives(output_dir):
            try:
                _write_geometry_snapshot_poison(output_dir, str(exc))
            except BaseException as poison_exc:
                raise ArtifactIntegrityError(
                    "private geometry revalidation failed after density "
                    "derivation and the poison marker could not be "
                    f"persisted; original_error={exc!r}"
                ) from poison_exc
        if isinstance(exc, ArtifactIntegrityError):
            raise
        raise ArtifactIntegrityError(
            "private geometry snapshot could not be revalidated"
        ) from exc


def _density_task(
    task: tuple[
        int,
        str,
        int,
        Optional[int],
        str,
        str,
        str,
        str,
        GeometryRecord,
        dict[str, Any],
    ],
) -> tuple[
    int,
    str,
    int,
    Optional[int],
    str,
    str,
    str,
    str,
    Optional[DensityGridResult],
    Optional[dict[str, Any]],
]:
    (
        row_index,
        smiles,
        source_index,
        train_ordinal,
        geometry_artifact,
        geometry_artifact_sha256,
        geometry_key,
        geometry_payload_sha256,
        geometry,
        config,
    ) = task
    try:
        result = build_promolecular_density(
            geometry.atomic_numbers,
            geometry.coords,
            grid_size=config["grid_size"],
            spacing=config["spacing"],
            box_padding=config["box_padding"],
            atomic_sigma=config["atomic_sigma"],
            conformer_index=config["conformer_index"],
            conformer_mask=geometry.conformer_mask,
            strict=config["strict"],
            discrete_normalize=config["discrete_normalize"],
        )
        return (
            row_index,
            smiles,
            source_index,
            train_ordinal,
            geometry_artifact,
            geometry_artifact_sha256,
            geometry_key,
            geometry_payload_sha256,
            result,
            None,
        )
    except (RuntimeError, ValueError) as exc:
        return (
            row_index,
            smiles,
            source_index,
            train_ordinal,
            geometry_artifact,
            geometry_artifact_sha256,
            geometry_key,
            geometry_payload_sha256,
            None,
            {
                "row_index": row_index,
                "source_index": source_index,
                "sdf_ordinal": train_ordinal,
                "train_ordinal": train_ordinal,
                "smiles": smiles,
                "smiles_hash": smiles_hash(smiles),
                "geometry_artifact": geometry_artifact,
                "geometry_artifact_sha256": geometry_artifact_sha256,
                "geometry_key": geometry_key,
                "geometry_payload_sha256": geometry_payload_sha256,
                "stage": "promolecular_density",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )


def _next_shard_id(output_dir: Path) -> int:
    ids = [
        int(path.stem.split("_")[-1])
        for path in output_dir.glob("density_*.json")
        if path.stem.split("_")[-1].isdigit()
    ]
    return max(ids, default=-1) + 1


class _FailureJournal:
    """Stream failures and publish the complete current-run log atomically."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.target = output_dir / "failures.jsonl"
        self._handle: Any = None
        self._temporary: Optional[Path] = None
        self.count = 0

    def __enter__(self) -> "_FailureJournal":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        removed_temporary = False
        for temporary in self.output_dir.glob(".failures.jsonl.*.tmp"):
            temporary.unlink()
            removed_temporary = True
        if removed_temporary:
            _fsync_directory(self.output_dir)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=self.output_dir,
            prefix=".failures.jsonl.",
            suffix=".tmp",
            delete=False,
        )
        self._handle = handle
        self._temporary = Path(handle.name)
        return self

    def record(self, failure: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("failure journal is not open")
        self._handle.write(
            json.dumps(
                failure,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        self.count += 1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is None or self._temporary is None:
            return
        temporary = self._temporary
        try:
            if exc_type is None:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            self._handle.close()
            if exc_type is None:
                os.replace(temporary, self.target)
                _fsync_directory(self.output_dir)
            elif temporary.exists():
                temporary.unlink()
                _fsync_directory(self.output_dir)
        finally:
            self._handle = None
            self._temporary = None


def _strict_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_verified_failure_journal(
    output_dir: Path,
    *,
    completed_index: Optional[_CompletedRowIndex] = None,
) -> dict[str, Any]:
    failure_path = output_dir / "failures.jsonl"
    digest = hashlib.sha256()
    record_count = 0
    pending: list[
        tuple[int, int, Optional[int], str, str, str, str, str]
    ] = []
    try:
        with failure_path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.endswith(b"\n") or raw_line == b"\n":
                    raise ArtifactIntegrityError(
                        "failure journal must contain one non-empty canonical "
                        f"JSON record per newline: line {line_number}"
                    )
                try:
                    failure = json.loads(
                        raw_line.decode("utf-8"),
                        object_pairs_hook=_strict_json_object,
                    )
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    raise ArtifactIntegrityError(
                        f"invalid failure journal JSON at line {line_number}"
                    ) from exc
                if (
                    not isinstance(failure, dict)
                    or set(failure) != FAILURE_RECORD_FIELDS
                ):
                    raise ArtifactIntegrityError(
                        "failure journal fields differ from schema at "
                        f"line {line_number}"
                    )
                try:
                    canonical_line = (
                        json.dumps(
                            failure,
                            allow_nan=False,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                        + b"\n"
                    )
                except (TypeError, ValueError) as exc:
                    raise ArtifactIntegrityError(
                        "failure journal contains a non-canonical value at "
                        f"line {line_number}"
                    ) from exc
                if raw_line != canonical_line:
                    raise ArtifactIntegrityError(
                        "failure journal record is not canonical JSON at "
                        f"line {line_number}"
                    )
                try:
                    row_index = _identity_integer(
                        failure["row_index"],
                        field=f"failures.jsonl:{line_number}.row_index",
                    )
                    source_index = _identity_integer(
                        failure["source_index"],
                        field=f"failures.jsonl:{line_number}.source_index",
                    )
                    train_ordinal = _record_ordinal(
                        failure,
                        context=f"failures.jsonl:{line_number}",
                    )
                except (KeyError, ValueError) as exc:
                    raise ArtifactIntegrityError(
                        "invalid failure identity at "
                        f"line {line_number}"
                    ) from exc
                if (
                    row_index is None
                    or row_index < 0
                    or source_index is None
                    or source_index < 0
                ):
                    raise ArtifactIntegrityError(
                        "failure row/source identity must be non-negative at "
                        f"line {line_number}"
                    )
                failure_smiles = failure["smiles"]
                if (
                    not isinstance(failure_smiles, str)
                    or not failure_smiles
                    or failure.get("smiles_hash")
                    != smiles_hash(failure_smiles)
                ):
                    raise ArtifactIntegrityError(
                        "failure SMILES identity mismatch at "
                        f"line {line_number}"
                    )
                provenance = _geometry_provenance(
                    failure["geometry_artifact"],
                    failure["geometry_artifact_sha256"],
                    failure["geometry_key"],
                    failure["geometry_payload_sha256"],
                    context=f"failures.jsonl:{line_number}",
                )
                if (
                    failure.get("stage") != "promolecular_density"
                    or not isinstance(failure.get("error_type"), str)
                    or not failure["error_type"]
                    or not isinstance(failure.get("message"), str)
                ):
                    raise ArtifactIntegrityError(
                        "failure diagnostic contract mismatch at "
                        f"line {line_number}"
                    )
                if completed_index is not None:
                    pending.append(
                        (
                            row_index,
                            source_index,
                            train_ordinal,
                            str(failure["smiles_hash"]),
                            *provenance,
                        )
                    )
                record_count += 1
                if completed_index is not None and len(pending) >= 512:
                    completed_index.register_failure_rows(pending)
                    pending.clear()
            after = os.fstat(handle.fileno())
        path_after = failure_path.stat()
    except ArtifactIntegrityError:
        raise
    except OSError as exc:
        raise ArtifactIntegrityError(
            f"cannot read failure journal: {failure_path}"
        ) from exc
    if (
        _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(path_after)
    ):
        raise ArtifactIntegrityError(
            "failure journal changed while it was audited"
        )
    if completed_index is not None and pending:
        completed_index.register_failure_rows(pending)
    return {
        "filename": failure_path.name,
        "size": int(after.st_size),
        "sha256": digest.hexdigest(),
        "record_count": record_count,
    }


def _consume_results(
    results: Iterable[
        tuple[
            int,
            str,
            int,
            Optional[int],
            str,
            str,
            str,
            str,
            Optional[DensityGridResult],
            Optional[dict[str, Any]],
        ]
    ],
    output_dir: Path,
    shard_size: int,
    first_shard_id: int,
    failure_journal: _FailureJournal,
) -> tuple[int, int]:
    pending: list[
        tuple[
            int,
            str,
            int,
            Optional[int],
            str,
            str,
            str,
            str,
            DensityGridResult,
        ]
    ] = []
    shard_id = first_shard_id
    successful = 0
    for (
        row_index,
        smiles,
        source_index,
        train_ordinal,
        geometry_artifact,
        geometry_artifact_sha256,
        geometry_key,
        geometry_payload_sha256,
        result,
        failure,
    ) in results:
        if failure is not None:
            failure_journal.record(failure)
            continue
        if result is None:
            raise RuntimeError("worker returned neither a density nor an error")
        pending.append(
            (
                row_index,
                smiles,
                source_index,
                train_ordinal,
                geometry_artifact,
                geometry_artifact_sha256,
                geometry_key,
                geometry_payload_sha256,
                result,
            )
        )
        if len(pending) >= shard_size:
            write_density_shard(pending, output_dir, shard_id)
            successful += len(pending)
            shard_id += 1
            pending = []
    if pending:
        write_density_shard(pending, output_dir, shard_id)
        successful += len(pending)
    return successful, failure_journal.count


def _load_yaml_config(
    path: Optional[str],
    *,
    pinned_bytes: Optional[bytes] = None,
) -> dict[str, Any]:
    if path is None:
        if pinned_bytes is not None:
            raise DensityConfigError(
                "configuration bytes were supplied without a path"
            )
        return {}
    if pinned_bytes is None:
        pinned_bytes, _descriptor = _read_pinned_bytes(Path(path))
    try:
        config_text = pinned_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DensityConfigError("YAML configuration must be UTF-8") from exc
    loaded = yaml.safe_load(config_text)
    if not isinstance(loaded, dict):
        raise DensityConfigError("YAML root must be a mapping")
    allowed_top = {"grid", "density", "storage", "encoder"}
    unknown_top = set(loaded) - allowed_top
    if unknown_top:
        raise DensityConfigError(
            f"unknown YAML sections: {sorted(unknown_top)}"
        )
    if "grid" not in loaded or "density" not in loaded:
        raise DensityConfigError("YAML must contain grid and density sections")
    grid = loaded["grid"]
    density = loaded["density"]
    if not isinstance(grid, dict) or not isinstance(density, dict):
        raise DensityConfigError("grid and density sections must be mappings")
    allowed_grid = {"size", "spacing", "resolution", "box_padding"}
    allowed_density = {
        "method",
        "atomic_sigma",
        "normalize",
    }
    if set(grid) - allowed_grid:
        raise DensityConfigError(
            f"unknown grid keys: {sorted(set(grid) - allowed_grid)}"
        )
    if set(density) - allowed_density:
        raise DensityConfigError(
            f"unknown density keys: {sorted(set(density) - allowed_density)}"
        )
    required_grid = {"size", "box_padding"}
    if not required_grid.issubset(grid) or not (
        "spacing" in grid or "resolution" in grid
    ):
        raise DensityConfigError(
            "grid requires size, box_padding, and spacing or resolution"
        )
    if "spacing" in grid and "resolution" in grid and (
        grid["spacing"] != grid["resolution"]
    ):
        raise DensityConfigError("grid spacing and resolution conflict")
    required_density = {"method", "normalize"}
    if not required_density.issubset(density):
        raise DensityConfigError("density requires method and normalize")
    if density["method"] not in {"gaussian_promol", "promolecular_gaussian"}:
        raise DensityConfigError(
            "density.method must identify the Gaussian promolecular approximation"
        )
    if not isinstance(density["normalize"], bool):
        raise DensityConfigError("density.normalize must be boolean")
    storage = loaded.get("storage")
    if storage is not None:
        if not isinstance(storage, dict):
            raise DensityConfigError("storage section must be a mapping")
        unknown_storage = set(storage) - {"format", "dtype", "compress"}
        if unknown_storage:
            raise DensityConfigError(
                f"unknown storage keys: {sorted(unknown_storage)}"
            )
        if storage.get("format", "npz") != "npz":
            raise DensityConfigError("storage.format must be npz")
        if storage.get("dtype", "float32") != "float32":
            raise DensityConfigError("storage.dtype must be float32")
        if "compress" in storage and not isinstance(storage["compress"], bool):
            raise DensityConfigError("storage.compress must be boolean")
        if storage.get("compress", True) is not True:
            raise DensityConfigError(
                "storage.compress must be true because density shards use "
                "compressed NPZ"
            )
    encoder = loaded.get("encoder")
    if encoder is not None:
        if not isinstance(encoder, dict):
            raise DensityConfigError("encoder section must be a mapping")
        unknown_encoder = set(encoder) - {"in_channels", "grid_size"}
        if unknown_encoder:
            raise DensityConfigError(
                f"unknown encoder keys: {sorted(unknown_encoder)}"
            )
        if encoder.get("in_channels", 1) != 1:
            raise DensityConfigError("encoder.in_channels must equal 1")
        if "grid_size" in encoder and encoder["grid_size"] != grid["size"]:
            raise DensityConfigError(
                "encoder.grid_size must equal grid.size"
            )

    return {
        "grid_size": grid.get("size"),
        "spacing": grid.get("spacing", grid.get("resolution")),
        "box_padding": grid.get("box_padding"),
        "atomic_sigma": density.get("atomic_sigma"),
        "discrete_normalize": density.get("normalize"),
    }


def _resolved_config(
    args: argparse.Namespace,
    *,
    pinned_config_bytes: Optional[bytes] = None,
) -> dict[str, Any]:
    file_config = _load_yaml_config(
        args.config,
        pinned_bytes=pinned_config_bytes,
    )

    def choose(name: str, default: Any) -> Any:
        command_line = getattr(args, name)
        if command_line is not None:
            return command_line
        file_value = file_config.get(name)
        return default if file_value is None else file_value

    sigma_value = choose("atomic_sigma", None)
    if isinstance(sigma_value, str):
        if sigma_value.lower() == "element":
            sigma = None
        else:
            try:
                sigma = float(sigma_value)
            except ValueError as exc:
                raise DensityConfigError(
                    "atomic_sigma must be 'element' or a positive number"
                ) from exc
    else:
        sigma = sigma_value
    conformer_index = (
        None
        if args.conformer_index.lower() == "mean"
        else int(args.conformer_index)
    )
    config: dict[str, object] = {
        "grid_size": choose("grid_size", 32),
        "spacing": choose("spacing", 0.75),
        "box_padding": choose("box_padding", 4.0),
        "atomic_sigma": sigma,
        "conformer_index": conformer_index,
        "strict": not args.allow_overflow,
        "discrete_normalize": choose("discrete_normalize", True),
    }
    return validate_density_config(config)


def _collect_run_metadata(
    output_dir: Path,
    failure_descriptor: dict[str, Any],
    config: dict[str, Any],
    *,
    run_fingerprint_sha256: str,
    geometry_run_fingerprint_sha256: str,
    expected_successful_records: Optional[int] = None,
    expected_geometry_records: Optional[int] = None,
    expected_shards: Optional[list[dict[str, Any]]] = None,
    validate_payloads: bool = False,
) -> dict[str, Any]:
    if (
        not _is_sha256(run_fingerprint_sha256)
        or not _is_sha256(geometry_run_fingerprint_sha256)
    ):
        raise ArtifactIntegrityError(
            "statistics fingerprint provenance is invalid"
        )
    observed_failure_descriptor = _read_verified_failure_journal(output_dir)
    if _canonical_json(observed_failure_descriptor) != _canonical_json(
        failure_descriptor
    ):
        raise ArtifactIntegrityError(
            "failure journal changed after the final provenance audit"
        )
    failure_count = int(failure_descriptor["record_count"])
    expected_normalization_requested = (
        "discrete_electron_count"
        if bool(config["discrete_normalize"])
        else "continuous_gaussian"
    )
    successful_records = 0
    overflow_records = 0
    integration_error_sum = 0.0
    integration_error_max: Optional[float] = None
    normalization_requested_counts: dict[str, int] = {}
    normalization_applied_counts: dict[str, int] = {}
    shards: list[dict[str, Any]] = []
    for sidecar in _iter_verified_density_sidecars(
        output_dir,
        verify_checksums=True,
        validate_payloads=validate_payloads,
        reconcile=False,
    ):
        shards.append(
            {
                "shard_id": int(sidecar["shard_id"]),
                "filename": str(sidecar["filename"]),
                "sha256": str(sidecar["sha256"]),
                "record_count": int(sidecar["record_count"]),
            }
        )
        for entry in sidecar["records"]:
            error = abs(float(entry["integration_error"]))
            if not np.isfinite(error):
                raise ArtifactIntegrityError(
                    "density integration_error must be finite"
                )
            successful_records += 1
            overflow_records += int(bool(entry["overflow"]))
            requested = str(entry["normalization_requested"])
            applied = str(entry["normalization_applied"])
            if requested != expected_normalization_requested:
                raise ArtifactIntegrityError(
                    "density normalization request differs from run config"
                )
            normalization_requested_counts[requested] = (
                normalization_requested_counts.get(requested, 0) + 1
            )
            normalization_applied_counts[applied] = (
                normalization_applied_counts.get(applied, 0) + 1
            )
            integration_error_sum += error
            integration_error_max = (
                error
                if integration_error_max is None
                else max(integration_error_max, error)
            )
    if (
        expected_shards is not None
        and _canonical_json(shards) != _canonical_json(expected_shards)
    ):
        raise ArtifactIntegrityError(
            "density shard inventory changed after provenance audit"
        )
    if (
        expected_successful_records is not None
        and successful_records != expected_successful_records
    ):
        raise ArtifactIntegrityError(
            "density shard count changed after the final provenance audit"
        )
    if (
        expected_geometry_records is not None
        and successful_records + int(failure_count)
        != expected_geometry_records
    ):
        raise ArtifactIntegrityError(
            "final density success/failure accounting does not cover the "
            "geometry snapshot"
        )
    stats = {
        "schema": DENSITY_SCHEMA,
        "run_fingerprint_sha256": run_fingerprint_sha256,
        "geometry_run_fingerprint_sha256": (
            geometry_run_fingerprint_sha256
        ),
        "method": "promolecular_gaussian",
        "successful_records": successful_records,
        "failed_records": int(failure_count),
        "geometry_records": successful_records + int(failure_count),
        "overflow_records": overflow_records,
        "normalization_requested": normalization_requested_counts,
        "normalization_applied": normalization_applied_counts,
        "absolute_integration_error": {
            "mean": (
                None
                if successful_records == 0
                else integration_error_sum / successful_records
            ),
            "max": integration_error_max,
        },
        "config": config,
        "shards": shards,
        "failures": failure_descriptor,
    }
    return stats


def _write_run_metadata(
    output_dir: Path,
    failure_descriptor: dict[str, Any],
    config: dict[str, Any],
    *,
    run_fingerprint_sha256: str,
    geometry_run_fingerprint_sha256: str,
    expected_successful_records: Optional[int] = None,
    expected_geometry_records: Optional[int] = None,
    expected_shards: Optional[list[dict[str, Any]]] = None,
    validate_payloads: bool = False,
    statistics_path: Optional[Path] = None,
) -> dict[str, Any]:
    stats = _collect_run_metadata(
        output_dir,
        failure_descriptor,
        config,
        run_fingerprint_sha256=run_fingerprint_sha256,
        geometry_run_fingerprint_sha256=(
            geometry_run_fingerprint_sha256
        ),
        expected_successful_records=expected_successful_records,
        expected_geometry_records=expected_geometry_records,
        expected_shards=expected_shards,
        validate_payloads=validate_payloads,
    )
    _atomic_write_text(
        (
            output_dir / "statistics.json"
            if statistics_path is None
            else statistics_path
        ),
        json.dumps(
            stats,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return stats


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
        raise ArtifactIntegrityError(
            "publication metadata is not canonical JSON"
        ) from exc


def _load_verified_density_run_state(output_dir: Path) -> dict[str, Any]:
    state_path = output_dir / "run_state.json"
    content, _descriptor = _read_pinned_bytes(state_path)
    try:
        state = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"invalid density run state: {state_path}"
        ) from exc
    fingerprint = state.get("fingerprint") if isinstance(state, dict) else None
    if (
        not isinstance(state, dict)
        or set(state) != {"schema", "fingerprint"}
        or state.get("schema") != DENSITY_RUN_STATE_SCHEMA
        or not isinstance(fingerprint, dict)
        or set(fingerprint) != {"sha256", "inputs", "parameters"}
        or not _is_sha256(fingerprint.get("sha256"))
        or not isinstance(fingerprint.get("inputs"), dict)
        or set(fingerprint["inputs"]) != DENSITY_FINGERPRINT_INPUT_FIELDS
        or not isinstance(fingerprint.get("parameters"), dict)
    ):
        raise ArtifactIntegrityError(
            "density run-state structure is invalid"
        )
    for input_name, descriptor in fingerprint["inputs"].items():
        if input_name == "density_config" and descriptor is None:
            continue
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"name", "size", "sha256"}
            or not isinstance(descriptor.get("name"), str)
            or not descriptor["name"]
            or Path(descriptor["name"]).name != descriptor["name"]
            or not isinstance(descriptor.get("size"), int)
            or isinstance(descriptor["size"], bool)
            or descriptor["size"] < 0
            or not _is_sha256(descriptor.get("sha256"))
        ):
            raise ArtifactIntegrityError(
                f"invalid density run-state input: {input_name}"
            )
    for input_name, expected_name in (
        ("geometry_manifest", "manifest.json"),
        ("geometry_index", "geometry_index.json"),
        ("geometry_run_state", "run_state.json"),
    ):
        if fingerprint["inputs"][input_name]["name"] != expected_name:
            raise ArtifactIntegrityError(
                f"density run-state input name mismatch: {input_name}"
            )
    expected_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "inputs": fingerprint["inputs"],
                "parameters": fingerprint["parameters"],
            }
        ).encode("utf-8")
    ).hexdigest()
    if fingerprint["sha256"] != expected_sha256:
        raise ArtifactIntegrityError(
            "density run fingerprint is not self-consistent"
        )
    return fingerprint


def _verify_statistics_contract(
    statistics_path: Path,
    *,
    output_dir: Path,
    run_fingerprint_sha256: str,
    config: dict[str, Any],
    validate_payloads: bool,
    expected_shards: Optional[list[dict[str, Any]]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    content, file_descriptor = _read_pinned_bytes(statistics_path)
    try:
        statistics = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"invalid statistics publication: {statistics_path}"
        ) from exc
    expected_fields = {
        "schema",
        "run_fingerprint_sha256",
        "geometry_run_fingerprint_sha256",
        "method",
        "successful_records",
        "failed_records",
        "geometry_records",
        "overflow_records",
        "normalization_requested",
        "normalization_applied",
        "absolute_integration_error",
        "config",
        "shards",
        "failures",
    }
    if (
        not isinstance(statistics, dict)
        or set(statistics) != expected_fields
        or statistics.get("schema") != DENSITY_SCHEMA
        or statistics.get("run_fingerprint_sha256")
        != run_fingerprint_sha256
        or statistics.get("method") != "promolecular_gaussian"
        or not _is_sha256(
            statistics.get("geometry_run_fingerprint_sha256")
        )
        or any(
            isinstance(statistics.get(field), bool)
            or not isinstance(statistics.get(field), int)
            or statistics[field] < 0
            for field in (
                "successful_records",
                "failed_records",
                "geometry_records",
                "overflow_records",
            )
        )
        or _canonical_json(statistics.get("config"))
        != _canonical_json(config)
    ):
        raise ArtifactIntegrityError(
            "statistics publication contract is invalid"
        )
    failure_descriptor = statistics.get("failures")
    if (
        not isinstance(failure_descriptor, dict)
        or set(failure_descriptor)
        != {"filename", "size", "sha256", "record_count"}
        or failure_descriptor.get("filename") != "failures.jsonl"
        or not isinstance(failure_descriptor.get("size"), int)
        or isinstance(failure_descriptor["size"], bool)
        or failure_descriptor["size"] < 0
        or not _is_sha256(failure_descriptor.get("sha256"))
        or not isinstance(failure_descriptor.get("record_count"), int)
        or isinstance(failure_descriptor["record_count"], bool)
        or failure_descriptor["record_count"] < 0
    ):
        raise ArtifactIntegrityError(
            "statistics failure descriptor is invalid"
        )
    expected = _collect_run_metadata(
        output_dir,
        failure_descriptor,
        config,
        run_fingerprint_sha256=run_fingerprint_sha256,
        geometry_run_fingerprint_sha256=str(
            statistics["geometry_run_fingerprint_sha256"]
        ),
        expected_successful_records=int(
            statistics["successful_records"]
        ),
        expected_geometry_records=int(statistics["geometry_records"]),
        expected_shards=expected_shards,
        validate_payloads=validate_payloads,
    )
    if _canonical_json(statistics) != _canonical_json(expected):
        raise ArtifactIntegrityError(
            "statistics bytes differ from the verified density artifacts"
        )
    return statistics, {
        "size": int(file_descriptor["size"]),
        "sha256": str(file_descriptor["sha256"]),
    }


def _load_statistics_commit(
    output_dir: Path,
    *,
    run_fingerprint_sha256: str,
) -> dict[str, Any]:
    commit_path = output_dir / STATISTICS_COMMIT_FILENAME
    content, _descriptor = _read_pinned_bytes(commit_path)
    try:
        commit = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"invalid statistics commit: {commit_path}"
        ) from exc
    statistics_descriptor = (
        commit.get("statistics") if isinstance(commit, dict) else None
    )
    if (
        not isinstance(commit, dict)
        or set(commit)
        != {"schema", "run_fingerprint_sha256", "statistics"}
        or commit.get("schema") != STATISTICS_COMMIT_SCHEMA
        or commit.get("run_fingerprint_sha256")
        != run_fingerprint_sha256
        or not isinstance(statistics_descriptor, dict)
        or set(statistics_descriptor) != {"size", "sha256"}
        or not isinstance(statistics_descriptor.get("size"), int)
        or isinstance(statistics_descriptor["size"], bool)
        or statistics_descriptor["size"] < 0
        or not _is_sha256(statistics_descriptor.get("sha256"))
    ):
        raise ArtifactIntegrityError(
            "statistics commit contract is invalid"
        )
    return commit


def _cleanup_committed_geometry_residue(output_dir: Path) -> None:
    snapshot_dir = output_dir / GEOMETRY_SNAPSHOT_DIRNAME
    if snapshot_dir.exists() or snapshot_dir.is_symlink():
        _remove_geometry_snapshot(snapshot_dir)
    _cleanup_geometry_snapshot_transients(output_dir)
    _assert_no_private_geometry_residue(output_dir)


def _recover_committed_statistics(
    output_dir: Path,
    *,
    fingerprint_parameters: dict[str, Any],
    pinned_config_descriptor: Optional[dict[str, Any]],
    config: dict[str, Any],
) -> Optional[dict[str, Any]]:
    commit_path = output_dir / STATISTICS_COMMIT_FILENAME
    if not commit_path.exists() and not commit_path.is_symlink():
        return None
    if output_dir.joinpath(GEOMETRY_SNAPSHOT_POISON_FILENAME).exists():
        raise ArtifactIntegrityError(
            "cannot recover committed statistics from a poisoned snapshot"
        )
    fingerprint = _load_verified_density_run_state(output_dir)
    if (
        _canonical_json(fingerprint["parameters"])
        != _canonical_json(fingerprint_parameters)
        or _canonical_json(fingerprint["inputs"]["density_config"])
        != _canonical_json(pinned_config_descriptor)
    ):
        raise ArtifactIntegrityError(
            "committed statistics belong to different parameters or config"
        )
    commit = _load_statistics_commit(
        output_dir,
        run_fingerprint_sha256=str(fingerprint["sha256"]),
    )
    ready_path = output_dir / STATISTICS_READY_FILENAME
    published_path = output_dir / "statistics.json"
    ready_exists = ready_path.is_file() and not ready_path.is_symlink()
    published_exists = (
        published_path.is_file() and not published_path.is_symlink()
    )
    if ready_exists == published_exists:
        raise ArtifactIntegrityError(
            "committed statistics require exactly one ready or published file"
        )
    active_path = ready_path if ready_exists else published_path
    statistics, descriptor = _verify_statistics_contract(
        active_path,
        output_dir=output_dir,
        run_fingerprint_sha256=str(fingerprint["sha256"]),
        config=config,
        validate_payloads=False,
    )
    if int(statistics["failed_records"]) != 0:
        raise ArtifactIntegrityError(
            "a terminal statistics commit cannot contain failed records"
        )
    if _canonical_json(descriptor) != _canonical_json(commit["statistics"]):
        raise ArtifactIntegrityError(
            "committed statistics descriptor does not match its file"
        )

    _cleanup_committed_geometry_residue(output_dir)

    persisted_again = _load_verified_density_run_state(output_dir)
    commit_again = _load_statistics_commit(
        output_dir,
        run_fingerprint_sha256=str(fingerprint["sha256"]),
    )
    statistics_again, descriptor_again = _verify_statistics_contract(
        active_path,
        output_dir=output_dir,
        run_fingerprint_sha256=str(fingerprint["sha256"]),
        config=config,
        validate_payloads=False,
        expected_shards=list(statistics["shards"]),
    )
    if (
        _canonical_json(persisted_again) != _canonical_json(fingerprint)
        or _canonical_json(commit_again) != _canonical_json(commit)
        or _canonical_json(statistics_again) != _canonical_json(statistics)
        or _canonical_json(descriptor_again)
        != _canonical_json(commit["statistics"])
    ):
        raise ArtifactIntegrityError(
            "committed publication changed during crash recovery"
        )
    if ready_exists:
        os.replace(ready_path, published_path)
        _fsync_directory(output_dir)
        statistics, published_descriptor = _verify_statistics_contract(
            published_path,
            output_dir=output_dir,
            run_fingerprint_sha256=str(fingerprint["sha256"]),
            config=config,
            validate_payloads=False,
            expected_shards=list(statistics["shards"]),
        )
        if _canonical_json(published_descriptor) != _canonical_json(
            commit["statistics"]
        ):
            raise ArtifactIntegrityError(
                "published statistics differ from their permanent commit"
            )
    _assert_no_private_geometry_residue(output_dir)
    return statistics


def _publish_statistics(
    output_dir: Path,
    snapshot_dir: Path,
    *,
    failure_descriptor: dict[str, Any],
    config: dict[str, Any],
    fingerprint: dict[str, Any],
    geometry_run_fingerprint_sha256: str,
    expected_successful_records: int,
    expected_geometry_records: int,
    expected_shards: list[dict[str, Any]],
) -> dict[str, Any]:
    if int(failure_descriptor.get("record_count", -1)) != 0:
        raise ArtifactIntegrityError(
            "refusing to commit statistics while failures remain retryable"
        )
    ready_path = output_dir / STATISTICS_READY_FILENAME
    commit_path = output_dir / STATISTICS_COMMIT_FILENAME
    published_path = output_dir / "statistics.json"
    if any(
        path.exists() or path.is_symlink()
        for path in (ready_path, commit_path, published_path)
    ):
        raise ArtifactIntegrityError(
            "statistics publication targets must be absent before staging"
        )
    statistics = _write_run_metadata(
        output_dir,
        failure_descriptor,
        config,
        run_fingerprint_sha256=str(fingerprint["sha256"]),
        geometry_run_fingerprint_sha256=(
            geometry_run_fingerprint_sha256
        ),
        expected_successful_records=expected_successful_records,
        expected_geometry_records=expected_geometry_records,
        expected_shards=expected_shards,
        validate_payloads=False,
        statistics_path=ready_path,
    )
    verified_statistics, statistics_descriptor = (
        _verify_statistics_contract(
            ready_path,
            output_dir=output_dir,
            run_fingerprint_sha256=str(fingerprint["sha256"]),
            config=config,
            validate_payloads=False,
            expected_shards=expected_shards,
        )
    )
    if _canonical_json(verified_statistics) != _canonical_json(statistics):
        raise ArtifactIntegrityError(
            "statistics changed immediately after staging"
        )
    commit = {
        "schema": STATISTICS_COMMIT_SCHEMA,
        "run_fingerprint_sha256": str(fingerprint["sha256"]),
        "statistics": statistics_descriptor,
    }
    _atomic_write_text(
        commit_path,
        json.dumps(
            commit,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    persisted_commit = _load_statistics_commit(
        output_dir,
        run_fingerprint_sha256=str(fingerprint["sha256"]),
    )
    if _canonical_json(persisted_commit) != _canonical_json(commit):
        raise ArtifactIntegrityError(
            "statistics commit changed immediately after publication"
        )

    _remove_geometry_snapshot(snapshot_dir)
    _cleanup_committed_geometry_residue(output_dir)

    persisted_fingerprint = _load_verified_density_run_state(output_dir)
    persisted_commit = _load_statistics_commit(
        output_dir,
        run_fingerprint_sha256=str(fingerprint["sha256"]),
    )
    verified_statistics, descriptor_after_cleanup = (
        _verify_statistics_contract(
            ready_path,
            output_dir=output_dir,
            run_fingerprint_sha256=str(fingerprint["sha256"]),
            config=config,
            validate_payloads=False,
            expected_shards=expected_shards,
        )
    )
    if (
        _canonical_json(persisted_fingerprint)
        != _canonical_json(fingerprint)
        or _canonical_json(persisted_commit) != _canonical_json(commit)
        or _canonical_json(verified_statistics) != _canonical_json(statistics)
        or _canonical_json(descriptor_after_cleanup)
        != _canonical_json(statistics_descriptor)
    ):
        raise ArtifactIntegrityError(
            "committed publication changed before final promotion"
        )
    os.replace(ready_path, published_path)
    _fsync_directory(output_dir)
    published_statistics, published_descriptor = (
        _verify_statistics_contract(
            published_path,
            output_dir=output_dir,
            run_fingerprint_sha256=str(fingerprint["sha256"]),
            config=config,
            validate_payloads=False,
            expected_shards=expected_shards,
        )
    )
    if (
        _canonical_json(published_statistics) != _canonical_json(statistics)
        or _canonical_json(published_descriptor)
        != _canonical_json(statistics_descriptor)
    ):
        raise ArtifactIntegrityError(
            "published statistics differ from their permanent commit"
        )
    _assert_no_private_geometry_residue(output_dir)
    return published_statistics


def _audit_density_geometry_provenance(
    output_dir: Path,
    snapshot_dir: Path,
    expected_shards: dict[str, dict[str, Any]],
    expected_geometry_records: int,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    audit_path = output_dir / ".density_provenance_audit.sqlite3"
    with _CompletedRowIndex(audit_path) as completed_index:
        density_shards = completed_index.populate(
            output_dir,
            verify_checksums=True,
        )
        failure_descriptor = _read_verified_failure_journal(
            output_dir,
            completed_index=completed_index,
        )
        uncompleted_records = 0
        for _record in _iter_private_geometry_snapshot_records(
            snapshot_dir,
            output_dir,
            completed_rows=completed_index,
            expected_shards=expected_shards,
        ):
            uncompleted_records += 1
        if completed_index.geometry_count != expected_geometry_records:
            raise ArtifactIntegrityError(
                "final geometry provenance audit record count differs from "
                "the snapshot contract"
            )
        if completed_index.count + uncompleted_records != expected_geometry_records:
            raise ArtifactIntegrityError(
                "final density provenance audit did not cover every "
                "geometry record"
            )
        completed_index.validate_final_outcomes(expected_geometry_records)
        return completed_index.count, failure_descriptor, density_shards


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert geometry shards to normalized Gaussian promolecular "
            "density shards. This is an approximation, not a DFT calculation."
        )
    )
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", "--qm-config")
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--spacing", type=float)
    parser.add_argument("--box-padding", type=float)
    parser.add_argument(
        "--atomic-sigma",
        help="A positive scalar width or 'element' for covalent-radius widths.",
    )
    parser.add_argument(
        "--conformer-index",
        default="0",
        help="Zero-based conformer index or 'mean' for all valid conformers.",
    )
    parser.add_argument(
        "--allow-overflow",
        action="store_true",
        help="Flag insufficient coverage in metadata instead of failing the record.",
    )
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument(
        "--discrete-normalize",
        dest="discrete_normalize",
        action="store_true",
    )
    normalization.add_argument(
        "--continuous-normalize",
        dest="discrete_normalize",
        action="store_false",
    )
    parser.set_defaults(discrete_normalize=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-verify-checksums", action="store_true")
    args = parser.parse_args(argv)
    if args.num_workers <= 0 or args.worker_chunksize <= 0 or args.shard_size <= 0:
        parser.error("worker and shard sizes must be positive")
    if args.conformer_index.lower() != "mean":
        try:
            conformer_index = int(args.conformer_index)
        except ValueError:
            parser.error("--conformer-index must be a non-negative integer or 'mean'")
        if conformer_index < 0:
            parser.error("--conformer-index must be non-negative")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    pinned_config_bytes: Optional[bytes] = None
    pinned_config_descriptor: Optional[dict[str, Any]] = None
    if args.config is not None:
        (
            pinned_config_bytes,
            pinned_config_descriptor,
        ) = _read_pinned_bytes(Path(args.config))
    config = _resolved_config(
        args,
        pinned_config_bytes=pinned_config_bytes,
    )
    geometry_dir = Path(args.geometry_dir)
    output_dir = Path(args.output_dir)
    if args.resume:
        if not output_dir.is_dir():
            raise RuntimeError(
                f"cannot resume missing density output directory: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    with _OutputRunLock(output_dir):
        return _main_locked(
            args,
            config,
            pinned_config_descriptor,
            geometry_dir,
            output_dir,
        )


def _main_locked(
    args: argparse.Namespace,
    config: dict[str, Any],
    pinned_config_descriptor: Optional[dict[str, Any]],
    geometry_dir: Path,
    output_dir: Path,
) -> int:
    has_existing_run_artifacts = (
        next(
            (
                path
                for path in output_dir.iterdir()
                if path.name != WRITER_LOCK_FILENAME
            ),
            None,
        )
        is not None
    )
    if has_existing_run_artifacts and not args.resume:
        raise RuntimeError(
            f"{output_dir} already contains density generation artifacts; "
            "use --resume "
            "or choose an empty output directory"
        )
    fingerprint_inputs = {
        "geometry_manifest": geometry_dir / "manifest.json",
        "geometry_index": geometry_dir / "geometry_index.json",
        "geometry_run_state": geometry_dir / "run_state.json",
        "density_config": None,
    }
    fingerprint_parameters = {
        "schema": DENSITY_SCHEMA,
        "density": config,
        "num_workers": args.num_workers,
        "worker_chunksize": args.worker_chunksize,
        "shard_size": args.shard_size,
        "verify_checksums": not args.no_verify_checksums,
    }
    if args.resume:
        recovered_statistics = _recover_committed_statistics(
            output_dir,
            fingerprint_parameters=fingerprint_parameters,
            pinned_config_descriptor=pinned_config_descriptor,
            config=config,
        )
        if recovered_statistics is not None:
            recovered_failures = int(
                recovered_statistics["failed_records"]
            )
            print(
                json.dumps(
                    {
                        "recovered_committed_publication": True,
                        "failed": recovered_failures,
                        "output_dir": str(output_dir),
                        "method": "promolecular_gaussian",
                    },
                    ensure_ascii=False,
                )
            )
            return 1 if recovered_failures else 0
    if args.resume:
        fingerprint = _load_verified_density_run_state(output_dir)
        if (
            _canonical_json(fingerprint["parameters"])
            != _canonical_json(fingerprint_parameters)
            or _canonical_json(fingerprint["inputs"]["density_config"])
            != _canonical_json(pinned_config_descriptor)
        ):
            raise ArtifactIntegrityError(
                "resume parameters or pinned config differ from run state"
            )
    else:
        fingerprint = compute_run_fingerprint(
            fingerprint_inputs,
            fingerprint_parameters,
        )
        fingerprint["inputs"]["density_config"] = pinned_config_descriptor
        fingerprint["sha256"] = hashlib.sha256(
            _canonical_json(
                {
                    "inputs": fingerprint["inputs"],
                    "parameters": fingerprint["parameters"],
                }
            ).encode("utf-8")
        ).hexdigest()
    ensure_run_state(
        output_dir,
        fingerprint,
        resume=args.resume,
        state_schema=DENSITY_RUN_STATE_SCHEMA,
    )
    if args.resume:
        _invalidate_statistics_marker(output_dir)
    _cleanup_geometry_snapshot_transients(output_dir)
    reconcile_density_artifacts(output_dir)
    (
        geometry_snapshot_dir,
        expected_geometry_shards,
        expected_geometry_records,
    ) = _prepare_geometry_snapshot(
        geometry_dir,
        output_dir,
        fingerprint,
    )
    completed_index_path = output_dir / ".density_completed.sqlite3"
    with _CompletedRowIndex(completed_index_path) as completed_index:
        if args.resume:
            completed_index.populate(
                output_dir,
                verify_checksums=True,
            )
        skipped_by_resume = completed_index.count
        records = _iter_private_geometry_snapshot_records(
            geometry_snapshot_dir,
            output_dir,
            completed_rows=completed_index,
            expected_shards=expected_geometry_shards,
        )
        tasks = (
            (
                row_index,
                smiles,
                source_index,
                train_ordinal,
                geometry_artifact,
                geometry_artifact_sha256,
                geometry_key,
                geometry_payload_sha256,
                geometry,
                config,
            )
            for (
                row_index,
                smiles,
                source_index,
                train_ordinal,
                geometry_artifact,
                geometry_artifact_sha256,
                geometry_key,
                geometry_payload_sha256,
                geometry,
            ) in records
        )
        first_shard_id = _next_shard_id(output_dir)
        with _FailureJournal(output_dir) as failure_journal:
            if args.num_workers == 1:
                generated_successes, failure_count = _consume_results(
                    map(_density_task, tasks),
                    output_dir,
                    shard_size=args.shard_size,
                    first_shard_id=first_shard_id,
                    failure_journal=failure_journal,
                )
            else:
                with Pool(processes=args.num_workers) as pool:
                    results = pool.imap(
                        _density_task,
                        tasks,
                        chunksize=args.worker_chunksize,
                    )
                    generated_successes, failure_count = _consume_results(
                        results,
                        output_dir,
                        shard_size=args.shard_size,
                        first_shard_id=first_shard_id,
                        failure_journal=failure_journal,
                    )
        geometry_record_count = completed_index.geometry_count
        if geometry_record_count != expected_geometry_records:
            raise ArtifactIntegrityError(
                "streamed geometry count differs from finalized manifest: "
                f"{geometry_record_count} != {expected_geometry_records}"
            )
        accounted_records = (
            skipped_by_resume + generated_successes + failure_count
        )
        if accounted_records != geometry_record_count:
            raise ArtifactIntegrityError(
                "density accounting mismatch: "
                f"completed={skipped_by_resume}, "
                f"generated={generated_successes}, failed={failure_count}, "
                f"geometry={geometry_record_count}"
            )
    (
        final_geometry_shards,
        final_geometry_records,
        geometry_run_fingerprint_sha256,
    ) = (
        _revalidate_geometry_snapshot_or_poison(
            geometry_snapshot_dir,
            output_dir,
            fingerprint,
        )
    )
    if (
        final_geometry_shards != expected_geometry_shards
        or final_geometry_records != expected_geometry_records
    ):
        raise ArtifactIntegrityError(
            "geometry snapshot contract changed during density generation"
        )
    (
        finalized_density_records,
        failure_descriptor,
        finalized_density_shards,
    ) = _audit_density_geometry_provenance(
        output_dir,
        geometry_snapshot_dir,
        expected_geometry_shards,
        expected_geometry_records,
    )
    if finalized_density_records != skipped_by_resume + generated_successes:
        raise ArtifactIntegrityError(
            "final density provenance audit count differs from successful "
            "record accounting"
        )
    if int(failure_descriptor["record_count"]) != failure_count:
        raise ArtifactIntegrityError(
            "final failure journal count differs from generated failures"
        )
    ensure_run_state(
        output_dir,
        fingerprint,
        resume=True,
        state_schema=DENSITY_RUN_STATE_SCHEMA,
    )
    if failure_count:
        print(
            json.dumps(
                {
                    "skipped_by_resume": skipped_by_resume,
                    "generated": generated_successes,
                    "failed": failure_count,
                    "output_dir": str(output_dir),
                    "method": "promolecular_gaussian",
                    "publication_committed": False,
                },
                ensure_ascii=False,
            )
        )
        return 1
    _publish_statistics(
        output_dir,
        geometry_snapshot_dir,
        failure_descriptor=failure_descriptor,
        config=config,
        fingerprint=fingerprint,
        geometry_run_fingerprint_sha256=(
            geometry_run_fingerprint_sha256
        ),
        expected_successful_records=finalized_density_records,
        expected_geometry_records=expected_geometry_records,
        expected_shards=finalized_density_shards,
    )
    print(
        json.dumps(
            {
                "skipped_by_resume": skipped_by_resume,
                "generated": generated_successes,
                "failed": failure_count,
                "output_dir": str(output_dir),
                "method": "promolecular_gaussian",
            },
            ensure_ascii=False,
        )
    )
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

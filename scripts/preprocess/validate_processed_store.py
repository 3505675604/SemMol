"""Offline integrity and schema validation for a published SemMol store."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.datasets.pcqm_dataset import (  # noqa: E402
    MODALITY_ORDER,
    PCQMMultimodalDataset,
    load_manifest_view,
)
from src.datasets.storage import ShardedRecordStore  # noqa: E402
from src.utils.io import sha256_file  # noqa: E402


SUPPORTED_BUILD_SCHEMAS = {
    "semmol.pcqm_store_build.v1",
    "semmol.moleculenet_store_build.v1",
}
BUILD_MANIFEST_FIELDS = {
    "semmol.pcqm_store_build.v1": {
        "schema",
        "selection_manifest",
        "selection_metadata",
        "tokenizer",
        "geometry",
        "target_sizes",
        "n_bins",
        "storage",
        "features",
        "status",
        "record_count",
        "shard_count",
        "attempted_candidates",
        "failed_candidates",
        "failure_rate",
        "density_extent_preflight",
        "failure_log",
        "views",
    },
    "semmol.moleculenet_store_build.v1": {
        "schema",
        "dataset_name",
        "source",
        "split",
        "tokenizer",
        "features",
        "storage",
        "task",
        "status",
        "record_count",
        "shard_count",
        "views",
        "failure_log",
    },
}


def deterministic_validation_indices(
    record_count: int,
    sample_count: Optional[int],
    *,
    seed: int,
) -> np.ndarray:
    if not isinstance(record_count, int) or isinstance(record_count, bool):
        raise TypeError("record_count must be an integer")
    if record_count < 0:
        raise ValueError("record_count must be non-negative")
    if sample_count is None:
        return np.arange(record_count, dtype=np.int64)
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise TypeError("sample_count must be an integer or None")
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    sample_count = min(sample_count, record_count)
    if sample_count == record_count:
        return np.arange(record_count, dtype=np.int64)
    generator = np.random.default_rng(int(seed))
    return np.sort(
        generator.choice(
            record_count,
            size=sample_count,
            replace=False,
        ).astype(np.int64, copy=False)
    )


def _validate_build_manifest(store_dir: Path) -> dict[str, Any]:
    path = store_dir / "build-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"build manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in SUPPORTED_BUILD_SCHEMAS:
        raise RuntimeError(
            f"unsupported store build schema: {schema!r}"
        )
    expected_fields = set(BUILD_MANIFEST_FIELDS[str(schema)])
    if schema == "semmol.pcqm_store_build.v1" and "selection_current" in payload:
        expected_fields.add("selection_current")
    if set(payload) != expected_fields:
        raise RuntimeError("build manifest field inventory is not exact")
    if payload.get("status") != "complete":
        raise RuntimeError(f"store build status is not complete: {path}")
    record_count = payload.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count < 0
    ):
        raise RuntimeError("build manifest has an invalid record_count")
    views = payload.get("views")
    if not isinstance(views, dict) or not views:
        raise RuntimeError("build manifest has no published views")
    failure_log = payload.get("failure_log")
    expected_failure_fields = {"path", "sha256", "record_count"}
    if schema == "semmol.moleculenet_store_build.v1":
        expected_failure_fields.add("resolved_record_count")
    if (
        not isinstance(failure_log, dict)
        or set(failure_log) != expected_failure_fields
        or not isinstance(failure_log.get("record_count"), int)
        or isinstance(failure_log.get("record_count"), bool)
        or failure_log["record_count"] < 0
    ):
        raise RuntimeError("build manifest failure-log descriptor is invalid")
    relative = Path(str(failure_log.get("path", "")))
    failure_path = (store_dir / relative).resolve()
    if relative.is_absolute() or (
        failure_path != store_dir and store_dir not in failure_path.parents
    ):
        raise RuntimeError("build manifest failure log escapes the store")
    if not failure_path.is_file():
        raise FileNotFoundError(
            f"build manifest failure log is missing: {failure_path}"
        )
    expected = str(failure_log.get("sha256", ""))
    actual = sha256_file(failure_path)
    if expected != actual:
        raise RuntimeError(
            f"failure-log checksum mismatch: {failure_path}"
        )
    failure_keys: set[tuple[int, str]] = set()
    with failure_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    f"failure-log line {line_number} is not an object"
                )
            source_index = row.get("source_index")
            key = (
                -1 if source_index is None else int(source_index),
                str(row.get("stage", "")),
            )
            if key in failure_keys:
                raise RuntimeError("failure log contains duplicate records")
            failure_keys.add(key)
    if len(failure_keys) != failure_log["record_count"]:
        raise RuntimeError("failure-log record_count is inconsistent")
    if schema == "semmol.moleculenet_store_build.v1" and (
        failure_log.get("resolved_record_count") != len(failure_keys)
    ):
        raise RuntimeError("failure-log resolved_record_count is inconsistent")
    return payload


def _resolve_manifest_descriptors(
    store_dir: Path,
    build_manifest: Mapping[str, Any],
    *,
    manifests: Sequence[os.PathLike[str] | str],
) -> list[tuple[Path, Optional[int], Optional[str], str]]:
    views = build_manifest.get("views")
    if not isinstance(views, Mapping) or not views:
        raise RuntimeError("build manifest has no view descriptors")
    published: dict[Path, tuple[int, str, str]] = {}
    for name, raw_descriptor in views.items():
        if not isinstance(name, str) or not isinstance(raw_descriptor, Mapping):
            raise RuntimeError("build manifest contains an invalid view descriptor")
        if set(raw_descriptor) != {"path", "sha256", "record_count"}:
            raise RuntimeError(
                f"published view {name!r} descriptor inventory is invalid"
            )
        relative = Path(str(raw_descriptor.get("path", "")))
        path = (store_dir / relative).resolve()
        if relative.is_absolute() or (
            path != store_dir and store_dir not in path.parents
        ):
            raise RuntimeError(f"published view {name!r} escapes the store")
        expected_count = raw_descriptor.get("record_count")
        expected_sha = str(raw_descriptor.get("sha256", ""))
        if (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count < 0
            or len(expected_sha) != 64
            or set(expected_sha) - set("0123456789abcdef")
        ):
            raise RuntimeError(f"published view {name!r} descriptor is invalid")
        if not path.is_file():
            raise FileNotFoundError(f"published view is missing: {path}")
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"published view checksum mismatch: {path}")
        if path in published:
            raise RuntimeError("multiple build views resolve to the same file")
        published[path] = (expected_count, expected_sha, name)

    if not manifests:
        return [
            (path, count, checksum, name)
            for path, (count, checksum, name) in published.items()
        ]

    result: list[tuple[Path, Optional[int], Optional[str], str]] = []
    seen: set[Path] = set()
    for raw_path in manifests:
        path = Path(raw_path).resolve()
        if path in seen:
            raise RuntimeError(f"manifest was requested more than once: {path}")
        seen.add(path)
        descriptor = published.get(path)
        if descriptor is None:
            result.append((path, None, None, path.name))
        else:
            count, checksum, name = descriptor
            result.append((path, count, checksum, name))
    return result


def _validate_manifest_records(
    store_dir: Path,
    manifest_path: Path,
    modalities: Sequence[str],
    *,
    sample_count: Optional[int],
    seed: int,
) -> int:
    dataset = PCQMMultimodalDataset(
        store_dir=store_dir,
        manifest_path=manifest_path,
        modalities=modalities,
        strict=True,
    )
    try:
        indices = deterministic_validation_indices(
            len(dataset),
            sample_count,
            seed=seed,
        )
        for dataset_index in indices.tolist():
            sample = dataset[dataset_index]
            if int(sample["source_index"]) != int(
                dataset.manifest.source_indices[dataset_index]
            ):
                raise RuntimeError(
                    "dataset source_index changed after record reconstruction"
                )
        return len(indices)
    finally:
        dataset.close()


def validate_store(
    *,
    store_dir: os.PathLike[str] | str,
    manifests: Sequence[os.PathLike[str] | str],
    modalities: Sequence[str],
    sample_count: Optional[int],
    seed: int,
    verify_shard_checksums: bool,
) -> dict[str, Any]:
    directory = Path(store_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    build_manifest = _validate_build_manifest(directory)
    store = ShardedRecordStore(directory)
    try:
        if len(store) != int(build_manifest["record_count"]):
            raise RuntimeError(
                "build-manifest record_count differs from store.json"
            )
        tokenizer_contract = build_manifest.get("tokenizer")
        if (
            not isinstance(tokenizer_contract, Mapping)
            or store.metadata.tokenizer_sha256
            != str(tokenizer_contract.get("artifact_sha256", ""))
            or store.metadata.tokenizer_vocab_size
            != int(tokenizer_contract.get("vocab_size", -1))
        ):
            raise RuntimeError(
                "build-manifest tokenizer contract differs from store.json"
            )
        if verify_shard_checksums:
            store.verify_shard_checksums()
        raw_indices = deterministic_validation_indices(
            len(store),
            sample_count,
            seed=seed,
        )
        for record_index in raw_indices.tolist():
            record = store[record_index]
            for key in ("sample_id", "source_index", "smiles"):
                if key not in record:
                    raise RuntimeError(
                        f"record_index={record_index} is missing {key}"
                    )
    finally:
        store.close()

    descriptors = _resolve_manifest_descriptors(
        directory,
        build_manifest,
        manifests=manifests,
    )
    manifest_results: dict[str, Any] = {}
    for manifest_path, expected_count, expected_sha, view_name in descriptors:
        view = load_manifest_view(
            manifest_path,
            int(build_manifest["record_count"]),
        )
        if expected_count is not None and len(view.record_indices) != expected_count:
            raise RuntimeError(
                f"published view count mismatch: {manifest_path}"
            )
        actual_sha = sha256_file(manifest_path)
        if expected_sha is not None and actual_sha != expected_sha:
            raise RuntimeError(
                f"published view checksum changed during validation: {manifest_path}"
            )
        checked = _validate_manifest_records(
            directory,
            manifest_path,
            modalities,
            sample_count=(
                None
                if sample_count is None
                else min(sample_count, len(view.record_indices))
            ),
            seed=seed,
        )
        manifest_results[str(manifest_path)] = {
            "view": view_name,
            "record_count": len(view.record_indices),
            "validated_records": checked,
            "sha256": actual_sha,
            "published": expected_sha is not None,
        }
    return {
        "store_dir": str(directory),
        "record_count": int(build_manifest["record_count"]),
        "validated_store_records": len(raw_indices),
        "verified_shard_checksums": bool(verify_shard_checksums),
        "modalities": list(modalities),
        "manifests": manifest_results,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify checksums and decode SemMol processed-store records"
    )
    parser.add_argument("--store-dir", required=True)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help=(
            "manifest to validate; repeat as needed. When omitted, every "
            "checksummed view in build-manifest.json is validated."
        ),
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=MODALITY_ORDER,
        default=list(MODALITY_ORDER),
    )
    validation_scope = parser.add_mutually_exclusive_group()
    validation_scope.add_argument(
        "--full",
        action="store_true",
        help="decode every record",
    )
    validation_scope.add_argument(
        "--sample-count",
        type=int,
        default=1024,
        help="deterministic number of records to decode",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--skip-shard-checksums", action="store_true")
    args = parser.parse_args(argv)
    if args.sample_count is not None and args.sample_count < 0:
        parser.error("--sample-count must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = validate_store(
        store_dir=args.store_dir,
        manifests=args.manifest,
        modalities=args.modalities,
        sample_count=None if args.full else args.sample_count,
        seed=args.seed,
        verify_shard_checksums=not args.skip_shard_checksums,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import numpy as np
import pytest

from scripts.preprocess.validate_processed_store import (
    _resolve_manifest_descriptors,
    deterministic_validation_indices,
)


def test_full_validation_indices_cover_every_record() -> None:
    indices = deterministic_validation_indices(
        record_count=7,
        sample_count=None,
        seed=42,
    )

    np.testing.assert_array_equal(indices, np.arange(7, dtype=np.int64))


def test_sampled_validation_indices_are_sorted_unique_and_deterministic() -> None:
    first = deterministic_validation_indices(100, 13, seed=3407)
    second = deterministic_validation_indices(100, 13, seed=3407)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 13
    assert np.all(first[1:] > first[:-1])


def test_validation_sample_is_clamped_to_small_store() -> None:
    np.testing.assert_array_equal(
        deterministic_validation_indices(4, 5, seed=1),
        np.arange(4, dtype=np.int64),
    )


def test_build_views_are_auto_discovered_and_integrity_checked(tmp_path) -> None:
    view_dir = tmp_path / "views"
    view_dir.mkdir()
    view_path = view_dir / "train.npz"
    with view_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            record_index=np.array([0, 1], dtype=np.int64),
            source_index=np.array([10, 11], dtype=np.int64),
        )
    import hashlib

    digest = hashlib.sha256(view_path.read_bytes()).hexdigest()
    build_manifest = {
        "schema": "semmol.moleculenet_store_build.v1",
        "status": "complete",
        "record_count": 2,
        "views": {
            "train": {
                "path": "views/train.npz",
                "sha256": digest,
                "record_count": 2,
            }
        },
    }

    descriptors = _resolve_manifest_descriptors(
        tmp_path,
        build_manifest,
        manifests=[],
    )

    assert descriptors == [(view_path.resolve(), 2, digest, "train")]


def test_auto_discovered_view_rejects_checksum_mismatch(tmp_path) -> None:
    view_dir = tmp_path / "views"
    view_dir.mkdir()
    view_path = view_dir / "train.npz"
    view_path.write_bytes(b"not-the-committed-view")
    build_manifest = {
        "views": {
            "train": {
                "path": "views/train.npz",
                "sha256": "0" * 64,
                "record_count": 1,
            }
        }
    }

    with pytest.raises(RuntimeError, match="checksum"):
        _resolve_manifest_descriptors(
            tmp_path,
            build_manifest,
            manifests=[],
        )

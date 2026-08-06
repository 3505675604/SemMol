from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.datasets.pcqm_dataset import (
    DatasetRecordError,
    MissingModalityError,
    PCQMMultimodalDataset,
)
from src.datasets.storage import (
    LmdbShardWriter,
    StoreMetadata,
    StoreSchemaError,
    write_store_metadata,
)


def _record(source_index: int, include_qm: bool = True) -> dict:
    record = {
        "sample_id": f"pcqm:{source_index}",
        "source_index": source_index,
        "smiles": "CC",
        "gap": 5.5,
        "input_ids": np.array([2, 5, 3], dtype=np.int32),
        "token_spans": np.array([[-1, -1], [0, 2], [-1, -1]], dtype=np.int32),
        "graph": {
            "node_feat": np.zeros((2, 9), dtype=np.int16),
            "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int32),
            "edge_feat": np.zeros((2, 3), dtype=np.int8),
            "num_nodes": 2,
        },
        "geometry": {
            "atomic_numbers": np.array([6, 6], dtype=np.int16),
            "coords": np.zeros((1, 2, 3), dtype=np.float32),
            "conformer_mask": np.array([True]),
            "energies": np.array([np.nan], dtype=np.float32),
            "energy_mask": np.array([False]),
            "heavy_atom_indices": np.array([0, 1], dtype=np.int32),
            "sources": ["official_dft"],
        },
    }
    record["graph"]["node_feat"][:, 0] = 5
    if include_qm:
        record["density"] = {
            "grid": np.full((1, 4, 4, 4), 1.5, dtype=np.float16),
            "origin": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            "spacing": 0.5,
            "neutral_atom_electron_count": 12.0,
            "integrated_electrons": 12.0,
            "prequantization_integrated_electrons": 12.0,
            "overflow": False,
            "overflow_axes": np.array([False, False, False]),
            "atomic_sigmas": np.array([0.34, 0.34], dtype=np.float32),
            "method": "promolecular_gaussian",
            "box_padding": 2.0,
            "conformers_used": np.array([0], dtype=np.int16),
            "conformer_reduction": "single",
            "conformer_alignment": "none",
            "normalization_requested": "discrete_electron_count",
            "normalization_applied": "discrete_electron_count",
        }
    return record


def _build_store(tmp_path, records: list[dict]):
    store_dir = tmp_path / "store"
    with LmdbShardWriter(
        store_dir=store_dir,
        shard_id=0,
        start_index=0,
        expected_records=len(records),
        map_size=16 * 1024 * 1024,
    ) as writer:
        for record_index, record in enumerate(records):
            writer.put(record_index, record)
    write_store_metadata(
        store_dir,
        StoreMetadata(
            schema_version=1,
            record_count=len(records),
            records_per_shard=len(records),
            modalities=("1d", "2d", "3d", "qm"),
            tokenizer_sha256="b" * 64,
            tokenizer_vocab_size=16,
            shards=("shard-000000.lmdb",),
        ),
    )
    manifest_path = tmp_path / "subset.npz"
    np.savez(
        manifest_path,
        record_index=np.arange(len(records), dtype=np.int64),
        source_index=np.array([r["source_index"] for r in records], dtype=np.int64),
    )
    (store_dir / "build-manifest.json").write_text(
        json.dumps(
            {
                "schema": "semmol.pcqm_store_build.v1",
                "status": "complete",
                "record_count": len(records),
                "tokenizer": {
                    "artifact_sha256": "b" * 64,
                    "vocab_size": 16,
                },
                "views": {},
            }
        ),
        encoding="utf-8",
    )
    return store_dir, manifest_path


def test_pcqm_dataset_reconstructs_tensor_modalities_from_safe_records(tmp_path) -> None:
    store_dir, manifest_path = _build_store(tmp_path, [_record(7), _record(9)])

    dataset = PCQMMultimodalDataset(
        store_dir=store_dir,
        manifest_path=manifest_path,
        modalities=("1d", "2d", "3d", "qm"),
        strict=True,
    )
    sample = dataset[1]

    assert sample["sample_id"] == "pcqm:9"
    assert sample["input_ids"].dtype.is_floating_point is False
    assert sample["graph"].x.shape == (2, 9)
    assert sample["coords"].shape == (1, 2, 3)
    assert sample["qm_grid"].shape == (1, 4, 4, 4)


def test_pcqm_dataset_raises_in_strict_mode_when_requested_modality_is_missing(tmp_path) -> None:
    store_dir, manifest_path = _build_store(tmp_path, [_record(7, include_qm=False)])
    dataset = PCQMMultimodalDataset(
        store_dir=store_dir,
        manifest_path=manifest_path,
        modalities=("1d", "2d", "3d", "qm"),
        strict=True,
    )

    with pytest.raises(MissingModalityError, match="qm"):
        _ = dataset[0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "input_ids",
            np.array([2, -1, 3], dtype=np.int32),
            "input_ids",
        ),
        (
            "input_ids",
            np.array([2, 16, 3], dtype=np.int32),
            "tokenizer_vocab_size",
        ),
    ],
)
def test_pcqm_dataset_rejects_token_ids_outside_store_vocabulary(
    tmp_path,
    field,
    value,
    message,
) -> None:
    record = _record(7)
    record[field] = value
    store_dir, manifest_path = _build_store(tmp_path, [record])
    dataset = PCQMMultimodalDataset(store_dir, manifest_path)

    with pytest.raises(DatasetRecordError, match=message):
        _ = dataset[0]


def test_pcqm_dataset_enforces_ogb_graph_schema_and_category_ranges(tmp_path) -> None:
    bad_shape = _record(7)
    bad_shape["graph"]["node_feat"] = np.zeros((2, 8), dtype=np.int16)
    store_dir, manifest_path = _build_store(tmp_path / "shape", [bad_shape])
    dataset = PCQMMultimodalDataset(store_dir, manifest_path)
    with pytest.raises(DatasetRecordError, match=r"\(N, 9\)"):
        _ = dataset[0]

    bad_category = _record(8)
    bad_category["graph"]["edge_feat"][0, 0] = -1
    store_dir, manifest_path = _build_store(tmp_path / "category", [bad_category])
    dataset = PCQMMultimodalDataset(store_dir, manifest_path)
    with pytest.raises(DatasetRecordError, match="edge_feat"):
        _ = dataset[0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record["geometry"].update(
                {
                    "coords": np.empty((0, 2, 3), dtype=np.float32),
                    "conformer_mask": np.empty((0,), dtype=np.bool_),
                    "energies": np.empty((0,), dtype=np.float32),
                    "energy_mask": np.empty((0,), dtype=np.bool_),
                    "sources": [],
                }
            ),
            "at least one conformer",
        ),
        (
            lambda record: record["geometry"]["coords"].__setitem__(
                (0, 0, 0),
                np.nan,
            ),
            "NaN/Inf",
        ),
        (
            lambda record: record["density"].pop("origin"),
            "origin",
        ),
        (
            lambda record: record["density"].update(
                {"grid": np.empty((1, 0, 4, 4), dtype=np.float16)}
            ),
            "空间维",
        ),
    ],
)
def test_pcqm_dataset_rejects_invalid_geometry_and_density(
    tmp_path,
    mutate,
    message,
) -> None:
    record = _record(7)
    mutate(record)
    store_dir, manifest_path = _build_store(tmp_path, [record])
    dataset = PCQMMultimodalDataset(store_dir, manifest_path)

    with pytest.raises(DatasetRecordError, match=message):
        _ = dataset[0]


def test_pcqm_dataset_rejects_masked_nonfinite_label(tmp_path) -> None:
    record = _record(7)
    record["labels"] = np.array([np.nan], dtype=np.float32)
    record["label_mask"] = np.array([True], dtype=np.bool_)
    store_dir, manifest_path = _build_store(tmp_path, [record])
    dataset = PCQMMultimodalDataset(store_dir, manifest_path)

    with pytest.raises(DatasetRecordError, match="非有限"):
        _ = dataset[0]


def test_pcqm_manifest_rejects_float_indices_without_truncating(tmp_path) -> None:
    store_dir, manifest_path = _build_store(tmp_path, [_record(7)])
    np.savez(
        manifest_path,
        record_index=np.array([0.5], dtype=np.float64),
        source_index=np.array([7], dtype=np.int64),
    )

    with pytest.raises(DatasetRecordError, match="一维整数数组"):
        PCQMMultimodalDataset(store_dir, manifest_path)


def test_pcqm_dataset_rejects_noncanonical_heavy_atom_order(tmp_path) -> None:
    record = _record(7)
    record["geometry"]["heavy_atom_indices"] = np.array([1, 0], dtype=np.int32)
    store_dir, manifest_path = _build_store(tmp_path, [record])
    dataset = PCQMMultimodalDataset(store_dir, manifest_path)

    with pytest.raises(DatasetRecordError, match="canonical"):
        _ = dataset[0]


def test_pcqm_dataset_requires_complete_store_build_manifest(tmp_path) -> None:
    store_dir, manifest_path = _build_store(tmp_path, [_record(7)])
    build_path = store_dir / "build-manifest.json"
    payload = json.loads(build_path.read_text(encoding="utf-8"))
    payload["status"] = "building"
    build_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StoreSchemaError, match="complete"):
        PCQMMultimodalDataset(store_dir, manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("record_count", 2, "record_count"),
        ("tokenizer", {"artifact_sha256": "c" * 64, "vocab_size": 16}, "tokenizer"),
        ("tokenizer", {"artifact_sha256": "b" * 64, "vocab_size": 17}, "tokenizer"),
    ],
)
def test_pcqm_dataset_binds_build_manifest_to_store_metadata(
    tmp_path,
    field,
    value,
    message,
) -> None:
    store_dir, manifest_path = _build_store(tmp_path, [_record(7)])
    build_path = store_dir / "build-manifest.json"
    payload = json.loads(build_path.read_text(encoding="utf-8"))
    payload[field] = value
    build_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StoreSchemaError, match=message):
        PCQMMultimodalDataset(store_dir, manifest_path)


def test_pcqm_dataset_validates_registered_view_count_and_hash(tmp_path) -> None:
    store_dir, _ = _build_store(tmp_path, [_record(7), _record(9)])
    views_dir = store_dir / "views"
    views_dir.mkdir()
    registered = views_dir / "pcqm_2.npz"
    np.savez(
        registered,
        record_index=np.array([0, 1], dtype=np.int64),
        source_index=np.array([7, 9], dtype=np.int64),
    )
    build_path = store_dir / "build-manifest.json"
    payload = json.loads(build_path.read_text(encoding="utf-8"))
    payload["views"] = {
        "2": {
            "path": "views/pcqm_2.npz",
            "record_count": 2,
            "sha256": hashlib.sha256(
                registered.read_bytes()
            ).hexdigest(),
        }
    }
    build_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = PCQMMultimodalDataset(store_dir, registered)
    dataset.close()

    payload["views"]["2"]["record_count"] = 1
    build_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetRecordError, match="record_count"):
        PCQMMultimodalDataset(store_dir, registered)

    payload["views"]["2"]["record_count"] = 2
    payload["views"]["2"]["sha256"] = "0" * 64
    build_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetRecordError, match="SHA-256"):
        PCQMMultimodalDataset(store_dir, registered)


def test_pcqm_dataset_allows_unregistered_custom_subset(tmp_path) -> None:
    store_dir, custom_subset = _build_store(
        tmp_path,
        [_record(7), _record(9)],
    )
    registered = store_dir / "registered.npz"
    np.savez(
        registered,
        record_index=np.array([0], dtype=np.int64),
        source_index=np.array([7], dtype=np.int64),
    )
    build_path = store_dir / "build-manifest.json"
    payload = json.loads(build_path.read_text(encoding="utf-8"))
    payload["views"] = {
        "registered": {
            "path": registered.name,
            "record_count": 1,
            "sha256": hashlib.sha256(
                registered.read_bytes()
            ).hexdigest(),
        }
    }
    build_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = PCQMMultimodalDataset(store_dir, custom_subset)
    try:
        assert len(dataset) == 2
    finally:
        dataset.close()

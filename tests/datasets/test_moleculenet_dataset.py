from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.datasets.moleculenet_dataset import (
    MoleculeNetDataset,
    MoleculeNetRegistryError,
    extract_moleculenet_rows,
    get_moleculenet_spec,
)
from src.datasets.storage import (
    LmdbShardWriter,
    StoreMetadata,
    write_store_metadata,
)


def _build_moleculenet_store(tmp_path):
    store_dir = tmp_path / "bbbp"
    record = {
        "sample_id": "bbbp:0",
        "source_index": 0,
        "smiles": "CC",
        "input_ids": np.array([2, 5, 3], dtype=np.int32),
        "token_spans": np.array(
            [[-1, -1], [0, 2], [-1, -1]],
            dtype=np.int32,
        ),
        "labels": np.array([1.0], dtype=np.float32),
        "label_mask": np.array([True], dtype=np.bool_),
        "dataset_name": "bbbp",
        "task_type": "classification",
        "label_columns": ["p_np"],
    }
    with LmdbShardWriter(
        store_dir=store_dir,
        shard_id=0,
        start_index=0,
        expected_records=1,
        map_size=16 * 1024 * 1024,
    ) as writer:
        writer.put(0, record)
    write_store_metadata(
        store_dir,
        StoreMetadata(
            schema_version=1,
            record_count=1,
            records_per_shard=1,
            modalities=("1d",),
            tokenizer_sha256="b" * 64,
            tokenizer_vocab_size=16,
            shards=("shard-000000.lmdb",),
        ),
    )
    view_path = store_dir / "train.npz"
    np.savez(
        view_path,
        record_index=np.array([0], dtype=np.int64),
        source_index=np.array([0], dtype=np.int64),
    )
    build_path = store_dir / "build-manifest.json"
    build_path.write_text(
        json.dumps(
            {
                "schema": "semmol.moleculenet_store_build.v1",
                "status": "complete",
                "dataset_name": "bbbp",
                "record_count": 1,
                "tokenizer": {
                    "artifact_sha256": "b" * 64,
                    "vocab_size": 16,
                },
                "source": {
                    "label_columns": ["p_np"],
                },
                "task": {
                    "type": "classification",
                    "num_tasks": 1,
                    "main_metric": "roc_auc",
                },
                "views": {
                    "train": {
                        "path": view_path.name,
                        "record_count": 1,
                        "sha256": hashlib.sha256(
                            view_path.read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return store_dir, view_path, build_path


def test_bace_registry_uses_expected_smiles_and_label_columns() -> None:
    spec = get_moleculenet_spec("bace")

    assert spec.smiles_column == "mol"
    assert spec.label_columns == ("Class",)
    assert spec.task_type == "classification"


def test_extract_rows_preserves_original_row_indices_and_missing_label_mask() -> None:
    frame = pd.DataFrame(
        {
            "smiles": ["CC", "N", "O"],
            "NR-AR": [1.0, np.nan, 0.0],
            "NR-AR-LBD": [np.nan, 1.0, 0.0],
            "NR-AhR": [0.0, 1.0, 0.0],
            "NR-Aromatase": [0.0, 1.0, 0.0],
            "NR-ER": [0.0, 1.0, 0.0],
            "NR-ER-LBD": [0.0, 1.0, 0.0],
            "NR-PPAR-gamma": [0.0, 1.0, 0.0],
            "SR-ARE": [0.0, 1.0, 0.0],
            "SR-ATAD5": [0.0, 1.0, 0.0],
            "SR-HSE": [0.0, 1.0, 0.0],
            "SR-MMP": [0.0, 1.0, 0.0],
            "SR-p53": [0.0, 1.0, 0.0],
        },
        index=[4, 8, 12],
    )

    rows = extract_moleculenet_rows(frame, get_moleculenet_spec("tox21"))

    assert rows.row_indices.tolist() == [4, 8, 12]
    assert rows.labels.shape == (3, 12)
    assert rows.label_mask[0, :2].tolist() == [True, False]
    assert rows.label_mask[1, :2].tolist() == [False, True]


def test_toxcast_dynamic_columns_require_exactly_617_tasks() -> None:
    frame = pd.DataFrame({"smiles": ["CC"], "assay_0": [1.0], "assay_1": [0.0]})

    with pytest.raises(MoleculeNetRegistryError, match="617"):
        extract_moleculenet_rows(frame, get_moleculenet_spec("toxcast"))


def test_nonempty_malformed_label_is_not_silently_treated_as_missing() -> None:
    frame = pd.DataFrame(
        {
            "smiles": ["CC", "CO"],
            "Class": [1.0, "not-a-label"],
        }
    )

    with pytest.raises(MoleculeNetRegistryError, match=r"row=1.*Class"):
        extract_moleculenet_rows(frame, get_moleculenet_spec("bace"))


def test_extract_rows_rejects_fractional_dataframe_index() -> None:
    frame = pd.DataFrame(
        {
            "smiles": ["CC"],
            "p_np": [1.0],
        },
        index=pd.Index([0.5]),
    )

    with pytest.raises(MoleculeNetRegistryError, match="integer row numbers"):
        extract_moleculenet_rows(frame, get_moleculenet_spec("bbbp"))


def test_moleculenet_dataset_reuses_base_publication_contract(tmp_path) -> None:
    store_dir, view_path, _ = _build_moleculenet_store(tmp_path)

    dataset = MoleculeNetDataset(
        "bbbp",
        store_dir,
        view_path,
        modalities=("1d",),
    )
    try:
        sample = dataset[0]
        assert sample["dataset_name"] == "bbbp"
        assert sample["task_type"] == "classification"
    finally:
        dataset.close()


def test_moleculenet_dataset_keeps_registry_specific_contract(tmp_path) -> None:
    store_dir, view_path, build_path = _build_moleculenet_store(tmp_path)
    payload = json.loads(build_path.read_text(encoding="utf-8"))
    payload["dataset_name"] = "bace"
    build_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MoleculeNetRegistryError, match="dataset"):
        MoleculeNetDataset(
            "bbbp",
            store_dir,
            view_path,
            modalities=("1d",),
        )

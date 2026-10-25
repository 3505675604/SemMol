from argparse import Namespace
from decimal import Decimal
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import scripts.preprocess.generate_3d_conformer as conformer_module
from scripts.preprocess.generate_3d_conformer import (
    ArtifactIntegrityError,
    FailureJournal,
    ResumeStateError,
    _optional_integer,
    _write_run_metadata,
    compute_run_fingerprint,
    ensure_run_state,
    iter_pending_work_items,
    load_geometry_by_source_index,
    load_completed_rows,
    load_work_items,
    parse_args,
    prepare_work_database,
    reconcile_geometry_artifacts,
    validate_official_sdf_work_contract,
    verify_run_fingerprint,
    write_geometry_index,
    write_geometry_shard,
)
from src.molecular.geometry import GeometryRecord


def test_geometry_shard_is_safe_atomic_and_checksummed(tmp_path):
    record = GeometryRecord(
        atomic_numbers=np.array([6, 1], dtype=np.int64),
        coords=np.zeros((1, 2, 3), dtype=np.float32),
        energies=np.array([np.nan], dtype=np.float32),
        conformer_mask=np.array([True]),
        conformer_source=np.array(["etkdg_unoptimized"]),
        heavy_atom_indices=np.array([0], dtype=np.int64),
        heavy_atom_mapping=np.array([0], dtype=np.int64),
        canonical_smiles="C",
        reason="force_field_unavailable",
    )

    metadata = write_geometry_shard(
        [(3, "C", 9, record)],
        tmp_path,
        shard_id=2,
    )
    shard_path = tmp_path / metadata["filename"]
    raw = shard_path.read_bytes()

    assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]
    with np.load(shard_path, allow_pickle=False) as stored:
        assert stored["r000000__coords"].shape == (1, 2, 3)
        assert stored["r000000__conformer_source"].dtype.kind == "U"
    sidecar = json.loads((tmp_path / "shard_000002.json").read_text("utf-8"))
    assert sidecar["records"][0]["row_index"] == 3
    assert sidecar["records"][0]["source_index"] == 9
    assert load_completed_rows(tmp_path, verify_checksums=True) == {3}

    index_metadata = write_geometry_index(tmp_path)
    assert index_metadata["record_count"] == 1
    with np.load(tmp_path / "geometry_index.npz", allow_pickle=False) as index:
        assert index["source_index"].tolist() == [9]
        assert index["row_index"].tolist() == [3]
        assert index["sdf_ordinal"].tolist() == [-1]
        assert index["shard_id"].tolist() == [2]
        assert index["record_ordinal"].tolist() == [0]
    loaded = load_geometry_by_source_index(tmp_path, source_index=9)
    assert loaded.atomic_numbers.tolist() == [6, 1]
    assert loaded.conformer_source.tolist() == ["etkdg_unoptimized"]


def test_manifest_joins_by_source_index_and_keeps_train_sdf_ordinal(tmp_path):
    input_path = tmp_path / "molecules.csv"
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "source_index": [100, 200],
            "smiles": ["C", "CO"],
        }
    ).to_csv(input_path, index=False)
    pd.DataFrame(
        {
            "source_index": [200],
            "train_ordinal": [0],
        }
    ).to_csv(manifest_path, index=False)

    items = load_work_items(
        input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=manifest_path,
    )

    assert items == [
        {
            "row_index": 200,
            "source_index": 200,
            "sdf_ordinal": 0,
            "smiles": "CO",
        }
    ]


def test_negative_one_train_ordinal_is_normalized_for_etkdg_fallback(tmp_path):
    input_path = tmp_path / "molecules.csv"
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"source_index": [300], "smiles": ["N"]}).to_csv(
        input_path,
        index=False,
    )
    pd.DataFrame({"source_index": [300], "train_ordinal": [-1]}).to_csv(
        manifest_path,
        index=False,
    )

    items = load_work_items(
        input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=manifest_path,
    )

    assert items[0]["sdf_ordinal"] is None


def test_resume_rejects_changed_inputs_or_parameters(tmp_path):
    input_path = tmp_path / "input.csv"
    input_path.write_text("smiles\nC\n", encoding="utf-8")
    first = compute_run_fingerprint(
        {"input": input_path},
        {"seed": 42, "num_conformers": 5},
    )
    changed = compute_run_fingerprint(
        {"input": input_path},
        {"seed": 43, "num_conformers": 5},
    )

    ensure_run_state(tmp_path / "out", first, resume=False)
    ensure_run_state(tmp_path / "out", first, resume=True)
    with pytest.raises(ResumeStateError):
        ensure_run_state(tmp_path / "out", changed, resume=True)


def test_orphan_shard_or_sidecar_is_reported_as_corruption(tmp_path):
    (tmp_path / "shard_000001.npz").write_bytes(b"orphan")

    with pytest.raises(ArtifactIntegrityError):
        load_completed_rows(tmp_path, verify_checksums=False)


def test_official_sdf_does_not_silently_ignore_multiple_workers():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                "input.csv",
                "--output-dir",
                "out",
                "--official-sdf",
                "train.sdf",
                "--num-workers",
                "2",
            ]
        )


def test_streaming_work_database_preserves_manifest_order_and_input_ordinal(
    tmp_path,
):
    input_path = tmp_path / "source.csv"
    manifest_path = tmp_path / "selection.csv"
    pd.DataFrame(
        {
            "source_index": [100, 200, 300],
            "smiles": ["C", "CO", "N"],
            "train_ordinal": [7, 3, 9],
        }
    ).to_csv(input_path, index=False)
    pd.DataFrame(
        {
            "source_index": [300, 100],
        }
    ).to_csv(manifest_path, index=False)

    connection = prepare_work_database(
        tmp_path / "work.sqlite3",
        input_path=input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=manifest_path,
        fingerprint_sha256="a" * 64,
        chunk_size=1,
    )
    try:
        items = list(iter_pending_work_items(connection))
    finally:
        connection.close()

    assert [item["source_index"] for item in items] == [300, 100]
    assert [item["sdf_ordinal"] for item in items] == [9, 7]


def test_official_sdf_contract_accepts_sparse_unique_ordinals(tmp_path):
    input_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "source_index": [10, 20, 30],
            "smiles": ["C", "N", "O"],
            "train_ordinal": [9, 2, 100],
        }
    ).to_csv(input_path, index=False)
    connection = prepare_work_database(
        tmp_path / "work.sqlite3",
        input_path=input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=None,
        fingerprint_sha256="c" * 64,
        chunk_size=1,
    )
    try:
        validate_official_sdf_work_contract(connection)
        ordered = list(
            iter_pending_work_items(connection, order_by="sdf_ordinal")
        )
    finally:
        connection.close()

    assert [item["sdf_ordinal"] for item in ordered] == [2, 9, 100]


def test_official_sdf_merge_uses_sparse_absolute_positions(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "source_index": [10, 20],
            "smiles": ["C", "N"],
            "train_ordinal": [5, 2],
        }
    ).to_csv(input_path, index=False)
    connection = prepare_work_database(
        tmp_path / "work.sqlite3",
        input_path=input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=None,
        fingerprint_sha256="4" * 64,
        chunk_size=1,
    )
    monkeypatch.setattr(
        conformer_module,
        "iter_sdf_molecules",
        lambda _path: iter(
            (ordinal, f"slot-{ordinal}") for ordinal in range(6)
        ),
    )
    monkeypatch.setattr(
        conformer_module,
        "_sdf_result_for_item",
        lambda item, sdf_mol, **_kwargs: (item, sdf_mol, None),
    )
    try:
        results = list(
            conformer_module._sdf_results_from_database(
                connection,
                tmp_path / "unused.sdf",
                num_conformers=1,
                prune_rms_thresh=0.5,
                seed=42,
                energy_property=None,
                optimize=False,
            )
        )
    finally:
        connection.close()

    assert [
        (item["sdf_ordinal"], record)
        for item, record, failure in results
        if failure is None
    ] == [(2, "slot-2"), (5, "slot-5")]


def test_optional_integer_rejects_non_float_fractional_scalars():
    with pytest.raises(ValueError, match="must be integral"):
        _optional_integer(
            Decimal("1.5"),
            field="train_ordinal",
            row_number=0,
        )


def test_official_sdf_contract_rejects_missing_ordinal(tmp_path):
    input_path = tmp_path / "source.csv"
    pd.DataFrame({"source_index": [10], "smiles": ["C"]}).to_csv(
        input_path,
        index=False,
    )
    connection = prepare_work_database(
        tmp_path / "work.sqlite3",
        input_path=input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=None,
        fingerprint_sha256="d" * 64,
        chunk_size=1,
    )
    try:
        with pytest.raises(ValueError, match="non-negative.*every work item"):
            validate_official_sdf_work_contract(connection)
    finally:
        connection.close()


def test_official_sdf_contract_rejects_negative_ordinal(tmp_path):
    input_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "source_index": [10],
            "smiles": ["C"],
            "train_ordinal": [-1],
        }
    ).to_csv(input_path, index=False)
    connection = prepare_work_database(
        tmp_path / "work.sqlite3",
        input_path=input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=None,
        fingerprint_sha256="e" * 64,
        chunk_size=1,
    )
    try:
        with pytest.raises(ValueError, match="non-negative.*every work item"):
            validate_official_sdf_work_contract(connection)
    finally:
        connection.close()


def test_official_sdf_contract_rejects_duplicate_ordinal_across_chunks(
    tmp_path,
):
    input_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "source_index": [10, 20],
            "smiles": ["C", "N"],
            "train_ordinal": [4, 4],
        }
    ).to_csv(input_path, index=False)
    connection = prepare_work_database(
        tmp_path / "work.sqlite3",
        input_path=input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=None,
        fingerprint_sha256="f" * 64,
        chunk_size=1,
    )
    try:
        with pytest.raises(ValueError, match="official SDF ordinals.*unique"):
            validate_official_sdf_work_contract(connection)
    finally:
        connection.close()


def test_streaming_database_rejects_manifest_source_ordinal_conflict(
    tmp_path,
):
    input_path = tmp_path / "source.csv"
    manifest_path = tmp_path / "selection.csv"
    pd.DataFrame(
        {
            "source_index": [10],
            "smiles": ["C"],
            "train_ordinal": [3],
        }
    ).to_csv(input_path, index=False)
    pd.DataFrame(
        {
            "source_index": [10],
            "train_ordinal": [4],
        }
    ).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="manifest SDF ordinal conflict"):
        prepare_work_database(
            tmp_path / "work.sqlite3",
            input_path=input_path,
            smiles_col="smiles",
            source_index_col="source_index",
            sdf_ordinal_col=None,
            manifest_path=manifest_path,
            fingerprint_sha256="1" * 64,
            chunk_size=1,
        )


def test_streaming_database_rejects_same_named_manifest_smiles_conflict(
    tmp_path,
):
    input_path = tmp_path / "source.csv"
    manifest_path = tmp_path / "selection.csv"
    pd.DataFrame({"source_index": [10], "smiles": ["C"]}).to_csv(
        input_path,
        index=False,
    )
    pd.DataFrame({"source_index": [10], "smiles": ["N"]}).to_csv(
        manifest_path,
        index=False,
    )

    with pytest.raises(ValueError, match="manifest SMILES conflict"):
        prepare_work_database(
            tmp_path / "work.sqlite3",
            input_path=input_path,
            smiles_col="smiles",
            source_index_col="source_index",
            sdf_ordinal_col=None,
            manifest_path=manifest_path,
            fingerprint_sha256="2" * 64,
            chunk_size=1,
        )


def test_streaming_database_ignores_differently_named_canonical_smiles(
    tmp_path,
):
    input_path = tmp_path / "source.csv"
    manifest_path = tmp_path / "selection.csv"
    pd.DataFrame({"source_index": [10], "smiles": ["C"]}).to_csv(
        input_path,
        index=False,
    )
    pd.DataFrame(
        {
            "source_index": [10],
            "canonical_smiles": ["N"],
        }
    ).to_csv(manifest_path, index=False)
    connection = prepare_work_database(
        tmp_path / "work.sqlite3",
        input_path=input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=manifest_path,
        fingerprint_sha256="3" * 64,
        chunk_size=1,
    )
    try:
        items = list(iter_pending_work_items(connection))
    finally:
        connection.close()

    assert items[0]["smiles"] == "C"


def test_small_compatibility_loader_inherits_input_ordinal(tmp_path):
    input_path = tmp_path / "source.csv"
    manifest_path = tmp_path / "selection.csv"
    pd.DataFrame(
        {
            "source_index": [10, 20],
            "smiles": ["C", "N"],
            "train_ordinal": [4, 8],
        }
    ).to_csv(input_path, index=False)
    pd.DataFrame({"source_index": [20]}).to_csv(manifest_path, index=False)

    items = load_work_items(
        input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=manifest_path,
    )

    assert items == [
        {
            "row_index": 20,
            "smiles": "N",
            "source_index": 20,
            "sdf_ordinal": 8,
        }
    ]


def test_failure_journal_streams_and_deduplicates_resume_rows(tmp_path):
    input_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "source_index": [4, 8],
            "smiles": ["C", "N"],
        }
    ).to_csv(input_path, index=False)
    database_path = tmp_path / "work.sqlite3"
    connection = prepare_work_database(
        database_path,
        input_path=input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=None,
        fingerprint_sha256="b" * 64,
        chunk_size=1,
    )
    failure = {
        "row_index": 4,
        "source_index": 4,
        "sdf_ordinal": None,
        "smiles": "C",
        "stage": "etkdg_fallback",
        "error_type": "ValueError",
        "message": "failed",
    }
    with FailureJournal(connection, tmp_path / "failures.jsonl") as journal:
        assert journal.record(failure) is True
        assert journal.record(failure) is False
    connection.close()

    resumed = prepare_work_database(
        database_path,
        input_path=input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=None,
        fingerprint_sha256="b" * 64,
        chunk_size=1,
    )
    try:
        assert [item["row_index"] for item in iter_pending_work_items(resumed)] == [8]
    finally:
        resumed.close()
    lines = (tmp_path / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_reconcile_recovers_new_orphan_shard_sidecar(tmp_path):
    record = GeometryRecord(
        atomic_numbers=np.array([6, 1], dtype=np.int64),
        coords=np.zeros((1, 2, 3), dtype=np.float32),
        energies=np.array([np.nan], dtype=np.float32),
        conformer_mask=np.array([True]),
        conformer_source=np.array(["etkdg_unoptimized"]),
        heavy_atom_indices=np.array([0], dtype=np.int64),
        heavy_atom_mapping=np.array([0], dtype=np.int64),
        canonical_smiles="C",
    )
    write_geometry_shard([(5, "C", 11, 2, record)], tmp_path, shard_id=0)
    (tmp_path / "shard_000000.json").unlink()

    reconcile_geometry_artifacts(tmp_path, verify_checksums=True)

    recovered = json.loads(
        (tmp_path / "shard_000000.json").read_text(encoding="utf-8")
    )
    assert recovered["records"][0]["source_index"] == 11
    assert recovered["records"][0]["train_ordinal"] == 2
    assert load_completed_rows(tmp_path) == {5}


def test_reconcile_removes_sidecar_whose_data_was_never_published(tmp_path):
    sidecar = tmp_path / "shard_000000.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": "semmol.geometry.v1",
                "shard_id": 0,
                "filename": "shard_000000.npz",
                "sha256": "0" * 64,
                "record_count": 1,
                "records": [{"row_index": 1}],
            }
        ),
        encoding="utf-8",
    )

    reconcile_geometry_artifacts(tmp_path, verify_checksums=True)

    assert not sidecar.exists()


def test_reconcile_discards_orphan_geometry_index_for_safe_rebuild(tmp_path):
    record = GeometryRecord(
        atomic_numbers=np.array([6], dtype=np.int64),
        coords=np.zeros((1, 1, 3), dtype=np.float32),
        energies=np.array([np.nan], dtype=np.float32),
        conformer_mask=np.array([True]),
        conformer_source=np.array(["official_dft"]),
        heavy_atom_indices=np.array([0], dtype=np.int64),
        heavy_atom_mapping=np.array([0], dtype=np.int64),
        canonical_smiles="C",
    )
    write_geometry_shard([(1, "C", 3, 0, record)], tmp_path, shard_id=0)
    expected = write_geometry_index(tmp_path)
    (tmp_path / "geometry_index.json").unlink()

    reconcile_geometry_artifacts(tmp_path, verify_checksums=True)

    assert expected["record_count"] == 1
    assert not (tmp_path / "geometry_index.npz").exists()
    assert not (tmp_path / "geometry_index.json").exists()


def test_run_manifest_embeds_the_verified_input_fingerprint(tmp_path):
    input_path = tmp_path / "source.csv"
    selection_path = tmp_path / "selection.csv"
    sdf_path = tmp_path / "train.sdf"
    input_path.write_text("source_index,smiles\n1,C\n", encoding="utf-8")
    selection_path.write_text(
        "source_index,train_ordinal\n1,0\n",
        encoding="utf-8",
    )
    sdf_path.write_bytes(b"official sdf bytes")
    parameters = {"seed": 42, "num_conformers": 5}
    fingerprint = compute_run_fingerprint(
        {
            "input": input_path,
            "manifest": selection_path,
            "official_sdf": sdf_path,
        },
        parameters,
    )
    record = GeometryRecord(
        atomic_numbers=np.array([6], dtype=np.int64),
        coords=np.zeros((1, 1, 3), dtype=np.float32),
        energies=np.array([np.nan], dtype=np.float32),
        conformer_mask=np.array([True]),
        conformer_source=np.array(["official_dft"]),
        heavy_atom_indices=np.array([0], dtype=np.int64),
        heavy_atom_mapping=np.array([0], dtype=np.int64),
        canonical_smiles="C",
    )
    write_geometry_shard([(1, "C", 1, 0, record)], tmp_path, shard_id=0)
    ensure_run_state(tmp_path, fingerprint, resume=False)
    connection = prepare_work_database(
        tmp_path / "geometry_work.sqlite3",
        input_path=input_path,
        smiles_col="smiles",
        source_index_col="source_index",
        sdf_ordinal_col=None,
        manifest_path=selection_path,
        fingerprint_sha256=fingerprint["sha256"],
        expected_run_fingerprint=fingerprint,
        chunk_size=1,
    )
    snapshot = conformer_module._prepare_official_sdf_snapshot(
        tmp_path,
        sdf_path,
        fingerprint["inputs"]["official_sdf"],
    )
    try:
        conformer_module._sync_completed_rows(
            connection,
            tmp_path,
            verify_checksums=True,
        )
        _write_run_metadata(
            tmp_path,
            failure_count=0,
            arguments=Namespace(
                num_conformers=5,
                prune_rms_thresh=0.5,
                seed=42,
                no_optimize=False,
                official_sdf=str(sdf_path),
            ),
            connection=connection,
            requested_count=1,
            official_sdf_snapshot=snapshot,
            run_fingerprint=fingerprint,
            fingerprint_inputs={
                "input": input_path,
                "manifest": selection_path,
                "official_sdf": sdf_path,
            },
            fingerprint_parameters=parameters,
        )
    finally:
        connection.close()

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_fingerprint"] == fingerprint
    assert manifest["run_fingerprint"]["inputs"] == {
        "input": {
            "name": input_path.name,
            "size": input_path.stat().st_size,
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "manifest": {
            "name": selection_path.name,
            "size": selection_path.stat().st_size,
            "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        },
        "official_sdf": {
            "name": sdf_path.name,
            "size": sdf_path.stat().st_size,
            "sha256": hashlib.sha256(sdf_path.read_bytes()).hexdigest(),
        },
    }


def test_run_fingerprint_verification_detects_input_toctou(tmp_path):
    input_path = tmp_path / "source.csv"
    input_path.write_text("smiles\nC\n", encoding="utf-8")
    inputs = {"input": input_path}
    parameters = {"seed": 42}
    fingerprint = compute_run_fingerprint(inputs, parameters)
    input_path.write_text("smiles\nN\n", encoding="utf-8")

    with pytest.raises(ResumeStateError, match="changed during generation"):
        verify_run_fingerprint(inputs, parameters, fingerprint)

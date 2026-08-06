from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.preprocess.build_pcqm_store import (
    AcceptedCandidate,
    GeometryRepository,
    SelectionAccumulator,
    TokenizerSnapshot,
    _audit_pinned_geometry,
    _cleanup_private_snapshot_temps,
    _converge_staged_shards,
    _contract_payload,
    _density_extent_fits,
    _prepare_geometry_snapshot,
    _prepare_selection_snapshot,
    _prepare_tokenizer_snapshot,
    _validate_geometry_run_contract,
    _validate_selection_provenance,
    resolve_tokenizer_snapshot,
    target_quotas,
    validate_resume_indices,
    verify_tokenizer_snapshot,
)
from src.datasets.feature_building import FeatureBuildConfig
from src.molecular.espf_tokenizer import ESPFTokenizer


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometry_run_fingerprint(seed: int = 42) -> dict:
    payload = {
        "inputs": {
            "input": {
                "name": "source.csv",
                "size": 16,
                "sha256": "a" * 64,
            },
            "manifest": None,
            "official_sdf": None,
        },
        "parameters": {
            "schema": "semmol.geometry.v1",
            "smiles_col": "smiles",
            "source_index_col": "source_index",
            "sdf_ordinal_col": None,
            "sdf_energy_property": None,
            "num_conformers": 5,
            "prune_rms_thresh": 0.5,
            "seed": seed,
            "optimize": True,
            "num_workers": 1,
            "worker_chunksize": 32,
            "shard_size": 1000,
            "table_chunk_size": 100_000,
            "verify_checksums": True,
        },
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        **payload,
    }


def test_selection_accumulator_backfills_failures_and_keeps_targets_nested() -> None:
    accumulator = SelectionAccumulator((10, 20), n_bins=2)

    for gap_bin in range(2):
        for selection_rank in range(20):
            source_index = gap_bin * 100 + selection_rank
            if source_index in {1, 3, 104}:
                continue
            accumulator.accept(
                source_index=source_index,
                gap_bin=gap_bin,
                selection_rank=selection_rank,
                record_index=accumulator.accepted_count,
            )
            if accumulator.bin_is_full(gap_bin):
                break

    accumulator.validate_complete()
    views = accumulator.views()

    assert len(views[10]["record_index"]) == 10
    assert len(views[20]["record_index"]) == 20
    assert set(views[10]["source_index"]).issubset(
        set(views[20]["source_index"])
    )
    assert views[10]["source_index"][:5].tolist() == [0, 2, 4, 5, 6]


def test_target_quotas_match_total_for_non_divisible_sizes() -> None:
    assert target_quotas(23, 10) == (3, 3, 3, 2, 2, 2, 2, 2, 2, 2)


def test_resume_index_validation_rejects_non_contiguous_record_indices() -> None:
    accepted = [
        AcceptedCandidate(0, 7, 0, 0, 0),
        AcceptedCandidate(2, 8, 0, 1, 1),
    ]

    with pytest.raises(ValueError, match="contiguous"):
        validate_resume_indices(accepted)


def test_resume_index_validation_accepts_exactly_contiguous_indices() -> None:
    accepted = [
        AcceptedCandidate(0, 7, 0, 0, 0),
        AcceptedCandidate(1, 8, 0, 1, 1),
    ]

    result = validate_resume_indices(accepted)

    np.testing.assert_array_equal(
        result["record_index"],
        np.array([0, 1], dtype=np.int64),
    )


def test_density_extent_preflight_uses_fixed_grid_physical_coverage() -> None:
    coords = np.array(
        [[[0.0, 0.0, 0.0], [7.4, 0.0, 0.0]]],
        dtype=np.float32,
    )
    assert _density_extent_fits(
        coords,
        np.array([True]),
        conformer_index=0,
        grid_size=32,
        spacing=0.5,
        padding=4.0,
    )
    coords[0, 1, 0] = 7.6
    assert not _density_extent_fits(
        coords,
        np.array([True]),
        conformer_index=0,
        grid_size=32,
        spacing=0.5,
        padding=4.0,
    )


def test_selection_provenance_requires_exact_schema_and_generation(tmp_path) -> None:
    metadata_path = tmp_path / "selection.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "semmol.pcqm_selection.v2",
                "generation_id": "generation-a",
                "input": {
                    "official_split_column": "official_split",
                    "official_split_counts": {"train": 10},
                    "integrity": {
                        "size_bytes": 123,
                        "sha256": "a" * 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema"):
        _validate_selection_provenance(
            metadata_path,
            allow_unverified_split=False,
            expected_generation_id="generation-a",
        )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["schema"] = "semmol.pcqm_selection.v1"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="generation"):
        _validate_selection_provenance(
            metadata_path,
            allow_unverified_split=False,
            expected_generation_id="generation-b",
        )


def test_selection_provenance_requires_source_integrity(tmp_path) -> None:
    metadata_path = tmp_path / "selection.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "semmol.pcqm_selection.v1",
                "generation_id": "generation-a",
                "input": {
                    "official_split_column": "official_split",
                    "official_split_counts": {"train": 10},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source integrity"):
        _validate_selection_provenance(
            metadata_path,
            allow_unverified_split=False,
            expected_generation_id="generation-a",
        )


def test_resume_convergence_removes_index_without_published_lmdb(tmp_path) -> None:
    orphan = tmp_path / "build-index-000000.npz"
    with orphan.open("wb") as stream:
        np.savez_compressed(stream, record_index=np.array([0], dtype=np.int64))

    _converge_staged_shards(tmp_path)

    assert not orphan.exists()


def test_geometry_repository_keeps_compact_index_and_lazy_sidecars(
    tmp_path,
) -> None:
    run_fingerprint = _geometry_run_fingerprint()
    artifact = tmp_path / "shard_000000.npz"
    with artifact.open("wb") as stream:
        np.savez_compressed(stream, marker=np.array([1], dtype=np.int8))
    artifact_sha = _sha256(artifact)
    sidecar = {
        "schema": "semmol.geometry.v1",
        "shard_id": 0,
        "filename": artifact.name,
        "sha256": artifact_sha,
        "record_count": 1,
        "records": [
            {
                "key": "r000000",
                "row_index": 3,
                "source_index": 7,
                "sdf_ordinal": None,
                "train_ordinal": None,
            }
        ],
    }
    (tmp_path / "shard_000000.json").write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )
    index_path = tmp_path / "geometry_index.npz"
    with index_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            source_index=np.array([7], dtype=np.int64),
            row_index=np.array([3], dtype=np.int64),
            sdf_ordinal=np.array([-1], dtype=np.int64),
            shard_id=np.array([0], dtype=np.int32),
            record_ordinal=np.array([0], dtype=np.int32),
        )
    index_sha = _sha256(index_path)
    (tmp_path / "geometry_index.json").write_text(
        json.dumps(
            {
                "schema": "semmol.geometry_index.v2",
                "filename": index_path.name,
                "sha256": index_sha,
                "record_count": 1,
                "sorted_by": ["source_index", "row_index"],
                "lookup": (
                    "numpy.searchsorted(source_index, requested_source_index)"
                ),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "semmol.geometry.v1",
                "shards": [
                    {
                        "shard_id": 0,
                        "filename": artifact.name,
                        "sha256": artifact_sha,
                        "record_count": 1,
                    }
                ],
                "successful_records": 1,
                "failed_records": 0,
                "run_fingerprint": run_fingerprint,
                "source_index": {
                    "artifact": index_path.name,
                    "metadata": "geometry_index.json",
                    "sha256": index_sha,
                    "record_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_state.json").write_text(
        json.dumps(
            {
                "schema": "semmol.geometry_run_state.v1",
                "fingerprint": run_fingerprint,
            }
        ),
        encoding="utf-8",
    )

    repository = GeometryRepository(
        tmp_path,
        verify_checksums=False,
        validate_index_inventory=True,
    )
    try:
        assert repository.source_indices.dtype == np.int64
        assert repository.shard_ids.dtype == np.int32
        assert repository.record_ordinals.dtype == np.int32
        assert not repository._records
    finally:
        repository.close()
    geometry_contract = {
        "manifest_sha256": _sha256(tmp_path / "manifest.json"),
        "run_state_sha256": _sha256(tmp_path / "run_state.json"),
        "index_metadata_sha256": _sha256(
            tmp_path / "geometry_index.json"
        ),
        "index_artifact_sha256": index_sha,
    }
    _audit_pinned_geometry(
        tmp_path,
        geometry_contract,
        max_open_shards=1,
    )
    snapshot_path = tmp_path.parent / f"{tmp_path.name}-geometry-snapshot"
    _prepare_geometry_snapshot(
        tmp_path,
        snapshot_path,
        geometry_contract,
        max_open_shards=1,
    )
    artifact.write_bytes(b"mutated after private snapshot")
    _audit_pinned_geometry(
        snapshot_path,
        geometry_contract,
        max_open_shards=1,
    )


def test_geometry_run_contract_rejects_manifest_run_state_mismatch() -> None:
    manifest = {
        "schema": "semmol.geometry.v1",
        "run_fingerprint": _geometry_run_fingerprint(seed=42),
    }
    run_state = {
        "schema": "semmol.geometry_run_state.v1",
        "fingerprint": _geometry_run_fingerprint(seed=43),
    }

    with pytest.raises(RuntimeError, match="manifest/run-state fingerprint"):
        _validate_geometry_run_contract(manifest, run_state)


def test_geometry_run_contract_rejects_json_type_only_mismatch() -> None:
    manifest_fingerprint = _geometry_run_fingerprint(seed=42)
    run_state_fingerprint = json.loads(json.dumps(manifest_fingerprint))
    run_state_fingerprint["parameters"]["seed"] = 42.0
    manifest = {
        "schema": "semmol.geometry.v1",
        "run_fingerprint": manifest_fingerprint,
    }
    run_state = {
        "schema": "semmol.geometry_run_state.v1",
        "fingerprint": run_state_fingerprint,
    }

    with pytest.raises(RuntimeError, match="manifest/run-state fingerprint"):
        _validate_geometry_run_contract(manifest, run_state)


def test_geometry_run_contract_requires_input_size_and_sha256() -> None:
    fingerprint = _geometry_run_fingerprint()
    del fingerprint["inputs"]["input"]["size"]
    manifest = {
        "schema": "semmol.geometry.v1",
        "run_fingerprint": fingerprint,
    }
    run_state = {
        "schema": "semmol.geometry_run_state.v1",
        "fingerprint": fingerprint,
    }

    with pytest.raises(RuntimeError, match="input descriptor"):
        _validate_geometry_run_contract(manifest, run_state)


def test_pcqm_build_contract_pins_geometry_run_state(tmp_path) -> None:
    selection_manifest = tmp_path / "selection.parquet"
    selection_metadata = tmp_path / "selection.json"
    selection_manifest.write_bytes(b"selection")
    selection_metadata.write_text("{}", encoding="utf-8")
    geometry_dir = tmp_path / "geometry"
    geometry_dir.mkdir()
    run_fingerprint = _geometry_run_fingerprint()
    (geometry_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "semmol.geometry.v1",
                "run_fingerprint": run_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    run_state_path = geometry_dir / "run_state.json"
    run_state_path.write_text(
        json.dumps(
            {
                "schema": "semmol.geometry_run_state.v1",
                "fingerprint": run_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    index_path = geometry_dir / "geometry_index.npz"
    index_path.write_bytes(b"index")
    (geometry_dir / "geometry_index.json").write_text(
        json.dumps(
            {
                "schema": "semmol.geometry_index.v2",
                "filename": index_path.name,
                "sha256": _sha256(index_path),
                "record_count": 0,
                "sorted_by": ["source_index", "row_index"],
                "lookup": (
                    "numpy.searchsorted(source_index, requested_source_index)"
                ),
            }
        ),
        encoding="utf-8",
    )

    contract = _contract_payload(
        selection_manifest=selection_manifest,
        selection_manifest_sha256=_sha256(selection_manifest),
        selection_metadata=selection_metadata,
        selection_metadata_sha256=_sha256(selection_metadata),
        tokenizer_snapshot=TokenizerSnapshot(
            root=tmp_path,
            load_path=tmp_path / "tokenizer",
            artifact_sha256="b" * 64,
            vocab_size=32,
        ),
        geometry_dir=geometry_dir,
        target_sizes=(10,),
        n_bins=2,
        records_per_shard=5,
        map_size=1_000_000,
        compression_level=3,
        feature_config=FeatureBuildConfig(),
    )

    assert contract["geometry"]["run_state_sha256"] == _sha256(run_state_path)


def test_selection_snapshot_is_private_and_hash_pinned(tmp_path) -> None:
    source = tmp_path / "selection.parquet"
    snapshot = tmp_path / ".selection.snapshot.parquet"
    source.write_bytes(b"selection version A")
    expected_sha256 = _sha256(source)

    prepared = _prepare_selection_snapshot(
        source,
        snapshot,
        expected_sha256=expected_sha256,
    )
    source.write_bytes(b"selection version B")

    assert prepared == snapshot
    assert snapshot.read_bytes() == b"selection version A"
    assert _sha256(snapshot) == expected_sha256


def test_selection_snapshot_rejects_bytes_changed_after_contract_pin(
    tmp_path,
) -> None:
    source = tmp_path / "selection.parquet"
    snapshot = tmp_path / ".selection.snapshot.parquet"
    source.write_bytes(b"selection version A")
    expected_sha256 = _sha256(source)
    source.write_bytes(b"selection version B")

    with pytest.raises(RuntimeError, match="changed while creating"):
        _prepare_selection_snapshot(
            source,
            snapshot,
            expected_sha256=expected_sha256,
        )

    assert not snapshot.exists()


def test_changed_selection_snapshot_poison_blocks_resume(tmp_path) -> None:
    source = tmp_path / "selection.parquet"
    snapshot = tmp_path / ".selection.snapshot.parquet"
    source.write_bytes(b"selection version A")
    expected_sha256 = _sha256(source)
    _prepare_selection_snapshot(
        source,
        snapshot,
        expected_sha256=expected_sha256,
    )
    snapshot.write_bytes(b"selection version B")

    with pytest.raises(RuntimeError, match="differs from the build contract"):
        _prepare_selection_snapshot(
            source,
            snapshot,
            expected_sha256=expected_sha256,
        )
    with pytest.raises(RuntimeError, match="poisoned"):
        _prepare_selection_snapshot(
            source,
            snapshot,
            expected_sha256=expected_sha256,
        )


def test_tokenizer_generation_snapshot_is_private_and_hash_pinned(
    tmp_path,
) -> None:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer = ESPFTokenizer.train(
        ["CO", "CCO", "C[NH3+]"],
        min_frequency=1,
        max_merges=4,
    )
    tokenizer.save_pretrained(tokenizer_root)
    original = resolve_tokenizer_snapshot(tokenizer_root)
    private_path = tmp_path / ".tokenizer-generation.snapshot"

    private = _prepare_tokenizer_snapshot(original, private_path)
    original_config = original.load_path / "tokenizer_config.json"
    original_config.write_text(
        original_config.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    assert private.load_path == private_path
    verify_tokenizer_snapshot(private)


def test_changed_tokenizer_snapshot_poison_blocks_resume(tmp_path) -> None:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer = ESPFTokenizer.train(
        ["CO", "CCO"],
        min_frequency=1,
        max_merges=2,
    )
    tokenizer.save_pretrained(tokenizer_root)
    original = resolve_tokenizer_snapshot(tokenizer_root)
    private_path = tmp_path / ".tokenizer-generation.snapshot"
    _prepare_tokenizer_snapshot(original, private_path)
    (private_path / "vocab.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="differs from the build contract"):
        _prepare_tokenizer_snapshot(original, private_path)
    with pytest.raises(RuntimeError, match="poisoned"):
        _prepare_tokenizer_snapshot(original, private_path)


def test_private_snapshot_temp_cleanup_is_targeted(tmp_path) -> None:
    selection_temp = (
        tmp_path / "..selection-manifest.snapshot.parquet.tmp-dead"
    )
    geometry_temp = tmp_path / "..geometry-input.snapshot.tmp-dead"
    tokenizer_temp = tmp_path / "..tokenizer-generation.snapshot.tmp-dead"
    poison_temp = (
        tmp_path / "..geometry-snapshot.poisoned.json.tmp-dead"
    )
    deletion_temp = tmp_path / "..geometry-input.snapshot.delete-dead"
    unrelated = tmp_path / ".keep-me.tmp-dead"
    selection_temp.write_bytes(b"partial")
    geometry_temp.mkdir()
    tokenizer_temp.mkdir()
    poison_temp.write_bytes(b"partial")
    deletion_temp.mkdir()
    unrelated.write_bytes(b"keep")

    _cleanup_private_snapshot_temps(tmp_path)

    assert not selection_temp.exists()
    assert not geometry_temp.exists()
    assert not tokenizer_temp.exists()
    assert not poison_temp.exists()
    assert not deletion_temp.exists()
    assert unrelated.read_bytes() == b"keep"

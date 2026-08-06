import json

import numpy as np
import pytest

from scripts.preprocess.generate_3d_conformer import write_geometry_shard
from scripts.preprocess.generate_qm_density import (
    ArtifactIntegrityError,
    DensityConfigError,
    _CompletedRowIndex,
    _FailureJournal,
    _cleanup_geometry_snapshot_transients,
    _invalidate_statistics_marker,
    _prepare_geometry_snapshot,
    _record_ordinal,
    iter_geometry_records,
    _load_yaml_config,
    reconcile_density_artifacts,
    write_density_shard,
)
from src.molecular.electron_density import build_promolecular_density
from src.molecular.geometry import GeometryRecord
from src.molecular.rdkit_utils import smiles_hash


GEOMETRY_ARTIFACT = "shard_000000.npz"
GEOMETRY_ARTIFACT_SHA256 = "a" * 64
GEOMETRY_KEY = "r000000"
GEOMETRY_PAYLOAD_SHA256 = "b" * 64


def _density_record(
    row_index,
    smiles,
    source_index,
    sdf_ordinal,
    result,
):
    return (
        row_index,
        smiles,
        source_index,
        sdf_ordinal,
        GEOMETRY_ARTIFACT,
        GEOMETRY_ARTIFACT_SHA256,
        GEOMETRY_KEY,
        GEOMETRY_PAYLOAD_SHA256,
        result,
    )


def test_density_shard_preserves_grid_metadata_contract(tmp_path):
    result = build_promolecular_density(
        np.array([1], dtype=np.int64),
        np.zeros((1, 3), dtype=np.float32),
        grid_size=16,
        spacing=0.5,
        box_padding=2.0,
    )

    metadata = write_density_shard(
        [_density_record(4, "C", 4, None, result)],
        tmp_path,
        shard_id=1,
    )

    with np.load(tmp_path / metadata["filename"], allow_pickle=False) as stored:
        assert stored["r000000__grid"].shape == (16, 16, 16)
        assert stored["r000000__origin"].shape == (3,)
        assert stored["r000000__spacing"].item() == 0.5
        assert stored["r000000__electron_count"].item() == 1.0
        stored_integral = float(
            stored["r000000__grid"].sum(dtype=np.float64)
            * float(stored["r000000__spacing"]) ** 3
        )
        assert (
            stored["r000000__integrated_electrons"].item()
            == stored_integral
        )
        assert (
            stored[
                "r000000__prequantization_integrated_electrons"
            ].item()
            == result.integrated_electrons
        )
        assert not stored["r000000__overflow"].item()
        assert (
            stored["r000000__normalization_requested"].item()
            == "discrete_electron_count"
        )
        assert (
            stored["r000000__normalization_applied"].item()
            == "discrete_electron_count"
        )
    assert (
        metadata["records"][0]["normalization_requested"]
        == "discrete_electron_count"
    )
    assert (
        metadata["records"][0]["normalization_applied"]
        == "discrete_electron_count"
    )


def test_density_shard_preserves_geometry_identity(tmp_path):
    result = build_promolecular_density(
        np.array([6], dtype=np.int64),
        np.zeros((1, 3), dtype=np.float32),
        grid_size=16,
        spacing=0.5,
        box_padding=2.0,
    )

    metadata = write_density_shard(
        [_density_record(4, "C", 104, 7, result)],
        tmp_path,
        shard_id=0,
    )

    entry = metadata["records"][0]
    assert entry["row_index"] == 4
    assert entry["source_index"] == 104
    assert entry["sdf_ordinal"] == 7
    assert entry["train_ordinal"] == 7
    assert entry["geometry_artifact"] == GEOMETRY_ARTIFACT
    assert (
        entry["geometry_artifact_sha256"]
        == GEOMETRY_ARTIFACT_SHA256
    )
    assert entry["geometry_key"] == GEOMETRY_KEY
    assert entry["geometry_payload_sha256"] == GEOMETRY_PAYLOAD_SHA256


def test_density_shard_normalizes_minus_one_ordinal_to_none(tmp_path):
    result = build_promolecular_density(
        np.array([1], dtype=np.int64),
        np.zeros((1, 3), dtype=np.float32),
        grid_size=8,
        spacing=0.5,
        box_padding=2.0,
        strict=False,
        discrete_normalize=True,
    )

    metadata = write_density_shard(
        [_density_record(6, "[H]", 106, -1, result)],
        tmp_path,
        shard_id=0,
    )

    entry = metadata["records"][0]
    assert entry["sdf_ordinal"] is None
    assert entry["train_ordinal"] is None
    assert entry["overflow"] is True
    assert entry["normalization_requested"] == "discrete_electron_count"
    assert entry["normalization_applied"] == "continuous_gaussian"


def test_density_shard_rejects_ordinal_less_than_minus_one(tmp_path):
    result = build_promolecular_density(
        np.array([1], dtype=np.int64),
        np.zeros((1, 3), dtype=np.float32),
        grid_size=16,
        spacing=0.5,
        box_padding=2.0,
    )

    with pytest.raises(ValueError, match="sdf_ordinal"):
        write_density_shard(
            [_density_record(6, "[H]", 106, -2, result)],
            tmp_path,
            shard_id=0,
        )


def test_reconcile_recovers_published_density_without_sidecar(tmp_path):
    result = build_promolecular_density(
        np.array([1], dtype=np.int64),
        np.zeros((1, 3), dtype=np.float32),
        grid_size=16,
        spacing=0.5,
        box_padding=2.0,
    )
    expected = write_density_shard(
        [_density_record(2, "[H]", 12, -1, result)],
        tmp_path,
        shard_id=3,
    )
    (tmp_path / "density_000003.json").unlink()

    reconcile_density_artifacts(tmp_path)

    recovered = json.loads(
        (tmp_path / "density_000003.json").read_text(encoding="utf-8")
    )
    assert recovered["sha256"] == expected["sha256"]
    assert recovered["records"][0]["source_index"] == 12
    assert (
        recovered["records"][0]["geometry_artifact_sha256"]
        == GEOMETRY_ARTIFACT_SHA256
    )
    with _CompletedRowIndex(tmp_path / ".completed.sqlite3") as index:
        index.populate(tmp_path, verify_checksums=True)
        assert index.count == 1


def test_ordinal_aliases_are_compared_after_canonicalization():
    assert (
        _record_ordinal(
            {"sdf_ordinal": -1, "train_ordinal": None},
            context="record",
        )
        is None
    )
    with pytest.raises(ValueError, match="train_ordinal"):
        _record_ordinal(
            {"sdf_ordinal": None, "train_ordinal": -2},
            context="record",
        )
    with pytest.raises(ArtifactIntegrityError, match="conflicting"):
        _record_ordinal(
            {"sdf_ordinal": None, "train_ordinal": 7},
            context="record",
        )
    with pytest.raises(ArtifactIntegrityError, match="conflicting"):
        _record_ordinal(
            {"sdf_ordinal": -1, "train_ordinal": 7},
            context="record",
        )


def test_completed_index_rejects_duplicate_source_identity(tmp_path):
    result = build_promolecular_density(
        np.array([1], dtype=np.int64),
        np.zeros((1, 3), dtype=np.float32),
        grid_size=16,
        spacing=0.5,
        box_padding=2.0,
    )
    write_density_shard(
        [_density_record(1, "[H]", 99, 0, result)],
        tmp_path,
        shard_id=0,
    )
    write_density_shard(
        [
            (
                2,
                "[H]",
                99,
                1,
                "shard_000001.npz",
                "c" * 64,
                "r000000",
                "d" * 64,
                result,
            )
        ],
        tmp_path,
        shard_id=1,
    )

    with _CompletedRowIndex(tmp_path / ".completed.sqlite3") as index:
        with pytest.raises(ArtifactIntegrityError, match="source_index"):
            index.populate(tmp_path, verify_checksums=True)


def test_geometry_stream_preserves_source_and_train_ordinal(tmp_path):
    geometry_dir = tmp_path / "geometry"
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
    write_geometry_shard(
        [(5, "C", 105, 9, record)],
        geometry_dir,
        shard_id=0,
    )

    with _CompletedRowIndex(tmp_path / ".completed.sqlite3") as index:
        items = list(
            iter_geometry_records(
                geometry_dir,
                verify_checksums=True,
                completed_rows=index,
            )
        )

    assert items[0][:4] == (5, "C", 105, 9)
    assert items[0][4] == "shard_000000.npz"
    assert items[0][5] == json.loads(
        (geometry_dir / "shard_000000.json").read_text(encoding="utf-8")
    )["sha256"]
    assert items[0][6] == "r000000"
    assert len(items[0][7]) == 64
    assert isinstance(items[0][8], GeometryRecord)


def test_completed_index_rejects_geometry_payload_provenance_mismatch(tmp_path):
    result = build_promolecular_density(
        np.array([1], dtype=np.int64),
        np.zeros((1, 3), dtype=np.float32),
        grid_size=16,
        spacing=0.5,
        box_padding=2.0,
    )
    write_density_shard(
        [_density_record(1, "[H]", 99, None, result)],
        tmp_path,
        shard_id=0,
    )

    with _CompletedRowIndex(tmp_path / ".completed.sqlite3") as index:
        index.populate(tmp_path, verify_checksums=True)
        index.register_geometry_rows(
            [
                (
                    1,
                    99,
                    None,
                    smiles_hash("[H]"),
                    GEOMETRY_ARTIFACT,
                    GEOMETRY_ARTIFACT_SHA256,
                    GEOMETRY_KEY,
                    "c" * 64,
                )
            ]
        )
        with pytest.raises(ArtifactIntegrityError, match="provenance"):
            index.validate_completed_coverage()


def test_incomplete_geometry_snapshot_is_poisoned(tmp_path):
    output_dir = tmp_path / "density"
    snapshot_dir = output_dir / ".geometry_snapshot"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "density_000000.json").write_text(
        "{}",
        encoding="utf-8",
    )
    fingerprint = {
        "sha256": "a" * 64,
        "inputs": {},
        "parameters": {},
    }

    with pytest.raises(ArtifactIntegrityError, match="snapshot"):
        _prepare_geometry_snapshot(
            tmp_path / "geometry",
            output_dir,
            fingerprint,
        )

    assert (output_dir / ".geometry_snapshot.poison.json").is_file()


def test_first_snapshot_build_failure_is_retryable_before_derivation(tmp_path):
    output_dir = tmp_path / "density"
    output_dir.mkdir()
    fingerprint = {
        "sha256": "a" * 64,
        "inputs": {},
        "parameters": {},
    }

    with pytest.raises(ArtifactIntegrityError, match="fingerprint"):
        _prepare_geometry_snapshot(
            tmp_path / "geometry",
            output_dir,
            fingerprint,
        )

    assert not (output_dir / ".geometry_snapshot.building").exists()
    assert not (output_dir / ".geometry_snapshot.poison.json").exists()


def test_private_snapshot_transients_converge_without_poison(tmp_path):
    building = tmp_path / ".geometry_snapshot.building"
    tombstone = tmp_path / (
        ".geometry_snapshot.delete-"
        "0123456789abcdef0123456789abcdef"
    )
    building.mkdir()
    tombstone.mkdir()
    (building / "partial").write_bytes(b"partial")
    (tombstone / "retired").write_bytes(b"retired")

    _cleanup_geometry_snapshot_transients(tmp_path)

    assert not building.exists()
    assert not tombstone.exists()
    assert not (tmp_path / ".geometry_snapshot.poison.json").exists()


def test_resume_invalidates_statistics_before_other_recovery_mutations(
    tmp_path,
):
    statistics = tmp_path / "statistics.json"
    ready = tmp_path / ".statistics.ready.json"
    statistics.write_text('{"complete":true}\n', encoding="utf-8")
    ready.write_text('{"complete":false}\n', encoding="utf-8")

    _invalidate_statistics_marker(tmp_path)

    assert not statistics.exists()
    assert not ready.exists()


def test_failure_journal_replaces_target_only_after_success(tmp_path):
    target = tmp_path / "failures.jsonl"
    target.write_text('{"old":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="interrupt"):
        with _FailureJournal(tmp_path) as journal:
            journal.record({"row_index": 1, "message": "new"})
            raise RuntimeError("interrupt")
    assert target.read_text(encoding="utf-8") == '{"old":true}\n'

    with _FailureJournal(tmp_path) as journal:
        journal.record({"row_index": 2, "message": "final"})
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [{"message": "final", "row_index": 2}]


def test_yaml_configuration_rejects_unknown_density_keys(tmp_path):
    config = tmp_path / "qm.yaml"
    config.write_text(
        "\n".join(
            [
                "grid:",
                "  size: 32",
                "  resolution: 0.5",
                "  box_padding: 4.0",
                "density:",
                "  method: gaussian_promol",
                "  normalize: true",
                "  unsupported: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DensityConfigError):
        _load_yaml_config(str(config))


@pytest.mark.parametrize(
    "extra_line",
    ["  atomic_cutoff: 4.0", "  element_z_max: 118"],
)
def test_yaml_rejects_parameters_the_density_implementation_does_not_use(
    tmp_path,
    extra_line,
):
    config = tmp_path / "qm.yaml"
    config.write_text(
        "\n".join(
            [
                "grid:",
                "  size: 32",
                "  resolution: 0.75",
                "  box_padding: 4.0",
                "density:",
                "  method: promolecular_gaussian",
                "  normalize: true",
                extra_line,
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DensityConfigError, match="unknown density"):
        _load_yaml_config(str(config))

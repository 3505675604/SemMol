from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.datasets.feature_building import (
    FeatureBuildConfig,
    FeatureBuildError,
    MultimodalFeatureBuilder,
    tokenizer_artifact_sha256,
)
from src.molecular.espf_tokenizer import ESPFTokenizer
from src.molecular.espf_tokenizer import TokenizerFormatError
from src.molecular.geometry import GeometryInputError, generate_conformers


def _tokenizer(tmp_path) -> ESPFTokenizer:
    tokenizer = ESPFTokenizer.train(
        ["CO", "CCO", "C[NH3+]"],
        min_frequency=1,
        max_merges=4,
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    return ESPFTokenizer.from_pretrained(tmp_path / "tokenizer")


def test_tokenizer_artifact_hash_rejects_a_changed_active_artifact(tmp_path) -> None:
    tokenizer = _tokenizer(tmp_path)
    directory = tmp_path / "tokenizer"
    assert tokenizer.generation_id is not None
    config_path = (
        directory
        / "generations"
        / tokenizer.generation_id
        / "tokenizer_config.json"
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TokenizerFormatError, match="hash mismatch"):
        tokenizer_artifact_sha256(directory)


def test_feature_builder_uses_one_canonical_atom_order_for_graph_and_geometry(
    tmp_path,
) -> None:
    tokenizer = _tokenizer(tmp_path)
    geometry = generate_conformers(
        "OC",
        num_conformers=1,
        seed=7,
        on_invalid="raise",
    )
    assert geometry is not None
    builder = MultimodalFeatureBuilder(
        tokenizer,
        FeatureBuildConfig(
            max_smiles_length=32,
        ),
    )

    record = builder.build_record(
        smiles="OC",
        source_index=9,
        sample_namespace="unit",
        geometry=geometry,
    )

    assert record["smiles"] == "CO"
    graph_atomic_numbers = builder.graph_atomic_numbers(record["smiles"])
    heavy = record["geometry"]["heavy_atom_indices"]
    np.testing.assert_array_equal(
        record["geometry"]["atomic_numbers"][heavy],
        graph_atomic_numbers,
    )
    assert isinstance(record["geometry"]["sources"], list)
    assert "density" not in record


def test_feature_builder_rejects_geometry_from_a_different_molecule(tmp_path) -> None:
    tokenizer = _tokenizer(tmp_path)
    wrong_geometry = generate_conformers(
        "CCC",
        num_conformers=1,
        seed=11,
        on_invalid="raise",
    )
    assert wrong_geometry is not None
    builder = MultimodalFeatureBuilder(
        tokenizer,
        FeatureBuildConfig(
            max_smiles_length=32,
        ),
    )

    with pytest.raises(FeatureBuildError, match="geometry_alignment"):
        builder.build_record(
            smiles="CO",
            source_index=4,
            sample_namespace="unit",
            geometry=wrong_geometry,
        )


def test_default_builder_selects_one_lowest_energy_conformer_from_supplied_ensemble(tmp_path):
    geometry = generate_conformers("CO", num_conformers=1, seed=3, on_invalid="raise")
    assert geometry is not None
    ensemble = replace(
        geometry,
        coords=np.stack((geometry.coords[0], geometry.coords[0] + 1.0)),
        energies=np.array([3.0, 1.0], dtype=np.float32),
        conformer_mask=np.ones(2, dtype=np.bool_),
        conformer_source=np.array(["etkdg_mmff", "etkdg_mmff"]),
    )
    builder = MultimodalFeatureBuilder(_tokenizer(tmp_path), FeatureBuildConfig())
    result = builder.build_record(smiles="CO", source_index=1, sample_namespace="unit", geometry=ensemble)
    assert result["geometry"]["coords"].shape[0] == 1
    assert result["geometry"]["energies"].tolist() == [1.0]
    np.testing.assert_array_equal(result["geometry"]["coords"][0], ensemble.coords[1])


def test_feature_builder_rejects_overlength_smiles_without_truncating(tmp_path) -> None:
    tokenizer = _tokenizer(tmp_path)
    builder = MultimodalFeatureBuilder(
        tokenizer,
        FeatureBuildConfig(
            max_smiles_length=3,
        ),
    )

    with pytest.raises(FeatureBuildError, match="untruncated token length"):
        builder.build_record(
            smiles="COCOCOCOCO",
            source_index=4,
            sample_namespace="unit",
        )


def test_alignment_rejects_repeated_element_position_permutation() -> None:
    geometry = generate_conformers(
        "CCN",
        num_conformers=1,
        seed=13,
        on_invalid="raise",
    )
    assert geometry is not None
    with pytest.raises(GeometryInputError, match="canonically cover"):
        replace(
            geometry,
            heavy_atom_indices=np.array([1, 0, 2], dtype=np.int64),
        )

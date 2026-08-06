from __future__ import annotations

import hashlib

import pytest

from scripts.preprocess.build_moleculenet_store import (
    _failure_key,
    ordered_valid_source_indices,
    validate_split_content_contract,
    validate_split_contract,
    validate_split_partition,
)
from src.datasets.scaffold_split import generate_scaffold


def _content_audit(
    smiles_by_source,
    split_indices,
    invalid,
):
    scaffold_hashes = {}
    scaffold_members = {}
    for split_name, source_indices in split_indices.items():
        for source_index in source_indices:
            scaffold = generate_scaffold(smiles_by_source[source_index])
            assert scaffold is not None
            scaffold_hash = hashlib.sha256(
                scaffold.encode("utf-8")
            ).hexdigest()
            previous = scaffold_hashes.setdefault(
                scaffold_hash,
                split_name,
            )
            assert previous == split_name
            scaffold_members.setdefault(scaffold_hash, []).append(source_index)
    return {
        "scaffold_hashes": scaffold_hashes,
        "scaffold_members": scaffold_members,
        "invalid": invalid,
    }


def test_moleculenet_store_order_preserves_original_source_indices() -> None:
    split_indices = {
        "train": [9, 2],
        "valid": [7],
        "test": [4, 12],
    }

    assert ordered_valid_source_indices(split_indices) == [2, 4, 7, 9, 12]


def test_split_partition_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_split_partition(
            {"train": [1, 2], "valid": [2], "test": [3]},
            available_source_indices={1, 2, 3},
        )


def test_split_partition_rejects_unknown_original_rows() -> None:
    with pytest.raises(ValueError, match="not present"):
        validate_split_partition(
            {"train": [1], "valid": [2], "test": [99]},
            available_source_indices={1, 2, 3},
        )


def test_split_contract_rejects_wrong_dataset_name() -> None:
    with pytest.raises(ValueError, match="dataset_name"):
        validate_split_contract(
            metadata={
                "dataset_name": "tox21",
                "invalid": [],
            },
            expected_dataset_name="bbbp",
            split_indices={"train": [0], "valid": [1], "test": [2]},
            available_source_indices={0, 1, 2},
        )


def test_split_contract_requires_assigned_and_invalid_to_cover_raw_rows() -> None:
    with pytest.raises(ValueError, match="cover"):
        validate_split_contract(
            metadata={
                "dataset_name": "bbbp",
                "invalid": [],
            },
            expected_dataset_name="bbbp",
            split_indices={"train": [0], "valid": [1], "test": []},
            available_source_indices={0, 1, 2},
        )

    validate_split_contract(
        metadata={
            "dataset_name": "bbbp",
            "invalid": [{"source_index": 2, "reason": "invalid_smiles"}],
        },
        expected_dataset_name="bbbp",
        split_indices={"train": [0], "valid": [1], "test": []},
        available_source_indices={0, 1, 2},
    )


def test_failure_key_deduplicates_same_source_and_stage() -> None:
    first = {"source_index": 7, "stage": "density", "message": "one"}
    second = {"source_index": 7, "stage": "density", "message": "two"}
    assert _failure_key(first) == _failure_key(second)


def test_split_content_contract_binds_scaffolds_to_current_smiles() -> None:
    split_indices = {"train": [0], "valid": [1], "test": []}
    smiles = {0: "CC", 1: "CCC", 2: "not-a-smiles"}
    metadata = _content_audit(
        smiles,
        split_indices,
        [{"source_index": 2, "reason": "invalid_smiles"}],
    )

    validate_split_content_contract(
        metadata=metadata,
        split_indices=split_indices,
        smiles_by_source=smiles,
    )

    stale_smiles = {**smiles, 0: "c1ccccc1"}
    with pytest.raises(ValueError, match="scaffold"):
        validate_split_content_contract(
            metadata=metadata,
            split_indices=split_indices,
            smiles_by_source=stale_smiles,
        )


def test_split_content_contract_rejects_invalid_row_that_became_valid() -> None:
    split_indices = {"train": [0], "valid": [1], "test": []}
    original_smiles = {0: "CC", 1: "CCC", 2: "not-a-smiles"}
    metadata = _content_audit(
        original_smiles,
        split_indices,
        [{"source_index": 2, "reason": "invalid_smiles"}],
    )

    with pytest.raises(ValueError, match="became valid"):
        validate_split_content_contract(
            metadata=metadata,
            split_indices=split_indices,
            smiles_by_source={**original_smiles, 2: "CCO"},
        )

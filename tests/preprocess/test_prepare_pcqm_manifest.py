from __future__ import annotations

import numpy as np
import pytest

from scripts.preprocess.prepare_pcqm_manifest import (
    SPLIT_NAMES,
    build_split_lookup,
    normalize_pcqm_item,
)


def test_build_split_lookup_maps_train_ordinal_without_reordering_source_indices() -> None:
    split_lookup, train_ordinal = build_split_lookup(
        dataset_size=6,
        split_indices={
            "train": np.array([4, 1, 5]),
            "valid": np.array([0]),
            "test-dev": np.array([3]),
            "test-challenge": np.array([2]),
        },
    )

    assert [SPLIT_NAMES[int(code)] for code in split_lookup] == [
        "valid",
        "train",
        "test-challenge",
        "test-dev",
        "train",
        "train",
    ]
    assert train_ordinal.tolist() == [-1, 1, -1, -1, 0, 2]


def test_build_split_lookup_rejects_duplicate_or_unassigned_indices() -> None:
    with pytest.raises(ValueError, match="重复"):
        build_split_lookup(
            dataset_size=3,
            split_indices={
                "train": np.array([0, 1]),
                "valid": np.array([1]),
                "test-dev": np.array([2]),
                "test-challenge": np.array([], dtype=np.int64),
            },
        )

    with pytest.raises(ValueError, match="未分配"):
        build_split_lookup(
            dataset_size=3,
            split_indices={
                "train": np.array([0]),
                "valid": np.array([1]),
                "test-dev": np.array([], dtype=np.int64),
                "test-challenge": np.array([], dtype=np.int64),
            },
        )


def test_normalize_pcqm_item_requires_smiles_and_numeric_gap() -> None:
    assert normalize_pcqm_item(("CC", 5.75)) == ("CC", 5.75)

    with pytest.raises(ValueError, match="SMILES"):
        normalize_pcqm_item(("", 5.75))
    with pytest.raises(ValueError, match="gap"):
        normalize_pcqm_item(("CC", "not-a-number"))


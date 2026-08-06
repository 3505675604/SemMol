import json

import numpy as np
import pandas as pd
import pytest

from scripts.preprocess.build_scaffold_splits import (
    MOLECULENET_REGISTRY,
    build_dataset_split,
)
from src.datasets.scaffold_split import (
    generate_scaffold,
    load_scaffold_split,
    save_scaffold_split,
    scaffold_split,
)


def test_scaffold_split_preserves_raw_row_indices_and_reports_invalid_smiles():
    smiles = ["c1ccccc1", None, "Cc1ccccc1", "not-smiles", "CCO"]
    row_indices = [10, 14, 21, 30, 45]

    train, valid, test, report = scaffold_split(
        smiles,
        row_indices=row_indices,
        frac_train=0.6,
        frac_valid=0.2,
        frac_test=0.2,
        seed=4,
        invalid_policy="report",
        return_report=True,
    )

    assert sorted(train + valid + test) == [10, 21, 45]
    assert report["invalid"] == [
        {"source_index": 14, "reason": "missing_smiles"},
        {"source_index": 30, "reason": "invalid_smiles"},
    ]


def test_scaffold_split_never_leaks_a_scaffold_and_is_order_independent():
    smiles = [
        "c1ccccc1",
        "Cc1ccccc1",
        "Oc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "CC",
        "CCC",
        "CCCC",
    ]
    row_indices = [90, 12, 77, 40, 2, 61, 33, 105]

    split_a = scaffold_split(smiles, row_indices=row_indices, seed=17)
    split_b = scaffold_split(
        list(reversed(smiles)), row_indices=list(reversed(row_indices)), seed=17
    )

    assert tuple(map(sorted, split_a)) == tuple(map(sorted, split_b))
    split_by_index = {
        index: split_name
        for split_name, indices in zip(("train", "valid", "test"), split_a)
        for index in indices
    }
    scaffold_splits = {}
    for index, smiles_value in zip(row_indices, smiles):
        scaffold_splits.setdefault(generate_scaffold(smiles_value), set()).add(
            split_by_index[index]
        )
    assert all(len(split_names) == 1 for split_names in scaffold_splits.values())


def test_group_bin_packing_places_an_unavoidably_large_group_in_train():
    smiles = ["c1ccccc1"] * 7 + ["c1ccncc1"] + ["C1CCCCC1"] + ["C1CCOCC1"]

    train, valid, test, report = scaffold_split(
        smiles,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=2,
        return_report=True,
    )

    assert len(train) == 8
    assert len(valid) == 1
    assert len(test) == 1
    assert set(range(7)) <= set(train)
    assert report["optimization"]["exact_target_achieved"] is True
    assert report["optimization"]["optimality"] == "global"


def test_singleton_scaffolds_are_collapsed_without_losing_exact_targets():
    smiles = [f"N{'C' * (index + 2)}" for index in range(30)]

    train, valid, test, report = scaffold_split(
        smiles,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=3,
        return_report=True,
        dp_state_limit=3,
    )

    assert [len(train), len(valid), len(test)] == [24, 3, 3]
    assert report["optimization"]["exact_target_achieved"] is True
    assert report["optimization"]["singleton_groups_collapsed"] == 30
    assert report["optimization"]["global_minimum_proven"] is True


def test_infeasible_exact_ratio_reports_globally_minimal_deviation():
    smiles = ["c1ccccc1"] * 6 + ["c1ccncc1"] * 4

    train, valid, test, report = scaffold_split(
        smiles,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=7,
        return_report=True,
        dp_state_limit=10_000,
    )

    assert [len(train), len(valid), len(test)] == [10, 0, 0]
    assert report["optimization"]["target_counts"] == {
        "train": 8,
        "valid": 1,
        "test": 1,
    }
    assert report["optimization"]["exact_target_achieved"] is False
    assert report["optimization"]["optimality"] == "global"
    assert report["optimization"]["absolute_deviation"] == 4


def test_invalid_smiles_raise_by_default():
    with pytest.raises(ValueError, match=r"source_index=6.*invalid_smiles"):
        scaffold_split(["CC", "bad"], row_indices=[5, 6])


def test_scaffold_split_rejects_non_integer_original_row_indices():
    with pytest.raises(ValueError, match="row_indices.*integers"):
        scaffold_split(["CC"], row_indices=[1.5])


def test_split_writer_rejects_non_integer_indices(tmp_path):
    with pytest.raises(ValueError, match="split indices.*integers"):
        save_scaffold_split(
            tmp_path / "invalid.json",
            [1.5],
            [],
            [],
        )


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_safe_split_round_trip_includes_schema_and_scaffold_statistics(tmp_path, suffix):
    path = tmp_path / f"bbbp_scaffold{suffix}"
    scaffold_groups = {"benzene": [4, 8], "acyclic": [15]}
    save_scaffold_split(
        path,
        [4, 8],
        [],
        [15],
        dataset_name="bbbp",
        fractions=(0.8, 0.1, 0.1),
        seed=11,
        scaffold_groups=scaffold_groups,
        invalid=[{"source_index": 22, "reason": "invalid_smiles"}],
        optimization={
            "algorithm": "exact_two_dimensional_group_dp",
            "optimality": "global",
            "exact_target_achieved": False,
        },
    )

    assert load_scaffold_split(path) == ([4, 8], [], [15])
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        with np.load(path, allow_pickle=False) as arrays:
            payload = json.loads(str(arrays["metadata_json"].item()))
    assert payload["schema"] == "semmol.scaffold_split.v1"
    assert payload["dataset_name"] == "bbbp"
    assert payload["fractions"] == {"train": 0.8, "valid": 0.1, "test": 0.1}
    assert payload["statistics"]["invalid_count"] == 1
    assert len(payload["scaffold_hashes"]) == 2
    assert set(payload["scaffold_members"]) == set(payload["scaffold_hashes"])
    assert payload["optimization"]["optimality"] == "global"


def test_loader_rejects_tampered_counts_and_overlapping_indices(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(
        json.dumps(
            {
                "schema": "semmol.scaffold_split.v1",
                "dataset_name": "broken",
                "seed": 1,
                "fractions": {"train": 0.8, "valid": 0.1, "test": 0.1},
                "statistics": {
                    "split_counts": {"train": 2, "valid": 1, "test": 0},
                    "valid_count": 3,
                    "invalid_count": 0,
                    "scaffolds": {},
                },
                "scaffold_hashes": {},
                "invalid": [],
                "optimization": {},
                "indices": {"train": [1], "valid": [1], "test": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="count|disjoint"):
        load_scaffold_split(path)


def test_loader_rejects_tampered_optimization_audit(tmp_path):
    train, valid, test, report = scaffold_split(
        ["CC", "CCC", "CCCC", "CCCCC"],
        return_report=True,
    )
    path = tmp_path / "tampered_optimization.json"
    save_scaffold_split(
        path,
        train,
        valid,
        test,
        scaffold_groups=report["scaffold_groups"],
        invalid=report["invalid"],
        optimization=report["optimization"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["optimization"]["absolute_deviation"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="optimization.*deviation"):
        load_scaffold_split(path)


def test_loader_rejects_scaffold_membership_that_crosses_splits(tmp_path):
    path = tmp_path / "tampered_membership.json"
    save_scaffold_split(
        path,
        [4],
        [],
        [15],
        scaffold_groups={"benzene": [4], "acyclic": [15]},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    train_scaffold = next(
        scaffold_hash
        for scaffold_hash, split_name in payload["scaffold_hashes"].items()
        if split_name == "train"
    )
    payload["scaffold_members"][train_scaffold].append(15)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="scaffold.*crosses|multiple scaffold"):
        load_scaffold_split(path)


def test_moleculenet_registry_is_complete_and_builder_keeps_dataframe_index(tmp_path):
    assert set(MOLECULENET_REGISTRY) == {
        "bace",
        "bbbp",
        "clintox",
        "tox21",
        "toxcast",
        "sider",
        "freesolv",
        "esol",
        "lipophilicity",
    }
    csv_path = tmp_path / "bbbp.csv"
    pd.DataFrame(
        {
            "smiles": ["c1ccccc1", None, "c1ccncc1"],
            "label": [1.0, None, 0.0],
        },
        index=[101, 205, 309],
    ).to_csv(csv_path, index=True, index_label="row_id")

    output_path, statistics = build_dataset_split(
        dataset_name="bbbp",
        csv_path=csv_path,
        output_dir=tmp_path / "splits",
        smiles_col="smiles",
        row_index_col="row_id",
        fractions=(0.8, 0.1, 0.1),
        seed=5,
        output_format="json",
    )

    train, valid, test = load_scaffold_split(output_path)
    assert sorted(train + valid + test) == [101, 309]
    assert statistics["input_rows"] == 3
    assert statistics["invalid_rows"] == 1


def test_builder_rejects_integral_float_raw_indices(tmp_path):
    csv_path = tmp_path / "bbbp_float_indices.csv"
    pd.DataFrame(
        {
            "row_id": [1.0, 2.0],
            "smiles": ["CC", "CCC"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="row index.*integer dtype"):
        build_dataset_split(
            dataset_name="bbbp",
            csv_path=csv_path,
            output_dir=tmp_path / "splits",
            smiles_col="smiles",
            row_index_col="row_id",
            fractions=(0.8, 0.1, 0.1),
            seed=5,
        )

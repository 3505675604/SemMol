import hashlib
import json

import pandas as pd
import pytest

import scripts.preprocess.filter_pcqm as filter_pcqm_module
from scripts.preprocess.filter_pcqm import (
    assign_gap_bins,
    build_selection_order,
    canonicalize_and_describe,
    filter_pcqm_dataset,
    parse_args,
    prepare_records,
    resolve_pcqm_generation,
    target_quotas,
    validate_capacity,
    write_backfilled_selection,
)


def test_cli_defaults_use_the_pcqm_processed_namespace():
    args = parse_args([])

    assert args.source_manifest == "data/processed/pcqm/source_manifest.parquet"
    assert args.output_dir == "data/processed/pcqm/manifests"


def test_prepare_records_keeps_smallest_source_index_for_canonical_duplicate():
    frame = pd.DataFrame(
        {
            "smiles": ["C(C)O", "CCO", "not-smiles", "CCN", "CCC"],
            "gap": [1.0, 2.0, 3.0, float("inf"), None],
        },
        index=[11, 7, 13, 17, 19],
    )

    records, failures, stats = prepare_records(
        frame, smiles_col="smiles", gap_col="gap", source_indices=frame.index
    )

    assert [(row["source_index"], row["canonical_smiles"]) for row in records] == [
        (7, "CCO")
    ]
    assert sorted(item["reason"] for item in failures) == [
        "duplicate_canonical_smiles",
        "invalid_smiles",
        "non_finite_gap",
        "non_finite_gap",
    ]
    assert stats == {
        "input_rows": 5,
        "valid_unique_rows": 1,
        "invalid_smiles": 1,
        "non_finite_gap": 2,
        "duplicate_canonical_smiles": 1,
    }


def test_acyclic_scaffolds_use_connectivity_not_a_shared_empty_key():
    ethane = canonicalize_and_describe("CC")
    propane = canonicalize_and_describe("CCC")
    oxygen_ethane = canonicalize_and_describe("CO")

    assert ethane["scaffold_kind"] == "acyclic_connectivity"
    assert propane["scaffold_kind"] == "acyclic_connectivity"
    assert ethane["scaffold"] != propane["scaffold"]
    assert ethane["scaffold"] == oxygen_ethane["scaffold"]
    assert ethane["scaffold"]


def test_gap_bins_are_balanced_and_ties_are_broken_by_source_index():
    records = [
        {
            "source_index": source_index,
            "canonical_smiles": f"C{'C' * source_index}",
            "gap": 1.0 if source_index < 12 else 2.0,
            "scaffold": str(source_index),
            "scaffold_kind": "acyclic_connectivity",
        }
        for source_index in reversed(range(20))
    ]

    binned, metadata = assign_gap_bins(records, n_bins=10)

    counts = [sum(row["gap_bin"] == gap_bin for row in binned) for gap_bin in range(10)]
    assert counts == [2] * 10
    assert [
        row["source_index"]
        for row in sorted(binned, key=lambda row: (row["gap_bin"], row["source_index"]))[:12]
    ] == list(range(12))
    assert metadata["tie_breaker"] == ["gap", "source_index"]
    assert len(metadata["bins"]) == 10


def test_non_divisible_gap_bins_use_the_same_remainder_rule_as_target_quotas():
    records = [
        {
            "source_index": source_index,
            "canonical_smiles": "C",
            "gap": float(source_index),
            "scaffold": str(source_index),
            "scaffold_kind": "acyclic_connectivity",
        }
        for source_index in range(23)
    ]

    binned, _ = assign_gap_bins(records, n_bins=10)

    counts = [sum(row["gap_bin"] == gap_bin for row in binned) for gap_bin in range(10)]
    assert counts == target_quotas(23, 10)


def test_target_quotas_are_exact_and_nested_capacity_is_preflighted():
    assert target_quotas(23, 10) == [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]
    bin_counts = {gap_bin: 4 for gap_bin in range(10)}
    validate_capacity(bin_counts, [20, 30], n_bins=10)

    with pytest.raises(ValueError, match=r"target_size=41.*gap_bin=0.*required=5.*available=4"):
        validate_capacity(bin_counts, [41], n_bins=10)


def test_selection_order_uses_scaffold_rounds_and_is_deterministic():
    smiles = ["CC", "CCC", "CCCC", "CCO", "CCCO", "c1ccccc1", "Cc1ccccc1"]
    records = []
    for source_index, smiles_value in enumerate(smiles):
        description = canonicalize_and_describe(smiles_value)
        records.append({"source_index": source_index, **description, "gap": 1.0, "gap_bin": 0})

    first = build_selection_order(
        records, seed=9, mode="approximate", bucket_size=3, exact_threshold=2
    )
    second = build_selection_order(
        list(reversed(records)), seed=9, mode="approximate", bucket_size=3, exact_threshold=2
    )

    assert [row["source_index"] for row in first] == [
        row["source_index"] for row in second
    ]
    first_occurrence = {}
    second_occurrence = {}
    for rank, row in enumerate(first):
        scaffold = row["scaffold"]
        if scaffold in first_occurrence:
            second_occurrence.setdefault(scaffold, rank)
        else:
            first_occurrence[scaffold] = rank
    assert max(first_occurrence.values()) < min(second_occurrence.values())
    assert sorted(row["selection_rank"] for row in first) == list(range(len(first)))


def _write_source_manifest(tmp_path, input_format, frame):
    source = tmp_path / f"pcqm.{input_format}"
    if input_format == "csv":
        frame.to_csv(source, index=False)
    else:
        frame.to_parquet(source, index=False)
    return source


def test_filter_requires_official_split_unless_explicitly_overridden(tmp_path):
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "source_index": range(20),
            "smiles": [f"C{'C' * index}N" for index in range(20)],
            "gap": [float(index) for index in range(20)],
        }
    )
    source = _write_source_manifest(tmp_path, "parquet", frame)

    with pytest.raises(KeyError, match="official_split"):
        filter_pcqm_dataset(
            source_manifest=source,
            output_dir=tmp_path / "strict",
            target_sizes=(10,),
            gap_col="gap",
            selection_mode="exact",
        )

    outputs = filter_pcqm_dataset(
        source_manifest=source,
        output_dir=tmp_path / "legacy",
        target_sizes=(10,),
        gap_col="gap",
        selection_mode="exact",
        allow_missing_official_split=True,
    )
    assert outputs["manifest"].is_file()


def test_parallel_workers_do_not_change_the_materialized_ranking(tmp_path):
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "source_index": range(20),
            "smiles": [
                "c1ccccc1",
                "Cc1ccccc1",
                *[f"N{'C' * (index + 2)}" for index in range(2, 20)],
            ],
            "gap": [float(index) for index in range(20)],
            "official_split": ["train"] * 20,
        }
    )
    source = _write_source_manifest(tmp_path, "parquet", frame)
    common = {
        "source_manifest": source,
        "target_sizes": (10,),
        "gap_col": "gap",
        "selection_mode": "exact",
        "block_size": 7,
    }

    serial = filter_pcqm_dataset(
        **common, output_dir=tmp_path / "serial", workers=1
    )
    parallel = filter_pcqm_dataset(
        **common, output_dir=tmp_path / "parallel", workers=2
    )

    columns = ["source_index", "gap_bin", "selection_rank"]
    serial_ranking = pd.read_parquet(serial["manifest"], columns=columns).sort_values(
        columns
    )
    parallel_ranking = pd.read_parquet(
        parallel["manifest"], columns=columns
    ).sort_values(columns)
    pd.testing.assert_frame_equal(
        serial_ranking.reset_index(drop=True),
        parallel_ranking.reset_index(drop=True),
    )


@pytest.mark.parametrize("input_format", ["csv", "parquet"])
def test_filter_commits_generation_with_full_ranking_and_nested_targets(
    tmp_path, input_format
):
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "source_index": [1000 + index for index in range(30)] + [9999],
            "smiles": [f"C{'C' * index}N" for index in range(30)] + ["[He]"],
            "gap": [float(index // 3) for index in range(30)] + [100.0],
            "official_split": ["train"] * 30 + ["valid"],
        }
    )
    source = _write_source_manifest(tmp_path, input_format, frame)

    outputs = filter_pcqm_dataset(
        source_manifest=source,
        output_dir=tmp_path / "out",
        target_sizes=(10, 20),
        smiles_col="smiles",
        gap_col="gap",
        seed=3,
        selection_mode="exact",
        workers=1,
        block_size=7,
    )

    manifest = pd.read_parquet(outputs["manifest"])
    selected_10 = set(
        manifest.loc[
            manifest["selected_targets"].map(lambda values: 10 in values), "source_index"
        ]
    )
    selected_20 = set(
        manifest.loc[
            manifest["selected_targets"].map(lambda values: 20 in values), "source_index"
        ]
    )
    assert len(selected_10) == 10
    assert len(selected_20) == 20
    assert selected_10 < selected_20
    assert 9999 not in set(manifest["source_index"])
    assert set(manifest["source_index"]) == set(range(1000, 1030))
    assert {
        "source_index",
        "canonical_smiles",
        "gap",
        "gap_bin",
        "scaffold",
        "scaffold_kind",
        "selection_rank",
        "selected_targets",
    } <= set(manifest.columns)
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["selection"]["algorithm"] == "scaffold_round_exact_maxmin"
    assert metadata["selection"]["complete_candidate_ranking"] is True
    assert metadata["input"]["integrity"] == {
        "size_bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    pointer = json.loads(outputs["current"].read_text(encoding="utf-8"))
    assert pointer["schema"] == "semmol.pcqm_selection.current.v1"
    assert pointer["targets"] == {
        "10": "pcqm_10_indices.json",
        "20": "pcqm_20_indices.json",
    }
    assert outputs["manifest"].parent.name == pointer["generation_id"]
    assert outputs["metadata"].parent == outputs["manifest"].parent
    resolved = resolve_pcqm_generation(outputs["current"])
    assert resolved["manifest"] == outputs["manifest"]
    assert resolved["target_10"] == outputs["target_10"]
    assert resolved["target_20"] == outputs["target_20"]
    assert resolved["current_sha256"] == hashlib.sha256(
        outputs["current"].read_bytes()
    ).hexdigest()
    assert resolved["manifest_sha256"] == pointer["files"][
        pointer["manifest"]
    ]["sha256"]
    assert resolved["metadata_sha256"] == pointer["files"][
        pointer["metadata"]
    ]["sha256"]
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_cross_block_duplicate_failures_reference_the_final_minimum_winner(tmp_path):
    pytest.importorskip("pyarrow")
    unique_rows = [
        {
            "source_index": source_index,
            "smiles": f"N{'C' * (source_index + 2)}",
            "gap": float(source_index),
            "official_split": "train",
        }
        for source_index in range(20)
    ]
    duplicate_rows = [
        {"source_index": 100, "smiles": "CCO", "gap": 1.0, "official_split": "train"},
        {"source_index": 50, "smiles": "OCC", "gap": 1.0, "official_split": "train"},
        {"source_index": 10_000, "smiles": "C(C)O", "gap": 1.0, "official_split": "train"},
    ]
    interleaved = (
        unique_rows[:5]
        + [duplicate_rows[0]]
        + unique_rows[5:12]
        + [duplicate_rows[1]]
        + unique_rows[12:]
        + [duplicate_rows[2]]
    )
    source = _write_source_manifest(tmp_path, "parquet", pd.DataFrame(interleaved))

    outputs = filter_pcqm_dataset(
        source_manifest=source,
        output_dir=tmp_path / "out",
        target_sizes=(10,),
        gap_col="gap",
        selection_mode="exact",
        block_size=2,
    )

    failures = [
        json.loads(line)
        for line in outputs["failures"].read_text(encoding="utf-8").splitlines()
    ]
    duplicates = [
        item
        for item in failures
        if item["reason"] == "duplicate_canonical_smiles"
        and item["canonical_smiles"] == "CCO"
    ]
    assert sorted(item["source_index"] for item in duplicates) == [100, 10_000]
    assert {item["kept_source_index"] for item in duplicates} == {50}


def test_failed_generation_does_not_replace_current_pointer(tmp_path):
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "source_index": range(20),
            "smiles": [f"N{'C' * (index + 2)}" for index in range(20)],
            "gap": [float(index) for index in range(20)],
            "official_split": ["train"] * 20,
        }
    )
    source = _write_source_manifest(tmp_path, "parquet", frame)
    output_dir = tmp_path / "selection"
    first = filter_pcqm_dataset(
        source_manifest=source,
        output_dir=output_dir,
        target_sizes=(10,),
        gap_col="gap",
        selection_mode="exact",
    )
    pointer_before = first["current"].read_bytes()

    with pytest.raises(ValueError, match="insufficient"):
        filter_pcqm_dataset(
            source_manifest=source,
            output_dir=output_dir,
            target_sizes=(30,),
            gap_col="gap",
            selection_mode="exact",
        )

    assert first["current"].read_bytes() == pointer_before
    assert resolve_pcqm_generation(first["current"])["manifest"] == first["manifest"]


def test_source_change_during_selection_preserves_current_pointer(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "source_index": range(20),
            "smiles": [f"N{'C' * (index + 2)}" for index in range(20)],
            "gap": [float(index) for index in range(20)],
            "official_split": ["train"] * 20,
        }
    )
    source = _write_source_manifest(tmp_path, "csv", frame)
    output_dir = tmp_path / "selection"
    first = filter_pcqm_dataset(
        source_manifest=source,
        output_dir=output_dir,
        target_sizes=(10,),
        gap_col="gap",
        selection_mode="exact",
    )
    pointer_before = first["current"].read_bytes()
    original_iterator = filter_pcqm_module._iter_input_blocks

    def mutate_after_read(*args, **kwargs):
        yield from original_iterator(*args, **kwargs)
        original_bytes = source.read_bytes()
        source.write_bytes(original_bytes.replace(b"train", b"valid", 1))

    monkeypatch.setattr(
        filter_pcqm_module,
        "_iter_input_blocks",
        mutate_after_read,
    )

    with pytest.raises(RuntimeError, match="source manifest changed"):
        filter_pcqm_dataset(
            source_manifest=source,
            output_dir=output_dir,
            target_sizes=(10,),
            gap_col="gap",
            selection_mode="exact",
        )

    assert first["current"].read_bytes() == pointer_before
    assert resolve_pcqm_generation(first["current"])["manifest"] == first["manifest"]
    assert not list(output_dir.glob(".pcqm_generation_*.tmp"))


def test_source_path_swap_cannot_change_the_pinned_selection_bytes(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyarrow")
    source_frame = pd.DataFrame(
        {
            "source_index": range(20),
            "smiles": [f"N{'C' * (index + 2)}" for index in range(20)],
            "gap": [float(index) for index in range(20)],
            "official_split": ["train"] * 20,
        }
    )
    replacement_frame = source_frame.copy()
    replacement_frame["smiles"] = [
        f"O{'C' * (index + 2)}" for index in range(20)
    ]
    source = _write_source_manifest(tmp_path, "csv", source_frame)
    replacement_path = tmp_path / "replacement.csv"
    replacement_frame.to_csv(replacement_path, index=False)
    original_bytes = source.read_bytes()
    replacement_bytes = replacement_path.read_bytes()
    original_iterator = filter_pcqm_module._iter_input_blocks

    def swap_original_path_while_reading(path, **kwargs):
        source.write_bytes(replacement_bytes)
        try:
            yield from original_iterator(path, **kwargs)
        finally:
            source.write_bytes(original_bytes)

    monkeypatch.setattr(
        filter_pcqm_module,
        "_iter_input_blocks",
        swap_original_path_while_reading,
    )
    outputs = filter_pcqm_dataset(
        source_manifest=source,
        output_dir=tmp_path / "selection",
        target_sizes=(10,),
        gap_col="gap",
        selection_mode="exact",
    )

    manifest = pd.read_parquet(outputs["manifest"])
    assert manifest["canonical_smiles"].str.contains("N", regex=False).all()
    assert not manifest["canonical_smiles"].str.contains(
        "O",
        regex=False,
    ).any()


def test_backfill_skips_exclusions_and_keeps_exact_nested_per_bin_quotas(tmp_path):
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "source_index": [2000 + index for index in range(40)],
            "smiles": [f"N{'C' * (index + 2)}" for index in range(40)],
            "gap": [float(index // 4) for index in range(40)],
            "official_split": ["train"] * 40,
        }
    )
    source = _write_source_manifest(tmp_path, "parquet", frame)
    outputs = filter_pcqm_dataset(
        source_manifest=source,
        output_dir=tmp_path / "selection",
        target_sizes=(10, 20),
        gap_col="gap",
        selection_mode="exact",
    )
    ranked = pd.read_parquet(outputs["manifest"])
    exclusions = set(
        ranked.loc[ranked["selection_rank"] == 0, "source_index"].astype(int)
    )

    backfilled_path = write_backfilled_selection(
        manifest_path=outputs["manifest"],
        output_path=tmp_path / "backfilled.parquet",
        target_sizes=(10, 20),
        excluded_source_indices=exclusions,
        batch_size=7,
        staging_dir=tmp_path / "backfill_staging",
    )

    backfilled = pd.read_parquet(backfilled_path)
    selected_10 = set(
        backfilled.loc[
            backfilled["selected_targets"].map(lambda values: 10 in values),
            "source_index",
        ]
    )
    selected_20 = set(
        backfilled.loc[
            backfilled["selected_targets"].map(lambda values: 20 in values),
            "source_index",
        ]
    )
    assert len(selected_10) == 10
    assert len(selected_20) == 20
    assert selected_10 < selected_20
    assert exclusions.isdisjoint(selected_20)
    for target in (10, 20):
        selected = backfilled[
            backfilled["selected_targets"].map(lambda values: target in values)
        ]
        assert selected.groupby("gap_bin").size().tolist() == target_quotas(target, 10)
    assert not list((tmp_path / "backfill_staging").glob("*.sqlite"))


def test_backfill_rejects_non_integer_manifest_index_columns(tmp_path):
    pytest.importorskip("pyarrow")
    manifest = tmp_path / "invalid_ranking.parquet"
    pd.DataFrame(
        {
            "source_index": [1.5],
            "gap_bin": [0],
            "selection_rank": [0],
        }
    ).to_parquet(manifest, index=False)

    with pytest.raises(ValueError, match=r"source_index.*integer"):
        write_backfilled_selection(
            manifest_path=manifest,
            output_path=tmp_path / "backfilled.parquet",
            target_sizes=(1,),
        )


def test_exact_mode_rejects_rounds_above_the_configured_limit():
    records = []
    for source_index in range(5):
        description = canonicalize_and_describe(f"N{'C' * (source_index + 2)}")
        records.append(
            {
                "source_index": source_index,
                **description,
                "gap": 1.0,
                "gap_bin": 0,
            }
        )

    with pytest.raises(ValueError, match=r"exact.*limit=4.*candidates=5"):
        build_selection_order(
            records,
            seed=1,
            mode="exact",
            exact_max_candidates=4,
        )

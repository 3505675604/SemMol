from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.preprocess.build_espf_vocab import (
    FailureThresholdExceeded,
    build_espf_vocab,
    parse_args,
)
from src.molecular.espf_tokenizer import ESPFTokenizer


def _artifact_bytes(output_dir: Path) -> dict[str, bytes]:
    root_manifest_path = output_dir / "tokenizer_manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    files = {"tokenizer_manifest.json": root_manifest_path.read_bytes()}
    for name, descriptor in root_manifest["artifacts"].items():
        files[name] = (output_dir / descriptor["path"]).read_bytes()
    return files


def test_argparse_exposes_streaming_and_failure_policy_options(tmp_path) -> None:
    args = parse_args(
        [
            "--input",
            str(tmp_path / "molecules.parquet"),
            "--output-dir",
            str(tmp_path / "tokenizer"),
            "--smiles-col",
            "canonical_smiles",
            "--min-frequency",
            "7",
            "--max-merges",
            "none",
            "--vocab-size",
            "512",
            "--chunk-size",
            "4096",
            "--canonicalize",
            "rdkit",
            "--max-failures",
            "10",
            "--max-failure-rate",
            "0.01",
        ]
    )

    assert args.smiles_col == "canonical_smiles"
    assert args.min_frequency == 7
    assert args.max_merges is None
    assert args.vocab_size == 512
    assert args.chunk_size == 4096
    assert args.canonicalize == "rdkit"
    assert args.max_failures == 10
    assert args.max_failure_rate == 0.01


def test_csv_build_writes_loadable_artifacts_statistics_and_failures(
    tmp_path,
) -> None:
    input_path = tmp_path / "molecules.csv"
    input_path.write_text(
        "smiles,label\nCCO,1\nCCO,2\nC C,3\n,4\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "espf"

    stats = build_espf_vocab(
        input_path=input_path,
        output_dir=output_dir,
        smiles_col="smiles",
        min_frequency=2,
        max_merges=2,
        vocab_size=None,
        chunk_size=2,
        canonicalize="none",
        max_failures=2,
        max_failure_rate=0.5,
    )

    tokenizer = ESPFTokenizer.from_pretrained(output_dir)
    failures = [
        json.loads(line)
        for line in (output_dir / "failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    persisted_stats = json.loads(
        (output_dir / "statistics.json").read_text(encoding="utf-8")
    )
    assert tokenizer.decode(tokenizer.encode("CCO", max_len=8)) == "CCO"
    assert stats == persisted_stats
    assert stats["rows_total"] == 4
    assert stats["rows_accepted"] == 2
    assert stats["rows_failed"] == 2
    assert stats["canonicalization"] == "none"
    assert [failure["row_number"] for failure in failures] == [4, 5]


def test_failure_threshold_aborts_before_tokenizer_artifacts(tmp_path) -> None:
    input_path = tmp_path / "molecules.csv"
    input_path.write_text("smiles\nCC\nbad q\n", encoding="utf-8")
    output_dir = tmp_path / "espf"

    with pytest.raises(FailureThresholdExceeded):
        build_espf_vocab(
            input_path=input_path,
            output_dir=output_dir,
            smiles_col="smiles",
            min_frequency=1,
            max_merges=1,
            vocab_size=None,
            chunk_size=1,
            canonicalize="none",
            max_failures=0,
            max_failure_rate=0.0,
        )

    assert (output_dir / "failures.jsonl").is_file()
    assert (output_dir / "statistics.json").is_file()
    assert not (output_dir / "tokenizer_manifest.json").exists()


def test_repeated_csv_build_is_byte_deterministic(tmp_path) -> None:
    input_path = tmp_path / "molecules.csv"
    input_path.write_text("smiles\nCO\nCN\nCO\nCN\n", encoding="utf-8")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    kwargs = dict(
        input_path=input_path,
        smiles_col="smiles",
        min_frequency=1,
        max_merges=2,
        vocab_size=None,
        chunk_size=1,
        canonicalize="none",
        max_failures=0,
        max_failure_rate=0.0,
    )

    build_espf_vocab(output_dir=first_dir, **kwargs)
    build_espf_vocab(output_dir=second_dir, **kwargs)

    assert _artifact_bytes(first_dir) == _artifact_bytes(second_dir)


def test_parquet_input_is_streamed_by_row_group(tmp_path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    input_path = tmp_path / "molecules.parquet"
    table = pyarrow.table({"smiles": ["CCO", "CCN", "CCO"]})
    parquet.write_table(table, input_path, row_group_size=1)
    output_dir = tmp_path / "espf"

    stats = build_espf_vocab(
        input_path=input_path,
        output_dir=output_dir,
        smiles_col="smiles",
        min_frequency=2,
        max_merges=2,
        vocab_size=None,
        chunk_size=2,
        canonicalize="none",
        max_failures=0,
        max_failure_rate=0.0,
    )

    assert stats["input_format"] == "parquet"
    assert stats["rows_accepted"] == 3
    assert ESPFTokenizer.from_pretrained(output_dir).merges

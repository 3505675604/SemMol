"""Build safe deterministic scaffold splits for nine MoleculeNet datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.datasets.moleculenet_dataset import (  # noqa: E402
    MOLECULENET_REGISTRY,
    get_moleculenet_spec,
    resolve_moleculenet_csv,
)
from src.datasets.scaffold_split import save_scaffold_split, scaffold_split  # noqa: E402


def _validate_columns(
    frame: pd.DataFrame,
    *,
    smiles_col: str,
    row_index_col: str | None,
    csv_path: Path,
) -> None:
    required = [smiles_col]
    if row_index_col is not None:
        required.append(row_index_col)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{csv_path}: missing required columns {missing}; "
            f"available={list(frame.columns)}"
        )


def build_dataset_split(
    *,
    dataset_name: str,
    csv_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    smiles_col: str,
    fractions: Sequence[float],
    seed: int,
    output_format: str = "json",
    row_index_col: str | None = None,
    dp_state_limit: int = 250_000,
) -> tuple[Path, dict[str, int]]:
    """Build one dataset split without filtering or reindexing label rows."""

    if output_format not in {"json", "npz"}:
        raise ValueError("output_format must be 'json' or 'npz'")
    source = Path(csv_path)
    frame = pd.read_csv(source)
    _validate_columns(
        frame, smiles_col=smiles_col, row_index_col=row_index_col, csv_path=source
    )
    if row_index_col is None:
        row_indices = list(range(len(frame)))
    else:
        index_series = frame[row_index_col]
        if index_series.isna().any():
            raise ValueError(f"{source}: row index column {row_index_col!r} contains NA")
        if (
            not pd.api.types.is_integer_dtype(index_series.dtype)
            or pd.api.types.is_bool_dtype(index_series.dtype)
        ):
            raise ValueError(
                f"{source}: row index column {row_index_col!r} "
                "must have an integer dtype"
            )
        row_indices = index_series.astype("int64").tolist()
        if any(index < 0 for index in row_indices):
            raise ValueError(
                f"{source}: row index column {row_index_col!r} must be non-negative"
            )
    if len(set(row_indices)) != len(row_indices):
        raise ValueError(f"{source}: original row indices are not unique")

    frac_train, frac_valid, frac_test = map(float, fractions)
    train, valid, test, report = scaffold_split(
        frame[smiles_col].tolist(),
        row_indices=row_indices,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed,
        invalid_policy="report",
        return_report=True,
        dp_state_limit=dp_state_limit,
    )
    destination = Path(output_dir) / f"{dataset_name}_scaffold.{output_format}"
    save_scaffold_split(
        destination,
        train,
        valid,
        test,
        dataset_name=dataset_name,
        fractions=fractions,
        seed=seed,
        scaffold_groups=report["scaffold_groups"],
        invalid=report["invalid"],
        optimization=report["optimization"],
    )
    statistics = {
        "input_rows": len(frame),
        "valid_rows": report["valid_count"],
        "invalid_rows": len(report["invalid"]),
        "train_rows": len(train),
        "valid_split_rows": len(valid),
        "test_rows": len(test),
        "scaffold_groups": len(report["scaffold_groups"]),
        "absolute_split_deviation": report["optimization"]["absolute_deviation"],
    }
    return destination, statistics


def build_all_splits(
    *,
    raw_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    datasets: Sequence[str],
    fractions: Sequence[float],
    seed: int,
    output_format: str,
    row_index_col: str | None = None,
    skip_missing_files: bool = False,
    dp_state_limit: int = 250_000,
) -> dict[str, Mapping[str, Any]]:
    unknown = sorted(set(datasets) - set(MOLECULENET_REGISTRY))
    if unknown:
        raise KeyError(
            f"unknown datasets {unknown}; registered={sorted(MOLECULENET_REGISTRY)}"
        )
    summary: dict[str, Mapping[str, Any]] = {}
    for dataset_name in datasets:
        spec = get_moleculenet_spec(dataset_name)
        try:
            csv_path = resolve_moleculenet_csv(raw_root, dataset_name)
        except FileNotFoundError as exc:
            if skip_missing_files:
                summary[dataset_name] = {"status": "missing", "error": str(exc)}
                continue
            raise
        output_path, statistics = build_dataset_split(
            dataset_name=dataset_name,
            csv_path=csv_path,
            output_dir=output_dir,
            smiles_col=spec.smiles_column,
            row_index_col=row_index_col,
            fractions=fractions,
            seed=seed,
            output_format=output_format,
            dp_state_limit=dp_state_limit,
        )
        summary[dataset_name] = {
            "status": "written",
            "input": str(csv_path),
            "output": str(output_path),
            "statistics": statistics,
        }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build safe scaffold splits for MoleculeNet datasets"
    )
    parser.add_argument(
        "--raw-root",
        "--raw_root",
        dest="raw_root",
        default="data/raw/MoleculeNet",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="data/splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frac",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALID", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(MOLECULENET_REGISTRY),
        default=sorted(MOLECULENET_REGISTRY),
    )
    parser.add_argument(
        "--output-format",
        "--output_format",
        dest="output_format",
        choices=("json", "npz"),
        default="json",
    )
    parser.add_argument(
        "--dp-state-limit",
        "--dp_state_limit",
        dest="dp_state_limit",
        type=int,
        default=250_000,
    )
    parser.add_argument(
        "--row-index-col",
        "--row_index_col",
        dest="row_index_col",
        default=None,
        help="optional CSV column containing stable original row indices",
    )
    parser.add_argument(
        "--skip-missing-files",
        "--skip_missing_files",
        dest="skip_missing_files",
        action="store_true",
        help="report absent registered files instead of failing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_all_splits(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        datasets=args.datasets,
        fractions=args.frac,
        seed=args.seed,
        output_format=args.output_format,
        row_index_col=args.row_index_col,
        skip_missing_files=args.skip_missing_files,
        dp_state_limit=args.dp_state_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

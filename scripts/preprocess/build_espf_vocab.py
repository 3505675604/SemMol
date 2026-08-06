"""Build deterministic atom-wise ESPF/BPE tokenizer artifacts.

CSV input is consumed with ``pandas.read_csv(..., chunksize=...)``.  Parquet
input is consumed with ``pyarrow.parquet.ParquetFile.iter_batches`` and thus
requires a Parquet engine on the server.  Accepted SMILES are spooled once to
disk.  They are atomized once more into a compact spool and passed to the Rust
``tokenizers`` BPE trainer, whose incremental occurrence index/priority queue
does not rescan and rewrite all three million rows after every merge.

Filtering is explicit and deterministic:

* null, non-string, empty, or whitespace-containing values are rejected;
* the in-project lossless SMILES scanner must cover every character;
* ``--canonicalize none`` (default) preserves the input byte-for-byte;
* ``--canonicalize rdkit`` parses and writes canonical isomeric SMILES, and
  rejects values RDKit cannot parse.

``--output-dir`` receives an atomically replaced ``tokenizer_manifest.json``;
it points to one immutable generation containing ``vocab.json``,
``merges.txt``, ``tokenizer_config.json``, ``metadata.json``, and a hash
manifest.  ``statistics.json`` and ``failures.jsonl`` remain at the root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import pandas as pd

# Allow direct execution from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.molecular.espf_tokenizer import (  # noqa: E402
    ESPFTokenizer,
    TokenizationError,
)

PathLike = os.PathLike[str] | str


class FailureThresholdExceeded(RuntimeError):
    """Raised after reports are written when rejected rows exceed policy."""


def _optional_non_negative_int(value: str) -> Optional[int]:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer or 'none', got {value!r}"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _json_text(value: Dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _input_format(input_path: Path) -> str:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(
        f"unsupported input extension {input_path.suffix!r}; "
        "expected .csv, .parquet, or .pq"
    )


def _iter_csv_values(
    input_path: Path,
    smiles_col: str,
    chunk_size: int,
) -> Iterator[Tuple[int, Any]]:
    logical_index = 0
    try:
        chunks = pd.read_csv(
            input_path,
            usecols=[smiles_col],
            chunksize=chunk_size,
            dtype={smiles_col: "string"},
        )
        for chunk in chunks:
            for value in chunk[smiles_col].array:
                # Logical CSV row: header is row 1, first record is row 2.
                yield logical_index + 2, value
                logical_index += 1
    except ValueError as error:
        if "Usecols do not match columns" in str(error):
            raise KeyError(
                f"CSV column {smiles_col!r} was not found in {input_path}"
            ) from error
        raise


def _iter_parquet_values(
    input_path: Path,
    smiles_col: str,
    chunk_size: int,
) -> Iterator[Tuple[int, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Parquet input requires pyarrow; install it in the Linux "
            "preprocessing environment"
        ) from error

    parquet_file = parquet.ParquetFile(input_path)
    if smiles_col not in parquet_file.schema_arrow.names:
        raise KeyError(
            f"Parquet column {smiles_col!r} was not found in {input_path}; "
            f"available columns: {parquet_file.schema_arrow.names}"
        )
    logical_index = 0
    for batch in parquet_file.iter_batches(
        batch_size=chunk_size,
        columns=[smiles_col],
        use_threads=False,
    ):
        values = batch.column(0).to_pylist()
        for value in values:
            # Parquet has no physical line; report one-based data-row numbers.
            yield logical_index + 1, value
            logical_index += 1


def iter_input_values(
    input_path: PathLike,
    smiles_col: str,
    chunk_size: int,
) -> Iterator[Tuple[int, Any]]:
    """Yield stable logical row numbers and raw SMILES values."""
    path = Path(input_path)
    input_format = _input_format(path)
    if input_format == "csv":
        yield from _iter_csv_values(path, smiles_col, chunk_size)
    else:
        yield from _iter_parquet_values(path, smiles_col, chunk_size)


def _prepare_smiles(value: Any, canonicalize: str) -> str:
    if value is None or value is pd.NA:
        raise ValueError("SMILES value is null")
    if not isinstance(value, str):
        raise TypeError(
            f"SMILES value must be a string, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("SMILES value is empty")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError("SMILES value contains whitespace")

    # Always apply the in-project lossless scanner before optional chemistry
    # canonicalization, so no input characters can disappear unnoticed.
    ESPFTokenizer.atom_tokenize(value)
    if canonicalize == "none":
        return value
    if canonicalize != "rdkit":
        raise ValueError(f"unknown canonicalization mode {canonicalize!r}")

    try:
        from rdkit import Chem
    except ImportError as error:
        raise RuntimeError(
            "RDKit canonicalization requested but rdkit is unavailable"
        ) from error

    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError("RDKit could not parse SMILES")
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    ESPFTokenizer.atom_tokenize(canonical)
    return canonical


def _failure_record(row_number: int, value: Any, error: Exception) -> Dict[str, Any]:
    if value is None or value is pd.NA:
        serializable_value: Any = None
    elif isinstance(value, float) and not math.isfinite(value):
        serializable_value = repr(value)
    elif isinstance(value, (str, int, float, bool)):
        serializable_value = value
    else:
        serializable_value = repr(value)
    return {
        "row_number": row_number,
        "value": serializable_value,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _threshold_exceeded(
    rows_total: int,
    rows_failed: int,
    max_failures: int,
    max_failure_rate: float,
) -> bool:
    failure_rate = rows_failed / rows_total if rows_total else 0.0
    return rows_failed > max_failures or failure_rate > max_failure_rate


def build_espf_vocab(
    *,
    input_path: PathLike,
    output_dir: PathLike,
    smiles_col: str = "smiles",
    min_frequency: int = 50,
    max_merges: Optional[int] = None,
    vocab_size: Optional[int] = 8192,
    chunk_size: int = 65_536,
    canonicalize: str = "none",
    max_failures: int = 0,
    max_failure_rate: float = 0.0,
) -> Dict[str, Any]:
    """Stream input, train ESPF/BPE, and atomically publish all reports."""
    input_file = Path(input_path)
    destination = Path(output_dir)
    if not input_file.is_file():
        raise FileNotFoundError(f"input dataset does not exist: {input_file}")
    input_format = _input_format(input_file)
    if not isinstance(smiles_col, str):
        raise TypeError("smiles_col must be a string")
    if not smiles_col:
        raise ValueError("smiles_col must not be empty")
    if not isinstance(min_frequency, int) or isinstance(min_frequency, bool):
        raise TypeError("min_frequency must be an integer")
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")
    if max_merges is not None and (
        not isinstance(max_merges, int) or isinstance(max_merges, bool)
    ):
        raise TypeError("max_merges must be an integer or None")
    if max_merges is not None and max_merges < 0:
        raise ValueError("max_merges must be non-negative or None")
    if vocab_size is not None and (
        not isinstance(vocab_size, int) or isinstance(vocab_size, bool)
    ):
        raise TypeError("vocab_size must be an integer or None")
    if vocab_size is not None and vocab_size < 5:
        raise ValueError("vocab_size must be at least 5")
    if max_merges is None and vocab_size is None:
        raise ValueError("at least one of max_merges or vocab_size is required")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not isinstance(canonicalize, str):
        raise TypeError("canonicalize must be a string")
    if canonicalize not in {"none", "rdkit"}:
        raise ValueError("canonicalize must be 'none' or 'rdkit'")
    if not isinstance(max_failures, int) or isinstance(max_failures, bool):
        raise TypeError("max_failures must be an integer")
    if max_failures < 0:
        raise ValueError("max_failures must be non-negative")
    if (
        not isinstance(max_failure_rate, (int, float))
        or isinstance(max_failure_rate, bool)
        or not math.isfinite(float(max_failure_rate))
    ):
        raise TypeError("max_failure_rate must be a finite number")
    if not 0.0 <= max_failure_rate <= 1.0:
        raise ValueError("max_failure_rate must be between 0 and 1")

    destination.mkdir(parents=True, exist_ok=True)
    failures_path = destination / "failures.jsonl"
    statistics_path = destination / "statistics.json"
    rows_total = 0
    rows_accepted = 0
    rows_failed = 0
    failures_temporary: Optional[Path] = None

    with tempfile.TemporaryDirectory(
        prefix=".semmol_espf_build_", dir=destination
    ) as working_directory:
        corpus_path = Path(working_directory) / "accepted.smi"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination,
                prefix=".failures.jsonl.",
                suffix=".tmp",
                delete=False,
            ) as failures_handle, corpus_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as corpus_handle:
                failures_temporary = Path(failures_handle.name)
                for row_number, value in iter_input_values(
                    input_file, smiles_col, chunk_size
                ):
                    rows_total += 1
                    try:
                        smiles = _prepare_smiles(value, canonicalize)
                    except (TokenizationError, TypeError, ValueError) as error:
                        rows_failed += 1
                        failures_handle.write(
                            json.dumps(
                                _failure_record(row_number, value, error),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )
                        failures_handle.write("\n")
                        continue
                    corpus_handle.write(smiles)
                    corpus_handle.write("\n")
                    rows_accepted += 1
                failures_handle.flush()
                os.fsync(failures_handle.fileno())
                corpus_handle.flush()
                os.fsync(corpus_handle.fileno())

            os.replace(failures_temporary, failures_path)
            failures_temporary = None

            failure_rate = rows_failed / rows_total if rows_total else 0.0
            exceeded = _threshold_exceeded(
                rows_total=rows_total,
                rows_failed=rows_failed,
                max_failures=max_failures,
                max_failure_rate=max_failure_rate,
            )
            statistics: Dict[str, Any] = {
                "schema_version": 1,
                "status": "failure_threshold_exceeded" if exceeded else "training",
                "input_path": str(input_file.resolve()),
                "input_format": input_format,
                "smiles_column": smiles_col,
                "canonicalization": canonicalize,
                "filter_policy": (
                    "reject-null-nonstring-empty-whitespace-or-uncovered"
                ),
                "rows_total": rows_total,
                "rows_accepted": rows_accepted,
                "rows_failed": rows_failed,
                "failure_rate": failure_rate,
                "max_failures": max_failures,
                "max_failure_rate": max_failure_rate,
                "min_frequency": min_frequency,
                "max_merges": max_merges,
                "target_vocab_size": vocab_size,
                "chunk_size": chunk_size,
            }
            if exceeded:
                _atomic_write_text(statistics_path, _json_text(statistics))
                raise FailureThresholdExceeded(
                    "rejected rows exceeded policy: "
                    f"{rows_failed}/{rows_total} failures "
                    f"(max_failures={max_failures}, "
                    f"max_failure_rate={max_failure_rate})"
                )
            if rows_accepted == 0:
                statistics["status"] = "empty_corpus"
                _atomic_write_text(statistics_path, _json_text(statistics))
                raise ValueError("no valid SMILES remained after filtering")

            def accepted_smiles() -> Iterator[str]:
                with corpus_path.open("r", encoding="utf-8") as corpus_source:
                    for line in corpus_source:
                        yield line[:-1] if line.endswith("\n") else line

            tokenizer = ESPFTokenizer.train(
                accepted_smiles(),
                min_frequency=min_frequency,
                max_merges=max_merges,
                vocab_size=vocab_size,
                temp_dir=working_directory,
            )
            tokenizer.metadata.update(
                {
                    "canonicalization": canonicalize,
                    "filter_policy": (
                        "reject-null-nonstring-empty-whitespace-or-uncovered"
                    ),
                }
            )
            saved_artifacts = tokenizer.save_pretrained(destination)
            statistics.update(
                {
                    "status": "complete",
                    "vocab_size": tokenizer.vocab_size,
                    "num_merges": len(tokenizer.merges),
                    "corpus_sha256": tokenizer.metadata["corpus_sha256"],
                    "atom_token_count": tokenizer.metadata["atom_token_count"],
                    "tokenizer_generation_id": tokenizer.generation_id,
                    "artifact_hashes": tokenizer.artifact_hashes,
                    "tokenizer_manifest": str(
                        saved_artifacts["tokenizer_manifest"].resolve()
                    ),
                }
            )
            _atomic_write_text(statistics_path, _json_text(statistics))
            return statistics
        finally:
            if failures_temporary is not None and failures_temporary.exists():
                failures_temporary.unlink()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream CSV/Parquet SMILES and build deterministic atom-wise "
            "ESPF/BPE artifacts with the incremental Rust BPE trainer."
        )
    )
    parser.add_argument(
        "--input",
        "--input-path",
        "--raw-csv",
        "--raw_csv",
        dest="input_path",
        type=Path,
        required=True,
        help="Input .csv, .parquet, or .pq dataset",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        "--output",
        type=Path,
        required=True,
        help="Directory receiving tokenizer artifacts and reports",
    )
    parser.add_argument(
        "--smiles-col",
        "--smiles_col",
        dest="smiles_col",
        default="smiles",
        help="SMILES column name",
    )
    parser.add_argument(
        "--min-frequency",
        "--min_frequency",
        dest="min_frequency",
        type=int,
        default=50,
        help="Minimum corpus frequency for a pair to be merged",
    )
    parser.add_argument(
        "--max-merges",
        "--max_merges",
        dest="max_merges",
        type=_optional_non_negative_int,
        default=None,
        help="Maximum ordered BPE merges, or 'none' when vocab-size is set",
    )
    parser.add_argument(
        "--vocab-size",
        "--vocab_size",
        "--target-vocab-size",
        dest="vocab_size",
        type=int,
        default=8192,
        help="Optional total vocabulary cap, including five special tokens",
    )
    parser.add_argument(
        "--chunk-size",
        "--chunk_size",
        "--corpus-chunk-size",
        dest="chunk_size",
        type=int,
        default=65_536,
        help="CSV chunk size or Parquet batch size",
    )
    parser.add_argument(
        "--canonicalize",
        choices=("none", "rdkit"),
        default="none",
        help="Preserve exact input or canonicalize with RDKit",
    )
    parser.add_argument(
        "--max-failures",
        "--max_failures",
        dest="max_failures",
        type=int,
        default=0,
        help="Maximum rejected row count before a non-zero exit",
    )
    parser.add_argument(
        "--max-failure-rate",
        "--max_failure_rate",
        "--failure-threshold",
        dest="max_failure_rate",
        type=float,
        default=0.0,
        help="Maximum rejected fraction in [0,1] before a non-zero exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        statistics = build_espf_vocab(
            input_path=args.input_path,
            output_dir=args.output_dir,
            smiles_col=args.smiles_col,
            min_frequency=args.min_frequency,
            max_merges=args.max_merges,
            vocab_size=args.vocab_size,
            chunk_size=args.chunk_size,
            canonicalize=args.canonicalize,
            max_failures=args.max_failures,
            max_failure_rate=args.max_failure_rate,
        )
    except FailureThresholdExceeded as error:
        print(f"[ESPF] {error}", file=sys.stderr)
        return 2
    print(
        "[ESPF] complete: "
        f"accepted={statistics['rows_accepted']}, "
        f"failed={statistics['rows_failed']}, "
        f"vocab_size={statistics['vocab_size']}, "
        f"merges={statistics['num_merges']} -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

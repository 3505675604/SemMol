"""Build deterministic, nested PCQM selection manifests.

The pipeline canonicalizes isomeric SMILES, removes exact molecular
duplicates, creates ten deterministic equal-frequency HOMO-LUMO-gap strata,
and ranks every valid molecule.  Diversity ranking is scaffold-round-first
and uses Morgan radius-2/2048-bit fingerprints with Tanimoto distance.

The default ranking is a hierarchical approximate MaxMin procedure.  It
MinHash-sorts candidates into bounded buckets, performs exact MaxMin ordering
inside each bucket, orders bucket representatives recursively, and merges
bucket members by rounds.  Its work is bounded by the bucket size instead of
growing as O(N * requested_samples).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

SCHEMA_VERSION = "semmol.pcqm_selection.v1"
N_GAP_BINS = 10
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
DEFAULT_BUCKET_SIZE = 1024
DEFAULT_EXACT_THRESHOLD = 4096
_MINHASH_PERMUTATIONS = 16
_MINHASH_PRIME = 4_294_967_311


def _stable_hash(seed: int, value: Any) -> int:
    payload = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _acyclic_connectivity_skeleton(mol: Chem.Mol) -> str:
    skeleton = Chem.RWMol(mol)
    for atom in skeleton.GetAtoms():
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    for bond in skeleton.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(
        skeleton.GetMol(), canonical=True, isomericSmiles=False, allHsExplicit=True
    )


def canonicalize_and_describe(smiles: Any) -> dict[str, str]:
    """Return canonical SMILES and a non-empty scaffold description.

    Raises:
        ValueError: if *smiles* is missing, empty, or cannot be parsed.
    """

    if smiles is None or (not isinstance(smiles, str) and pd.isna(smiles)):
        raise ValueError("missing_smiles")
    text = str(smiles).strip()
    if not text:
        raise ValueError("missing_smiles")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError("invalid_smiles")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    murcko = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False
    )
    if murcko:
        scaffold = Chem.MolToSmiles(
            Chem.MolFromSmiles(murcko), canonical=True, isomericSmiles=False
        )
        kind = "bemis_murcko"
    else:
        scaffold = _acyclic_connectivity_skeleton(mol)
        kind = "acyclic_connectivity"
    if not scaffold:
        raise ValueError("scaffold_generation_failed")
    return {
        "canonical_smiles": canonical,
        "scaffold": scaffold,
        "scaffold_kind": kind,
    }


def extract_scaffold(smiles: str) -> str:
    """Compatibility wrapper returning the non-empty scaffold key."""

    return canonicalize_and_describe(smiles)["scaffold"]


def _prepare_one(argument: tuple[int, Any, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source_index, smiles, gap_value = argument
    try:
        gap = float(gap_value)
    except (TypeError, ValueError):
        return None, {"source_index": int(source_index), "reason": "non_finite_gap"}
    if not math.isfinite(gap):
        return None, {"source_index": int(source_index), "reason": "non_finite_gap"}
    try:
        description = canonicalize_and_describe(smiles)
    except ValueError as exc:
        return None, {"source_index": int(source_index), "reason": str(exc)}
    return {
        "source_index": int(source_index),
        **description,
        "gap": gap,
    }, None


def prepare_records(
    frame: pd.DataFrame,
    *,
    smiles_col: str,
    gap_col: str,
    source_indices: Iterable[int] | None = None,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Canonicalize one frame and deterministically remove exact duplicates."""

    missing = [name for name in (smiles_col, gap_col) if name not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns {missing}; available={list(frame.columns)}")
    indices = list(range(len(frame))) if source_indices is None else list(source_indices)
    if len(indices) != len(frame):
        raise ValueError("source_indices must have the same length as frame")
    arguments = list(zip(indices, frame[smiles_col].tolist(), frame[gap_col].tolist()))
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        processed = map(_prepare_one, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        processed = executor.map(_prepare_one, arguments, chunksize=max(1, len(arguments) // (workers * 8)))

    candidates: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    try:
        for record, failure in processed:
            if failure is not None:
                failures.append(failure)
                reason_counts[failure["reason"]] += 1
                continue
            if record is None:
                raise RuntimeError("row preparation returned neither a record nor a failure")
            canonical = record["canonical_smiles"]
            current = candidates.get(canonical)
            if current is None:
                candidates[canonical] = record
                continue
            if record["source_index"] < current["source_index"]:
                loser, winner = current, record
                candidates[canonical] = record
            else:
                loser, winner = record, current
            failures.append(
                {
                    "source_index": loser["source_index"],
                    "reason": "duplicate_canonical_smiles",
                    "canonical_smiles": canonical,
                    "kept_source_index": winner["source_index"],
                }
            )
            reason_counts["duplicate_canonical_smiles"] += 1
    finally:
        if workers != 1:
            executor.shutdown(wait=True)

    records = sorted(candidates.values(), key=lambda row: row["source_index"])
    stats = {
        "input_rows": len(frame),
        "valid_unique_rows": len(records),
        "invalid_smiles": reason_counts["invalid_smiles"]
        + reason_counts["missing_smiles"]
        + reason_counts["scaffold_generation_failed"],
        "non_finite_gap": reason_counts["non_finite_gap"],
        "duplicate_canonical_smiles": reason_counts["duplicate_canonical_smiles"],
    }
    return records, failures, stats


def assign_gap_bins(
    records: Sequence[Mapping[str, Any]], n_bins: int = N_GAP_BINS
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign equal-frequency bins using (gap, source_index) rank.

    Sorting by source index makes ties deterministic.  Unlike value-only
    quantiles, repeated gap values do not collapse bins.
    """

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    ranked = sorted(records, key=lambda row: (float(row["gap"]), int(row["source_index"])))
    n_records = len(ranked)
    output: list[dict[str, Any]] = []
    members_by_bin: list[list[dict[str, Any]]] = [[] for _ in range(n_bins)]
    bin_sizes = target_quotas(n_records, n_bins)
    bin_ends = np.cumsum(bin_sizes).tolist()
    gap_bin = 0
    for rank, record in enumerate(ranked):
        while gap_bin < n_bins - 1 and rank >= bin_ends[gap_bin]:
            gap_bin += 1
        binned_record = {**record, "gap_bin": int(gap_bin)}
        output.append(binned_record)
        members_by_bin[gap_bin].append(binned_record)
    output.sort(key=lambda row: row["source_index"])

    bin_metadata = []
    for gap_bin in range(n_bins):
        members = members_by_bin[gap_bin]
        gaps = [float(row["gap"]) for row in members]
        bin_metadata.append(
            {
                "gap_bin": gap_bin,
                "count": len(members),
                "min_gap": min(gaps) if gaps else None,
                "max_gap": max(gaps) if gaps else None,
            }
        )
    return output, {
        "method": "equal_frequency_rank",
        "n_bins": n_bins,
        "tie_breaker": ["gap", "source_index"],
        "bins": bin_metadata,
    }


def target_quotas(target_size: int, n_bins: int = N_GAP_BINS) -> list[int]:
    if not isinstance(target_size, (int, np.integer)) or isinstance(target_size, bool):
        raise ValueError("target_size must be a strict integer")
    if target_size < 0:
        raise ValueError("target_size must be non-negative")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    base, remainder = divmod(int(target_size), n_bins)
    return [base + (1 if gap_bin < remainder else 0) for gap_bin in range(n_bins)]


def _validated_target_sizes(target_sizes: Sequence[int]) -> list[int]:
    values = list(target_sizes)
    if not values:
        raise ValueError("target_sizes must not be empty")
    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, bool)
        for value in values
    ):
        raise ValueError("target_sizes must contain strict integers")
    targets = sorted(set(int(value) for value in values))
    if any(target < 1 for target in targets):
        raise ValueError("all target_sizes must be positive")
    return targets


def validate_capacity(
    bin_counts: Mapping[int, int],
    target_sizes: Sequence[int],
    n_bins: int = N_GAP_BINS,
) -> None:
    for target_size in _validated_target_sizes(target_sizes):
        for gap_bin, required in enumerate(target_quotas(target_size, n_bins)):
            available = int(bin_counts.get(gap_bin, 0))
            if available < required:
                raise ValueError(
                    "insufficient valid unique molecules during preflight: "
                    f"target_size={target_size}, gap_bin={gap_bin}, "
                    f"required={required}, available={available}"
                )


def _morgan_fp_from_smiles(smiles: str, n_bits: int = MORGAN_BITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"canonical SMILES unexpectedly failed to parse: {smiles!r}")
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, MORGAN_RADIUS, nBits=n_bits
    )


def _exact_maxmin_order(
    fps: Sequence[Any], identities: Sequence[int], seed: int
) -> list[int]:
    n_items = len(fps)
    if n_items <= 1:
        return list(range(n_items))
    fingerprint_list = list(fps)
    stable_keys = [_stable_hash(seed, identity) for identity in identities]
    first = min(range(n_items), key=lambda index: (stable_keys[index], identities[index]))
    order = [first]
    selected = np.zeros(n_items, dtype=bool)
    selected[first] = True
    min_distances = np.full(n_items, np.inf, dtype=np.float64)
    while len(order) < n_items:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fingerprint_list[order[-1]], fingerprint_list
            ),
            dtype=np.float64,
        )
        np.minimum(min_distances, 1.0 - similarities, out=min_distances)
        min_distances[selected] = -1.0
        maximum = float(min_distances.max())
        tied = np.flatnonzero(np.isclose(min_distances, maximum, rtol=0.0, atol=1e-12))
        next_index = min(
            tied.tolist(), key=lambda index: (stable_keys[index], identities[index])
        )
        selected[next_index] = True
        order.append(next_index)
    return order


@lru_cache(maxsize=None)
def _minhash_coefficients(seed: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            1 + (_stable_hash(seed, f"a:{permutation}") % (_MINHASH_PRIME - 1)),
            _stable_hash(seed, f"b:{permutation}") % _MINHASH_PRIME,
        )
        for permutation in range(_MINHASH_PERMUTATIONS)
    )


def _minhash_signature(
    fp: Any, coefficients: Sequence[tuple[int, int]]
) -> tuple[int, ...]:
    bits = list(fp.GetOnBits())
    if not bits:
        return tuple([_MINHASH_PRIME] * _MINHASH_PERMUTATIONS)
    return tuple(
        min((a * bit + b) % _MINHASH_PRIME for bit in bits)
        for a, b in coefficients
    )


def _hierarchical_approximate_order(
    fps: Sequence[Any],
    identities: Sequence[int],
    *,
    seed: int,
    bucket_size: int,
    exact_threshold: int,
) -> list[int]:
    n_items = len(fps)
    if n_items <= exact_threshold:
        return _exact_maxmin_order(fps, identities, seed)

    coefficients = _minhash_coefficients(seed)
    signatures = [_minhash_signature(fp, coefficients) for fp in fps]
    sorted_indices = sorted(
        range(n_items),
        key=lambda index: (
            signatures[index],
            _stable_hash(seed, identities[index]),
            identities[index],
        ),
    )
    buckets = [
        sorted_indices[start : start + bucket_size]
        for start in range(0, n_items, bucket_size)
    ]
    local_orders: list[list[int]] = []
    for bucket_number, bucket in enumerate(buckets):
        relative = _exact_maxmin_order(
            [fps[index] for index in bucket],
            [identities[index] for index in bucket],
            seed + 1_000_003 * (bucket_number + 1),
        )
        local_orders.append([bucket[index] for index in relative])

    representative_indices = [order[0] for order in local_orders]
    representative_relative_order = _hierarchical_approximate_order(
        [fps[index] for index in representative_indices],
        [identities[index] for index in representative_indices],
        seed=seed + 97_409,
        bucket_size=bucket_size,
        exact_threshold=exact_threshold,
    )
    ordered_buckets = [local_orders[index] for index in representative_relative_order]
    merged: list[int] = []
    for member_round in range(max(map(len, ordered_buckets))):
        for bucket in ordered_buckets:
            if member_round < len(bucket):
                merged.append(bucket[member_round])
    return merged


def build_selection_order(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    mode: str = "approximate",
    bucket_size: int = DEFAULT_BUCKET_SIZE,
    exact_threshold: int = DEFAULT_EXACT_THRESHOLD,
    exact_max_candidates: int = 20_000,
) -> list[dict[str, Any]]:
    """Rank candidates with scaffold rounds followed by molecular diversity."""

    if mode not in {"approximate", "exact"}:
        raise ValueError("mode must be 'approximate' or 'exact'")
    if bucket_size < 2:
        raise ValueError("bucket_size must be at least 2")
    if exact_threshold < 2:
        raise ValueError("exact_threshold must be at least 2")
    if exact_max_candidates < 1:
        raise ValueError("exact_max_candidates must be positive")
    if mode == "exact" and len(records) > exact_max_candidates:
        raise ValueError(
            "exact selection exceeds configured total safety limit: "
            f"limit={exact_max_candidates}, candidates={len(records)}"
        )

    scaffold_members: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        scaffold_members[str(record["scaffold"])].append(record)
    for members in scaffold_members.values():
        members.sort(key=lambda row: int(row["source_index"]))

    ordered: list[Mapping[str, Any]] = []
    max_scaffold_size = max(map(len, scaffold_members.values()), default=0)
    for scaffold_round in range(max_scaffold_size):
        round_members = [
            members[scaffold_round]
            for scaffold, members in sorted(scaffold_members.items())
            if scaffold_round < len(members)
        ]
        round_members.sort(key=lambda row: int(row["source_index"]))
        if mode == "exact" and len(round_members) > exact_max_candidates:
            raise ValueError(
                "exact selection exceeds configured safety limit: "
                f"limit={exact_max_candidates}, candidates={len(round_members)}"
            )
        fps = [_morgan_fp_from_smiles(str(row["canonical_smiles"])) for row in round_members]
        identities = [int(row["source_index"]) for row in round_members]
        if mode == "exact":
            relative_order = _exact_maxmin_order(fps, identities, seed + scaffold_round)
        else:
            relative_order = _hierarchical_approximate_order(
                fps,
                identities,
                seed=seed + scaffold_round,
                bucket_size=bucket_size,
                exact_threshold=exact_threshold,
            )
        ordered.extend(round_members[index] for index in relative_order)
    return [
        {**record, "selection_rank": rank}
        for rank, record in enumerate(ordered)
    ]


def farthest_point_sampling(
    smiles_list: Sequence[str],
    num_samples: int,
    seed: int = 3407,
    n_bits: int = MORGAN_BITS,
    candidate_cap: int | None = None,
) -> list[int]:
    """Compatibility API using exact MaxMin without truncating candidates.

    ``candidate_cap`` is accepted for older callers but never used to discard
    candidates.  A cap below the requested quota is rejected explicitly.
    """

    if num_samples < 0 or num_samples > len(smiles_list):
        raise ValueError(
            f"num_samples={num_samples} exceeds candidate count={len(smiles_list)}"
        )
    if candidate_cap is not None and candidate_cap < num_samples:
        raise ValueError(
            f"candidate_cap={candidate_cap} is smaller than quota={num_samples}"
        )
    fps = [_morgan_fp_from_smiles(smiles, n_bits=n_bits) for smiles in smiles_list]
    return _exact_maxmin_order(fps, list(range(len(fps))), seed)[:num_samples]


def _iter_input_blocks(
    path: Path,
    *,
    required_columns: Sequence[str],
    optional_columns: Sequence[str],
    block_size: int,
) -> Iterator[tuple[pd.DataFrame, list[int]]]:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    suffix = path.suffix.lower()
    offset = 0
    if suffix == ".csv":
        with path.open("rb") as stream:
            available = pd.read_csv(stream, nrows=0).columns.tolist()
            missing = [
                column for column in required_columns if column not in available
            ]
            if missing:
                raise KeyError(
                    f"missing required columns {missing}; available={available}"
                )
            selected_columns = list(required_columns) + [
                column
                for column in optional_columns
                if (
                    column
                    and column in available
                    and column not in required_columns
                )
            ]
            stream.seek(0)
            for block in pd.read_csv(
                stream,
                usecols=selected_columns,
                chunksize=block_size,
            ):
                source_indices = list(range(offset, offset + len(block)))
                offset += len(block)
                yield block, source_indices
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires pyarrow; install pyarrow on the target server"
            ) from exc
        parquet_file = pq.ParquetFile(path)
        available = parquet_file.schema.names
        missing = [
            column for column in required_columns if column not in available
        ]
        if missing:
            raise KeyError(
                f"missing required columns {missing}; available={available}"
            )
        selected_columns = list(required_columns) + [
            column
            for column in optional_columns
            if column and column in available and column not in required_columns
        ]
        for batch in parquet_file.iter_batches(
            batch_size=block_size, columns=selected_columns
        ):
            import pandas as pd
            block = batch.to_pandas(types_mapper=pd.ArrowDtype)
            source_indices = list(range(offset, offset + len(block)))
            offset += len(block)
            yield block, source_indices
        return
    raise ValueError("input must have a .csv, .parquet, or .pq extension")


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet manifest output requires pyarrow; install pyarrow on the target server"
        ) from exc
    return pa, pq


def _validated_source_indices(
    values: Sequence[Any], *, column_name: str
) -> list[int]:
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError(f"source index column {column_name!r} contains missing values")
    if not pd.api.types.is_integer_dtype(series.dtype):
        raise ValueError(
            f"source index column {column_name!r} must have an integer dtype"
        )
    numeric = series.astype("int64")
    if (numeric < 0).any():
        raise ValueError(
            f"source index column {column_name!r} must contain non-negative indices"
        )
    return numeric.tolist()


def _manifest_schema(pa):
    return pa.schema(
        [
            ("source_index", pa.int64()),
            ("canonical_smiles", pa.string()),
            ("gap", pa.float64()),
            ("gap_bin", pa.int8()),
            ("scaffold", pa.string()),
            ("scaffold_kind", pa.string()),
            ("selection_rank", pa.int64()),
            ("selected_targets", pa.list_(pa.int64())),
        ]
    )


def _backfill_schema(pa, metadata: Mapping[bytes, bytes] | None = None):
    schema = pa.schema(
        [
            ("source_index", pa.int64()),
            ("gap_bin", pa.int8()),
            ("selection_rank", pa.int64()),
            ("selected_targets", pa.list_(pa.int64())),
        ]
    )
    return schema.with_metadata(metadata) if metadata else schema


def _chunks(values: Sequence[int], size: int = 900) -> Iterator[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _connect_staging(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    connection.executescript(
        """
        CREATE TABLE source_rows (
            source_index INTEGER PRIMARY KEY
        );
        CREATE TABLE valid_rows (
            source_index INTEGER PRIMARY KEY,
            canonical_smiles TEXT NOT NULL,
            gap REAL NOT NULL,
            scaffold TEXT NOT NULL,
            scaffold_kind TEXT NOT NULL
        );
        """
    )
    return connection


def _register_source_indices(
    connection: sqlite3.Connection, source_indices: Sequence[int]
) -> None:
    if len(set(source_indices)) != len(source_indices):
        counts = Counter(source_indices)
        duplicate = min(index for index, count in counts.items() if count > 1)
        raise ValueError(
            "source indices must be unique within each input block; "
            f"duplicate source_index={duplicate}"
        )
    for chunk in _chunks(source_indices):
        placeholders = ",".join("?" for _ in chunk)
        existing = connection.execute(
            f"SELECT source_index FROM source_rows "
            f"WHERE source_index IN ({placeholders}) ORDER BY source_index LIMIT 1",
            tuple(chunk),
        ).fetchone()
        if existing is not None:
            raise ValueError(
                "source indices must be globally unique; "
                f"duplicate source_index={existing[0]}"
            )
    connection.executemany(
        "INSERT INTO source_rows(source_index) VALUES (?)",
        ((int(index),) for index in source_indices),
    )


def _prepare_block_without_deduplication(
    frame: pd.DataFrame,
    *,
    smiles_col: str,
    gap_col: str,
    source_indices: Sequence[int],
    executor: ProcessPoolExecutor | None,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arguments = list(
        zip(source_indices, frame[smiles_col].tolist(), frame[gap_col].tolist())
    )
    if executor is None:
        processed = map(_prepare_one, arguments)
    else:
        processed = executor.map(
            _prepare_one,
            arguments,
            chunksize=max(1, len(arguments) // max(1, workers * 8)),
        )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record, failure in processed:
        if failure is not None:
            failures.append(failure)
        elif record is not None:
            records.append(record)
        else:
            raise RuntimeError("row preparation returned neither a record nor a failure")
    return records, failures


def _materialize_unique_candidates(
    connection: sqlite3.Connection,
    failure_stream,
    failure_counts: Counter[str],
) -> int:
    connection.executescript(
        """
        CREATE INDEX valid_rows_canonical_idx
            ON valid_rows(canonical_smiles, source_index);
        CREATE TABLE canonical_winners AS
            SELECT canonical_smiles, MIN(source_index) AS winner_source_index
            FROM valid_rows
            GROUP BY canonical_smiles;
        CREATE UNIQUE INDEX canonical_winners_idx
            ON canonical_winners(canonical_smiles);
        CREATE TABLE candidates AS
            SELECT v.source_index,
                   v.canonical_smiles,
                   v.gap,
                   v.scaffold,
                   v.scaffold_kind,
                   CAST(NULL AS INTEGER) AS gap_bin,
                   CAST(NULL AS INTEGER) AS scaffold_round,
                   CAST(NULL AS BLOB) AS fingerprint,
                   CAST(NULL AS BLOB) AS minhash_key,
                   CAST(NULL AS INTEGER) AS selection_rank
            FROM valid_rows AS v
            JOIN canonical_winners AS w
              ON v.canonical_smiles = w.canonical_smiles
             AND v.source_index = w.winner_source_index;
        CREATE UNIQUE INDEX candidates_source_idx ON candidates(source_index);
        """
    )
    duplicate_cursor = connection.execute(
        """
        SELECT v.source_index, v.canonical_smiles, w.winner_source_index
        FROM valid_rows AS v
        JOIN canonical_winners AS w
          ON v.canonical_smiles = w.canonical_smiles
        WHERE v.source_index <> w.winner_source_index
        ORDER BY v.source_index
        """
    )
    duplicate_count = 0
    for source_index, canonical_smiles, winner_source_index in duplicate_cursor:
        failure_stream.write(
            json.dumps(
                {
                    "source_index": int(source_index),
                    "reason": "duplicate_canonical_smiles",
                    "canonical_smiles": canonical_smiles,
                    "kept_source_index": int(winner_source_index),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        duplicate_count += 1
    failure_counts["duplicate_canonical_smiles"] += duplicate_count
    connection.commit()
    return int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])


def _assign_gap_bins_external(
    connection: sqlite3.Connection, candidate_count: int
) -> tuple[dict[int, int], dict[str, Any]]:
    bin_sizes = target_quotas(candidate_count, N_GAP_BINS)
    updates: list[tuple[int, int]] = []
    gap_bin = 0
    consumed_in_bin = 0
    for (source_index,) in connection.execute(
        "SELECT source_index FROM candidates ORDER BY gap, source_index"
    ):
        while gap_bin < N_GAP_BINS - 1 and consumed_in_bin >= bin_sizes[gap_bin]:
            gap_bin += 1
            consumed_in_bin = 0
        updates.append((gap_bin, int(source_index)))
        consumed_in_bin += 1
        if len(updates) >= 50_000:
            connection.executemany(
                "UPDATE candidates SET gap_bin=? WHERE source_index=?", updates
            )
            updates.clear()
    if updates:
        connection.executemany(
            "UPDATE candidates SET gap_bin=? WHERE source_index=?", updates
        )
    connection.executescript(
        """
        CREATE INDEX candidates_gap_order_idx
            ON candidates(gap_bin, gap, source_index);
        CREATE TABLE candidates_ranked AS
            SELECT source_index,
                   canonical_smiles,
                   gap,
                   scaffold,
                   scaffold_kind,
                   gap_bin,
                   ROW_NUMBER() OVER (
                       PARTITION BY gap_bin, scaffold
                       ORDER BY source_index
                   ) - 1 AS scaffold_round,
                   fingerprint,
                   minhash_key,
                   selection_rank
            FROM candidates;
        DROP TABLE candidates;
        ALTER TABLE candidates_ranked RENAME TO candidates;
        CREATE UNIQUE INDEX candidates_source_idx ON candidates(source_index);
        CREATE INDEX candidates_round_idx
            ON candidates(gap_bin, scaffold_round, minhash_key, source_index);
        """
    )
    connection.commit()
    bins = []
    counts: dict[int, int] = {}
    for current_bin in range(N_GAP_BINS):
        count, minimum, maximum = connection.execute(
            "SELECT COUNT(*), MIN(gap), MAX(gap) "
            "FROM candidates WHERE gap_bin=?",
            (current_bin,),
        ).fetchone()
        counts[current_bin] = int(count)
        bins.append(
            {
                "gap_bin": current_bin,
                "count": int(count),
                "min_gap": float(minimum) if minimum is not None else None,
                "max_gap": float(maximum) if maximum is not None else None,
            }
        )
    return counts, {
        "method": "equal_frequency_rank_balanced_remainder_first",
        "n_bins": N_GAP_BINS,
        "tie_breaker": ["gap", "source_index"],
        "bins": bins,
    }


def _fingerprint_to_blob(fingerprint: Any) -> bytes:
    return bytes(DataStructs.BitVectToBinaryText(fingerprint))


def _fingerprint_from_blob(blob: bytes):
    return DataStructs.CreateFromBinaryText(bytes(blob))


def _fingerprint_payload(argument: tuple[int, str, int]) -> tuple[int, bytes, bytes]:
    source_index, canonical_smiles, seed = argument
    fingerprint = _morgan_fp_from_smiles(canonical_smiles)
    signature = _minhash_signature(fingerprint, _minhash_coefficients(seed))
    minhash_key = struct.pack(f">{_MINHASH_PERMUTATIONS}Q", *signature)
    return (
        int(source_index),
        _fingerprint_to_blob(fingerprint),
        minhash_key,
    )


def _materialize_fingerprints_once(
    connection: sqlite3.Connection,
    *,
    seed: int,
    block_size: int,
    executor: ProcessPoolExecutor | None,
    workers: int,
) -> None:
    cursor = connection.execute(
        "SELECT source_index, canonical_smiles FROM candidates ORDER BY source_index"
    )
    while True:
        rows = cursor.fetchmany(block_size)
        if not rows:
            break
        arguments = (
            (int(source_index), canonical_smiles, seed)
            for source_index, canonical_smiles in rows
        )
        if executor is None:
            payloads = map(_fingerprint_payload, arguments)
        else:
            payloads = executor.map(
                _fingerprint_payload,
                arguments,
                chunksize=max(1, len(rows) // max(1, workers * 8)),
            )
        updates = []
        for source_index, fingerprint_blob, minhash_key in payloads:
            updates.append(
                (
                    sqlite3.Binary(fingerprint_blob),
                    sqlite3.Binary(minhash_key),
                    int(source_index),
                )
            )
        connection.executemany(
            "UPDATE candidates SET fingerprint=?, minhash_key=? "
            "WHERE source_index=?",
            updates,
        )
        connection.commit()


def _rank_one_round_approximate(
    connection: sqlite3.Connection,
    *,
    gap_bin: int,
    scaffold_round: int,
    seed: int,
    bucket_size: int,
    exact_threshold: int,
) -> Iterator[int]:
    connection.executescript(
        """
        DELETE FROM bucket_members;
        DELETE FROM bucket_order;
        """
    )
    cursor = connection.execute(
        """
        SELECT source_index, fingerprint
        FROM candidates
        WHERE gap_bin=? AND scaffold_round=?
        ORDER BY minhash_key, source_index
        """,
        (gap_bin, scaffold_round),
    )
    representatives: list[Any] = []
    representative_ids: list[int] = []
    bucket_id = 0
    while True:
        bucket = cursor.fetchmany(bucket_size)
        if not bucket:
            break
        identities = [int(row[0]) for row in bucket]
        fingerprints = [_fingerprint_from_blob(row[1]) for row in bucket]
        local_order = _exact_maxmin_order(
            fingerprints,
            identities,
            seed + 1_000_003 * (bucket_id + 1),
        )
        connection.executemany(
            "INSERT INTO bucket_members(bucket_id, local_rank, source_index) "
            "VALUES (?, ?, ?)",
            (
                (bucket_id, local_rank, identities[relative_index])
                for local_rank, relative_index in enumerate(local_order)
            ),
        )
        representative_index = local_order[0]
        representatives.append(fingerprints[representative_index])
        representative_ids.append(identities[representative_index])
        bucket_id += 1
    if len(representatives) <= exact_threshold:
        representative_order = _exact_maxmin_order(
            representatives, representative_ids, seed + 97_409
        )
    else:
        representative_order = _hierarchical_approximate_order(
            representatives,
            representative_ids,
            seed=seed + 97_409,
            bucket_size=bucket_size,
            exact_threshold=exact_threshold,
        )
    connection.executemany(
        "INSERT INTO bucket_order(bucket_id, representative_order) VALUES (?, ?)",
        (
            (bucket_index, order_rank)
            for order_rank, bucket_index in enumerate(representative_order)
        ),
    )
    connection.commit()
    ordered_cursor = connection.execute(
        """
        SELECT m.source_index
        FROM bucket_members AS m
        JOIN bucket_order AS o ON m.bucket_id = o.bucket_id
        ORDER BY m.local_rank, o.representative_order
        """
    )
    for (source_index,) in ordered_cursor:
        yield int(source_index)


def _rank_candidates_external(
    connection: sqlite3.Connection,
    *,
    seed: int,
    mode: str,
    bucket_size: int,
    exact_threshold: int,
    exact_max_candidates: int,
) -> None:
    if mode not in {"approximate", "exact"}:
        raise ValueError("selection_mode must be 'approximate' or 'exact'")
    total_candidates = int(
        connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    )
    if mode == "exact" and total_candidates > exact_max_candidates:
        raise ValueError(
            "exact selection exceeds configured total safety limit: "
            f"limit={exact_max_candidates}, candidates={total_candidates}"
        )
    connection.executescript(
        """
        CREATE TABLE bucket_members (
            bucket_id INTEGER NOT NULL,
            local_rank INTEGER NOT NULL,
            source_index INTEGER NOT NULL
        );
        CREATE TABLE bucket_order (
            bucket_id INTEGER PRIMARY KEY,
            representative_order INTEGER NOT NULL
        );
        CREATE INDEX bucket_members_merge_idx
            ON bucket_members(local_rank, bucket_id);
        CREATE TABLE ranking_rounds AS
            SELECT gap_bin,
                   scaffold_round,
                   COUNT(*) AS candidate_count,
                   CASE WHEN COUNT(*) = 1
                        THEN MIN(source_index)
                        ELSE NULL
                   END AS singleton_source_index
            FROM candidates
            GROUP BY gap_bin, scaffold_round;
        CREATE UNIQUE INDEX ranking_rounds_order_idx
            ON ranking_rounds(gap_bin, scaffold_round);
        """
    )
    next_rank = [0] * N_GAP_BINS
    last_gap_bin = -1
    last_scaffold_round = -1
    while True:
        rounds = connection.execute(
            """
            SELECT gap_bin, scaffold_round, candidate_count,
                   singleton_source_index
            FROM ranking_rounds
            WHERE gap_bin > ?
               OR (gap_bin = ? AND scaffold_round > ?)
            ORDER BY gap_bin, scaffold_round
            LIMIT 1024
            """,
            (last_gap_bin, last_gap_bin, last_scaffold_round),
        ).fetchall()
        if not rounds:
            break
        updates: list[tuple[int, int]] = []
        for gap_bin, scaffold_round, round_count, singleton_source_index in rounds:
            round_seed = seed + 10_007 * int(gap_bin) + int(scaffold_round)
            ordered_sources: Iterable[int]
            if int(round_count) == 1:
                if singleton_source_index is None:
                    raise RuntimeError("singleton ranking round is missing its source")
                ordered_sources = (int(singleton_source_index),)
            elif mode == "exact":
                if int(round_count) > exact_max_candidates:
                    raise ValueError(
                        "exact selection exceeds configured safety limit: "
                        f"limit={exact_max_candidates}, candidates={round_count}, "
                        f"gap_bin={gap_bin}, scaffold_round={scaffold_round}"
                    )
                rows = connection.execute(
                    """
                    SELECT source_index, fingerprint
                    FROM candidates
                    WHERE gap_bin=? AND scaffold_round=?
                    ORDER BY source_index
                    """,
                    (gap_bin, scaffold_round),
                ).fetchall()
                identities = [int(row[0]) for row in rows]
                fingerprints = [_fingerprint_from_blob(row[1]) for row in rows]
                relative_order = _exact_maxmin_order(
                    fingerprints, identities, round_seed
                )
                ordered_sources = (
                    identities[index] for index in relative_order
                )
            else:
                ordered_sources = _rank_one_round_approximate(
                    connection,
                    gap_bin=int(gap_bin),
                    scaffold_round=int(scaffold_round),
                    seed=round_seed,
                    bucket_size=bucket_size,
                    exact_threshold=exact_threshold,
                )
            for source_index in ordered_sources:
                updates.append((next_rank[int(gap_bin)], int(source_index)))
                next_rank[int(gap_bin)] += 1
                if len(updates) >= 50_000:
                    connection.executemany(
                        "UPDATE candidates SET selection_rank=? WHERE source_index=?",
                        updates,
                    )
                    updates.clear()
            if int(round_count) > 1:
                if updates:
                    connection.executemany(
                        "UPDATE candidates SET selection_rank=? WHERE source_index=?",
                        updates,
                    )
                    updates.clear()
                connection.commit()
            last_gap_bin = int(gap_bin)
            last_scaffold_round = int(scaffold_round)
        if updates:
            connection.executemany(
                "UPDATE candidates SET selection_rank=? WHERE source_index=?",
                updates,
            )
            updates.clear()
        connection.commit()
    connection.executescript(
        """
        DROP TABLE bucket_members;
        DROP TABLE bucket_order;
        DROP TABLE ranking_rounds;
        CREATE UNIQUE INDEX candidates_selection_rank_idx
            ON candidates(gap_bin, selection_rank);
        """
    )
    for gap_bin in range(N_GAP_BINS):
        count, minimum, maximum, distinct_count = connection.execute(
            """
            SELECT COUNT(*), MIN(selection_rank), MAX(selection_rank),
                   COUNT(DISTINCT selection_rank)
            FROM candidates WHERE gap_bin=?
            """,
            (gap_bin,),
        ).fetchone()
        if count and (
            int(minimum) != 0
            or int(maximum) != int(count) - 1
            or int(distinct_count) != int(count)
        ):
            raise RuntimeError(
                f"incomplete selection ranking for gap_bin={gap_bin}"
            )
    connection.commit()


def _write_manifest_from_database(
    connection: sqlite3.Connection,
    path: Path,
    *,
    target_sizes: Sequence[int],
    batch_size: int,
) -> int:
    pa, pq = _require_pyarrow()
    schema = _manifest_schema(pa)
    quotas = {
        target: target_quotas(target, N_GAP_BINS)
        for target in target_sizes
    }
    cursor = connection.execute(
        """
        SELECT source_index, canonical_smiles, gap, gap_bin, scaffold,
               scaffold_kind, selection_rank
        FROM candidates
        ORDER BY gap_bin, selection_rank
        """
    )
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    written = 0
    try:
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            selected_targets = [
                [
                    int(target)
                    for target in target_sizes
                    if int(row[6]) < quotas[target][int(row[3])]
                ]
                for row in rows
            ]
            arrays = [
                pa.array([int(row[0]) for row in rows], type=pa.int64()),
                pa.array([row[1] for row in rows], type=pa.string()),
                pa.array([float(row[2]) for row in rows], type=pa.float64()),
                pa.array([int(row[3]) for row in rows], type=pa.int8()),
                pa.array([row[4] for row in rows], type=pa.string()),
                pa.array([row[5] for row in rows], type=pa.string()),
                pa.array([int(row[6]) for row in rows], type=pa.int64()),
                pa.array(selected_targets, type=pa.list_(pa.int64())),
            ]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
            written += len(rows)
    finally:
        writer.close()
    return written


def _write_target_indices_from_database(
    connection: sqlite3.Connection,
    path: Path,
    *,
    target_size: int,
) -> None:
    quotas = target_quotas(target_size, N_GAP_BINS)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            '{"schema":"semmol.pcqm_indices.v1","target_size":'
            + str(int(target_size))
            + ',"source_indices":['
        )
        first = True
        for gap_bin, quota in enumerate(quotas):
            cursor = connection.execute(
                """
                SELECT source_index FROM candidates
                WHERE gap_bin=? AND selection_rank<?
                ORDER BY selection_rank
                """,
                (gap_bin, quota),
            )
            for (source_index,) in cursor:
                if not first:
                    stream.write(",")
                stream.write(str(int(source_index)))
                first = False
                count += 1
        stream.write("]}\n")
    if count != target_size:
        raise RuntimeError(
            f"internal selection error for target={target_size}: produced={count}"
        )


def _cleanup_tree(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as cleanup_error:
        print(
            f"warning: could not remove temporary directory {path}: {cleanup_error}",
            file=sys.stderr,
        )


def _cleanup_staging_database(path: Path, *, strict: bool = False) -> None:
    errors: list[tuple[Path, OSError]] = []
    for candidate in (
        path,
        Path(str(path) + "-journal"),
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
    ):
        try:
            candidate.unlink(missing_ok=True)
        except OSError as cleanup_error:
            errors.append((candidate, cleanup_error))
            print(
                f"warning: could not remove staging file {candidate}: {cleanup_error}",
                file=sys.stderr,
            )
    if strict and errors:
        failed_paths = ", ".join(str(candidate) for candidate, _error in errors)
        raise RuntimeError(
            f"could not remove staging database files before commit: {failed_paths}"
        )


def _file_integrity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(stream.fileno())
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field, None) != getattr(after, field, None)
        for field in stable_fields
    ):
        raise RuntimeError(f"file changed while hashing: {path}")
    current = path.stat()
    if any(
        getattr(after, field, None) != getattr(current, field, None)
        for field in stable_fields
    ):
        raise RuntimeError(f"file identity changed while hashing: {path}")
    return {
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def _prepare_source_snapshot(
    source: Path,
    destination: Path,
    *,
    expected_integrity: Mapping[str, Any],
) -> Path:
    """Copy and pin the exact source bytes consumed by this generation."""
    temporary = _temporary_path(destination)
    try:
        with source.open("rb") as source_stream, temporary.open(
            "wb"
        ) as snapshot_stream:
            shutil.copyfileobj(
                source_stream,
                snapshot_stream,
                length=1024 * 1024,
            )
            snapshot_stream.flush()
            os.fsync(snapshot_stream.fileno())
        if _file_integrity(temporary) != dict(expected_integrity):
            raise RuntimeError(
                "source manifest changed while creating the private snapshot"
            )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def resolve_pcqm_generation(
    current_path: str | os.PathLike[str], *, verify_integrity: bool = True
) -> dict[str, Any]:
    """Resolve and optionally verify an atomically committed PCQM generation."""

    pointer_path = Path(current_path)
    pointer_raw = pointer_path.read_bytes()
    pointer_sha256 = hashlib.sha256(pointer_raw).hexdigest()
    try:
        pointer = json.loads(pointer_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CURRENT must contain valid UTF-8 JSON") from exc
    if not isinstance(pointer, dict):
        raise ValueError("CURRENT must contain a JSON object")
    if pointer.get("schema") != "semmol.pcqm_selection.current.v1":
        raise ValueError(f"unsupported CURRENT schema: {pointer.get('schema')!r}")
    generation_id = pointer.get("generation_id")
    generation_path = pointer.get("generation_path")
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or not isinstance(generation_path, str)
        or Path(generation_path).parts
        != ("pcqm_generations", generation_id)
    ):
        raise ValueError("CURRENT contains an invalid generation path")
    generation = (pointer_path.parent / generation_path).resolve()
    output_root = pointer_path.parent.resolve()
    if output_root not in generation.parents:
        raise ValueError("CURRENT generation_path escapes the output directory")
    if generation_id != generation.name:
        raise ValueError("CURRENT generation_id does not match generation_path")
    names = {
        "manifest": pointer.get("manifest"),
        "metadata": pointer.get("metadata"),
        "failures": pointer.get("failures"),
    }
    target_names = pointer.get("targets")
    if not isinstance(target_names, dict):
        raise ValueError("CURRENT is missing the committed target-file mapping")
    for target_text, name in target_names.items():
        try:
            target = int(target_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("CURRENT contains an invalid target size") from exc
        if target < 1 or str(target) != target_text:
            raise ValueError("CURRENT contains an invalid target size")
        names[f"target_{target}"] = name
    if any(not isinstance(name, str) or Path(name).name != name for name in names.values()):
        raise ValueError("CURRENT contains invalid generation filenames")
    if len(set(names.values())) != len(names):
        raise ValueError("CURRENT maps multiple artifacts to the same filename")
    resolved = {key: generation / name for key, name in names.items()}
    inventory = pointer.get("files")
    if not isinstance(inventory, dict):
        raise ValueError("CURRENT is missing the committed file inventory")
    if set(inventory) != set(names.values()):
        raise ValueError(
            "CURRENT file inventory does not exactly match the artifact mapping"
        )
    for filename, expected in inventory.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("CURRENT inventory contains an invalid filename")
        path = generation / filename
        if (
            not path.is_file()
            or not isinstance(expected, dict)
            or set(expected) != {"size_bytes", "sha256"}
            or not isinstance(expected.get("size_bytes"), int)
            or isinstance(expected["size_bytes"], bool)
            or expected["size_bytes"] < 0
            or not isinstance(expected.get("sha256"), str)
            or len(expected["sha256"]) != 64
            or set(expected["sha256"]) - set("0123456789abcdef")
        ):
            raise ValueError(f"committed file {filename} is missing from the generation")
        if verify_integrity and _file_integrity(path) != expected:
            raise ValueError(f"integrity verification failed for {filename}")
    for key, path in resolved.items():
        if path.name not in inventory:
            raise ValueError(f"committed {key} file is missing from the inventory")
    resolved["current"] = pointer_path
    resolved["generation"] = generation
    resolved["current_sha256"] = pointer_sha256
    resolved["manifest_sha256"] = str(
        inventory[str(names["manifest"])]["sha256"]
    )
    resolved["metadata_sha256"] = str(
        inventory[str(names["metadata"])]["sha256"]
    )
    return resolved


def filter_pcqm_dataset(
    source_manifest: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] = "data/processed/pcqm/manifests",
    target_sizes: Sequence[int] = (1_000_000, 3_000_000),
    seed: int = 3407,
    smiles_col: str = "smiles",
    gap_col: str = "homolumogap",
    source_index_col: str = "source_index",
    official_split_col: str = "official_split",
    allow_missing_official_split: bool = False,
    selection_mode: str = "approximate",
    bucket_size: int = DEFAULT_BUCKET_SIZE,
    exact_threshold: int = DEFAULT_EXACT_THRESHOLD,
    exact_max_candidates: int = 20_000,
    block_size: int = 100_000,
    workers: int = 1,
    staging_dir: str | os.PathLike[str] | None = None,
    input_path: str | os.PathLike[str] | None = None,
    raw_csv_path: str | os.PathLike[str] | None = None,
) -> dict[str, Path]:
    """Build one atomically committed external-memory selection generation."""

    supplied = [
        Path(value)
        for value in (source_manifest, input_path, raw_csv_path)
        if value is not None
    ]
    if not supplied:
        source_path = Path("data/processed/pcqm/source_manifest.parquet")
    else:
        source_path = supplied[0]
        if any(path != source_path for path in supplied[1:]):
            raise ValueError(
                "source_manifest, input_path, and raw_csv_path refer to different files"
            )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    _require_pyarrow()
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if bucket_size < 2:
        raise ValueError("bucket_size must be at least 2")
    if exact_threshold < 2:
        raise ValueError("exact_threshold must be at least 2")
    if exact_max_candidates < 1:
        raise ValueError("exact_max_candidates must be positive")
    if selection_mode not in {"approximate", "exact"}:
        raise ValueError("selection_mode must be 'approximate' or 'exact'")
    column_names = [smiles_col, gap_col, source_index_col, official_split_col]
    if any(not name for name in column_names) or len(set(column_names)) != 4:
        raise ValueError(
            "smiles, gap, source-index, and official-split columns must be distinct"
        )
    targets = _validated_target_sizes(target_sizes)
    source_integrity = _file_integrity(source_path)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    generations_root = output_root / "pcqm_generations"
    generations_root.mkdir(parents=True, exist_ok=True)
    generation_id = uuid.uuid4().hex
    temporary_generation = output_root / f".pcqm_generation_{generation_id}.tmp"
    committed_generation = generations_root / generation_id
    temporary_generation.mkdir()
    source_snapshot_path = (
        temporary_generation
        / f".source-manifest.snapshot{source_path.suffix.lower()}"
    )
    staging_root = Path(staging_dir) if staging_dir is not None else temporary_generation
    staging_path = staging_root / f".pcqm_selection_{generation_id}.sqlite"
    manifest_name = "pcqm_selection_manifest.parquet"
    metadata_name = "pcqm_selection_metadata.json"
    failures_name = "pcqm_selection_failures.jsonl"
    manifest_path = temporary_generation / manifest_name
    metadata_path = temporary_generation / metadata_name
    failures_path = temporary_generation / failures_name
    target_paths = {
        target: temporary_generation / f"pcqm_{target}_indices.json"
        for target in targets
    }

    failure_counts: Counter[str] = Counter()
    official_split_counts: Counter[str] = Counter()
    input_rows = 0
    train_candidate_rows = 0
    excluded_non_train_rows = 0
    connection: sqlite3.Connection | None = None
    executor: ProcessPoolExecutor | None = None
    try:
        _prepare_source_snapshot(
            source_path,
            source_snapshot_path,
            expected_integrity=source_integrity,
        )
        staging_root.mkdir(parents=True, exist_ok=True)
        if workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
            )
        connection = _connect_staging(staging_path)
        required_columns = [smiles_col, gap_col, source_index_col]
        optional_columns: list[str] = []
        if allow_missing_official_split:
            optional_columns.append(official_split_col)
        else:
            required_columns.append(official_split_col)
        with failures_path.open("w", encoding="utf-8", newline="\n") as failure_stream:
            for block, _physical_indices in _iter_input_blocks(
                source_snapshot_path,
                required_columns=tuple(required_columns),
                optional_columns=tuple(optional_columns),
                block_size=block_size,
            ):
                input_rows += len(block)
                source_indices = _validated_source_indices(
                    block[source_index_col].tolist(),
                    column_name=source_index_col,
                )
                _register_source_indices(connection, source_indices)
                if official_split_col in block.columns:
                    normalized_split = (
                        block[official_split_col]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )
                    if "" in set(normalized_split):
                        raise ValueError(
                            f"official split column {official_split_col!r} contains "
                            "missing or empty values"
                        )
                    official_split_counts.update(normalized_split.tolist())
                    train_mask = normalized_split.eq("train").to_numpy()
                else:
                    train_mask = np.ones(len(block), dtype=bool)
                excluded_non_train_rows += int((~train_mask).sum())
                block = block.loc[train_mask]
                source_indices = [
                    source_index
                    for source_index, is_train in zip(
                        source_indices, train_mask.tolist()
                    )
                    if is_train
                ]
                train_candidate_rows += len(block)
                records, failures = _prepare_block_without_deduplication(
                    block,
                    smiles_col=smiles_col,
                    gap_col=gap_col,
                    source_indices=source_indices,
                    executor=executor,
                    workers=workers,
                )
                for failure in failures:
                    failure_counts[failure["reason"]] += 1
                    failure_stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                connection.executemany(
                    """
                    INSERT INTO valid_rows(
                        source_index, canonical_smiles, gap, scaffold, scaffold_kind
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            int(record["source_index"]),
                            record["canonical_smiles"],
                            float(record["gap"]),
                            record["scaffold"],
                            record["scaffold_kind"],
                        )
                        for record in records
                    ),
                )
                connection.commit()
            candidate_count = _materialize_unique_candidates(
                connection, failure_stream, failure_counts
            )

        bin_counts, gap_metadata = _assign_gap_bins_external(
            connection, candidate_count
        )
        validate_capacity(bin_counts, targets, n_bins=N_GAP_BINS)
        if selection_mode == "exact" and candidate_count > exact_max_candidates:
            raise ValueError(
                "exact selection exceeds configured total safety limit: "
                f"limit={exact_max_candidates}, candidates={candidate_count}"
            )
        _materialize_fingerprints_once(
            connection,
            seed=seed,
            block_size=block_size,
            executor=executor,
            workers=workers,
        )
        if executor is not None:
            executor.shutdown(wait=True)
            executor = None
        _rank_candidates_external(
            connection,
            seed=seed,
            mode=selection_mode,
            bucket_size=bucket_size,
            exact_threshold=exact_threshold,
            exact_max_candidates=exact_max_candidates,
        )
        algorithm = (
            "scaffold_round_exact_maxmin"
            if selection_mode == "exact"
            else "scaffold_round_minhash_bucketed_maxmin"
        )
        final_source_integrity = _file_integrity(source_path)
        final_snapshot_integrity = _file_integrity(source_snapshot_path)
        if final_snapshot_integrity != source_integrity:
            raise RuntimeError(
                "private source snapshot changed during selection processing; "
                f"before={source_integrity}, after={final_snapshot_integrity}"
            )
        if final_source_integrity != source_integrity:
            raise RuntimeError(
                "source manifest changed during selection processing; "
                f"before={source_integrity}, after={final_source_integrity}"
            )
        metadata = {
            "schema": SCHEMA_VERSION,
            "generation_id": generation_id,
            "input": {
                "path": str(source_path),
                "format": source_path.suffix.lower().lstrip("."),
                "integrity": source_integrity,
                "smiles_column": smiles_col,
                "gap_column": gap_col,
                "source_index_column": source_index_col,
                "official_split_column": (
                    official_split_col
                    if official_split_counts
                    else None
                ),
                "official_split_required": not allow_missing_official_split,
                "official_split_counts": dict(sorted(official_split_counts.items())),
                "rows": input_rows,
                "train_candidate_rows": train_candidate_rows,
                "excluded_non_train_rows": excluded_non_train_rows,
            },
            "canonicalization": {
                "isomeric": True,
                "deduplication_key": "canonical_smiles",
                "duplicate_winner": "minimum_source_index",
            },
            "execution": {
                "storage": "sqlite_external_staging",
                "block_size": block_size,
                "workers": workers,
                "process_start_method": "spawn" if workers > 1 else None,
                "fingerprint_materializations_per_candidate": 1,
                "parquet_writer": "streaming_column_batches",
            },
            "statistics": {
                "input_rows": input_rows,
                "valid_unique_rows": candidate_count,
                "failure_counts": dict(sorted(failure_counts.items())),
                "gap_bin_counts": {
                    str(gap_bin): int(bin_counts[gap_bin])
                    for gap_bin in range(N_GAP_BINS)
                },
            },
            "gap_binning": gap_metadata,
            "selection": {
                "algorithm": algorithm,
                "algorithm_scope": "maxmin_within_scaffold_rounds",
                "seed": seed,
                "target_sizes": targets,
                "target_quotas": {
                    str(target): target_quotas(target, N_GAP_BINS)
                    for target in targets
                },
                "complete_candidate_ranking": True,
                "selection_rank_scope": "within_gap_bin",
                "selected_targets_semantics": "initial_selection_only",
                "scaffold_policy": "round_robin_members",
                "fingerprint": {
                    "type": "Morgan",
                    "radius": MORGAN_RADIUS,
                    "n_bits": MORGAN_BITS,
                    "distance": "1-Tanimoto",
                },
                "bucket_size": bucket_size if selection_mode == "approximate" else None,
                "exact_threshold": (
                    exact_threshold if selection_mode == "approximate" else None
                ),
                "exact_max_candidates": (
                    exact_max_candidates if selection_mode == "exact" else None
                ),
                "approximation": (
                    {
                        "bucketing": "deterministic_minhash_lexicographic",
                        "minhash_permutations": _MINHASH_PERMUTATIONS,
                        "bucket_merge": "round_robin",
                        "global_fps_claim": False,
                    }
                    if selection_mode == "approximate"
                    else None
                ),
            },
        }

        written = _write_manifest_from_database(
            connection,
            manifest_path,
            target_sizes=targets,
            batch_size=block_size,
        )
        if written != candidate_count:
            raise RuntimeError(
                f"manifest row count mismatch: expected={candidate_count}, written={written}"
            )
        _write_json(metadata_path, metadata)
        for target, destination in target_paths.items():
            _write_target_indices_from_database(
                connection,
                destination,
                target_size=target,
            )
        generation_files = {
            manifest_name: _file_integrity(manifest_path),
            metadata_name: _file_integrity(metadata_path),
            failures_name: _file_integrity(failures_path),
            **{
                path.name: _file_integrity(path)
                for path in target_paths.values()
            },
        }
        connection.close()
        connection = None
        _cleanup_staging_database(staging_path, strict=True)
        source_snapshot_path.unlink()
        os.replace(temporary_generation, committed_generation)
        current_path = output_root / "pcqm_selection_CURRENT.json"
        current_temporary = _temporary_path(current_path)
        try:
            _write_json(
                current_temporary,
                {
                    "schema": "semmol.pcqm_selection.current.v1",
                    "generation_id": generation_id,
                    "generation_path": str(
                        Path("pcqm_generations") / generation_id
                    ),
                    "manifest": manifest_name,
                    "metadata": metadata_name,
                    "failures": failures_name,
                    "targets": {
                        str(target): path.name
                        for target, path in target_paths.items()
                    },
                    "files": generation_files,
                },
            )
            os.replace(current_temporary, current_path)
        except BaseException:
            current_temporary.unlink(missing_ok=True)
            raise
        committed_targets = {
            target: committed_generation / path.name
            for target, path in target_paths.items()
        }
        return {
            "manifest": committed_generation / manifest_name,
            "metadata": committed_generation / metadata_name,
            "failures": committed_generation / failures_name,
            "current": current_path,
            **{
                f"target_{target}": path
                for target, path in committed_targets.items()
            },
        }
    except BaseException:
        if connection is not None:
            connection.close()
        _cleanup_staging_database(staging_path)
        _cleanup_tree(temporary_generation)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def _query_excluded_in_batch(
    connection: sqlite3.Connection, source_indices: Sequence[int]
) -> set[int]:
    excluded: set[int] = set()
    for chunk in _chunks(source_indices):
        placeholders = ",".join("?" for _ in chunk)
        excluded.update(
            int(row[0])
            for row in connection.execute(
                f"SELECT source_index FROM exclusions "
                f"WHERE source_index IN ({placeholders})",
                tuple(chunk),
            )
        )
    return excluded


def write_backfilled_selection(
    *,
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    target_sizes: Sequence[int],
    excluded_source_indices: Iterable[int] = (),
    batch_size: int = 65_536,
    staging_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Stream a nested, per-bin exact selection after applying exclusions."""

    pa, pq = _require_pyarrow()
    source = Path(manifest_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("backfill output_path must differ from manifest_path")
    targets = _validated_target_sizes(target_sizes)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not source.is_file():
        raise FileNotFoundError(source)
    quotas = {target: target_quotas(target, N_GAP_BINS) for target in targets}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    exclusion_db: Path | None = None
    connection: sqlite3.Connection | None = None
    writer = None
    try:
        staging_root = (
            Path(staging_dir) if staging_dir is not None else destination.parent
        )
        staging_root.mkdir(parents=True, exist_ok=True)
        descriptor, exclusion_db_name = tempfile.mkstemp(
            prefix=".pcqm_exclusions.",
            suffix=".sqlite",
            dir=staging_root,
        )
        exclusion_db = Path(exclusion_db_name)
        os.close(descriptor)
        connection = sqlite3.connect(exclusion_db)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute(
            "CREATE TABLE exclusions(source_index INTEGER PRIMARY KEY)"
        )
        connection.execute(
            "CREATE TABLE manifest_seen(source_index INTEGER PRIMARY KEY)"
        )
        exclusion_count = 0
        pending = []
        for value in excluded_source_indices:
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or int(value) < 0
            ):
                raise ValueError(
                    "excluded_source_indices must contain non-negative integers"
                )
            pending.append((int(value),))
            if len(pending) >= batch_size:
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO exclusions(source_index) VALUES (?)",
                    pending,
                )
                exclusion_count += connection.total_changes - before
                pending.clear()
                connection.commit()
        if pending:
            before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO exclusions(source_index) VALUES (?)",
                pending,
            )
            exclusion_count += connection.total_changes - before
            connection.commit()
        metadata = {
            b"schema": b"semmol.pcqm_backfilled_selection.v1",
            b"target_sizes": json.dumps(targets).encode("utf-8"),
            b"target_quotas": json.dumps(
                {str(target): quotas[target] for target in targets},
                sort_keys=True,
            ).encode("utf-8"),
            b"exclusion_count": str(exclusion_count).encode("ascii"),
        }
        schema = _backfill_schema(pa, metadata)
        parquet_file = pq.ParquetFile(source)
        required = {"source_index", "gap_bin", "selection_rank"}
        missing = sorted(required - set(parquet_file.schema.names))
        if missing:
            raise ValueError(f"selection manifest missing columns {missing}")
        for column_name in sorted(required):
            column_type = parquet_file.schema_arrow.field(column_name).type
            if not pa.types.is_integer(column_type):
                raise ValueError(
                    f"selection manifest column {column_name!r} must be integer, "
                    f"got {column_type}"
                )
        writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        selected_counts = {
            target: [0] * N_GAP_BINS for target in targets
        }
        expected_rank = [0] * N_GAP_BINS
        last_gap_bin = -1
        output_columns = [[], [], [], []]

        def flush_output() -> None:
            if not output_columns[0]:
                return
            arrays = [
                pa.array(output_columns[0], type=pa.int64()),
                pa.array(output_columns[1], type=pa.int8()),
                pa.array(output_columns[2], type=pa.int64()),
                pa.array(output_columns[3], type=pa.list_(pa.int64())),
            ]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
            for column in output_columns:
                column.clear()

        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["source_index", "gap_bin", "selection_rank"],
        ):
            if any(record_batch.column(index).null_count for index in range(3)):
                raise ValueError(
                    "selection manifest index/ranking columns must not contain nulls"
                )
            source_values = [
                int(value) for value in record_batch.column(0).to_pylist()
            ]
            if any(value < 0 for value in source_values):
                raise ValueError(
                    "selection manifest source_index values must be non-negative"
                )
            if len(set(source_values)) != len(source_values):
                raise ValueError("selection manifest contains duplicate source_index")
            for chunk in _chunks(source_values):
                placeholders = ",".join("?" for _ in chunk)
                existing = connection.execute(
                    f"SELECT source_index FROM manifest_seen "
                    f"WHERE source_index IN ({placeholders}) LIMIT 1",
                    tuple(chunk),
                ).fetchone()
                if existing is not None:
                    raise ValueError(
                        "selection manifest contains duplicate source_index="
                        f"{existing[0]}"
                    )
            connection.executemany(
                "INSERT INTO manifest_seen(source_index) VALUES (?)",
                ((value,) for value in source_values),
            )
            connection.commit()
            bin_values = [int(value) for value in record_batch.column(1).to_pylist()]
            rank_values = [
                int(value) for value in record_batch.column(2).to_pylist()
            ]
            excluded = _query_excluded_in_batch(connection, source_values)
            for source_index, gap_bin, selection_rank in zip(
                source_values, bin_values, rank_values
            ):
                if not 0 <= gap_bin < N_GAP_BINS:
                    raise ValueError(f"invalid gap_bin={gap_bin} in manifest")
                if gap_bin < last_gap_bin:
                    raise ValueError("manifest is not ordered by gap_bin")
                last_gap_bin = gap_bin
                if selection_rank != expected_rank[gap_bin]:
                    raise ValueError(
                        "manifest ranking is incomplete or unordered: "
                        f"gap_bin={gap_bin}, expected_rank={expected_rank[gap_bin]}, "
                        f"actual_rank={selection_rank}"
                    )
                expected_rank[gap_bin] += 1
                if source_index in excluded:
                    continue
                selected_for = []
                for target in targets:
                    if selected_counts[target][gap_bin] < quotas[target][gap_bin]:
                        selected_counts[target][gap_bin] += 1
                        selected_for.append(target)
                if selected_for:
                    output_columns[0].append(source_index)
                    output_columns[1].append(gap_bin)
                    output_columns[2].append(selection_rank)
                    output_columns[3].append(selected_for)
                    if len(output_columns[0]) >= batch_size:
                        flush_output()
        flush_output()
        for target in targets:
            if selected_counts[target] != quotas[target]:
                raise ValueError(
                    "insufficient non-excluded candidates for exact backfill: "
                    f"target_size={target}, required={quotas[target]}, "
                    f"produced={selected_counts[target]}"
                )
        writer.close()
        writer = None
        os.replace(temporary, destination)
        return destination
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if connection is not None:
            connection.close()
        if exclusion_db is not None:
            _cleanup_staging_database(exclusion_db)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic nested PCQM selection manifests"
    )
    parser.add_argument(
        "--source-manifest",
        "--source_manifest",
        "--input",
        "--raw_csv",
        dest="source_manifest",
        default="data/processed/pcqm/source_manifest.parquet",
        help="strict source manifest (.csv/.parquet) with source_index and official_split",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="data/processed/pcqm/manifests",
    )
    parser.add_argument(
        "--target-sizes",
        "--target_sizes",
        dest="target_sizes",
        type=int,
        nargs="+",
        default=[1_000_000, 3_000_000],
    )
    parser.add_argument(
        "--smiles-col",
        "--smiles_col",
        dest="smiles_col",
        default="smiles",
    )
    parser.add_argument(
        "--gap-col",
        "--gap_col",
        dest="gap_col",
        default="homolumogap",
    )
    parser.add_argument(
        "--source-index-col",
        "--source_index_col",
        dest="source_index_col",
        default="source_index",
        help="required stable original row-index column",
    )
    parser.add_argument(
        "--official-split-col",
        "--official_split_col",
        dest="official_split_col",
        default="official_split",
        help="official split column; only exact value 'train' is eligible",
    )
    parser.add_argument(
        "--allow-missing-official-split",
        "--allow_missing_official_split",
        dest="allow_missing_official_split",
        action="store_true",
        help="legacy override: treat every row as train when official_split is absent",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--selection-mode",
        "--selection_mode",
        dest="selection_mode",
        choices=("approximate", "exact"),
        default="approximate",
    )
    parser.add_argument(
        "--bucket-size",
        "--bucket_size",
        dest="bucket_size",
        type=int,
        default=DEFAULT_BUCKET_SIZE,
    )
    parser.add_argument(
        "--exact-threshold",
        "--exact_threshold",
        dest="exact_threshold",
        type=int,
        default=DEFAULT_EXACT_THRESHOLD,
    )
    parser.add_argument(
        "--exact-max-candidates",
        "--exact_max_candidates",
        dest="exact_max_candidates",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--block-size",
        "--block_size",
        dest="block_size",
        type=int,
        default=100_000,
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--staging-dir",
        "--staging_dir",
        dest="staging_dir",
        default=None,
        help="optional high-capacity local directory for SQLite staging",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = filter_pcqm_dataset(
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        target_sizes=args.target_sizes,
        seed=args.seed,
        smiles_col=args.smiles_col,
        gap_col=args.gap_col,
        source_index_col=args.source_index_col,
        official_split_col=args.official_split_col,
        allow_missing_official_split=args.allow_missing_official_split,
        selection_mode=args.selection_mode,
        bucket_size=args.bucket_size,
        exact_threshold=args.exact_threshold,
        exact_max_candidates=args.exact_max_candidates,
        block_size=args.block_size,
        workers=args.workers,
        staging_dir=args.staging_dir,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

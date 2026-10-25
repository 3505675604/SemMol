"""Auditable inner scaffold folds drawn exclusively from the outer train split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_scaffold_folds(
    source_indices: Sequence[int],
    scaffold_members: Mapping[str, Sequence[int]],
    *,
    n_folds: int = 10,
    seed: int = 3407,
) -> list[list[int]]:
    """Assign complete scaffold groups to balanced folds without label access."""
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")
    selected = set(int(x) for x in source_indices)
    if len(selected) != len(source_indices) or any(x < 0 for x in selected):
        raise ValueError("training source indices must be unique and non-negative")
    seen: set[int] = set()
    groups: list[tuple[str, list[int]]] = []
    for scaffold, members in scaffold_members.items():
        group = sorted(selected.intersection(int(x) for x in members))
        if not group:
            continue
        if seen.intersection(group):
            raise ValueError("a training molecule belongs to multiple scaffold groups")
        seen.update(group)
        groups.append((str(scaffold), group))
    if seen != selected:
        raise ValueError("scaffold metadata does not cover every training molecule")
    if len(groups) < n_folds:
        raise ValueError("not enough distinct training scaffolds for the requested folds")

    def tie(value: str) -> bytes:
        return hashlib.sha256(f"{seed}\0{value}".encode()).digest()

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for scaffold, members in sorted(groups, key=lambda x: (-len(x[1]), tie(x[0]))):
        index = min(range(n_folds), key=lambda i: (len(folds[i]), tie(f"{scaffold}:{i}")))
        folds[index].extend(members)
    return [sorted(fold) for fold in folds]


def build_training_folds(
    train_manifest: Path,
    scaffold_path: Path,
    output_dir: Path,
    *,
    n_folds: int = 10,
    seed: int = 3407,
) -> list[tuple[Path, Path]]:
    """Write NPZ folds and a provenance sidecar; never open outer valid/test."""
    import numpy as np

    train_manifest = train_manifest.resolve()
    scaffold_path = scaffold_path.resolve()
    with np.load(train_manifest, allow_pickle=False) as archive:
        records = np.asarray(archive["record_index"])
        sources = np.asarray(archive["source_index"])
    if records.ndim != 1 or sources.ndim != 1 or len(records) != len(sources):
        raise ValueError("train manifest must contain matching 1D record/source indices")
    if records.dtype.kind not in "iu" or sources.dtype.kind not in "iu":
        raise ValueError("manifest indices must be integers")
    if len(set(map(int, records))) != len(records):
        raise ValueError("training manifest contains duplicate records")
    metadata = json.loads(scaffold_path.read_text(encoding="utf-8"))
    outer_train = set(map(int, metadata["indices"]["train"]))
    if not set(map(int, sources)).issubset(outer_train):
        raise ValueError("CV pool includes molecules outside the outer 80% train split")
    folds = assign_scaffold_folds(
        sources.tolist(), metadata["scaffold_members"], n_folds=n_folds, seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []
    for index, held_out in enumerate(folds):
        validation_mask = np.isin(sources, np.asarray(held_out, dtype=np.int64))
        train_path = (output_dir / f"fold_{index:02d}_train.npz").resolve()
        valid_path = (output_dir / f"fold_{index:02d}_valid.npz").resolve()
        for path, mask in ((train_path, ~validation_mask), (valid_path, validation_mask)):
            np.savez(path, record_index=records[mask], source_index=sources[mask])
        sidecar = {
            "schema": "semmol.inner_scaffold_cv.v1",
            "purpose": "development-only",
            "pool": "outer_training_split_only",
            "parent_train_manifest": str(train_manifest),
            "parent_train_sha256": file_sha256(train_manifest),
            "scaffold_path": str(scaffold_path),
            "scaffold_sha256": file_sha256(scaffold_path),
            "n_folds": n_folds,
            "fold_index": index,
            "fold_seed": seed,
            "train_manifest": str(train_path),
            "train_sha256": file_sha256(train_path),
            "valid_manifest": str(valid_path),
            "valid_sha256": file_sha256(valid_path),
            "train_count": int((~validation_mask).sum()),
            "valid_count": int(validation_mask.sum()),
        }
        valid_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        pairs.append((train_path, valid_path))
    return pairs


def validate_fold_sidecar(sidecar_path: Path) -> dict:
    """Validate hashes, pool membership and group separation before model loading."""
    import numpy as np

    info = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if info.get("schema") != "semmol.inner_scaffold_cv.v1":
        raise ValueError("unsupported CV provenance schema")
    for field in ("parent_train", "scaffold", "train", "valid"):
        path_key = "scaffold_path" if field == "scaffold" else f"{field}_manifest"
        if file_sha256(Path(info[path_key])) != info[f"{field}_sha256"]:
            raise ValueError(f"{field} artifact changed since fold construction")

    def pairs(path: str) -> set[tuple[int, int]]:
        with np.load(path, allow_pickle=False) as archive:
            r, s = archive["record_index"], archive["source_index"]
            result = set(zip(map(int, r), map(int, s)))
            if len(result) != len(r) or len(r) != len(s):
                raise ValueError("fold manifest has duplicates or mismatched indices")
            return result

    train, valid = pairs(info["train_manifest"]), pairs(info["valid_manifest"])
    parent = pairs(info["parent_train_manifest"])
    if not train or not valid or train & valid or train | valid != parent:
        raise ValueError("folds must be a disjoint complete partition of outer training records")
    metadata = json.loads(Path(info["scaffold_path"]).read_text(encoding="utf-8"))
    train_sources = {source for _, source in train}
    valid_sources = {source for _, source in valid}
    if not (train_sources | valid_sources).issubset(set(metadata["indices"]["train"])):
        raise ValueError("fold provenance contains outer validation or test molecules")
    for members in metadata["scaffold_members"].values():
        members = set(members)
        if members & train_sources and members & valid_sources:
            raise ValueError("a scaffold crosses the inner train/validation boundary")
    return info

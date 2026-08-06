"""MoleculeNet data contract and sharded Dataset for nine downstream benchmarks.

This module only defines trusted column mappings, missing-label masks, and
training record read interfaces. Raw CSV SMILES cleaning, 3D/QM feature
construction, and scaffold splitting are done offline by preprocessing scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .pcqm_dataset import MODALITY_ORDER, PCQMMultimodalDataset


PathLike = Union[str, Path]


class MoleculeNetRegistryError(ValueError):
    """Dataset name, file, or columns do not conform to the registered MoleculeNet contract."""


@dataclass(frozen=True)
class MoleculeNetSpec:
    name: str
    directory_candidates: Tuple[str, ...]
    file_candidates: Tuple[str, ...]
    smiles_column: str
    label_columns: Tuple[str, ...]
    task_type: str
    num_tasks: int
    main_metric: str
    dynamic_label_columns: bool = False

    def validate(self) -> None:
        if self.task_type not in {"classification", "regression"}:
            raise MoleculeNetRegistryError(
                f"{self.name} task_type is invalid: {self.task_type}"
            )
        if self.num_tasks <= 0:
            raise MoleculeNetRegistryError(f"{self.name} num_tasks must be positive")
        if not self.dynamic_label_columns and len(self.label_columns) != self.num_tasks:
            raise MoleculeNetRegistryError(
                f"{self.name} label_columns={len(self.label_columns)} "
                f"!= num_tasks={self.num_tasks}"
            )
        if self.dynamic_label_columns and self.label_columns:
            raise MoleculeNetRegistryError(
                f"{self.name} dynamic label registration cannot also specify fixed label_columns"
            )


TOX21_LABELS = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


MOLECULENET_REGISTRY: Dict[str, MoleculeNetSpec] = {
    "bace": MoleculeNetSpec(
        name="bace",
        directory_candidates=("BACE", "bace"),
        file_candidates=("bace.csv", "BACE.csv", "raw.csv"),
        smiles_column="mol",
        label_columns=("Class",),
        task_type="classification",
        num_tasks=1,
        main_metric="roc_auc",
    ),
    "bbbp": MoleculeNetSpec(
        name="bbbp",
        directory_candidates=("BBBP", "bbbp"),
        file_candidates=("bbbp.csv", "BBBP.csv", "raw.csv"),
        smiles_column="smiles",
        label_columns=("p_np",),
        task_type="classification",
        num_tasks=1,
        main_metric="roc_auc",
    ),
    "clintox": MoleculeNetSpec(
        name="clintox",
        directory_candidates=("ClinTox", "clintox"),
        file_candidates=("clintox.csv", "clintox.csv.gz", "raw.csv"),
        smiles_column="smiles",
        label_columns=("FDA_APPROVED", "CT_TOX"),
        task_type="classification",
        num_tasks=2,
        main_metric="roc_auc",
    ),
    "tox21": MoleculeNetSpec(
        name="tox21",
        directory_candidates=("Tox21", "tox21"),
        file_candidates=("tox21.csv", "tox21.csv.gz", "raw.csv"),
        smiles_column="smiles",
        label_columns=TOX21_LABELS,
        task_type="classification",
        num_tasks=12,
        main_metric="roc_auc",
    ),
    "toxcast": MoleculeNetSpec(
        name="toxcast",
        directory_candidates=("ToxCast", "toxcast"),
        file_candidates=("toxcast_data.csv", "toxcast.csv", "raw.csv"),
        smiles_column="smiles",
        label_columns=(),
        task_type="classification",
        num_tasks=617,
        main_metric="roc_auc",
        dynamic_label_columns=True,
    ),
    "sider": MoleculeNetSpec(
        name="sider",
        directory_candidates=("SIDER", "sider"),
        file_candidates=("sider.csv", "sider.csv.gz", "raw.csv"),
        smiles_column="smiles",
        label_columns=(),
        task_type="classification",
        num_tasks=27,
        main_metric="roc_auc",
        dynamic_label_columns=True,
    ),
    "freesolv": MoleculeNetSpec(
        name="freesolv",
        directory_candidates=("FreeSolv", "freesolv"),
        file_candidates=("freesolv.csv", "FreeSolv.csv", "SAMPL.csv", "raw.csv"),
        smiles_column="smiles",
        label_columns=("expt",),
        task_type="regression",
        num_tasks=1,
        main_metric="rmse",
    ),
    "esol": MoleculeNetSpec(
        name="esol",
        directory_candidates=("ESOL", "esol", "Delaney"),
        file_candidates=("delaney-processed.csv", "esol.csv", "raw.csv"),
        smiles_column="smiles",
        label_columns=("measured log solubility in mols per litre",),
        task_type="regression",
        num_tasks=1,
        main_metric="rmse",
    ),
    "lipophilicity": MoleculeNetSpec(
        name="lipophilicity",
        directory_candidates=("Lipophilicity", "lipophilicity"),
        file_candidates=("Lipophilicity.csv", "lipophilicity.csv", "raw.csv"),
        smiles_column="smiles",
        label_columns=("exp",),
        task_type="regression",
        num_tasks=1,
        main_metric="rmse",
    ),
}

for _registered_spec in MOLECULENET_REGISTRY.values():
    _registered_spec.validate()


@dataclass(frozen=True)
class MoleculeNetRows:
    row_indices: np.ndarray
    smiles: Tuple[str, ...]
    labels: np.ndarray
    label_mask: np.ndarray
    label_columns: Tuple[str, ...]

    def validate(self) -> None:
        row_count = len(self.row_indices)
        if (
            self.row_indices.ndim != 1
            or self.row_indices.dtype.kind not in {"i", "u"}
            or np.any(self.row_indices < 0)
            or len(np.unique(self.row_indices)) != row_count
        ):
            raise MoleculeNetRegistryError(
                "row_indices must be a unique, non-negative, 1D integer array"
            )
        if len(self.smiles) != row_count:
            raise MoleculeNetRegistryError("smiles length does not match row_indices")
        if any(not isinstance(smiles, str) for smiles in self.smiles):
            raise MoleculeNetRegistryError("smiles must be a tuple of strings")
        if self.labels.ndim != 2 or self.labels.shape[0] != row_count:
            raise MoleculeNetRegistryError("labels must be (N, T)")
        if self.labels.dtype.kind != "f" or np.any(np.isinf(self.labels)):
            raise MoleculeNetRegistryError("labels must be a float array without Inf")
        if (
            self.label_mask.shape != self.labels.shape
            or self.label_mask.dtype.kind != "b"
            or not np.array_equal(self.label_mask, np.isfinite(self.labels))
        ):
            raise MoleculeNetRegistryError(
                "label_mask must exactly match the finite positions of labels"
            )
        if self.labels.shape[1] != len(self.label_columns):
            raise MoleculeNetRegistryError("labels column count does not match label_columns")
        if (
            any(
                not isinstance(column, str) or not column
                for column in self.label_columns
            )
            or len(set(self.label_columns)) != len(self.label_columns)
        ):
            raise MoleculeNetRegistryError(
                "label_columns must be a unique non-empty tuple of strings"
            )


def get_moleculenet_spec(name: str) -> MoleculeNetSpec:
    if not isinstance(name, str) or not name.strip():
        raise MoleculeNetRegistryError("MoleculeNet dataset name must be a non-empty string")
    normalized = name.strip().lower()
    try:
        return MOLECULENET_REGISTRY[normalized]
    except KeyError as exc:
        raise MoleculeNetRegistryError(
            f"Unknown MoleculeNet dataset {name!r}; "
            f"available: {sorted(MOLECULENET_REGISTRY)}"
        ) from exc


def resolve_moleculenet_csv(raw_root: PathLike, name: str) -> Path:
    spec = get_moleculenet_spec(name)
    root = Path(raw_root)
    attempted = []
    for directory in spec.directory_candidates:
        for filename in spec.file_candidates:
            candidate = root / directory / filename
            attempted.append(str(candidate))
            if candidate.is_file():
                return candidate
    for filename in spec.file_candidates:
        candidate = root / filename
        attempted.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find {spec.name} raw CSV; tried: {attempted}"
    )


def _resolve_label_columns(
    frame: pd.DataFrame,
    spec: MoleculeNetSpec,
) -> Tuple[str, ...]:
    if spec.smiles_column not in frame.columns:
        raise MoleculeNetRegistryError(
            f"{spec.name} missing SMILES column {spec.smiles_column!r}; "
            f"existing columns: {list(frame.columns)}"
        )
    if spec.dynamic_label_columns:
        label_columns = tuple(
            str(column)
            for column in frame.columns
            if str(column) != spec.smiles_column
        )
        if len(label_columns) != spec.num_tasks:
            raise MoleculeNetRegistryError(
                f"{spec.name} expected {spec.num_tasks} task columns, got "
                f"{len(label_columns)}; please confirm you are using the standard MoleculeNet file"
            )
        return label_columns
    missing = [column for column in spec.label_columns if column not in frame.columns]
    if missing:
        raise MoleculeNetRegistryError(
            f"{spec.name} missing label columns {missing}; existing columns: {list(frame.columns)}"
        )
    return spec.label_columns


def extract_moleculenet_rows(
    frame: pd.DataFrame,
    spec: MoleculeNetSpec,
) -> MoleculeNetRows:
    """Extract SMILES, labels, and missing-label mask from a raw DataFrame.

    Does not call ``dropna``, so output ``row_indices`` always point to the
    original DataFrame rows. SMILES validity is checked by the RDKit
    preprocessing stage; missing SMILES are represented as empty strings here
    so that failure reports can still locate the original row.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be pandas.DataFrame")
    spec.validate()
    label_columns = _resolve_label_columns(frame, spec)
    raw_row_indices = np.asarray(frame.index)
    if raw_row_indices.dtype.kind not in {"i", "u"}:
        raise MoleculeNetRegistryError(
            "DataFrame index must be raw integer row numbers; implicit truncation or conversion is not allowed"
        )
    if raw_row_indices.dtype.kind == "u" and raw_row_indices.size and int(
        raw_row_indices.max()
    ) > np.iinfo(np.int64).max:
        raise MoleculeNetRegistryError("DataFrame index exceeds int64 range")
    row_indices = raw_row_indices.astype(np.int64, copy=False)
    if np.any(row_indices < 0):
        raise MoleculeNetRegistryError("DataFrame index cannot be negative")
    if len(np.unique(row_indices)) != len(row_indices):
        raise MoleculeNetRegistryError("DataFrame index contains duplicate row numbers")

    smiles_series = frame[spec.smiles_column]
    smiles = tuple(
        "" if pd.isna(value) else str(value).strip()
        for value in smiles_series.tolist()
    )
    labels = np.full(
        (len(frame), len(label_columns)),
        np.nan,
        dtype=np.float32,
    )
    for task_index, column in enumerate(label_columns):
        raw = frame[column]
        blank_strings = raw.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        missing = raw.isna() | blank_strings
        numeric = pd.to_numeric(raw.mask(missing), errors="coerce")
        numeric_values = numeric.to_numpy(
            dtype=np.float64,
            copy=True,
            na_value=np.nan,
        )
        invalid = ~missing.to_numpy(dtype=np.bool_) & ~np.isfinite(
            numeric_values
        )
        if np.any(invalid):
            position = int(np.flatnonzero(invalid)[0])
            raise MoleculeNetRegistryError(
                f"{spec.name} invalid label row={int(row_indices[position])}, "
                f"column={column!r}, value={raw.iloc[position]!r}"
            )
        finite = np.isfinite(numeric_values)
        float32_limit = np.finfo(np.float32).max
        overflow = finite & (np.abs(numeric_values) > float32_limit)
        if np.any(overflow):
            position = int(np.flatnonzero(overflow)[0])
            raise MoleculeNetRegistryError(
                f"{spec.name} label exceeds float32 row={int(row_indices[position])}, "
                f"column={column!r}, value={raw.iloc[position]!r}"
            )
        converted = numeric_values.astype(np.float32)
        labels[:, task_index] = converted
    label_mask = np.isfinite(labels)

    if spec.task_type == "classification":
        finite_values = labels[label_mask]
        invalid_values = finite_values[
            (finite_values != 0.0) & (finite_values != 1.0)
        ]
        if invalid_values.size:
            unique = np.unique(invalid_values)[:10].tolist()
            raise MoleculeNetRegistryError(
                f"{spec.name} classification labels must be 0/1/NaN, found {unique}"
            )

    rows = MoleculeNetRows(
        row_indices=row_indices,
        smiles=smiles,
        labels=labels,
        label_mask=label_mask,
        label_columns=label_columns,
    )
    rows.validate()
    return rows


class MoleculeNetDataset(PCQMMultimodalDataset):
    """MoleculeNet sharded Dataset with task registration validation."""

    def __init__(
        self,
        dataset_name: str,
        store_dir: PathLike,
        manifest_path: PathLike,
        modalities: Sequence[str] = MODALITY_ORDER,
        strict: bool = True,
        expected_tokenizer_sha256: Optional[str] = None,
        readahead: bool = False,
        max_open_shards: int = 16,
    ) -> None:
        self.spec = get_moleculenet_spec(dataset_name)
        try:
            super().__init__(
                store_dir=store_dir,
                manifest_path=manifest_path,
                modalities=modalities,
                strict=strict,
                expected_tokenizer_sha256=expected_tokenizer_sha256,
                readahead=readahead,
                max_open_shards=max_open_shards,
            )
            if (
                self.build_manifest.get("schema")
                != "semmol.moleculenet_store_build.v1"
                or self.build_manifest.get("dataset_name")
                != self.spec.name
            ):
                raise MoleculeNetRegistryError(
                    "MoleculeNet build manifest schema/dataset does not match"
                )
            task_metadata = self.build_manifest.get("task")
            if (
                not isinstance(task_metadata, Mapping)
                or set(task_metadata) != {"type", "num_tasks", "main_metric"}
                or task_metadata.get("type") != self.spec.task_type
                or not isinstance(task_metadata.get("num_tasks"), int)
                or isinstance(task_metadata.get("num_tasks"), bool)
                or task_metadata.get("num_tasks") != self.spec.num_tasks
                or task_metadata.get("main_metric") != self.spec.main_metric
            ):
                raise MoleculeNetRegistryError(
                    "MoleculeNet build manifest task contract does not match registry"
                )
            source_metadata = self.build_manifest.get("source")
            raw_label_columns = (
                source_metadata.get("label_columns")
                if isinstance(source_metadata, Mapping)
                else None
            )
            if (
                isinstance(raw_label_columns, (str, bytes))
                or not isinstance(raw_label_columns, Sequence)
                or len(raw_label_columns) != self.spec.num_tasks
                or any(
                    not isinstance(column, str) or not column
                    for column in raw_label_columns
                )
                or len(set(raw_label_columns)) != len(raw_label_columns)
                or (
                    not self.spec.dynamic_label_columns
                    and tuple(raw_label_columns) != self.spec.label_columns
                )
            ):
                raise MoleculeNetRegistryError(
                    "MoleculeNet build manifest label column contract does not match registry"
                )
            self.label_columns = tuple(raw_label_columns)
        except BaseException:
            if hasattr(self, "store"):
                self.close()
            raise

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        if "labels" not in sample or "label_mask" not in sample:
            raise MoleculeNetRegistryError(
                f"{self.spec.name} record missing labels/label_mask"
            )
        if int(sample["labels"].numel()) != self.spec.num_tasks:
            raise MoleculeNetRegistryError(
                f"{self.spec.name} record task count {sample['labels'].numel()} "
                f"!= {self.spec.num_tasks}"
            )
        if sample.get("dataset_name") != self.spec.name:
            raise MoleculeNetRegistryError("record dataset_name does not match registry")
        if sample.get("task_type") != self.spec.task_type:
            raise MoleculeNetRegistryError("record task_type does not match registry")
        if tuple(sample.get("label_columns", ())) != self.label_columns:
            raise MoleculeNetRegistryError("record label_columns does not match build contract")
        if self.spec.task_type == "classification":
            finite = sample["labels"][sample["label_mask"]]
            if finite.numel() and not bool(
                torch.all((finite == 0.0) | (finite == 1.0))
            ):
                raise MoleculeNetRegistryError("classification labels must be 0/1/NaN")
        sample["dataset_name"] = self.spec.name
        sample["task_type"] = self.spec.task_type
        sample["label_columns"] = self.label_columns
        return sample
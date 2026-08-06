"""Offline construction of one aligned SemMol multimodal record.

This module is used only by preprocessing programs.  Training datasets read
the resulting safe records and never invoke RDKit, conformer generation, or
density construction in DataLoader workers.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
from src.molecular.electron_density import build_promolecular_density
from src.molecular.espf_tokenizer import ESPFTokenizer
from src.molecular.geometry import GeometryRecord, generate_conformers
from src.molecular.graph import mol_to_pyg_graph
from src.molecular.rdkit_utils import canonicalize_smiles, smiles_to_mol


PathLike = Union[str, os.PathLike[str]]


class FeatureBuildError(RuntimeError):
    """A deterministic, attributable failure while constructing one record."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        source_index: Optional[int] = None,
    ) -> None:
        if not stage:
            raise ValueError("stage must be non-empty")
        self.stage = str(stage)
        self.source_index = None if source_index is None else int(source_index)
        self.detail = str(message)
        prefix = self.stage
        if self.source_index is not None:
            prefix += f"[source_index={self.source_index}]"
        super().__init__(f"{prefix}: {self.detail}")

    def to_dict(self, smiles: str) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "smiles": str(smiles),
            "stage": self.stage,
            "error_type": type(self).__name__,
            "message": self.detail,
        }


def _strict_positive_integer(name: str, value: Any) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _strict_finite_float(
    name: str,
    value: Any,
    *,
    allow_zero: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < 0 or (normalized == 0 and not allow_zero):
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")
    return normalized


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Validated preprocessing parameters shared by PCQM and MoleculeNet."""

    max_smiles_length: int = 256
    generated_conformers: int = 3
    prune_rms_threshold: float = 0.5
    geometry_seed: int = 42
    optimize_geometry: bool = True
    grid_size: int = 32
    grid_spacing: float = 0.75
    grid_padding: float = 4.0
    atomic_sigma: Optional[float] = None
    density_conformer_index: Optional[int] = 0
    strict_density_bounds: bool = True
    discrete_density_normalization: bool = True
    density_storage_dtype: str = "float16"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_smiles_length",
            _strict_positive_integer(
                "max_smiles_length",
                self.max_smiles_length,
            ),
        )
        if self.max_smiles_length < 3:
            raise ValueError("max_smiles_length must be at least 3")
        object.__setattr__(
            self,
            "generated_conformers",
            _strict_positive_integer(
                "generated_conformers",
                self.generated_conformers,
            ),
        )
        object.__setattr__(
            self,
            "grid_size",
            _strict_positive_integer("grid_size", self.grid_size),
        )
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        object.__setattr__(
            self,
            "prune_rms_threshold",
            _strict_finite_float(
                "prune_rms_threshold",
                self.prune_rms_threshold,
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "grid_spacing",
            _strict_finite_float(
                "grid_spacing",
                self.grid_spacing,
                allow_zero=False,
            ),
        )
        object.__setattr__(
            self,
            "grid_padding",
            _strict_finite_float(
                "grid_padding",
                self.grid_padding,
                allow_zero=True,
            ),
        )
        if not isinstance(self.geometry_seed, (int, np.integer)) or isinstance(
            self.geometry_seed,
            bool,
        ):
            raise TypeError("geometry_seed must be an integer")
        object.__setattr__(self, "geometry_seed", int(self.geometry_seed))
        if not 0 <= self.geometry_seed <= 0x7FFFFFFF:
            raise ValueError(
                "geometry_seed must be within RDKit's signed 32-bit range"
            )
        if not isinstance(self.optimize_geometry, bool):
            raise TypeError("optimize_geometry must be bool")
        if not isinstance(self.strict_density_bounds, bool):
            raise TypeError("strict_density_bounds must be bool")
        if not isinstance(self.discrete_density_normalization, bool):
            raise TypeError("discrete_density_normalization must be bool")
        if self.atomic_sigma is not None:
            object.__setattr__(
                self,
                "atomic_sigma",
                _strict_finite_float(
                    "atomic_sigma",
                    self.atomic_sigma,
                    allow_zero=False,
                ),
            )
        if self.density_conformer_index is not None:
            if not isinstance(
                self.density_conformer_index,
                (int, np.integer),
            ) or isinstance(self.density_conformer_index, bool):
                raise TypeError(
                    "density_conformer_index must be a non-negative integer or None"
                )
            if int(self.density_conformer_index) < 0:
                raise ValueError(
                    "density_conformer_index must be non-negative or None"
                )
            object.__setattr__(
                self,
                "density_conformer_index",
                int(self.density_conformer_index),
            )
            if self.density_conformer_index >= self.generated_conformers:
                raise ValueError(
                    "density_conformer_index must be smaller than "
                    "generated_conformers so fallback geometries remain usable"
                )
        if not isinstance(self.density_storage_dtype, str):
            raise TypeError("density_storage_dtype must be a string")
        if self.density_storage_dtype not in {"float16", "float32"}:
            raise ValueError(
                "density_storage_dtype must be 'float16' or 'float32'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_smiles_length": self.max_smiles_length,
            "generated_conformers": self.generated_conformers,
            "prune_rms_threshold": self.prune_rms_threshold,
            "geometry_seed": self.geometry_seed,
            "optimize_geometry": self.optimize_geometry,
            "grid_size": self.grid_size,
            "grid_spacing": self.grid_spacing,
            "grid_padding": self.grid_padding,
            "atomic_sigma": self.atomic_sigma,
            "density_conformer_index": self.density_conformer_index,
            "strict_density_bounds": self.strict_density_bounds,
            "discrete_density_normalization": (
                self.discrete_density_normalization
            ),
            "density_storage_dtype": self.density_storage_dtype,
        }


def tokenizer_artifact_sha256(tokenizer_dir: PathLike) -> str:
    """Return the validated active tokenizer generation/content hash."""

    directory = Path(tokenizer_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"tokenizer directory does not exist: {directory}")
    tokenizer = ESPFTokenizer.from_pretrained(directory)
    if tokenizer.generation_id is not None:
        return tokenizer.generation_id

    # Explicit legacy fallback: bind only the files that define encoding
    # semantics, never mutable reports or unrelated historical directories.
    excluded_names = {
        "failures.jsonl",
        "statistics.json",
    }
    files = sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.name not in excluded_names
            and ".tmp" not in path.name
        ),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not files:
        raise FileNotFoundError(
            f"tokenizer directory contains no immutable artifacts: {directory}"
        )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def deterministic_molecule_seed(
    canonical_smiles: str,
    source_index: int,
    base_seed: int,
) -> int:
    if not isinstance(canonical_smiles, str) or not canonical_smiles:
        raise ValueError("canonical_smiles must be a non-empty string")
    for name, value in (
        ("source_index", source_index),
        ("base_seed", base_seed),
    ):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if int(source_index) < 0:
        raise ValueError("source_index must be non-negative")
    if not 0 <= int(base_seed) <= 0x7FFFFFFF:
        raise ValueError("base_seed must be within RDKit's signed 32-bit range")
    material = (
        f"{int(base_seed)}\0{int(source_index)}\0{canonical_smiles}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(material, digest_size=8).digest(),
        byteorder="big",
        signed=False,
    ) & 0x7FFFFFFF


class MultimodalFeatureBuilder:
    """Construct aligned 1D, 2D, 3D, and promolecular-grid features."""

    def __init__(
        self,
        tokenizer: ESPFTokenizer,
        config: FeatureBuildConfig,
    ) -> None:
        if not isinstance(tokenizer, ESPFTokenizer):
            raise TypeError("tokenizer must be ESPFTokenizer")
        if not isinstance(config, FeatureBuildConfig):
            raise TypeError("config must be FeatureBuildConfig")
        self.tokenizer = tokenizer
        self.config = config

    @staticmethod
    def graph_atomic_numbers(canonical_smiles: str) -> np.ndarray:
        mol = smiles_to_mol(canonical_smiles, on_invalid="raise")
        if mol is None:
            raise RuntimeError("strict SMILES parsing unexpectedly returned None")
        return np.fromiter(
            (atom.GetAtomicNum() for atom in mol.GetAtoms()),
            dtype=np.int64,
            count=mol.GetNumAtoms(),
        )

    def _canonicalize(self, smiles: str, source_index: int) -> str:
        try:
            canonical = canonicalize_smiles(smiles, on_invalid="raise")
        except (RuntimeError, TypeError, ValueError) as exc:
            raise FeatureBuildError(
                "canonicalization",
                str(exc),
                source_index=source_index,
            ) from exc
        if canonical is None:
            raise FeatureBuildError(
                "canonicalization",
                "strict canonicalization returned no SMILES",
                source_index=source_index,
            )
        return canonical

    def _tokens(
        self,
        canonical_smiles: str,
        source_index: int,
    ) -> tuple[np.ndarray, np.ndarray, bool, int]:
        try:
            encoded = self.tokenizer.encode_plus(
                canonical_smiles,
                max_length=None,
                add_special_tokens=True,
                padding=False,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise FeatureBuildError(
                "tokenization",
                str(exc),
                source_index=source_index,
            ) from exc
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int32)
        token_spans = np.asarray(encoded["token_spans"], dtype=np.int32)
        if input_ids.ndim != 1 or input_ids.size == 0:
            raise FeatureBuildError(
                "tokenization",
                "tokenizer produced an empty or non-vector encoding",
                source_index=source_index,
            )
        if token_spans.shape != (input_ids.size, 2):
            raise FeatureBuildError(
                "tokenization",
                "token spans do not align with input IDs",
                source_index=source_index,
            )
        if input_ids.size > self.config.max_smiles_length:
            raise FeatureBuildError(
                "tokenization",
                (
                    f"untruncated token length {input_ids.size} exceeds "
                    f"max_smiles_length={self.config.max_smiles_length}"
                ),
                source_index=source_index,
            )
        negative = token_spans < 0
        sentinel = np.all(token_spans == -1, axis=1)
        if np.any(np.any(negative, axis=1) & ~sentinel):
            raise FeatureBuildError(
                "tokenization",
                "negative token spans must use the complete (-1, -1) sentinel",
                source_index=source_index,
            )
        content_spans = token_spans[~sentinel]
        if (
            content_spans.size == 0
            or not sentinel[0]
            or not sentinel[-1]
            or int(content_spans[0, 0]) != 0
            or int(content_spans[-1, 1]) != len(canonical_smiles)
            or np.any(content_spans[:, 1] <= content_spans[:, 0])
            or np.any(content_spans[1:, 0] != content_spans[:-1, 1])
        ):
            raise FeatureBuildError(
                "tokenization",
                "token spans do not cover the complete canonical SMILES",
                source_index=source_index,
            )
        unknown_count = int(
            np.count_nonzero(input_ids == self.tokenizer.unk_token_id)
        )
        return input_ids, token_spans, False, unknown_count

    def _graph(
        self,
        canonical_smiles: str,
        source_index: int,
    ) -> tuple[dict[str, Any], np.ndarray]:
        try:
            mol = smiles_to_mol(canonical_smiles, on_invalid="raise")
            if mol is None:
                raise RuntimeError("strict SMILES parsing returned None")
            graph = mol_to_pyg_graph(mol)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise FeatureBuildError(
                "graph",
                str(exc),
                source_index=source_index,
            ) from exc
        atomic_numbers = np.fromiter(
            (atom.GetAtomicNum() for atom in mol.GetAtoms()),
            dtype=np.int64,
            count=mol.GetNumAtoms(),
        )
        return {
            "node_feat": graph.x.detach().cpu().numpy().astype(
                np.int16,
                copy=False,
            ),
            "edge_index": graph.edge_index.detach().cpu().numpy().astype(
                np.int32,
                copy=False,
            ),
            "edge_feat": graph.edge_attr.detach().cpu().numpy().astype(
                np.int16,
                copy=False,
            ),
            "num_nodes": int(graph.num_nodes),
        }, atomic_numbers

    def _geometry(
        self,
        canonical_smiles: str,
        source_index: int,
        geometry: Optional[GeometryRecord],
    ) -> GeometryRecord:
        record = geometry
        if record is not None and not isinstance(record, GeometryRecord):
            raise TypeError("geometry must be a GeometryRecord or None")
        if record is None:
            try:
                record = generate_conformers(
                    canonical_smiles,
                    num_conformers=self.config.generated_conformers,
                    prune_rms_thresh=self.config.prune_rms_threshold,
                    seed=deterministic_molecule_seed(
                        canonical_smiles,
                        source_index,
                        self.config.geometry_seed,
                    ),
                    optimize=self.config.optimize_geometry,
                    on_invalid="raise",
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise FeatureBuildError(
                    "geometry",
                    str(exc),
                    source_index=source_index,
                ) from exc
        if record is None:
            raise FeatureBuildError(
                "geometry",
                "geometry generation produced no record",
                source_index=source_index,
            )
        try:
            geometry_canonical = canonicalize_smiles(
                record.canonical_smiles,
                on_invalid="raise",
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise FeatureBuildError(
                "geometry_alignment",
                f"invalid geometry canonical_smiles: {exc}",
                source_index=source_index,
            ) from exc
        if geometry_canonical != canonical_smiles:
            raise FeatureBuildError(
                "geometry_alignment",
                "geometry belongs to a different canonical molecule",
                source_index=source_index,
            )
        return record

    @staticmethod
    def _validate_alignment(
        graph_atomic_numbers: np.ndarray,
        geometry: GeometryRecord,
        source_index: int,
    ) -> None:
        heavy = np.asarray(geometry.heavy_atom_indices, dtype=np.int64)
        if heavy.size != graph_atomic_numbers.size:
            raise FeatureBuildError(
                "geometry_alignment",
                "2D node count differs from the number of canonical heavy atoms",
                source_index=source_index,
            )
        # Both geometry producers reorder the stored all-atom arrays so that
        # canonical heavy atom i is stored at position i.  Requiring the exact
        # mapping (rather than merely equal element counts) prevents a
        # repeated-element permutation such as C(i)<->C(j) from passing.
        expected_heavy = np.arange(graph_atomic_numbers.size, dtype=np.int64)
        if not np.array_equal(heavy, expected_heavy):
            raise FeatureBuildError(
                "geometry_alignment",
                (
                    "heavy_atom_indices do not map canonical graph atom i to "
                    "stored geometry atom i"
                ),
                source_index=source_index,
            )
        geometry_numbers = np.asarray(geometry.atomic_numbers, dtype=np.int64)
        if not np.array_equal(geometry_numbers[heavy], graph_atomic_numbers):
            raise FeatureBuildError(
                "geometry_alignment",
                "2D and 3D canonical atom orders do not have equal elements",
                source_index=source_index,
            )
        source_mapping = np.asarray(
            geometry.heavy_atom_mapping,
            dtype=np.int64,
        )
        if (
            source_mapping.shape != expected_heavy.shape
            or np.unique(source_mapping).size != source_mapping.size
        ):
            raise FeatureBuildError(
                "geometry_alignment",
                "heavy_atom_mapping is not a one-to-one canonical/source map",
                source_index=source_index,
            )
        sources = tuple(str(item) for item in geometry.conformer_source.tolist())
        if sources and all(item.startswith("etkdg_") for item in sources):
            if not np.array_equal(source_mapping, expected_heavy):
                raise FeatureBuildError(
                    "geometry_alignment",
                    "generated geometry lost its canonical heavy-atom mapping",
                    source_index=source_index,
                )

    @staticmethod
    def _geometry_mapping(record: GeometryRecord) -> dict[str, Any]:
        return {
            "atomic_numbers": record.atomic_numbers.astype(np.int16, copy=False),
            "coords": record.coords.astype(np.float32, copy=False),
            "conformer_mask": record.conformer_mask.astype(np.bool_, copy=False),
            "energies": record.energies.astype(np.float32, copy=False),
            "energy_mask": record.energy_mask.astype(np.bool_, copy=False),
            "heavy_atom_indices": record.heavy_atom_indices.astype(
                np.int32,
                copy=False,
            ),
            "sources": [str(item) for item in record.conformer_source.tolist()],
        }

    def _density(
        self,
        geometry: GeometryRecord,
        source_index: int,
    ) -> dict[str, Any]:
        try:
            result = build_promolecular_density(
                geometry.atomic_numbers,
                geometry.coords,
                grid_size=self.config.grid_size,
                spacing=self.config.grid_spacing,
                box_padding=self.config.grid_padding,
                atomic_sigma=self.config.atomic_sigma,
                conformer_index=self.config.density_conformer_index,
                conformer_mask=geometry.conformer_mask,
                strict=self.config.strict_density_bounds,
                discrete_normalize=self.config.discrete_density_normalization,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise FeatureBuildError(
                "density",
                str(exc),
                source_index=source_index,
            ) from exc
        storage_dtype = np.dtype(self.config.density_storage_dtype)
        stored_grid = result.grid.astype(storage_dtype, copy=False)[None, ...]
        stored_integrated_electrons = float(
            stored_grid.astype(np.float64, copy=False).sum(dtype=np.float64)
            * (float(result.spacing) ** 3)
        )
        return {
            "grid": stored_grid,
            "origin": result.origin.astype(np.float32, copy=False),
            "spacing": float(result.spacing),
            "neutral_atom_electron_count": float(result.electron_count),
            "integrated_electrons": stored_integrated_electrons,
            "prequantization_integrated_electrons": float(
                result.integrated_electrons
            ),
            "overflow": bool(result.overflow),
            "overflow_axes": result.overflow_axes.astype(np.bool_, copy=False),
            "atomic_sigmas": result.atomic_sigmas.astype(
                np.float32,
                copy=False,
            ),
            "method": str(result.method),
            "box_padding": float(result.box_padding),
            "conformers_used": result.conformers_used.astype(
                np.int16,
                copy=False,
            ),
            "conformer_reduction": str(result.conformer_reduction),
            "conformer_alignment": str(result.conformer_alignment),
            "normalization_requested": str(result.normalization_requested),
            "normalization_applied": str(result.normalization_applied),
        }

    def build_record(
        self,
        *,
        smiles: str,
        source_index: int,
        sample_namespace: str,
        geometry: Optional[GeometryRecord] = None,
        labels: Optional[Sequence[float] | np.ndarray] = None,
        label_mask: Optional[Sequence[bool] | np.ndarray] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build one record or raise a stage-specific failure."""

        if not isinstance(source_index, (int, np.integer)) or isinstance(
            source_index,
            bool,
        ):
            raise TypeError("source_index must be an integer")
        normalized_source_index = int(source_index)
        if normalized_source_index < 0:
            raise ValueError("source_index must be non-negative")
        if not isinstance(sample_namespace, str) or not sample_namespace.strip():
            raise ValueError("sample_namespace must be a non-empty string")
        sample_namespace = sample_namespace.strip()
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a Mapping or None")

        canonical = self._canonicalize(smiles, normalized_source_index)
        (
            input_ids,
            token_spans,
            smiles_truncated,
            unknown_token_count,
        ) = self._tokens(
            canonical,
            normalized_source_index,
        )
        graph, graph_atomic_numbers = self._graph(
            canonical,
            normalized_source_index,
        )
        geometry_record = self._geometry(
            canonical,
            normalized_source_index,
            geometry,
        )
        self._validate_alignment(
            graph_atomic_numbers,
            geometry_record,
            normalized_source_index,
        )
        density = self._density(geometry_record, normalized_source_index)

        record: dict[str, Any] = {
            "sample_id": f"{sample_namespace}:{normalized_source_index}",
            "source_index": normalized_source_index,
            "smiles": canonical,
            "input_ids": input_ids,
            "token_spans": token_spans,
            "graph": graph,
            "geometry": self._geometry_mapping(geometry_record),
            "density": density,
            "quality": {
                "geometry_reason": geometry_record.reason,
                "geometry_sources": [
                    str(item)
                    for item in geometry_record.conformer_source.tolist()
                ],
                "density_overflow": bool(density["overflow"]),
                "smiles_truncated": smiles_truncated,
                "unknown_token_count": unknown_token_count,
            },
        }
        if metadata is not None:
            for key, value in metadata.items():
                if key in record:
                    raise ValueError(
                        f"metadata key {key!r} collides with a core record field"
                    )
                if not isinstance(key, str) or not key:
                    raise TypeError("metadata keys must be non-empty strings")
                record[key] = value

        if labels is not None:
            raw_labels = np.asarray(labels)
            if raw_labels.ndim != 1 or raw_labels.dtype.kind not in {"i", "u", "f"}:
                raise ValueError("labels must be a one-dimensional real numeric array")
            label_array = raw_labels.astype(np.float32, copy=False)
            if np.any(np.isinf(label_array)):
                raise ValueError("labels cannot contain positive or negative infinity")
            mask_array = (
                np.isfinite(label_array)
                if label_mask is None
                else np.asarray(label_mask)
            )
            if (
                mask_array.shape != label_array.shape
                or mask_array.dtype.kind != "b"
            ):
                raise ValueError(
                    "label_mask must be a boolean array with the same shape as labels"
                )
            if not np.array_equal(mask_array, np.isfinite(label_array)):
                raise ValueError(
                    "label_mask must exactly match the finite positions in labels"
                )
            record["labels"] = label_array
            record["label_mask"] = mask_array.astype(np.bool_, copy=False)
        elif label_mask is not None:
            raise ValueError("label_mask cannot be supplied without labels")
        return record

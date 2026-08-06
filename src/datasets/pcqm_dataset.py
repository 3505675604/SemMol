"""PCQM4Mv2 Sharded Multimodal Dataset.

Dataset only reads LMDB records that have been built and verified offline, and does not execute RDKit in the training worker.
Analysis, conformation generation or density calculation. This can avoid CPU jitter and random image inconsistency during multi-card training.
and the metadata bottleneck caused by millions of small files.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .storage import ShardedRecordStore, StoreSchemaError
from ..utils.io import sha256_file


PathLike = Union[str, Path]
MODALITY_ORDER = ("1d", "2d", "3d", "qm")
OGB_ATOM_FEATURE_DIMS = tuple(int(value) for value in get_atom_feature_dims())
OGB_BOND_FEATURE_DIMS = tuple(int(value) for value in get_bond_feature_dims())
MANIFEST_VALIDATION_CHUNK_SIZE = 262_144
DENSITY_REQUIRED_FIELDS = {
    "grid",
    "origin",
    "spacing",
    "neutral_atom_electron_count",
    "integrated_electrons",
    "prequantization_integrated_electrons",
    "overflow",
    "overflow_axes",
    "atomic_sigmas",
    "method",
    "box_padding",
    "conformers_used",
    "conformer_reduction",
    "conformer_alignment",
    "normalization_requested",
    "normalization_applied",
}
SUPPORTED_BUILD_MANIFEST_SCHEMAS = {
    "semmol.pcqm_store_build.v1",
    "semmol.moleculenet_store_build.v1",
}


class DatasetRecordError(RuntimeError):
    """The training record field, shape, or source index does not satisfy the Dataset contract."""


class MissingModalityError(DatasetRecordError):
    """strict Dataset requested a modal that does not exist in the record."""


class ManifestError(DatasetRecordError):
    """The subset manifest is missing or the field is invalid."""


@dataclass(frozen=True)
class ManifestView:
    """Minimal manifest memory view required for Dataset loading."""

    record_indices: np.ndarray
    source_indices: np.ndarray

    def validate(self, store_length: int) -> None:
        if self.record_indices.ndim != 1 or self.source_indices.ndim != 1:
            raise ManifestError("record_index/source_index must be a one-dimensional array")
        if len(self.record_indices) != len(self.source_indices):
            raise ManifestError("The length of record_index and source_index are inconsistent")
        if self.record_indices.dtype.kind not in {"i", "u"}:
            raise ManifestError("record_index must be an integer dtype")
        if self.source_indices.dtype.kind not in {"i", "u"}:
            raise ManifestError("source_index must be an integer dtype")
        if np.any(self.source_indices < 0):
            raise ManifestError("source_index cannot be negative")
        if np.any(self.record_indices < 0) or np.any(self.record_indices >= store_length):
            raise ManifestError(
                f"manifest record_index is outside [0, {store_length})"
            )
        if _contains_duplicate_indices(
            self.record_indices,
            bitmap_size=store_length,
        ):
            raise ManifestError("subset manifest contains duplicate record_index")
        source_bitmap_size = (
            int(self.source_indices.max()) + 1
            if len(self.source_indices)
            else 0
        )
        if _contains_duplicate_indices(
            self.source_indices,
            bitmap_size=source_bitmap_size,
        ):
            raise ManifestError("subset manifest contains duplicate source_index")


def _contains_duplicate_indices(
    values: np.ndarray,
    *,
    bitmap_size: int,
) -> bool:
    """Check uniqueness with a memory-controlled chunked bitmap.

    Conventional PCQM source/record index is close to dense, and bitmap only needs a few MiB; for abnormal sparse
    The huge source index, return to sorting once, avoid allocating memory according to the malicious maximum value.
    """

    count = len(values)
    if count < 2:
        return False
    dense_limit = max(1_000_000, count * 8)
    if 0 <= bitmap_size <= dense_limit:
        seen = np.zeros(bitmap_size, dtype=np.bool_)
        for start in range(0, count, MANIFEST_VALIDATION_CHUNK_SIZE):
            chunk = values[start:start + MANIFEST_VALIDATION_CHUNK_SIZE]
            unique = np.unique(chunk)
            if len(unique) != len(chunk) or np.any(seen[unique]):
                return True
            seen[unique] = True
        return False
    ordered = np.sort(values, kind="quicksort")
    return bool(np.any(ordered[1:] == ordered[:-1]))


def _load_npz_manifest(path: Path) -> ManifestView:
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            required = {"record_index", "source_index"}
            missing = required - names
            if missing:
                raise ManifestError(f"NPZ manifest is missing fields: {sorted(missing)}")
            raw_record_indices = np.asarray(archive["record_index"])
            raw_source_indices = np.asarray(archive["source_index"])
            for name, values in (
                ("record_index", raw_record_indices),
                ("source_index", raw_source_indices),
            ):
                if values.ndim != 1 or values.dtype.kind not in {"i", "u"}:
                    raise ManifestError(
                        f"NPZ {name} must be a one-dimensional integer array; implicit numeric conversion is not allowed"
                    )
                if values.dtype.kind == "u" and values.size and int(
                    values.max()
                ) > np.iinfo(np.int64).max:
                    raise ManifestError(f"NPZ {name} is outside the int64 range")
            record_indices = raw_record_indices.astype(np.int64, copy=False)
            source_indices = raw_source_indices.astype(np.int64, copy=False)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"Unable to read NPZ manifest: {path}") from exc
    return ManifestView(record_indices, source_indices)


def _load_parquet_manifest(path: Path) -> ManifestView:
    try:
        import pyarrow as arrow
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ManifestError(
            "Reading Parquet manifest requires pyarrow, please install according to the server environment file"
        ) from exc
    try:
        parquet_file = parquet.ParquetFile(path, memory_map=True)
        arrow_schema = parquet_file.schema_arrow
        for name in ("record_index", "source_index"):
            field_index = arrow_schema.get_field_index(name)
            if field_index < 0:
                raise ManifestError(
                    "Parquet manifest is missing record_index/source_index"
                )
            if not arrow.types.is_int64(arrow_schema.field(field_index).type):
                raise ManifestError(
                    f"Parquet {name} must be int64; implicit numeric conversion is not allowed"
                )
        row_count = int(parquet_file.metadata.num_rows)
        record_indices = np.empty(row_count, dtype=np.int64)
        source_indices = np.empty(row_count, dtype=np.int64)
        offset = 0
        for batch in parquet_file.iter_batches(
            batch_size=MANIFEST_VALIDATION_CHUNK_SIZE,
            columns=["record_index", "source_index"],
        ):
            batch_size = batch.num_rows
            record_column = batch.schema.get_field_index("record_index")
            source_column = batch.schema.get_field_index("source_index")
            if record_column < 0 or source_column < 0:
                raise ManifestError(
                    "Parquet manifest is missing record_index/source_index"
                )
            if (
                batch.column(record_column).null_count
                or batch.column(source_column).null_count
            ):
                raise ManifestError(
                    "Parquet record_index/source_index does not allow null"
                )
            record_indices[offset:offset + batch_size] = np.asarray(
                batch.column(record_column).to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            source_indices[offset:offset + batch_size] = np.asarray(
                batch.column(source_column).to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            offset += batch_size
        if offset != row_count:
            raise ManifestError(
                f"Parquet manifest row count {offset} != metadata {row_count}"
            )
    except Exception as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError(f"Unable to read Parquet manifest: {path}") from exc
    return ManifestView(record_indices, source_indices)


def load_manifest_view(path: PathLike, store_length: int) -> ManifestView:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"subset manifest does not exist: {manifest_path}")
    suffix = manifest_path.suffix.lower()
    if suffix == ".npz":
        view = _load_npz_manifest(manifest_path)
    elif suffix in {".parquet", ".pq"}:
        view = _load_parquet_manifest(manifest_path)
    else:
        raise ManifestError(
            f"unsupported manifest format {suffix!r}; only .npz/.parquet are supported"
        )
    view.validate(store_length)
    return view


def _validate_published_store_manifest(
    store: ShardedRecordStore,
    requested_manifest_path: Path,
) -> tuple[dict[str, Any], Optional[Mapping[str, Any]]]:
    build_manifest_path = store.store_dir / "build-manifest.json"
    if not build_manifest_path.is_file():
        raise StoreSchemaError(
            f"published store is missing build-manifest.json: {build_manifest_path}"
        )
    try:
        payload = json.loads(
            build_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreSchemaError(
            f"Unable to parse build-manifest.json: {build_manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise StoreSchemaError("build-manifest.json root node must be an object")
    schema = payload.get("schema")
    if (
        not isinstance(schema, str)
        or schema not in SUPPORTED_BUILD_MANIFEST_SCHEMAS
    ):
        raise StoreSchemaError(
            f"unsupported build manifest schema: {schema!r}"
        )
    if payload.get("status") != "complete":
        raise StoreSchemaError("store build manifest not yet complete")
    record_count = payload.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count != len(store)
    ):
        raise StoreSchemaError(
            "build manifest record_count is inconsistent with store.json"
        )
    tokenizer = payload.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or not isinstance(tokenizer.get("artifact_sha256"), str)
        or not isinstance(tokenizer.get("vocab_size"), int)
        or isinstance(tokenizer.get("vocab_size"), bool)
        or tokenizer.get("artifact_sha256")
        != store.metadata.tokenizer_sha256
        or tokenizer.get("vocab_size")
        != store.metadata.tokenizer_vocab_size
    ):
        raise StoreSchemaError(
            "build manifest tokenizer contract inconsistent with store.json"
        )
    views = payload.get("views")
    if not isinstance(views, Mapping):
        raise StoreSchemaError("build manifest views must be objects")

    requested = requested_manifest_path.resolve()
    registered: Optional[Mapping[str, Any]] = None
    registered_paths: set[Path] = set()
    hexadecimal = set("0123456789abcdef")
    for view_name, descriptor in views.items():
        if (
            not isinstance(view_name, str)
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "sha256", "record_count"}
        ):
            raise StoreSchemaError(
                f"build manifest view {view_name!r} has an invalid descriptor"
            )
        relative = Path(str(descriptor.get("path", "")))
        view_path = (store.store_dir / relative).resolve()
        expected_count = descriptor.get("record_count")
        expected_sha256 = str(descriptor.get("sha256", ""))
        if (
            relative.is_absolute()
            or (
                view_path != store.store_dir
                and store.store_dir not in view_path.parents
            )
            or not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count < 0
            or expected_count > len(store)
            or len(expected_sha256) != 64
            or set(expected_sha256) - hexadecimal
            or view_path in registered_paths
        ):
            raise StoreSchemaError(
                f"build manifest view {view_name!r} has an invalid contract"
            )
        registered_paths.add(view_path)
        if view_path == requested:
            registered = descriptor
    return payload, registered


def _numeric_array(
    value: Any,
    *,
    name: str,
    ndim: Optional[int] = None,
    kinds: str = "biuf",
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in set(kinds):
        raise DatasetRecordError(f"{name} dtype={array.dtype} is not an allowed numeric type")
    if ndim is not None and array.ndim != ndim:
        raise DatasetRecordError(f"{name} ndim={array.ndim}, expected {ndim}")
    return array


def _boolean_array(
    value: Any,
    *,
    name: str,
    ndim: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise DatasetRecordError(f"{name} dtype={array.dtype} must be bool")
    if ndim is not None and array.ndim != ndim:
        raise DatasetRecordError(f"{name} ndim={array.ndim}, expected {ndim}")
    return array


def _finite_real_scalar(
    value: Any,
    *,
    name: str,
    minimum: Optional[float] = None,
    exclusive_minimum: bool = False,
) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"i", "u", "f"}:
        raise DatasetRecordError(f"{name} must be a real scalar")
    normalized = float(array.item())
    if not math.isfinite(normalized):
        raise DatasetRecordError(f"{name} must be a finite real number")
    if minimum is not None:
        invalid = (
            normalized <= minimum
            if exclusive_minimum
            else normalized < minimum
        )
        if invalid:
            relation = "greater than" if exclusive_minimum else "no less than"
            raise DatasetRecordError(f"{name} must be {relation} {minimum}")
    return normalized


def _build_graph(graph: Mapping[str, Any]) -> Data:
    required = {"node_feat", "edge_index", "edge_feat", "num_nodes"}
    missing = required - set(graph)
    if missing:
        raise DatasetRecordError(f"graph is missing fields: {sorted(missing)}")
    node_feat = _numeric_array(
        graph["node_feat"],
        name="graph.node_feat",
        ndim=2,
        kinds="iu",
    )
    edge_index = _numeric_array(
        graph["edge_index"],
        name="graph.edge_index",
        ndim=2,
        kinds="iu",
    )
    edge_feat = _numeric_array(
        graph["edge_feat"],
        name="graph.edge_feat",
        ndim=2,
        kinds="iu",
    )
    raw_num_nodes = graph["num_nodes"]
    if (
        not isinstance(raw_num_nodes, (int, np.integer))
        or isinstance(raw_num_nodes, bool)
    ):
        raise DatasetRecordError("graph.num_nodes must be an integer")
    num_nodes = int(raw_num_nodes)
    if num_nodes <= 0:
        raise DatasetRecordError("graph.num_nodes must be a positive integer")
    if node_feat.shape[0] != num_nodes:
        raise DatasetRecordError(
            f"graph.node_feat row count {node_feat.shape[0]} != num_nodes {num_nodes}"
        )
    if edge_index.shape[0] != 2:
        raise DatasetRecordError("graph.edge_index shape[0] must be 2")
    if edge_index.shape[1] != edge_feat.shape[0]:
        raise DatasetRecordError("The number of edges in graph.edge_index is inconsistent with the number of rows in edge_feat")
    if node_feat.shape[1] != len(OGB_ATOM_FEATURE_DIMS):
        raise DatasetRecordError(
            f"graph.node_feat must be (N, {len(OGB_ATOM_FEATURE_DIMS)})"
        )
    if edge_feat.shape[1] != len(OGB_BOND_FEATURE_DIMS):
        raise DatasetRecordError(
            f"graph.edge_feat must be (E, {len(OGB_BOND_FEATURE_DIMS)})"
        )
    for column, category_count in enumerate(OGB_ATOM_FEATURE_DIMS):
        values = node_feat[:, column]
        if values.size and (
            int(values.min()) < 0 or int(values.max()) >= category_count
        ):
            raise DatasetRecordError(
                f"graph.node_feat column {column} is outside [0, {category_count})"
            )
    for column, category_count in enumerate(OGB_BOND_FEATURE_DIMS):
        values = edge_feat[:, column]
        if values.size and (
            int(values.min()) < 0 or int(values.max()) >= category_count
        ):
            raise DatasetRecordError(
                f"graph.edge_feat column {column} is outside [0, {category_count})"
            )
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise DatasetRecordError("graph.edge_index contains out-of-bounds nodes")
    return Data(
        x=torch.as_tensor(node_feat, dtype=torch.long),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(edge_feat, dtype=torch.long),
        num_nodes=num_nodes,
    )


def _load_geometry(geometry: Mapping[str, Any]) -> Dict[str, Any]:
    required = {
        "atomic_numbers",
        "coords",
        "conformer_mask",
        "energies",
        "energy_mask",
        "heavy_atom_indices",
    }
    missing = required - set(geometry)
    if missing:
        raise DatasetRecordError(f"geometry is missing fields: {sorted(missing)}")

    atomic_numbers = _numeric_array(
        geometry["atomic_numbers"],
        name="geometry.atomic_numbers",
        ndim=1,
        kinds="iu",
    )
    coords = _numeric_array(
        geometry["coords"],
        name="geometry.coords",
        ndim=3,
        kinds="f",
    )
    conformer_mask = _boolean_array(
        geometry["conformer_mask"],
        name="geometry.conformer_mask",
        ndim=1,
    )
    energies = _numeric_array(
        geometry["energies"],
        name="geometry.energies",
        ndim=1,
        kinds="f",
    )
    energy_mask = _boolean_array(
        geometry["…1838 tokens truncated…plied",
    ):
        if not isinstance(density[key], str) or not density[key]:
            raise DatasetRecordError(
                f"density.{key} must be a non-empty string"
            )
    if density["method"] != "promolecular_gaussian":
        raise DatasetRecordError("density.method is not a supported promolecular_gaussian")
    if density["conformer_reduction"] not in {"single", "mean"}:
        raise DatasetRecordError("density.conformer_reduction is not supported")
    if density["conformer_alignment"] not in {"none", "heavy_atom_kabsch"}:
        raise DatasetRecordError("density.conformer_alignment is not supported")
    normalization_values = {
        "discrete_electron_count",
        "continuous_gaussian",
    }
    requested_normalization = density["normalization_requested"]
    applied_normalization = density["normalization_applied"]
    if requested_normalization not in normalization_values:
        raise DatasetRecordError("density.normalization_requested is not supported")
    if applied_normalization not in normalization_values:
        raise DatasetRecordError("density.normalization_applied is not supported")
    expected_applied = (
        "discrete_electron_count"
        if (
            requested_normalization == "discrete_electron_count"
            and not bool(density["overflow"])
        )
        else "continuous_gaussian"
    )
    if applied_normalization != expected_applied:
        raise DatasetRecordError(
            "Density normalization request, actual policy and overflow are inconsistent"
        )
    if (
        applied_normalization == "discrete_electron_count"
        and not math.isclose(
            numeric_metadata["prequantization_integrated_electrons"],
            numeric_metadata["neutral_atom_electron_count"],
            rel_tol=1e-9,
            abs_tol=1e-7,
        )
    ):
        raise DatasetRecordError(
            "Pre-quantization integral of discrete normalized density is inconsistent with electron number"
        )
    conformers_used = _numeric_array(
        density["conformers_used"],
        name="density.conformers_used",
        ndim=1,
        kinds="iu",
    )
    if (
        conformers_used.size == 0
        or int(conformers_used.min()) < 0
        or np.unique(conformers_used).size != conformers_used.size
    ):
        raise DatasetRecordError(
            "density.conformers_used must be a non-empty array of unique non-negative integers"
        )
    if (
        density["conformer_reduction"] == "single"
        and conformers_used.size != 1
    ):
        raise DatasetRecordError("single density must use exactly one conformation")
    if (
        density["conformer_reduction"] == "mean"
        and conformers_used.size > 1
        and density["conformer_alignment"] != "heavy_atom_kabsch"
    ):
        raise DatasetRecordError("Multi-conformation mean density must be recorded heavy_atom_kabsch")
    metadata = {
        key: value
        for key, value in density.items()
        if key != "grid"
    }
    metadata["origin"] = origin.astype(np.float32, copy=False)
    metadata["spacing"] = spacing
    return {
        "qm_grid": torch.as_tensor(grid, dtype=torch.float32),
        "qm_metadata": metadata,
    }


class PCQMMultimodalDataset(Dataset):
    """Read the PCQM4Mv2 four-modal sample from the secure sharding record library.

    Args:
        store_dir: Directory containing ``store.json`` and LMDB shards.
        manifest_path: NPZ containing only ``record_index``/``source_index``, or with the same name
            Parquet subset manifest of columns.
        modalities: The modalities to be returned, the value is ``1d/2d/3d/qm``.
        strict: If True, a ``MissingModalityError`` will be thrown if any request modality is missing;
            Missing fields are omitted when False, expressed by Collator via ``modality_mask``.
        expected_tokenizer_sha256: Optional tokenizer product hash to prevent model vocabulary from being confused with
            The feature database is inconsistent.
    """

    def __init__(
        self,
        store_dir: PathLike,
        manifest_path: PathLike,
        modalities: Sequence[str] = MODALITY_ORDER,
        strict: bool = True,
        expected_tokenizer_sha256: Optional[str] = None,
        readahead: bool = False,
        max_open_shards: int = 16,
    ) -> None:
        if isinstance(modalities, (str, bytes)) or not isinstance(
            modalities,
            Sequence,
        ):
            raise TypeError("modalities must be a sequence of strings")
        if any(not isinstance(modality, str) for modality in modalities):
            raise TypeError("modalities must be a sequence of strings")
        if not isinstance(strict, bool):
            raise TypeError("strict must be bool")
        if not isinstance(readahead, bool):
            raise TypeError("readahead must be bool")
        if expected_tokenizer_sha256 is not None and (
            not isinstance(expected_tokenizer_sha256, str)
            or len(expected_tokenizer_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_tokenizer_sha256
            )
        ):
            raise ValueError(
                "expected_tokenizer_sha256 must be a 64-bit lowercase hexadecimal string"
            )
        requested = tuple(modality.lower() for modality in modalities)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("modalities must be non-empty and non-repeating")
        unknown = set(requested) - set(MODALITY_ORDER)
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")

        self.store = ShardedRecordStore(
            store_dir,
            readahead=readahead,
            max_open_shards=max_open_shards,
        )
        try:
            resolved_manifest_path = Path(manifest_path).resolve()
            (
                self.build_manifest,
                registered_descriptor,
            ) = _validate_published_store_manifest(
                self.store,
                resolved_manifest_path,
            )
            self.manifest = load_manifest_view(
                resolved_manifest_path,
                len(self.store),
            )
            if registered_descriptor is not None:
                if len(self.manifest.record_indices) != int(
                    registered_descriptor["record_count"]
                ):
                    raise ManifestError(
                        "registered view record_count is inconsistent with manifest content"
                    )
                if sha256_file(resolved_manifest_path) != str(
                    registered_descriptor["sha256"]
                ):
                    raise ManifestError(
                        "registered view SHA-256 is inconsistent with build manifest"
                    )
            self.modalities = requested
            self.strict = strict
            if expected_tokenizer_sha256 is not None:
                actual = self.store.metadata.tokenizer_sha256
                if actual != expected_tokenizer_sha256:
                    raise StoreSchemaError(
                        f"tokenizer hash mismatch: store={actual}, "
                        f"expected={expected_tokenizer_sha256}"
                    )
            unavailable = (
                set(self.modalities)
                - set(self.store.metadata.modalities)
            )
            if unavailable and self.strict:
                raise MissingModalityError(
                    f"store.json does not declare requested modalities: {sorted(unavailable)}"
                )
        except BaseException:
            self.store.close()
            raise

    def __len__(self) -> int:
        return len(self.manifest.record_indices)

    def _missing(self, modality: str, record_index: int) -> None:
        if self.strict:
            raise MissingModalityError(
                f"record_index={record_index} is missing requested modality {modality}"
            )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise TypeError("Dataset index must be an integer")
        dataset_index = int(index)
        if dataset_index < 0:
            dataset_index += len(self)
        if dataset_index < 0 or dataset_index >= len(self):
            raise IndexError(f"Dataset index={index} is out of bounds")

        record_index = int(self.manifest.record_indices[dataset_index])
        expected_source_index = int(self.manifest.source_indices[dataset_index])
        record = self.store[record_index]
        required_metadata = {"sample_id", "source_index", "smiles"}
        missing_metadata = required_metadata - set(record)
        if missing_metadata:
            raise DatasetRecordError(
                f"record_index={record_index} is missing metadata {sorted(missing_metadata)}"
            )
        raw_source_index = record["source_index"]
        if not isinstance(raw_source_index, (int, np.integer)) or isinstance(
            raw_source_index,
            bool,
        ):
            raise DatasetRecordError("record.source_index must be an integer")
        source_index = int(raw_source_index)
        if source_index != expected_source_index:
            raise DatasetRecordError(
                f"manifest source_index={expected_source_index} and record "
                f"source_index={source_index} are inconsistent"
            )

        raw_sample_id = record["sample_id"]
        raw_smiles = record["smiles"]
        if not isinstance(raw_sample_id, str) or not raw_sample_id:
            raise DatasetRecordError("record.sample_id must be a non-empty string")
        if not isinstance(raw_smiles, str) or not raw_smiles:
            raise DatasetRecordError("record.smiles must be a non-empty string")
        sample: Dict[str, Any] = {
            "record_index": record_index,
            "sample_id": raw_sample_id,
            "source_index": source_index,
            "smiles": raw_smiles,
        }
        if "gap" in record:
            gap = _finite_real_scalar(record["gap"], name="gap")
            sample["gap"] = gap
        if "quality" in record:
            sample["quality"] = record["quality"]
        for metadata_key in ("dataset_name", "task_type"):
            if metadata_key in record:
                metadata_value = record[metadata_key]
                if not isinstance(metadata_value, str) or not metadata_value:
                    raise DatasetRecordError(
                        f"record.{metadata_key} must be a non-empty string"
                    )
                sample[metadata_key] = metadata_value
        if "label_columns" in record:
            raw_label_columns = record["label_columns"]
            if (
                isinstance(raw_label_columns, (str, bytes))
                or not isinstance(raw_label_columns, (list, tuple))
                or any(
                    not isinstance(column, str) or not column
                    for column in raw_label_columns
                )
                or len(set(raw_label_columns)) != len(raw_label_columns)
            ):
                raise DatasetRecordError(
                    "record.label_columns must be a unique non-empty string array"
                )
            sample["label_columns"] = tuple(raw_label_columns)

        if "1d" in self.modalities:
            if "input_ids" not in record:
                self._missing("1d", record_index)
            else:
                input_ids = _numeric_array(
                    record["input_ids"],
                    name="input_ids",
                    ndim=1,
                    kinds="iu",
                )
                if input_ids.size == 0:
                    raise DatasetRecordError("input_ids cannot be empty")
                if int(input_ids.min()) < 0:
                    raise DatasetRecordError("input_ids cannot contain negative numbers")
                vocab_size = self.store.metadata.tokenizer_vocab_size
                if int(input_ids.max()) >= vocab_size:
                    raise DatasetRecordError(
                        f"input_ids are outside tokenizer_vocab_size={vocab_size}"
                    )
                sample["input_ids"] = torch.as_tensor(input_ids, dtype=torch.long)
                spans = record.get("token_spans")
                if spans is not None:
                    token_spans = _numeric_array(
                        spans,
                        name="token_spans",
                        ndim=2,
                        kinds="iu",
                    )
                    if token_spans.shape != (len(input_ids), 2):
                        raise DatasetRecordError(
                            "token_spans must be (token_count, 2)"
                        )
                    negative = token_spans < 0
                    sentinel = np.all(token_spans == -1, axis=1)
                    if np.any(np.any(negative, axis=1) & ~sentinel):
                        raise DatasetRecordError(
                            "Negative values of token_spans can only use the complete (-1, -1) sentinel"
                        )
                    content_spans = token_spans[~sentinel]
                    if (
                        content_spans.size == 0
                        or not sentinel[0]
                        or not sentinel[-1]
                        or int(content_spans[0, 0]) != 0
                        or int(content_spans[-1, 1]) != len(raw_smiles)
                        or np.any(content_spans[:, 1] <= content_spans[:, 0])
                        or np.any(
                            content_spans[1:, 0] != content_spans[:-1, 1]
                        )
                    ):
                        raise DatasetRecordError(
                            "token_spans must cover the entire canonical SMILES continuously"
                        )
                    sample["token_spans"] = torch.as_tensor(
                        token_spans,
                        dtype=torch.long,
                    )

        if "2d" in self.modalities:
            graph = record.get("graph")
            if graph is None:
                self._missing("2d", record_index)
            elif not isinstance(graph, Mapping):
                raise DatasetRecordError("graph must be Mapping")
            else:
                sample["graph"] = _build_graph(graph)

        if "3d" in self.modalities:
            geometry = record.get("geometry")
            if geometry is None:
                self._missing("3d", record_index)
            elif not isinstance(geometry, Mapping):
                raise DatasetRecordError("geometry must be Mapping")
            else:
                sample.update(_load_geometry(geometry))

        if "qm" in self.modalities:
            density = record.get("density")
            if density is None:
                self._missing("qm", record_index)
            elif not isinstance(density, Mapping):
                raise DatasetRecordError("density must be Mapping")
            else:
                sample.update(_load_density(density))

        if "labels" in record:
            labels = _numeric_array(
                record["labels"],
                name="labels",
                ndim=1,
                kinds="f",
            )
            label_mask = _boolean_array(
                record.get("label_mask", np.isfinite(labels)),
                name="label_mask",
                ndim=1,
            )
            if label_mask.shape != labels.shape:
                raise DatasetRecordError("label_mask is inconsistent with labels shape")
            if np.any(np.isinf(labels)):
                raise DatasetRecordError("labels cannot contain plus or minus infinity")
            if not np.array_equal(label_mask, np.isfinite(labels)):
                raise DatasetRecordError(
                    "label_mask must exactly match the finite value position of labels"
                )
            label_columns = sample.get("label_columns")
            if label_columns is not None and len(label_columns) != len(labels):
                raise DatasetRecordError(
                    "The number of record.label_columns is inconsistent with labels"
                )
            sample["labels"] = torch.as_tensor(labels, dtype=torch.float32)
            sample["label_mask"] = torch.as_tensor(label_mask, dtype=torch.bool)

        if "graph" in sample and "atomic_numbers" in sample:
            heavy = sample["heavy_atom_indices"]
            graph = sample["graph"]
            if graph.num_nodes != int(heavy.numel()):
                raise DatasetRecordError("2D node count is inconsistent with 3D canonical heavy atoms")
            geometry_atomic_numbers = sample["atomic_numbers"][heavy]
            graph_atomic_numbers = graph.x[:, 0] + 1
            if not torch.equal(geometry_atomic_numbers, graph_atomic_numbers):
                raise DatasetRecordError("2D/3D canonical atomic number is inconsistent with the element")

        if "qm_metadata" in sample and "atomic_numbers" in sample:
            density_metadata = sample["qm_metadata"]
            atomic_sigmas = np.asarray(density_metadata["atomic_sigmas"])
            if atomic_sigmas.shape != (int(sample["atomic_numbers"].numel()),):
                raise DatasetRecordError(
                    "density.atomic_sigmas is inconsistent with 3D atomic numbers"
                )
            expected_electrons = float(sample["atomic_numbers"].sum().item())
            if not math.isclose(
                float(density_metadata["neutral_atom_electron_count"]),
                expected_electrons,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise DatasetRecordError(
                    "density electron count is inconsistent with 3D atomic_numbers"
                )
            conformers_used = np.asarray(
                density_metadata["conformers_used"],
                dtype=np.int64,
            )
            conformer_count = int(sample["coords"].shape[0])
            if int(conformers_used.max()) >= conformer_count:
                raise DatasetRecordError("density.conformers_used out of bounds")
            valid_conformers = sample["conformer_mask"].cpu().numpy()
            if not np.all(valid_conformers[conformers_used]):
                raise DatasetRecordError("density uses an invalid conformation")
            if density_metadata["conformer_reduction"] == "mean":
                expected_used = np.flatnonzero(valid_conformers)
                if not np.array_equal(conformers_used, expected_used):
                    raise DatasetRecordError(
                        "mean density must use all valid conformations"
                    )
        return sample

    def close(self) -> None:
        self.store.close()

    def __getstate__(self) -> Dict[str, Any]:
        self.close()
        return dict(self.__dict__)

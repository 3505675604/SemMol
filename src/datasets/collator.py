"""SemMol multimodal batch processing and dynamic pretraining perturbations.

The base Collator only handles lossless padding/assembly; the pretraining Collator
dynamically generates MLM, node/edge masks, and 3D coordinate noise at each batch
fetch. Random seeds are derived from sample_id and epoch, so they do not depend on
DataLoader worker scheduling order.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch_geometric.data import Batch, Data

from src.molecular.graph import (
    OGB_ATOM_FEATURE_CARDINALITIES,
    OGB_BOND_FEATURE_CARDINALITIES,
)


MODALITY_ORDER = ("1d", "2d", "3d", "qm")


class CollationError(RuntimeError):
    """Sample fields or shapes within a batch cannot be safely assembled."""


def _validate_ratio(name: str, value: float) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {ratio}")
    return ratio


def _present_modalities(sample: Mapping[str, Any]) -> Tuple[bool, bool, bool, bool]:
    return (
        "input_ids" in sample,
        "graph" in sample,
        "atomic_numbers" in sample and "coords" in sample,
        "qm_grid" in sample,
    )


def _require_tensor(sample: Mapping[str, Any], key: str, ndim: int) -> torch.Tensor:
    value = sample.get(key)
    if not isinstance(value, torch.Tensor):
        raise CollationError(f"{key} must be torch.Tensor")
    if value.ndim != ndim:
        raise CollationError(f"{key}.ndim={value.ndim}, expected {ndim}")
    return value


class MultimodalDataCollator:
    """Dynamically pad four-modality input, and explicitly output each validity mask."""

    def __init__(
        self,
        pad_token_id: int = 0,
        allow_partial_modalities: bool = False,
    ) -> None:
        if not isinstance(pad_token_id, Integral) or isinstance(
            pad_token_id,
            bool,
        ):
            raise TypeError("pad_token_id must be an integer")
        if not isinstance(allow_partial_modalities, bool):
            raise TypeError("allow_partial_modalities must be bool")
        if pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")
        self.pad_token_id = int(pad_token_id)
        self.allow_partial_modalities = bool(allow_partial_modalities)

    def _collate_tokens(
        self,
        batch: Sequence[Mapping[str, Any]],
        present: torch.Tensor,
        output: MutableMapping[str, Any],
    ) -> None:
        available = [
            _require_tensor(sample, "input_ids", 1)
            for sample, has_tokens in zip(batch, present.tolist())
            if has_tokens
        ]
        if not available:
            return
        max_length = max(int(tokens.numel()) for tokens in available)
        if max_length <= 0:
            raise CollationError("input_ids cannot be empty")
        batch_size = len(batch)
        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        has_any_spans = any(
            has_tokens and "token_spans" in sample
            for sample, has_tokens in zip(batch, present.tolist())
        )
        if has_any_spans and any(
            has_tokens and "token_spans" not in sample
            for sample, has_tokens in zip(batch, present.tolist())
        ):
            raise CollationError(
                "1D modality in the same batch cannot mix records with and without token_spans"
            )
        token_spans = (
            torch.full((batch_size, max_length, 2), -1, dtype=torch.long)
            if has_any_spans
            else None
        )

        for row, (sample, has_tokens) in enumerate(zip(batch, present.tolist())):
            if not has_tokens:
                continue
            tokens = _require_tensor(sample, "input_ids", 1).to(dtype=torch.long)
            length = int(tokens.numel())
            input_ids[row, :length] = tokens
            attention_mask[row, :length] = True
            spans = sample.get("token_spans")
            if spans is not None:
                if not isinstance(spans, torch.Tensor) or spans.shape != (length, 2):
                    raise CollationError(
                        f"sample {row} token_spans shape must be ({length}, 2)"
                    )
                token_spans[row, :length] = spans.to(dtype=torch.long)

        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        if token_spans is not None:
            output["token_spans"] = token_spans

    def _collate_graphs(
        self,
        batch: Sequence[Mapping[str, Any]],
        present: torch.Tensor,
        output: MutableMapping[str, Any],
    ) -> None:
        graphs: List[Data] = []
        sample_indices: List[int] = []
        for sample_index, (sample, has_graph) in enumerate(
            zip(batch, present.tolist())
        ):
            if not has_graph:
                continue
            graph = sample.get("graph")
            if not isinstance(graph, Data):
                raise CollationError(f"sample {sample_index} graph must be PyG Data")
            if graph.x is None or graph.x.ndim != 2:
                raise CollationError(f"sample {sample_index} graph.x must be 2D")
            if graph.edge_index is None or graph.edge_index.shape[0] != 2:
                raise CollationError(
                    f"sample {sample_index} graph.edge_index must be (2, E)"
                )
            if graph.edge_attr is None or graph.edge_attr.ndim != 2:
                raise CollationError(
                    f"sample {sample_index} graph.edge_attr must be 2D"
                )
            if graph.edge_index.shape[1] != graph.edge_attr.shape[0]:
                raise CollationError(
                    f"sample {sample_index} graph edge count does not match edge_attr"
                )
            graphs.append(graph)
            sample_indices.append(sample_index)
        if graphs:
            output["graph"] = Batch.from_data_list(graphs)
            output["graph_sample_index"] = torch.tensor(
                sample_indices,
                dtype=torch.long,
            )

    def _collate_geometry(
        self,
        batch: Sequence[Mapping[str, Any]],
        present: torch.Tensor,
        output: MutableMapping[str, Any],
    ) -> None:
        shapes = []
        for row, (sample, has_geometry) in enumerate(zip(batch, present.tolist())):
            if not has_geometry:
                continue
            atomic_numbers = _require_tensor(sample, "atomic_numbers", 1)
            coords = _require_tensor(sample, "coords", 3)
            if coords.shape[1] != atomic_numbers.numel() or coords.shape[2] != 3:
                raise CollationError(
                    f"sample {row} coords must be (C, {atomic_numbers.numel()}, 3)"
                )
            if atomic_numbers.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise CollationError(f"sample {row} atomic_numbers must be integer")
            if (
                atomic_numbers.numel() == 0
                or int(atomic_numbers.min()) < 1
                or int(atomic_numbers.max()) > 118
            ):
                raise CollationError(
                    f"sample {row} atomic_numbers must be in [1, 118]"
                )
            if (
                coords.shape[0] == 0
                or not coords.is_floating_point()
                or not bool(torch.isfinite(coords).all())
            ):
                raise CollationError(
                    f"sample {row} coords must be a non-empty finite float tensor"
                )
            shapes.append((int(coords.shape[0]), int(coords.shape[1])))
        if not shapes:
            return

        batch_size = len(batch)
        max_conformers = max(shape[0] for shape in shapes)
        max_atoms = max(shape[1] for shape in shapes)
        atomic_numbers_batch = torch.zeros((batch_size, max_atoms), dtype=torch.long)
        coords_batch = torch.zeros(
            (batch_size, max_conformers, max_atoms, 3),
            dtype=torch.float32,
        )
        atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
        conformer_mask = torch.zeros(
            (batch_size, max_conformers),
            dtype=torch.bool,
        )
        energies = torch.full(
            (batch_size, max_conformers),
            float("nan"),
            dtype=torch.float32,
        )
        energy_mask = torch.zeros(
            (batch_size, max_conformers),
            dtype=torch.bool,
        )
        heavy_atom_indices = torch.full(
            (batch_size, max_atoms),
            -1,
            dtype=torch.long,
        )
        heavy_atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
        conformer_sources: List[Tuple[str, ...]] = []

        for row, (sample, has_geometry) in enumerate(zip(batch, present.tolist())):
            if not has_geometry:
                conformer_sources.append(())
                continue
            atomic_numbers = _require_tensor(
                sample,
                "atomic_numbers",
                1,
            ).to(dtype=torch.long)
            coords = _require_tensor(sample, "coords", 3).to(dtype=torch.float32)
            conformer_count, atom_count = int(coords.shape[0]), int(coords.shape[1])
            sample_conformer_mask = sample.get(
                "conformer_mask",
                torch.ones(conformer_count, dtype=torch.bool),
            )
            if (
                not isinstance(sample_conformer_mask, torch.Tensor)
                or sample_conformer_mask.shape != (conformer_count,)
                or sample_conformer_mask.dtype != torch.bool
                or not bool(sample_conformer_mask.any())
            ):
                raise CollationError(
                    f"sample {row} conformer_mask shape must be ({conformer_count},)"
                )
            sample_energies = sample.get(
                "energies",
                torch.full((conformer_count,), float("nan")),
            )
            sample_energy_mask = sample.get(
                "energy_mask",
                torch.isfinite(sample_energies)
                if isinstance(sample_energies, torch.Tensor)
                else None,
            )
            if (
                not isinstance(sample_energies, torch.Tensor)
                or sample_energies.shape != (conformer_count,)
                or not sample_energies.is_floating_point()
                or not isinstance(sample_energy_mask, torch.Tensor)
                or sample_energy_mask.shape != (conformer_count,)
                or sample_energy_mask.dtype != torch.bool
            ):
                raise CollationError(
                    f"sample {row} energies/energy_mask does not match conformer count"
                )
            if (
                bool(torch.isinf(sample_energies).any())
                or not torch.equal(
                    sample_energy_mask,
                    torch.isfinite(sample_energies),
                )
                or bool((sample_energy_mask & ~sample_conformer_mask).any())
            ):
                raise CollationError(
                    f"sample {row} energy_mask does not match valid finite energies"
                )
            heavy = sample.get(
                "heavy_atom_indices",
                torch.arange(atom_count, dtype=torch.long),
            )
            if (
                not isinstance(heavy, torch.Tensor)
                or heavy.ndim != 1
                or heavy.dtype not in {
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                }
            ):
                raise CollationError(f"sample {row} heavy_atom_indices must be 1D")
            if heavy.numel() and (
                int(heavy.min()) < 0 or int(heavy.max()) >= atom_count
            ):
                raise CollationError(f"sample {row} heavy_atom_indices out of bounds")
            expected_heavy = torch.nonzero(
                atomic_numbers != 1,
                as_tuple=False,
            ).flatten()
            if not torch.equal(heavy.to(dtype=torch.long), expected_heavy):
                raise CollationError(
                    f"sample {row} heavy_atom_indices must cover all non-hydrogen atoms"
                )
            raw_sources = sample.get("conformer_sources", ())
            if (
                isinstance(raw_sources, (str, bytes))
                or not isinstance(raw_sources, (list, tuple))
                or len(raw_sources) != conformer_count
                or any(
                    not isinstance(source, str) or not source
                    for source in raw_sources
                )
            ):
                raise CollationError(
                    f"sample {row} conformer_sources does not match conformer count"
                )

            atomic_numbers_batch[row, :atom_count] = atomic_numbers
            coords_batch[row, :conformer_count, :atom_count] = coords
            atom_mask[row, :atom_count] = True
            conformer_mask[row, :conformer_count] = sample_conformer_mask.to(
                dtype=torch.bool
            )
            energies[row, :conformer_count] = sample_energies.to(dtype=torch.float32)
            energy_mask[row, :conformer_count] = sample_energy_mask.to(dtype=torch.bool)
            heavy_count = int(heavy.numel())
            heavy_atom_indices[row, :heavy_count] = heavy.to(dtype=torch.long)
            heavy_atom_mask[row, :heavy_count] = True
            conformer_sources.append(
                tuple(raw_sources)
            )

        output.update(
            {
                "atomic_numbers": atomic_numbers_batch,
                "coords": coords_batch,
                "atom_mask": atom_mask,
                "conformer_mask": conformer_mask,
                "energies": energies,
                "energy_mask": energy_mask,
                "heavy_atom_indices": heavy_atom_indices,
                "heavy_atom_mask": heavy_atom_mask,
                "conformer_sources": conformer_sources,
            }
        )

    def _collate_density(
        self,
        batch: Sequence[Mapping[str, Any]],
        present: torch.Tensor,
        output: MutableMapping[str, Any],
    ) -> None:
        available = [
            _require_tensor(sample, "qm_grid", 4)
            for sample, has_density in zip(batch, present.tolist())
            if has_density
        ]
        if not available:
            return
        expected_shape = tuple(available[0].shape)
        if expected_shape[0] != 1:
            raise CollationError("qm_grid channel dimension must be 1")
        for grid in available[1:]:
            if tuple(grid.shape) != expected_shape:
                raise CollationError(
                    f"qm_grid shapes within batch are inconsistent: {tuple(grid.shape)} != "
                    f"{expected_shape}"
                )
        qm_grid = torch.zeros(
            (len(batch),) + expected_shape,
            dtype=torch.float32,
        )
        qm_mask = torch.zeros(len(batch), dtype=torch.bool)
        qm_metadata: List[Mapping[str, Any]] = []
        for row, (sample, has_density) in enumerate(zip(batch, present.tolist())):
            if has_density:
                qm_grid[row] = _require_tensor(sample, "qm_grid", 4).to(
                    dtype=torch.float32
                )
                qm_mask[row] = True
                metadata = sample.get("qm_metadata", {})
                if not isinstance(metadata, Mapping):
                    raise CollationError(f"sample {row} qm_metadata must be Mapping")
                qm_metadata.append(metadata)
            else:
                qm_metadata.append({})
        output["qm_grid"] = qm_grid
        output["qm_mask"] = qm_mask
        output["qm_metadata"] = qm_metadata

    def _collate_labels(
        self,
        batch: Sequence[Mapping[str, Any]],
        output: MutableMapping[str, Any],
    ) -> None:
        label_samples = [
            _require_tensor(sample, "labels", 1)
            for sample in batch
            if "labels" in sample
        ]
        if not label_samples:
            return
        task_count = int(label_samples[0].numel())
        if any(int(labels.numel()) != task_count for labels in label_samples):
            raise CollationError("labels task count is inconsistent within batch")
        labels_batch = torch.full(
            (len(batch), task_count),
            float("nan"),
            dtype=torch.float32,
        )
        label_mask = torch.zeros((len(batch), task_count), dtype=torch.bool)
        for row, sample in enumerate(batch):
            if "labels" not in sample:
                continue
            labels = _require_tensor(sample, "labels", 1).to(dtype=torch.float32)
            mask = sample.get("label_mask", torch.isfinite(labels))
            if (
                not isinstance(mask, torch.Tensor)
                or mask.shape != labels.shape
                or mask.dtype != torch.bool
            ):
                raise CollationError(f"sample {row} label_mask does not match labels shape")
            if torch.isinf(labels).any() or not torch.equal(
                mask,
                torch.isfinite(labels),
            ):
                raise CollationError(
                    f"sample {row} label_mask does not match finite positions of labels"
                )
            labels_batch[row] = labels
            label_mask[row] = mask
        output["labels"] = labels_batch
        output["label_mask"] = label_mask

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise CollationError("cannot collate empty batch")
        for row, sample in enumerate(batch):
            if not isinstance(sample, Mapping):
                raise CollationError(f"sample {row} is not a Mapping")
            for required_key in ("sample_id", "source_index", "smiles"):
                if required_key not in sample:
                    raise CollationError(f"sample {row} missing {required_key}")
            if (
                not isinstance(sample["sample_id"], str)
                or not sample["sample_id"]
            ):
                raise CollationError(f"sample {row} sample_id must be a non-empty string")
            if (
                not isinstance(sample["source_index"], Integral)
                or isinstance(sample["source_index"], bool)
                or int(sample["source_index"]) < 0
            ):
                raise CollationError(f"sample {row} source_index must be a non-negative integer")
            if not isinstance(sample["smiles"], str) or not sample["smiles"]:
                raise CollationError(f"sample {row} smiles must be a non-empty string")
            if ("atomic_numbers" in sample) != ("coords" in sample):
                raise CollationError(
                    f"sample {row} 3D modality must contain both atomic_numbers and coords"
                )
        has_record_indices = ["record_index" in sample for sample in batch]
        if any(has_record_indices) and not all(has_record_indices):
            raise CollationError(
                "cannot mix samples with and without record_index in the same batch"
            )
        if all(has_record_indices):
            for row, sample in enumerate(batch):
                if (
                    not isinstance(sample["record_index"], Integral)
                    or isinstance(sample["record_index"], bool)
                    or int(sample["record_index"]) < 0
                ):
                    raise CollationError(
                        f"sample {row} record_index must be a non-negative integer"
                    )

        presence = torch.tensor(
            [_present_modalities(sample) for sample in batch],
            dtype=torch.bool,
        )
        if not self.allow_partial_modalities:
            for column, modality in enumerate(MODALITY_ORDER):
                count = int(presence[:, column].sum())
                if count not in {0, len(batch)}:
                    raise CollationError(
                        f"modality {modality} is present in only {count}/{len(batch)} samples "
                        f"within the batch; if missing modalities are expected, "
                        f"explicitly set allow_partial_modalities=True"
                    )

        output: Dict[str, Any] = {
            "sample_id": [sample["sample_id"] for sample in batch],
            "source_index": torch.tensor(
                [int(sample["source_index"]) for sample in batch],
                dtype=torch.long,
            ),
            "smiles": [sample["smiles"] for sample in batch],
            "modality_mask": presence,
        }
        if all(has_record_indices):
            output["record_index"] = torch.tensor(
                [int(sample["record_index"]) for sample in batch],
                dtype=torch.long,
            )

        self._collate_tokens(batch, presence[:, 0], output)
        self._collate_graphs(batch, presence[:, 1], output)
        self._collate_geometry(batch, presence[:, 2], output)
        self._collate_density(batch, presence[:, 3], output)
        self._collate_labels(batch, output)
        return output


class PretrainingDataCollator(MultimodalDataCollator):
    """Dynamically create multi-modal self-supervised targets on top of the base batch."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        mask_token_id: int,
        vocab_size: int,
        special_token_ids: Sequence[int],
        smiles_mask_ratio: float = 0.15,
        node_mask_ratio: float = 0.15,
        edge_mask_ratio: float = 0.15,
        geo_noise_std: float = 1.0,
        seed: int = 3407,
        allow_partial_modalities: bool = False,
        node_mask_token_ids: Sequence[int] = OGB_ATOM_FEATURE_CARDINALITIES,
        edge_mask_token_ids: Sequence[int] = OGB_BOND_FEATURE_CARDINALITIES,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            allow_partial_modalities=allow_partial_modalities,
        )
        for name, value in (
            ("mask_token_id", mask_token_id),
            ("vocab_size", vocab_size),
            ("seed", seed),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if pad_token_id >= vocab_size:
            raise ValueError("pad_token_id must be within vocabulary range")
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError("mask_token_id must be within vocabulary range")
        if isinstance(special_token_ids, (str, bytes)) or not isinstance(
            special_token_ids,
            Sequence,
        ):
            raise TypeError("special_token_ids must be an integer sequence")
        if any(
            not isinstance(item, Integral) or isinstance(item, bool)
            for item in special_token_ids
        ):
            raise TypeError("special_token_ids must be an integer sequence")
        normalized_special = tuple(
            sorted({int(item) for item in special_token_ids})
        )
        if any(item < 0 or item >= vocab_size for item in normalized_special):
            raise ValueError("special_token_ids must be within vocabulary range")
        if pad_token_id not in normalized_special:
            raise ValueError("special_token_ids must include pad_token_id")
        if mask_token_id not in normalized_special:
            raise ValueError("special_token_ids must include mask_token_id")
        random_token_ids = [
            token_id
            for token_id in range(vocab_size)
            if token_id not in set(normalized_special)
        ]
        if not random_token_ids:
            raise ValueError("vocabulary must contain at least one non-special token")

        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)
        self.special_token_ids = normalized_special
        self._random_token_ids = torch.tensor(random_token_ids, dtype=torch.long)
        self.smiles_mask_ratio = _validate_ratio(
            "smiles_mask_ratio",
            smiles_mask_ratio,
        )
        self.node_mask_ratio = _validate_ratio("node_mask_ratio", node_mask_ratio)
        self.edge_mask_ratio = _validate_ratio("edge_mask_ratio", edge_mask_ratio)
        if not isinstance(geo_noise_std, Real) or isinstance(geo_noise_std, bool):
            raise TypeError("geo_noise_std must be a real number")
        if not math.isfinite(float(geo_noise_std)) or geo_noise_std < 0:
            raise ValueError("geo_noise_std cannot be negative")
        if seed < 0 or seed > 2**63 - 1:
            raise ValueError("seed must be in [0, 2**63 - 1]")
        self.geo_noise_std = float(geo_noise_std)
        self.seed = int(seed)
        self.node_mask_token_ids = self._validate_graph_mask_tokens(
            "node_mask_token_ids",
            node_mask_token_ids,
            expected_values=OGB_ATOM_FEATURE_CARDINALITIES,
        )
        self.edge_mask_token_ids = self._validate_graph_mask_tokens(
            "edge_mask_token_ids",
            edge_mask_token_ids,
            expected_values=OGB_BOND_FEATURE_CARDINALITIES,
        )
        # DataLoader executes collate_fn inside worker processes when
        # num_workers > 0.  A normal integer would be copied into each worker
        # and persistent workers would therefore keep using epoch zero.
        # Shared CPU tensor storage makes set_epoch() visible to every worker
        # without rebuilding the DataLoader at each epoch.
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @staticmethod
    def _validate_graph_mask_tokens(
        name: str,
        values: Sequence[int],
        *,
        expected_values: Sequence[int],
    ) -> torch.Tensor:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{name} must be an integer sequence")
        if any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in values
        ):
            raise TypeError(f"{name} must be an integer sequence")
        normalized = tuple(int(value) for value in values)
        expected = tuple(int(value) for value in expected_values)
        if normalized != expected:
            raise ValueError(
                f"{name} must equal the reserved OGB mask indices {expected}"
            )
        return torch.tensor(normalized, dtype=torch.long)

    @property
    def epoch(self) -> int:
        return int(self._shared_epoch.item())

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, Integral) or isinstance(epoch, bool):
            raise TypeError("epoch must be an integer")
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self._shared_epoch.fill_(int(epoch))

    def _generator(self, sample_id: str, task: str) -> torch.Generator:
        material = f"{self.seed}|{self.epoch}|{task}|{sample_id}".encode("utf-8")
        digest = hashlib.blake2b(material, digest_size=8).digest()
        derived_seed = int.from_bytes(digest, byteorder="big", signed=False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derived_seed)
        return generator

    @staticmethod
    def _choose_positions(
        candidates: torch.Tensor,
        ratio: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if ratio <= 0 or candidates.numel() == 0:
            return candidates.new_empty((0,), dtype=torch.long)
        count = min(
            int(candidates.numel()),
            max(1, int(math.ceil(float(candidates.numel()) * ratio))),
        )
        order = torch.randperm(int(candidates.numel()), generator=generator)
        return candidates[order[:count]]

    def _apply_mlm(self, output: MutableMapping[str, Any]) -> None:
        if "input_ids" not in output:
            return
        input_ids = output["input_ids"].clone()
        attention_mask = output["attention_mask"]
        labels = torch.full_like(input_ids, -100)
        special_ids = torch.tensor(self.special_token_ids, dtype=torch.long)
        for row, sample_id in enumerate(output["sample_id"]):
            eligible = attention_mask[row].clone()
            if special_ids.numel():
                eligible &= ~torch.isin(input_ids[row], special_ids)
            candidates = torch.nonzero(eligible, as_tuple=False).flatten()
            generator = self._generator(sample_id, "mlm")
            selected = self._choose_positions(
                candidates,
                self.smiles_mask_ratio,
                generator,
            )
            if selected.numel() == 0:
                continue
            labels[row, selected] = input_ids[row, selected]
            decisions = torch.rand(selected.numel(), generator=generator)
            mask_positions = selected[decisions < 0.8]
            random_positions = selected[(decisions >= 0.8) & (decisions < 0.9)]
            input_ids[row, mask_positions] = self.mask_token_id
            if random_positions.numel():
                choices = torch.randint(
                    low=0,
                    high=int(self._random_token_ids.numel()),
                    size=(int(random_positions.numel()),),
                    generator=generator,
                )
                input_ids[row, random_positions] = self._random_token_ids[choices]
        output["input_ids"] = input_ids
        output["mlm_labels"] = labels

    def _apply_node_mask(self, output: MutableMapping[str, Any]) -> None:
        graph = output.get("graph")
        if not isinstance(graph, Batch):
            return
        node_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        node_labels = torch.full_like(graph.x, -100)
        graph_sample_index = output["graph_sample_index"]
        for graph_index, sample_index in enumerate(graph_sample_index.tolist()):
            start = int(graph.ptr[graph_index])
            end = int(graph.ptr[graph_index + 1])
            candidates = torch.arange(start, end, dtype=torch.long)
            generator = self._generator(
                output["sample_id"][sample_index],
                "node_mask",
            )
            selected = self._choose_positions(
                candidates,
                self.node_mask_ratio,
                generator,
            )
            if selected.numel():
                node_mask[selected] = True
                node_labels[selected] = graph.x[selected]
        graph.x = graph.x.clone()
        if graph.x.ndim != 2 or graph.x.shape[1] != self.node_mask_token_ids.numel():
            raise CollationError(
                "graph.x fields do not match configured node mask token schema"
            )
        graph.x[node_mask] = self.node_mask_token_ids.to(graph.x.device)
        output["node_mask"] = node_mask
        output["node_labels"] = node_labels

    @staticmethod
    def _undirected_edge_groups(
        edge_index: torch.Tensor,
        edge_positions: torch.Tensor,
    ) -> List[torch.Tensor]:
        groups: Dict[Tuple[int, int], List[int]] = {}
        for position in edge_positions.tolist():
            source = int(edge_index[0, position])
            target = int(edge_index[1, position])
            key = (source, target) if source <= target else (target, source)
            groups.setdefault(key, []).append(position)
        return [
            torch.tensor(positions, dtype=torch.long)
            for _, positions in sorted(groups.items())
        ]

    def _apply_edge_mask(self, output: MutableMapping[str, Any]) -> None:
        graph = output.get("graph")
        if not isinstance(graph, Batch):
            return
        edge_count = int(graph.edge_index.shape[1])
        edge_mask = torch.zeros(edge_count, dtype=torch.bool)
        edge_labels = torch.full_like(graph.edge_attr, -100)
        if edge_count == 0:
            output["edge_mask"] = edge_mask
            output["edge_labels"] = edge_labels
            return

        edge_graph_index = graph.batch[graph.edge_index[0]]
        graph_sample_index = output["graph_sample_index"]
        for graph_index, sample_index in enumerate(graph_sample_index.tolist()):
            positions = torch.nonzero(
                edge_graph_index == graph_index,
                as_tuple=False,
            ).flatten()
            groups = self._undirected_edge_groups(graph.edge_index, positions)
            if not groups or self.edge_mask_ratio <= 0:
                continue
            generator = self._generator(
                output["sample_id"][sample_index],
                "edge_mask",
            )
            group_candidates = torch.arange(len(groups), dtype=torch.long)
            selected_groups = self._choose_positions(
                group_candidates,
                self.edge_mask_ratio,
                generator,
            )
            if selected_groups.numel():
                selected_edges = torch.cat(
                    [groups[int(index)] for index in selected_groups],
                    dim=0,
                )
                edge_mask[selected_edges] = True
                edge_labels[selected_edges] = graph.edge_attr[selected_edges]
        graph.edge_attr = graph.edge_attr.clone()
        if (
            graph.edge_attr.ndim != 2
            or graph.edge_attr.shape[1] != self.edge_mask_token_ids.numel()
        ):
            raise CollationError(
                "graph.edge_attr fields do not match configured edge mask token schema"
            )
        graph.edge_attr[edge_mask] = self.edge_mask_token_ids.to(
            graph.edge_attr.device
        )
        output["edge_mask"] = edge_mask
        output["edge_labels"] = edge_labels

    def _apply_geometry_noise(self, output: MutableMapping[str, Any]) -> None:
        if "coords" not in output:
            return
        clean_coords = output["coords"].clone()
        coord_noise = torch.zeros_like(clean_coords)
        if self.geo_noise_std > 0:
            valid = (
                output["conformer_mask"][:, :, None]
                & output["atom_mask"][:, None, :]
            )
            for row, sample_id in enumerate(output["sample_id"]):
                generator = self._generator(sample_id, "geometry_noise")
                noise = torch.randn(
                    clean_coords[row].shape,
                    generator=generator,
                    dtype=clean_coords.dtype,
                )
                noise *= self.geo_noise_std
                row_valid = valid[row]
                valid_values = row_valid[:, :, None].to(
                    dtype=noise.dtype
                )
                noise *= valid_values
                atom_counts = row_valid.sum(
                    dim=1,
                    keepdim=True,
                ).to(dtype=noise.dtype)
                mean_noise = noise.sum(
                    dim=1,
                    keepdim=True,
                ) / atom_counts.clamp_min(1.0).unsqueeze(-1)
                noise = (noise - mean_noise) * valid_values
                variance_scale = torch.sqrt(
                    atom_counts
                    / (atom_counts - 1.0).clamp_min(1.0)
                )
                variance_scale *= atom_counts > 1.0
                noise *= variance_scale.unsqueeze(-1)
                coord_noise[row] = noise
        output["clean_coords"] = clean_coords
        output["coord_noise"] = coord_noise
        output["coords"] = clean_coords + coord_noise

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        output = super().__call__(batch)
        self._apply_mlm(output)
        self._apply_node_mask(output)
        self._apply_edge_mask(output)
        self._apply_geometry_noise(output)
        return output


class FinetuningDataCollator(MultimodalDataCollator):
    """Semantic alias: downstream tasks only do basic assembly, no self-supervised perturbations."""
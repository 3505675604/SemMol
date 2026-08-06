"""Complete SemMol assembly for pretraining and downstream prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Final

import torch
import torch.distributed as dist
from torch import Tensor, nn

from src.models.alignment.acsm import ACSMOutput, AnchorCentersSoftMatching
from src.models.encoders import (
    EncoderOutput,
    GeoEncoder,
    GraphEncoder,
    GraphEncoderOutput,
    QMEncoder,
    SMILESEncoder,
)
from src.models.heads import (
    GeometryDenoisingHead,
    GeometryDenoisingPrediction,
    GraphReconstructionHead,
    GraphReconstructionPrediction,
    MLMPrediction,
    MaskedLanguageModelingHead,
    PropertyPredictor,
)
from src.models.projection import ProjectionBank, ProjectionOutput
from src.models.semantic.dcl import (
    DCLAssignment,
    DCLUpdate,
    DynamicCentralLibrary,
)


_MODALITIES: Final[tuple[str, ...]] = ("1d", "2d", "3d", "qm")
_ANCHOR_MODALITIES: Final[frozenset[str]] = frozenset({"1d", "2d", "3d"})
_ENCODER_SECTION: Final[dict[str, str]] = {
    "1d": "smiles",
    "2d": "graph",
    "3d": "geo",
    "qm": "qm",
}
_MODALITY_COLUMN: Final[dict[str, int]] = {
    modality: index for index, modality in enumerate(_MODALITIES)
}
_ENCODER_OUTPUT_DIMENSION: Final[dict[str, str]] = {
    "1d": "shared_dim",
    "2d": "hidden_size",
    "3d": "target_dim",
    "qm": "hidden_size",
}
_BATCH_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "1d": ("input_ids", "attention_mask"),
    "2d": ("graph", "graph_sample_index"),
    "3d": ("atomic_numbers", "coords", "atom_mask", "conformer_mask"),
    "qm": ("qm_grid", "qm_mask"),
}


@dataclass(frozen=True)
class ModalityRepresentation:
    """Encoder and shared-space projection outputs for one modality."""

    encoder_output: EncoderOutput
    projection_output: ProjectionOutput


@dataclass(frozen=True)
class SemMolPretrainingOutput:
    """Outputs needed by reconstruction, clustering, and ACSM losses."""

    batch_size: int
    representations: dict[str, ModalityRepresentation]
    anchor_modality: str
    anchor_sample_index: Tensor
    anchor_embedding: Tensor
    dcl_assignments: dict[str, DCLAssignment]
    acsm_output: ACSMOutput | None
    dcl_updates: dict[str, DCLUpdate]
    dcl_initialized: bool
    alignment_ready: bool
    warmup_active: bool
    mlm_prediction: MLMPrediction | None = None
    graph_prediction: GraphReconstructionPrediction | None = None
    geo_prediction: GeometryDenoisingPrediction | None = None

    @property
    def h_anchor(self) -> Tensor:
        return self.anchor_embedding

    @property
    def h_pos_final(self) -> Tensor | None:
        if self.acsm_output is None:
            return None
        return self.acsm_output.positive_embedding

    @property
    def neg_sim_dict(self) -> dict[str, Tensor]:
        if self.acsm_output is None:
            return {}
        return {
            modality: match.negative_similarities
            for modality, match in self.acsm_output.modality_matches.items()
        }

    @property
    def mlm_logits(self) -> Tensor | None:
        if self.mlm_prediction is None:
            return None
        return self.mlm_prediction.logits

    @property
    def node_logits(self) -> tuple[Tensor, ...] | None:
        if self.graph_prediction is None:
            return None
        return self.graph_prediction.node_logits

    @property
    def edge_logits(self) -> tuple[Tensor, ...] | None:
        if self.graph_prediction is None:
            return None
        return self.graph_prediction.edge_logits

    @property
    def pred_noise(self) -> Tensor | None:
        if self.geo_prediction is None:
            return None
        return self.geo_prediction.predicted_noise

    def projected(self, modality: str) -> Tensor | None:
        normalized = _normalize_modality("modality", modality)
        representation = self.representations.get(normalized)
        if representation is None:
            return None
        return representation.projection_output.normalized

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow dictionary for trainer code that prefers mappings."""

        output = {
            "batch_size": self.batch_size,
            "representations": self.representations,
            "anchor_modality": self.anchor_modality,
            "anchor_sample_index": self.anchor_sample_index,
            "anchor_embedding": self.anchor_embedding,
            "dcl_assignments": self.dcl_assignments,
            "acsm_output": self.acsm_output,
            "dcl_updates": self.dcl_updates,
            "dcl_initialized": self.dcl_initialized,
            "alignment_ready": self.alignment_ready,
            "warmup_active": self.warmup_active,
            "mlm_prediction": self.mlm_prediction,
            "graph_prediction": self.graph_prediction,
            "geo_prediction": self.geo_prediction,
            "h_anchor": self.h_anchor,
            "h_pos_final": self.h_pos_final,
            "neg_sim_dict": self.neg_sim_dict,
            "mlm_logits": self.mlm_logits,
            "node_logits": self.node_logits,
            "edge_logits": self.edge_logits,
            "pred_noise": self.pred_noise,
        }
        for modality in ("1d", "2d", "3d", "qm"):
            output[f"h_{modality}"] = self.projected(modality)
        return output

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str):
            raise TypeError("SemMol output keys must be strings")
        output = self.as_dict()
        if key not in output:
            raise KeyError(key)
        return output[key]


@dataclass(frozen=True)
class SemMolFinetuningOutput:
    """Raw property predictions and their fused semantic representation."""

    predictions: Tensor
    fused_features: Tensor
    anchor_sample_index: Tensor
    anchor_representation: ModalityRepresentation
    acsm_output: ACSMOutput


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive, got {normalized}")
    return normalized


def _nonnegative_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} cannot be negative")
    return normalized


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _mapping_copy(name: str, value: object) -> dict[str, Any]:
    return dict(_mapping(name, value))


def _normalize_modality(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in _MODALITIES:
        raise ValueError(
            f"unsupported {name}={value!r}; expected one of {_MODALITIES}"
        )
    return normalized


class SemMol(nn.Module):
    """Assemble encoders, projections, target DCLs, ACSM, and an optional head.

    Configuration paths must be resolved by the entrypoint before construction.
    DCL snapshots are always consumed by ACSM before the current mini-batch EMA
    update, preventing current-batch information from leaking into retrieval.
    Every target DCL is updated in a deterministic modality order, including
    empty local tensors, so distributed collectives remain rank-consistent.
    """

    SUPPORTED_MODALITIES = _MODALITIES
    SUPPORTED_ANCHORS = tuple(
        modality for modality in _MODALITIES if modality in _ANCHOR_MODALITIES
    )

    def __init__(
        self,
        *,
        encoders: Mapping[str, Any],
        projection: Mapping[str, Any],
        dcl: Mapping[str, Any],
        acsm: Mapping[str, Any],
        modalities: Sequence[str] = ("1d", "2d", "3d"),
        anchor_modality: str = "1d",
        task: Mapping[str, Any] | None = None,
        head: Mapping[str, Any] | None = None,
        pretraining_heads: Mapping[str, Any] | None = None,
        freeze_encoders: bool = False,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")
        if not isinstance(freeze_encoders, bool):
            raise TypeError("freeze_encoders must be bool")

        self.validate_values = validate_values
        self.modalities = self._normalize_modalities(modalities)
        if task is None and "qm" in self.modalities:
            raise ValueError(
                "the manuscript defines pretraining objectives and DCL/ACSM "
                "centers for 1d, 2d, and 3d only; QM pretraining requires an "
                "explicit differentiable objective and cannot be enabled"
            )
        self.anchor_modality = _normalize_modality(
            "anchor_modality",
            anchor_modality,
        )
        if self.anchor_modality not in _ANCHOR_MODALITIES:
            raise ValueError(
                "QM is an optional target extension and cannot be the anchor; "
                "choose one of '1d', '2d', or '3d'"
            )
        if self.anchor_modality not in self.modalities:
            raise ValueError(
                f"anchor_modality={self.anchor_modality!r} is not enabled"
            )
        self.target_modalities = tuple(
            modality
            for modality in self.modalities
            if modality != self.anchor_modality
        )
        if not self.target_modalities:
            raise ValueError(
                "SemMol requires at least one enabled target modality"
            )

        encoder_options = _mapping_copy("encoders", encoders)
        allowed_encoder_sections = {
            "shared_dim",
            *tuple(_ENCODER_SECTION.values()),
        }
        unknown_encoder_sections = (
            set(encoder_options) - allowed_encoder_sections
        )
        if unknown_encoder_sections:
            raise ValueError(
                "unknown encoder configuration sections: "
                f"{sorted(unknown_encoder_sections)}"
            )
        if "shared_dim" not in encoder_options:
            raise ValueError("encoders.shared_dim is required")
        self.encoder_dim = _positive_integer(
            "encoders.shared_dim",
            encoder_options["shared_dim"],
        )
        self.encoders = self._build_encoders(encoder_options)

        projection_options = _mapping_copy("projection", projection)
        if "modalities" in projection_options:
            raise ValueError(
                "projection.modalities is owned by SemMol and must be omitted"
            )
        projection_options.setdefault("input_dim", self.encoder_dim)
        projection_options.setdefault("validate_values", validate_values)
        self.projection = ProjectionBank(
            modalities=self.modalities,
            **projection_options,
        )
        if self.projection.input_dim != self.encoder_dim:
            raise ValueError(
                "projection.input_dim must equal encoders.shared_dim: "
                f"{self.projection.input_dim} != {self.encoder_dim}"
            )
        self._validate_encoder_dimensions()
        self.feature_dim = int(self.projection.output_dim)

        dcl_options = _mapping_copy("dcl", dcl)
        self.dcl_warmup_steps = _nonnegative_integer(
            "dcl.warmup_steps",
            dcl_options.pop("warmup_steps", 0),
        )
        dcl_options.setdefault("feature_dim", self.feature_dim)
        dcl_options.setdefault("validate_values", validate_values)
        configured_dcl_dim = _positive_integer(
            "dcl.feature_dim",
            dcl_options["feature_dim"],
        )
        if configured_dcl_dim != self.feature_dim:
            raise ValueError(
                "dcl.feature_dim must equal projection.output_dim: "
                f"{configured_dcl_dim} != {self.feature_dim}"
            )
        self.dcls = nn.ModuleDict(
            {
                modality: DynamicCentralLibrary(**dcl_options)
                for modality in self.target_modalities
            }
        )

        acsm_options = _mapping_copy("acsm", acsm)
        acsm_options.setdefault("feature_dim", self.feature_dim)
        acsm_options.setdefault("validate_values", validate_values)
        configured_acsm_dim = _positive_integer(
            "acsm.feature_dim",
            acsm_options["feature_dim"],
        )
        if configured_acsm_dim != self.feature_dim:
            raise ValueError(
                "acsm.feature_dim must equal projection.output_dim: "
                f"{configured_acsm_dim} != {self.feature_dim}"
            )
        self.acsm = AnchorCentersSoftMatching(**acsm_options)
        if self.acsm.learnable_temperature:
            raise ValueError(
                "SemMol uses a fixed ACSM temperature so data-driven DCL "
                "initialization and warmup remain safe with standard DDP; "
                "set acsm.learnable_temperature=false"
            )

        self.property_head = self._build_property_head(task, head)
        self.pretraining_heads = self._build_pretraining_heads(
            task,
            pretraining_heads,
        )
        self.freeze_encoders = False
        self.set_encoders_frozen(freeze_encoders)

    @staticmethod
    def _normalize_modalities(modalities: Sequence[str]) -> tuple[str, ...]:
        if isinstance(modalities, (str, bytes)) or not isinstance(
            modalities,
            Sequence,
        ):
            raise TypeError("modalities must be a sequence")
        normalized = tuple(
            _normalize_modality(f"modalities[{index}]", modality)
            for index, modality in enumerate(modalities)
        )
        if not normalized:
            raise ValueError("modalities must contain at least one modality")
        if len(set(normalized)) != len(normalized):
            raise ValueError("modalities must be unique")
        selected = set(normalized)
        return tuple(
            modality for modality in _MODALITIES if modality in selected
        )

    def _build_encoders(
        self,
        options: Mapping[str, Any],
    ) -> nn.ModuleDict:
        modules: dict[str, nn.Module] = {}
        constructors: dict[str, type[nn.Module]] = {
            "1d": SMILESEncoder,
            "2d": GraphEncoder,
            "3d": GeoEncoder,
            "qm": QMEncoder,
        }
        dimension_key = {
            "1d": "shared_dim",
            "2d": "hidden_size",
            "3d": "target_dim",
            "qm": "hidden_size",
        }
        for modality in self.modalities:
            section = _ENCODER_SECTION[modality]
            if section not in options:
                raise ValueError(
                    f"encoders.{section} is required for modality {modality!r}"
                )
            modality_options = _mapping_copy(
                f"encoders.{section}",
                options[section],
            )
            modality_options.setdefault(
                dimension_key[modality],
                self.encoder_dim,
            )
            modality_options.setdefault(
                "validate_values",
                self.validate_values,
            )
            modules[modality] = constructors[modality](**modality_options)
        return nn.ModuleDict(modules)

    def _validate_encoder_dimensions(self) -> None:
        for modality in self.modalities:
            encoder = self.encoders[modality]
            attribute = _ENCODER_OUTPUT_DIMENSION[modality]
            actual_dimension = getattr(encoder, attribute, None)
            if (
                not isinstance(actual_dimension, Integral)
                or isinstance(actual_dimension, bool)
            ):
                raise TypeError(
                    f"{modality} encoder must expose integer {attribute}"
                )
            if int(actual_dimension) != self.projection.input_dim:
                raise ValueError(
                    f"{modality} encoder {attribute}={actual_dimension} "
                    "does not match projection.input_dim="
                    f"{self.projection.input_dim}"
                )

    def _build_property_head(
        self,
        task: Mapping[str, Any] | None,
        head: Mapping[str, Any] | None,
    ) -> PropertyPredictor | None:
        if task is None:
            if head is not None:
                raise ValueError(
                    "head options require task configuration"
                )
            return None
        task_options = _mapping("task", task)
        if "type" not in task_options or "num_tasks" not in task_options:
            raise ValueError("task.type and task.num_tasks are required")
        head_options = (
            {} if head is None else _mapping_copy("head", head)
        )
        reserved = {"input_dim", "num_tasks", "task_type"}
        conflicts = reserved.intersection(head_options)
        if conflicts:
            raise ValueError(
                "head cannot override SemMol-owned options: "
                f"{sorted(conflicts)}"
            )
        head_options.setdefault("validate_values", self.validate_values)
        return PropertyPredictor(
            input_dim=2 * self.feature_dim,
            num_tasks=task_options["num_tasks"],
            task_type=task_options["type"],
            **head_options,
        )

    @staticmethod
    def _pretraining_head_section(
        options: Mapping[str, Any],
        name: str,
        *,
        allowed: frozenset[str],
    ) -> tuple[bool, dict[str, Any]]:
        section = (
            {}
            if name not in options
            else _mapping_copy(f"pretraining_heads.{name}", options[name])
        )
        enabled = section.pop("enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError(
                f"pretraining_heads.{name}.enabled must be bool"
            )
        unknown = set(section) - allowed
        if unknown:
            raise ValueError(
                f"unknown pretraining_heads.{name} options: "
                f"{sorted(unknown)}"
            )
        return enabled, section

    def _build_pretraining_heads(
        self,
        task: Mapping[str, Any] | None,
        configuration: Mapping[str, Any] | None,
    ) -> nn.ModuleDict:
        if task is not None:
            if configuration is not None:
                raise ValueError(
                    "pretraining_heads cannot be configured for a "
                    "downstream task"
                )
            return nn.ModuleDict()

        options = (
            {}
            if configuration is None
            else _mapping_copy("pretraining_heads", configuration)
        )
        unknown_sections = set(options) - {"mlm", "graph", "geo"}
        if unknown_sections:
            raise ValueError(
                "unknown pretraining head sections: "
                f"{sorted(unknown_sections)}"
            )

        modules: dict[str, nn.Module] = {}
        mlm_enabled, mlm_options = self._pretraining_head_section(
            options,
            "mlm",
            allowed=frozenset({"bias", "layer_norm_eps"}),
        )
        if mlm_enabled and "1d" in self.modalities:
            smiles_encoder = self.encoders["1d"]
            vocab_size = getattr(smiles_encoder, "vocab_size", None)
            if (
                not isinstance(vocab_size, Integral)
                or isinstance(vocab_size, bool)
            ):
                raise TypeError(
                    "1d encoder must expose an integer vocab_size"
                )
            modules["mlm"] = MaskedLanguageModelingHead(
                input_dim=self.encoder_dim,
                vocab_size=int(vocab_size),
                validate_values=self.validate_values,
                **mlm_options,
            )

        graph_enabled, graph_options = self._pretraining_head_section(
            options,
            "graph",
            allowed=frozenset({"bias", "layer_norm_eps"}),
        )
        if graph_enabled and "2d" in self.modalities:
            graph_encoder = self.encoders["2d"]
            if not isinstance(graph_encoder, GraphEncoder):
                raise TypeError(
                    "2d pretraining requires GraphEncoder"
                )
            node_cardinalities = tuple(
                int(cardinality) - 1
                for cardinality in graph_encoder.node_feature_cardinalities
            )
            edge_cardinalities = tuple(
                int(cardinality) - 1
                for cardinality in graph_encoder.edge_feature_cardinalities
            )
            modules["graph"] = GraphReconstructionHead(
                input_dim=graph_encoder.hidden_size,
                embedding_dim=self.feature_dim,
                node_cardinalities=node_cardinalities,
                edge_cardinalities=edge_cardinalities,
                validate_values=self.validate_values,
                **graph_options,
            )

        geo_enabled, geo_options = self._pretraining_head_section(
            options,
            "geo",
            allowed=frozenset(
                {
                    "hidden_dim",
                    "num_radial",
                    "cutoff",
                    "max_num_neighbors",
                    "layer_norm_eps",
                    "eps",
                }
            ),
        )
        if geo_enabled and "3d" in self.modalities:
            modules["geo"] = GeometryDenoisingHead(
                context_dim=self.feature_dim,
                validate_values=self.validate_values,
                **geo_options,
            )
        return nn.ModuleDict(modules)

    def set_encoders_frozen(self, frozen: bool = True) -> None:
        """Change encoder trainability while preserving downstream pruning."""

        if not isinstance(frozen, bool):
            raise TypeError("frozen must be bool")
        self.freeze_encoders = frozen
        downstream = self.property_head is not None
        for modality in self.modalities:
            encoder_trainable = (
                not frozen
                and (not downstream or modality == self.anchor_modality)
            )
            self.encoders[modality].requires_grad_(encoder_trainable)
            if encoder_trainable:
                self.encoders[modality].train(self.training)
            else:
                self.encoders[modality].eval()

            projection_trainable = (
                not downstream or modality == self.anchor_modality
            )
            self.projection.projections[modality].requires_grad_(
                projection_trainable
            )
            if projection_trainable:
                self.projection.projections[modality].train(self.training)
            else:
                self.projection.projections[modality].eval()

    def train(self, mode: bool = True) -> "SemMol":
        if not isinstance(mode, bool):
            raise TypeError("mode must be bool")
        super().train(mode)
        downstream = self.property_head is not None
        for modality in self.modalities:
            if self.freeze_encoders or (
                downstream and modality != self.anchor_modality
            ):
                self.encoders[modality].eval()
            if downstream and modality != self.anchor_modality:
                self.projection.projections[modality].eval()
        return self

    @staticmethod
    def _sequence_length(name: str, value: object) -> int:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"batch[{name!r}] must be a sequence")
        return len(value)

    def _batch_size(self, batch: Mapping[str, Any]) -> int:
        candidates: list[tuple[str, int]] = []
        for key in ("sample_id", "smiles"):
            if key in batch:
                candidates.append(
                    (key, self._sequence_length(key, batch[key]))
                )

        tensor_keys = (
            "source_index",
            "record_index",
            "modality_mask",
            "input_ids",
            "attention_mask",
            "atomic_numbers",
            "coords",
            "atom_mask",
            "conformer_mask",
            "qm_grid",
            "qm_mask",
            "labels",
            "label_mask",
        )
        for key in tensor_keys:
            if key not in batch:
                continue
            value = batch[key]
            if not isinstance(value, Tensor):
                raise TypeError(f"batch[{key!r}] must be a torch.Tensor")
            if value.ndim == 0:
                raise ValueError(
                    f"batch[{key!r}] must have a batch dimension"
                )
            candidates.append((key, int(value.shape[0])))

        if not candidates:
            raise ValueError(
                "cannot infer batch size; provide sample_id, source_index, "
                "modality_mask, or a full-batch modality tensor"
            )
        batch_size = candidates[0][1]
        inconsistent = {
            name: size for name, size in candidates if size != batch_size
        }
        if inconsistent:
            all_sizes = {name: size for name, size in candidates}
            raise ValueError(
                f"inconsistent batch dimensions: {all_sizes}"
            )
        if batch_size <= 0:
            raise ValueError("SemMol does not accept an empty full batch")

        modality_mask = batch.get("modality_mask")
        if modality_mask is not None:
            if modality_mask.ndim != 2 or modality_mask.shape[1] != len(
                _MODALITIES
            ):
                raise ValueError(
                    "modality_mask must have shape "
                    f"[batch, {len(_MODALITIES)}]"
                )
            if modality_mask.dtype != torch.bool:
                raise TypeError("modality_mask must be bool")
        return batch_size

    @staticmethod
    def _modality_key_state(
        batch: Mapping[str, Any],
        modality: str,
    ) -> bool:
        required = _BATCH_KEYS[modality]
        present = tuple(key in batch for key in required)
        if any(present) and not all(present):
            missing = [
                key for key, exists in zip(required, present) if not exists
            ]
            raise KeyError(
                f"incomplete {modality} batch inputs; missing {missing}"
            )
        return all(present)

    def _validate_presence_index(
        self,
        batch: Mapping[str, Any],
        modality: str,
        output: EncoderOutput,
    ) -> None:
        if not self.validate_values or "modality_mask" not in batch:
            return
        modality_mask = batch["modality_mask"]
        expected = torch.nonzero(
            modality_mask[:, _MODALITY_COLUMN[modality]],
            as_tuple=False,
        ).flatten()
        if expected.device != output.sample_index.device:
            raise ValueError(
                "modality_mask and encoder sample_index must share a device"
            )
        if not torch.equal(expected, output.sample_index):
            raise ValueError(
                f"{modality} encoder sample_index disagrees with modality_mask"
            )

    def _encode_modality(
        self,
        batch: Mapping[str, Any],
        modality: str,
        *,
        batch_size: int,
    ) -> ModalityRepresentation | None:
        if not self._modality_key_state(batch, modality):
            return None
        encoder = self.encoders[modality]
        if modality == "1d":
            modality_mask = batch.get("modality_mask")
            presence = (
                None
                if modality_mask is None
                else modality_mask[:, _MODALITY_COLUMN[modality]]
            )
            encoder_output = encoder(
                batch["input_ids"],
                batch["attention_mask"],
                presence,
            )
        elif modality == "2d":
            encoder_output = encoder(
                batch["graph"],
                batch["graph_sample_index"],
                batch_size=batch_size,
            )
        elif modality == "3d":
            encoder_output = encoder(
                batch["atomic_numbers"],
                batch["coords"],
                batch["atom_mask"],
                batch["conformer_mask"],
            )
        else:
            encoder_output = encoder(
                batch["qm_grid"],
                batch["qm_mask"],
            )
        if not isinstance(encoder_output, EncoderOutput):
            raise TypeError(
                f"{modality} encoder must return EncoderOutput"
            )
        self._validate_presence_index(batch, modality, encoder_output)
        projection_output = self.projection(
            modality,
            encoder_output,
            full_batch_size=batch_size,
        )
        return ModalityRepresentation(
            encoder_output=encoder_output,
            projection_output=projection_output,
        )

    def _encode_enabled_modalities(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
    ) -> dict[str, ModalityRepresentation]:
        representations: dict[str, ModalityRepresentation] = {}
        for modality in self.modalities:
            representation = self._encode_modality(
                batch,
                modality,
                batch_size=batch_size,
            )
            if representation is not None:
                representations[modality] = representation
        return representations

    def _anchor_representation(
        self,
        representations: Mapping[str, ModalityRepresentation],
    ) -> ModalityRepresentation:
        if self.anchor_modality not in representations:
            required = _BATCH_KEYS[self.anchor_modality]
            raise KeyError(
                f"anchor modality {self.anchor_modality!r} is absent; "
                f"required batch keys are {required}"
            )
        anchor = representations[self.anchor_modality]
        return anchor

    def _empty_representation(self) -> ModalityRepresentation:
        reference = self.dcls[self.target_modalities[0]].centers
        anchor_projection = self.projection.projections[
            self.anchor_modality
        ]
        projection_parameter = next(
            (
                parameter
                for parameter in anchor_projection.parameters()
                if parameter.requires_grad
            ),
            None,
        )
        if projection_parameter is None:
            projection_parameter = next(
                (
                    parameter
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ),
                None,
            )
        parameter_zero = reference.reshape(-1)[:0].sum()
        if projection_parameter is not None:
            if projection_parameter.device != reference.device:
                raise RuntimeError(
                    "trainable model parameters and DCL buffers must share "
                    "a device"
                )
            parameter_zero = (
                projection_parameter.reshape(-1)[:1].sum() * 0.0
            ).to(dtype=reference.dtype)
        encoder_output = EncoderOutput(
            global_embedding=(
                reference.new_empty((0, self.encoder_dim))
                + parameter_zero
            ),
            sample_index=torch.empty(
                (0,),
                dtype=torch.long,
                device=reference.device,
            ),
            tokens=(
                reference.new_empty((0, 0, self.encoder_dim))
                + parameter_zero
            ),
            token_mask=torch.empty(
                (0, 0),
                dtype=torch.bool,
                device=reference.device,
            ),
        )
        projection_output = ProjectionOutput(
            raw=(
                reference.new_empty((0, self.feature_dim))
                + parameter_zero
            ),
            normalized=(
                reference.new_empty((0, self.feature_dim))
                + parameter_zero
            ),
        )
        return ModalityRepresentation(
            encoder_output=encoder_output,
            projection_output=projection_output,
        )

    def _features_for_dcl(
        self,
        representations: Mapping[str, ModalityRepresentation],
        modality: str,
    ) -> Tensor:
        if modality in representations:
            return representations[modality].projection_output.normalized
        library = self.dcls[modality]
        return library.centers.new_empty((0, self.feature_dim))

    def _warmup_active(self) -> bool:
        if self.dcl_warmup_steps == 0:
            return False
        return any(
            int(self.dcls[modality].update_count.item())
            < self.dcl_warmup_steps
            for modality in self.target_modalities
        )

    def _synchronize_update_decision(self, update_dcl: bool) -> bool:
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() <= 1
        ):
            return update_dcl
        reference = self.dcls[self.target_modalities[0]].centers
        minimum = reference.new_tensor(
            int(update_dcl),
            dtype=torch.long,
        )
        maximum = minimum.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if int(minimum.item()) != int(maximum.item()):
            raise RuntimeError(
                "update_dcl and model training mode must agree across ranks"
            )
        return update_dcl

    def _all_dcls_initialized(self) -> bool:
        return all(
            self.dcls[modality].is_initialized
            for modality in self.target_modalities
        )

    def _synchronize_dcls(self) -> None:
        for modality in self.target_modalities:
            self.dcls[modality].synchronize_distributed_state()

    def _center_snapshots(self) -> dict[str, Tensor]:
        return {
            modality: self.dcls[modality].snapshot_centers()
            for modality in self.target_modalities
        }

    def _restore_clean_graph(
        self,
        batch: Mapping[str, Any],
    ) -> Any | None:
        graph = batch.get("graph")
        if graph is None:
            return None

        restoration_fields = (
            ("x", "node_mask", "node_labels"),
            ("edge_attr", "edge_mask", "edge_labels"),
        )
        available_pairs = 0
        clean_graph = graph.clone()
        for attribute, mask_key, labels_key in restoration_fields:
            mask_present = mask_key in batch
            labels_present = labels_key in batch
            if mask_present != labels_present:
                missing = labels_key if mask_present else mask_key
                raise KeyError(
                    f"graph clean-view restoration requires batch[{missing!r}]"
                )
            if not mask_present:
                continue
            available_pairs += 1
            values = getattr(clean_graph, attribute, None)
            mask = batch[mask_key]
            labels = batch[labels_key]
            if not isinstance(values, Tensor):
                raise TypeError(f"graph.{attribute} must be a torch.Tensor")
            if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
                raise TypeError(f"batch[{mask_key!r}] must be a bool Tensor")
            if not isinstance(labels, Tensor):
                raise TypeError(
                    f"batch[{labels_key!r}] must be a torch.Tensor"
                )
            if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
                raise ValueError(
                    f"batch[{mask_key!r}] must have shape "
                    f"[{values.shape[0]}]"
                )
            if labels.shape != values.shape:
                raise ValueError(
                    f"batch[{labels_key!r}] must have shape "
                    f"{tuple(values.shape)}"
                )
            if labels.dtype != values.dtype:
                raise TypeError(
                    f"batch[{labels_key!r}] dtype must equal graph."
                    f"{attribute} dtype"
                )
            if len({values.device, mask.device, labels.device}) != 1:
                raise ValueError(
                    f"graph.{attribute}, {mask_key}, and {labels_key} "
                    "must share a device"
                )
            restored_values = values.clone()
            restored_values[mask] = labels[mask]
            setattr(clean_graph, attribute, restored_values)

        if available_pairs == 0:
            return None
        return clean_graph

    def _clean_graph_embedding(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
        graph_sample_index: Tensor,
    ) -> Tensor | None:
        clean_graph = self._restore_clean_graph(batch)
        if clean_graph is None:
            return None
        graph_encoder = self.encoders["2d"]
        if not isinstance(graph_encoder, GraphEncoder):
            raise TypeError("2d pretraining requires GraphEncoder")
        graph_projection = self.projection.projections["2d"]
        encoder_training = graph_encoder.training
        projection_training = graph_projection.training
        try:
            graph_encoder.eval()
            graph_projection.eval()
            with torch.no_grad():
                clean_output = graph_encoder(
                    clean_graph,
                    graph_sample_index,
                    batch_size=batch_size,
                )
                clean_projection = self.projection(
                    "2d",
                    clean_output,
                    full_batch_size=batch_size,
                )
        finally:
            graph_encoder.train(encoder_training)
            graph_projection.train(projection_training)
        if not torch.equal(
            clean_output.sample_index,
            graph_sample_index,
        ):
            raise RuntimeError(
                "clean and corrupted graph views changed sample ordering"
            )
        return clean_projection.normalized.detach()

    def _pretraining_predictions(
        self,
        batch: Mapping[str, Any],
        representations: Mapping[str, ModalityRepresentation],
        *,
        batch_size: int,
    ) -> tuple[
        MLMPrediction | None,
        GraphReconstructionPrediction | None,
        GeometryDenoisingPrediction | None,
    ]:
        mlm_prediction: MLMPrediction | None = None
        graph_prediction: GraphReconstructionPrediction | None = None
        geo_prediction: GeometryDenoisingPrediction | None = None

        if (
            "mlm" in self.pretraining_heads
            and "1d" in representations
            and all(key in batch for key in _BATCH_KEYS["1d"])
        ):
            output_1d = representations["1d"].encoder_output
            mlm_prediction = self.pretraining_heads["mlm"](
                output_1d.tokens,
                output_1d.sample_index,
                output_1d.token_mask,
            )

        if (
            "graph" in self.pretraining_heads
            and "2d" in representations
            and all(key in batch for key in _BATCH_KEYS["2d"])
        ):
            graph_representation = representations["2d"]
            graph_output = graph_representation.encoder_output
            if not isinstance(graph_output, GraphEncoderOutput):
                raise TypeError(
                    "2d encoder must return GraphEncoderOutput during "
                    "pretraining"
                )
            corrupted_embedding = (
                graph_representation.projection_output.normalized
            )
            clean_embedding = self._clean_graph_embedding(
                batch,
                batch_size=batch_size,
                graph_sample_index=graph_output.sample_index,
            )
            graph_prediction = self.pretraining_heads["graph"](
                graph_output.node_embedding,
                graph_output.edge_embedding,
                graph_output.sample_index,
                corrupted_embedding,
                clean_embedding,
            )

        if (
            "geo" in self.pretraining_heads
            and "3d" in representations
            and all(key in batch for key in _BATCH_KEYS["3d"])
        ):
            required = (
                "atomic_numbers",
                "coords",
                "atom_mask",
                "conformer_mask",
            )
            missing = tuple(key for key in required if key not in batch)
            if missing:
                raise KeyError(
                    "geometry denoising prediction is missing batch inputs "
                    f"{missing}"
                )
            geo_representation = representations["3d"]
            geo_output = geo_representation.encoder_output
            geo_prediction = self.pretraining_heads["geo"](
                batch["atomic_numbers"],
                batch["coords"],
                batch["atom_mask"],
                batch["conformer_mask"],
                geo_representation.projection_output.normalized,
                geo_output.sample_index,
            )
        return mlm_prediction, graph_prediction, geo_prediction

    def forward_pretrain(
        self,
        batch: Mapping[str, Any],
        *,
        update_dcl: bool | None = None,
    ) -> SemMolPretrainingOutput:
        """Encode a batch, create loss inputs, then update target DCLs."""

        batch = _mapping("batch", batch)
        if update_dcl is None:
            should_update_dcl = self.training
        elif isinstance(update_dcl, bool):
            should_update_dcl = update_dcl
        else:
            raise TypeError("update_dcl must be bool or None")
        should_update_dcl = self._synchronize_update_decision(
            should_update_dcl
        )

        batch_size = self._batch_size(batch)
        representations = self._encode_enabled_modalities(
            batch,
            batch_size=batch_size,
        )
        if self.anchor_modality not in representations:
            representations[self.anchor_modality] = (
                self._empty_representation()
            )
        anchor = self._anchor_representation(representations)
        anchor_embedding = anchor.projection_output.normalized
        (
            mlm_prediction,
            graph_prediction,
            geo_prediction,
        ) = self._pretraining_predictions(
            batch,
            representations,
            batch_size=batch_size,
        )
        target_features = {
            modality: self._features_for_dcl(
                representations,
                modality,
            )
            for modality in self.target_modalities
        }
        self._synchronize_dcls()

        initialized_before_update = self._all_dcls_initialized()
        warmup_active = self._warmup_active()
        alignment_ready = initialized_before_update and not warmup_active

        dcl_assignments: dict[str, DCLAssignment] = {}
        acsm_output: ACSMOutput | None = None
        if alignment_ready:
            center_snapshots = self._center_snapshots()
            dcl_assignments = {
                modality: self.dcls[modality].assign(
                    target_features[modality]
                )
                for modality in self.target_modalities
            }
            acsm_output = self.acsm(
                anchor_embedding,
                center_snapshots,
                anchor_modality=self.anchor_modality,
            )

        dcl_updates: dict[str, DCLUpdate] = {}
        if should_update_dcl:
            for modality in self.target_modalities:
                dcl_updates[modality] = self.dcls[
                    modality
                ].update_centers(
                    target_features[modality],
                    synchronize_state=False,
                )

        return SemMolPretrainingOutput(
            batch_size=batch_size,
            representations=representations,
            anchor_modality=self.anchor_modality,
            anchor_sample_index=anchor.encoder_output.sample_index,
            anchor_embedding=anchor_embedding,
            dcl_assignments=dcl_assignments,
            acsm_output=acsm_output,
            dcl_updates=dcl_updates,
            dcl_initialized=self._all_dcls_initialized(),
            alignment_ready=alignment_ready,
            warmup_active=warmup_active,
            mlm_prediction=mlm_prediction,
            graph_prediction=graph_prediction,
            geo_prediction=geo_prediction,
        )

    def forward_finetune(
        self,
        batch: Mapping[str, Any],
    ) -> SemMolFinetuningOutput:
        """Predict properties from an Anchor query and frozen DCL semantics."""

        if self.property_head is None:
            raise RuntimeError(
                "finetuning requires task configuration and a property head"
            )
        batch = _mapping("batch", batch)
        batch_size = self._batch_size(batch)
        anchor = self._encode_modality(
            batch,
            self.anchor_modality,
            batch_size=batch_size,
        )
        if anchor is None:
            required = _BATCH_KEYS[self.anchor_modality]
            raise KeyError(
                f"finetuning requires anchor inputs {required}"
            )
        expected_index = torch.arange(
            batch_size,
            dtype=torch.long,
            device=anchor.encoder_output.sample_index.device,
        )
        if not torch.equal(
            anchor.encoder_output.sample_index,
            expected_index,
        ):
            raise ValueError(
                "finetuning requires the anchor modality for every sample "
                "in original batch order"
            )
        self._synchronize_dcls()
        if not self._all_dcls_initialized():
            uninitialized = [
                modality
                for modality in self.target_modalities
                if not self.dcls[modality].is_initialized
            ]
            raise RuntimeError(
                "cannot finetune before loading initialized DCL centers for "
                f"{uninitialized}"
            )

        acsm_output = self.acsm(
            anchor.projection_output.normalized,
            self._center_snapshots(),
            anchor_modality=self.anchor_modality,
        )
        fused_features = torch.cat(
            (
                acsm_output.anchor_embedding,
                acsm_output.positive_embedding,
            ),
            dim=-1,
        )
        predictions = self.property_head(fused_features)
        return SemMolFinetuningOutput(
            predictions=predictions,
            fused_features=fused_features,
            anchor_sample_index=anchor.encoder_output.sample_index,
            anchor_representation=anchor,
            acsm_output=acsm_output,
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        mode: str = "pretrain",
        *,
        update_dcl: bool | None = None,
    ) -> SemMolPretrainingOutput | SemMolFinetuningOutput:
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("mode must be a non-empty string")
        normalized_mode = mode.strip().lower()
        if normalized_mode == "pretrain":
            return self.forward_pretrain(
                batch,
                update_dcl=update_dcl,
            )
        if normalized_mode == "finetune":
            if update_dcl is not None:
                raise ValueError(
                    "update_dcl is only valid in pretrain mode"
                )
            return self.forward_finetune(batch)
        raise ValueError(
            f"unsupported mode={mode!r}; expected 'pretrain' or 'finetune'"
        )


__all__ = [
    "ModalityRepresentation",
    "SemMol",
    "SemMolFinetuningOutput",
    "SemMolPretrainingOutput",
]

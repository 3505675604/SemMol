"""Unified multi-task objective for SemMol self-supervised pretraining."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Final

from torch import Tensor, nn

from src.losses.acsm_loss import (
    ACSMContrastiveLoss,
    ACSMContrastiveLossOutput,
)
from src.losses.common import LossComponent, connected_zero, normalized_loss
from src.losses.geo_loss import (
    CoordinateDenoisingLoss,
    CoordinateDenoisingLossOutput,
)
from src.losses.graph_loss import (
    GraphReconstructionLoss,
    GraphReconstructionLossOutput,
)
from src.losses.mlm_loss import MaskedLanguageModelingLoss
from src.models.heads.pretraining_heads import (
    GeometryDenoisingPrediction,
    GraphReconstructionPrediction,
    MLMPrediction,
)
from src.models.semmol import SemMolPretrainingOutput


_WEIGHT_KEYS: Final[tuple[str, ...]] = (
    "mlm",
    "graph",
    "geo",
    "pseudo",
    "alignment",
)
_LEGACY_AMBIGUOUS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "acsm",
        "cluster",
        "align",
        "weight_1d",
        "weight_2d",
        "weight_3d",
        "weight_acsm",
    }
)


@dataclass(frozen=True)
class SemMolPretrainLossOutput:
    """Joint scalar loss with every unweighted component kept as a Tensor."""

    total_loss: Tensor
    mlm_loss: LossComponent
    graph_loss: GraphReconstructionLossOutput
    geo_loss: CoordinateDenoisingLossOutput
    pseudo_loss: LossComponent
    alignment_loss: LossComponent
    acsm: ACSMContrastiveLossOutput
    component_counts: dict[str, Tensor]
    weighted_losses: dict[str, Tensor]


def _weight(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} loss weight must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(
            f"{name} loss weight must be finite and non-negative"
        )
    return normalized


def _configured_weights(
    loss_config: Mapping[str, object] | None,
    *,
    mlm_weight: object,
    graph_weight: object,
    geo_weight: object,
    pseudo_weight: object,
    alignment_weight: object,
) -> dict[str, float]:
    weights = {
        "mlm": _weight("mlm", mlm_weight),
        "graph": _weight("graph", graph_weight),
        "geo": _weight("geo", geo_weight),
        "pseudo": _weight("pseudo", pseudo_weight),
        "alignment": _weight("alignment", alignment_weight),
    }
    if loss_config is None:
        return weights
    if not isinstance(loss_config, Mapping):
        raise TypeError("loss_config must be a mapping or None")

    normalized_config: dict[str, object] = {}
    for raw_key, value in loss_config.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(
                "loss_config keys must be non-empty strings"
            )
        key = raw_key.strip().lower()
        if key in normalized_config:
            raise ValueError(
                f"duplicate normalized loss_config key {key!r}"
            )
        normalized_config[key] = value

    ambiguous = set(normalized_config) & _LEGACY_AMBIGUOUS_KEYS
    if ambiguous:
        raise ValueError(
            "legacy loss keys are intentionally unsupported because they "
            "double-count or ambiguously combine paper objectives: "
            f"{sorted(ambiguous)}. Use only {_WEIGHT_KEYS}."
        )
    unknown = set(normalized_config) - set(_WEIGHT_KEYS)
    if unknown:
        raise ValueError(
            f"unknown loss_config keys {sorted(unknown)}; "
            f"expected only {_WEIGHT_KEYS}"
        )
    for key, value in normalized_config.items():
        weights[key] = _weight(key, value)
    return weights


class SemMolPretrainTotalLoss(nn.Module):
    """Combine reconstruction, ACSM pseudo-pair, and alignment objectives.

    The top-level pseudo and alignment weights are installed in
    :class:`ACSMContrastiveLoss` and are not multiplied a second time here.
    A direct YAML ``loss`` mapping can contain exactly ``mlm``, ``graph``,
    ``geo``, ``pseudo``, and ``alignment``.
    """

    def __init__(
        self,
        loss_config: Mapping[str, object] | None = None,
        *,
        mlm_weight: float = 1.0,
        graph_weight: float = 1.0,
        geo_weight: float = 0.5,
        pseudo_weight: float = 0.1,
        alignment_weight: float = 0.01,
        temperature: float = 0.07,
        modality_weights: Mapping[str, object] | None = None,
        warmup_epochs: int = 5,
        alignment_metric: str = "mse",
        ignore_index: int = -100,
        graph_node_weight: float = 1.0,
        graph_edge_weight: float = 1.0,
        graph_structure_weight: float = 0.1,
        geo_mse_weight: float = 1.0,
        geo_cosine_weight: float = 1.0,
        distributed_sync: bool = True,
        eps: float = 1.0e-8,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(distributed_sync, bool):
            raise TypeError("distributed_sync must be bool")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")
        weights = _configured_weights(
            loss_config,
            mlm_weight=mlm_weight,
            graph_weight=graph_weight,
            geo_weight=geo_weight,
            pseudo_weight=pseudo_weight,
            alignment_weight=alignment_weight,
        )
        self.mlm_weight = weights["mlm"]
        self.graph_weight = weights["graph"]
        self.geo_weight = weights["geo"]
        self.pseudo_weight = weights["pseudo"]
        self.alignment_weight = weights["alignment"]
        self.distributed_sync = distributed_sync

        self.mlm = MaskedLanguageModelingLoss(
            ignore_index=ignore_index,
            distributed_sync=distributed_sync,
            validate_values=validate_values,
        )
        self.graph = GraphReconstructionLoss(
            node_weight=graph_node_weight,
            edge_weight=graph_edge_weight,
            structure_weight=graph_structure_weight,
            ignore_index=ignore_index,
            distributed_sync=distributed_sync,
            eps=eps,
            validate_values=validate_values,
        )
        self.geo = CoordinateDenoisingLoss(
            mse_weight=geo_mse_weight,
            cosine_weight=geo_cosine_weight,
            distributed_sync=distributed_sync,
            eps=eps,
            validate_values=validate_values,
        )
        self.acsm = ACSMContrastiveLoss(
            temperature=temperature,
            modality_weights=modality_weights,
            pseudo_weight=self.pseudo_weight,
            alignment_weight=self.alignment_weight,
            warmup_epochs=warmup_epochs,
            alignment_metric=alignment_metric,
            distributed_sync=distributed_sync,
            eps=eps,
            validate_values=validate_values,
        )

    def _empty_component(self, reference: Tensor) -> LossComponent:
        return normalized_loss(
            connected_zero(reference),
            0,
            reference=reference,
            distributed_sync=self.distributed_sync,
        )

    def _empty_graph(
        self,
        reference: Tensor,
    ) -> GraphReconstructionLossOutput:
        node = self._empty_component(reference)
        edge = self._empty_component(reference)
        structure = self._empty_component(reference)
        loss = (
            self.graph.node_weight * node.loss
            + self.graph.edge_weight * edge.loss
            + self.graph.structure_weight * structure.loss
        )
        return GraphReconstructionLossOutput(
            loss=loss,
            node=node,
            edge=edge,
            structure=structure,
        )

    def _empty_geo(
        self,
        reference: Tensor,
    ) -> CoordinateDenoisingLossOutput:
        mse = self._empty_component(reference)
        direction = self._empty_component(reference)
        loss = (
            self.geo.mse_weight * mse.loss
            + self.geo.cosine_weight * direction.loss
        )
        return CoordinateDenoisingLossOutput(
            loss=loss,
            mse=mse,
            direction=direction,
        )

    @staticmethod
    def _batch_tensor(
        batch: Mapping[str, object],
        key: str,
    ) -> Tensor:
        if key not in batch:
            raise KeyError(
                f"pretraining prediction requires batch[{key!r}]"
            )
        value = batch[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"batch[{key!r}] must be a Tensor")
        return value

    def _mlm_component(
        self,
        prediction: MLMPrediction | None,
        batch: Mapping[str, object],
        reference: Tensor,
    ) -> LossComponent:
        if prediction is None:
            return self._empty_component(reference)
        if not isinstance(prediction, MLMPrediction):
            raise TypeError(
                "outputs.mlm_prediction must be an MLMPrediction or None"
            )
        labels = self._batch_tensor(batch, "mlm_labels")
        return self.mlm.compute(
            prediction.logits,
            labels,
            prediction.sample_index,
        )

    def _graph_component(
        self,
        prediction: GraphReconstructionPrediction | None,
        batch: Mapping[str, object],
        reference: Tensor,
    ) -> GraphReconstructionLossOutput:
        if prediction is None:
            return self._empty_graph(reference)
        if not isinstance(prediction, GraphReconstructionPrediction):
            raise TypeError(
                "outputs.graph_prediction must be a "
                "GraphReconstructionPrediction or None"
            )
        node_labels = self._batch_tensor(batch, "node_labels")
        edge_labels = self._batch_tensor(batch, "edge_labels")
        return self.graph.compute(
            prediction.node_logits,
            node_labels,
            prediction.edge_logits,
            edge_labels,
            corrupted_embedding=prediction.corrupted_embedding,
            clean_embedding=prediction.clean_embedding,
        )

    def _geo_component(
        self,
        prediction: GeometryDenoisingPrediction | None,
        batch: Mapping[str, object],
        reference: Tensor,
    ) -> CoordinateDenoisingLossOutput:
        if prediction is None:
            return self._empty_geo(reference)
        if not isinstance(prediction, GeometryDenoisingPrediction):
            raise TypeError(
                "outputs.geo_prediction must be a "
                "GeometryDenoisingPrediction or None"
            )
        target_noise = self._batch_tensor(batch, "coord_noise")
        return self.geo.compute(
            prediction.predicted_noise,
            target_noise,
            valid_mask=prediction.valid_mask,
        )

    def compute(
        self,
        outputs: SemMolPretrainingOutput,
        batch: Mapping[str, object],
        epoch: int = 0,
    ) -> SemMolPretrainLossOutput:
        """Compute every objective in a fixed DDP collective order."""

        if not isinstance(outputs, SemMolPretrainingOutput):
            raise TypeError(
                "outputs must be a SemMolPretrainingOutput, not a loose mapping"
            )
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        reference = outputs.anchor_embedding
        if not isinstance(reference, Tensor):
            raise TypeError("outputs.anchor_embedding must be a Tensor")
        if not reference.is_floating_point():
            raise TypeError("outputs.anchor_embedding must be floating point")

        mlm = self._mlm_component(
            outputs.mlm_prediction,
            batch,
            reference,
        )
        graph = self._graph_component(
            outputs.graph_prediction,
            batch,
            reference,
        )
        geo = self._geo_component(
            outputs.geo_prediction,
            batch,
            reference,
        )
        acsm = self.acsm.compute(
            outputs.acsm_output,
            epoch=epoch,
            reference=reference,
        )

        weighted_losses = {
            "mlm": self.mlm_weight * mlm.loss,
            "graph": self.graph_weight * graph.loss,
            "geo": self.geo_weight * geo.loss,
            "pseudo": (
                self.pseudo_weight
                * acsm.pseudo_scale
                * acsm.pseudo.loss
            ),
            "alignment": (
                self.alignment_weight * acsm.alignment.loss
            ),
        }
        total_loss = (
            weighted_losses["mlm"]
            + weighted_losses["graph"]
            + weighted_losses["geo"]
            + acsm.loss
        )
        component_counts = {
            "mlm": mlm.global_count,
            "graph_node": graph.node.global_count,
            "graph_edge": graph.edge.global_count,
            "graph_structure": graph.structure.global_count,
            "geo_mse": geo.mse.global_count,
            "geo_direction": geo.direction.global_count,
            "pseudo": acsm.pseudo.global_count,
            "alignment": acsm.alignment.global_count,
        }
        return SemMolPretrainLossOutput(
            total_loss=total_loss,
            mlm_loss=mlm,
            graph_loss=graph,
            geo_loss=geo,
            pseudo_loss=acsm.pseudo,
            alignment_loss=acsm.alignment,
            acsm=acsm,
            component_counts=component_counts,
            weighted_losses=weighted_losses,
        )

    def forward(
        self,
        outputs: SemMolPretrainingOutput,
        batch: Mapping[str, object],
        epoch: int = 0,
    ) -> Tensor:
        return self.compute(outputs, batch, epoch=epoch).total_loss


__all__ = ["SemMolPretrainLossOutput", "SemMolPretrainTotalLoss"]

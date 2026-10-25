"""Losses for reconstructing the categorical and semantic content of 2D graphs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import LossComponent, connected_zero, normalized_loss


_NODE_FIELD_COUNT = 9
_EDGE_FIELD_COUNT = 3


def _nonnegative_weight(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _positive_finite(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return normalized


def _validate_logits_sequence(
    name: str,
    logits: Sequence[Tensor],
    labels: Tensor,
    *,
    expected_fields: int,
    ignore_index: int,
    validate_values: bool,
) -> tuple[Tensor, ...]:
    if isinstance(logits, (str, bytes, Tensor)) or not isinstance(
        logits, Sequence
    ):
        raise TypeError(f"{name} must be a sequence of tensors")
    normalized = tuple(logits)
    if len(normalized) != expected_fields:
        raise ValueError(
            f"{name} must contain {expected_fields} field logits, "
            f"got {len(normalized)}"
        )
    if not isinstance(labels, Tensor):
        raise TypeError(f"{name.replace('_logits', '_labels')} must be a tensor")
    expected_shape = (labels.shape[0], expected_fields) if labels.ndim == 2 else None
    if labels.ndim != 2 or labels.shape[1] != expected_fields:
        raise ValueError(
            f"{name.replace('_logits', '_labels')} must have shape "
            f"[items, {expected_fields}], got {tuple(labels.shape)}"
        )
    if labels.dtype != torch.long:
        raise TypeError(
            f"{name.replace('_logits', '_labels')} must be torch.long, "
            f"got {labels.dtype}"
        )

    first = normalized[0]
    for field_index, field_logits in enumerate(normalized):
        if not isinstance(field_logits, Tensor):
            raise TypeError(f"{name}[{field_index}] must be a tensor")
        if field_logits.ndim != 2:
            raise ValueError(
                f"{name}[{field_index}] must have shape [items, classes], "
                f"got {tuple(field_logits.shape)}"
            )
        if field_logits.shape[0] != expected_shape[0]:
            raise ValueError(
                f"{name}[{field_index}] item count must match labels: "
                f"{field_logits.shape[0]} != {expected_shape[0]}"
            )
        if field_logits.shape[1] <= 0:
            raise ValueError(f"{name}[{field_index}] must define at least one class")
        if not field_logits.is_floating_point():
            raise TypeError(
                f"{name}[{field_index}] must be floating point, "
                f"got {field_logits.dtype}"
            )
        if field_logits.device != labels.device:
            raise ValueError(
                f"{name}[{field_index}] and labels must be on the same device"
            )
        if field_logits.device != first.device:
            raise ValueError(f"all tensors in {name} must be on the same device")
        if field_logits.dtype != first.dtype:
            raise TypeError(f"all tensors in {name} must have the same dtype")
        if (
            validate_values
            and field_logits.numel() > 0
            and not bool(torch.isfinite(field_logits).all())
        ):
            raise ValueError(f"{name}[{field_index}] contains NaN or infinite values")

        if validate_values and labels.shape[0] > 0:
            field_labels = labels[:, field_index]
            valid_labels = field_labels != ignore_index
            invalid_labels = valid_labels & (
                (field_labels < 0) | (field_labels >= field_logits.shape[1])
            )
            if bool(invalid_labels.any()):
                raise ValueError(
                    f"{name.replace('_logits', '_labels')}[:, {field_index}] "
                    f"contains a class outside [0, {field_logits.shape[1]})"
                )
    return normalized


def _classification_component(
    logits: tuple[Tensor, ...],
    labels: Tensor,
    *,
    ignore_index: int,
    distributed_sync: bool,
) -> LossComponent:
    reference = connected_zero(logits[0])
    numerator = reference
    local_count = torch.zeros((), dtype=torch.long, device=labels.device)

    for field_index, field_logits in enumerate(logits):
        if field_index > 0:
            reference = reference + connected_zero(field_logits)
        field_labels = labels[:, field_index]
        valid = field_labels != ignore_index
        local_count = local_count + valid.sum(dtype=torch.long)
        if field_logits.shape[0] == 0:
            numerator = numerator + connected_zero(field_logits)
        else:
            calculation_logits = (
                field_logits.float()
                if field_logits.dtype in (torch.float16, torch.bfloat16)
                else field_logits
            )
            numerator = numerator + F.cross_entropy(
                calculation_logits,
                field_labels,
                ignore_index=ignore_index,
                reduction="sum",
            )

    return normalized_loss(
        numerator,
        local_count,
        reference=reference,
        distributed_sync=distributed_sync,
    )


def _zero_component(
    reference: Tensor,
    *,
    distributed_sync: bool,
) -> LossComponent:
    zero = connected_zero(reference)
    return normalized_loss(
        zero,
        0,
        reference=zero,
        distributed_sync=distributed_sync,
    )


@dataclass(frozen=True)
class GraphReconstructionLossOutput:
    """Scalar graph loss and its independently normalized components."""

    loss: Tensor
    node: LossComponent
    edge: LossComponent
    structure: LossComponent


class GraphReconstructionLoss(nn.Module):
    """Reconstruct all categorical graph fields and preserve graph semantics.

    The nine atom fields and three bond fields are separate classification
    tasks because their vocabularies differ. Their summed cross-entropies are
    normalized by the number of non-ignored field labels, including the
    global count when distributed synchronization is enabled.

    The structure term is active only when both a genuinely corrupted graph
    embedding and its clean counterpart are available. A missing clean view
    yields a graph-connected zero; silently reusing the corrupted tensor as
    its own target would create a meaningless zero loss and is rejected.
    """

    def __init__(
        self,
        *,
        node_weight: float = 1.0,
        edge_weight: float = 1.0,
        structure_weight: float = 0.1,
        ignore_index: int = -100,
        distributed_sync: bool = True,
        eps: float = 1.0e-8,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        self.node_weight = _nonnegative_weight("node_weight", node_weight)
        self.edge_weight = _nonnegative_weight("edge_weight", edge_weight)
        self.structure_weight = _nonnegative_weight(
            "structure_weight", structure_weight
        )
        if not isinstance(ignore_index, Integral) or isinstance(ignore_index, bool):
            raise TypeError("ignore_index must be an integer")
        if not isinstance(distributed_sync, bool):
            raise TypeError("distributed_sync must be bool")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")
        self.ignore_index = int(ignore_index)
        self.distributed_sync = distributed_sync
        self.eps = _positive_finite("eps", eps)
        self.validate_values = validate_values

    def _structure_component(
        self,
        corrupted_embedding: Tensor | None,
        clean_embedding: Tensor | None,
        *,
        fallback_reference: Tensor,
    ) -> LossComponent:
        if corrupted_embedding is None or clean_embedding is None:
            reference = fallback_reference
            if corrupted_embedding is not None:
                if not isinstance(corrupted_embedding, Tensor):
                    raise TypeError("corrupted_embedding must be a tensor")
                if not corrupted_embedding.is_floating_point():
                    raise TypeError("corrupted_embedding must be floating point")
                if (
                    corrupted_embedding.ndim != 2
                    or corrupted_embedding.shape[1] <= 0
                ):
                    raise ValueError(
                        "corrupted_embedding must have shape "
                        "[compact_batch, positive_dim]"
                    )
                if corrupted_embedding.device != fallback_reference.device:
                    raise ValueError(
                        "corrupted_embedding and categorical logits must be "
                        "on the same device"
                    )
                if (
                    self.validate_values
                    and corrupted_embedding.numel() > 0
                    and not bool(torch.isfinite(corrupted_embedding).all())
                ):
                    raise ValueError(
                        "corrupted_embedding contains NaN or infinite values"
                    )
                reference = reference + connected_zero(corrupted_embedding)
            if clean_embedding is not None:
                if not isinstance(clean_embedding, Tensor):
                    raise TypeError("clean_embedding must be a tensor")
                if not clean_embedding.is_floating_point():
                    raise TypeError("clean_embedding must be floating point")
                if clean_embedding.ndim != 2 or clean_embedding.shape[1] <= 0:
                    raise ValueError(
                        "clean_embedding must have shape "
                        "[compact_batch, positive_dim]"
                    )
                if clean_embedding.device != fallback_reference.device:
                    raise ValueError(
                        "clean_embedding and categorical logits must be "
                        "on the same device"
                    )
                if (
                    self.validate_values
                    and clean_embedding.numel() > 0
                    and not bool(torch.isfinite(clean_embedding).all())
                ):
                    raise ValueError(
                        "clean_embedding contains NaN or infinite values"
                    )
                reference = reference + connected_zero(clean_embedding)
            return _zero_component(
                reference,
                distributed_sync=self.distributed_sync,
            )
        if not isinstance(corrupted_embedding, Tensor):
            raise TypeError("corrupted_embedding must be a tensor")
        if not isinstance(clean_embedding, Tensor):
            raise TypeError("clean_embedding must be a tensor")
        if corrupted_embedding is clean_embedding:
            raise ValueError(
                "corrupted_embedding and clean_embedding must be different tensors"
            )
        if corrupted_embedding.ndim != 2:
            raise ValueError(
                "corrupted_embedding must have shape [compact_batch, dim], got "
                f"{tuple(corrupted_embedding.shape)}"
            )
        if clean_embedding.ndim != 2:
            raise ValueError(
                "clean_embedding must have shape [compact_batch, dim], got "
                f"{tuple(clean_embedding.shape)}"
            )
        if corrupted_embedding.shape != clean_embedding.shape:
            raise ValueError(
                "corrupted_embedding and clean_embedding must have the same shape"
            )
        if corrupted_embedding.shape[1] <= 0:
            raise ValueError("graph embeddings must have a positive feature dimension")
        if not corrupted_embedding.is_floating_point():
            raise TypeError("corrupted_embedding must be floating point")
        if not clean_embedding.is_floating_point():
            raise TypeError("clean_embedding must be floating point")
        if corrupted_embedding.dtype != clean_embedding.dtype:
            raise TypeError("graph embeddings must have the same dtype")
        if corrupted_embedding.device != clean_embedding.device:
            raise ValueError("graph embeddings must be on the same device")
        if corrupted_embedding.device != fallback_reference.device:
            raise ValueError(
                "graph embeddings and categorical logits must be on the same device"
            )
        if self.validate_values:
            if (
                corrupted_embedding.numel() > 0
                and not bool(torch.isfinite(corrupted_embedding).all())
            ):
                raise ValueError(
                    "corrupted_embedding contains NaN or infinite values"
                )
            if clean_embedding.numel() > 0 and not bool(
                torch.isfinite(clean_embedding).all()
            ):
                raise ValueError("clean_embedding contains NaN or infinite values")

        calculation_dtype = (
            torch.float32
            if corrupted_embedding.dtype in (torch.float16, torch.bfloat16)
            else corrupted_embedding.dtype
        )
        corrupted = F.normalize(
            corrupted_embedding.to(dtype=calculation_dtype),
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        clean = F.normalize(
            clean_embedding.to(dtype=calculation_dtype),
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        numerator = (1.0 - (corrupted * clean).sum(dim=-1)).sum()
        reference = (
            connected_zero(corrupted_embedding)
            + connected_zero(clean_embedding)
        )
        return normalized_loss(
            numerator,
            corrupted_embedding.shape[0],
            reference=reference,
            distributed_sync=self.distributed_sync,
        )

    def compute(
        self,
        node_logits: Sequence[Tensor],
        node_labels: Tensor,
        edge_logits: Sequence[Tensor],
        edge_labels: Tensor,
        *,
        corrupted_embedding: Tensor | None = None,
        clean_embedding: Tensor | None = None,
    ) -> GraphReconstructionLossOutput:
        """Return the weighted loss together with all normalized components."""

        normalized_node_logits = _validate_logits_sequence(
            "node_logits",
            node_logits,
            node_labels,
            expected_fields=_NODE_FIELD_COUNT,
            ignore_index=self.ignore_index,
            validate_values=self.validate_values,
        )
        normalized_edge_logits = _validate_logits_sequence(
            "edge_logits",
            edge_logits,
            edge_labels,
            expected_fields=_EDGE_FIELD_COUNT,
            ignore_index=self.ignore_index,
            validate_values=self.validate_values,
        )
        if normalized_node_logits[0].device != normalized_edge_logits[0].device:
            raise ValueError("node_logits and edge_logits must be on the same device")

        node = _classification_component(
            normalized_node_logits,
            node_labels,
            ignore_index=self.ignore_index,
            distributed_sync=self.distributed_sync,
        )
        edge = _classification_component(
            normalized_edge_logits,
            edge_labels,
            ignore_index=self.ignore_index,
            distributed_sync=self.distributed_sync,
        )
        structure_reference = (
            connected_zero(normalized_node_logits[0])
            + connected_zero(normalized_edge_logits[0])
        )
        structure = self._structure_component(
            corrupted_embedding,
            clean_embedding,
            fallback_reference=structure_reference,
        )
        loss = (
            self.node_weight * node.loss
            + self.edge_weight * edge.loss
            + self.structure_weight * structure.loss
        )
        return GraphReconstructionLossOutput(
            loss=loss,
            node=node,
            edge=edge,
            structure=structure,
        )

    def forward(
        self,
        node_logits: Sequence[Tensor],
        node_labels: Tensor,
        edge_logits: Sequence[Tensor],
        edge_labels: Tensor,
        *,
        corrupted_embedding: Tensor | None = None,
        clean_embedding: Tensor | None = None,
    ) -> Tensor:
        return self.compute(
            node_logits,
            node_labels,
            edge_logits,
            edge_labels,
            corrupted_embedding=corrupted_embedding,
            clean_embedding=clean_embedding,
        ).loss


__all__ = ["GraphReconstructionLoss", "GraphReconstructionLossOutput"]

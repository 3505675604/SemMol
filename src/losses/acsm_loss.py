"""Stable ACSM pseudo-pair contrastive and feature-alignment losses."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from src.losses.common import LossComponent, connected_zero, normalized_loss
from src.models.alignment.acsm import ACSMOutput, ModalityMatch


_ALIGNMENT_METRICS: Final[frozenset[str]] = frozenset({"cosine", "mse"})


@dataclass(frozen=True)
class ACSMContrastiveLossOutput:
    """Separated ACSM losses and their weighted training objective."""

    loss: Tensor
    pseudo: LossComponent
    alignment: LossComponent
    modality_losses: dict[str, LossComponent]
    pseudo_scale: Tensor


def _finite_nonnegative(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _positive_finite(name: str, value: object) -> float:
    normalized = _finite_nonnegative(name, value)
    if normalized == 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _alignment_metric(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("alignment_metric must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in _ALIGNMENT_METRICS:
        raise ValueError(
            "unsupported alignment_metric="
            f"{value!r}; expected one of {sorted(_ALIGNMENT_METRICS)}"
        )
    return normalized


def _modality_weights(
    values: Mapping[str, object] | None,
) -> dict[str, float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("modality_weights must be a mapping or None")
    normalized: dict[str, float] = {}
    for raw_name, raw_weight in values.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(
                "modality_weights keys must be non-empty strings"
            )
        name = raw_name.strip().lower()
        if name in normalized:
            raise ValueError(
                f"duplicate normalized modality weight for {name!r}"
            )
        normalized[name] = _finite_nonnegative(
            f"modality_weights[{name!r}]",
            raw_weight,
        )
    return normalized


class ACSMContrastiveLoss(nn.Module):
    """Compute manuscript Equation 10 and its feature-alignment regularizer.

    Equation 10 is evaluated with a masked ``logsumexp``. Rows without a
    trusted negative are excluded instead of being interpreted as zero-loss
    training examples. Modality weights apply to valid modality-sample pairs
    and are normalized by their global weighted count. Consequently, changing
    all modality weights by the same factor does not change the loss scale.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        *,
        modality_weights: Mapping[str, object] | None = None,
        pseudo_weight: float = 0.1,
        alignment_weight: float = 0.01,
        warmup_epochs: int = 5,
        alignment_metric: str = "mse",
        distributed_sync: bool = True,
        eps: float = 1.0e-8,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = _positive_finite("temperature", temperature)
        self.modality_weights = _modality_weights(modality_weights)
        self.pseudo_weight = _finite_nonnegative(
            "pseudo_weight",
            pseudo_weight,
        )
        self.alignment_weight = _finite_nonnegative(
            "alignment_weight",
            alignment_weight,
        )
        self.warmup_epochs = _nonnegative_integer(
            "warmup_epochs",
            warmup_epochs,
        )
        self.alignment_metric = _alignment_metric(alignment_metric)
        self.eps = _positive_finite("eps", eps)
        if not isinstance(distributed_sync, bool):
            raise TypeError("distributed_sync must be bool")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")
        self.distributed_sync = distributed_sync
        self.validate_values = validate_values

    def _reference(
        self,
        acsm_output: ACSMOutput | None,
        reference: Tensor | None,
    ) -> Tensor:
        if reference is not None and not isinstance(reference, Tensor):
            raise TypeError("reference must be a Tensor or None")
        if acsm_output is not None:
            if not isinstance(acsm_output, ACSMOutput):
                raise TypeError("acsm_output must be an ACSMOutput or None")
            output_reference = acsm_output.anchor_embedding
            if not isinstance(output_reference, Tensor):
                raise TypeError(
                    "ACSMOutput.anchor_embedding must be a Tensor"
                )
            if reference is not None and (
                reference.device != output_reference.device
            ):
                raise ValueError(
                    "reference and ACSMOutput tensors must share a device"
                )
            return output_reference
        if reference is None:
            raise ValueError(
                "reference is required when acsm_output is None"
            )
        return reference

    def _empty_component(self, reference: Tensor) -> LossComponent:
        return normalized_loss(
            connected_zero(reference),
            0,
            reference=reference,
            distributed_sync=self.distributed_sync,
        )

    def _pseudo_scale(self, epoch: int, reference: Tensor) -> Tensor:
        if not isinstance(epoch, Integral) or isinstance(epoch, bool):
            raise TypeError("epoch must be an integer")
        epoch_index = int(epoch)
        if self.warmup_epochs == 0:
            scale = 1.0
        else:
            scale = min(
                max(float(epoch_index), 0.0) / float(self.warmup_epochs),
                1.0,
            )
        return reference.new_tensor(scale, dtype=torch.float32)

    def _validate_embeddings(
        self,
        acsm_output: ACSMOutput,
    ) -> tuple[Tensor, Tensor]:
        anchor = acsm_output.anchor_embedding
        positive = acsm_output.positive_embedding
        if not isinstance(anchor, Tensor) or not isinstance(positive, Tensor):
            raise TypeError(
                "ACSM anchor_embedding and positive_embedding must be Tensors"
            )
        if anchor.ndim != 2 or positive.ndim != 2:
            raise ValueError(
                "ACSM embeddings must both have shape [batch, feature_dim]"
            )
        if anchor.shape != positive.shape:
            raise ValueError(
                "ACSM anchor and positive embeddings must have identical "
                f"shapes, got {tuple(anchor.shape)} and {tuple(positive.shape)}"
            )
        if not anchor.is_floating_point() or not positive.is_floating_point():
            raise TypeError("ACSM embeddings must be floating point")
        if anchor.device != positive.device:
            raise ValueError(
                "ACSM anchor and positive embeddings must share a device"
            )
        prepared_anchor = F.normalize(
            anchor.float(),
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        prepared_positive = F.normalize(
            positive.float(),
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        if self.validate_values and (
            not bool(torch.isfinite(prepared_anchor).all())
            or not bool(torch.isfinite(prepared_positive).all())
        ):
            raise ValueError("ACSM embeddings contain NaN or infinite values")
        return prepared_anchor, prepared_positive

    def _ordered_modalities(
        self,
        acsm_output: ACSMOutput,
    ) -> tuple[tuple[str, ...], dict[str, ModalityMatch]]:
        if not isinstance(acsm_output.modality_matches, Mapping):
            raise TypeError("ACSM modality_matches must be a mapping")
        normalized_matches: dict[str, ModalityMatch] = {}
        for raw_name, match in acsm_output.modality_matches.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(
                    "ACSM modality names must be non-empty strings"
                )
            name = raw_name.strip().lower()
            if name in normalized_matches:
                raise ValueError(
                    f"duplicate normalized ACSM modality {name!r}"
                )
            if not isinstance(match, ModalityMatch):
                raise TypeError(
                    f"ACSM match for {name!r} must be a ModalityMatch"
                )
            normalized_matches[name] = match

        target_modalities_list: list[str] = []
        for raw_name in acsm_output.target_modalities:
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(
                    "ACSM target_modalities entries must be non-empty strings"
                )
            target_modalities_list.append(raw_name.strip().lower())
        target_modalities = tuple(target_modalities_list)
        if (
            len(set(target_modalities)) != len(target_modalities)
            or set(target_modalities) != set(normalized_matches)
        ):
            raise ValueError(
                "ACSM target_modalities must contain each modality_matches "
                "key exactly once"
            )
        unknown_weights = set(self.modality_weights) - set(normalized_matches)
        if unknown_weights:
            raise ValueError(
                "modality_weights contains targets absent from ACSM output: "
                f"{sorted(unknown_weights)}"
            )
        return tuple(sorted(normalized_matches)), normalized_matches

    def _validate_distributed_signature(
        self,
        *,
        output_present: bool,
        ordered_modalities: tuple[str, ...],
        reference: Tensor,
    ) -> None:
        if (
            not self.distributed_sync
            or not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() <= 1
        ):
            return
        encoded = "\0".join(ordered_modalities).encode("utf-8")
        digest = hashlib.blake2b(encoded, digest_size=8).digest()
        modality_signature = (
            int.from_bytes(digest, byteorder="big", signed=False)
            & ((1 << 62) - 1)
        )
        state = torch.tensor(
            [
                int(output_present),
                len(ordered_modalities),
                modality_signature,
            ],
            dtype=torch.long,
            device=reference.device,
        )
        minimum = state.clone()
        maximum = state.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if not torch.equal(minimum, maximum):
            raise RuntimeError(
                "ACSM output presence and target modalities must agree "
                "across all distributed ranks"
            )

    def _match_numerator(
        self,
        match: ModalityMatch,
        *,
        positive_logits: Tensor,
        batch_size: int,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        negative = match.negative_similarities
        negative_mask = match.negative_mask
        valid_rows = match.valid_negative_rows
        negative_count = match.negative_count
        if not isinstance(negative, Tensor) or negative.ndim != 2:
            raise ValueError(
                "negative_similarities must have shape [batch, negatives]"
            )
        if (
            negative.shape[0] != batch_size
            or negative.device != reference.device
        ):
            raise ValueError(
                "negative_similarities must match the ACSM batch and device"
            )
        if not negative.is_floating_point():
            raise TypeError("negative_similarities must be floating point")
        if (
            not isinstance(negative_mask, Tensor)
            or negative_mask.dtype != torch.bool
            or negative_mask.shape != negative.shape
            or negative_mask.device != negative.device
        ):
            raise ValueError(
                "negative_mask must be a bool Tensor matching "
                "negative_similarities"
            )
        if (
            not isinstance(valid_rows, Tensor)
            or valid_rows.dtype != torch.bool
            or valid_rows.shape != (batch_size,)
            or valid_rows.device != negative.device
        ):
            raise ValueError(
                "valid_negative_rows must be a bool Tensor with shape [batch]"
            )
        if (
            not isinstance(negative_count, Tensor)
            or negative_count.shape != (batch_size,)
            or negative_count.device != negative.device
        ):
            raise ValueError(
                "negative_count must be a Tensor with shape [batch]"
            )
        if negative_count.dtype != torch.long:
            raise TypeError("negative_count must use torch.long dtype")
        if negative_count.requires_grad:
            raise ValueError("negative_count must not require gradients")
        mask_count = negative_mask.sum(dim=-1)
        if not torch.equal(valid_rows, mask_count > 0):
            raise ValueError(
                "valid_negative_rows is inconsistent with negative_mask"
            )
        if not torch.equal(negative_count, mask_count):
            raise ValueError("negative_count is inconsistent with negative_mask")
        if self.validate_values and negative_mask.any() and not bool(
            torch.isfinite(negative[negative_mask]).all()
        ):
            raise ValueError(
                "valid ACSM negative similarities contain non-finite values"
            )

        negative_logits = negative.float() / self.temperature
        negative_logits = negative_logits.masked_fill(
            ~negative_mask,
            float("-inf"),
        )
        denominator_logits = torch.cat(
            (positive_logits.unsqueeze(-1), negative_logits),
            dim=-1,
        )
        per_row = (
            torch.logsumexp(denominator_logits, dim=-1)
            - positive_logits
        )
        return per_row[valid_rows].sum(), valid_rows.sum()

    def _alignment_numerator(
        self,
        anchor: Tensor,
        positive: Tensor,
    ) -> Tensor:
        if self.alignment_metric == "cosine":
            return (
                1.0 - torch.sum(anchor * positive, dim=-1)
            ).sum()
        return torch.mean((anchor - positive).square(), dim=-1).sum()

    def compute(
        self,
        acsm_output: ACSMOutput | None,
        epoch: int = 0,
        reference: Tensor | None = None,
    ) -> ACSMContrastiveLossOutput:
        """Return ACSM components while preserving differentiable zero cases."""

        loss_reference = self._reference(acsm_output, reference)
        pseudo_scale = self._pseudo_scale(epoch, loss_reference)
        if acsm_output is None:
            self._validate_distributed_signature(
                output_present=False,
                ordered_modalities=(),
                reference=loss_reference,
            )
            pseudo = self._empty_component(loss_reference)
            alignment = self._empty_component(loss_reference)
            return ACSMContrastiveLossOutput(
                loss=connected_zero(loss_reference),
                pseudo=pseudo,
                alignment=alignment,
                modality_losses={},
                pseudo_scale=pseudo_scale,
            )

        anchor, positive = self._validate_embeddings(acsm_output)
        ordered_modalities, normalized_matches = self._ordered_modalities(
            acsm_output
        )
        self._validate_distributed_signature(
            output_present=True,
            ordered_modalities=ordered_modalities,
            reference=anchor,
        )
        positive_logits = torch.sum(anchor * positive, dim=-1)
        positive_logits = positive_logits / self.temperature

        modality_losses: dict[str, LossComponent] = {}
        modality_numerators: dict[str, Tensor] = {}
        modality_counts: dict[str, Tensor] = {}
        for modality in ordered_modalities:
            numerator, count = self._match_numerator(
                normalized_matches[modality],
                positive_logits=positive_logits,
                batch_size=int(anchor.shape[0]),
                reference=anchor,
            )
            modality_numerators[modality] = numerator
            modality_counts[modality] = count
            modality_losses[modality] = normalized_loss(
                numerator,
                count,
                reference=anchor,
                distributed_sync=self.distributed_sync,
            )

        active_weights = {
            modality: self.modality_weights.get(modality, 1.0)
            for modality in ordered_modalities
            if self.modality_weights.get(modality, 1.0) > 0.0
        }
        if active_weights:
            active_modalities = tuple(active_weights)
            pseudo_count = torch.stack(
                tuple(
                    modality_counts[modality]
                    for modality in active_modalities
                )
            ).sum()
            global_counts = torch.stack(
                tuple(
                    modality_losses[modality].global_count
                    for modality in active_modalities
                )
            )
            global_count = global_counts.sum()

            configured_weights = anchor.new_tensor(
                tuple(
                    active_weights[modality]
                    for modality in active_modalities
                ),
                dtype=torch.float64,
            )
            globally_valid_weights = torch.where(
                global_counts > 0,
                configured_weights,
                torch.zeros_like(configured_weights),
            )
            weight_scale = globally_valid_weights.max().clamp_min(
                torch.finfo(torch.float64).tiny
            )
            relative_weights = (
                globally_valid_weights / weight_scale
            ).to(dtype=anchor.dtype)
            weighted_global_count = torch.sum(
                global_counts.to(dtype=anchor.dtype) * relative_weights
            )
            count_normalizer = (
                global_count.to(dtype=anchor.dtype)
                / weighted_global_count.clamp_min(1.0)
            )
            pseudo_numerator = sum(
                modality_numerators[modality]
                * relative_weights[index]
                * count_normalizer
                for index, modality in enumerate(active_modalities)
            )
        else:
            pseudo_numerator = connected_zero(anchor)
            pseudo_count = anchor.new_zeros((), dtype=torch.long)
        pseudo = normalized_loss(
            pseudo_numerator,
            pseudo_count,
            reference=anchor,
            distributed_sync=self.distributed_sync,
        )

        alignment_numerator = self._alignment_numerator(anchor, positive)
        alignment = normalized_loss(
            alignment_numerator,
            int(anchor.shape[0]),
            reference=anchor,
            distributed_sync=self.distributed_sync,
        )
        loss = (
            self.pseudo_weight * pseudo_scale * pseudo.loss
            + self.alignment_weight * alignment.loss
        )
        return ACSMContrastiveLossOutput(
            loss=loss,
            pseudo=pseudo,
            alignment=alignment,
            modality_losses=modality_losses,
            pseudo_scale=pseudo_scale,
        )

    def forward(
        self,
        acsm_output: ACSMOutput | None,
        epoch: int = 0,
        reference: Tensor | None = None,
    ) -> Tensor:
        return self.compute(
            acsm_output,
            epoch=epoch,
            reference=reference,
        ).loss


__all__ = ["ACSMContrastiveLoss", "ACSMContrastiveLossOutput"]

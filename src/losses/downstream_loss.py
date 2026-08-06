"""Masked downstream classification and regression objectives."""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import LossComponent, connected_zero, normalized_loss


_TASK_TYPES = frozenset({"classification", "regression", "multiclass"})
_REGRESSION_LOSSES = frozenset({"mse", "mae", "huber"})
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class DownstreamTaskLoss(nn.Module):
    """Loss over finite, explicitly enabled downstream labels.

    Binary and multilabel classification consumes raw logits with
    ``binary_cross_entropy_with_logits``. Multiclass classification consumes
    one class-index target per logit row. Missing targets are removed before
    any arithmetic, so NaN labels can never contaminate the numerator.
    """

    def __init__(
        self,
        task_type: str = "classification",
        loss_type: str = "mse",
        huber_delta: float = 1.0,
        distributed_sync: bool = True,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("task_type must be a non-empty string")
        normalized_task_type = task_type.strip().lower()
        if normalized_task_type not in _TASK_TYPES:
            raise ValueError(
                f"unsupported task_type={task_type!r}; expected one of "
                f"{sorted(_TASK_TYPES)}"
            )
        if not isinstance(loss_type, str) or not loss_type.strip():
            raise ValueError("loss_type must be a non-empty string")
        normalized_loss_type = loss_type.strip().lower()
        if normalized_loss_type not in _REGRESSION_LOSSES:
            raise ValueError(
                f"unsupported loss_type={loss_type!r}; expected one of "
                f"{sorted(_REGRESSION_LOSSES)}"
            )
        if not isinstance(huber_delta, Real) or isinstance(huber_delta, bool):
            raise TypeError("huber_delta must be a real number")
        normalized_delta = float(huber_delta)
        if not math.isfinite(normalized_delta) or normalized_delta <= 0.0:
            raise ValueError("huber_delta must be positive and finite")
        if not isinstance(distributed_sync, bool):
            raise TypeError("distributed_sync must be bool")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")

        self.task_type = normalized_task_type
        self.loss_type = normalized_loss_type
        self.huber_delta = normalized_delta
        self.distributed_sync = distributed_sync
        self.validate_values = validate_values

    @staticmethod
    def _validate_real_targets(targets: Tensor) -> None:
        if targets.is_complex() or targets.dtype == torch.bool:
            raise TypeError(
                "targets must have a real numeric dtype, got "
                f"{targets.dtype}"
            )
        if not targets.is_floating_point() and targets.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                "targets must have a real numeric dtype, got "
                f"{targets.dtype}"
            )

    @staticmethod
    def _mask_tensor(mask: Tensor | None, targets: Tensor) -> Tensor:
        if mask is None:
            return torch.ones_like(targets, dtype=torch.bool)
        if not isinstance(mask, Tensor):
            raise TypeError("mask must be a torch.Tensor or None")
        if mask.shape != targets.shape:
            raise ValueError(
                "mask and targets must have the same shape: "
                f"{tuple(mask.shape)} != {tuple(targets.shape)}"
            )
        if mask.device != targets.device:
            raise ValueError(
                "mask and targets must be on the same device: "
                f"{mask.device} != {targets.device}"
            )
        if mask.dtype == torch.bool:
            return mask
        if mask.is_complex() or (
            not mask.is_floating_point() and mask.dtype not in _INTEGER_DTYPES
        ):
            raise TypeError(
                f"mask must be boolean or numeric 0/1, got {mask.dtype}"
            )
        if mask.numel() > 0:
            if mask.is_floating_point() and not bool(torch.isfinite(mask).all()):
                raise ValueError("mask cannot contain NaN or infinite values")
            if bool(torch.any((mask != 0) & (mask != 1))):
                raise ValueError("numeric mask values must be exactly 0 or 1")
        return mask.to(dtype=torch.bool)

    def _valid_mask(
        self,
        targets: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        explicit_mask = self._mask_tensor(mask, targets)
        return explicit_mask & torch.isfinite(targets)

    def _elementwise_numerator(
        self,
        predictions: Tensor,
        targets: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        valid_predictions = predictions[valid_mask]
        if valid_predictions.numel() == 0:
            return connected_zero(predictions)

        calculation_predictions = (
            valid_predictions.float()
            if valid_predictions.dtype in (torch.float16, torch.bfloat16)
            else valid_predictions
        )
        valid_targets = targets[valid_mask].to(
            dtype=calculation_predictions.dtype
        )
        if self.validate_values and not bool(
            torch.isfinite(valid_predictions).all()
        ):
            raise ValueError("predictions contain non-finite values at valid labels")

        if self.task_type == "classification":
            if self.validate_values and bool(
                torch.any((valid_targets < 0) | (valid_targets > 1))
            ):
                raise ValueError("classification targets must be in [0, 1]")
            return F.binary_cross_entropy_with_logits(
                calculation_predictions,
                valid_targets,
                reduction="sum",
            )
        if self.loss_type == "mse":
            return F.mse_loss(
                calculation_predictions,
                valid_targets,
                reduction="sum",
            )
        if self.loss_type == "mae":
            return F.l1_loss(
                calculation_predictions,
                valid_targets,
                reduction="sum",
            )
        return F.smooth_l1_loss(
            calculation_predictions,
            valid_targets,
            reduction="sum",
            beta=self.huber_delta,
        )

    def _multiclass_numerator(
        self,
        predictions: Tensor,
        targets: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        class_count = int(predictions.shape[-1])
        flat_predictions = predictions.reshape(-1, class_count)
        flat_valid = valid_mask.reshape(-1)
        valid_predictions = flat_predictions[flat_valid]
        if valid_predictions.shape[0] == 0:
            return connected_zero(predictions)
        raw_targets = targets.reshape(-1)[flat_valid]
        if raw_targets.is_floating_point() and bool(
            torch.any(raw_targets != torch.round(raw_targets))
        ):
            raise ValueError("multiclass targets must contain integer class indices")
        class_targets = raw_targets.to(dtype=torch.long)
        if bool(torch.any(class_targets < 0)) or bool(
            torch.any(class_targets >= class_count)
        ):
            raise ValueError(
                "multiclass targets must be in "
                f"[0, {class_count}) at valid positions"
            )
        if self.validate_values and not bool(
            torch.isfinite(valid_predictions).all()
        ):
            raise ValueError("predictions contain non-finite values at valid labels")
        calculation_predictions = (
            valid_predictions.float()
            if valid_predictions.dtype in (torch.float16, torch.bfloat16)
            else valid_predictions
        )
        return F.cross_entropy(
            calculation_predictions,
            class_targets,
            reduction="sum",
        )

    def compute(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> LossComponent:
        if not isinstance(predictions, Tensor):
            raise TypeError("predictions must be a torch.Tensor")
        if not isinstance(targets, Tensor):
            raise TypeError("targets must be a torch.Tensor")
        if not predictions.is_floating_point():
            raise TypeError(
                "predictions must be floating point raw logits or values, got "
                f"{predictions.dtype}"
            )
        self._validate_real_targets(targets)
        if predictions.device != targets.device:
            raise ValueError(
                "predictions and targets must be on the same device: "
                f"{predictions.device} != {targets.device}"
            )

        if self.task_type == "multiclass":
            if predictions.ndim < 2:
                raise ValueError(
                    "multiclass predictions must have shape [..., classes]"
                )
            if predictions.shape[-1] < 2:
                raise ValueError(
                    "multiclass predictions require at least two classes"
                )
            if predictions.shape[:-1] != targets.shape:
                raise ValueError(
                    "multiclass targets must match all prediction dimensions "
                    "except classes: "
                    f"{tuple(targets.shape)} != {tuple(predictions.shape[:-1])}"
                )
        else:
            if predictions.ndim < 1:
                raise ValueError(
                    "classification/regression predictions must have at least "
                    "one dimension"
                )
            if predictions.shape != targets.shape:
                raise ValueError(
                    "predictions and targets must have the same shape: "
                    f"{tuple(predictions.shape)} != {tuple(targets.shape)}"
                )

        valid_mask = self._valid_mask(targets, mask)
        local_count = valid_mask.sum()
        if self.task_type == "multiclass":
            numerator = self._multiclass_numerator(
                predictions,
                targets,
                valid_mask,
            )
        else:
            numerator = self._elementwise_numerator(
                predictions,
                targets,
                valid_mask,
            )

        return normalized_loss(
            numerator,
            local_count,
            reference=predictions,
            distributed_sync=self.distributed_sync,
        )

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self.compute(predictions, targets, mask).loss


__all__ = ["DownstreamTaskLoss"]

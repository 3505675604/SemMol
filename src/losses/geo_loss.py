"""Masked losses for equivariant molecular coordinate denoising."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import LossComponent, connected_zero, normalized_loss


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


@dataclass(frozen=True)
class CoordinateDenoisingLossOutput:
    """Combined denoising loss and its magnitude and direction components."""

    loss: Tensor
    mse: LossComponent
    direction: LossComponent


class CoordinateDenoisingLoss(nn.Module):
    """Compare predicted and target noise/displacement vectors.

    SemMol supplies tensors shaped ``[batch, conformer, atom, 3]``. General
    ``[..., 3]`` vector batches are also accepted when ``valid_mask`` is
    supplied directly. The cosine term acts on displacement directions, not
    on absolute coordinates relative to an arbitrary coordinate origin.
    """

    def __init__(
        self,
        *,
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        distributed_sync: bool = True,
        eps: float = 1.0e-8,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        self.mse_weight = _nonnegative_weight("mse_weight", mse_weight)
        self.cosine_weight = _nonnegative_weight(
            "cosine_weight", cosine_weight
        )
        if not isinstance(distributed_sync, bool):
            raise TypeError("distributed_sync must be bool")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")
        self.distributed_sync = distributed_sync
        self.eps = _positive_finite("eps", eps)
        self.validate_values = validate_values

    @staticmethod
    def _explicit_mask(predicted: Tensor, valid_mask: Tensor) -> Tensor:
        if not isinstance(valid_mask, Tensor):
            raise TypeError("valid_mask must be a tensor")
        if valid_mask.shape != predicted.shape[:-1]:
            raise ValueError(
                f"valid_mask must have shape {tuple(predicted.shape[:-1])}, "
                f"got {tuple(valid_mask.shape)}"
            )
        if valid_mask.dtype != torch.bool:
            raise TypeError(f"valid_mask must be bool, got {valid_mask.dtype}")
        if valid_mask.device != predicted.device:
            raise ValueError("valid_mask and predicted must be on the same device")
        return valid_mask

    @staticmethod
    def _molecular_mask(
        predicted: Tensor,
        atom_mask: Tensor,
        conformer_mask: Tensor,
    ) -> Tensor:
        if predicted.ndim != 4:
            raise ValueError(
                "atom_mask and conformer_mask require predicted shape "
                f"[batch, conformer, atom, 3], got {tuple(predicted.shape)}"
            )
        if not isinstance(atom_mask, Tensor):
            raise TypeError("atom_mask must be a tensor")
        if not isinstance(conformer_mask, Tensor):
            raise TypeError("conformer_mask must be a tensor")
        batch_size, conformer_count, atom_count, _ = predicted.shape
        if atom_mask.shape != (batch_size, atom_count):
            raise ValueError(
                f"atom_mask must have shape {(batch_size, atom_count)}, "
                f"got {tuple(atom_mask.shape)}"
            )
        if conformer_mask.shape != (batch_size, conformer_count):
            raise ValueError(
                "conformer_mask must have shape "
                f"{(batch_size, conformer_count)}, "
                f"got {tuple(conformer_mask.shape)}"
            )
        if atom_mask.dtype != torch.bool:
            raise TypeError(f"atom_mask must be bool, got {atom_mask.dtype}")
        if conformer_mask.dtype != torch.bool:
            raise TypeError(
                f"conformer_mask must be bool, got {conformer_mask.dtype}"
            )
        if atom_mask.device != predicted.device:
            raise ValueError("atom_mask and predicted must be on the same device")
        if conformer_mask.device != predicted.device:
            raise ValueError(
                "conformer_mask and predicted must be on the same device"
            )
        return conformer_mask.unsqueeze(-1) & atom_mask.unsqueeze(1)

    def _resolve_mask(
        self,
        predicted: Tensor,
        *,
        valid_mask: Tensor | None,
        atom_mask: Tensor | None,
        conformer_mask: Tensor | None,
    ) -> Tensor:
        if valid_mask is not None:
            if atom_mask is not None or conformer_mask is not None:
                raise ValueError(
                    "valid_mask cannot be combined with atom_mask or "
                    "conformer_mask"
                )
            return self._explicit_mask(predicted, valid_mask)
        if (atom_mask is None) != (conformer_mask is None):
            raise ValueError(
                "atom_mask and conformer_mask must be provided together"
            )
        if atom_mask is not None and conformer_mask is not None:
            return self._molecular_mask(predicted, atom_mask, conformer_mask)
        return torch.ones(
            predicted.shape[:-1],
            dtype=torch.bool,
            device=predicted.device,
        )

    def _validate_inputs(self, predicted: Tensor, target: Tensor) -> None:
        if not isinstance(predicted, Tensor):
            raise TypeError("predicted must be a tensor")
        if not isinstance(target, Tensor):
            raise TypeError("target must be a tensor")
        if predicted.ndim < 1 or predicted.shape[-1] != 3:
            raise ValueError(
                "predicted must have shape [..., 3], got "
                f"{tuple(predicted.shape)}"
            )
        if target.shape != predicted.shape:
            raise ValueError(
                "target must have the same shape as predicted: "
                f"{tuple(target.shape)} != {tuple(predicted.shape)}"
            )
        if not predicted.is_floating_point():
            raise TypeError(f"predicted must be floating point, got {predicted.dtype}")
        if not target.is_floating_point():
            raise TypeError(f"target must be floating point, got {target.dtype}")
        if predicted.dtype != target.dtype:
            raise TypeError("predicted and target must have the same dtype")
        if predicted.device != target.device:
            raise ValueError("predicted and target must be on the same device")
        if self.validate_values:
            if predicted.numel() > 0 and not bool(torch.isfinite(predicted).all()):
                raise ValueError("predicted contains NaN or infinite values")
            if target.numel() > 0 and not bool(torch.isfinite(target).all()):
                raise ValueError("target contains NaN or infinite values")

    def compute(
        self,
        predicted: Tensor,
        target: Tensor,
        *,
        valid_mask: Tensor | None = None,
        atom_mask: Tensor | None = None,
        conformer_mask: Tensor | None = None,
    ) -> CoordinateDenoisingLossOutput:
        """Return the weighted scalar and independently normalized components."""

        self._validate_inputs(predicted, target)
        mask = self._resolve_mask(
            predicted,
            valid_mask=valid_mask,
            atom_mask=atom_mask,
            conformer_mask=conformer_mask,
        )
        reference = connected_zero(predicted) + connected_zero(target)
        predicted_vectors = predicted[mask]
        target_vectors = target[mask]
        calculation_dtype = (
            torch.float32
            if predicted.dtype in (torch.float16, torch.bfloat16)
            else predicted.dtype
        )
        predicted_for_loss = predicted_vectors.to(dtype=calculation_dtype)
        target_for_loss = target_vectors.to(dtype=calculation_dtype)

        mse_numerator = (predicted_for_loss - target_for_loss).square().sum()
        mse_count = mask.sum(dtype=torch.long) * predicted.shape[-1]
        mse = normalized_loss(
            mse_numerator,
            mse_count,
            reference=reference,
            distributed_sync=self.distributed_sync,
        )

        nonzero_target = target_for_loss.norm(p=2.0, dim=-1) > self.eps
        cosine_distance = 1.0 - F.cosine_similarity(
            predicted_for_loss,
            target_for_loss,
            dim=-1,
            eps=self.eps,
        )
        direction_numerator = cosine_distance[nonzero_target].sum()
        direction = normalized_loss(
            direction_numerator,
            nonzero_target.sum(dtype=torch.long),
            reference=reference,
            distributed_sync=self.distributed_sync,
        )
        loss = (
            self.mse_weight * mse.loss
            + self.cosine_weight * direction.loss
        )
        return CoordinateDenoisingLossOutput(
            loss=loss,
            mse=mse,
            direction=direction,
        )

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        *,
        valid_mask: Tensor | None = None,
        atom_mask: Tensor | None = None,
        conformer_mask: Tensor | None = None,
    ) -> Tensor:
        return self.compute(
            predicted,
            target,
            valid_mask=valid_mask,
            atom_mask=atom_mask,
            conformer_mask=conformer_mask,
        ).loss


__all__ = ["CoordinateDenoisingLoss", "CoordinateDenoisingLossOutput"]

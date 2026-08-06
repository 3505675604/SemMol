"""Shared loss reduction primitives with DDP-correct normalization."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor
from torch import distributed as dist


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class LossComponent:
    """A local loss numerator and the counts used to normalize it."""

    loss: Tensor
    numerator: Tensor
    local_count: Tensor
    global_count: Tensor


def connected_zero(reference: Tensor) -> Tensor:
    """Return a scalar zero connected to ``reference`` without reading values."""

    if not isinstance(reference, Tensor):
        raise TypeError("reference must be a torch.Tensor")
    if not reference.is_floating_point():
        raise TypeError(
            f"reference must be floating point, got {reference.dtype}"
        )
    return reference.reshape(-1)[:0].sum()


def _count_tensor(local_count: int | Tensor, reference: Tensor) -> Tensor:
    if isinstance(local_count, Tensor):
        if local_count.ndim != 0:
            raise ValueError(
                "local_count tensor must be scalar, got "
                f"shape {tuple(local_count.shape)}"
            )
        if local_count.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                "local_count tensor must have an integer dtype, got "
                f"{local_count.dtype}"
            )
        if local_count.device != reference.device:
            raise ValueError(
                "local_count and reference must be on the same device: "
                f"{local_count.device} != {reference.device}"
            )
        if local_count.requires_grad:
            raise ValueError("local_count must not require gradients")
        count = local_count.detach().to(dtype=torch.long).clone()
    else:
        if not isinstance(local_count, Integral) or isinstance(local_count, bool):
            raise TypeError("local_count must be an integer or scalar tensor")
        count = torch.tensor(
            int(local_count),
            dtype=torch.long,
            device=reference.device,
        )

    if bool(count < 0):
        raise ValueError("local_count cannot be negative")
    return count


def normalized_loss(
    numerator: Tensor,
    local_count: int | Tensor,
    *,
    reference: Tensor,
    distributed_sync: bool = False,
) -> LossComponent:
    """Normalize a local sum so DDP gradients equal a global element mean.

    Only the detached valid-element count is synchronized. When DDP averages
    gradients across ``world_size`` ranks, multiplying each rank's local
    numerator by ``world_size / global_count`` produces the gradient of the
    global valid-element mean without all-reducing an autograd tensor.
    """

    if not isinstance(numerator, Tensor):
        raise TypeError("numerator must be a torch.Tensor")
    if not isinstance(reference, Tensor):
        raise TypeError("reference must be a torch.Tensor")
    if numerator.ndim != 0:
        raise ValueError(
            f"numerator must be scalar, got shape {tuple(numerator.shape)}"
        )
    if not numerator.is_floating_point():
        raise TypeError(
            f"numerator must be floating point, got {numerator.dtype}"
        )
    if not reference.is_floating_point():
        raise TypeError(
            f"reference must be floating point, got {reference.dtype}"
        )
    if numerator.device != reference.device:
        raise ValueError(
            "numerator and reference must be on the same device: "
            f"{numerator.device} != {reference.device}"
        )
    if not isinstance(distributed_sync, bool):
        raise TypeError("distributed_sync must be bool")

    normalized_local_count = _count_tensor(local_count, reference)
    global_count = normalized_local_count.clone()
    world_size = 1
    if (
        distributed_sync
        and dist.is_available()
        and dist.is_initialized()
    ):
        world_size = dist.get_world_size()
        if world_size > 1:
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)

    calculation_dtype = (
        torch.float32
        if numerator.dtype in (torch.float16, torch.bfloat16)
        else numerator.dtype
    )
    calculation_numerator = numerator.to(dtype=calculation_dtype)
    denominator = global_count.clamp_min(1).to(
        dtype=calculation_dtype
    )
    scaled_loss = (
        calculation_numerator / denominator
    ) * float(world_size)
    zero = connected_zero(reference).to(dtype=calculation_dtype)
    loss = torch.where(global_count > 0, scaled_loss, zero)

    return LossComponent(
        loss=loss,
        numerator=calculation_numerator,
        local_count=normalized_local_count,
        global_count=global_count,
    )


__all__ = ["LossComponent", "connected_zero", "normalized_loss"]

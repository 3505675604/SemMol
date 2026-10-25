"""Masked language-modeling loss for compact or full-batch labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import LossComponent, connected_zero, normalized_loss


class MaskedLanguageModelingLoss(nn.Module):
    """Cross-entropy over masked SMILES token positions only."""

    def __init__(
        self,
        ignore_index: int = -100,
        distributed_sync: bool = True,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
            raise TypeError("ignore_index must be an integer")
        if not isinstance(distributed_sync, bool):
            raise TypeError("distributed_sync must be bool")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")

        self.ignore_index = ignore_index
        self.distributed_sync = distributed_sync
        self.validate_values = validate_values

    @staticmethod
    def _validate_sample_index(
        sample_index: Tensor,
        *,
        compact_size: int,
        device: torch.device,
    ) -> None:
        if not isinstance(sample_index, Tensor):
            raise TypeError("sample_index must be a torch.Tensor")
        if sample_index.ndim != 1:
            raise ValueError(
                "sample_index must have shape [compact_batch], got "
                f"{tuple(sample_index.shape)}"
            )
        if sample_index.dtype != torch.long:
            raise TypeError(
                f"sample_index must be torch.long, got {sample_index.dtype}"
            )
        if sample_index.device != device:
            raise ValueError(
                "sample_index and logits must be on the same device: "
                f"{sample_index.device} != {device}"
            )
        if sample_index.numel() != compact_size:
            raise ValueError(
                "sample_index length must match logits compact batch: "
                f"{sample_index.numel()} != {compact_size}"
            )
        if sample_index.numel() > 0:
            if bool(torch.any(sample_index < 0)):
                raise ValueError("sample_index cannot contain negative indices")
            if sample_index.numel() > 1 and not bool(
                torch.all(sample_index[1:] > sample_index[:-1])
            ):
                raise ValueError(
                    "sample_index must be strictly increasing and unique"
                )

    def _select_targets(
        self,
        logits: Tensor,
        target_ids: Tensor,
        sample_index: Tensor | None,
    ) -> Tensor:
        compact_size, sequence_length, _ = logits.shape
        if target_ids.shape[1] != sequence_length:
            raise ValueError(
                "target_ids sequence length must match logits: "
                f"{target_ids.shape[1]} != {sequence_length}"
            )

        if sample_index is None:
            if target_ids.shape[0] != compact_size:
                raise ValueError(
                    "target_ids batch must match logits when sample_index is "
                    f"omitted: {target_ids.shape[0]} != {compact_size}"
                )
            return target_ids

        self._validate_sample_index(
            sample_index,
            compact_size=compact_size,
            device=logits.device,
        )
        if target_ids.shape[0] == compact_size:
            return target_ids
        if sample_index.numel() > 0 and bool(
            torch.any(sample_index >= target_ids.shape[0])
        ):
            raise IndexError(
                "sample_index contains an index outside target_ids batch "
                f"size {target_ids.shape[0]}"
            )
        return target_ids.index_select(0, sample_index)

    def compute(
        self,
        logits: Tensor,
        target_ids: Tensor,
        sample_index: Tensor | None = None,
    ) -> LossComponent:
        if not isinstance(logits, Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if not isinstance(target_ids, Tensor):
            raise TypeError("target_ids must be a torch.Tensor")
        if logits.ndim != 3:
            raise ValueError(
                "logits must have shape [compact_batch, sequence, vocab], got "
                f"{tuple(logits.shape)}"
            )
        if target_ids.ndim != 2:
            raise ValueError(
                "target_ids must have shape [batch, sequence], got "
                f"{tuple(target_ids.shape)}"
            )
        if not logits.is_floating_point():
            raise TypeError(f"logits must be floating point, got {logits.dtype}")
        if target_ids.dtype != torch.long:
            raise TypeError(
                f"target_ids must be torch.long, got {target_ids.dtype}"
            )
        if target_ids.device != logits.device:
            raise ValueError(
                "target_ids and logits must be on the same device: "
                f"{target_ids.device} != {logits.device}"
            )
        if logits.shape[2] <= 0:
            raise ValueError("logits vocabulary dimension must be positive")
        if (
            self.validate_values
            and logits.numel() > 0
            and not bool(torch.isfinite(logits).all())
        ):
            raise ValueError("logits contain NaN or infinite values")

        selected_targets = self._select_targets(
            logits,
            target_ids,
            sample_index,
        )
        vocabulary_size = int(logits.shape[2])
        flat_logits = logits.reshape(-1, vocabulary_size)
        flat_targets = selected_targets.reshape(-1)
        valid_mask = flat_targets != self.ignore_index
        local_count = valid_mask.sum()
        valid_targets = flat_targets[valid_mask]
        if valid_targets.numel() == 0:
            numerator = connected_zero(logits)
        else:
            if self.validate_values and (
                bool(torch.any(valid_targets < 0))
                or bool(torch.any(valid_targets >= vocabulary_size))
            ):
                raise ValueError(
                    "target_ids contain a valid-position token outside "
                    f"[0, {vocabulary_size})"
                )
            valid_logits = flat_logits[valid_mask]
            calculation_logits = (
                valid_logits.float()
                if valid_logits.dtype in (torch.float16, torch.bfloat16)
                else valid_logits
            )
            numerator = F.cross_entropy(
                calculation_logits,
                valid_targets,
                reduction="sum",
            )

        return normalized_loss(
            numerator,
            local_count,
            reference=logits,
            distributed_sync=self.distributed_sync,
        )

    def forward(
        self,
        logits: Tensor,
        target_ids: Tensor,
        sample_index: Tensor | None = None,
    ) -> Tensor:
        return self.compute(logits, target_ids, sample_index).loss


__all__ = ["MaskedLanguageModelingLoss"]

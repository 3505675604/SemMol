"""Strict evaluation metrics and distributed prediction gathering.

The functions in this module keep missing labels explicit: a NaN target is a
missing value, while an infinity is always an input error.  Classification
scores are deliberately consumed as supplied, so callers may supply raw logits
from :class:`~src.models.heads.property_predictor.PropertyPredictor` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final, Sequence, TypeAlias

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from torch import Tensor


ArrayLike: TypeAlias = np.ndarray | Tensor

_TORCH_DTYPE_CODES: Final[dict[torch.dtype, int]] = {
    torch.bool: 0,
    torch.uint8: 1,
    torch.int8: 2,
    torch.int16: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.float16: 6,
    torch.bfloat16: 7,
    torch.float32: 8,
    torch.float64: 9,
}

_NUMPY_TORCH_DTYPES: Final[frozenset[torch.dtype]] = frozenset(
    {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Per-task and macro ROC-AUC values for a multitask classification run."""

    macro_roc_auc: float
    task_names: tuple[str, ...]
    per_task_roc_auc: tuple[float, ...]
    valid_sample_counts: tuple[int, ...]
    positive_sample_counts: tuple[int, ...]
    negative_sample_counts: tuple[int, ...]
    eligible: tuple[bool, ...]
    eligible_task_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation of the metrics."""

        return {
            "macro_roc_auc": self.macro_roc_auc,
            "eligible_task_count": self.eligible_task_count,
            "per_task": {
                name: {
                    "roc_auc": self.per_task_roc_auc[index],
                    "valid_sample_count": self.valid_sample_counts[index],
                    "positive_sample_count": self.positive_sample_counts[index],
                    "negative_sample_count": self.negative_sample_counts[index],
                    "eligible": self.eligible[index],
                }
                for index, name in enumerate(self.task_names)
            },
        }


@dataclass(frozen=True)
class RegressionMetrics:
    """Per-task and macro RMSE, MAE, and R2 for multitask regression."""

    macro_rmse: float
    macro_mae: float
    macro_r2: float
    task_names: tuple[str, ...]
    per_task_rmse: tuple[float, ...]
    per_task_mae: tuple[float, ...]
    per_task_r2: tuple[float, ...]
    valid_sample_counts: tuple[int, ...]
    eligible: tuple[bool, ...]
    r2_eligible: tuple[bool, ...]
    eligible_task_count: int
    r2_eligible_task_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation of the metrics."""

        return {
            "macro_rmse": self.macro_rmse,
            "macro_mae": self.macro_mae,
            "macro_r2": self.macro_r2,
            "eligible_task_count": self.eligible_task_count,
            "r2_eligible_task_count": self.r2_eligible_task_count,
            "per_task": {
                name: {
                    "rmse": self.per_task_rmse[index],
                    "mae": self.per_task_mae[index],
                    "r2": self.per_task_r2[index],
                    "valid_sample_count": self.valid_sample_counts[index],
                    "eligible": self.eligible[index],
                    "r2_eligible": self.r2_eligible[index],
                }
                for index, name in enumerate(self.task_names)
            },
        }


@dataclass(frozen=True)
class IndexedPredictions:
    """CPU prediction tensors, sorted and de-duplicated by ``source_index``."""

    source_index: Tensor
    predictions: Tensor
    targets: Tensor
    mask: Tensor

    @property
    def source_indices(self) -> Tensor:
        """Plural alias for ``source_index``."""

        return self.source_index


def _as_numpy(value: ArrayLike, *, name: str) -> np.ndarray:
    if isinstance(value, Tensor):
        if value.layout != torch.strided:
            raise TypeError(f"{name} must use the torch.strided layout")
        if value.is_quantized:
            raise TypeError(f"{name} must not be a quantized tensor")
        if value.dtype not in _NUMPY_TORCH_DTYPES:
            raise TypeError(f"{name} has unsupported torch dtype {value.dtype}")
        detached = value.detach().resolve_conj().resolve_neg().cpu()
        if detached.dtype == torch.bfloat16:
            detached = detached.to(dtype=torch.float32)
        return detached.numpy()
    if isinstance(value, np.ndarray):
        return value
    raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor")


def _require_real_array(array: np.ndarray, *, name: str) -> None:
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError(f"{name} must have a real boolean, integer, or floating dtype")


def _normalize_matrix(
    predictions: ArrayLike,
    targets: ArrayLike,
    mask: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = _as_numpy(predictions, name="predictions")
    labels = _as_numpy(targets, name="targets")
    _require_real_array(scores, name="predictions")
    _require_real_array(labels, name="targets")
    if scores.ndim not in {1, 2}:
        raise ValueError(
            f"predictions must have shape [N] or [N, T], got {tuple(scores.shape)}"
        )
    if labels.shape != scores.shape:
        raise ValueError("predictions and targets must have exactly the same shape")

    if mask is None:
        normalized_mask = ~np.isnan(labels) if labels.dtype.kind == "f" else np.ones_like(
            labels, dtype=np.bool_
        )
    else:
        raw_mask = _as_numpy(mask, name="mask")
        if raw_mask.shape != scores.shape:
            raise ValueError("mask must have exactly the same shape as predictions")
        normalized_mask = _normalize_mask(raw_mask)

    if labels.dtype.kind == "f":
        if np.any(np.isinf(labels)):
            raise ValueError("targets must not contain infinite values")
        nan_targets = np.isnan(labels)
    else:
        nan_targets = np.zeros(labels.shape, dtype=np.bool_)
    if np.any(normalized_mask & nan_targets):
        raise ValueError("mask must not mark NaN targets as valid")

    normalized_scores = _as_2d(scores, name="predictions")
    normalized_labels = _as_2d(labels, name="targets")
    normalized_mask = _as_2d(normalized_mask, name="mask")
    valid = normalized_mask & ~_as_2d(nan_targets, name="targets")
    if np.any(valid) and not np.all(np.isfinite(normalized_scores[valid])):
        raise ValueError("predictions contain NaN or infinite values at valid positions")
    return normalized_scores, normalized_labels, valid


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype.kind == "b":
        return mask.astype(np.bool_, copy=False)
    if mask.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("mask must have a boolean dtype or contain only 0 and 1")
    if mask.dtype.kind == "f" and not np.all(np.isfinite(mask)):
        raise ValueError("mask must not contain NaN or infinite values")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only 0 and 1")
    return mask.astype(np.bool_)


def _as_2d(array: np.ndarray, *, name: str) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"{name} must have shape [N] or [N, T]")


def _task_names(task_names: Sequence[str] | None, task_count: int) -> tuple[str, ...]:
    if task_names is None:
        return tuple(f"task_{index}" for index in range(task_count))
    if isinstance(task_names, (str, bytes)):
        raise TypeError("task_names must be a sequence of unique non-empty strings")
    if not isinstance(task_names, Sequence):
        raise TypeError("task_names must be a sequence of unique non-empty strings")
    names = tuple(task_names)
    if len(names) != task_count:
        raise ValueError(
            f"task_names must contain {task_count} names, got {len(names)}"
        )
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("task_names must contain only non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("task_names must be unique")
    return names


def _macro(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate_classification(
    targets: ArrayLike,
    predictions: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    task_names: Sequence[str] | None = None,
) -> ClassificationMetrics:
    """Evaluate classification predictions against binary ground-truth targets.

    ``targets`` (``y_true``) contains binary labels or NaN missing values.
    ``predictions`` (``y_score``) may be probabilities or raw logits.  A task
    is ROC-AUC eligible only when its valid labels include both classes.
    """

    scores, labels, valid = _normalize_matrix(predictions, targets, mask)
    names = _task_names(task_names, scores.shape[1])

    aucs: list[float] = []
    valid_counts: list[int] = []
    positive_counts: list[int] = []
    negative_counts: list[int] = []
    eligible: list[bool] = []
    macro_values: list[float] = []
    for task_index in range(scores.shape[1]):
        task_valid = valid[:, task_index]
        task_labels = labels[task_valid, task_index]
        if task_labels.size and not np.all((task_labels == 0) | (task_labels == 1)):
            raise ValueError(
                f"targets for task {names[task_index]!r} must be strictly 0 or 1 "
                "at valid positions"
            )
        positive_count = int(np.count_nonzero(task_labels == 1))
        negative_count = int(np.count_nonzero(task_labels == 0))
        is_eligible = positive_count > 0 and negative_count > 0
        auc = float("nan")
        if is_eligible:
            auc = float(roc_auc_score(task_labels, scores[task_valid, task_index]))
            macro_values.append(auc)
        aucs.append(auc)
        valid_counts.append(int(task_labels.size))
        positive_counts.append(positive_count)
        negative_counts.append(negative_count)
        eligible.append(is_eligible)

    return ClassificationMetrics(
        macro_roc_auc=_macro(macro_values),
        task_names=names,
        per_task_roc_auc=tuple(aucs),
        valid_sample_counts=tuple(valid_counts),
        positive_sample_counts=tuple(positive_counts),
        negative_sample_counts=tuple(negative_counts),
        eligible=tuple(eligible),
        eligible_task_count=len(macro_values),
    )


def evaluate_regression(
    targets: ArrayLike,
    predictions: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    task_names: Sequence[str] | None = None,
) -> RegressionMetrics:
    """Evaluate regression predictions against continuous ground-truth targets.

    ``targets`` (``y_true``) contains values or NaN missing values, and
    ``predictions`` (``y_score``) contains corresponding model outputs.
    """

    predictions_2d, targets_2d, valid = _normalize_matrix(predictions, targets, mask)
    names = _task_names(task_names, predictions_2d.shape[1])

    rmses: list[float] = []
    maes: list[float] = []
    r2s: list[float] = []
    valid_counts: list[int] = []
    eligible: list[bool] = []
    r2_eligible: list[bool] = []
    rmse_values: list[float] = []
    mae_values: list[float] = []
    r2_values: list[float] = []
    for task_index in range(predictions_2d.shape[1]):
        task_valid = valid[:, task_index]
        task_predictions = np.asarray(
            predictions_2d[task_valid, task_index], dtype=np.float64
        )
        task_targets = np.asarray(
            targets_2d[task_valid, task_index], dtype=np.float64
        )
        count = int(task_targets.size)
        rmse = float("nan")
        mae = float("nan")
        r2 = float("nan")
        if count:
            if not np.all(np.isfinite(task_predictions)) or not np.all(
                np.isfinite(task_targets)
            ):
                raise ValueError("valid targets and predictions must fit finite float64")
            with np.errstate(over="ignore", invalid="ignore"):
                errors = task_predictions - task_targets
                rmse = float(np.sqrt(np.mean(np.square(errors))))
                mae = float(np.mean(np.abs(errors)))
            rmse_values.append(rmse)
            mae_values.append(mae)
        can_score_r2 = False
        if count >= 2:
            with np.errstate(over="ignore", invalid="ignore"):
                centered_targets = task_targets - np.mean(task_targets, dtype=np.float64)
                total_sum = float(np.sum(np.square(centered_targets), dtype=np.float64))
            can_score_r2 = bool(np.isfinite(total_sum) and total_sum > 0.0)
        if can_score_r2:
            with np.errstate(over="ignore", invalid="ignore"):
                residual_sum = float(
                    np.sum(np.square(task_predictions - task_targets), dtype=np.float64)
                )
            r2 = float(1.0 - residual_sum / total_sum)
            r2_values.append(r2)
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        valid_counts.append(count)
        eligible.append(count > 0)
        r2_eligible.append(can_score_r2)

    return RegressionMetrics(
        macro_rmse=_macro(rmse_values),
        macro_mae=_macro(mae_values),
        macro_r2=_macro(r2_values),
        task_names=names,
        per_task_rmse=tuple(rmses),
        per_task_mae=tuple(maes),
        per_task_r2=tuple(r2s),
        valid_sample_counts=tuple(valid_counts),
        eligible=tuple(eligible),
        r2_eligible=tuple(r2_eligible),
        eligible_task_count=len(rmse_values),
        r2_eligible_task_count=len(r2_values),
    )


def _validate_gather_inputs(
    source_index: Tensor,
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> None:
    for name, value in (
        ("source_index", source_index),
        ("predictions", predictions),
        ("targets", targets),
        ("mask", mask),
    ):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if source_index.ndim != 1:
        raise ValueError("source_index must have shape [N]")
    if source_index.dtype != torch.int64:
        raise TypeError("source_index must have dtype torch.int64")
    if predictions.ndim not in {1, 2}:
        raise ValueError("predictions must have shape [N] or [N, T]")
    if targets.shape != predictions.shape or mask.shape != predictions.shape:
        raise ValueError("predictions, targets, and mask must have exactly the same shape")
    if source_index.shape[0] != predictions.shape[0]:
        raise ValueError("source_index and predictions must have the same leading length")
    if not predictions.is_floating_point():
        raise TypeError("predictions must have a floating-point dtype")
    if predictions.dtype not in _TORCH_DTYPE_CODES:
        raise TypeError(f"predictions has unsupported dtype {predictions.dtype}")
    if targets.dtype not in _TORCH_DTYPE_CODES or targets.is_complex():
        raise TypeError("targets must have a real boolean, integer, or floating dtype")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    device = predictions.device
    if source_index.device != device or targets.device != device or mask.device != device:
        raise ValueError("source_index, predictions, targets, and mask must be on one device")
    if source_index.numel() and bool(torch.any(source_index < 0)):
        raise ValueError("source_index must contain only non-negative values")


def _distributed_metadata(
    local_length: int,
    predictions: Tensor,
    targets: Tensor,
    dst: int,
    valid_dst: bool,
) -> tuple[list[int], int]:
    """Gather and validate small fixed-width metadata without Python objects."""

    world_size = dist.get_world_size()
    width = 1 if predictions.ndim == 1 else int(predictions.shape[1])
    local = torch.tensor(
        [
            local_length,
            predictions.ndim,
            width,
            _TORCH_DTYPE_CODES[predictions.dtype],
            _TORCH_DTYPE_CODES[targets.dtype],
            dst,
            int(valid_dst),
        ],
        device=predictions.device,
        dtype=torch.int64,
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    values = [item.cpu().tolist() for item in gathered]
    reference = values[0][1:5]
    if any(item[1:5] != reference for item in values[1:]):
        raise ValueError("DDP ranks must use matching prediction shape and dtypes")
    gathered_dst = {int(item[5]) for item in values}
    if len(gathered_dst) != 1:
        raise ValueError("DDP ranks must use the same dst rank")
    if any(int(item[6]) != 1 for item in values):
        raise TypeError("DDP dst must be an integer rank representable as int64")
    resolved_dst = gathered_dst.pop()
    if not 0 <= resolved_dst < world_size:
        raise ValueError(
            f"DDP dst must be in [0, {world_size}), got {resolved_dst}"
        )
    return [int(item[0]) for item in values], resolved_dst


def _all_gather_padded(tensor: Tensor, lengths: Sequence[int], max_length: int) -> list[Tensor]:
    padded = tensor.new_zeros((max_length, *tensor.shape[1:]))
    if tensor.shape[0]:
        padded[: tensor.shape[0]] = tensor
    gathered = [torch.empty_like(padded) for _ in lengths]
    dist.all_gather(gathered, padded)
    return [item[:length] for item, length in zip(gathered, lengths)]


def _targets_equal(left: Tensor, right: Tensor) -> bool:
    if torch.equal(left, right):
        return True
    if not left.is_floating_point():
        return False
    if not torch.equal(torch.isnan(left), torch.isnan(right)):
        return False
    if not torch.equal(torch.isposinf(left), torch.isposinf(right)):
        return False
    if not torch.equal(torch.isneginf(left), torch.isneginf(right)):
        return False
    return bool(torch.equal(torch.nan_to_num(left), torch.nan_to_num(right)))


def _deduplicate_sorted(
    source_index: Tensor,
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> IndexedPredictions:
    order = torch.argsort(source_index, stable=True)
    source_index = source_index[order]
    predictions = predictions[order]
    targets = targets[order]
    mask = mask[order]
    keep_rows: list[int] = []
    previous = -1
    for row in range(source_index.shape[0]):
        current = int(source_index[row].item())
        if current != previous:
            keep_rows.append(row)
            previous = current
            continue
        kept = keep_rows[-1]
        if not _targets_equal(targets[row], targets[kept]) or not torch.equal(
            mask[row], mask[kept]
        ):
            raise ValueError(
                f"duplicate source_index={current} has inconsistent targets or mask"
            )
        if not bool(torch.allclose(predictions[row], predictions[kept], equal_nan=True)):
            raise ValueError(f"duplicate source_index={current} has inconsistent predictions")
    kept = torch.tensor(keep_rows, device=source_index.device, dtype=torch.long)
    return IndexedPredictions(
        source_index=source_index[kept].detach().cpu(),
        predictions=predictions[kept].detach().cpu(),
        targets=targets[kept].detach().cpu(),
        mask=mask[kept].detach().cpu(),
    )


def gather_indexed_predictions(
    source_index: Tensor,
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    *,
    dst: int = 0,
) -> IndexedPredictions | None:
    """Collect DDP evaluation outputs on ``dst``, sorted by source index.

    Each rank may contribute zero rows.  DDP uses length gathering, fixed-size
    padding, and tensor-only ``all_gather`` calls; duplicated sampler padding
    is removed after checking that every duplicated record agrees.
    """

    _validate_gather_inputs(source_index, predictions, targets, mask)
    valid_dst = isinstance(dst, Integral) and not isinstance(dst, bool)
    normalized_dst = int(dst) if valid_dst else -1
    if normalized_dst < -(2**63) or normalized_dst > 2**63 - 1:
        valid_dst = False
        normalized_dst = -1
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    if not distributed:
        if not valid_dst:
            raise TypeError("dst must be an integer rank representable as int64")
        if not 0 <= normalized_dst < world_size:
            raise ValueError(f"dst must be in [0, {world_size}), got {dst}")
        return _deduplicate_sorted(source_index, predictions, targets, mask)

    lengths, resolved_dst = _distributed_metadata(
        source_index.shape[0], predictions, targets, normalized_dst, valid_dst
    )
    max_length = max(lengths)
    if max_length == 0:
        if rank != resolved_dst:
            return None
        return _deduplicate_sorted(source_index, predictions, targets, mask)
    gathered_source = _all_gather_padded(source_index, lengths, max_length)
    if predictions.ndim == 2 and predictions.shape[1] == 0:
        if rank != resolved_dst:
            return None
        all_source_index = torch.cat(gathered_source, dim=0)
        empty_shape = (all_source_index.shape[0], 0)
        return _deduplicate_sorted(
            all_source_index,
            predictions.new_empty(empty_shape),
            targets.new_empty(empty_shape),
            mask.new_empty(empty_shape),
        )
    gathered_predictions = _all_gather_padded(predictions, lengths, max_length)
    gathered_targets = _all_gather_padded(targets, lengths, max_length)
    gathered_mask = _all_gather_padded(mask, lengths, max_length)
    if rank != resolved_dst:
        return None
    return _deduplicate_sorted(
        torch.cat(gathered_source, dim=0),
        torch.cat(gathered_predictions, dim=0),
        torch.cat(gathered_targets, dim=0),
        torch.cat(gathered_mask, dim=0),
    )


__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
    "IndexedPredictions",
    "evaluate_classification",
    "evaluate_regression",
    "gather_indexed_predictions",
]

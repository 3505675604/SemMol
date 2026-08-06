"""Strict diagnostic metrics for SemMol experiments.

The diagnostics in this module keep eligibility explicit.  A diagnostic that
cannot be estimated from the supplied observations is represented by ``NaN``
together with the corresponding count and eligibility flag.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter

from src.models.alignment.acsm import ACSMOutput
from src.models.semmol import ModalityRepresentation


ArrayLike: TypeAlias = np.ndarray | Tensor


@dataclass(frozen=True)
class GradientConflictPairResult:
    """Gradient geometry for one pair of loss terms."""

    loss_a: str
    loss_b: str
    dot: float
    cosine: float
    conflict: bool | None
    eligible: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "loss_a": self.loss_a,
            "loss_b": self.loss_b,
            "dot": self.dot,
            "cosine": self.cosine,
            "conflict": self.conflict,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class GradientConflictResult:
    """Pairwise gradient conflicts for one optimization step."""

    loss_names: tuple[str, ...]
    pairs: tuple[GradientConflictPairResult, ...]
    pair_count: int
    eligible_pair_count: int
    conflict_count: int
    conflict_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "loss_names": self.loss_names,
            "pair_count": self.pair_count,
            "eligible_pair_count": self.eligible_pair_count,
            "conflict_count": self.conflict_count,
            "conflict_rate": self.conflict_rate,
            "pairs": tuple(pair.as_dict() for pair in self.pairs),
        }


@dataclass(frozen=True)
class GradientConflictPairAccumulation:
    """Eligible and conflicting observations for a loss pair."""

    loss_a: str
    loss_b: str
    eligible_count: int
    conflict_count: int
    conflict_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "loss_a": self.loss_a,
            "loss_b": self.loss_b,
            "eligible_count": self.eligible_count,
            "conflict_count": self.conflict_count,
            "conflict_rate": self.conflict_rate,
        }


@dataclass(frozen=True)
class GradientConflictAccumulatorResult:
    """Gradient-conflict rate accumulated across optimization steps."""

    step_count: int
    pairs: tuple[GradientConflictPairAccumulation, ...]
    eligible_pair_count: int
    conflict_count: int
    conflict_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "step_count": self.step_count,
            "eligible_pair_count": self.eligible_pair_count,
            "conflict_count": self.conflict_count,
            "conflict_rate": self.conflict_rate,
            "pairs": tuple(pair.as_dict() for pair in self.pairs),
        }


class GradientConflictAccumulator:
    """Accumulate the paper's conflict rate over eligible pair-step units."""

    def __init__(self) -> None:
        self._step_count = 0
        self._pair_order: tuple[tuple[str, str], ...] | None = None
        self._eligible_counts: dict[tuple[str, str], int] = {}
        self._conflict_counts: dict[tuple[str, str], int] = {}

    def update(self, result: GradientConflictResult) -> None:
        """Add one step without treating ineligible pairs as non-conflicts."""

        if not isinstance(result, GradientConflictResult):
            raise TypeError("result must be a GradientConflictResult")
        pair_order = tuple((pair.loss_a, pair.loss_b) for pair in result.pairs)
        if self._pair_order is None:
            self._pair_order = pair_order
            self._eligible_counts = {pair: 0 for pair in pair_order}
            self._conflict_counts = {pair: 0 for pair in pair_order}
        elif pair_order != self._pair_order:
            raise ValueError(
                "every accumulated result must use the same ordered loss pairs"
            )

        for pair in result.pairs:
            key = (pair.loss_a, pair.loss_b)
            if not pair.eligible:
                continue
            if pair.conflict is None:
                raise ValueError("an eligible pair must define conflict")
            self._eligible_counts[key] += 1
            self._conflict_counts[key] += int(pair.conflict)
        self._step_count += 1

    def compute(self) -> GradientConflictAccumulatorResult:
        """Return an immutable snapshot of the accumulated statistics."""

        pair_order = self._pair_order or ()
        pair_results: list[GradientConflictPairAccumulation] = []
        total_eligible = 0
        total_conflicts = 0
        for loss_a, loss_b in pair_order:
            key = (loss_a, loss_b)
            eligible_count = self._eligible_counts[key]
            conflict_count = self._conflict_counts[key]
            rate = (
                float(conflict_count / eligible_count)
                if eligible_count
                else float("nan")
            )
            pair_results.append(
                GradientConflictPairAccumulation(
                    loss_a=loss_a,
                    loss_b=loss_b,
                    eligible_count=eligible_count,
                    conflict_count=conflict_count,
                    conflict_rate=rate,
                )
            )
            total_eligible += eligible_count
            total_conflicts += conflict_count
        conflict_rate = (
            float(total_conflicts / total_eligible)
            if total_eligible
            else float("nan")
        )
        return GradientConflictAccumulatorResult(
            step_count=self._step_count,
            pairs=tuple(pair_results),
            eligible_pair_count=total_eligible,
            conflict_count=total_conflicts,
            conflict_rate=conflict_rate,
        )

    def as_dict(self) -> dict[str, object]:
        return self.compute().as_dict()

    def reset(self) -> None:
        self._step_count = 0
        self._pair_order = None
        self._eligible_counts = {}
        self._conflict_counts = {}


@dataclass(frozen=True)
class ModalConsistencyPairResult:
    """Mean cosine consistency for one aligned modality pair."""

    modality_a: str
    modality_b: str
    mean_cosine_similarity: float
    sample_count: int
    eligible: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "modality_a": self.modality_a,
            "modality_b": self.modality_b,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "sample_count": self.sample_count,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class ModalConsistencyResult:
    """Pairwise and macro cross-modal representation consistency."""

    modalities: tuple[str, ...]
    cohort: str
    pairs: tuple[ModalConsistencyPairResult, ...]
    macro_consistency: float
    eligible_pair_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "modalities": self.modalities,
            "cohort": self.cohort,
            "macro_consistency": self.macro_consistency,
            "eligible_pair_count": self.eligible_pair_count,
            "pairs": tuple(pair.as_dict() for pair in self.pairs),
        }


@dataclass(frozen=True)
class NoiseVarianceRatioResult:
    """Per-dimension and macro center-to-feature variance ratios."""

    dimension_ratios: tuple[float, ...]
    eligible: tuple[bool, ...]
    mean_ratio: float
    effective_noise: float
    dimension_count: int
    eligible_dimension_count: int
    noise_rate: float
    ddof: int

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension_ratios": self.dimension_ratios,
            "eligible": self.eligible,
            "mean_ratio": self.mean_ratio,
            "effective_noise": self.effective_noise,
            "dimension_count": self.dimension_count,
            "eligible_dimension_count": self.eligible_dimension_count,
            "noise_rate": self.noise_rate,
            "ddof": self.ddof,
        }


@dataclass(frozen=True)
class OutlierTaskError:
    """Extreme-value error diagnostics for one regression task."""

    task_name: str
    threshold: float
    r_ext: float
    conditional_tail_mse: float
    valid_count: int
    tail_count: int
    eligible: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "task_name": self.task_name,
            "threshold": self.threshold,
            "r_ext": self.r_ext,
            "conditional_tail_mse": self.conditional_tail_mse,
            "valid_count": self.valid_count,
            "tail_count": self.tail_count,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class OutlierErrorResult:
    """Per-task and equally weighted macro extreme-value errors."""

    task_names: tuple[str, ...]
    tasks: tuple[OutlierTaskError, ...]
    macro_r_ext: float
    macro_conditional_tail_mse: float
    eligible_task_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "macro_r_ext": self.macro_r_ext,
            "macro_conditional_tail_mse": self.macro_conditional_tail_mse,
            "eligible_task_count": self.eligible_task_count,
            "per_task": {
                task.task_name: task.as_dict() for task in self.tasks
            },
        }


@dataclass(frozen=True)
class ScaffoldPORGroupResult:
    """Top-k positive-over-random enrichment for one scaffold."""

    scaffold_id: Hashable
    size: int
    positive_count: int
    hits_at_k: float
    baseline_positive_rate: float
    por_at_k: float
    eligible: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "scaffold_id": self.scaffold_id,
            "size": self.size,
            "positive_count": self.positive_count,
            "hits_at_k": self.hits_at_k,
            "baseline_positive_rate": self.baseline_positive_rate,
            "por_at_k": self.por_at_k,
            "eligible": self.eligible,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScaffoldPORResult:
    """Macro and group-size-weighted POR@Scaffold summaries."""

    top_k: int
    tie_policy: str
    groups: tuple[ScaffoldPORGroupResult, ...]
    group_count: int
    eligible_group_count: int
    eligible_sample_count: int
    macro_por: float
    weighted_por: float

    def as_dict(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "tie_policy": self.tie_policy,
            "group_count": self.group_count,
            "eligible_group_count": self.eligible_group_count,
            "eligible_sample_count": self.eligible_sample_count,
            "macro_por": self.macro_por,
            "weighted_por": self.weighted_por,
            "per_group": {
                group.scaffold_id: group.as_dict() for group in self.groups
            },
        }


@dataclass(frozen=True)
class MatchingEntropyModalityResult:
    """Shannon-entropy summaries for one ACSM target modality."""

    modality: str
    center_count: int
    sample_count: int
    mean_entropy: float
    std_entropy: float
    mean_normalized_entropy: float
    std_normalized_entropy: float
    mean_effective_center_count: float
    std_effective_center_count: float

    def as_dict(self) -> dict[str, object]:
        return {
            "modality": self.modality,
            "center_count": self.center_count,
            "sample_count": self.sample_count,
            "mean_entropy": self.mean_entropy,
            "std_entropy": self.std_entropy,
            "mean_normalized_entropy": self.mean_normalized_entropy,
            "std_normalized_entropy": self.std_normalized_entropy,
            "mean_effective_center_count": self.mean_effective_center_count,
            "std_effective_center_count": self.std_effective_center_count,
        }


@dataclass(frozen=True)
class MatchingEntropyResult:
    """Per-modality Shannon entropy of ACSM positive weights."""

    modalities: tuple[str, ...]
    per_modality: tuple[MatchingEntropyModalityResult, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "modalities": self.modalities,
            "per_modality": {
                item.modality: item.as_dict() for item in self.per_modality
            },
        }


def _nonempty_unique_names(
    values: Sequence[object],
    *,
    name: str,
) -> tuple[str, ...]:
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}[{index}] must be a non-empty string")
        item = value.strip()
        if item in normalized:
            raise ValueError(f"{name} must contain unique names")
        normalized.append(item)
    return tuple(normalized)


def _finite_nonnegative_real(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _real_numpy(value: ArrayLike, *, name: str) -> np.ndarray:
    if isinstance(value, Tensor):
        if value.layout != torch.strided:
            raise TypeError(f"{name} must use the torch.strided layout")
        if value.is_quantized or value.is_complex():
            raise TypeError(f"{name} must be a real, non-quantized tensor")
        detached = value.detach().resolve_conj().resolve_neg().cpu()
        if detached.dtype == torch.bfloat16:
            detached = detached.to(dtype=torch.float32)
        try:
            array = detached.numpy()
        except TypeError as error:
            raise TypeError(f"{name} has an unsupported tensor dtype") from error
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor")
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError(f"{name} must have a real numeric dtype")
    return array


def _matrix(value: ArrayLike, *, name: str) -> np.ndarray:
    array = _real_numpy(value, name=name)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D]")
    return array


def _supervised_matrices(
    targets: ArrayLike,
    predictions: ArrayLike,
    mask: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = _real_numpy(targets, name="targets")
    scores = _real_numpy(predictions, name="predictions")
    if labels.ndim not in {1, 2}:
        raise ValueError("targets must have shape [N] or [N, T]")
    if scores.shape != labels.shape:
        raise ValueError("targets and predictions must have exactly the same shape")

    if labels.dtype.kind == "f":
        if np.any(np.isinf(labels)):
            raise ValueError("targets must not contain infinite values")
        missing = np.isnan(labels)
    else:
        missing = np.zeros(labels.shape, dtype=np.bool_)

    if mask is None:
        valid = ~missing
    else:
        raw_mask = _real_numpy(mask, name="mask")
        if raw_mask.shape != labels.shape:
            raise ValueError("mask must have exactly the same shape as targets")
        if raw_mask.dtype.kind == "f" and not np.all(np.isfinite(raw_mask)):
            raise ValueError("mask must not contain NaN or infinite values")
        if not np.all((raw_mask == 0) | (raw_mask == 1)):
            raise ValueError("mask must contain only 0 and 1")
        valid = raw_mask.astype(np.bool_, copy=False)
        if np.any(valid & missing):
            raise ValueError("mask must not mark NaN targets as valid")
        valid &= ~missing

    labels_2d = labels.reshape(-1, 1) if labels.ndim == 1 else labels
    scores_2d = scores.reshape(-1, 1) if scores.ndim == 1 else scores
    valid_2d = valid.reshape(-1, 1) if valid.ndim == 1 else valid
    if np.any(valid_2d) and not np.all(np.isfinite(scores_2d[valid_2d])):
        raise ValueError(
            "predictions contain NaN or infinite values at valid positions"
        )
    return labels_2d, scores_2d, valid_2d


def _task_name_tuple(
    task_names: Sequence[str] | None,
    task_count: int,
) -> tuple[str, ...]:
    if task_names is None:
        return tuple(f"task_{index}" for index in range(task_count))
    if isinstance(task_names, (str, bytes)) or not isinstance(
        task_names, Sequence
    ):
        raise TypeError("task_names must be a sequence")
    names = _nonempty_unique_names(tuple(task_names), name="task_names")
    if len(names) != task_count:
        raise ValueError(f"task_names must contain {task_count} names")
    return names


def compute_gradient_conflicts(
    losses: Mapping[str, Tensor],
    named_parameters: Sequence[tuple[str, Parameter]],
    distributed: bool = True,
    norm_eps: float = 1.0e-12,
) -> GradientConflictResult:
    """Compute pairwise loss-gradient conflicts without altering ``.grad``."""

    if not isinstance(losses, Mapping):
        raise TypeError("losses must be a mapping")
    raw_loss_items = tuple(losses.items())
    normalized_loss_names = _nonempty_unique_names(
        tuple(name for name, _ in raw_loss_items),
        name="losses",
    )
    if len(raw_loss_items) < 2:
        raise ValueError("losses must contain at least two loss terms")
    loss_items = tuple(
        sorted(
            (
                (name, raw_item[1])
                for name, raw_item in zip(
                    normalized_loss_names,
                    raw_loss_items,
                )
            ),
            key=lambda item: item[0],
        )
    )
    loss_names = tuple(name for name, _ in loss_items)
    if not isinstance(distributed, bool):
        raise TypeError("distributed must be bool")
    epsilon = _finite_nonnegative_real("norm_eps", norm_eps)

    normalized_parameters: list[tuple[str, Parameter]] = []
    if isinstance(named_parameters, (str, bytes)) or not isinstance(
        named_parameters, Sequence
    ):
        raise TypeError("named_parameters must be a sequence")
    parameter_names: list[object] = []
    parameter_ids: set[int] = set()
    for index, item in enumerate(named_parameters):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"named_parameters[{index}] must be a (name, Parameter) tuple"
            )
        parameter_name, parameter = item
        parameter_names.append(parameter_name)
        if not isinstance(parameter, Parameter):
            raise TypeError(
                f"named_parameters[{index}][1] must be a Parameter"
            )
        if id(parameter) in parameter_ids:
            raise ValueError("named_parameters must contain unique objects")
        parameter_ids.add(id(parameter))
        if not parameter.is_floating_point():
            raise TypeError(f"parameter {parameter_name!r} must be floating point")
        if not parameter.requires_grad:
            raise ValueError(f"parameter {parameter_name!r} must require gradients")
        normalized_parameters.append((str(parameter_name), parameter))
    normalized_parameter_names = _nonempty_unique_names(
        parameter_names,
        name="named_parameters",
    )
    if not normalized_parameters:
        raise ValueError("named_parameters must not be empty")
    normalized_parameters = sorted(
        (
            (name, item[1])
            for name, item in zip(
                normalized_parameter_names,
                normalized_parameters,
            )
        ),
        key=lambda item: item[0],
    )

    for loss_name, loss in loss_items:
        if not isinstance(loss, Tensor):
            raise TypeError(f"loss {loss_name!r} must be a torch.Tensor")
        if loss.ndim != 0:
            raise ValueError(f"loss {loss_name!r} must be a scalar tensor")
        if not loss.is_floating_point():
            raise TypeError(f"loss {loss_name!r} must be floating point")
        if not loss.requires_grad:
            raise ValueError(f"loss {loss_name!r} must require gradients")

    parameter_values = tuple(parameter for _, parameter in normalized_parameters)
    synchronize = (
        distributed
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    world_size = dist.get_world_size() if synchronize else 1
    if synchronize:
        local_signature = (
            loss_names,
            tuple(
                (
                    name,
                    tuple(int(size) for size in parameter.shape),
                    str(parameter.dtype),
                )
                for name, parameter in normalized_parameters
            ),
        )
        gathered_signatures: list[object] = [None] * world_size
        dist.all_gather_object(gathered_signatures, local_signature)
        if any(signature != local_signature for signature in gathered_signatures):
            raise ValueError(
                "DDP ranks must use identical sorted loss names and parameter "
                "names, shapes, and dtypes for gradient diagnostics"
            )

    gradients: dict[str, tuple[Tensor, ...]] = {}
    participation: dict[str, tuple[bool, ...]] = {}
    for loss_name, loss in loss_items:
        raw_gradients = torch.autograd.grad(
            loss,
            parameter_values,
            allow_unused=True,
            retain_graph=True,
        )
        aligned: list[Tensor] = []
        participating: list[bool] = []
        for parameter, raw_gradient in zip(parameter_values, raw_gradients):
            participates_locally = raw_gradient is not None
            if raw_gradient is None:
                gradient = torch.zeros_like(
                    parameter,
                    dtype=torch.float64,
                )
            else:
                gradient = raw_gradient.detach()
                if gradient.layout != torch.strided:
                    gradient = gradient.to_dense()
                gradient = gradient.to(dtype=torch.float64).clone()
            participates_globally = participates_locally
            if synchronize:
                flag = torch.tensor(
                    int(participates_locally),
                    dtype=torch.int64,
                    device=parameter.device,
                )
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                gradient.div_(world_size)
                participates_globally = bool(flag.item())
            if gradient.numel() and not bool(torch.isfinite(gradient).all()):
                raise ValueError(
                    f"loss {loss_name!r} produced non-finite gradients"
                )
            aligned.append(gradient)
            participating.append(participates_globally)
        gradients[loss_name] = tuple(aligned)
        participation[loss_name] = tuple(participating)

    pair_results: list[GradientConflictPairResult] = []
    conflict_count = 0
    eligible_count = 0
    for loss_a, loss_b in combinations(loss_names, 2):
        common_parameter = any(
            left and right
            for left, right in zip(
                participation[loss_a],
                participation[loss_b],
            )
        )
        dot = 0.0
        norm_a_squared = 0.0
        norm_b_squared = 0.0
        for gradient_a, gradient_b in zip(
            gradients[loss_a],
            gradients[loss_b],
        ):
            dot += float(torch.sum(gradient_a * gradient_b).item())
            norm_a_squared += float(
                torch.sum(gradient_a * gradient_a).item()
            )
            norm_b_squared += float(
                torch.sum(gradient_b * gradient_b).item()
            )
        if not all(
            math.isfinite(value)
            for value in (dot, norm_a_squared, norm_b_squared)
        ):
            raise ValueError("gradient geometry exceeds finite float64 range")
        norm_a = math.sqrt(max(norm_a_squared, 0.0))
        norm_b = math.sqrt(max(norm_b_squared, 0.0))
        eligible = common_parameter and norm_a > epsilon and norm_b > epsilon
        if eligible:
            cosine = float(dot / (norm_a * norm_b))
            conflict = bool(dot < 0.0)
            eligible_count += 1
            conflict_count += int(conflict)
            pair_dot = float(dot)
        else:
            pair_dot = float("nan")
            cosine = float("nan")
            conflict = None
        pair_results.append(
            GradientConflictPairResult(
                loss_a=loss_a,
                loss_b=loss_b,
                dot=pair_dot,
                cosine=cosine,
                conflict=conflict,
                eligible=eligible,
            )
        )
    conflict_rate = (
        float(conflict_count / eligible_count)
        if eligible_count
        else float("nan")
    )
    return GradientConflictResult(
        loss_names=loss_names,
        pairs=tuple(pair_results),
        pair_count=len(pair_results),
        eligible_pair_count=eligible_count,
        conflict_count=conflict_count,
        conflict_rate=conflict_rate,
    )


def compute_gradient_conflict_rate(
    losses: Mapping[str, Tensor],
    named_parameters: Sequence[tuple[str, Parameter]],
    distributed: bool = True,
    norm_eps: float = 1.0e-12,
) -> float:
    """Return only the eligible-pair conflict rate."""

    return compute_gradient_conflicts(
        losses,
        named_parameters,
        distributed=distributed,
        norm_eps=norm_eps,
    ).conflict_rate


def compute_modal_consistency(
    representations: Mapping[str, ModalityRepresentation],
    modalities: Sequence[str] = ("1d", "2d", "3d"),
    cohort: str = "complete",
    norm_eps: float = 1.0e-12,
) -> ModalConsistencyResult:
    """Align samples by index and average pairwise cosine similarities."""

    if not isinstance(representations, Mapping):
        raise TypeError("representations must be a mapping")
    if isinstance(modalities, (str, bytes)) or not isinstance(
        modalities, Sequence
    ):
        raise TypeError("modalities must be a sequence")
    modality_names = _nonempty_unique_names(tuple(modalities), name="modalities")
    if len(modality_names) < 2:
        raise ValueError("modalities must contain at least two names")
    if not isinstance(cohort, str) or cohort not in {
        "complete",
        "pairwise_available",
    }:
        raise ValueError("cohort must be 'complete' or 'pairwise_available'")
    epsilon = _finite_nonnegative_real("norm_eps", norm_eps)

    features: dict[str, np.ndarray] = {}
    row_by_index: dict[str, dict[int, int]] = {}
    feature_dimension: int | None = None
    for modality in modality_names:
        if modality not in representations:
            raise KeyError(f"representations is missing modality {modality!r}")
        representation = representations[modality]
        if not isinstance(representation, ModalityRepresentation):
            raise TypeError(
                f"representations[{modality!r}] must be a ModalityRepresentation"
            )
        projected = representation.projection_output.normalized
        sample_index = representation.encoder_output.sample_index
        if not isinstance(projected, Tensor) or projected.ndim != 2:
            raise ValueError(
                f"{modality!r} normalized projection must have shape [N, D]"
            )
        if not projected.is_floating_point():
            raise TypeError(f"{modality!r} normalized projection must be floating point")
        if not isinstance(sample_index, Tensor) or sample_index.ndim != 1:
            raise ValueError(f"{modality!r} sample_index must have shape [N]")
        if sample_index.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(f"{modality!r} sample_index must have an integer dtype")
        if sample_index.shape[0] != projected.shape[0]:
            raise ValueError(
                f"{modality!r} sample_index and projection lengths must match"
            )
        if sample_index.numel() and bool(torch.any(sample_index < 0)):
            raise ValueError(
                f"{modality!r} sample_index must contain non-negative values"
            )
        if feature_dimension is None:
            feature_dimension = int(projected.shape[1])
            if feature_dimension < 1:
                raise ValueError("projection feature dimension must be positive")
        elif projected.shape[1] != feature_dimension:
            raise ValueError("all modality projections must share a feature dimension")

        index_values = tuple(int(item) for item in sample_index.detach().cpu().tolist())
        if len(set(index_values)) != len(index_values):
            raise ValueError(f"{modality!r} sample_index must be unique")
        prepared = projected.detach().to(dtype=torch.float64).cpu().numpy()
        if prepared.size and not np.all(np.isfinite(prepared)):
            raise ValueError(f"{modality!r} projection contains non-finite values")
        norms = np.linalg.norm(prepared, axis=1)
        if norms.size and np.any(norms <= epsilon):
            raise ValueError(
                f"{modality!r} projection contains zero-norm rows"
            )
        features[modality] = prepared
        row_by_index[modality] = {
            sample_id: row for row, sample_id in enumerate(index_values)
        }

    complete_ids: set[int] | None = None
    if cohort == "complete":
        for modality in modality_names:
            identifiers = set(row_by_index[modality])
            complete_ids = (
                identifiers
                if complete_ids is None
                else complete_ids.intersection(identifiers)
            )

    pair_results: list[ModalConsistencyPairResult] = []
    macro_values: list[float] = []
    for modality_a, modality_b in combinations(modality_names, 2):
        if cohort == "complete":
            shared_ids = complete_ids or set()
        else:
            shared_ids = set(row_by_index[modality_a]).intersection(
                row_by_index[modality_b]
            )
        ordered_ids = sorted(shared_ids)
        count = len(ordered_ids)
        if count:
            rows_a = [row_by_index[modality_a][item] for item in ordered_ids]
            rows_b = [row_by_index[modality_b][item] for item in ordered_ids]
            values_a = features[modality_a][rows_a]
            values_b = features[modality_b][rows_b]
            cosine = np.sum(values_a * values_b, axis=1) / (
                np.linalg.norm(values_a, axis=1)
                * np.linalg.norm(values_b, axis=1)
            )
            mean = float(np.mean(cosine, dtype=np.float64))
            macro_values.append(mean)
            eligible = True
        else:
            mean = float("nan")
            eligible = False
        pair_results.append(
            ModalConsistencyPairResult(
                modality_a=modality_a,
                modality_b=modality_b,
                mean_cosine_similarity=mean,
                sample_count=count,
                eligible=eligible,
            )
        )
    return ModalConsistencyResult(
        modalities=modality_names,
        cohort=cohort,
        pairs=tuple(pair_results),
        macro_consistency=(
            float(np.mean(macro_values, dtype=np.float64))
            if macro_values
            else float("nan")
        ),
        eligible_pair_count=len(macro_values),
    )


def compute_noise_variance_ratio(
    features: ArrayLike,
    centers: ArrayLike,
    *,
    noise_rate: float,
    ddof: int = 0,
) -> NoiseVarianceRatioResult:
    """Compute Note 8 center-to-feature variance ratios by dimension."""

    feature_values = _matrix(features, name="features")
    center_values = _matrix(centers, name="centers")
    if feature_values.shape[1] != center_values.shape[1]:
        raise ValueError("features and centers must share dimension D")
    if feature_values.shape[1] < 1:
        raise ValueError("features and centers must have a positive dimension D")
    rate = _finite_nonnegative_real("noise_rate", noise_rate)
    if not isinstance(ddof, Integral) or isinstance(ddof, bool):
        raise TypeError("ddof must be an integer")
    normalized_ddof = int(ddof)
    if normalized_ddof < 0:
        raise ValueError("ddof must be non-negative")
    if normalized_ddof >= feature_values.shape[0]:
        raise ValueError("ddof must be smaller than the feature sample count")
    if normalized_ddof >= center_values.shape[0]:
        raise ValueError("ddof must be smaller than the center sample count")
    if not np.all(np.isfinite(feature_values)):
        raise ValueError("features must contain only finite values")
    if not np.all(np.isfinite(center_values)):
        raise ValueError("centers must contain only finite values")

    feature_variance = np.var(
        feature_values.astype(np.float64, copy=False),
        axis=0,
        ddof=normalized_ddof,
        dtype=np.float64,
    )
    center_variance = np.var(
        center_values.astype(np.float64, copy=False),
        axis=0,
        ddof=normalized_ddof,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(feature_variance)) or not np.all(
        np.isfinite(center_variance)
    ):
        raise ValueError("variances must be representable as finite float64 values")
    positive_feature_variance = feature_variance > 0.0
    ratios = np.full(feature_variance.shape, np.nan, dtype=np.float64)
    candidate_ratios = np.full(feature_variance.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        candidate_ratios[positive_feature_variance] = (
            center_variance[positive_feature_variance]
            / feature_variance[positive_feature_variance]
        )
    eligible = positive_feature_variance & np.isfinite(candidate_ratios)
    ratios[eligible] = candidate_ratios[eligible]
    eligible_count = int(np.count_nonzero(eligible))
    mean_ratio = (
        float(np.mean(ratios[eligible], dtype=np.float64))
        if eligible_count
        else float("nan")
    )
    if eligible_count and not math.isfinite(mean_ratio):
        raise ValueError("mean variance ratio exceeds finite float64 range")
    effective_noise = (
        float(rate * mean_ratio) if eligible_count else float("nan")
    )
    if eligible_count and not math.isfinite(effective_noise):
        raise ValueError("effective noise exceeds finite float64 range")
    return NoiseVarianceRatioResult(
        dimension_ratios=tuple(float(value) for value in ratios),
        eligible=tuple(bool(value) for value in eligible),
        mean_ratio=mean_ratio,
        effective_noise=effective_noise,
        dimension_count=int(feature_values.shape[1]),
        eligible_dimension_count=eligible_count,
        noise_rate=rate,
        ddof=normalized_ddof,
    )


def compute_noise_robustness(
    features: ArrayLike,
    centers: ArrayLike,
    *,
    noise_rate: float,
    ddof: int = 0,
) -> NoiseVarianceRatioResult:
    """Strict synonym for :func:`compute_noise_variance_ratio`."""

    return compute_noise_variance_ratio(
        features,
        centers,
        noise_rate=noise_rate,
        ddof=ddof,
    )


def _thresholds(value: object, task_count: int) -> tuple[float, ...]:
    if isinstance(value, Real) and not isinstance(value, bool):
        raw_values: tuple[object, ...] = (value,) * task_count
    elif isinstance(value, Tensor):
        array = _real_numpy(value, name="threshold")
        if array.ndim == 0:
            raw_values = (array.item(),) * task_count
        elif array.ndim == 1 and array.shape[0] == task_count:
            raw_values = tuple(array.tolist())
        else:
            raise ValueError("threshold must be scalar or have length T")
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            raw_values = (value.item(),) * task_count
        elif value.ndim == 1 and value.shape[0] == task_count:
            raw_values = tuple(value.tolist())
        else:
            raise ValueError("threshold must be scalar or have length T")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = tuple(value)
        if len(raw_values) != task_count:
            raise ValueError("threshold sequence must have length T")
    else:
        raise TypeError("threshold must be a real scalar or length-T sequence")

    thresholds: list[float] = []
    for index, raw_value in enumerate(raw_values):
        if not isinstance(raw_value, Real) or isinstance(raw_value, bool):
            raise TypeError(f"threshold[{index}] must be a real number")
        threshold = float(raw_value)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(f"threshold[{index}] must be positive and finite")
        thresholds.append(threshold)
    return tuple(thresholds)


def compute_outlier_error(
    targets: ArrayLike,
    predictions: ArrayLike,
    *,
    threshold: object,
    mask: ArrayLike | None = None,
    task_names: Sequence[str] | None = None,
) -> OutlierErrorResult:
    """Compute empirical-integral and conditional MSE on extreme targets."""

    labels, scores, valid = _supervised_matrices(targets, predictions, mask)
    names = _task_name_tuple(task_names, labels.shape[1])
    task_thresholds = _thresholds(threshold, labels.shape[1])
    task_results: list[OutlierTaskError] = []
    macro_r_ext: list[float] = []
    macro_conditional: list[float] = []
    for task_index, task_name in enumerate(names):
        task_valid = valid[:, task_index]
        valid_targets = labels[task_valid, task_index].astype(
            np.float64, copy=False
        )
        valid_predictions = scores[task_valid, task_index].astype(
            np.float64, copy=False
        )
        valid_count = int(valid_targets.size)
        tail = np.abs(valid_targets) > task_thresholds[task_index]
        tail_count = int(np.count_nonzero(tail))
        if tail_count:
            tail_errors = valid_predictions[tail] - valid_targets[tail]
            squared_tail_errors = np.square(tail_errors)
            conditional_tail_mse = float(
                np.mean(squared_tail_errors, dtype=np.float64)
            )
            r_ext = float(
                np.sum(squared_tail_errors, dtype=np.float64) / valid_count
            )
            if not math.isfinite(r_ext) or not math.isfinite(
                conditional_tail_mse
            ):
                raise ValueError("tail errors must fit finite float64 values")
            macro_r_ext.append(r_ext)
            macro_conditional.append(conditional_tail_mse)
            eligible = True
        else:
            r_ext = float("nan")
            conditional_tail_mse = float("nan")
            eligible = False
        task_results.append(
            OutlierTaskError(
                task_name=task_name,
                threshold=task_thresholds[task_index],
                r_ext=r_ext,
                conditional_tail_mse=conditional_tail_mse,
                valid_count=valid_count,
                tail_count=tail_count,
                eligible=eligible,
            )
        )
    return OutlierErrorResult(
        task_names=names,
        tasks=tuple(task_results),
        macro_r_ext=(
            float(np.mean(macro_r_ext, dtype=np.float64))
            if macro_r_ext
            else float("nan")
        ),
        macro_conditional_tail_mse=(
            float(np.mean(macro_conditional, dtype=np.float64))
            if macro_conditional
            else float("nan")
        ),
        eligible_task_count=len(macro_r_ext),
    )


def compute_tail_error(
    targets: ArrayLike,
    predictions: ArrayLike,
    *,
    threshold: object,
    mask: ArrayLike | None = None,
    task_names: Sequence[str] | None = None,
) -> OutlierErrorResult:
    """Strict synonym for :func:`compute_outlier_error`."""

    return compute_outlier_error(
        targets,
        predictions,
        threshold=threshold,
        mask=mask,
        task_names=task_names,
    )


def _validate_scaffold_identifier(value: object, *, path: str) -> Hashable:
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{path} must be hashable") from error
    if isinstance(value, (float, complex, np.floating, np.complexfloating)):
        if not bool(np.isfinite(value)):
            raise ValueError(f"{path} must not contain NaN or infinite values")
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_scaffold_identifier(
                item,
                path=f"{path}[{index}]",
            )
    elif isinstance(value, frozenset):
        for index, item in enumerate(value):
            _validate_scaffold_identifier(
                item,
                path=f"{path}<item_{index}>",
            )
    return value


def _hashable_ids(value: object, expected_length: int) -> tuple[Hashable, ...]:
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError("scaffold_ids must have shape [N]")
        raw_values = tuple(value.detach().cpu().tolist())
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("scaffold_ids must have shape [N]")
        raw_values = tuple(value.tolist())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = tuple(value)
    else:
        raise TypeError("scaffold_ids must be a one-dimensional sequence")
    if len(raw_values) != expected_length:
        raise ValueError("scaffold_ids must have the same length as targets")
    normalized: list[Hashable] = []
    for index, item in enumerate(raw_values):
        normalized.append(
            _validate_scaffold_identifier(
                item,
                path=f"scaffold_ids[{index}]",
            )
        )
    return tuple(normalized)


def _stable_source_indices(value: object, expected_length: int) -> np.ndarray:
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError("source_index must have shape [N]")
        raw_values = tuple(value.detach().cpu().tolist())
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("source_index must have shape [N]")
        raw_values = tuple(value.tolist())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = tuple(value)
    else:
        raise TypeError("source_index must be a one-dimensional sequence")
    if len(raw_values) != expected_length:
        raise ValueError("source_index must have the same length as targets")
    normalized: list[int] = []
    for index, item in enumerate(raw_values):
        if not isinstance(item, Integral) or isinstance(item, bool):
            raise TypeError(f"source_index[{index}] must be an integer")
        normalized_item = int(item)
        if normalized_item < 0:
            raise ValueError(
                f"source_index[{index}] must be non-negative"
            )
        if normalized_item > np.iinfo(np.int64).max:
            raise ValueError(
                f"source_index[{index}] exceeds the int64 range"
            )
        normalized.append(normalized_item)
    if len(set(normalized)) != len(normalized):
        raise ValueError("source_index must contain unique values")
    return np.asarray(normalized, dtype=np.int64)


def compute_por_at_scaffold(
    targets: ArrayLike,
    predictions: ArrayLike,
    scaffold_ids: Sequence[Hashable] | np.ndarray | Tensor,
    *,
    top_k: int,
    mask: ArrayLike | None = None,
    source_index: Sequence[int] | np.ndarray | Tensor | None = None,
    tie_policy: str = "expected",
) -> ScaffoldPORResult:
    """Compute positive-over-random enrichment within scaffold groups."""

    labels, scores, valid = _supervised_matrices(targets, predictions, mask)
    if labels.shape[1] != 1:
        raise ValueError("POR@Scaffold accepts a single binary task")
    valid_scores = scores[valid]
    exact_float64_integer_limit = 2**53
    if scores.dtype.kind == "i" and np.any(
        (valid_scores < -exact_float64_integer_limit)
        | (valid_scores > exact_float64_integer_limit)
    ):
        raise ValueError(
            "valid integer predictions must be exactly representable as float64"
        )
    if scores.dtype.kind == "u" and np.any(
        valid_scores > exact_float64_integer_limit
    ):
        raise ValueError(
            "valid integer predictions must be exactly representable as float64"
        )
    k = _positive_integer("top_k", top_k)
    if not isinstance(tie_policy, str) or tie_policy not in {
        "expected",
        "stable",
    }:
        raise ValueError("tie_policy must be 'expected' or 'stable'")
    identifiers = _hashable_ids(scaffold_ids, labels.shape[0])
    stable_indices: np.ndarray | None = None
    if tie_policy == "stable":
        if source_index is None:
            raise ValueError("source_index is required for tie_policy='stable'")
        stable_indices = _stable_source_indices(source_index, labels.shape[0])
    elif source_index is not None:
        _stable_source_indices(source_index, labels.shape[0])

    valid_vector = valid[:, 0]
    valid_labels = labels[valid_vector, 0]
    if valid_labels.size and not np.all(
        (valid_labels == 0) | (valid_labels == 1)
    ):
        raise ValueError("targets must be strictly 0 or 1 at valid positions")

    rows_by_group: dict[Hashable, list[int]] = {}
    for row, scaffold_id in enumerate(identifiers):
        rows_by_group.setdefault(scaffold_id, [])
        if valid_vector[row]:
            rows_by_group[scaffold_id].append(row)

    group_results: list[ScaffoldPORGroupResult] = []
    eligible_values: list[float] = []
    weighted_sum = 0.0
    eligible_size = 0
    for scaffold_id, rows in rows_by_group.items():
        size = len(rows)
        group_labels = labels[rows, 0].astype(np.float64, copy=False)
        positive_count = int(np.count_nonzero(group_labels == 1.0))
        if size < k:
            hits = float("nan")
            baseline = (
                float(positive_count / size) if size else float("nan")
            )
            por = float("nan")
            eligible = False
            reason = "group_too_small"
        elif positive_count == 0:
            hits = 0.0
            baseline = 0.0
            por = float("nan")
            eligible = False
            reason = "no_positive_labels"
        else:
            group_scores = scores[rows, 0].astype(np.float64, copy=False)
            if tie_policy == "expected":
                cutoff = float(np.partition(group_scores, size - k)[size - k])
                above = group_scores > cutoff
                tied = group_scores == cutoff
                needed = k - int(np.count_nonzero(above))
                tied_count = int(np.count_nonzero(tied))
                hits = float(
                    np.sum(group_labels[above], dtype=np.float64)
                    + needed
                    * np.sum(group_labels[tied], dtype=np.float64)
                    / tied_count
                )
            else:
                if stable_indices is None:
                    raise RuntimeError("stable source indices were not prepared")
                group_sources = stable_indices[rows]
                order = np.lexsort((group_sources, -group_scores))
                hits = float(
                    np.sum(group_labels[order[:k]], dtype=np.float64)
                )
            baseline = float(positive_count / size)
            por = float((hits / k) / baseline)
            eligible = True
            reason = "eligible"
            eligible_values.append(por)
            weighted_sum += size * por
            eligible_size += size
        group_results.append(
            ScaffoldPORGroupResult(
                scaffold_id=scaffold_id,
                size=size,
                positive_count=positive_count,
                hits_at_k=hits,
                baseline_positive_rate=baseline,
                por_at_k=por,
                eligible=eligible,
                reason=reason,
            )
        )
    return ScaffoldPORResult(
        top_k=k,
        tie_policy=tie_policy,
        groups=tuple(group_results),
        group_count=len(group_results),
        eligible_group_count=len(eligible_values),
        eligible_sample_count=eligible_size,
        macro_por=(
            float(np.mean(eligible_values, dtype=np.float64))
            if eligible_values
            else float("nan")
        ),
        weighted_por=(
            float(weighted_sum / eligible_size)
            if eligible_size
            else float("nan")
        ),
    )


def scaffold_topk_enrichment(
    targets: ArrayLike,
    predictions: ArrayLike,
    scaffold_ids: Sequence[Hashable] | np.ndarray | Tensor,
    *,
    top_k: int,
    mask: ArrayLike | None = None,
    source_index: Sequence[int] | np.ndarray | Tensor | None = None,
    tie_policy: str = "expected",
) -> ScaffoldPORResult:
    """Strict synonym for :func:`compute_por_at_scaffold`."""

    return compute_por_at_scaffold(
        targets,
        predictions,
        scaffold_ids,
        top_k=top_k,
        mask=mask,
        source_index=source_index,
        tie_policy=tie_policy,
    )


def _matching_weights(
    matching: ACSMOutput | Mapping[str, ArrayLike] | ArrayLike,
    modality: str | None,
) -> tuple[tuple[str, ArrayLike], ...]:
    if modality is not None and (
        not isinstance(modality, str) or not modality.strip()
    ):
        raise ValueError("modality must be a non-empty string or None")
    if isinstance(matching, ACSMOutput):
        if modality is not None:
            raise ValueError("modality is only valid for a single weight matrix")
        return tuple(
            (name, matching.modality_matches[name].positive_weights)
            for name in matching.target_modalities
        )
    if isinstance(matching, Mapping):
        if modality is not None:
            raise ValueError("modality is only valid for a single weight matrix")
        items = tuple(matching.items())
        names = _nonempty_unique_names(
            tuple(name for name, _ in items),
            name="matching",
        )
        return tuple((name, item[1]) for name, item in zip(names, items))
    if isinstance(matching, (Tensor, np.ndarray)):
        name = "weights" if modality is None else modality.strip()
        return ((name, matching),)
    raise TypeError(
        "matching must be an ACSMOutput, mapping, or weight matrix"
    )


def compute_matching_entropy(
    matching: ACSMOutput | Mapping[str, ArrayLike] | ArrayLike,
    *,
    modality: str | None = None,
    atol: float = 1.0e-6,
) -> MatchingEntropyResult:
    """Summarize exact Shannon entropy of ACSM positive weights."""

    tolerance = _finite_nonnegative_real("atol", atol)
    weight_items = _matching_weights(matching, modality)
    if not weight_items:
        raise ValueError("matching must contain at least one modality")
    summaries: list[MatchingEntropyModalityResult] = []
    for name, weights in weight_items:
        values = _matrix(weights, name=f"matching[{name!r}]").astype(
            np.float64, copy=False
        )
        sample_count, center_count = values.shape
        if center_count < 1:
            raise ValueError(f"matching[{name!r}] must have K >= 1")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"matching[{name!r}] must contain finite weights")
        if np.any(values < 0.0):
            raise ValueError(f"matching[{name!r}] must contain non-negative weights")
        row_sums = np.sum(values, axis=1, dtype=np.float64)
        if sample_count and not np.all(
            np.isclose(row_sums, 1.0, rtol=0.0, atol=tolerance)
        ):
            raise ValueError(
                f"rows of matching[{name!r}] must sum to one within atol"
            )

        entropies = np.zeros(sample_count, dtype=np.float64)
        for row in range(sample_count):
            positive = values[row] > 0.0
            entropies[row] = -np.sum(
                values[row, positive] * np.log(values[row, positive]),
                dtype=np.float64,
            )
        if center_count == 1:
            normalized_entropies = np.zeros(sample_count, dtype=np.float64)
        else:
            normalized_entropies = entropies / math.log(center_count)
        effective_counts = np.exp(entropies)

        mean_entropy = (
            float(np.mean(entropies, dtype=np.float64))
            if sample_count
            else float("nan")
        )
        mean_normalized = (
            float(np.mean(normalized_entropies, dtype=np.float64))
            if sample_count
            else float("nan")
        )
        mean_effective = (
            float(np.mean(effective_counts, dtype=np.float64))
            if sample_count
            else float("nan")
        )
        if sample_count >= 2:
            std_entropy = float(np.std(entropies, ddof=1, dtype=np.float64))
            std_normalized = float(
                np.std(normalized_entropies, ddof=1, dtype=np.float64)
            )
            std_effective = float(
                np.std(effective_counts, ddof=1, dtype=np.float64)
            )
        else:
            std_entropy = float("nan")
            std_normalized = float("nan")
            std_effective = float("nan")
        summaries.append(
            MatchingEntropyModalityResult(
                modality=name,
                center_count=center_count,
                sample_count=sample_count,
                mean_entropy=mean_entropy,
                std_entropy=std_entropy,
                mean_normalized_entropy=mean_normalized,
                std_normalized_entropy=std_normalized,
                mean_effective_center_count=mean_effective,
                std_effective_center_count=std_effective,
            )
        )
    return MatchingEntropyResult(
        modalities=tuple(item.modality for item in summaries),
        per_modality=tuple(summaries),
    )


__all__ = [
    "GradientConflictAccumulator",
    "GradientConflictAccumulatorResult",
    "GradientConflictPairAccumulation",
    "GradientConflictPairResult",
    "GradientConflictResult",
    "MatchingEntropyModalityResult",
    "MatchingEntropyResult",
    "ModalConsistencyPairResult",
    "ModalConsistencyResult",
    "NoiseVarianceRatioResult",
    "OutlierErrorResult",
    "OutlierTaskError",
    "ScaffoldPORGroupResult",
    "ScaffoldPORResult",
    "compute_gradient_conflict_rate",
    "compute_gradient_conflicts",
    "compute_matching_entropy",
    "compute_modal_consistency",
    "compute_noise_robustness",
    "compute_noise_variance_ratio",
    "compute_outlier_error",
    "compute_por_at_scaffold",
    "compute_tail_error",
    "scaffold_topk_enrichment",
]

"""Attribution utilities for SemMol's fused semantic latent features.

The features analysed here are the anchor embedding and ACSM-retrieved DCL
center embeddings supplied to the property head.  They are not attributions
to the original molecular inputs or to an end-to-end encoder pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.alignment.acsm import ACSMOutput
from src.models.semmol import SemMolFinetuningOutput


_OUTPUT_SPACES: Final[frozenset[str]] = frozenset(
    {"value", "logit", "probability"}
)


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive, got {normalized}")
    return normalized


def _positive_finite_real(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return normalized


def _normalize_name(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip().lower()


def _normalize_group_names(
    group_names: Sequence[str],
    *,
    expected_count: int | None = None,
) -> tuple[str, ...]:
    if isinstance(group_names, (str, bytes)) or not isinstance(
        group_names, Sequence
    ):
        raise TypeError("group_names must be a sequence of strings")
    normalized = tuple(
        _normalize_name(f"group_names[{index}]", group_name)
        for index, group_name in enumerate(group_names)
    )
    if len(normalized) < 2:
        raise ValueError(
            "group_names must contain one anchor and at least one target group"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("group_names must be unique")
    if expected_count is not None and len(normalized) != expected_count:
        raise ValueError(
            f"expected {expected_count} group names, got {len(normalized)}"
        )
    return normalized


def _normalize_modalities(
    name: str,
    modalities: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(modalities, (str, bytes)) or not isinstance(
        modalities, Sequence
    ):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(
        _normalize_name(f"{name}[{index}]", modality)
        for index, modality in enumerate(modalities)
    )
    if not normalized:
        raise ValueError(f"{name} must contain at least one modality")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique modalities")
    return normalized


def _validate_feature_matrix(
    name: str,
    value: object,
    *,
    expected_shape: tuple[int, int] | None = None,
    expected_device: torch.device | None = None,
    expected_dtype: torch.dtype | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [batch, feature_dim], got "
            f"{tuple(value.shape)}"
        )
    if expected_shape is not None and tuple(value.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
        )
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} must have non-empty batch and feature axes")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a real floating-point dtype")
    if expected_device is not None and value.device != expected_device:
        raise ValueError(
            f"{name} must be on device {expected_device}, got {value.device}"
        )
    if expected_dtype is not None and value.dtype != expected_dtype:
        raise TypeError(
            f"{name} must use dtype {expected_dtype}, got {value.dtype}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values")
    return value


def _validate_blocks(
    name: str,
    value: object,
    *,
    expected_groups: int | None = None,
    expected_feature_dim: int | None = None,
    require_nonempty_batch: bool = True,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 3:
        raise ValueError(
            f"{name} must have shape [batch, groups, feature_dim], got "
            f"{tuple(value.shape)}"
        )
    batch_size, group_count, feature_dim = value.shape
    if require_nonempty_batch and batch_size <= 0:
        raise ValueError(f"{name} must contain at least one sample")
    if group_count < 2:
        raise ValueError(
            f"{name} must contain one anchor and at least one target group"
        )
    if feature_dim <= 0:
        raise ValueError(f"{name} feature_dim must be positive")
    if expected_groups is not None and group_count != expected_groups:
        raise ValueError(
            f"{name} has {group_count} groups, expected {expected_groups}"
        )
    if (
        expected_feature_dim is not None
        and feature_dim != expected_feature_dim
    ):
        raise ValueError(
            f"{name} has feature_dim={feature_dim}, expected "
            f"{expected_feature_dim}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a real floating-point dtype")
    if value.numel() > 0 and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values")
    return value


def _module_float_reference(module: nn.Module) -> Tensor:
    references = [
        tensor
        for tensor in (*tuple(module.parameters()), *tuple(module.buffers()))
        if tensor.is_floating_point()
    ]
    if not references:
        raise ValueError("property_head must contain floating-point state")
    first = references[0]
    for reference in references[1:]:
        if reference.device != first.device:
            raise ValueError(
                "all floating-point property_head state must share a device"
            )
        if reference.dtype != first.dtype:
            raise TypeError(
                "all floating-point property_head state must share a dtype"
            )
    return first


def _validate_predictor_blocks(
    predictor: "SemanticFusionPredictor",
    name: str,
    blocks: object,
) -> Tensor:
    validated = _validate_blocks(
        name,
        blocks,
        expected_groups=len(predictor.group_names),
        expected_feature_dim=predictor.feature_dim,
    )
    reference = _module_float_reference(predictor.property_head)
    if validated.device != reference.device:
        raise ValueError(
            f"{name} and property_head must share a device: "
            f"{validated.device} != {reference.device}"
        )
    if validated.dtype != reference.dtype:
        raise TypeError(
            f"{name} and property_head must share a dtype: "
            f"{validated.dtype} != {reference.dtype}"
        )
    return validated


def _validate_attribution_inputs(
    predictor: object,
    foreground: object,
    background: object,
) -> tuple["SemanticFusionPredictor", Tensor, Tensor]:
    if not isinstance(predictor, SemanticFusionPredictor):
        raise TypeError("predictor must be a SemanticFusionPredictor")
    foreground_tensor = _validate_predictor_blocks(
        predictor, "foreground", foreground
    )
    background_tensor = _validate_predictor_blocks(
        predictor, "background", background
    )
    if background_tensor.shape[0] <= 0:
        raise ValueError("background must contain at least one training sample")
    if foreground_tensor.shape[1:] != background_tensor.shape[1:]:
        raise ValueError(
            "foreground and background must have matching group and feature "
            "dimensions"
        )
    if foreground_tensor.device != background_tensor.device:
        raise ValueError("foreground and background must share a device")
    if foreground_tensor.dtype != background_tensor.dtype:
        raise TypeError("foreground and background must share a dtype")
    return predictor, foreground_tensor, background_tensor


def _normalize_ids(
    name: str,
    values: Sequence[Hashable] | None,
    *,
    expected_length: int,
) -> tuple[Hashable, ...]:
    if values is None:
        return tuple(range(expected_length))
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of hashable IDs")
    normalized = tuple(values)
    if len(normalized) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} IDs, got {len(normalized)}"
        )
    seen: set[Hashable] = set()
    for index, value in enumerate(normalized):
        if not isinstance(value, Hashable):
            raise TypeError(f"{name}[{index}] must be hashable")
        if isinstance(value, Real) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name}[{index}] must be finite")
        if value in seen:
            raise ValueError(f"{name} must contain unique IDs")
        seen.add(value)
    return normalized


def _detach_tensor(value: Tensor) -> Tensor:
    return value.detach()


def _readonly_numpy(value: object) -> np.ndarray:
    if isinstance(value, Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    readonly = np.array(array, copy=True)
    readonly.setflags(write=False)
    return readonly


def _validate_prediction_tensor(
    name: str,
    value: object,
    *,
    batch_size: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[0] != batch_size:
        raise ValueError(
            f"{name} must have shape [{batch_size}, tasks], got "
            f"{tuple(value.shape)}"
        )
    if value.shape[1] <= 0:
        raise ValueError(f"{name} must contain at least one task")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a real floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values")
    return value


@contextmanager
def _preserve_training_states(
    predictor: "SemanticFusionPredictor",
) -> Iterator[None]:
    """Evaluate temporarily and restore every module's exact prior state."""

    module_states = tuple(
        (module, bool(module.training)) for module in predictor.modules()
    )
    try:
        predictor.eval()
        yield
    finally:
        for module, training in module_states:
            module.training = training


@dataclass(frozen=True)
class SemanticFeatureBatch:
    """Detached semantic latent blocks and their local sample indices."""

    blocks: Tensor
    sample_index: Tensor
    group_names: tuple[str, ...]
    fusion_eps: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocks": self.blocks,
            "sample_index": self.sample_index,
            "group_names": self.group_names,
            "fusion_eps": self.fusion_eps,
        }


@dataclass(frozen=True)
class GroupShapleyResult:
    """Exact interventional Shapley values for semantic feature groups."""

    phi: Tensor
    base: Tensor
    full: Tensor
    additivity_residual: Tensor
    group_names: tuple[str, ...]
    output_space: str
    foreground_ids: tuple[Hashable, ...]
    background_ids: tuple[Hashable, ...]

    @property
    def values(self) -> Tensor:
        return self.phi

    @property
    def base_values(self) -> Tensor:
        return self.base

    @property
    def full_predictions(self) -> Tensor:
        return self.full

    def as_dict(self) -> dict[str, Any]:
        return {
            "phi": self.phi,
            "base": self.base,
            "full": self.full,
            "additivity_residual": self.additivity_residual,
            "group_names": self.group_names,
            "output_space": self.output_space,
            "foreground_ids": self.foreground_ids,
            "background_ids": self.background_ids,
        }


@dataclass(frozen=True)
class ModalityOcclusionResult:
    """Signed full-minus-occluded effects for semantic feature groups."""

    effects: Tensor
    full: Tensor
    occluded: Tensor
    group_names: tuple[str, ...]
    output_space: str
    foreground_ids: tuple[Hashable, ...]
    background_ids: tuple[Hashable, ...]

    @property
    def values(self) -> Tensor:
        return self.effects

    @property
    def full_predictions(self) -> Tensor:
        return self.full

    def as_dict(self) -> dict[str, Any]:
        return {
            "effects": self.effects,
            "full": self.full,
            "occluded": self.occluded,
            "group_names": self.group_names,
            "output_space": self.output_space,
            "foreground_ids": self.foreground_ids,
            "background_ids": self.background_ids,
        }


@dataclass(frozen=True)
class FeatureSHAPResult:
    """Gradient SHAP attributions in the semantic latent feature space."""

    shap_values: np.ndarray
    base: np.ndarray
    full: np.ndarray
    additivity_residual: np.ndarray
    task_indices: tuple[int, ...]
    group_names: tuple[str, ...]
    output_space: str
    foreground_ids: tuple[Hashable, ...]
    background_ids: tuple[Hashable, ...]

    @property
    def values(self) -> np.ndarray:
        return self.shap_values

    @property
    def base_values(self) -> np.ndarray:
        return self.base

    @property
    def full_predictions(self) -> np.ndarray:
        return self.full

    def as_dict(self) -> dict[str, Any]:
        return {
            "shap_values": self.shap_values,
            "base": self.base,
            "full": self.full,
            "additivity_residual": self.additivity_residual,
            "task_indices": self.task_indices,
            "group_names": self.group_names,
            "output_space": self.output_space,
            "foreground_ids": self.foreground_ids,
            "background_ids": self.background_ids,
        }


@dataclass(frozen=True)
class SHAPSummary:
    """Paper-facing group totals and signed feature heatmap data."""

    task_index: int
    group_names: tuple[str, ...]
    top_prediction_sample_ids: tuple[Hashable, ...]
    top_prediction_scores: np.ndarray
    mean_absolute_group_contributions: np.ndarray
    group_contribution_proportions: np.ndarray
    heatmap: np.ndarray
    heatmap_feature_keys: tuple[tuple[str, int], ...]
    heatmap_sample_ids: tuple[Hashable, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_index": self.task_index,
            "group_names": self.group_names,
            "top_prediction_sample_ids": self.top_prediction_sample_ids,
            "top_prediction_scores": self.top_prediction_scores,
            "mean_absolute_group_contributions": (
                self.mean_absolute_group_contributions
            ),
            "group_contribution_proportions": (
                self.group_contribution_proportions
            ),
            "heatmap": self.heatmap,
            "heatmap_feature_keys": self.heatmap_feature_keys,
            "heatmap_sample_ids": self.heatmap_sample_ids,
        }


def extract_semantic_feature_batch(
    output: SemMolFinetuningOutput,
    *,
    anchor_modality: str,
    fusion_eps: float,
    target_modalities: Sequence[str] | None = None,
) -> SemanticFeatureBatch:
    """Extract the property head's semantic latent sources from model output.

    Target groups are ACSM-retrieved DCL center embeddings.  Group names are
    derived from the explicit modalities, for example ``1d_anchor``,
    ``retrieved_2d_centers``, and ``retrieved_3d_centers`` in the standard
    configuration.  ``fusion_eps`` must be the originating model's ACSM eps.
    """

    if not isinstance(output, SemMolFinetuningOutput):
        raise TypeError("output must be a SemMolFinetuningOutput")
    if not isinstance(output.acsm_output, ACSMOutput):
        raise TypeError("output.acsm_output must be an ACSMOutput")

    anchor_name = _normalize_name("anchor_modality", anchor_modality)
    normalized_fusion_eps = _positive_finite_real("fusion_eps", fusion_eps)
    acsm = output.acsm_output
    actual_targets = _normalize_modalities(
        "output.acsm_output.target_modalities",
        acsm.target_modalities,
    )
    if target_modalities is not None:
        requested_targets = _normalize_modalities(
            "target_modalities", target_modalities
        )
        if requested_targets != actual_targets:
            raise ValueError(
                "target_modalities must exactly match "
                "output.acsm_output.target_modalities in order"
            )
    if anchor_name in actual_targets:
        raise ValueError("anchor_modality cannot also be a target modality")
    match_keys = tuple(acsm.modality_matches)
    normalized_match_keys = _normalize_modalities(
        "output.acsm_output.modality_matches keys", match_keys
    )
    if normalized_match_keys != actual_targets:
        raise ValueError(
            "ACSM modality_matches keys must exactly match target_modalities "
            "in order"
        )

    anchor = _validate_feature_matrix(
        "output.acsm_output.anchor_embedding", acsm.anchor_embedding
    )
    batch_size, feature_dim = anchor.shape
    target_blocks: list[Tensor] = []
    for modality, raw_key in zip(actual_targets, match_keys):
        match = acsm.modality_matches[raw_key]
        positive = _validate_feature_matrix(
            f"positive_embedding for target {modality!r}",
            match.positive_embedding,
            expected_shape=(batch_size, feature_dim),
            expected_device=anchor.device,
            expected_dtype=anchor.dtype,
        )
        target_blocks.append(positive)

    fused_target = _validate_feature_matrix(
        "output.acsm_output.positive_embedding",
        acsm.positive_embedding,
        expected_shape=(batch_size, feature_dim),
        expected_device=anchor.device,
        expected_dtype=anchor.dtype,
    )
    reconstructed_target = F.normalize(
        torch.stack(target_blocks, dim=0).sum(dim=0),
        p=2.0,
        dim=-1,
        eps=normalized_fusion_eps,
    )
    if not torch.allclose(
        reconstructed_target,
        fused_target,
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError(
            "output.acsm_output.positive_embedding is inconsistent with the "
            "ordered target positive embeddings and fusion_eps"
        )

    fused_features = _validate_feature_matrix(
        "output.fused_features",
        output.fused_features,
        expected_shape=(batch_size, 2 * feature_dim),
        expected_device=anchor.device,
        expected_dtype=anchor.dtype,
    )
    reconstructed_features = torch.cat((anchor, fused_target), dim=-1)
    if not torch.allclose(
        reconstructed_features,
        fused_features,
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError(
            "output.fused_features is inconsistent with the concatenated "
            "anchor and ACSM positive embedding"
        )
    sample_index = output.anchor_sample_index
    if not isinstance(sample_index, Tensor):
        raise TypeError("output.anchor_sample_index must be a torch.Tensor")
    if sample_index.ndim != 1 or sample_index.shape[0] != batch_size:
        raise ValueError(
            "output.anchor_sample_index must have shape "
            f"[{batch_size}], got {tuple(sample_index.shape)}"
        )
    if sample_index.dtype != torch.long:
        raise TypeError("output.anchor_sample_index must use torch.long dtype")
    if sample_index.device != anchor.device:
        raise ValueError(
            "output.anchor_sample_index and semantic blocks must share a device"
        )
    expected_index = torch.arange(
        batch_size, dtype=torch.long, device=sample_index.device
    )
    if not torch.equal(sample_index, expected_index):
        raise ValueError(
            "output.anchor_sample_index must contain every sample exactly once "
            "in original batch order"
        )

    group_names = (
        f"{anchor_name}_anchor",
        *(f"retrieved_{modality}_centers" for modality in actual_targets),
    )
    normalized_group_names = _normalize_group_names(
        group_names, expected_count=1 + len(actual_targets)
    )
    blocks = torch.stack((anchor, *target_blocks), dim=1).detach()
    return SemanticFeatureBatch(
        blocks=blocks,
        sample_index=sample_index.detach(),
        group_names=normalized_group_names,
        fusion_eps=normalized_fusion_eps,
    )


class SemanticFusionPredictor(nn.Module):
    """Apply an existing property head to grouped SemMol latent features.

    Group zero is the anchor.  Remaining retrieved-center groups are summed,
    L2-normalized, and concatenated with the anchor exactly as in SemMol
    finetuning.  The supplied property head is referenced directly; it is not
    copied, replaced, or retrained.
    """

    def __init__(
        self,
        property_head: nn.Module,
        group_names: Sequence[str],
        *,
        eps: float,
        output_space: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(property_head, nn.Module):
            raise TypeError("property_head must be a torch.nn.Module")
        input_dim = getattr(property_head, "input_dim", None)
        if not isinstance(input_dim, Integral) or isinstance(input_dim, bool):
            raise TypeError("property_head.input_dim must be an integer")
        normalized_input_dim = int(input_dim)
        if normalized_input_dim <= 0 or normalized_input_dim % 2 != 0:
            raise ValueError(
                "property_head.input_dim must be a positive even dimension"
            )
        task_type = getattr(property_head, "task_type", None)
        if not isinstance(task_type, str):
            raise TypeError(
                "property_head.task_type must be 'classification' or "
                "'regression'"
            )
        normalized_task_type = task_type.strip().lower()
        if normalized_task_type not in {"classification", "regression"}:
            raise ValueError(
                "property_head.task_type must be 'classification' or "
                "'regression'"
            )
        num_tasks = getattr(property_head, "num_tasks", None)
        self.num_tasks = _positive_integer("property_head.num_tasks", num_tasks)
        self.group_names = _normalize_group_names(group_names)
        self.feature_dim = normalized_input_dim // 2
        self.eps = _positive_finite_real("eps", eps)

        if output_space is None:
            normalized_output_space = (
                "logit"
                if normalized_task_type == "classification"
                else "value"
            )
        else:
            normalized_output_space = _normalize_name(
                "output_space", output_space
            )
        if normalized_output_space not in _OUTPUT_SPACES:
            raise ValueError(
                f"unsupported output_space={output_space!r}; expected one of "
                f"{sorted(_OUTPUT_SPACES)}"
            )
        if normalized_task_type == "regression":
            if normalized_output_space != "value":
                raise ValueError(
                    "regression property heads require output_space='value'"
                )
        elif normalized_output_space == "value":
            raise ValueError(
                "classification property heads require output_space='logit' "
                "or 'probability'"
            )

        _module_float_reference(property_head)
        self.property_head = property_head
        self.task_type = normalized_task_type
        self.output_space = normalized_output_space

    def forward(self, blocks: Tensor) -> Tensor:
        validated = _validate_predictor_blocks(self, "blocks", blocks)
        target_semantics = validated[:, 1:, :].sum(dim=1)
        target_semantics = F.normalize(
            target_semantics,
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        fused = torch.cat((validated[:, 0, :], target_semantics), dim=-1)
        raw = _validate_prediction_tensor(
            "property_head output",
            self.property_head(fused),
            batch_size=validated.shape[0],
        )
        if raw.shape[1] != self.num_tasks:
            raise ValueError(
                f"property_head produced {raw.shape[1]} tasks, expected "
                f"{self.num_tasks}"
            )
        if self.output_space == "probability":
            return torch.sigmoid(raw)
        return raw


def _mean_predictor_output(
    predictor: SemanticFusionPredictor,
    blocks: Tensor,
    *,
    batch_size: int,
) -> Tensor:
    total: Tensor | None = None
    row_count = int(blocks.shape[0])
    for start in range(0, row_count, batch_size):
        predictions = predictor(blocks[start : start + batch_size])
        chunk_sum = predictions.sum(dim=0)
        total = chunk_sum if total is None else total + chunk_sum
    if total is None:
        raise ValueError("background must contain at least one sample")
    return total / float(row_count)


def exact_group_shapley(
    predictor: SemanticFusionPredictor,
    foreground: Tensor,
    background: Tensor,
    *,
    foreground_ids: Sequence[Hashable] | None = None,
    background_ids: Sequence[Hashable] | None = None,
    max_groups: int = 8,
    background_batch_size: int = 256,
) -> GroupShapleyResult:
    """Compute exact interventional Shapley values for latent groups.

    Missing groups are independently replaced by rows from a non-empty
    training background, and each coalition value is the background
    expectation.  This enumerates every coalition and is intentionally
    limited to a small number of semantic groups.
    """

    predictor, foreground, background = _validate_attribution_inputs(
        predictor, foreground, background
    )
    normalized_max_groups = _positive_integer("max_groups", max_groups)
    chunk_size = _positive_integer(
        "background_batch_size", background_batch_size
    )
    batch_size, group_count, _ = foreground.shape
    if group_count > normalized_max_groups:
        raise ValueError(
            f"exact Shapley enumeration supports at most "
            f"max_groups={normalized_max_groups}, got {group_count}"
        )
    normalized_foreground_ids = _normalize_ids(
        "foreground_ids", foreground_ids, expected_length=batch_size
    )
    normalized_background_ids = _normalize_ids(
        "background_ids",
        background_ids,
        expected_length=int(background.shape[0]),
    )

    with _preserve_training_states(predictor):
        with torch.no_grad():
            full = _validate_prediction_tensor(
                "full predictions",
                predictor(foreground),
                batch_size=batch_size,
            )
            base = _mean_predictor_output(
                predictor, background, batch_size=chunk_size
            )
            task_count = int(full.shape[1])
            coalition_count = 1 << group_count
            phi = foreground.new_zeros((batch_size, task_count, group_count))
            factorial = tuple(
                math.factorial(index) for index in range(group_count + 1)
            )
            denominator = float(factorial[group_count])

            for sample_index in range(batch_size):
                coalition_values = foreground.new_empty(
                    (coalition_count, task_count)
                )
                coalition_values[0] = base
                coalition_values[-1] = full[sample_index]
                foreground_row = foreground[sample_index : sample_index + 1]
                for mask in range(1, coalition_count - 1):
                    present_groups = tuple(
                        group_index
                        for group_index in range(group_count)
                        if mask & (1 << group_index)
                    )
                    total: Tensor | None = None
                    for start in range(
                        0, int(background.shape[0]), chunk_size
                    ):
                        background_chunk = background[start : start + chunk_size]
                        hybrid = background_chunk.clone()
                        for group_index in present_groups:
                            hybrid[:, group_index, :] = foreground_row[
                                0, group_index, :
                            ]
                        chunk_sum = predictor(hybrid).sum(dim=0)
                        total = chunk_sum if total is None else total + chunk_sum
                    if total is None:
                        raise ValueError(
                            "background must contain at least one sample"
                        )
                    coalition_values[mask] = total / float(
                        background.shape[0]
                    )

                for group_index in range(group_count):
                    group_bit = 1 << group_index
                    for mask in range(coalition_count):
                        if mask & group_bit:
                            continue
                        subset_size = mask.bit_count()
                        weight = (
                            factorial[subset_size]
                            * factorial[group_count - subset_size - 1]
                            / denominator
                        )
                        phi[sample_index, :, group_index] += weight * (
                            coalition_values[mask | group_bit]
                            - coalition_values[mask]
                        )

            residual = full - (base.unsqueeze(0) + phi.sum(dim=-1))

    return GroupShapleyResult(
        phi=_detach_tensor(phi),
        base=_detach_tensor(base),
        full=_detach_tensor(full),
        additivity_residual=_detach_tensor(residual),
        group_names=predictor.group_names,
        output_space=predictor.output_space,
        foreground_ids=normalized_foreground_ids,
        background_ids=normalized_background_ids,
    )


def modality_occlusion(
    predictor: SemanticFusionPredictor,
    foreground: Tensor,
    background: Tensor,
    *,
    foreground_ids: Sequence[Hashable] | None = None,
    background_ids: Sequence[Hashable] | None = None,
) -> ModalityOcclusionResult:
    """Measure signed effects after one-group background-mean replacement.

    This is a standalone occlusion analysis.  It is not a SHAP estimator and
    its signed effects are not absolute-value normalized.
    """

    predictor, foreground, background = _validate_attribution_inputs(
        predictor, foreground, background
    )
    batch_size, group_count, _ = foreground.shape
    normalized_foreground_ids = _normalize_ids(
        "foreground_ids", foreground_ids, expected_length=batch_size
    )
    normalized_background_ids = _normalize_ids(
        "background_ids",
        background_ids,
        expected_length=int(background.shape[0]),
    )

    with _preserve_training_states(predictor):
        with torch.no_grad():
            full = predictor(foreground)
            replacement = background.mean(dim=0)
            occluded_by_group: list[Tensor] = []
            for group_index in range(group_count):
                occluded_blocks = foreground.clone()
                occluded_blocks[:, group_index, :] = replacement[group_index]
                occluded_by_group.append(predictor(occluded_blocks))
            occluded = torch.stack(occluded_by_group, dim=-1)
            effects = full.unsqueeze(-1) - occluded

    return ModalityOcclusionResult(
        effects=_detach_tensor(effects),
        full=_detach_tensor(full),
        occluded=_detach_tensor(occluded),
        group_names=predictor.group_names,
        output_space=predictor.output_space,
        foreground_ids=normalized_foreground_ids,
        background_ids=normalized_background_ids,
    )


def _normalize_gradient_shap_values(
    raw_values: object,
    *,
    batch_size: int,
    task_count: int,
    group_count: int,
    feature_dim: int,
) -> np.ndarray:
    expected_feature_shape = (batch_size, group_count, feature_dim)
    if isinstance(raw_values, list):
        if len(raw_values) != task_count:
            raise ValueError(
                "GradientExplainer returned a list whose length does not "
                f"match the {task_count} model tasks"
            )
        per_task: list[np.ndarray] = []
        for task_index, task_values in enumerate(raw_values):
            array = _readonly_numpy(task_values)
            if array.shape == (*expected_feature_shape, 1):
                array = array[..., 0]
            if array.shape != expected_feature_shape:
                raise ValueError(
                    f"GradientExplainer task {task_index} values must have "
                    f"shape {expected_feature_shape}, got {array.shape}"
                )
            per_task.append(array)
        normalized = np.stack(per_task, axis=1)
    else:
        array = _readonly_numpy(raw_values)
        if array.shape == expected_feature_shape:
            if task_count != 1:
                raise ValueError(
                    "GradientExplainer returned single-output values for a "
                    f"{task_count}-task model"
                )
            normalized = array[:, np.newaxis, :, :]
        elif array.shape == (*expected_feature_shape, task_count):
            normalized = np.moveaxis(array, -1, 1)
        elif array.shape == (
            batch_size,
            task_count,
            group_count,
            feature_dim,
        ):
            normalized = array
        elif array.shape == (
            task_count,
            batch_size,
            group_count,
            feature_dim,
        ):
            normalized = np.moveaxis(array, 0, 1)
        else:
            raise ValueError(
                "unsupported GradientExplainer output shape "
                f"{array.shape}; expected a per-task list, "
                f"{expected_feature_shape}, or an array with a task axis"
            )
    if not np.issubdtype(normalized.dtype, np.floating):
        raise TypeError(
            "GradientExplainer values must use a real floating-point dtype"
        )
    if not bool(np.isfinite(normalized).all()):
        raise ValueError("GradientExplainer values contain NaN or infinity")
    return _readonly_numpy(normalized)


def gradient_feature_shap(
    predictor: SemanticFusionPredictor,
    foreground: Tensor,
    background: Tensor,
    *,
    task_index: int | None = None,
    foreground_ids: Sequence[Hashable] | None = None,
    background_ids: Sequence[Hashable] | None = None,
) -> FeatureSHAPResult:
    """Run SHAP GradientExplainer on grouped semantic latent features.

    When ``task_index`` is supplied, the explainer receives a one-output
    wrapper and therefore computes gradients only for that requested task.
    """

    try:
        import shap
    except ImportError as error:
        raise ImportError(
            "gradient_feature_shap requires shap==0.44.1; install the SHAP "
            "dependency before running latent feature attribution"
        ) from error

    predictor, foreground, background = _validate_attribution_inputs(
        predictor, foreground, background
    )
    batch_size, group_count, feature_dim = foreground.shape
    if task_index is not None:
        if not isinstance(task_index, Integral) or isinstance(task_index, bool):
            raise TypeError("task_index must be an integer or None")
        normalized_task_index = int(task_index)
        if not 0 <= normalized_task_index < predictor.num_tasks:
            raise IndexError(
                f"task_index={normalized_task_index} is outside [0, "
                f"{predictor.num_tasks})"
            )
        task_indices = (normalized_task_index,)
        explainer_task_count = 1
    else:
        task_indices = tuple(range(predictor.num_tasks))
        explainer_task_count = predictor.num_tasks
    normalized_foreground_ids = _normalize_ids(
        "foreground_ids", foreground_ids, expected_length=batch_size
    )
    normalized_background_ids = _normalize_ids(
        "background_ids",
        background_ids,
        expected_length=int(background.shape[0]),
    )

    with _preserve_training_states(predictor):
        with torch.no_grad():
            all_full_predictions = _validate_prediction_tensor(
                "full predictions",
                predictor(foreground),
                batch_size=batch_size,
            )
            all_background_predictions = _validate_prediction_tensor(
                "background predictions",
                predictor(background),
                batch_size=int(background.shape[0]),
            )
            if task_index is None:
                full_tensor = all_full_predictions
                background_predictions = all_background_predictions
            else:
                full_tensor = all_full_predictions[
                    :, normalized_task_index : normalized_task_index + 1
                ]
                background_predictions = all_background_predictions[
                    :, normalized_task_index : normalized_task_index + 1
                ]
            base_tensor = background_predictions.mean(dim=0)

        if task_index is None:
            explainer_model: nn.Module = predictor
        else:
            class _SelectedTaskPredictor(nn.Module):
                def __init__(
                    self,
                    base_predictor: SemanticFusionPredictor,
                    selected_task: int,
                ) -> None:
                    super().__init__()
                    self.base_predictor = base_predictor
                    self.selected_task = selected_task

                def forward(self, blocks: Tensor) -> Tensor:
                    predictions = self.base_predictor(blocks)
                    return predictions[
                        :, self.selected_task : self.selected_task + 1
                    ]

            explainer_model = _SelectedTaskPredictor(
                predictor, normalized_task_index
            )

        explainer = shap.GradientExplainer(explainer_model, background)
        raw_values = explainer.shap_values(foreground)
        normalized_values = _normalize_gradient_shap_values(
            raw_values,
            batch_size=batch_size,
            task_count=explainer_task_count,
            group_count=group_count,
            feature_dim=feature_dim,
        )

    selected_values = _readonly_numpy(normalized_values)
    full = _readonly_numpy(full_tensor)
    base = _readonly_numpy(base_tensor)
    residual = _readonly_numpy(
        full - (base[np.newaxis, :] + selected_values.sum(axis=(2, 3)))
    )
    return FeatureSHAPResult(
        shap_values=selected_values,
        base=base,
        full=full,
        additivity_residual=residual,
        task_indices=task_indices,
        group_names=predictor.group_names,
        output_space=predictor.output_space,
        foreground_ids=normalized_foreground_ids,
        background_ids=normalized_background_ids,
    )


def _stable_source_id_key(value: Hashable) -> tuple[str, object]:
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return ("number", float(value))
    return (
        f"{type(value).__module__}.{type(value).__qualname__}",
        repr(value),
    )


def _prediction_scores_for_task(
    prediction_scores: object,
    *,
    sample_count: int,
    task_index: int,
    selected_task_position: int,
    selected_task_count: int,
) -> np.ndarray:
    scores = _readonly_numpy(prediction_scores)
    if not np.issubdtype(scores.dtype, np.floating):
        raise TypeError("prediction_scores must use a real floating-point dtype")
    if scores.ndim == 1:
        if scores.shape[0] != sample_count:
            raise ValueError(
                f"prediction_scores must contain {sample_count} rows"
            )
        selected = scores
    elif scores.ndim == 2 and scores.shape[0] == sample_count:
        if scores.shape[1] == selected_task_count:
            selected = scores[:, selected_task_position]
        elif task_index < scores.shape[1]:
            selected = scores[:, task_index]
        else:
            raise ValueError(
                "prediction_scores task axis does not contain the requested "
                f"task_index={task_index}"
            )
    else:
        raise ValueError(
            "prediction_scores must have shape [samples] or [samples, tasks]"
        )
    if not bool(np.isfinite(selected).all()):
        raise ValueError("prediction_scores contain NaN or infinity")
    return _readonly_numpy(selected)


def summarize_shap_values(
    result: FeatureSHAPResult,
    prediction_scores: object,
    source_ids: Sequence[Hashable],
    *,
    task_index: int,
    top_prediction_count: int = 30,
    heatmap_molecule_count: int = 20,
    top_feature_count: int = 50,
) -> SHAPSummary:
    """Summarize one task's latent-feature SHAP values for paper figures.

    The returned arrays contain figure-ready data only; this function neither
    plots nor writes files.
    """

    if not isinstance(result, FeatureSHAPResult):
        raise TypeError("result must be a FeatureSHAPResult")
    if not isinstance(task_index, Integral) or isinstance(task_index, bool):
        raise TypeError("task_index must be an integer")
    normalized_task_index = int(task_index)
    if normalized_task_index not in result.task_indices:
        raise ValueError(
            f"task_index={normalized_task_index} is not present in "
            f"result.task_indices={result.task_indices}"
        )
    local_task_index = result.task_indices.index(normalized_task_index)
    top_count = _positive_integer(
        "top_prediction_count", top_prediction_count
    )
    heatmap_count = _positive_integer(
        "heatmap_molecule_count", heatmap_molecule_count
    )
    feature_count = _positive_integer("top_feature_count", top_feature_count)

    values = np.asarray(result.shap_values)
    if values.ndim != 4:
        raise ValueError(
            "result.shap_values must have shape [samples, tasks, groups, "
            "features]"
        )
    sample_count, selected_task_count, group_count, feature_dim = values.shape
    if sample_count <= 0:
        raise ValueError("result.shap_values must contain at least one sample")
    if selected_task_count != len(result.task_indices):
        raise ValueError(
            "result.shap_values task axis must match result.task_indices"
        )
    group_names = _normalize_group_names(
        result.group_names, expected_count=group_count
    )
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("result.shap_values must be real floating point")
    if not bool(np.isfinite(values).all()):
        raise ValueError("result.shap_values contain NaN or infinity")

    normalized_source_ids = _normalize_ids(
        "source_ids", source_ids, expected_length=sample_count
    )
    if normalized_source_ids != result.foreground_ids:
        raise ValueError(
            "source_ids must exactly match result.foreground_ids in order"
        )
    scores = _prediction_scores_for_task(
        prediction_scores,
        sample_count=sample_count,
        task_index=normalized_task_index,
        selected_task_position=local_task_index,
        selected_task_count=selected_task_count,
    )

    ordered_indices = sorted(
        range(sample_count),
        key=lambda index: (
            -float(scores[index]),
            _stable_source_id_key(normalized_source_ids[index]),
        ),
    )
    selected_top_count = min(top_count, sample_count)
    top_indices = ordered_indices[:selected_top_count]
    task_values = values[:, local_task_index, :, :]
    top_values = task_values[top_indices]
    mean_absolute_group = np.abs(top_values).sum(axis=-1).mean(axis=0)
    total_group_contribution = float(mean_absolute_group.sum())
    if total_group_contribution == 0.0:
        proportions = np.full(group_count, np.nan, dtype=np.float64)
    else:
        proportions = mean_absolute_group / total_group_contribution

    selected_heatmap_count = min(heatmap_count, selected_top_count)
    heatmap_indices = top_indices[:selected_heatmap_count]
    heatmap_source_values = task_values[heatmap_indices]
    mean_absolute_features = np.abs(heatmap_source_values).mean(axis=0)
    flat_feature_indices = sorted(
        range(group_count * feature_dim),
        key=lambda index: (
            -float(mean_absolute_features.reshape(-1)[index]),
            index // feature_dim,
            index % feature_dim,
        ),
    )
    selected_feature_count = min(feature_count, group_count * feature_dim)
    flat_feature_indices = flat_feature_indices[:selected_feature_count]
    feature_keys = tuple(
        (group_names[index // feature_dim], index % feature_dim)
        for index in flat_feature_indices
    )
    heatmap = np.stack(
        [
            heatmap_source_values[
                :, index // feature_dim, index % feature_dim
            ]
            for index in flat_feature_indices
        ],
        axis=1,
    )

    return SHAPSummary(
        task_index=normalized_task_index,
        group_names=group_names,
        top_prediction_sample_ids=tuple(
            normalized_source_ids[index] for index in top_indices
        ),
        top_prediction_scores=_readonly_numpy(scores[top_indices]),
        mean_absolute_group_contributions=_readonly_numpy(
            mean_absolute_group
        ),
        group_contribution_proportions=_readonly_numpy(proportions),
        heatmap=_readonly_numpy(heatmap),
        heatmap_feature_keys=feature_keys,
        heatmap_sample_ids=tuple(
            normalized_source_ids[index] for index in heatmap_indices
        ),
    )


__all__ = [
    "FeatureSHAPResult",
    "GroupShapleyResult",
    "ModalityOcclusionResult",
    "SHAPSummary",
    "SemanticFeatureBatch",
    "SemanticFusionPredictor",
    "exact_group_shapley",
    "extract_semantic_feature_batch",
    "gradient_feature_shap",
    "modality_occlusion",
    "summarize_shap_values",
]

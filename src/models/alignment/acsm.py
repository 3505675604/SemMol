"""Anchor-centers soft matching in the shared molecular embedding space."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_WEIGHTING_MODES: Final[frozenset[str]] = frozenset(
    {"softmax", "uniform", "top1"}
)
_NEGATIVE_SELECTION_MODES: Final[frozenset[str]] = frozenset(
    {"debiased", "hard", "all"}
)
@dataclass(frozen=True)
class ModalityMatch:
    """ACSM retrieval result for one target modality.

    Loss code must apply ``negative_mask`` and restrict reductions to
    ``valid_negative_rows``. A threshold can legitimately leave one anchor
    without any trusted negatives; its masked logits are all ``-inf`` and must
    not be passed to an unmasked softmax or mean.
    """

    positive_embedding: Tensor
    positive_indices: Tensor
    positive_similarities: Tensor
    positive_weights: Tensor
    all_similarities: Tensor
    negative_similarities: Tensor
    negative_mask: Tensor
    negative_count: Tensor
    valid_negative_rows: Tensor


@dataclass(frozen=True)
class ACSMOutput:
    """Complete one-to-many matching result for all target modalities."""

    anchor_embedding: Tensor
    positive_embedding: Tensor
    modality_matches: dict[str, ModalityMatch]
    target_modalities: tuple[str, ...]


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive, got {normalized}")
    return normalized


def _finite_real(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _choice(name: str, value: object, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise ValueError(
            f"unsupported {name}={value!r}; expected one of {sorted(allowed)}"
        )
    return normalized


class AnchorCentersSoftMatching(nn.Module):
    """Retrieve and fuse multiple semantic centers for each anchor.

    ``negative_selection='hard'`` follows the threshold direction in the
    manuscript's Equation 9 and retains non-positive centers whose similarity
    is at least the denoising threshold. ``'debiased'`` provides the opposite
    ablation: it removes those ambiguous high-similarity candidates.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_retrieve: int = 16,
        temperature: float = 0.07,
        denoise_threshold: float = 0.5,
        *,
        learnable_temperature: bool = False,
        weighting: str = "softmax",
        negative_selection: str = "hard",
        max_negatives: int | None = 10,
        eps: float = 1.0e-8,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_integer("feature_dim", feature_dim)
        self.num_retrieve = _positive_integer(
            "num_retrieve", num_retrieve
        )
        initial_temperature = _finite_real("temperature", temperature)
        if initial_temperature <= 0.0:
            raise ValueError("temperature must be positive")
        threshold = _finite_real("denoise_threshold", denoise_threshold)
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(
                "denoise_threshold must be a cosine similarity in [-1, 1]"
            )
        epsilon = _finite_real("eps", eps)
        if epsilon <= 0.0:
            raise ValueError("eps must be positive")
        if not isinstance(learnable_temperature, bool):
            raise TypeError("learnable_temperature must be bool")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")
        if max_negatives is not None:
            max_negatives = _positive_integer(
                "max_negatives", max_negatives
            )

        self.weighting = _choice(
            "weighting", weighting, _WEIGHTING_MODES
        )
        self.negative_selection = _choice(
            "negative_selection",
            negative_selection,
            _NEGATIVE_SELECTION_MODES,
        )
        self.denoise_threshold = threshold
        self.max_negatives = max_negatives
        self.eps = epsilon
        self.validate_values = validate_values
        self.learnable_temperature = learnable_temperature

        if self.learnable_temperature:
            if self.weighting != "softmax":
                raise ValueError(
                    "learnable_temperature requires weighting='softmax'"
                )
            positive_offset = initial_temperature - self.eps
            if positive_offset <= 0.0:
                raise ValueError(
                    "a learnable temperature must be greater than eps"
                )
            if positive_offset > 20.0:
                initial_raw_temperature = positive_offset
            else:
                initial_raw_temperature = math.log(
                    math.expm1(positive_offset)
                )
            self.raw_temperature = nn.Parameter(
                torch.tensor(initial_raw_temperature, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "fixed_temperature",
                torch.tensor(initial_temperature, dtype=torch.float32),
            )

    @property
    def temperature(self) -> Tensor:
        """Positive scalar temperature used by positive-center retrieval."""

        if self.learnable_temperature:
            return F.softplus(self.raw_temperature.float()) + self.eps
        return self.fixed_temperature.float()

    def _validate_anchor(self, anchor: Tensor) -> Tensor:
        if not isinstance(anchor, Tensor):
            raise TypeError("anchor must be a torch.Tensor")
        if anchor.ndim != 2:
            raise ValueError(
                "anchor must have shape [batch, feature_dim], got "
                f"{tuple(anchor.shape)}"
            )
        if anchor.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected anchor feature_dim={self.feature_dim}, "
                f"got {anchor.shape[1]}"
            )
        if not anchor.is_floating_point():
            raise TypeError(
                f"anchor must be floating point, got {anchor.dtype}"
            )
        prepared = anchor.float()
        if (
            self.validate_values
            and prepared.numel() > 0
            and not bool(torch.isfinite(prepared).all())
        ):
            raise ValueError("anchor contains NaN or infinite values")
        return F.normalize(
            prepared,
            p=2.0,
            dim=-1,
            eps=self.eps,
        )

    def _validate_centers(
        self,
        modality: str,
        centers: Tensor,
        *,
        device: torch.device,
    ) -> Tensor:
        if not isinstance(modality, str) or not modality.strip():
            raise ValueError("target modality names must be non-empty strings")
        if not isinstance(centers, Tensor):
            raise TypeError(f"centers for {modality!r} must be a Tensor")
        if centers.ndim != 2:
            raise ValueError(
                f"centers for {modality!r} must have shape "
                f"[num_centers, {self.feature_dim}], got {tuple(centers.shape)}"
            )
        if centers.shape[1] != self.feature_dim:
            raise ValueError(
                f"centers for {modality!r} use feature_dim={centers.shape[1]}, "
                f"expected {self.feature_dim}"
            )
        if centers.shape[0] < self.num_retrieve:
            raise ValueError(
                f"num_retrieve={self.num_retrieve} exceeds the "
                f"{modality!r} center count {centers.shape[0]}"
            )
        if not centers.is_floating_point():
            raise TypeError(
                f"centers for {modality!r} must be floating point"
            )
        if centers.device != device:
            raise ValueError(
                f"anchor and {modality!r} centers must share a device: "
                f"{device} != {centers.device}"
            )
        prepared = centers.detach().float()
        if (
            self.validate_values
            and prepared.numel() > 0
            and not bool(torch.isfinite(prepared).all())
        ):
            raise ValueError(
                f"centers for {modality!r} contain NaN or infinite values"
            )
        return F.normalize(
            prepared,
            p=2.0,
            dim=-1,
            eps=self.eps,
        )

    def _positive_weights(self, similarities: Tensor) -> Tensor:
        if self.weighting == "softmax":
            return F.softmax(
                similarities / self.temperature,
                dim=-1,
            )
        if self.weighting == "uniform":
            return torch.full_like(
                similarities,
                1.0 / float(self.num_retrieve),
            )
        weights = torch.zeros_like(similarities)
        if weights.shape[0] > 0:
            weights[:, 0] = 1.0
        return weights

    def _negative_mask(
        self,
        similarities: Tensor,
        positive_indices: Tensor,
    ) -> Tensor:
        candidates = torch.ones_like(similarities, dtype=torch.bool)
        candidates.scatter_(1, positive_indices, False)
        if self.negative_selection == "debiased":
            candidates &= similarities < self.denoise_threshold
        elif self.negative_selection == "hard":
            candidates &= similarities >= self.denoise_threshold

        if self.max_negatives is None:
            return candidates
        selected_count = min(self.max_negatives, int(similarities.shape[1]))
        candidate_scores = similarities.masked_fill(
            ~candidates,
            float("-inf"),
        )
        selected_scores, selected_indices = torch.topk(
            candidate_scores,
            k=selected_count,
            dim=-1,
        )
        selected_valid = torch.isfinite(selected_scores)
        limited = torch.zeros_like(candidates)
        limited.scatter_(1, selected_indices, selected_valid)
        return limited

    def match_modality(
        self,
        anchor: Tensor,
        centers: Tensor,
        *,
        modality: str,
        anchor_is_normalized: bool = False,
    ) -> ModalityMatch:
        """Match one anchor batch against one target center library."""

        if not isinstance(anchor_is_normalized, bool):
            raise TypeError("anchor_is_normalized must be bool")
        if anchor_is_normalized:
            normalized_anchor = anchor
            if anchor.ndim != 2 or anchor.shape[1] != self.feature_dim:
                raise ValueError(
                    "normalized anchor has an incompatible shape"
                )
            if not anchor.is_floating_point():
                raise TypeError("normalized anchor must be floating point")
            normalized_anchor = F.normalize(
                anchor.float(),
                p=2.0,
                dim=-1,
                eps=self.eps,
            )
            if (
                self.validate_values
                and normalized_anchor.numel() > 0
                and not bool(torch.isfinite(normalized_anchor).all())
            ):
                raise ValueError("normalized anchor contains non-finite values")
        else:
            normalized_anchor = self._validate_anchor(anchor)
        normalized_centers = self._validate_centers(
            modality,
            centers,
            device=normalized_anchor.device,
        )

        similarities = normalized_anchor @ normalized_centers.transpose(0, 1)
        positive_similarities, positive_indices = torch.topk(
            similarities,
            k=self.num_retrieve,
            dim=-1,
        )
        positive_weights = self._positive_weights(positive_similarities)
        retrieved_centers = normalized_centers[positive_indices]
        positive_embedding = torch.sum(
            positive_weights.unsqueeze(-1) * retrieved_centers,
            dim=1,
        )

        negative_mask = self._negative_mask(
            similarities,
            positive_indices,
        )
        negative_similarities = similarities.masked_fill(
            ~negative_mask,
            float("-inf"),
        )
        negative_count = negative_mask.sum(dim=-1)
        return ModalityMatch(
            positive_embedding=positive_embedding,
            positive_indices=positive_indices,
            positive_similarities=positive_similarities,
            positive_weights=positive_weights,
            all_similarities=similarities,
            negative_similarities=negative_similarities,
            negative_mask=negative_mask,
            negative_count=negative_count,
            valid_negative_rows=negative_count > 0,
        )

    def forward(
        self,
        anchor: Tensor,
        centers_by_modality: Mapping[str, Tensor],
        *,
        anchor_modality: str,
    ) -> ACSMOutput:
        if not isinstance(centers_by_modality, Mapping):
            raise TypeError("centers_by_modality must be a mapping")
        if not centers_by_modality:
            raise ValueError("at least one target center library is required")
        if not isinstance(anchor_modality, str) or not anchor_modality.strip():
            raise ValueError("anchor_modality must be a non-empty string")
        normalized_anchor_name = anchor_modality.strip().lower()
        normalized_target_names = {
            str(name).strip().lower() for name in centers_by_modality
        }
        if normalized_anchor_name in normalized_target_names:
            raise ValueError(
                "centers_by_modality must exclude the anchor modality"
            )

        normalized_anchor = self._validate_anchor(anchor)
        modality_matches: dict[str, ModalityMatch] = {}
        positive_embeddings: list[Tensor] = []
        target_modalities: list[str] = []
        normalized_names: set[str] = set()
        for raw_modality, centers in centers_by_modality.items():
            if not isinstance(raw_modality, str) or not raw_modality.strip():
                raise ValueError(
                    "target modality names must be non-empty strings"
                )
            modality = raw_modality.strip().lower()
            if modality in normalized_names:
                raise ValueError(
                    f"duplicate normalized target modality {modality!r}"
                )
            normalized_names.add(modality)
            match = self.match_modality(
                normalized_anchor,
                centers,
                modality=modality,
                anchor_is_normalized=True,
            )
            modality_matches[modality] = match
            positive_embeddings.append(match.positive_embedding)
            target_modalities.append(modality)

        stacked = torch.stack(positive_embeddings, dim=0)
        fused_positive = stacked.sum(dim=0)
        fused_positive = F.normalize(
            fused_positive,
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        return ACSMOutput(
            anchor_embedding=normalized_anchor,
            positive_embedding=fused_positive,
            modality_matches=modality_matches,
            target_modalities=tuple(target_modalities),
        )


__all__ = [
    "ACSMOutput",
    "AnchorCentersSoftMatching",
    "ModalityMatch",
]

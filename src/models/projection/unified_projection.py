"""MLP projection into the shared multimodal representation space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ProjectionOutput:
    """Raw and numerically stable normalized shared-space embeddings."""

    raw: Tensor
    normalized: Tensor


def normalize_projection(features: Tensor, eps: float) -> Tensor:
    """L2-normalize in float32 so mixed-precision training remains stable."""

    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    return F.normalize(features.float(), p=2.0, dim=-1, eps=eps)


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    normalized_name = name.strip().lower()
    activations: dict[str, Callable[[], nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    if normalized_name not in activations:
        allowed = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation {name!r}; expected one of: {allowed}")
    return activations[normalized_name]


class UnifiedProjectionMLP(nn.Module):
    """Independent per-modality MLP for a common-dimensional embedding space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        *,
        hidden_dim: int = 512,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 1.0e-5,
        normalize_eps: float = 1.0e-8,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if layer_norm_eps <= 0.0:
            raise ValueError("layer_norm_eps must be positive")
        if normalize_eps <= 0.0:
            raise ValueError("normalize_eps must be positive")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_eps = normalize_eps
        self.validate_values = validate_values

        activation_factory = _activation_factory(activation)
        if num_layers == 1:
            modules: list[nn.Module] = [nn.Linear(input_dim, output_dim, bias=bias)]
        else:
            modules = []
            current_dim = input_dim
            for _ in range(num_layers - 1):
                modules.extend(
                    [
                        nn.Linear(current_dim, hidden_dim, bias=bias),
                        nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
                        activation_factory(),
                        nn.Dropout(dropout),
                    ]
                )
                current_dim = hidden_dim
            modules.append(nn.Linear(current_dim, output_dim, bias=bias))
        self.network = nn.Sequential(*modules)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: Tensor) -> ProjectionOutput:
        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [compact_batch, input_dim], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {features.shape[1]}"
            )
        if not features.is_floating_point():
            raise TypeError(f"features must be floating point, got {features.dtype}")
        if (
            self.validate_values
            and features.numel() > 0
            and not bool(torch.isfinite(features).all())
        ):
            raise ValueError("features contain NaN or infinite values")

        raw = self.network(features)
        normalized = normalize_projection(raw, self.normalize_eps)
        return ProjectionOutput(raw=raw, normalized=normalized)

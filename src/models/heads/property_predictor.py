"""Task-specific molecular property prediction head."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Callable, Final

import torch
from torch import Tensor, nn


_TASK_TYPES: Final[frozenset[str]] = frozenset(
    {"classification", "regression"}
)


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive, got {normalized}")
    return normalized


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("activation must be a non-empty string")
    activations: dict[str, Callable[[], nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    normalized = name.strip().lower()
    if normalized not in activations:
        raise ValueError(
            f"unsupported activation={name!r}; expected one of "
            f"{sorted(activations)}"
        )
    return activations[normalized]


class PropertyPredictor(nn.Module):
    """Predict raw multitask logits or continuous molecular properties.

    The head deliberately uses LayerNorm instead of BatchNorm so batches of
    size one, uneven DDP shards, and small scaffold-split datasets remain
    valid. Classification outputs are logits; sigmoid belongs in metrics or
    inference code, while training should use BCEWithLogitsLoss with the
    dataset's label mask.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_tasks: int = 1,
        *,
        hidden_dims: Sequence[int] = (256, 128),
        task_type: str = "classification",
        activation: str = "gelu",
        dropout: float = 0.2,
        layer_norm_eps: float = 1.0e-5,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer("input_dim", input_dim)
        self.num_tasks = _positive_integer("num_tasks", num_tasks)
        if isinstance(hidden_dims, (str, bytes)) or not isinstance(
            hidden_dims, Sequence
        ):
            raise TypeError("hidden_dims must be a sequence of positive integers")
        normalized_hidden = tuple(
            _positive_integer(f"hidden_dims[{index}]", value)
            for index, value in enumerate(hidden_dims)
        )
        if not normalized_hidden:
            raise ValueError("hidden_dims must contain at least one dimension")
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("task_type must be a non-empty string")
        normalized_task = task_type.strip().lower()
        if normalized_task not in _TASK_TYPES:
            raise ValueError(
                f"unsupported task_type={task_type!r}; expected one of "
                f"{sorted(_TASK_TYPES)}"
            )
        if not isinstance(dropout, Real) or isinstance(dropout, bool):
            raise TypeError("dropout must be a real number")
        normalized_dropout = float(dropout)
        if not math.isfinite(normalized_dropout) or not (
            0.0 <= normalized_dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(layer_norm_eps, Real) or isinstance(
            layer_norm_eps, bool
        ):
            raise TypeError("layer_norm_eps must be a real number")
        normalized_eps = float(layer_norm_eps)
        if not math.isfinite(normalized_eps) or normalized_eps <= 0.0:
            raise ValueError("layer_norm_eps must be positive and finite")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")

        self.hidden_dims = normalized_hidden
        self.task_type = normalized_task
        self.validate_values = validate_values
        activation_factory = _activation_factory(activation)

        modules: list[nn.Module] = []
        current_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            modules.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim, eps=normalized_eps),
                    activation_factory(),
                    nn.Dropout(normalized_dropout),
                ]
            )
            current_dim = hidden_dim
        modules.append(nn.Linear(current_dim, self.num_tasks))
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

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 2:
            raise ValueError(
                "features must have shape [batch, input_dim], got "
                f"{tuple(features.shape)}"
            )
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, got {features.shape[1]}"
            )
        if not features.is_floating_point():
            raise TypeError(
                f"features must be floating point, got {features.dtype}"
            )
        if (
            self.validate_values
            and features.numel() > 0
            and not bool(torch.isfinite(features).all())
        ):
            raise ValueError("features contain NaN or infinite values")
        outputs = self.network(features)
        if (
            self.validate_values
            and outputs.numel() > 0
            and not bool(torch.isfinite(outputs).all())
        ):
            raise FloatingPointError(
                "property predictor produced NaN or infinite values"
            )
        return outputs

    def predict(self, features: Tensor) -> Tensor:
        """Return probabilities for classification and values for regression."""

        outputs = self(features)
        if self.task_type == "classification":
            return torch.sigmoid(outputs)
        return outputs


__all__ = ["PropertyPredictor"]

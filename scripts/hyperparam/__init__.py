"""SemMol hyperparameter search package.

Provides grid search, random search, sensitivity analysis, and automated
reporting for the DCL/ACSM/training hyperparameters referenced in the
SemMol manuscript and reviewer response.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GridAxis:
    """One dimension of a hyperparameter search space."""

    path: str
    values: list[Any]
    value_type: str = "auto"

    def __post_init__(self) -> None:
        if not self.path or not self.path.strip():
            raise ValueError("axis path must be a non-empty string")
        if not self.values:
            raise ValueError(
                f"axis {self.path!r} must have at least one value"
            )
        if self.value_type not in {"auto", "int", "float", "choice", "bool"}:
            raise ValueError(
                f"axis {self.path!r} value_type must be one of "
                f"auto, int, float, choice, bool"
            )


@dataclass(frozen=True)
class GridDefinition:
    """A complete grid or random search specification."""

    name: str
    description: str
    axes: tuple[GridAxis, ...]
    search_strategy: str = "grid"
    constraints: tuple[dict[str, str], ...] = ()
    evaluation_mode: str = "pretrain"
    fast_epochs: int = 10
    metrics: tuple[str, ...] = ("train_loss",)
    direction: str = "minimize"
    seed: int = 3407
    max_trials: int | None = None
    random_trials: int = 50

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("grid name must be non-empty")
        if self.search_strategy not in {"grid", "random"}:
            raise ValueError("search_strategy must be 'grid' or 'random'")
        if self.evaluation_mode not in {"pretrain", "finetune"}:
            raise ValueError(
                "evaluation.mode must be 'pretrain' or 'finetune'"
            )
        if self.fast_epochs < 1:
            raise ValueError("fast_epochs must be at least 1")
        _DIRECTION_ALIASES = {
            "min": "minimize", "minimize": "minimize",
            "max": "maximize", "maximize": "maximize",
        }
        direction = _DIRECTION_ALIASES.get(self.direction)
        if direction is None:
            raise ValueError(
                f"evaluation.direction must be 'minimize' or 'maximize', "
                f"got {self.direction!r}"
            )
        object.__setattr__(self, "direction", direction)
        if not self.metrics:
            raise ValueError("evaluation.metrics must not be empty")


@dataclass(frozen=True)
class TrialSpec:
    """A single trial configuration ready for execution."""

    trial_index: int
    overrides: dict[str, Any]
    output_dir: Path
    config_path: Path


@dataclass(frozen=True)
class TrialResult:
    """Aggregated result from one completed trial."""

    trial_index: int
    grid_values: dict[str, Any]
    status: str
    metrics: dict[str, float]
    best_epoch: int | None
    wall_time_seconds: float
    error_message: str | None
    config_path: Path
    output_dir: Path


__all__ = [
    "GridAxis",
    "GridDefinition",
    "TrialResult",
    "TrialSpec",
]
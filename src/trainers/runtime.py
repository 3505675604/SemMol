"""Strict runtime construction shared by the training entry points."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from .checkpointing import configuration_fingerprint


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUIRED_DERIVED_VALUES = frozenset(
    {
        "train_loader_batch_count",
        "optimizer_steps_per_epoch",
        "total_optimizer_steps",
    }
)


@dataclass(frozen=True)
class LoadedExperimentConfiguration:
    """A validated experiment mapping and its filesystem anchors."""

    path: Path
    project_root: Path
    values: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("path must be an absolute pathlib.Path")
        if not isinstance(self.project_root, Path) or not self.project_root.is_absolute():
            raise ValueError("project_root must be an absolute pathlib.Path")
        if not isinstance(self.values, dict):
            raise TypeError("values must be a dictionary")
        if any(not isinstance(key, str) for key in self.values):
            raise TypeError("configuration keys must be strings")

    def section(self, name: str, *, required: bool = True) -> dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise ValueError("section name must be a non-empty string")
        if name not in self.values:
            if required:
                raise KeyError(f"configuration section {name!r} is required")
            return {}
        value = self.values[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"configuration section {name!r} must be a mapping")
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"configuration section {name!r} has non-string keys")
        return dict(value)

    def resolve_path(self, value: str | Path, *, name: str) -> Path:
        return resolve_project_path(value, project_root=self.project_root, name=name)


def _find_project_root(configuration_path: Path) -> Path:
    for candidate in (configuration_path.parent, *configuration_path.parents):
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate.resolve()
    raise RuntimeError(
        "cannot locate the SemMol project root above configuration path "
        f"{configuration_path}"
    )


def load_experiment_configuration(
    path: str | Path,
    *,
    expected_mode: str,
) -> LoadedExperimentConfiguration:
    """Load one YAML file and reject a mismatched experiment mode."""

    if not isinstance(path, (str, Path)):
        raise TypeError("configuration path must be a string or pathlib.Path")
    if not isinstance(expected_mode, str) or not expected_mode.strip():
        raise ValueError("expected_mode must be a non-empty string")
    normalized_mode = expected_mode.strip().lower()
    configuration_path = Path(path).expanduser().resolve()
    if not configuration_path.is_file():
        raise FileNotFoundError(
            f"configuration file does not exist: {configuration_path}"
        )
    with configuration_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("experiment YAML must contain a top-level mapping")
    if any(not isinstance(key, str) for key in payload):
        raise TypeError("experiment YAML keys must be strings")
    values = dict(payload)
    experiment = values.get("experiment")
    if not isinstance(experiment, Mapping):
        raise TypeError("configuration section 'experiment' must be a mapping")
    configured_mode = experiment.get("mode")
    if not isinstance(configured_mode, str) or not configured_mode.strip():
        raise ValueError("experiment.mode must be a non-empty string")
    if configured_mode.strip().lower() != normalized_mode:
        raise ValueError(
            f"expected experiment.mode={normalized_mode!r}, got {configured_mode!r}"
        )
    return LoadedExperimentConfiguration(
        path=configuration_path,
        project_root=_find_project_root(configuration_path),
        values=values,
    )


def resolve_project_path(
    value: str | Path,
    *,
    project_root: Path,
    name: str,
) -> Path:
    if not isinstance(project_root, Path) or not project_root.is_absolute():
        raise ValueError("project_root must be an absolute pathlib.Path")
    if not isinstance(name, str) or not name:
        raise ValueError("path name must be a non-empty string")
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a string or pathlib.Path")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} cannot be empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return normalized


def require_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = normalized < minimum if minimum_inclusive else normalized <= minimum
        if invalid:
            relation = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return normalized


def require_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def require_string_sequence(name: str, value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(require_string(f"{name}[{index}]", item) for index, item in enumerate(value))
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def optimizer_steps_per_epoch(batch_count: int, accumulation_steps: int) -> int:
    normalized_batches = require_int("batch_count", batch_count, minimum=1)
    normalized_accumulation = require_int(
        "accumulation_steps",
        accumulation_steps,
        minimum=1,
    )
    return math.ceil(normalized_batches / normalized_accumulation)


def build_optimizer(
    model: nn.Module,
    options: Mapping[str, Any],
) -> Optimizer:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(options, Mapping):
        raise TypeError("optimizer options must be a mapping")
    if any(not isinstance(key, str) for key in options):
        raise TypeError("optimizer option keys must be strings")
    unknown = set(options) - {"type", "lr", "weight_decay", "betas", "eps", "amsgrad"}
    if unknown:
        raise ValueError(f"unsupported optimizer options: {sorted(unknown)}")
    optimizer_type = require_string("optimizer.type", options.get("type", "adamw")).lower()
    if optimizer_type != "adamw":
        raise ValueError("only optimizer.type='adamw' is supported")
    learning_rate = require_real(
        "optimizer.lr",
        options.get("lr"),
        minimum=0.0,
        minimum_inclusive=False,
    )
    weight_decay = require_real(
        "optimizer.weight_decay",
        options.get("weight_decay", 0.0),
        minimum=0.0,
    )
    eps = require_real(
        "optimizer.eps",
        options.get("eps", 1.0e-8),
        minimum=0.0,
        minimum_inclusive=False,
    )
    raw_betas = options.get("betas", (0.9, 0.999))
    if isinstance(raw_betas, (str, bytes)) or not isinstance(raw_betas, Sequence):
        raise TypeError("optimizer.betas must be a two-item real sequence")
    if len(raw_betas) != 2:
        raise ValueError("optimizer.betas must contain exactly two values")
    betas = tuple(
        require_real(
            f"optimizer.betas[{index}]",
            beta,
            minimum=0.0,
            maximum=1.0,
        )
        for index, beta in enumerate(raw_betas)
    )
    if betas[0] >= 1.0 or betas[1] >= 1.0:
        raise ValueError("optimizer beta values must be smaller than 1")
    amsgrad = require_bool("optimizer.amsgrad", options.get("amsgrad", False))
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return AdamW(
        parameters,
        lr=learning_rate,
        betas=(betas[0], betas[1]),
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )


def build_scheduler(
    optimizer: Optimizer,
    options: Mapping[str, Any] | None,
    *,
    total_optimizer_steps: int,
) -> LRScheduler | None:
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    total_steps = require_int(
        "total_optimizer_steps",
        total_optimizer_steps,
        minimum=1,
    )
    if options is None:
        return None
    if not isinstance(options, Mapping):
        raise TypeError("scheduler options must be a mapping or None")
    if any(not isinstance(key, str) for key in options):
        raise TypeError("scheduler option keys must be strings")
    unknown = set(options) - {"type", "warmup_ratio", "min_lr"}
    if unknown:
        raise ValueError(f"unsupported scheduler options: {sorted(unknown)}")
    scheduler_type = require_string("scheduler.type", options.get("type", "cosine")).lower()
    if scheduler_type == "none":
        if set(options) - {"type"}:
            raise ValueError("scheduler.type='none' cannot have additional options")
        return None
    if scheduler_type != "cosine":
        raise ValueError("only scheduler.type='cosine' or 'none' is supported")
    warmup_ratio = require_real(
        "scheduler.warmup_ratio",
        options.get("warmup_ratio", 0.0),
        minimum=0.0,
        maximum=1.0,
    )
    if warmup_ratio >= 1.0:
        raise ValueError("scheduler.warmup_ratio must be smaller than 1")
    min_lr = require_real(
        "scheduler.min_lr",
        options.get("min_lr", 0.0),
        minimum=0.0,
    )
    if not optimizer.param_groups:
        raise ValueError("optimizer must contain at least one parameter group")
    base_lrs = tuple(
        require_real(
            f"optimizer.param_groups[{index}].lr",
            group.get("lr"),
            minimum=0.0,
            minimum_inclusive=False,
        )
        for index, group in enumerate(optimizer.param_groups)
    )
    if any(min_lr > base_lr for base_lr in base_lrs):
        raise ValueError("scheduler.min_lr cannot exceed an optimizer base learning rate")
    warmup_steps = 0
    if warmup_ratio > 0.0:
        warmup_steps = max(1, int(math.floor(total_steps * warmup_ratio)))

    def schedule_for(base_lr: float):
        floor = min_lr / base_lr

        def factor(step: int) -> float:
            update = min(max(int(step) + 1, 1), total_steps)
            if warmup_steps and update <= warmup_steps:
                return float(update) / float(warmup_steps)
            if warmup_steps:
                decay_steps = total_steps - warmup_steps
                progress = float(update - warmup_steps) / float(decay_steps)
            elif total_steps == 1:
                progress = 0.0
            else:
                progress = float(update - 1) / float(total_steps - 1)
            progress = min(
                1.0,
                max(0.0, progress),
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + (1.0 - floor) * cosine

        return factor

    return LambdaLR(
        optimizer,
        lr_lambda=[schedule_for(base_lr) for base_lr in base_lrs],
    )


def training_configuration_fingerprint(
    configuration: Mapping[str, Any],
    *,
    resolved_model_configuration: Mapping[str, Any],
    artifact_fingerprints: Mapping[str, str],
    derived_values: Mapping[str, Any],
) -> str:
    """Hash all declared trajectory inputs while ignoring output locations.

    ``artifact_fingerprints`` maps stable artifact names to lowercase SHA-256
    content digests. ``derived_values`` must include
    ``train_loader_batch_count``, ``optimizer_steps_per_epoch``, and
    ``total_optimizer_steps`` so a resumed LambdaLR recreates the same closure.
    """

    if not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a mapping")
    if not isinstance(resolved_model_configuration, Mapping):
        raise TypeError("resolved_model_configuration must be a mapping")
    if not resolved_model_configuration:
        raise ValueError("resolved_model_configuration cannot be empty")
    if not isinstance(artifact_fingerprints, Mapping):
        raise TypeError("artifact_fingerprints must be a mapping")
    if not artifact_fingerprints:
        raise ValueError("artifact_fingerprints cannot be empty")
    normalized_artifacts: dict[str, str] = {}
    for name, fingerprint in artifact_fingerprints.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("artifact fingerprint names must be non-empty strings")
        normalized_name = name.strip()
        if normalized_name in normalized_artifacts:
            raise ValueError(
                f"duplicate normalized artifact fingerprint name {normalized_name!r}"
            )
        if not isinstance(fingerprint, str) or _SHA256_PATTERN.fullmatch(
            fingerprint
        ) is None:
            raise ValueError(
                f"artifact_fingerprints[{normalized_name!r}] must be a "
                "lowercase SHA-256 hexadecimal digest"
            )
        normalized_artifacts[normalized_name] = fingerprint
    if not isinstance(derived_values, Mapping):
        raise TypeError("derived_values must be a mapping")
    if any(not isinstance(key, str) for key in derived_values):
        raise TypeError("derived_values keys must be strings")
    missing_derived = _REQUIRED_DERIVED_VALUES - set(derived_values)
    if missing_derived:
        raise ValueError(
            "derived_values is missing required trajectory values: "
            f"{sorted(missing_derived)}"
        )
    normalized_derived = copy.deepcopy(dict(derived_values))
    for name in _REQUIRED_DERIVED_VALUES:
        normalized_derived[name] = require_int(
            f"derived_values.{name}",
            normalized_derived[name],
            minimum=1,
        )
    if (
        normalized_derived["optimizer_steps_per_epoch"]
        > normalized_derived["train_loader_batch_count"]
    ):
        raise ValueError(
            "derived_values.optimizer_steps_per_epoch cannot exceed "
            "train_loader_batch_count"
        )
    if (
        normalized_derived["total_optimizer_steps"]
        < normalized_derived["optimizer_steps_per_epoch"]
    ):
        raise ValueError(
            "derived_values.total_optimizer_steps cannot be smaller than "
            "optimizer_steps_per_epoch"
        )
    semantic = copy.deepcopy(dict(configuration))
    semantic.pop("output", None)
    experiment = semantic.get("experiment")
    if isinstance(experiment, Mapping):
        normalized_experiment = dict(experiment)
        normalized_experiment.pop("name", None)
        semantic["experiment"] = normalized_experiment
    payload = {
        "experiment_configuration": semantic,
        "resolved_model_configuration": copy.deepcopy(
            dict(resolved_model_configuration)
        ),
        "artifact_fingerprints": normalized_artifacts,
        "derived_values": normalized_derived,
    }
    return configuration_fingerprint(payload)


__all__ = [
    "LoadedExperimentConfiguration",
    "build_optimizer",
    "build_scheduler",
    "load_experiment_configuration",
    "optimizer_steps_per_epoch",
    "require_bool",
    "require_int",
    "require_real",
    "require_string",
    "require_string_sequence",
    "resolve_project_path",
    "training_configuration_fingerprint",
]

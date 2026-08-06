"""Resolve project YAML references into a validated SemMol build specification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .semmol import SemMol


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_KEYS = ("encoders", "projection", "dcl", "acsm")
_OPTIONAL_REFERENCE_KEYS = ("pretraining_heads",)
_MODEL_KEYS = {
    *_REFERENCE_KEYS,
    *_OPTIONAL_REFERENCE_KEYS,
    "modalities",
    "anchor_modality",
    "head",
    "freeze_encoders",
    "validate_values",
}
_MODALITIES = ("1d", "2d", "3d", "qm")


@dataclass(frozen=True)
class ResolvedSemMolConfig:
    """Resolved model inputs plus runner-owned checkpoint and DDP settings."""

    model_options: dict[str, Any]
    pretrained_checkpoint: Path | None
    dropped_modalities: tuple[str, ...]
    distributed_options: dict[str, Any]

    def build(self) -> SemMol:
        return SemMol(**self.model_options)


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _load_yaml(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a YAML mapping: {path}")
    return dict(payload)


def _infer_project_root(config_path: Path | None) -> Path:
    if config_path is not None:
        for candidate in (config_path.parent, *config_path.parents):
            if (
                (candidate / "src").is_dir()
                and (candidate / "configs").is_dir()
            ):
                return candidate.resolve()
    return _PROJECT_ROOT


def _resolve_path(value: str | Path, *, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _configuration_source(
    config: Mapping[str, Any] | str | Path,
    *,
    project_root: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(config, Mapping):
        root = (
            _PROJECT_ROOT
            if project_root is None
            else Path(project_root).expanduser().resolve()
        )
        return dict(config), root
    if not isinstance(config, (str, Path)):
        raise TypeError("config must be a mapping or YAML path")
    provisional_root = (
        _PROJECT_ROOT
        if project_root is None
        else Path(project_root).expanduser().resolve()
    )
    config_path = _resolve_path(config, project_root=provisional_root)
    root = (
        _infer_project_root(config_path)
        if project_root is None
        else provisional_root
    )
    return _load_yaml(config_path, name="config"), root


def _resolve_reference(
    name: str,
    value: object,
    *,
    project_root: Path,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, Path)):
        path = _resolve_path(value, project_root=project_root)
        return _load_yaml(path, name=f"model.{name}")
    raise TypeError(f"model.{name} must be a mapping or YAML path")


def _resolve_encoder_local_paths(
    encoders: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved = dict(encoders)
    if "smiles" not in resolved:
        return resolved
    smiles = dict(_mapping("model.encoders.smiles", resolved["smiles"]))
    for key in ("tokenizer_dir", "cache_dir"):
        value = smiles.get(key)
        if value is None:
            continue
        if not isinstance(value, (str, Path)):
            raise TypeError(
                f"model.encoders.smiles.{key} must be a path"
            )
        if isinstance(value, str) and not value.strip():
            raise ValueError(
                f"model.encoders.smiles.{key} cannot be empty"
            )
        smiles[key] = str(
            _resolve_path(value, project_root=project_root)
        )
    resolved["smiles"] = smiles
    return resolved


def _normalize_modalities(
    name: str,
    value: object,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized: list[str] = []
    for index, modality in enumerate(value):
        if not isinstance(modality, str) or not modality.strip():
            raise ValueError(f"{name}[{index}] must be a non-empty string")
        candidate = modality.strip().lower()
        if candidate not in _MODALITIES:
            raise ValueError(
                f"unsupported {name}[{index}]={modality!r}; "
                f"expected one of {_MODALITIES}"
            )
        normalized.append(candidate)
    if not normalized and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique modalities")
    selected = set(normalized)
    return tuple(modality for modality in _MODALITIES if modality in selected)


def resolve_semmol_config(
    config: Mapping[str, Any] | str | Path,
    *,
    project_root: str | Path | None = None,
    task: Mapping[str, Any] | None = None,
) -> ResolvedSemMolConfig:
    """Resolve a full experiment config or a standalone ``model`` mapping."""

    root_config, root = _configuration_source(
        config,
        project_root=project_root,
    )
    if "model" in root_config:
        model_config = dict(_mapping("model", root_config["model"]))
        resolved_task = (
            task
            if task is not None
            else root_config.get("task")
        )
        checkpoint_value = root_config.get("pretrained_ckpt")
        drop_value = root_config.get("drop_modalities", ())
        distributed_value = root_config.get("distributed", {})
    else:
        model_config = dict(root_config)
        resolved_task = task
        checkpoint_value = None
        drop_value = ()
        distributed_value = {}

    unknown_model_keys = set(model_config) - _MODEL_KEYS
    if unknown_model_keys:
        raise ValueError(
            f"unsupported model configuration keys: "
            f"{sorted(unknown_model_keys)}"
        )
    missing_references = [
        name for name in _REFERENCE_KEYS if name not in model_config
    ]
    if missing_references:
        raise ValueError(
            f"model configuration is missing {missing_references}"
        )
    for name in _REFERENCE_KEYS:
        model_config[name] = _resolve_reference(
            name,
            model_config[name],
            project_root=root,
        )
    for name in _OPTIONAL_REFERENCE_KEYS:
        if name in model_config:
            model_config[name] = _resolve_reference(
                name,
                model_config[name],
                project_root=root,
            )
    model_config["encoders"] = _resolve_encoder_local_paths(
        model_config["encoders"],
        project_root=root,
    )

    configured_modalities = _normalize_modalities(
        "model.modalities",
        model_config.get("modalities", ("1d", "2d", "3d")),
        allow_empty=False,
    )
    dropped_modalities = _normalize_modalities(
        "drop_modalities",
        drop_value,
        allow_empty=True,
    )
    unknown_drops = set(dropped_modalities) - set(configured_modalities)
    if unknown_drops:
        raise ValueError(
            "drop_modalities contains modalities not enabled by the model: "
            f"{sorted(unknown_drops)}"
        )
    anchor_value = model_config.get("anchor_modality", "1d")
    if not isinstance(anchor_value, str) or not anchor_value.strip():
        raise ValueError("model.anchor_modality must be a non-empty string")
    anchor = anchor_value.strip().lower()
    if anchor in dropped_modalities:
        raise ValueError("drop_modalities cannot remove the anchor")
    active_modalities = tuple(
        modality
        for modality in configured_modalities
        if modality not in dropped_modalities
    )
    if len(active_modalities) < 2:
        raise ValueError(
            "at least one target modality must remain after ablation"
        )
    model_config["modalities"] = active_modalities
    model_config["anchor_modality"] = anchor

    if resolved_task is not None:
        model_config["task"] = dict(_mapping("task", resolved_task))

    distributed_options = dict(
        _mapping("distributed", distributed_value)
    )
    world_size = distributed_options.get("world_size", 1)
    if not isinstance(world_size, int) or isinstance(world_size, bool):
        raise TypeError("distributed.world_size must be an integer")
    if world_size <= 0:
        raise ValueError("distributed.world_size must be positive")
    if world_size > 1:
        if distributed_options.get("broadcast_buffers") is not False:
            raise ValueError(
                "distributed.broadcast_buffers must be false because DCL "
                "synchronizes variable-size buffers explicitly"
            )
        if (
            resolved_task is None
            and distributed_options.get("find_unused_parameters") is not True
        ):
            raise ValueError(
                "distributed.find_unused_parameters must be true for "
                "pretraining with rank-local missing modalities"
            )

    pretrained_checkpoint: Path | None = None
    if checkpoint_value is not None:
        if not isinstance(checkpoint_value, (str, Path)):
            raise TypeError("pretrained_ckpt must be a path")
        pretrained_checkpoint = _resolve_path(
            checkpoint_value,
            project_root=root,
        )

    return ResolvedSemMolConfig(
        model_options=model_config,
        pretrained_checkpoint=pretrained_checkpoint,
        dropped_modalities=dropped_modalities,
        distributed_options=distributed_options,
    )


def build_semmol(
    config: Mapping[str, Any] | str | Path,
    *,
    project_root: str | Path | None = None,
    task: Mapping[str, Any] | None = None,
) -> SemMol:
    """Resolve configuration references and construct the model."""

    return resolve_semmol_config(
        config,
        project_root=project_root,
        task=task,
    ).build()


__all__ = [
    "ResolvedSemMolConfig",
    "build_semmol",
    "resolve_semmol_config",
]

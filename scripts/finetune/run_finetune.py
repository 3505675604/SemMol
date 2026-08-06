"""Linux/torchrun entry point for strict ten-seed SemMol finetuning."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.distributed as dist

from src.datasets import (
    FinetuningDataCollator,
    MoleculeNetDataset,
    MoleculeNetSpec,
    create_dataloader,
    get_moleculenet_spec,
    tokenizer_artifact_sha256,
)
from src.losses import DownstreamTaskLoss
from src.models import ResolvedSemMolConfig, resolve_semmol_config
from src.molecular.espf_tokenizer import ESPFTokenizer
from src.trainers.benchmark import (
    PreparedFinetuningRun,
    TenSeedBenchmarkResult,
    TenSeedFinetuningRunner,
)
from src.trainers.checkpointing import (
    configuration_fingerprint,
    load_pretrained_semmol,
)
from src.trainers.common import DistributedContext, initialize_distributed
from src.trainers.finetune_trainer import (
    DownstreamTaskDefinition,
    FinetuningEpochResult,
    FinetuningTrainerConfig,
)
from src.trainers.runtime import (
    LoadedExperimentConfiguration,
    load_experiment_configuration,
    optimizer_steps_per_epoch,
    require_bool,
    require_int,
    require_real,
    require_string,
    require_string_sequence,
    training_configuration_fingerprint,
)
from src.utils.io import atomic_write_json, sha256_file


_TOP_LEVEL_KEYS = {
    "experiment",
    "model",
    "pretrained_ckpt",
    "drop_modalities",
    "data",
    "task",
    "train",
    "eval",
    "distributed",
    "output",
}
_MODEL_REFERENCE_KEYS = ("encoders", "projection", "dcl", "acsm")
_MODALITIES = ("1d", "2d", "3d", "qm")
_EXPERIMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_MAX_SEED = 2**63 - 1
_DETERMINISTIC = True
_CUDNN_BENCHMARK = False
_T = TypeVar("_T")


@dataclass(frozen=True)
class _RunOptions:
    experiment_name: str
    experiment_seed: int
    model: dict[str, Any]
    pretrained_checkpoint: Path
    dropped_modalities: tuple[str, ...]
    dataset_name: str
    store_dir: Path
    train_manifest: Path
    valid_manifest: Path
    test_manifest: Path
    tokenizer_dir: Path
    data_modalities: tuple[str, ...]
    strict_modalities: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    task_type: str
    num_tasks: int
    metrics: tuple[str, ...]
    main_metric: str
    batch_size: int
    epochs: int
    accumulation_steps: int
    gradient_clip_norm: float | None
    precision: str
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None
    early_stopping_enabled: bool
    early_stopping_patience: int
    early_stopping_mode: str
    seeds: tuple[int, ...]
    distributed: dict[str, Any]
    checkpoint_dir: Path
    log_dir: Path
    log_interval: int
    tensorboard: bool
    wandb: bool


@dataclass(frozen=True)
class _StaticInputs:
    tokenizer: ESPFTokenizer
    tokenizer_sha256: str
    spec: MoleculeNetSpec
    resolved_model: ResolvedSemMolConfig
    resolved_model_sha256: str
    artifact_fingerprints: dict[str, str]


@dataclass(frozen=True)
class _DatasetBundle:
    train: MoleculeNetDataset
    valid: MoleculeNetDataset
    test: MoleculeNetDataset
    label_columns: tuple[str, ...]
    split_sizes: dict[str, int]

    def close(self) -> None:
        failures: list[BaseException] = []
        for dataset in (self.train, self.valid, self.test):
            try:
                dataset.close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict ten-seed SemMol MoleculeNet finetuning.",
    )
    parser.add_argument("config", help="Path to a finetuning YAML file.")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional explicit device such as cpu, cuda, or cuda:0.",
    )
    return parser


def _exact_section(
    configuration: LoadedExperimentConfiguration,
    name: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    section = configuration.section(name)
    return _exact_mapping(
        f"configuration section {name!r}",
        section,
        required=required,
        optional=optional,
    )


def _exact_mapping(
    name: str,
    value: object,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    normalized = dict(value)
    allowed = required | (set() if optional is None else optional)
    missing = sorted(required - set(normalized))
    unknown = sorted(set(normalized) - allowed)
    if missing or unknown:
        raise ValueError(f"{name} has missing={missing}, unknown={unknown}")
    return normalized


def _path_reference(name: str, value: object) -> None:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} mapping keys must be strings")
        return
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a mapping or path")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} path cannot be empty")


def _modalities(
    name: str,
    value: object,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if allow_empty and isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes),
    ) and len(value) == 0:
        return ()
    normalized = tuple(
        item.lower() for item in require_string_sequence(name, value)
    )
    unknown = sorted(set(normalized) - set(_MODALITIES))
    if unknown:
        raise ValueError(f"{name} contains unsupported modalities: {unknown}")
    canonical = tuple(item for item in _MODALITIES if item in set(normalized))
    if normalized != canonical:
        raise ValueError(f"{name} must follow canonical order {_MODALITIES}")
    return normalized


def _optimizer_options(value: object) -> dict[str, Any]:
    options = _exact_mapping(
        "train.optimizer",
        value,
        required={"type", "lr", "weight_decay", "betas"},
        optional={"eps", "amsgrad"},
    )
    optimizer_type = require_string(
        "train.optimizer.type",
        options["type"],
    ).lower()
    if optimizer_type != "adamw":
        raise ValueError("train.optimizer.type must be 'adamw'")
    learning_rate = require_real(
        "train.optimizer.lr",
        options["lr"],
        minimum=0.0,
        minimum_inclusive=False,
    )
    weight_decay = require_real(
        "train.optimizer.weight_decay",
        options["weight_decay"],
        minimum=0.0,
    )
    raw_betas = options["betas"]
    if isinstance(raw_betas, (str, bytes)) or not isinstance(
        raw_betas,
        Sequence,
    ):
        raise TypeError("train.optimizer.betas must be a two-item sequence")
    if len(raw_betas) != 2:
        raise ValueError("train.optimizer.betas must contain exactly two values")
    betas = tuple(
        require_real(
            f"train.optimizer.betas[{index}]",
            beta,
            minimum=0.0,
            maximum=1.0,
        )
        for index, beta in enumerate(raw_betas)
    )
    if any(beta >= 1.0 for beta in betas):
        raise ValueError("train.optimizer beta values must be smaller than 1")
    eps = require_real(
        "train.optimizer.eps",
        options.get("eps", 1.0e-8),
        minimum=0.0,
        minimum_inclusive=False,
    )
    amsgrad = require_bool(
        "train.optimizer.amsgrad",
        options.get("amsgrad", False),
    )
    return {
        "type": optimizer_type,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "betas": betas,
        "eps": eps,
        "amsgrad": amsgrad,
    }


def _scheduler_options(
    value: object,
    *,
    optimizer_learning_rate: float,
) -> dict[str, Any] | None:
    if value is None:
        return None
    options = _exact_mapping(
        "train.scheduler",
        value,
        required={"type"},
        optional={"warmup_ratio", "min_lr"},
    )
    scheduler_type = require_string(
        "train.scheduler.type",
        options["type"],
    ).lower()
    if scheduler_type == "none":
        if set(options) != {"type"}:
            raise ValueError(
                "train.scheduler type='none' cannot define other options"
            )
        return {"type": "none"}
    if scheduler_type != "cosine":
        raise ValueError("train.scheduler.type must be 'cosine' or 'none'")
    warmup_ratio = require_real(
        "train.scheduler.warmup_ratio",
        options.get("warmup_ratio", 0.0),
        minimum=0.0,
        maximum=1.0,
    )
    if warmup_ratio >= 1.0:
        raise ValueError("train.scheduler.warmup_ratio must be smaller than 1")
    min_lr = require_real(
        "train.scheduler.min_lr",
        options.get("min_lr", 0.0),
        minimum=0.0,
    )
    if min_lr > optimizer_learning_rate:
        raise ValueError("train.scheduler.min_lr cannot exceed optimizer.lr")
    return {
        "type": scheduler_type,
        "warmup_ratio": warmup_ratio,
        "min_lr": min_lr,
    }


def _parse_configuration(
    configuration: LoadedExperimentConfiguration,
) -> _RunOptions:
    missing = sorted(_TOP_LEVEL_KEYS - set(configuration.values))
    unknown = sorted(set(configuration.values) - _TOP_LEVEL_KEYS)
    if missing or unknown:
        raise ValueError(
            "finetuning configuration top-level schema differs; "
            f"missing={missing}, unknown={unknown}"
        )

    experiment = _exact_section(
        configuration,
        "experiment",
        required={"name", "mode", "seed"},
    )
    experiment_name = require_string("experiment.name", experiment["name"])
    if _EXPERIMENT_NAME.fullmatch(experiment_name) is None:
        raise ValueError(
            "experiment.name may contain only letters, digits, '.', '_', and '-'"
        )
    if require_string("experiment.mode", experiment["mode"]).lower() != "finetune":
        raise ValueError("experiment.mode must be 'finetune'")
    experiment_seed = require_int(
        "experiment.seed",
        experiment["seed"],
        minimum=0,
        maximum=_MAX_SEED,
    )

    model = _exact_section(
        configuration,
        "model",
        required={
            "encoders",
            "projection",
            "dcl",
            "acsm",
            "modalities",
            "anchor_modality",
            "freeze_encoders",
        },
        optional={"head", "validate_values"},
    )
    for key in _MODEL_REFERENCE_KEYS:
        _path_reference(f"model.{key}", model[key])
    model_modalities = _modalities("model.modalities", model["modalities"])
    anchor_modality = require_string(
        "model.anchor_modality",
        model["anchor_modality"],
    ).lower()
    if anchor_modality not in model_modalities:
        raise ValueError("model.anchor_modality must be enabled in model.modalities")
    require_bool("model.freeze_encoders", model["freeze_encoders"])
    if "validate_values" in model:
        require_bool("model.validate_values", model["validate_values"])
    if "head" in model:
        _exact_mapping(
            "model.head",
            model["head"],
            required=set(),
            optional={
                "hidden_dims",
                "activation",
                "dropout",
                "layer_norm_eps",
                "validate_values",
            },
        )

    pretrained_value = configuration.values["pretrained_ckpt"]
    if not isinstance(pretrained_value, (str, Path)):
        raise TypeError("pretrained_ckpt must be a path")
    if isinstance(pretrained_value, str) and not pretrained_value.strip():
        raise ValueError("pretrained_ckpt cannot be empty")
    pretrained_checkpoint = configuration.resolve_path(
        pretrained_value,
        name="pretrained_ckpt",
    )
    dropped_modalities = _modalities(
        "drop_modalities",
        configuration.values["drop_modalities"],
        allow_empty=True,
    )
    if set(dropped_modalities) - set(model_modalities):
        raise ValueError("drop_modalities must be enabled in model.modalities")
    if anchor_modality in dropped_modalities:
        raise ValueError("drop_modalities cannot remove model.anchor_modality")
    active_modalities = tuple(
        modality
        for modality in model_modalities
        if modality not in dropped_modalities
    )
    if len(active_modalities) < 2:
        raise ValueError("finetuning requires an anchor and at least one target")

    data = _exact_section(
        configuration,
        "data",
        required={
            "dataset",
            "store_dir",
            "train_manifest",
            "valid_manifest",
            "test_manifest",
            "tokenizer_dir",
            "modalities",
            "strict_modalities",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
        },
    )
    dataset_name = require_string("data.dataset", data["dataset"]).lower()
    spec = get_moleculenet_spec(dataset_name)
    data_modalities = _modalities("data.modalities", data["modalities"])
    if anchor_modality != "1d" or data_modalities != (anchor_modality,):
        raise ValueError(
            "finetuning data.modalities must contain exactly the 1d model anchor"
        )
    strict_modalities = require_bool(
        "data.strict_modalities",
        data["strict_modalities"],
    )
    if not strict_modalities:
        raise ValueError("data.strict_modalities must be true for finetuning")
    num_workers = require_int(
        "data.num_workers",
        data["num_workers"],
        minimum=0,
    )
    pin_memory = require_bool("data.pin_memory", data["pin_memory"])
    persistent_workers = require_bool(
        "data.persistent_workers",
        data["persistent_workers"],
    )
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=true requires num_workers > 0")
    prefetch_factor = require_int(
        "data.prefetch_factor",
        data["prefetch_factor"],
        minimum=1,
    )
    store_dir = configuration.resolve_path(data["store_dir"], name="data.store_dir")
    train_manifest = configuration.resolve_path(
        data["train_manifest"],
        name="data.train_manifest",
    )
    valid_manifest = configuration.resolve_path(
        data["valid_manifest"],
        name="data.valid_manifest",
    )
    test_manifest = configuration.resolve_path(
        data["test_manifest"],
        name="data.test_manifest",
    )
    if len({train_manifest, valid_manifest, test_manifest}) != 3:
        raise ValueError("train, validation, and test manifests must be distinct")
    tokenizer_dir = configuration.resolve_path(
        data["tokenizer_dir"],
        name="data.tokenizer_dir",
    )

    task = _exact_section(
        configuration,
        "task",
        required={"type", "num_tasks", "metrics", "main_metric"},
    )
    task_type = require_string("task.type", task["type"]).lower()
    if task_type != spec.task_type:
        raise ValueError("task.type does not match the MoleculeNet registry")
    num_tasks = require_int("task.num_tasks", task["num_tasks"], minimum=1)
    if num_tasks != spec.num_tasks:
        raise ValueError("task.num_tasks does not match the MoleculeNet registry")
    metrics = tuple(
        metric.lower()
        for metric in require_string_sequence("task.metrics", task["metrics"])
    )
    expected_metrics = (
        ("roc_auc",)
        if task_type == "classification"
        else ("rmse", "mae", "r2")
    )
    if metrics != expected_metrics:
        raise ValueError(
            f"task.metrics must be exactly {expected_metrics} for {task_type}"
        )
    main_metric = require_string("task.main_metric", task["main_metric"]).lower()
    if main_metric != spec.main_metric or main_metric not in metrics:
        raise ValueError(
            "task.main_metric must match the MoleculeNet registry and task.metrics"
        )

    train = _exact_section(
        configuration,
        "train",
        required={
            "batch_size",
            "epochs",
            "accum_steps",
            "grad_clip",
            "mixed_precision",
            "optimizer",
            "scheduler",
            "early_stopping",
        },
    )
    batch_size = require_int("train.batch_size", train["batch_size"], minimum=1)
    epochs = require_int("train.epochs", train["epochs"], minimum=1)
    accumulation_steps = require_int(
        "train.accum_steps",
        train["accum_steps"],
        minimum=1,
    )
    gradient_clip_norm = (
        None
        if train["grad_clip"] is None
        else require_real(
            "train.grad_clip",
            train["grad_clip"],
            minimum=0.0,
            minimum_inclusive=False,
        )
    )
    raw_precision = require_string(
        "train.mixed_precision",
        train["mixed_precision"],
    ).lower()
    precision_aliases = {
        "none": "none",
        "fp32": "none",
        "amp": "fp16",
        "fp16": "fp16",
        "bf16": "bf16",
    }
    if raw_precision not in precision_aliases:
        raise ValueError(
            "train.mixed_precision must be one of none, fp32, amp, fp16, bf16"
        )
    precision = precision_aliases[raw_precision]
    optimizer = _optimizer_options(train["optimizer"])
    scheduler = _scheduler_options(
        train["scheduler"],
        optimizer_learning_rate=float(optimizer["lr"]),
    )
    early_stopping = _exact_mapping(
        "train.early_stopping",
        train["early_stopping"],
        required={"enabled", "patience", "mode"},
    )
    early_stopping_enabled = require_bool(
        "train.early_stopping.enabled",
        early_stopping["enabled"],
    )
    early_stopping_patience = require_int(
        "train.early_stopping.patience",
        early_stopping["patience"],
        minimum=1,
    )
    early_stopping_mode = require_string(
        "train.early_stopping.mode",
        early_stopping["mode"],
    ).lower()
    expected_mode = "max" if main_metric in {"roc_auc", "r2"} else "min"
    if early_stopping_mode != expected_mode:
        raise ValueError(
            f"train.early_stopping.mode must be {expected_mode!r} for "
            f"task.main_metric={main_metric!r}"
        )

    evaluation = _exact_section(
        configuration,
        "eval",
        required={"num_seeds", "seeds"},
    )
    if require_int("eval.num_seeds", evaluation["num_seeds"], minimum=1) != 10:
        raise ValueError("eval.num_seeds must be exactly 10")
    raw_seeds = evaluation["seeds"]
    if isinstance(raw_seeds, (str, bytes)) or not isinstance(raw_seeds, Sequence):
        raise TypeError("eval.seeds must be a sequence of integers")
    seeds = tuple(
        require_int(
            f"eval.seeds[{index}]",
            seed,
            minimum=0,
            maximum=_MAX_SEED,
        )
        for index, seed in enumerate(raw_seeds)
    )
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("eval.seeds must contain exactly ten unique seeds")
    if experiment_seed not in seeds:
        raise ValueError("experiment.seed must be included in eval.seeds")

    distributed = _exact_section(
        configuration,
        "distributed",
        required={
            "backend",
            "world_size",
            "broadcast_buffers",
            "sync_batchnorm",
        },
        optional={"find_unused_parameters"},
    )
    backend = require_string(
        "distributed.backend",
        distributed["backend"],
    ).lower()
    if backend not in {"nccl", "gloo"}:
        raise ValueError("distributed.backend must be 'nccl' or 'gloo'")
    world_size = require_int(
        "distributed.world_size",
        distributed["world_size"],
        minimum=1,
    )
    broadcast_buffers = require_bool(
        "distributed.broadcast_buffers",
        distributed["broadcast_buffers"],
    )
    if broadcast_buffers:
        raise ValueError("distributed.broadcast_buffers must be false for DCL")
    sync_batchnorm = require_bool(
        "distributed.sync_batchnorm",
        distributed["sync_batchnorm"],
    )
    if sync_batchnorm and world_size == 1:
        raise ValueError("sync_batchnorm=true requires world_size > 1")
    find_unused_parameters = require_bool(
        "distributed.find_unused_parameters",
        distributed.get("find_unused_parameters", True),
    )
    if world_size > 1 and not find_unused_parameters:
        raise ValueError(
            "distributed.find_unused_parameters must be true for downstream "
            "anchor-only execution"
        )
    normalized_distributed = {
        "backend": backend,
        "world_size": world_size,
        "broadcast_buffers": False,
        "sync_batchnorm": sync_batchnorm,
        "find_unused_parameters": find_unused_parameters,
    }

    output = _exact_section(
        configuration,
        "output",
        required={
            "checkpoint_dir",
            "log_dir",
            "save_best",
            "log_interval",
            "tensorboard",
            "wandb",
        },
    )
    if not require_bool("output.save_best", output["save_best"]):
        raise ValueError("output.save_best must be true")
    log_interval = require_int(
        "output.log_interval",
        output["log_interval"],
        minimum=1,
    )

    return _RunOptions(
        experiment_name=experiment_name,
        experiment_seed=experiment_seed,
        model=model,
        pretrained_checkpoint=pretrained_checkpoint,
        dropped_modalities=dropped_modalities,
        dataset_name=dataset_name,
        store_dir=store_dir,
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        test_manifest=test_manifest,
        tokenizer_dir=tokenizer_dir,
        data_modalities=data_modalities,
        strict_modalities=strict_modalities,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        task_type=task_type,
        num_tasks=num_tasks,
        metrics=metrics,
        main_metric=main_metric,
        batch_size=batch_size,
        epochs=epochs,
        accumulation_steps=accumulation_steps,
        gradient_clip_norm=gradient_clip_norm,
        precision=precision,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_mode=early_stopping_mode,
        seeds=seeds,
        distributed=normalized_distributed,
        checkpoint_dir=configuration.resolve_path(
            output["checkpoint_dir"],
            name="output.checkpoint_dir",
        ),
        log_dir=configuration.resolve_path(
            output["log_dir"],
            name="output.log_dir",
        ),
        log_interval=log_interval,
        tensorboard=require_bool("output.tensorboard", output["tensorboard"]),
        wandb=require_bool("output.wandb", output["wandb"]),
    )


def _collective_device(context: DistributedContext) -> torch.device:
    if context.distributed and str(dist.get_backend()).lower() == "nccl":
        return context.device
    return torch.device("cpu")


def _synchronized_local_stage(
    context: DistributedContext,
    operation: str,
    callback: Callable[[], _T],
) -> _T:
    """Coordinate a stage whose callback itself must not use collectives."""

    local_error: Exception | None = None
    result: Any = None
    try:
        result = callback()
    except Exception as exc:
        local_error = exc
    if not context.distributed:
        if local_error is not None:
            raise local_error
        return result

    error_flag = torch.tensor(
        int(local_error is not None),
        dtype=torch.int32,
        device=_collective_device(context),
    )
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    if int(error_flag.item()) == 0:
        return result
    description = (
        None
        if local_error is None
        else (type(local_error).__name__, str(local_error))
    )
    descriptions: list[tuple[str, str] | None] = [
        None for _ in range(context.world_size)
    ]
    dist.all_gather_object(descriptions, description)
    if local_error is not None:
        raise local_error
    failures = [
        f"rank {rank}: {item[0]}: {item[1]}"
        for rank, item in enumerate(descriptions)
        if item is not None
    ]
    raise RuntimeError(
        f"{operation} failed on another rank; " + "; ".join(failures)
    )


def _require_matching_signature(
    context: DistributedContext,
    operation: str,
    signature: Mapping[str, Any],
) -> None:
    if not context.distributed:
        return
    signatures: list[Mapping[str, Any] | None] = [
        None for _ in range(context.world_size)
    ]
    dist.all_gather_object(signatures, dict(signature))
    if any(candidate != signatures[0] for candidate in signatures):
        raise RuntimeError(f"{operation} differs across ranks: {signatures}")


def _rank_zero_stage(
    context: DistributedContext,
    operation: str,
    callback: Callable[[], _T],
) -> _T | None:
    result: Any = None
    local_error: BaseException | None = None
    if context.is_main_process:
        try:
            result = callback()
        except BaseException as exc:
            local_error = exc
    description = (
        None
        if local_error is None
        else (type(local_error).__name__, str(local_error))
    )
    if context.distributed:
        container: list[tuple[str, str] | None] = [description]
        dist.broadcast_object_list(container, src=0)
        description = container[0]
    if local_error is not None:
        raise local_error
    if description is not None:
        raise RuntimeError(
            f"rank zero failed during {operation} "
            f"({description[0]}): {description[1]}"
        )
    return result


def _load_static_inputs(
    configuration: LoadedExperimentConfiguration,
    options: _RunOptions,
) -> _StaticInputs:
    tokenizer = ESPFTokenizer.from_pretrained(options.tokenizer_dir)
    tokenizer_sha256 = tokenizer_artifact_sha256(options.tokenizer_dir)
    spec = get_moleculenet_spec(options.dataset_name)
    resolved_model = resolve_semmol_config(
        configuration.values,
        project_root=configuration.project_root,
    )
    if resolved_model.pretrained_checkpoint != options.pretrained_checkpoint:
        raise ValueError(
            "resolved pretrained checkpoint differs from validated pretrained_ckpt"
        )
    if resolved_model.dropped_modalities != options.dropped_modalities:
        raise ValueError(
            "resolved dropped modalities differ from validated drop_modalities"
        )
    resolved_modalities = tuple(resolved_model.model_options["modalities"])
    anchor = resolved_model.model_options["anchor_modality"]
    if anchor != "1d" or options.data_modalities != (anchor,):
        raise ValueError(
            "the resolved 1d model anchor must exactly match data.modalities"
        )
    if anchor not in resolved_modalities:
        raise ValueError("resolved model modalities do not contain their anchor")
    task_options = resolved_model.model_options.get("task")
    if not isinstance(task_options, Mapping):
        raise TypeError("resolved downstream model must contain task options")
    if (
        task_options.get("type") != options.task_type
        or task_options.get("num_tasks") != options.num_tasks
    ):
        raise ValueError("resolved model task does not match the registry contract")
    encoders = resolved_model.model_options.get("encoders")
    if not isinstance(encoders, Mapping):
        raise TypeError("resolved model.encoders must be a mapping")
    smiles_options = encoders.get("smiles")
    if not isinstance(smiles_options, Mapping):
        raise TypeError("resolved model.encoders.smiles must be a mapping")
    model_tokenizer_dir = configuration.resolve_path(
        smiles_options.get("tokenizer_dir"),
        name="resolved model.encoders.smiles.tokenizer_dir",
    )
    if model_tokenizer_dir != options.tokenizer_dir:
        raise ValueError(
            "data.tokenizer_dir and the resolved 1d encoder tokenizer differ"
        )
    if tokenizer.vocab_size <= 0:
        raise ValueError("the ESPF tokenizer vocabulary must not be empty")

    required_files = {
        "pretrained_checkpoint": options.pretrained_checkpoint,
        "train_manifest": options.train_manifest,
        "valid_manifest": options.valid_manifest,
        "test_manifest": options.test_manifest,
        "store_build_manifest": options.store_dir / "build-manifest.json",
    }
    for name, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    artifact_fingerprints = {
        "tokenizer_artifact": tokenizer_sha256,
        **{
            name: sha256_file(path)
            for name, path in required_files.items()
        },
    }
    resolved_model_sha256 = configuration_fingerprint(
        resolved_model.model_options
    )
    return _StaticInputs(
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        spec=spec,
        resolved_model=resolved_model,
        resolved_model_sha256=resolved_model_sha256,
        artifact_fingerprints=artifact_fingerprints,
    )


def _registered_split_descriptor(
    dataset: MoleculeNetDataset,
    *,
    split: str,
    manifest_path: Path,
    expected_sha256: str,
) -> None:
    views = dataset.build_manifest.get("views")
    if not isinstance(views, Mapping):
        raise TypeError("MoleculeNet build manifest views must be a mapping")
    descriptor = views.get(split)
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "path",
        "sha256",
        "record_count",
    }:
        raise ValueError(f"build manifest lacks the exact {split!r} view")
    relative = descriptor["path"]
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"build manifest {split!r} view path is invalid")
    registered_path = (dataset.store.store_dir / relative).resolve()
    if registered_path != manifest_path:
        raise ValueError(
            f"data.{split}_manifest is not the registered {split!r} view"
        )
    if descriptor["sha256"] != expected_sha256:
        raise ValueError(f"registered {split!r} manifest SHA-256 differs")
    count = descriptor["record_count"]
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count != len(dataset)
    ):
        raise ValueError(f"registered {split!r} record_count differs")


def _build_datasets(
    options: _RunOptions,
    static: _StaticInputs,
) -> _DatasetBundle:
    opened: list[MoleculeNetDataset] = []
    try:
        split_datasets: dict[str, MoleculeNetDataset] = {}
        for split, manifest_path in (
            ("train", options.train_manifest),
            ("valid", options.valid_manifest),
            ("test", options.test_manifest),
        ):
            dataset = MoleculeNetDataset(
                dataset_name=options.dataset_name,
                store_dir=options.store_dir,
                manifest_path=manifest_path,
                modalities=options.data_modalities,
                strict=options.strict_modalities,
                expected_tokenizer_sha256=static.tokenizer_sha256,
            )
            opened.append(dataset)
            split_datasets[split] = dataset
            _registered_split_descriptor(
                dataset,
                split=split,
                manifest_path=manifest_path,
                expected_sha256=static.artifact_fingerprints[
                    f"{split}_manifest"
                ],
            )

        train = split_datasets["train"]
        valid = split_datasets["valid"]
        test = split_datasets["test"]
        label_columns = train.label_columns
        if valid.label_columns != label_columns or test.label_columns != label_columns:
            raise ValueError("train/valid/test label columns must match exactly")
        if len(label_columns) != options.num_tasks:
            raise ValueError("dataset label columns differ from task.num_tasks")
        if not static.spec.dynamic_label_columns and (
            label_columns != static.spec.label_columns
        ):
            raise ValueError("dataset label columns differ from the registry")
        if any(
            dataset.build_manifest != train.build_manifest
            for dataset in (valid, test)
        ):
            raise ValueError("all fixed splits must use the same build manifest")

        split_sizes = {
            split: len(dataset)
            for split, dataset in split_datasets.items()
        }
        if any(size <= 0 for size in split_sizes.values()):
            raise ValueError("train, validation, and test splits must be non-empty")
        source_indices = {
            split: set(int(value) for value in dataset.manifest.source_indices)
            for split, dataset in split_datasets.items()
        }
        record_indices = {
            split: set(int(value) for value in dataset.manifest.record_indices)
            for split, dataset in split_datasets.items()
        }
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            if source_indices[left] & source_indices[right]:
                raise ValueError(
                    f"fixed {left}/{right} splits overlap in source_index"
                )
            if record_indices[left] & record_indices[right]:
                raise ValueError(
                    f"fixed {left}/{right} splits overlap in record_index"
                )
        return _DatasetBundle(
            train=train,
            valid=valid,
            test=test,
            label_columns=label_columns,
            split_sizes=split_sizes,
        )
    except BaseException:
        for dataset in reversed(opened):
            try:
                dataset.close()
            except BaseException:
                continue
        raise


def _task_definition(
    options: _RunOptions,
    datasets: _DatasetBundle,
) -> DownstreamTaskDefinition:
    return DownstreamTaskDefinition(
        task_type=options.task_type,
        num_tasks=options.num_tasks,
        task_names=datasets.label_columns,
        main_metric=options.main_metric,
        metric_direction=options.early_stopping_mode,
    )


class _ExperimentLogger:
    def __init__(
        self,
        *,
        writer: object | None,
        wandb_run: object | None,
    ) -> None:
        self.writer = writer
        self.wandb_run = wandb_run

    @classmethod
    def create(
        cls,
        *,
        options: _RunOptions,
        configuration: Mapping[str, Any],
    ) -> "_ExperimentLogger":
        writer: object | None = None
        wandb_run: object | None = None
        experiment_log_dir = options.log_dir / options.experiment_name
        experiment_log_dir.mkdir(parents=True, exist_ok=True)
        if not experiment_log_dir.is_dir():
            raise NotADirectoryError(
                f"experiment log path is not a directory: {experiment_log_dir}"
            )
        try:
            if options.tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(
                    log_dir=str(experiment_log_dir / "tensorboard")
                )
            if options.wandb:
                import wandb

                wandb_dir = experiment_log_dir / "wandb"
                wandb_dir.mkdir(parents=True, exist_ok=True)
                wandb_run = wandb.init(
                    project="SemMol",
                    name=options.experiment_name,
                    dir=str(wandb_dir),
                    config=dict(configuration),
                    reinit=True,
                )
                if wandb_run is None:
                    raise RuntimeError("wandb.init returned no run")
        except BaseException as primary_error:
            cleanup_failures: list[BaseException] = []
            if writer is not None:
                try:
                    close = getattr(writer, "close", None)
                    if callable(close):
                        close()
                except BaseException as cleanup_error:
                    cleanup_failures.append(cleanup_error)
            if wandb_run is not None:
                try:
                    finish = getattr(wandb_run, "finish", None)
                    if callable(finish):
                        finish(exit_code=1)
                except BaseException as cleanup_error:
                    cleanup_failures.append(cleanup_error)
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                for cleanup_error in cleanup_failures:
                    try:
                        add_note(
                            "logger initialization cleanup also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    except BaseException:
                        continue
            raise
        return cls(writer=writer, wandb_run=wandb_run)

    @staticmethod
    def _epoch_metrics(
        seed: int,
        result: FinetuningEpochResult,
    ) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "seed": seed,
            "epoch": result.epoch + 1,
            "train/loss": result.train_loss,
            "validation/loss": result.validation.loss,
            "validation/main_metric": result.validation.main_metric,
            "train/optimizer_steps": result.optimizer_steps,
            "early_stopping/bad_epochs": result.bad_epochs,
            "early_stopping/improved": int(result.improved),
        }
        for index, learning_rate in enumerate(result.learning_rates):
            metrics[f"train/lr_group_{index}"] = learning_rate
        return metrics

    def log_epoch(
        self,
        *,
        seed: int,
        seed_index: int,
        max_epochs: int,
        log_interval: int,
        result: FinetuningEpochResult,
    ) -> None:
        metrics = self._epoch_metrics(seed, result)
        epoch = result.epoch + 1
        global_step = seed_index * max_epochs + epoch
        if self.writer is not None:
            add_scalar = getattr(self.writer, "add_scalar", None)
            if not callable(add_scalar):
                raise TypeError("TensorBoard writer does not provide add_scalar")
            prefix = f"seed_{seed}"
            for name, value in metrics.items():
                if name not in {"seed", "epoch"}:
                    add_scalar(f"{prefix}/{name}", value, global_step=epoch)
        if self.wandb_run is not None:
            log = getattr(self.wandb_run, "log", None)
            if not callable(log):
                raise TypeError("Weights & Biases run does not provide log")
            log(metrics, step=global_step)
        if (
            epoch == 1
            or epoch % log_interval == 0
            or epoch == max_epochs
            or result.improved
        ):
            print(
                json.dumps(
                    {
                        "event": "finetune_epoch",
                        **metrics,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    def log_result(
        self,
        result: TenSeedBenchmarkResult,
        *,
        max_epochs: int,
    ) -> None:
        aggregate_metrics: dict[str, float | int] = {}
        for aggregate in result.aggregates:
            if aggregate.mean is not None:
                aggregate_metrics[f"benchmark/{aggregate.name}_mean"] = (
                    aggregate.mean
                )
            if aggregate.sample_standard_deviation is not None:
                aggregate_metrics[
                    f"benchmark/{aggregate.name}_sample_std"
                ] = aggregate.sample_standard_deviation
            aggregate_metrics[
                f"benchmark/{aggregate.name}_eligible_seeds"
            ] = aggregate.eligible_seed_count
        step = len(result.runs) * max_epochs + 1
        if self.writer is not None:
            add_scalar = getattr(self.writer, "add_scalar", None)
            if not callable(add_scalar):
                raise TypeError("TensorBoard writer does not provide add_scalar")
            for name, value in aggregate_metrics.items():
                add_scalar(name, value, global_step=step)
            flush = getattr(self.writer, "flush", None)
            if callable(flush):
                flush()
        if self.wandb_run is not None:
            log = getattr(self.wandb_run, "log", None)
            if not callable(log):
                raise TypeError("Weights & Biases run does not provide log")
            log(aggregate_metrics, step=step)

    def close(self, *, success: bool) -> None:
        failures: list[BaseException] = []
        if self.writer is not None:
            try:
                flush = getattr(self.writer, "flush", None)
                if callable(flush):
                    flush()
                close = getattr(self.writer, "close", None)
                if not callable(close):
                    raise TypeError("TensorBoard writer does not provide close")
                close()
            except BaseException as exc:
                failures.append(exc)
        if self.wandb_run is not None:
            try:
                finish = getattr(self.wandb_run, "finish", None)
                if not callable(finish):
                    raise TypeError("Weights & Biases run does not provide finish")
                finish(exit_code=0 if success else 1)
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]


def _initialize_logger(
    context: DistributedContext,
    *,
    options: _RunOptions,
    configuration: Mapping[str, Any],
) -> _ExperimentLogger:
    logger = _rank_zero_stage(
        context,
        "finetuning logger initialization",
        lambda: _ExperimentLogger.create(
            options=options,
            configuration=configuration,
        ),
    )
    if context.is_main_process:
        if not isinstance(logger, _ExperimentLogger):
            raise RuntimeError("rank zero did not create the finetuning logger")
        return logger
    return _ExperimentLogger(writer=None, wandb_run=None)


def _close_logger(
    context: DistributedContext,
    logger: _ExperimentLogger | None,
    *,
    success: bool,
) -> None:
    _rank_zero_stage(
        context,
        "finetuning logger close",
        lambda: None if logger is None else logger.close(success=success),
    )


class _SeedEpochCallback:
    """Same callable type on every rank; only rank zero performs I/O."""

    def __init__(
        self,
        *,
        context: DistributedContext,
        logger: _ExperimentLogger,
        seed: int,
        seed_index: int,
        max_epochs: int,
        log_interval: int,
    ) -> None:
        self.context = context
        self.logger = logger
        self.seed = seed
        self.seed_index = seed_index
        self.max_epochs = max_epochs
        self.log_interval = log_interval

    def __call__(self, result: FinetuningEpochResult) -> None:
        if not self.context.is_main_process:
            return
        self.logger.log_epoch(
            seed=self.seed,
            seed_index=self.seed_index,
            max_epochs=self.max_epochs,
            log_interval=self.log_interval,
            result=result,
        )


class _SeededFinetuningCollator(FinetuningDataCollator):
    """Attach the run seed to an otherwise deterministic finetuning collator."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        allow_partial_modalities: bool,
        seed: int,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            allow_partial_modalities=allow_partial_modalities,
        )
        self.seed = require_int(
            "collator seed",
            seed,
            minimum=0,
            maximum=_MAX_SEED,
        )


class _PreparedRunFactory:
    """Purely rank-local factory; the benchmark runner owns every collective."""

    def __init__(
        self,
        *,
        configuration: LoadedExperimentConfiguration,
        options: _RunOptions,
        context: DistributedContext,
        static: _StaticInputs,
        datasets: _DatasetBundle,
        task: DownstreamTaskDefinition,
        logger: _ExperimentLogger,
    ) -> None:
        self.configuration = configuration
        self.options = options
        self.context = context
        self.static = static
        self.datasets = datasets
        self.task = task
        self.logger = logger
        self.seed_indices = {
            seed: index for index, seed in enumerate(options.seeds)
        }

    def _collator(self, seed: int) -> _SeededFinetuningCollator:
        return _SeededFinetuningCollator(
            pad_token_id=self.static.tokenizer.pad_token_id,
            allow_partial_modalities=not self.options.strict_modalities,
            seed=seed,
        )

    def _loader(
        self,
        dataset: MoleculeNetDataset,
        *,
        collator: _SeededFinetuningCollator,
        shuffle: bool,
        seed: int,
    ) -> Any:
        return create_dataloader(
            dataset,
            batch_size=self.options.batch_size,
            collate_fn=collator,
            shuffle=shuffle,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory,
            drop_last=False,
            persistent_workers=self.options.persistent_workers,
            prefetch_factor=self.options.prefetch_factor,
            seed=seed,
        )

    def __call__(self, seed: int) -> PreparedFinetuningRun:
        if seed not in self.seed_indices:
            raise ValueError(f"runner requested an undeclared seed: {seed}")

        expected_checkpoint_sha256 = self.static.artifact_fingerprints[
            "pretrained_checkpoint"
        ]
        checkpoint_sha256_before = sha256_file(
            self.options.pretrained_checkpoint
        )
        if checkpoint_sha256_before != expected_checkpoint_sha256:
            raise RuntimeError(
                "pretrained checkpoint changed after startup validation"
            )
        model = self.static.resolved_model.build()
        transfer = load_pretrained_semmol(
            self.options.pretrained_checkpoint,
            model,
            map_location=torch.device("cpu"),
        )
        checkpoint_sha256_after = sha256_file(
            self.options.pretrained_checkpoint
        )
        if checkpoint_sha256_after != expected_checkpoint_sha256:
            raise RuntimeError(
                "pretrained checkpoint changed while preparing a seed"
            )
        if transfer.path != self.options.pretrained_checkpoint:
            raise ValueError("pretrained transfer resolved an unexpected path")
        if not transfer.missing_keys or any(
            not name.startswith("property_head.")
            for name in transfer.missing_keys
        ):
            raise RuntimeError(
                "pretrained transfer must leave only the fresh property head "
                "uninitialized"
            )
        model_values = tuple(model.parameters()) + tuple(model.buffers())
        if any(value.device.type != "cpu" for value in model_values):
            raise ValueError("Prepared model must remain on CPU before runner setup")

        train_collator = self._collator(seed)
        valid_collator = self._collator(seed)
        test_collator = self._collator(seed)
        train_loader = self._loader(
            self.datasets.train,
            collator=train_collator,
            shuffle=True,
            seed=seed,
        )
        valid_loader = self._loader(
            self.datasets.valid,
            collator=valid_collator,
            shuffle=False,
            seed=seed,
        )
        test_loader = self._loader(
            self.datasets.test,
            collator=test_collator,
            shuffle=False,
            seed=seed,
        )
        loader_batch_counts = {
            "train": len(train_loader),
            "valid": len(valid_loader),
            "test": len(test_loader),
        }
        if any(count <= 0 for count in loader_batch_counts.values()):
            raise ValueError("every finetuning DataLoader must contain a batch")
        steps_per_epoch = optimizer_steps_per_epoch(
            loader_batch_counts["train"],
            self.options.accumulation_steps,
        )
        total_optimizer_steps = steps_per_epoch * self.options.epochs
        if total_optimizer_steps <= 0 or not math.isfinite(
            float(total_optimizer_steps)
        ):
            raise ValueError("total optimizer steps must be positive and finite")

        seed_directory = (
            self.options.checkpoint_dir
            / self.options.experiment_name
            / f"seed_{seed}"
        )
        trainer_config = FinetuningTrainerConfig(
            max_epochs=self.options.epochs,
            best_checkpoint_path=seed_directory / "best.pt",
            latest_checkpoint_path=seed_directory / "latest.pt",
            gradient_accumulation_steps=self.options.accumulation_steps,
            precision=self.options.precision,
            gradient_clip_norm=self.options.gradient_clip_norm,
            early_stopping_patience=(
                self.options.early_stopping_patience
                if self.options.early_stopping_enabled
                else None
            ),
            min_improvement=0.0,
            non_blocking_transfer=(
                self.options.pin_memory and self.context.device.type == "cuda"
            ),
        )
        sampler_sizes = {
            "train": len(train_loader.sampler),
            "valid": len(valid_loader.sampler),
            "test": len(test_loader.sampler),
        }
        config_fingerprint = training_configuration_fingerprint(
            self.configuration.values,
            resolved_model_configuration=(
                self.static.resolved_model.model_options
            ),
            artifact_fingerprints=self.static.artifact_fingerprints,
            derived_values={
                "seed": seed,
                "train_loader_batch_count": loader_batch_counts["train"],
                "valid_loader_batch_count": loader_batch_counts["valid"],
                "test_loader_batch_count": loader_batch_counts["test"],
                "optimizer_steps_per_epoch": steps_per_epoch,
                "total_optimizer_steps": total_optimizer_steps,
                "world_size": self.context.world_size,
                "train_split_size": self.datasets.split_sizes["train"],
                "valid_split_size": self.datasets.split_sizes["valid"],
                "test_split_size": self.datasets.split_sizes["test"],
                "train_sampler_size": sampler_sizes["train"],
                "valid_sampler_size": sampler_sizes["valid"],
                "test_sampler_size": sampler_sizes["test"],
                "gradient_accumulation_steps": (
                    self.options.accumulation_steps
                ),
                "max_epochs": self.options.epochs,
                "label_columns": self.datasets.label_columns,
                "loss_type": (
                    "binary_cross_entropy_with_logits"
                    if self.options.task_type == "classification"
                    else "mse"
                ),
                "deterministic": _DETERMINISTIC,
                "cudnn_benchmark": _CUDNN_BENCHMARK,
                "find_unused_parameters": self.options.distributed[
                    "find_unused_parameters"
                ],
            },
        )
        callback = _SeedEpochCallback(
            context=self.context,
            logger=self.logger,
            seed=seed,
            seed_index=self.seed_indices[seed],
            max_epochs=self.options.epochs,
            log_interval=self.options.log_interval,
        )
        return PreparedFinetuningRun(
            seed=seed,
            model=model,
            loss_fn=DownstreamTaskLoss(
                task_type=self.options.task_type,
                loss_type="mse",
                distributed_sync=self.context.distributed,
                validate_values=True,
            ),
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            config=trainer_config,
            config_fingerprint=config_fingerprint,
            optimizer_options=self.options.optimizer,
            scheduler_options=self.options.scheduler,
            distributed_options={
                "broadcast_buffers": False,
                "sync_batchnorm": self.options.distributed["sync_batchnorm"],
                "find_unused_parameters": self.options.distributed[
                    "find_unused_parameters"
                ],
            },
            epoch_callback=callback,
        )


def _cleanup_runtime(context: DistributedContext) -> None:
    gc.collect()
    if context.device.type == "cuda":
        torch.cuda.empty_cache()


def _report_cleanup_failures(failures: Sequence[BaseException]) -> None:
    if not failures:
        return
    try:
        descriptions: list[str] = []
        for error in failures:
            try:
                detail = str(error)
            except BaseException:
                detail = "<unprintable exception>"
            descriptions.append(f"{type(error).__name__}: {detail}")
        print(
            "cleanup errors after finetuning failure: "
            + "; ".join(descriptions),
            file=sys.stderr,
            flush=True,
        )
    except BaseException:
        return


def _destroy_failed_process_group(context: DistributedContext) -> None:
    """Destroy this entry point's group without waiting on failed peers."""

    if (
        not context.initialized_here
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return
    if dist.get_rank() != context.rank or dist.get_world_size() != (
        context.world_size
    ):
        raise RuntimeError("refusing to destroy a different process group")
    dist.destroy_process_group()


def _run(arguments: argparse.Namespace) -> int:
    configuration = load_experiment_configuration(
        arguments.config,
        expected_mode="finetune",
    )
    bootstrap_distributed = configuration.section("distributed")

    context: DistributedContext | None = None
    datasets: _DatasetBundle | None = None
    logger: _ExperimentLogger | None = None
    primary_error: BaseException | None = None
    cleanup_failures: list[BaseException] = []
    completed = False
    try:
        context = initialize_distributed(
            bootstrap_distributed,
            requested_device=arguments.device,
        )
        options = _synchronized_local_stage(
            context,
            "strict finetuning configuration validation",
            lambda: _parse_configuration(configuration),
        )
        raw_configuration_sha256 = _synchronized_local_stage(
            context,
            "finetuning configuration canonicalization",
            lambda: configuration_fingerprint(configuration.values),
        )
        _require_matching_signature(
            context,
            "finetuning configuration contract",
            {
                "configuration_sha256": raw_configuration_sha256,
                "experiment": options.experiment_name,
                "experiment_seed": options.experiment_seed,
                "seeds": options.seeds,
                "world_size": options.distributed["world_size"],
                "find_unused_parameters": options.distributed[
                    "find_unused_parameters"
                ],
                "deterministic": _DETERMINISTIC,
                "cudnn_benchmark": _CUDNN_BENCHMARK,
            },
        )
        static = _synchronized_local_stage(
            context,
            "tokenizer, model, checkpoint, and artifact validation",
            lambda: _load_static_inputs(configuration, options),
        )
        _require_matching_signature(
            context,
            "static finetuning input contract",
            {
                "dataset": static.spec.name,
                "task_type": static.spec.task_type,
                "num_tasks": static.spec.num_tasks,
                "main_metric": static.spec.main_metric,
                "tokenizer_sha256": static.tokenizer_sha256,
                "resolved_model_sha256": static.resolved_model_sha256,
                "artifact_fingerprints": tuple(
                    sorted(static.artifact_fingerprints.items())
                ),
            },
        )

        datasets = _synchronized_local_stage(
            context,
            "fixed MoleculeNet split construction and validation",
            lambda: _build_datasets(options, static),
        )
        _require_matching_signature(
            context,
            "fixed MoleculeNet split contract",
            {
                "label_columns": datasets.label_columns,
                "split_sizes": tuple(sorted(datasets.split_sizes.items())),
                "modalities": options.data_modalities,
                "strict_modalities": options.strict_modalities,
            },
        )
        task = _synchronized_local_stage(
            context,
            "downstream task definition",
            lambda: _task_definition(options, datasets),
        )
        _require_matching_signature(
            context,
            "downstream task definition",
            task.as_dict(),
        )

        logger = _initialize_logger(
            context,
            options=options,
            configuration=configuration.values,
        )
        prepared_factory = _synchronized_local_stage(
            context,
            "prepared-run factory construction",
            lambda: _PreparedRunFactory(
                configuration=configuration,
                options=options,
                context=context,
                static=static,
                datasets=datasets,
                task=task,
                logger=logger,
            ),
        )
        runner = TenSeedFinetuningRunner(
            context=context,
            task=task,
            seeds=options.seeds,
            prepared_run_factory=prepared_factory,
            deterministic=_DETERMINISTIC,
            cudnn_benchmark=_CUDNN_BENCHMARK,
        )
        result = runner.run()
        _rank_zero_stage(
            context,
            "final benchmark logging",
            lambda: logger.log_result(result, max_epochs=options.epochs),
        )
        result_path = (
            options.log_dir
            / options.experiment_name
            / "ten_seed_results.json"
        )

        def publish_result() -> None:
            atomic_write_json(
                result_path,
                result.as_dict(),
                overwrite=True,
            )
            print(
                json.dumps(
                    {
                        "event": "ten_seed_benchmark_complete",
                        "experiment": options.experiment_name,
                        "path": str(result_path),
                        "seeds": list(options.seeds),
                        "world_size": context.world_size,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

        _rank_zero_stage(
            context,
            "atomic ten-seed result publication",
            publish_result,
        )
        completed = True
        return 0
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if primary_error is None:
            if context is not None:
                try:
                    _close_logger(context, logger, success=completed)
                except BaseException as exc:
                    cleanup_failures.append(exc)
            if context is not None and datasets is not None:
                try:
                    _synchronized_local_stage(
                        context,
                        "MoleculeNet dataset cleanup",
                        datasets.close,
                    )
                except BaseException as exc:
                    cleanup_failures.append(exc)
            if context is not None:
                try:
                    _synchronized_local_stage(
                        context,
                        "finetuning runtime cleanup",
                        lambda: _cleanup_runtime(context),
                    )
                except BaseException as exc:
                    cleanup_failures.append(exc)
                try:
                    context.close()
                except BaseException as exc:
                    cleanup_failures.append(exc)
            if cleanup_failures:
                raise cleanup_failures[0]
        else:
            if (
                context is not None
                and context.is_main_process
                and logger is not None
            ):
                try:
                    logger.close(success=False)
                except BaseException as exc:
                    cleanup_failures.append(exc)
            if datasets is not None:
                try:
                    datasets.close()
                except BaseException as exc:
                    cleanup_failures.append(exc)
            if context is not None:
                try:
                    _cleanup_runtime(context)
                except BaseException as exc:
                    cleanup_failures.append(exc)
                try:
                    _destroy_failed_process_group(context)
                except BaseException as exc:
                    cleanup_failures.append(exc)
            if cleanup_failures and (
                context is None or context.is_main_process
            ):
                _report_cleanup_failures(cleanup_failures)


def main() -> int:
    arguments = _argument_parser().parse_args()
    return _run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

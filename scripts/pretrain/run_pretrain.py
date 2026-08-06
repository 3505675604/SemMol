"""Linux/torchrun entry point for SemMol self-supervised pretraining."""

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
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, Subset

from src.datasets import (
    PCQMMultimodalDataset,
    PretrainingDataCollator,
    create_dataloader,
    tokenizer_artifact_sha256,
)
from src.losses import SemMolPretrainTotalLoss
from src.models import resolve_semmol_config
from src.molecular.espf_tokenizer import ESPFTokenizer
from src.trainers.common import (
    DistributedContext,
    initialize_distributed,
    seed_everything,
)
from src.trainers.checkpointing import configuration_fingerprint
from src.trainers.pretrain_trainer import (
    PretrainCheckpointConfig,
    PretrainEpochResult,
    PretrainFitResult,
    PretrainProgressResult,
    PretrainTrainer,
    PretrainTrainerConfig,
)
from src.trainers.runtime import (
    LoadedExperimentConfiguration,
    build_optimizer,
    build_scheduler,
    load_experiment_configuration,
    optimizer_steps_per_epoch,
    require_bool,
    require_int,
    require_real,
    require_string,
    require_string_sequence,
    training_configuration_fingerprint,
)
from src.utils.io import sha256_file


_TOP_LEVEL_SECTIONS = {
    "experiment",
    "model",
    "data",
    "train",
    "loss",
    "mask",
    "distributed",
    "output",
}
_MODALITIES = ("1d", "2d", "3d", "qm")
_EXPERIMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_DCL_CONTROL_ATTRIBUTES = (
    "num_clusters",
    "feature_dim",
    "ema_momentum",
    "init_method",
    "init_num_iters",
    "init_max_samples",
    "init_seed",
    "reassign_interval",
    "assignment_temperature",
    "center_l2_normalize",
    "distributed_sync",
    "eps",
    "validate_values",
)
_T = TypeVar("_T")


@dataclass(frozen=True)
class _RunOptions:
    experiment_name: str
    seed: int
    deterministic: bool
    cudnn_benchmark: bool
    model: dict[str, Any]
    store_dir: Path
    manifest_path: Path
    tokenizer_dir: Path
    modalities: tuple[str, ...]
    strict_modalities: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    debug_subset: int | None
    batch_size: int
    epochs: int
    accumulation_steps: int
    max_grad_norm: float | None
    precision: str
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None
    loss: dict[str, float]
    smiles_mask_ratio: float
    node_mask_ratio: float
    edge_mask_ratio: float
    geo_noise_std: float
    distributed: dict[str, Any]
    checkpoint_dir: Path
    log_dir: Path
    save_every_n_epochs: int
    resume_from: Path | None
    log_interval: int
    tensorboard: bool
    wandb: bool


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SemMol pretraining locally or through torchrun.",
    )
    parser.add_argument("config", help="Path to a pretraining YAML file.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path overriding output.resume.",
    )
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
    allowed = required | (set() if optional is None else optional)
    missing = sorted(required - set(section))
    unknown = sorted(set(section) - allowed)
    if missing or unknown:
        raise ValueError(
            f"configuration section {name!r} has missing={missing}, "
            f"unknown={unknown}"
        )
    return section


def _mapping_copy(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return dict(value)


def _path_reference(name: str, value: object) -> None:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} mapping keys must be strings")
        return
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a mapping or path")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} path cannot be empty")


def _modalities(name: str, value: object) -> tuple[str, ...]:
    normalized = tuple(
        modality.lower()
        for modality in require_string_sequence(name, value)
    )
    unknown = sorted(set(normalized) - set(_MODALITIES))
    if unknown:
        raise ValueError(f"{name} contains unsupported modalities: {unknown}")
    ordered = tuple(item for item in _MODALITIES if item in set(normalized))
    if normalized != ordered:
        raise ValueError(f"{name} must follow canonical order {_MODALITIES}")
    return normalized


def _optional_resume(
    configuration: LoadedExperimentConfiguration,
    configured: object,
    override: str | None,
) -> Path | None:
    selected = override if override is not None else configured
    if selected is None:
        return None
    if not isinstance(selected, (str, Path)):
        raise TypeError("output.resume must be a path or null")
    if isinstance(selected, str) and not selected.strip():
        raise ValueError("resume checkpoint path cannot be empty")
    return configuration.resolve_path(selected, name="resume checkpoint")


def _parse_configuration(
    configuration: LoadedExperimentConfiguration,
    *,
    resume_override: str | None,
) -> _RunOptions:
    missing_sections = sorted(_TOP_LEVEL_SECTIONS - set(configuration.values))
    unknown_sections = sorted(set(configuration.values) - _TOP_LEVEL_SECTIONS)
    if missing_sections or unknown_sections:
        raise ValueError(
            "pretraining configuration top-level schema differs; "
            f"missing={missing_sections}, unknown={unknown_sections}"
        )

    experiment = _exact_section(
        configuration,
        "experiment",
        required={"name", "mode", "seed", "deterministic", "cudnn_benchmark"},
    )
    experiment_name = require_string("experiment.name", experiment["name"])
    if _EXPERIMENT_NAME.fullmatch(experiment_name) is None:
        raise ValueError(
            "experiment.name may contain only letters, digits, '.', '_', and '-'"
        )
    mode = require_string("experiment.mode", experiment["mode"]).lower()
    if mode != "pretrain":
        raise ValueError("experiment.mode must be 'pretrain'")
    seed = require_int(
        "experiment.seed",
        experiment["seed"],
        minimum=0,
        maximum=2**63 - 1,
    )
    deterministic = require_bool(
        "experiment.deterministic",
        experiment["deterministic"],
    )
    cudnn_benchmark = require_bool(
        "experiment.cudnn_benchmark",
        experiment["cudnn_benchmark"],
    )
    if deterministic and cudnn_benchmark:
        raise ValueError(
            "deterministic and cudnn_benchmark cannot both be enabled"
        )

    model = _exact_section(
        configuration,
        "model",
        required={
            "encoders",
            "dcl",
            "acsm",
            "projection",
            "pretraining_heads",
            "modalities",
            "anchor_modality",
            "freeze_encoders",
        },
        optional={"validate_values"},
    )
    for reference_name in (
        "encoders",
        "dcl",
        "acsm",
        "projection",
        "pretraining_heads",
    ):
        _path_reference(f"model.{reference_name}", model[reference_name])
    model_modalities = _modalities("model.modalities", model["modalities"])
    if "qm" in model_modalities:
        raise ValueError(
            "QM has no configured manuscript pretraining objective and cannot "
            "be enabled for pretraining"
        )
    anchor_modality = require_string(
        "model.anchor_modality",
        model["anchor_modality"],
    ).lower()
    if anchor_modality not in model_modalities:
        raise ValueError("model.anchor_modality must be enabled in model.modalities")
    require_bool("model.freeze_encoders", model["freeze_encoders"])
    if "validate_values" in model:
        require_bool("model.validate_values", model["validate_values"])

    data = _exact_section(
        configuration,
        "data",
        required={
            "store_dir",
            "manifest_path",
            "tokenizer_dir",
            "modalities",
            "strict_modalities",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
        },
        optional={"debug_subset"},
    )
    data_modalities = _modalities("data.modalities", data["modalities"])
    if data_modalities != model_modalities:
        raise ValueError(
            "data.modalities must exactly match model.modalities for pretraining"
        )
    if "1d" not in data_modalities:
        raise ValueError("pretraining data must contain the configured 1d tokenizer")
    strict_modalities = require_bool(
        "data.strict_modalities",
        data["strict_modalities"],
    )
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
    debug_subset = (
        None
        if "debug_subset" not in data
        else require_int(
            "data.debug_subset",
            data["debug_subset"],
            minimum=1,
        )
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
        },
    )
    batch_size = require_int("train.batch_size", train["batch_size"], minimum=1)
    epochs = require_int("train.epochs", train["epochs"], minimum=1)
    accumulation_steps = require_int(
        "train.accum_steps",
        train["accum_steps"],
        minimum=1,
    )
    if accumulation_steps != 1:
        raise ValueError(
            "train.accum_steps must be 1 for SemMol pretraining: objectives "
            "have different valid-element counts and every microbatch updates "
            "the DCL online, so window loss division is not a valid large-batch "
            "mean"
        )
    max_grad_norm = (
        None
        if train["grad_clip"] is None
        else require_real(
            "train.grad_clip",
            train["grad_clip"],
            minimum=0.0,
            minimum_inclusive=False,
        )
    )
    precision = require_string(
        "train.mixed_precision",
        train["mixed_precision"],
    ).lower()
    if precision not in {"none", "fp32", "amp", "fp16", "bf16"}:
        raise ValueError(
            "train.mixed_precision must be one of none, fp32, amp, fp16, bf16"
        )
    optimizer_options = _mapping_copy("train.optimizer", train["optimizer"])
    scheduler_options = (
        None
        if train["scheduler"] is None
        else _mapping_copy("train.scheduler", train["scheduler"])
    )

    loss = _exact_section(
        configuration,
        "loss",
        required={"mlm", "graph", "geo", "pseudo", "alignment"},
    )
    normalized_loss = {
        name: require_real(f"loss.{name}", loss[name], minimum=0.0)
        for name in ("mlm", "graph", "geo", "pseudo", "alignment")
    }
    if all(value == 0.0 for value in normalized_loss.values()):
        raise ValueError("at least one pretraining loss weight must be positive")

    mask = _exact_section(
        configuration,
        "mask",
        required={
            "smiles_ratio",
            "node_ratio",
            "edge_ratio",
            "geo_noise_std",
        },
    )
    smiles_mask_ratio = require_real(
        "mask.smiles_ratio",
        mask["smiles_ratio"],
        minimum=0.0,
        maximum=1.0,
    )
    node_mask_ratio = require_real(
        "mask.node_ratio",
        mask["node_ratio"],
        minimum=0.0,
        maximum=1.0,
    )
    edge_mask_ratio = require_real(
        "mask.edge_ratio",
        mask["edge_ratio"],
        minimum=0.0,
        maximum=1.0,
    )
    geo_noise_std = require_real(
        "mask.geo_noise_std",
        mask["geo_noise_std"],
        minimum=0.0,
    )

    distributed = _exact_section(
        configuration,
        "distributed",
        required={
            "backend",
            "world_size",
            "broadcast_buffers",
            "sync_batchnorm",
            "find_unused_parameters",
        },
        optional={"sampler"},
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
    sync_batchnorm = require_bool(
        "distributed.sync_batchnorm",
        distributed["sync_batchnorm"],
    )
    find_unused = require_bool(
        "distributed.find_unused_parameters",
        distributed["find_unused_parameters"],
    )
    sampler = require_string(
        "distributed.sampler",
        distributed.get("sampler", "distributed"),
    ).lower()
    if sampler != "distributed":
        raise ValueError("distributed.sampler must be 'distributed'")
    if world_size > 1 and broadcast_buffers:
        raise ValueError("distributed.broadcast_buffers must be false for DCL")
    if world_size > 1 and not find_unused:
        raise ValueError(
            "distributed.find_unused_parameters must be true for pretraining"
        )
    if world_size == 1 and sync_batchnorm:
        raise ValueError("sync_batchnorm=true requires distributed.world_size > 1")
    normalized_distributed = {
        "backend": backend,
        "world_size": world_size,
        "broadcast_buffers": broadcast_buffers,
        "sync_batchnorm": sync_batchnorm,
        "find_unused_parameters": find_unused,
        "sampler": sampler,
    }

    output = _exact_section(
        configuration,
        "output",
        required={
            "checkpoint_dir",
            "log_dir",
            "save_every_n_epochs",
            "resume",
            "log_interval",
            "tensorboard",
            "wandb",
        },
    )
    save_every_n_epochs = require_int(
        "output.save_every_n_epochs",
        output["save_every_n_epochs"],
        minimum=1,
    )
    log_interval = require_int(
        "output.log_interval",
        output["log_interval"],
        minimum=1,
    )
    tensorboard = require_bool("output.tensorboard", output["tensorboard"])
    wandb_enabled = require_bool("output.wandb", output["wandb"])

    return _RunOptions(
        experiment_name=experiment_name,
        seed=seed,
        deterministic=deterministic,
        cudnn_benchmark=cudnn_benchmark,
        model=model,
        store_dir=configuration.resolve_path(
            data["store_dir"],
            name="data.store_dir",
        ),
        manifest_path=configuration.resolve_path(
            data["manifest_path"],
            name="data.manifest_path",
        ),
        tokenizer_dir=configuration.resolve_path(
            data["tokenizer_dir"],
            name="data.tokenizer_dir",
        ),
        modalities=data_modalities,
        strict_modalities=strict_modalities,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        debug_subset=debug_subset,
        batch_size=batch_size,
        epochs=epochs,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        precision=precision,
        optimizer=optimizer_options,
        scheduler=scheduler_options,
        loss=normalized_loss,
        smiles_mask_ratio=smiles_mask_ratio,
        node_mask_ratio=node_mask_ratio,
        edge_mask_ratio=edge_mask_ratio,
        geo_noise_std=geo_noise_std,
        distributed=normalized_distributed,
        checkpoint_dir=configuration.resolve_path(
            output["checkpoint_dir"],
            name="output.checkpoint_dir",
        ),
        log_dir=configuration.resolve_path(
            output["log_dir"],
            name="output.log_dir",
        ),
        save_every_n_epochs=save_every_n_epochs,
        resume_from=_optional_resume(
            configuration,
            output["resume"],
            resume_override,
        ),
        log_interval=log_interval,
        tensorboard=tensorboard,
        wandb=wandb_enabled,
    )


def _collective_device(context: DistributedContext) -> torch.device:
    if context.distributed and str(dist.get_backend()) == "nccl":
        return context.device
    return torch.device("cpu")


def _synchronized_local_stage(
    context: DistributedContext,
    operation: str,
    callback: Callable[[], _T],
) -> _T:
    """Finish rank-local work everywhere before any later collective stage."""

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
    local_description = (
        None
        if local_error is None
        else (type(local_error).__name__, str(local_error))
    )
    descriptions: list[tuple[str, str] | None] = [
        None for _ in range(context.world_size)
    ]
    dist.all_gather_object(descriptions, local_description)
    if local_error is not None:
        raise local_error
    failures = [
        f"rank {rank}: {description[0]}: {description[1]}"
        for rank, description in enumerate(descriptions)
        if description is not None
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
    authoritative = signatures[0]
    if any(candidate != authoritative for candidate in signatures):
        raise RuntimeError(
            f"{operation} differs across ranks: {signatures}"
        )


def _require_rank_zero_callback_layout(
    context: DistributedContext,
    operation: str,
    callback_count: int,
) -> None:
    if not isinstance(callback_count, int) or isinstance(callback_count, bool):
        raise TypeError("callback_count must be an integer")
    counts: list[int | None]
    if context.distributed:
        counts = [None for _ in range(context.world_size)]
        dist.all_gather_object(counts, callback_count)
    else:
        counts = [callback_count]
    expected = [1] + [0 for _ in range(context.world_size - 1)]
    if counts != expected:
        raise RuntimeError(
            f"{operation} must install one callback on rank zero and none on "
            f"other ranks; got {counts}"
        )


def _model_contract_signature(model: nn.Module) -> dict[str, Any]:
    parameters = tuple(
        (
            name,
            tuple(parameter.shape),
            str(parameter.dtype),
            bool(parameter.requires_grad),
        )
        for name, parameter in model.named_parameters()
    )
    buffers = tuple(
        (name, tuple(buffer.shape), str(buffer.dtype))
        for name, buffer in model.named_buffers()
    )
    if not parameters:
        raise ValueError("the resolved SemMol model has no parameters")
    return {
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "modalities": tuple(getattr(model, "modalities", ())),
        "anchor_modality": getattr(model, "anchor_modality", None),
        "pretraining_heads": tuple(
            getattr(model, "pretraining_heads", {}).keys()
        ),
        "dcl_configuration": tuple(
            (
                name,
                tuple(
                    (
                        attribute,
                        getattr(library, attribute, None),
                    )
                    for attribute in _DCL_CONTROL_ATTRIBUTES
                ),
            )
            for name, library in getattr(model, "dcls", {}).items()
        ),
        "parameters": parameters,
        "buffers": buffers,
    }


def _prepare_model_for_ddp(
    model: nn.Module,
    context: DistributedContext,
    distributed_options: Mapping[str, Any],
) -> nn.Module:
    prepared = model.to(context.device)
    if bool(distributed_options["sync_batchnorm"]):
        if not context.distributed:
            raise ValueError("SyncBatchNorm requires a distributed process group")
        if context.device.type != "cuda":
            raise ValueError("SyncBatchNorm DDP requires CUDA")
        prepared = nn.SyncBatchNorm.convert_sync_batchnorm(prepared)
    misplaced = [
        name
        for name, value in (
            *tuple(prepared.named_parameters()),
            *tuple(prepared.named_buffers()),
        )
        if value.device != context.device
    ]
    if misplaced:
        raise ValueError(
            f"model values were not moved to {context.device}: {misplaced[:8]}"
        )
    return prepared


def _wrap_prepared_model(
    model: nn.Module,
    context: DistributedContext,
    distributed_options: Mapping[str, Any],
) -> nn.Module:
    if not context.distributed:
        return model
    find_unused = bool(distributed_options["find_unused_parameters"])
    if context.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.device.index],
            output_device=context.device.index,
            broadcast_buffers=False,
            find_unused_parameters=find_unused,
        )
    return DistributedDataParallel(
        model,
        device_ids=None,
        broadcast_buffers=False,
        find_unused_parameters=find_unused,
    )


class _EpochLogger:
    def __init__(self, writer: object | None, wandb_run: object | None) -> None:
        self.writer = writer
        self.wandb_run = wandb_run

    @classmethod
    def create(
        cls,
        *,
        options: _RunOptions,
        configuration: Mapping[str, Any],
    ) -> "_EpochLogger":
        writer: object | None = None
        wandb_run: object | None = None
        experiment_log_dir = options.log_dir / options.experiment_name
        experiment_log_dir.mkdir(parents=True, exist_ok=True)
        try:
            if options.tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(
                    log_dir=str(experiment_log_dir / "tensorboard")
                )
            if options.wandb:
                import wandb

                wandb_directory = experiment_log_dir / "wandb"
                wandb_directory.mkdir(parents=True, exist_ok=True)
                wandb_run = wandb.init(
                    project="SemMol",
                    name=options.experiment_name,
                    dir=str(wandb_directory),
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
    def _metrics(result: PretrainEpochResult) -> dict[str, float | int]:
        train = result.training
        metrics: dict[str, float | int] = {
            "epoch": result.epoch + 1,
            "train/total_loss": train.losses.total_loss,
            "train/mlm_loss": train.losses.mlm_loss,
            "train/graph_loss": train.losses.graph_loss,
            "train/geo_loss": train.losses.geo_loss,
            "train/pseudo_loss": train.losses.pseudo_loss,
            "train/alignment_loss": train.losses.alignment_loss,
            "train/pseudo_scale": train.losses.pseudo_scale,
            "train/optimizer_steps": train.optimizer_steps,
            "train/skipped_optimizer_steps": train.skipped_optimizer_steps,
            "train/processed_samples": train.processed_samples,
        }
        for index, learning_rate in enumerate(train.learning_rates):
            metrics[f"train/lr_group_{index}"] = learning_rate
        if result.validation is not None:
            valid = result.validation
            metrics.update(
                {
                    "valid/total_loss": valid.losses.total_loss,
                    "valid/mlm_loss": valid.losses.mlm_loss,
                    "valid/graph_loss": valid.losses.graph_loss,
                    "valid/geo_loss": valid.losses.geo_loss,
                    "valid/pseudo_loss": valid.losses.pseudo_loss,
                    "valid/alignment_loss": valid.losses.alignment_loss,
                    "valid/processed_samples": valid.processed_samples,
                }
            )
        return metrics

    def log_epoch(self, result: PretrainEpochResult) -> None:
        metrics = self._metrics(result)
        step = result.epoch + 1
        if self.writer is not None:
            add_scalar = getattr(self.writer, "add_scalar", None)
            if not callable(add_scalar):
                raise TypeError("TensorBoard writer does not provide add_scalar")
            for name, value in metrics.items():
                if name != "epoch":
                    add_scalar(name, value, global_step=step)
        if self.wandb_run is not None:
            log = getattr(self.wandb_run, "log", None)
            if not callable(log):
                raise TypeError("Weights & Biases run does not provide log")
            log(metrics, step=step)

    @staticmethod
    def log_progress(result: PretrainProgressResult) -> None:
        print(
            json.dumps(
                {
                    "event": "train_progress",
                    "epoch": result.epoch + 1,
                    "completed_batches": result.completed_batches,
                    "total_batches": result.total_batches,
                    "optimizer_step": result.optimizer_step,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

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


def _rank_zero_logger(
    context: DistributedContext,
    *,
    options: _RunOptions,
    configuration: Mapping[str, Any],
) -> _EpochLogger | None:
    logger: _EpochLogger | None = None
    local_error: BaseException | None = None
    if context.is_main_process:
        try:
            logger = _EpochLogger.create(
                options=options,
                configuration=configuration,
            )
        except BaseException as exc:
            local_error = exc
    description = (
        None
        if local_error is None
        else (type(local_error).__name__, str(local_error))
    )
    if context.distributed:
        payload: list[tuple[str, str] | None] = [description]
        dist.broadcast_object_list(payload, src=0)
        description = payload[0]
    if local_error is not None:
        raise local_error
    if description is not None:
        raise RuntimeError(
            "rank zero failed to initialize logging "
            f"({description[0]}): {description[1]}"
        )
    return logger


def _close_rank_zero_logger(
    context: DistributedContext,
    logger: _EpochLogger | None,
    *,
    success: bool,
) -> None:
    local_error: BaseException | None = None
    if context.is_main_process and logger is not None:
        try:
            logger.close(success=success)
        except BaseException as exc:
            local_error = exc
    description = (
        None
        if local_error is None
        else (type(local_error).__name__, str(local_error))
    )
    if context.distributed:
        payload: list[tuple[str, str] | None] = [description]
        dist.broadcast_object_list(payload, src=0)
        description = payload[0]
    if local_error is not None:
        raise local_error
    if description is not None:
        raise RuntimeError(
            "rank zero failed to close logging "
            f"({description[0]}): {description[1]}"
        )


def _destroy_owned_process_group_locally(
    context: DistributedContext,
) -> None:
    """Destroy this entrypoint's process group without any collective."""

    if not context.initialized_here:
        return
    if not dist.is_available() or not dist.is_initialized():
        return
    active_rank = dist.get_rank()
    active_world_size = dist.get_world_size()
    if active_rank != context.rank or active_world_size != context.world_size:
        raise RuntimeError(
            "refusing to destroy a process group that differs from context"
        )
    dist.destroy_process_group()


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
            "cleanup errors after training failure: "
            + "; ".join(descriptions),
            file=sys.stderr,
            flush=True,
        )
    except BaseException:
        return


def _summary(
    options: _RunOptions,
    result: PretrainFitResult,
    *,
    tokenizer_sha256: str,
    world_size: int,
) -> dict[str, Any]:
    final_epoch = result.epochs[-1] if result.epochs else None
    return {
        "status": "completed",
        "experiment": options.experiment_name,
        "world_size": world_size,
        "tokenizer_sha256": tokenizer_sha256,
        "epochs_executed": len(result.epochs),
        "next_epoch": result.state.next_epoch,
        "optimizer_step": result.state.optimizer_step,
        "best_epoch": (
            None if result.state.best_epoch == -1 else result.state.best_epoch + 1
        ),
        "best_validation_loss": (
            None if result.state.best_epoch == -1 else result.state.best_metric
        ),
        "resumed_from": (
            None if result.resumed_from is None else str(result.resumed_from)
        ),
        "latest_checkpoint": (
            None
            if result.latest_checkpoint is None
            else str(result.latest_checkpoint)
        ),
        "best_checkpoint": (
            None if result.best_checkpoint is None else str(result.best_checkpoint)
        ),
        "periodic_checkpoints": [
            str(path) for path in result.periodic_checkpoints
        ],
        "final_train_loss": (
            None
            if final_epoch is None
            else final_epoch.training.losses.total_loss
        ),
        "final_validation_loss": (
            None
            if final_epoch is None or final_epoch.validation is None
            else final_epoch.validation.losses.total_loss
        ),
        "log_interval": options.log_interval,
    }


def _run(arguments: argparse.Namespace) -> int:
    configuration = load_experiment_configuration(
        arguments.config,
        expected_mode="pretrain",
    )
    options = _parse_configuration(
        configuration,
        resume_override=arguments.resume,
    )

    context: DistributedContext | None = None
    base_dataset: PCQMMultimodalDataset | None = None
    dataset: Dataset[Any] | None = None
    train_loader: Any = None
    model: nn.Module | None = None
    loss_fn: SemMolPretrainTotalLoss | None = None
    optimizer: Any = None
    scheduler: Any = None
    trainer: PretrainTrainer | None = None
    logger: _EpochLogger | None = None
    primary_error: BaseException | None = None
    cleanup_failures: list[BaseException] = []
    completed = False
    try:
        context = initialize_distributed(
            options.distributed,
            requested_device=arguments.device,
        )
        _synchronized_local_stage(
            context,
            "global random seed configuration",
            lambda: seed_everything(
                options.seed,
                deterministic=options.deterministic,
                cudnn_benchmark=options.cudnn_benchmark,
            ),
        )

        def load_static_inputs() -> tuple[Any, str, Any, dict[str, str], str]:
            tokenizer = ESPFTokenizer.from_pretrained(options.tokenizer_dir)
            tokenizer_sha256 = tokenizer_artifact_sha256(
                options.tokenizer_dir
            )
            resolved_model = resolve_semmol_config(
                configuration.values,
                project_root=configuration.project_root,
            )
            if resolved_model.pretrained_checkpoint is not None:
                raise ValueError("pretraining must not configure pretrained_ckpt")
            resolved_modalities = tuple(
                resolved_model.model_options["modalities"]
            )
            if resolved_modalities != options.modalities:
                raise ValueError(
                    "resolved model modalities differ from validated data "
                    "modalities"
                )
            encoders = _mapping_copy(
                "resolved model.encoders",
                resolved_model.model_options["encoders"],
            )
            resolved_dcl = _mapping_copy(
                "resolved model.dcl",
                resolved_model.model_options["dcl"],
            )
            dcl_distributed_sync = require_bool(
                "resolved model.dcl.distributed_sync",
                resolved_dcl.get("distributed_sync", True),
            )
            if context.distributed and not dcl_distributed_sync:
                raise ValueError(
                    "DDP pretraining requires resolved "
                    "model.dcl.distributed_sync=true"
                )
            smiles_options = _mapping_copy(
                "resolved model.encoders.smiles",
                encoders.get("smiles"),
            )
            model_tokenizer_dir = configuration.resolve_path(
                smiles_options.get("tokenizer_dir"),
                name="model.encoders.smiles.tokenizer_dir",
            )
            if model_tokenizer_dir != options.tokenizer_dir:
                raise ValueError(
                    "data.tokenizer_dir and model encoder tokenizer_dir must "
                    "match"
                )
            build_manifest_path = options.store_dir / "build-manifest.json"
            artifact_fingerprints = {
                "tokenizer_artifact": tokenizer_sha256,
                "selection_manifest": sha256_file(options.manifest_path),
                "store_build_manifest": sha256_file(build_manifest_path),
            }
            resolved_model_sha256 = configuration_fingerprint(
                resolved_model.model_options
            )
            return (
                tokenizer,
                tokenizer_sha256,
                resolved_model,
                artifact_fingerprints,
                resolved_model_sha256,
            )

        (
            tokenizer,
            tokenizer_sha256,
            resolved_model,
            artifact_fingerprints,
            resolved_model_sha256,
        ) = _synchronized_local_stage(
            context,
            "tokenizer, artifact, and resolved-model loading",
            load_static_inputs,
        )
        _require_matching_signature(
            context,
            "static pretraining input contract",
            {
                "tokenizer_sha256": tokenizer_sha256,
                "artifact_fingerprints": tuple(
                    sorted(artifact_fingerprints.items())
                ),
                "resolved_model_sha256": resolved_model_sha256,
            },
        )

        def build_dataset() -> tuple[Dataset[Any], int]:
            nonlocal base_dataset
            base_dataset = PCQMMultimodalDataset(
                store_dir=options.store_dir,
                manifest_path=options.manifest_path,
                modalities=options.modalities,
                strict=options.strict_modalities,
                expected_tokenizer_sha256=tokenizer_sha256,
            )
            dataset: Dataset[Any] = base_dataset
            base_length = len(base_dataset)
            if options.debug_subset is not None:
                if options.debug_subset > base_length:
                    raise ValueError(
                        "data.debug_subset exceeds the selected manifest length: "
                        f"{options.debug_subset} > {base_length}"
                    )
                dataset = Subset(
                    base_dataset,
                    range(options.debug_subset),
                )
            dataset_length = len(dataset)
            if dataset_length <= 0:
                raise ValueError("the selected pretraining dataset is empty")
            return dataset, dataset_length

        dataset, dataset_length = _synchronized_local_stage(
            context,
            "pretraining dataset construction",
            build_dataset,
        )
        _require_matching_signature(
            context,
            "pretraining dataset contract",
            {
                "dataset_length": dataset_length,
                "modalities": options.modalities,
                "strict_modalities": options.strict_modalities,
            },
        )

        def build_train_loader() -> tuple[Any, int]:
            special_token_ids = tuple(
                tokenizer.vocab[token] for token in tokenizer.special_tokens
            )
            collator = PretrainingDataCollator(
                pad_token_id=tokenizer.pad_token_id,
                mask_token_id=tokenizer.mask_token_id,
                vocab_size=tokenizer.vocab_size,
                special_token_ids=special_token_ids,
                smiles_mask_ratio=options.smiles_mask_ratio,
                node_mask_ratio=options.node_mask_ratio,
                edge_mask_ratio=options.edge_mask_ratio,
                geo_noise_std=options.geo_noise_std,
                seed=options.seed,
                allow_partial_modalities=not options.strict_modalities,
            )
            train_loader = create_dataloader(
                dataset,
                batch_size=options.batch_size,
                collate_fn=collator,
                shuffle=True,
                num_workers=options.num_workers,
                pin_memory=options.pin_memory,
                drop_last=False,
                persistent_workers=options.persistent_workers,
                prefetch_factor=options.prefetch_factor,
                seed=options.seed,
            )
            batch_count = len(train_loader)
            if batch_count <= 0:
                raise ValueError("the pretraining DataLoader has no batches")
            return train_loader, batch_count

        train_loader, train_loader_batch_count = _synchronized_local_stage(
            context,
            "pretraining collator and DataLoader construction",
            build_train_loader,
        )
        _require_matching_signature(
            context,
            "pretraining DataLoader contract",
            {
                "dataset_length": dataset_length,
                "batch_size": options.batch_size,
                "batch_count": train_loader_batch_count,
                "sampler_type": (
                    f"{type(train_loader.sampler).__module__}."
                    f"{type(train_loader.sampler).__qualname__}"
                ),
            },
        )

        model = _synchronized_local_stage(
            context,
            "rank-local SemMol construction",
            resolved_model.build,
        )
        initial_model_signature = _synchronized_local_stage(
            context,
            "rank-local SemMol contract extraction",
            lambda: _model_contract_signature(model),
        )
        _require_matching_signature(
            context,
            "rank-local SemMol contract",
            initial_model_signature,
        )
        model = _synchronized_local_stage(
            context,
            "rank-local SemMol device preparation",
            lambda: _prepare_model_for_ddp(
                model,
                context,
                options.distributed,
            ),
        )
        prepared_model_signature = _synchronized_local_stage(
            context,
            "prepared SemMol contract extraction",
            lambda: _model_contract_signature(model),
        )
        prepared_model_signature["device_type"] = context.device.type
        _require_matching_signature(
            context,
            "final pre-DDP SemMol contract",
            prepared_model_signature,
        )
        model = _wrap_prepared_model(
            model,
            context,
            options.distributed,
        )

        def build_training_objects() -> tuple[
            SemMolPretrainTotalLoss,
            Any,
            int,
            int,
            Any,
            PretrainTrainerConfig,
            str,
        ]:
            loss_fn = SemMolPretrainTotalLoss(
                loss_config=options.loss,
                distributed_sync=context.distributed,
            )
            optimizer = build_optimizer(model, options.optimizer)
            steps_per_epoch = optimizer_steps_per_epoch(
                train_loader_batch_count,
                options.accumulation_steps,
            )
            total_optimizer_steps = steps_per_epoch * options.epochs
            if total_optimizer_steps <= 0 or not math.isfinite(
                float(total_optimizer_steps)
            ):
                raise ValueError(
                    "computed total optimizer steps must be positive"
                )
            scheduler = build_scheduler(
                optimizer,
                options.scheduler,
                total_optimizer_steps=total_optimizer_steps,
            )
            checkpoint = PretrainCheckpointConfig(
                directory=options.checkpoint_dir,
                save_every_n_epochs=options.save_every_n_epochs,
                latest_filename=f"{options.experiment_name}_latest.pt",
                best_filename=f"{options.experiment_name}_best.pt",
                periodic_prefix=options.experiment_name,
            )
            trainer_config = PretrainTrainerConfig(
                epochs=options.epochs,
                checkpoint=checkpoint,
                accumulation_steps=options.accumulation_steps,
                precision=options.precision,
                max_grad_norm=options.max_grad_norm,
                non_blocking=(
                    options.pin_memory and context.device.type == "cuda"
                ),
            )
            config_fingerprint = training_configuration_fingerprint(
                configuration.values,
                resolved_model_configuration=resolved_model.model_options,
                artifact_fingerprints=artifact_fingerprints,
                derived_values={
                    "train_loader_batch_count": train_loader_batch_count,
                    "optimizer_steps_per_epoch": steps_per_epoch,
                    "total_optimizer_steps": total_optimizer_steps,
                    "world_size": context.world_size,
                },
            )
            return (
                loss_fn,
                optimizer,
                steps_per_epoch,
                total_optimizer_steps,
                scheduler,
                trainer_config,
                config_fingerprint,
            )

        (
            loss_fn,
            optimizer,
            steps_per_epoch,
            total_optimizer_steps,
            scheduler,
            trainer_config,
            config_fingerprint,
        ) = _synchronized_local_stage(
            context,
            "loss, optimizer, scheduler, and trainer configuration",
            build_training_objects,
        )
        _require_matching_signature(
            context,
            "pretraining optimizer trajectory contract",
            {
                "config_fingerprint": config_fingerprint,
                "steps_per_epoch": steps_per_epoch,
                "total_optimizer_steps": total_optimizer_steps,
                "optimizer_type": (
                    f"{type(optimizer).__module__}."
                    f"{type(optimizer).__qualname__}"
                ),
                "scheduler_type": (
                    None
                    if scheduler is None
                    else (
                        f"{type(scheduler).__module__}."
                        f"{type(scheduler).__qualname__}"
                    )
                ),
                "loss_weights": tuple(
                    options.loss[name]
                    for name in ("mlm", "graph", "geo", "pseudo", "alignment")
                ),
            },
        )
        logger = _rank_zero_logger(
            context,
            options=options,
            configuration=configuration.values,
        )
        trainer = PretrainTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            config=trainer_config,
            context=context,
            config_fingerprint=config_fingerprint,
            scheduler=scheduler,
            valid_loader=None,
        )
        callbacks = (
            (logger.log_epoch,)
            if context.is_main_process and logger is not None
            else ()
        )
        progress_callbacks = (
            (logger.log_progress,)
            if context.is_main_process and logger is not None
            else ()
        )
        _require_rank_zero_callback_layout(
            context,
            "epoch callback layout",
            len(callbacks),
        )
        _require_rank_zero_callback_layout(
            context,
            "progress callback layout",
            len(progress_callbacks),
        )
        result = trainer.fit(
            resume_from=options.resume_from,
            callbacks=callbacks,
            progress_interval=options.log_interval,
            progress_callbacks=progress_callbacks,
        )
        if context.is_main_process:
            print(
                json.dumps(
                    _summary(
                        options,
                        result,
                        tokenizer_sha256=tokenizer_sha256,
                        world_size=context.world_size,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
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
                    _close_rank_zero_logger(
                        context,
                        logger,
                        success=completed,
                    )
                except BaseException as exc:
                    cleanup_failures.append(exc)
            if base_dataset is not None:
                try:
                    base_dataset.close()
                except BaseException as exc:
                    cleanup_failures.append(exc)
            if context is not None:
                try:
                    context.close()
                except BaseException as exc:
                    cleanup_failures.append(exc)
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
            if base_dataset is not None:
                try:
                    base_dataset.close()
                except BaseException as exc:
                    cleanup_failures.append(exc)
            trainer = None
            optimizer = None
            scheduler = None
            loss_fn = None
            model = None
            train_loader = None
            dataset = None
            try:
                gc.collect()
            except BaseException as exc:
                cleanup_failures.append(exc)
            if context is not None:
                try:
                    _destroy_owned_process_group_locally(context)
                except BaseException as exc:
                    cleanup_failures.append(exc)
        if primary_error is None and cleanup_failures:
            raise cleanup_failures[0]
        if primary_error is not None and cleanup_failures:
            rank = 0 if context is None else context.rank
            if rank == 0:
                _report_cleanup_failures(cleanup_failures)


def main() -> int:
    arguments = _argument_parser().parse_args()
    return _run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

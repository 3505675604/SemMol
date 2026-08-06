"""Distributed pretraining orchestration for the complete SemMol objective."""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch

from src.datasets import PretrainingDataCollator, set_dataloader_epoch
from src.losses import (
    LossComponent,
    SemMolPretrainLossOutput,
    SemMolPretrainTotalLoss,
)
from src.models import SemMol, SemMolPretrainingOutput
from src.models.semantic.dcl import DynamicCentralLibrary

from .checkpointing import (
    TrainingCheckpointLoadResult,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .common import (
    DistributedContext,
    PrecisionMode,
    TrainerState,
    all_reduce_sum,
    broadcast_bool,
    move_batch_to_device,
    unwrap_model,
)


_T = TypeVar("_T")
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMPONENT_NAMES = (
    "mlm",
    "graph_node",
    "graph_edge",
    "graph_structure",
    "geo_mse",
    "geo_direction",
    "pseudo",
    "alignment",
)
_CHECKPOINT_EXTRA = {
    "trainer_kind": "pretrain",
    "trainer_schema_version": 1,
}
_MODALITY_ORDER = ("1d", "2d", "3d", "qm")
_MODALITY_COLUMN = {
    name: index for index, name in enumerate(_MODALITY_ORDER)
}
_MODALITY_INPUT_KEYS = {
    "1d": ("input_ids", "attention_mask"),
    "2d": ("graph", "graph_sample_index"),
    "3d": ("atomic_numbers", "coords", "atom_mask", "conformer_mask"),
    "qm": ("qm_grid", "qm_mask"),
}
_MODALITY_TARGET_KEYS = {
    "1d": ("mlm_labels",),
    "2d": ("node_mask", "node_labels", "edge_mask", "edge_labels"),
    "3d": ("clean_coords", "coord_noise"),
    "qm": (),
}
_MODALITY_OPTIONAL_KEYS = {
    "1d": ("token_spans",),
    "2d": (),
    "3d": (),
    "qm": ("qm_metadata",),
}
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


def _strict_integer(name: str, value: object, *, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _finite_float(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _checkpoint_filename(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty filename")
    if value != value.strip():
        raise ValueError(f"{name} cannot have surrounding whitespace")
    if any(character in value for character in ("/", "\\")):
        raise ValueError(f"{name} must not contain path separators")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.name != value or len(candidate.parts) != 1:
        raise ValueError(f"{name} must not contain a directory component")
    if candidate.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError(f"{name} must end in .pt or .pth")
    return value


def _periodic_prefix(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("periodic_prefix must be a non-empty string")
    if value != value.strip():
        raise ValueError("periodic_prefix cannot have surrounding whitespace")
    if any(character in value for character in ("/", "\\")):
        raise ValueError("periodic_prefix cannot contain path separators")
    if value in {".", ".."}:
        raise ValueError("periodic_prefix cannot be '.' or '..'")
    return value


@dataclass(frozen=True)
class PretrainCheckpointConfig:
    """Names and cadence for epoch-boundary pretraining checkpoints."""

    directory: str | Path
    save_every_n_epochs: int = 1
    latest_filename: str = "pretrain_latest.pt"
    best_filename: str = "pretrain_best.pt"
    periodic_prefix: str = "pretrain"

    def __post_init__(self) -> None:
        if not isinstance(self.directory, (str, Path)):
            raise TypeError("checkpoint directory must be a string or pathlib.Path")
        if isinstance(self.directory, str) and not self.directory.strip():
            raise ValueError("checkpoint directory cannot be empty")
        directory = Path(self.directory).expanduser()
        if directory.name in {"", ".", ".."}:
            raise ValueError("checkpoint directory must identify a concrete directory")
        object.__setattr__(self, "directory", directory)
        object.__setattr__(
            self,
            "save_every_n_epochs",
            _strict_integer(
                "save_every_n_epochs",
                self.save_every_n_epochs,
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "latest_filename",
            _checkpoint_filename("latest_filename", self.latest_filename),
        )
        object.__setattr__(
            self,
            "best_filename",
            _checkpoint_filename("best_filename", self.best_filename),
        )
        if self.latest_filename == self.best_filename:
            raise ValueError("latest_filename and best_filename must differ")
        object.__setattr__(
            self,
            "periodic_prefix",
            _periodic_prefix(self.periodic_prefix),
        )

    @property
    def latest_path(self) -> Path:
        return Path(self.directory) / self.latest_filename

    @property
    def best_path(self) -> Path:
        return Path(self.directory) / self.best_filename

    def periodic_path(self, completed_epoch: int) -> Path:
        completed = _strict_integer(
            "completed_epoch",
            completed_epoch,
            minimum=1,
        )
        return Path(self.directory) / (
            f"{self.periodic_prefix}_epoch_{completed:04d}.pt"
        )


@dataclass(frozen=True)
class PretrainTrainerConfig:
    """Validated runtime controls owned by the pretraining loop."""

    epochs: int
    checkpoint: PretrainCheckpointConfig
    accumulation_steps: int = 1
    precision: str = "none"
    max_grad_norm: float | None = 1.0
    non_blocking: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epochs",
            _strict_integer("epochs", self.epochs, minimum=1),
        )
        if not isinstance(self.checkpoint, PretrainCheckpointConfig):
            raise TypeError("checkpoint must be PretrainCheckpointConfig")
        accumulation_steps = _strict_integer(
            "accumulation_steps",
            self.accumulation_steps,
            minimum=1,
        )
        if accumulation_steps != 1:
            raise ValueError(
                "pretraining requires accumulation_steps == 1 because its "
                "objectives use different valid-element counts and every "
                "microbatch performs an online DCL update; dividing window "
                "losses cannot reproduce a correct large-batch mean"
            )
        object.__setattr__(self, "accumulation_steps", accumulation_steps)
        raw_precision = self.precision
        if not isinstance(raw_precision, str):
            raise TypeError("precision must be a string")
        precision = PrecisionMode(
            "none" if raw_precision == "fp32" else raw_precision
        )
        object.__setattr__(self, "precision", precision.mode)
        if self.max_grad_norm is not None:
            normalized_norm = _finite_float(
                "max_grad_norm",
                self.max_grad_norm,
            )
            if normalized_norm <= 0.0:
                raise ValueError("max_grad_norm must be positive or None")
            object.__setattr__(self, "max_grad_norm", normalized_norm)
        object.__setattr__(
            self,
            "non_blocking",
            _strict_bool("non_blocking", self.non_blocking),
        )


@dataclass(frozen=True)
class PretrainLossSummary:
    """Globally reduced, count-weighted objective values for one epoch."""

    total_loss: float
    mlm_loss: float
    graph_loss: float
    geo_loss: float
    pseudo_loss: float
    alignment_loss: float
    pseudo_scale: float
    component_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in (
            "total_loss",
            "mlm_loss",
            "graph_loss",
            "geo_loss",
            "pseudo_loss",
            "alignment_loss",
            "pseudo_scale",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(name, getattr(self, name)),
            )
        if not 0.0 <= self.pseudo_scale <= 1.0:
            raise ValueError("pseudo_scale must be between zero and one")
        if not isinstance(self.component_counts, tuple):
            raise TypeError("component_counts must be a tuple")
        names = tuple(name for name, _ in self.component_counts)
        if names != _COMPONENT_NAMES:
            raise ValueError(
                "component_counts must follow the complete objective schema"
            )
        normalized_counts: list[tuple[str, int]] = []
        for name, count in self.component_counts:
            normalized_counts.append(
                (name, _strict_integer(f"component_counts[{name}]", count, minimum=0))
            )
        object.__setattr__(self, "component_counts", tuple(normalized_counts))

    def counts_as_dict(self) -> dict[str, int]:
        return dict(self.component_counts)


@dataclass(frozen=True)
class PretrainTrainingResult:
    epoch: int
    losses: PretrainLossSummary
    microbatches: int
    processed_samples: int
    optimizer_steps: int
    skipped_optimizer_steps: int
    learning_rates: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epoch",
            _strict_integer("epoch", self.epoch, minimum=0),
        )
        if not isinstance(self.losses, PretrainLossSummary):
            raise TypeError("losses must be PretrainLossSummary")
        for name in (
            "microbatches",
            "processed_samples",
            "optimizer_steps",
            "skipped_optimizer_steps",
        ):
            minimum = 1 if name in {"microbatches", "processed_samples"} else 0
            object.__setattr__(
                self,
                name,
                _strict_integer(name, getattr(self, name), minimum=minimum),
            )
        if self.optimizer_steps + self.skipped_optimizer_steps <= 0:
            raise ValueError("an epoch must attempt at least one optimizer step")
        if not isinstance(self.learning_rates, tuple) or not self.learning_rates:
            raise ValueError("learning_rates must be a non-empty tuple")
        object.__setattr__(
            self,
            "learning_rates",
            tuple(
                _finite_float(
                    f"learning_rates[{index}]",
                    value,
                    minimum=0.0,
                )
                for index, value in enumerate(self.learning_rates)
            ),
        )


@dataclass(frozen=True)
class PretrainProgressResult:
    """A deterministic rank-zero progress event at a completed batch boundary."""

    epoch: int
    completed_batches: int
    total_batches: int
    optimizer_step: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epoch",
            _strict_integer("epoch", self.epoch, minimum=0),
        )
        object.__setattr__(
            self,
            "completed_batches",
            _strict_integer(
                "completed_batches",
                self.completed_batches,
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "total_batches",
            _strict_integer("total_batches", self.total_batches, minimum=1),
        )
        object.__setattr__(
            self,
            "optimizer_step",
            _strict_integer("optimizer_step", self.optimizer_step, minimum=0),
        )
        if self.completed_batches > self.total_batches:
            raise ValueError("completed_batches cannot exceed total_batches")


@dataclass(frozen=True)
class PretrainValidationResult:
    epoch: int
    losses: PretrainLossSummary
    microbatches: int
    processed_samples: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epoch",
            _strict_integer("epoch", self.epoch, minimum=0),
        )
        if not isinstance(self.losses, PretrainLossSummary):
            raise TypeError("losses must be PretrainLossSummary")
        object.__setattr__(
            self,
            "microbatches",
            _strict_integer("microbatches", self.microbatches, minimum=1),
        )
        object.__setattr__(
            self,
            "processed_samples",
            _strict_integer(
                "processed_samples",
                self.processed_samples,
                minimum=1,
            ),
        )


@dataclass(frozen=True)
class PretrainEpochResult:
    epoch: int
    training: PretrainTrainingResult
    validation: PretrainValidationResult | None
    improved: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epoch",
            _strict_integer("epoch", self.epoch, minimum=0),
        )
        if not isinstance(self.training, PretrainTrainingResult):
            raise TypeError("training must be PretrainTrainingResult")
        if self.training.epoch != self.epoch:
            raise ValueError("training epoch does not match epoch result")
        if self.validation is not None:
            if not isinstance(self.validation, PretrainValidationResult):
                raise TypeError("validation must be PretrainValidationResult or None")
            if self.validation.epoch != self.epoch:
                raise ValueError("validation epoch does not match epoch result")
        object.__setattr__(
            self,
            "improved",
            _strict_bool("improved", self.improved),
        )
        if self.validation is None and self.improved:
            raise ValueError("an epoch without validation cannot be marked improved")


@dataclass(frozen=True)
class PretrainFitResult:
    state: TrainerState
    epochs: tuple[PretrainEpochResult, ...]
    resumed_from: Path | None
    latest_checkpoint: Path | None
    best_checkpoint: Path | None
    periodic_checkpoints: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state, TrainerState):
            raise TypeError("state must be TrainerState")
        if not isinstance(self.epochs, tuple) or any(
            not isinstance(item, PretrainEpochResult) for item in self.epochs
        ):
            raise TypeError("epochs must be a tuple of PretrainEpochResult")
        for name in (
            "resumed_from",
            "latest_checkpoint",
            "best_checkpoint",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{name} must be pathlib.Path or None")
        if not isinstance(self.periodic_checkpoints, tuple) or any(
            not isinstance(path, Path) for path in self.periodic_checkpoints
        ):
            raise TypeError("periodic_checkpoints must be a tuple of Paths")


EpochEndCallback = Callable[[PretrainEpochResult], None]
ProgressCallback = Callable[[PretrainProgressResult], None]


def _collective_device(context: DistributedContext) -> torch.device:
    if not context.distributed:
        return context.device
    if context.distributed and str(dist.get_backend()) == "nccl":
        return context.device
    return torch.device("cpu")


def _active_rank_world() -> tuple[int, int]:
    active = dist.is_available() and dist.is_initialized()
    return (
        dist.get_rank() if active else 0,
        dist.get_world_size() if active else 1,
    )


def _active_collective_device(world_size: int) -> torch.device:
    if world_size > 1 and str(dist.get_backend()) == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL trainer coordination requires CUDA")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _active_distributed_call(
    operation: str,
    callback: Callable[[], _T],
) -> _T:
    """Coordinate constructor validation without trusting a supplied context."""

    rank, world_size = _active_rank_world()
    local_error: Exception | None = None
    result: Any = None
    try:
        result = callback()
    except Exception as exc:
        local_error = exc
    if world_size == 1:
        if local_error is not None:
            raise local_error
        return result

    error_flag = torch.tensor(
        int(local_error is not None),
        dtype=torch.int32,
        device=_active_collective_device(world_size),
    )
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    if int(error_flag.item()) == 0:
        return result

    local_description = (
        None
        if local_error is None
        else (type(local_error).__name__, str(local_error))
    )
    descriptions: list[tuple[str, str] | None] = [None] * world_size
    dist.all_gather_object(descriptions, local_description)
    if local_error is not None:
        raise local_error
    failures = [
        f"rank {failed_rank}: {description[0]}: {description[1]}"
        for failed_rank, description in enumerate(descriptions)
        if description is not None
    ]
    raise RuntimeError(
        f"{operation} failed on another rank; " + "; ".join(failures)
    )


def _distributed_call(
    context: DistributedContext,
    operation: str,
    callback: Callable[[], _T],
) -> _T:
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
    if int(error_flag.item()) != 0:
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
    return result


def _require_matching_integer(
    context: DistributedContext,
    name: str,
    value: int,
) -> int:
    normalized = _strict_integer(name, value, minimum=0)
    if not context.distributed:
        return normalized
    local = torch.tensor(
        normalized,
        dtype=torch.int64,
        device=_collective_device(context),
    )
    minimum = local.clone()
    maximum = local.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if int(minimum.item()) != int(maximum.item()):
        values: list[int | None] = [None for _ in range(context.world_size)]
        dist.all_gather_object(values, normalized)
        raise RuntimeError(f"{name} differs across ranks: {values}")
    return normalized


def _require_matching_bool(
    context: DistributedContext,
    name: str,
    value: bool,
) -> bool:
    _strict_bool(name, value)
    matched = _require_matching_integer(context, name, int(value))
    return bool(matched)


def _require_all_true(
    context: DistributedContext,
    name: str,
    value: bool,
) -> None:
    _strict_bool(name, value)
    if not context.distributed:
        if not value:
            raise RuntimeError(name)
        return
    local = torch.tensor(
        int(value),
        dtype=torch.int32,
        device=_collective_device(context),
    )
    dist.all_reduce(local, op=dist.ReduceOp.MIN)
    if int(local.item()) == 0:
        values: list[bool | None] = [None for _ in range(context.world_size)]
        dist.all_gather_object(values, value)
        failing = [rank for rank, valid in enumerate(values) if valid is False]
        raise RuntimeError(f"{name}; failing ranks={failing}")


def _coordinated_next(
    context: DistributedContext,
    iterator: Any,
    *,
    operation: str,
    expect_item: bool,
) -> Any:
    local_error: Exception | None = None
    ended = False
    item: Any = None
    try:
        item = next(iterator)
    except StopIteration:
        ended = True
    except Exception as exc:
        local_error = exc

    status = 2 if local_error is not None else (1 if ended else 0)
    if context.distributed:
        local_status = torch.tensor(
            status,
            dtype=torch.int32,
            device=_collective_device(context),
        )
        minimum = local_status.clone()
        maximum = local_status.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if int(maximum.item()) == 2:
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
                f"rank {rank}: {failure[0]}: {failure[1]}"
                for rank, failure in enumerate(descriptions)
                if failure is not None
            ]
            raise RuntimeError(
                f"{operation} failed on another rank; " + "; ".join(failures)
            )
        if int(minimum.item()) != int(maximum.item()):
            statuses: list[int | None] = [
                None for _ in range(context.world_size)
            ]
            dist.all_gather_object(statuses, status)
            raise RuntimeError(
                f"{operation} exhausted at different times across ranks: "
                f"{statuses}"
            )
    elif local_error is not None:
        raise local_error

    if expect_item and ended:
        raise RuntimeError(f"{operation} ended before its advertised length")
    if not expect_item and not ended:
        raise RuntimeError(f"{operation} yielded beyond its advertised length")
    return item


def _loss_components(
    output: SemMolPretrainLossOutput,
) -> tuple[LossComponent, ...]:
    return (
        output.mlm_loss,
        output.graph_loss.node,
        output.graph_loss.edge,
        output.graph_loss.structure,
        output.geo_loss.mse,
        output.geo_loss.direction,
        output.pseudo_loss,
        output.alignment_loss,
    )


def _validate_loss_output(
    output: object,
    *,
    device: torch.device,
) -> SemMolPretrainLossOutput:
    if not isinstance(output, SemMolPretrainLossOutput):
        raise TypeError("loss_fn.compute must return SemMolPretrainLossOutput")
    total = output.total_loss
    if not isinstance(total, Tensor) or total.ndim != 0:
        raise ValueError("total_loss must be a scalar tensor")
    if not total.is_floating_point():
        raise TypeError("total_loss must be floating point")
    if total.device != device:
        raise ValueError("total_loss must be on the trainer device")
    for name, component in zip(_COMPONENT_NAMES, _loss_components(output)):
        if not isinstance(component, LossComponent):
            raise TypeError(f"{name} must be a LossComponent")
        if component.numerator.ndim != 0 or not component.numerator.is_floating_point():
            raise ValueError(f"{name}.numerator must be a floating scalar")
        if component.numerator.device != device:
            raise ValueError(f"{name}.numerator must be on the trainer device")
        if not bool(torch.isfinite(component.numerator.detach()).item()):
            raise FloatingPointError(f"non-finite {name} numerator")
        for count_name, count in (
            ("local_count", component.local_count),
            ("global_count", component.global_count),
        ):
            if not isinstance(count, Tensor) or count.ndim != 0:
                raise ValueError(f"{name}.{count_name} must be a scalar tensor")
            if count.is_floating_point() or count.dtype == torch.bool:
                raise TypeError(f"{name}.{count_name} must have integer dtype")
            if count.device != device:
                raise ValueError(f"{name}.{count_name} must be on the trainer device")
            if int(count.detach().item()) < 0:
                raise ValueError(f"{name}.{count_name} cannot be negative")
    if not bool(torch.isfinite(total.detach()).item()):
        raise FloatingPointError("non-finite pretraining total_loss")
    pseudo_scale = output.acsm.pseudo_scale
    if not isinstance(pseudo_scale, Tensor) or pseudo_scale.ndim != 0:
        raise ValueError("pseudo_scale must be a scalar tensor")
    scale = float(pseudo_scale.detach().item())
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("pseudo_scale must be finite and within [0, 1]")
    return output


class _EpochLossAccumulator:
    def __init__(
        self,
        criterion: SemMolPretrainTotalLoss,
        context: DistributedContext,
        epoch: int,
    ) -> None:
        self.criterion = criterion
        self.context = context
        self.epoch = _strict_integer("epoch", epoch, minimum=0)
        device = _collective_device(context)
        self.numerators = torch.zeros(
            len(_COMPONENT_NAMES),
            dtype=torch.float64,
            device=device,
        )
        self.counts = torch.zeros(
            len(_COMPONENT_NAMES),
            dtype=torch.int64,
            device=device,
        )
        self.processed_samples = torch.zeros((), dtype=torch.int64, device=device)
        self.microbatches = 0

    def update(
        self,
        output: SemMolPretrainLossOutput,
        *,
        batch_size: int,
    ) -> None:
        normalized_batch_size = _strict_integer(
            "batch_size",
            batch_size,
            minimum=1,
        )
        for index, component in enumerate(_loss_components(output)):
            self.numerators[index].add_(
                component.numerator.detach().to(
                    device=self.numerators.device,
                    dtype=torch.float64,
                )
            )
            self.counts[index].add_(
                component.local_count.detach().to(
                    device=self.counts.device,
                    dtype=torch.int64,
                )
            )
        self.processed_samples.add_(normalized_batch_size)
        self.microbatches += 1

    def finalize(self) -> tuple[PretrainLossSummary, int, int]:
        if self.microbatches <= 0:
            raise RuntimeError("cannot finalize an empty epoch")
        reduced_numerators = all_reduce_sum(self.numerators)
        reduced_counts = all_reduce_sum(self.counts)
        reduced_samples = all_reduce_sum(self.processed_samples)
        if not isinstance(reduced_numerators, Tensor):
            raise TypeError("reduced numerators must be a tensor")
        if not isinstance(reduced_counts, Tensor):
            raise TypeError("reduced counts must be a tensor")
        if not isinstance(reduced_samples, Tensor):
            raise TypeError("reduced sample count must be a tensor")

        means: list[float] = []
        counts: list[int] = []
        for numerator, count in zip(reduced_numerators, reduced_counts):
            normalized_count = int(count.item())
            counts.append(normalized_count)
            means.append(
                0.0
                if normalized_count == 0
                else float(numerator.item()) / float(normalized_count)
            )
        component = dict(zip(_COMPONENT_NAMES, means))
        graph_loss = (
            self.criterion.graph.node_weight * component["graph_node"]
            + self.criterion.graph.edge_weight * component["graph_edge"]
            + self.criterion.graph.structure_weight
            * component["graph_structure"]
        )
        geo_loss = (
            self.criterion.geo.mse_weight * component["geo_mse"]
            + self.criterion.geo.cosine_weight * component["geo_direction"]
        )
        warmup_epochs = self.criterion.acsm.warmup_epochs
        pseudo_scale = (
            1.0
            if warmup_epochs == 0
            else min(float(self.epoch) / float(warmup_epochs), 1.0)
        )
        total_loss = (
            self.criterion.mlm_weight * component["mlm"]
            + self.criterion.graph_weight * graph_loss
            + self.criterion.geo_weight * geo_loss
            + self.criterion.pseudo_weight
            * pseudo_scale
            * component["pseudo"]
            + self.criterion.alignment_weight * component["alignment"]
        )
        summary = PretrainLossSummary(
            total_loss=total_loss,
            mlm_loss=component["mlm"],
            graph_loss=graph_loss,
            geo_loss=geo_loss,
            pseudo_loss=component["pseudo"],
            alignment_loss=component["alignment"],
            pseudo_scale=pseudo_scale,
            component_counts=tuple(zip(_COMPONENT_NAMES, counts)),
        )
        return (
            summary,
            self.microbatches,
            int(reduced_samples.item()),
        )


class PretrainTrainer:
    """Train SemMol with DCL updates and globally normalized objectives."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: SemMolPretrainTotalLoss,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: PretrainTrainerConfig,
        context: DistributedContext,
        config_fingerprint: str,
        scheduler: object | None = None,
        valid_loader: DataLoader | None = None,
    ) -> None:
        def validate_inputs() -> tuple[SemMol, PrecisionMode]:
            if not isinstance(context, DistributedContext):
                raise TypeError("context must be DistributedContext")
            self._validate_context(context)
            if not isinstance(config, PretrainTrainerConfig):
                raise TypeError("config must be PretrainTrainerConfig")
            if not isinstance(model, nn.Module):
                raise TypeError("model must be an nn.Module")
            core_model = unwrap_model(model)
            if not isinstance(core_model, SemMol):
                raise TypeError("the wrapped model must be SemMol")
            if context.distributed and not isinstance(
                model,
                DistributedDataParallel,
            ):
                raise TypeError(
                    "distributed pretraining requires a DDP-wrapped model"
                )
            if not context.distributed and isinstance(
                model,
                DistributedDataParallel,
            ):
                raise ValueError("a single-process context must not receive DDP")
            self._validate_model_device(model, context.device)
            if not isinstance(loss_fn, SemMolPretrainTotalLoss):
                raise TypeError("loss_fn must be SemMolPretrainTotalLoss")
            if loss_fn.distributed_sync != context.distributed:
                raise ValueError(
                    "loss_fn.distributed_sync must exactly match the trainer "
                    "distributed mode"
                )
            if not isinstance(optimizer, Optimizer):
                raise TypeError("optimizer must be a torch Optimizer")
            self._validate_optimizer(model, optimizer)
            self._validate_scheduler(scheduler, optimizer)
            self._validate_pretraining_contract(core_model, loss_fn)
            self._validate_dcl_contract(core_model, context)
            self._validate_loader("train_loader", train_loader, context)
            if valid_loader is not None:
                self._validate_loader("valid_loader", valid_loader, context)
                if valid_loader is train_loader:
                    raise ValueError(
                        "valid_loader must be distinct from train_loader"
                    )
            if (
                not isinstance(config_fingerprint, str)
                or _FINGERPRINT_PATTERN.fullmatch(config_fingerprint) is None
            ):
                raise ValueError(
                    "config_fingerprint must be a lowercase SHA-256 hex digest"
                )
            return core_model, PrecisionMode(config.precision)

        core_model, precision = _active_distributed_call(
            "PretrainTrainer input validation",
            validate_inputs,
        )

        def prepare_runtime() -> tuple[
            SemMolPretrainTotalLoss,
            torch.cuda.amp.GradScaler | None,
        ]:
            prepared_loss = loss_fn.to(context.device)
            prepared_scaler = precision.create_grad_scaler(context.device)
            return prepared_loss, prepared_scaler

        prepared_loss, prepared_scaler = _active_distributed_call(
            "PretrainTrainer runtime preparation",
            prepare_runtime,
        )

        self.model = model
        self.loss_fn = prepared_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.context = context
        self.config_fingerprint = config_fingerprint
        self.precision = precision
        self.scaler = prepared_scaler
        self._semmol = core_model
        self._state = TrainerState()
        self._resumed_from: Path | None = None
        self._poisoned = False
        self._fit_invoked = False
        self._validate_control_signature()

    @staticmethod
    def _validate_context(context: DistributedContext) -> None:
        active = dist.is_available() and dist.is_initialized()
        active_world = dist.get_world_size() if active else 1
        active_rank = dist.get_rank() if active else 0
        if active_world != context.world_size or active_rank != context.rank:
            raise RuntimeError("context does not match the active process group")
        if context.distributed != (active_world > 1):
            raise RuntimeError("context.distributed does not match world_size")
        if context.distributed and not active:
            raise RuntimeError("distributed context requires an initialized group")
        if (
            context.distributed
            and context.device.type == "cuda"
            and str(dist.get_backend()) != "nccl"
        ):
            raise ValueError("CUDA pretraining requires NCCL distributed collectives")

    def _validate_control_signature(self) -> None:
        if not self.context.distributed:
            return
        scheduler_type = (
            None
            if self.scheduler is None
            else (
                f"{type(self.scheduler).__module__}."
                f"{type(self.scheduler).__qualname__}"
            )
        )
        signature = {
            "epochs": self.config.epochs,
            "accumulation_steps": self.config.accumulation_steps,
            "precision": self.config.precision,
            "max_grad_norm": self.config.max_grad_norm,
            "non_blocking": self.config.non_blocking,
            "checkpoint_directory": str(self.config.checkpoint.directory),
            "save_every_n_epochs": (
                self.config.checkpoint.save_every_n_epochs
            ),
            "latest_filename": self.config.checkpoint.latest_filename,
            "best_filename": self.config.checkpoint.best_filename,
            "periodic_prefix": self.config.checkpoint.periodic_prefix,
            "config_fingerprint": self.config_fingerprint,
            "has_validation": self.valid_loader is not None,
            "scheduler_type": scheduler_type,
            "scaler_present": self.scaler is not None,
            "dcl_configuration": tuple(
                (
                    modality,
                    self._dcl_control_signature(
                        self._semmol.dcls[modality]
                    ),
                )
                for modality in self._semmol.target_modalities
            ),
            "loss_weights": (
                self.loss_fn.mlm_weight,
                self.loss_fn.graph_weight,
                self.loss_fn.geo_weight,
                self.loss_fn.pseudo_weight,
                self.loss_fn.alignment_weight,
                self.loss_fn.graph.node_weight,
                self.loss_fn.graph.edge_weight,
                self.loss_fn.graph.structure_weight,
                self.loss_fn.geo.mse_weight,
                self.loss_fn.geo.cosine_weight,
                self.loss_fn.acsm.warmup_epochs,
            ),
        }
        signatures: list[dict[str, Any] | None] = [
            None for _ in range(self.context.world_size)
        ]
        dist.all_gather_object(signatures, signature)
        if any(candidate != signature for candidate in signatures):
            raise RuntimeError(
                "pretraining control configuration differs across ranks: "
                f"{signatures}"
            )

    @staticmethod
    def _validate_model_device(model: nn.Module, device: torch.device) -> None:
        misplaced = [
            name
            for name, value in (
                *tuple(model.named_parameters()),
                *tuple(model.named_buffers()),
            )
            if value.device != device
        ]
        if misplaced:
            preview = misplaced[:8]
            suffix = "..." if len(misplaced) > len(preview) else ""
            raise ValueError(
                f"model parameters/buffers are not all on {device}: "
                f"{preview}{suffix}"
            )

    @staticmethod
    def _validate_optimizer(model: nn.Module, optimizer: Optimizer) -> None:
        model_parameters = {
            id(parameter): parameter for parameter in model.parameters()
        }
        optimized: set[int] = set()
        for group_index, group in enumerate(optimizer.param_groups):
            parameters = group.get("params")
            if not isinstance(parameters, list):
                raise TypeError(
                    f"optimizer.param_groups[{group_index}]['params'] must be a list"
                )
            for parameter in parameters:
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError("optimizer parameters must be nn.Parameter")
                identity = id(parameter)
                if identity not in model_parameters:
                    raise ValueError("optimizer contains a parameter outside the model")
                if identity in optimized:
                    raise ValueError("optimizer contains a duplicate parameter")
                optimized.add(identity)
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in optimized
        ]
        if missing:
            preview = missing[:8]
            suffix = "..." if len(missing) > len(preview) else ""
            raise ValueError(
                f"optimizer omits trainable model parameters: {preview}{suffix}"
            )

    @staticmethod
    def _validate_scheduler(
        scheduler: object | None,
        optimizer: Optimizer,
    ) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, ReduceLROnPlateau):
            raise TypeError(
                "ReduceLROnPlateau is unsupported because pretraining steps the "
                "scheduler without a validation metric"
            )
        for method in ("step", "state_dict", "load_state_dict"):
            if not callable(getattr(scheduler, method, None)):
                raise TypeError(f"scheduler must provide callable {method}()")
        missing = object()
        scheduler_optimizer = getattr(scheduler, "optimizer", missing)
        if scheduler_optimizer is missing:
            raise TypeError(
                "scheduler must expose the optimizer through scheduler.optimizer"
            )
        if scheduler_optimizer is not optimizer:
            raise ValueError(
                "scheduler.optimizer must be the trainer optimizer"
            )
        step = getattr(scheduler, "step")
        try:
            step_signature = inspect.signature(step)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "scheduler.step signature must be inspectable"
            ) from exc
        try:
            step_signature.bind()
        except TypeError as exc:
            raise TypeError(
                "scheduler.step must be callable without a metric or epoch "
                "argument"
            ) from exc

    @staticmethod
    def _validate_pretraining_contract(
        model: SemMol,
        loss_fn: SemMolPretrainTotalLoss,
    ) -> None:
        if model.property_head is not None:
            raise ValueError(
                "pretraining requires a SemMol model without a task head"
            )
        if "qm" in model.modalities:
            raise ValueError("QM has no implemented SemMol pretraining objective")
        required_heads = {
            "mlm": loss_fn.mlm_weight,
            "graph": loss_fn.graph_weight,
            "geo": loss_fn.geo_weight,
        }
        missing = [
            name
            for name, weight in required_heads.items()
            if weight > 0.0 and name not in model.pretraining_heads
        ]
        if missing:
            raise ValueError(
                "positive reconstruction loss weights require matching model "
                f"pretraining heads: {missing}"
            )

    @staticmethod
    def _dcl_control_signature(
        library: DynamicCentralLibrary,
    ) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (name, getattr(library, name))
            for name in _DCL_CONTROL_ATTRIBUTES
        )

    @staticmethod
    def _validate_dcl_contract(
        model: SemMol,
        context: DistributedContext,
    ) -> None:
        target_modalities = tuple(model.target_modalities)
        if tuple(model.dcls.keys()) != target_modalities:
            raise ValueError(
                "SemMol DCL keys must exactly follow target_modalities"
            )
        synchronization_flags: list[bool] = []
        control_signatures: list[tuple[tuple[str, Any], ...]] = []
        for modality in target_modalities:
            library = model.dcls[modality]
            if not isinstance(library, DynamicCentralLibrary):
                raise TypeError(
                    f"model.dcls[{modality!r}] must be DynamicCentralLibrary"
                )
            if not isinstance(library.distributed_sync, bool):
                raise TypeError(
                    f"model.dcls[{modality!r}].distributed_sync must be bool"
                )
            synchronization_flags.append(library.distributed_sync)
            control_signatures.append(
                PretrainTrainer._dcl_control_signature(library)
            )
        if len(set(control_signatures)) != 1:
            raise ValueError(
                "all target DCLs must use identical runtime configuration"
            )
        if len(set(synchronization_flags)) != 1:
            raise ValueError(
                "all target DCLs must use the same distributed_sync setting"
            )
        if context.distributed and not all(synchronization_flags):
            raise ValueError(
                "distributed pretraining requires distributed_sync=True for "
                "every target DCL"
            )

    @staticmethod
    def _validate_loader(
        name: str,
        loader: DataLoader,
        context: DistributedContext,
    ) -> None:
        if not isinstance(loader, DataLoader):
            raise TypeError(f"{name} must be a DataLoader")
        if not isinstance(loader.collate_fn, PretrainingDataCollator):
            raise TypeError(
                f"{name}.collate_fn must be PretrainingDataCollator so real "
                "self-supervised labels are present"
            )
        if not isinstance(loader.generator, torch.Generator):
            raise ValueError(
                f"{name} must have an explicit torch.Generator for resume"
            )
        sampler = loader.sampler
        if context.distributed:
            if not isinstance(sampler, DistributedSampler):
                raise TypeError(f"{name} must use DistributedSampler under DDP")
            if sampler.num_replicas != context.world_size:
                raise ValueError(f"{name} sampler world size differs from context")
            if sampler.rank != context.rank:
                raise ValueError(f"{name} sampler rank differs from context")

    @property
    def state(self) -> TrainerState:
        return self._state

    @property
    def is_usable(self) -> bool:
        """Whether model, optimizer, DCL, and trainer state may still be used."""

        return not self._poisoned

    def _require_usable(self, operation: str) -> None:
        if self._poisoned:
            raise RuntimeError(
                f"cannot {operation}: this trainer is fail-closed after an "
                "earlier mutable operation failed; rebuild all training objects"
            )

    def _mark_unusable(self) -> None:
        self._poisoned = True

    def _loaders(self) -> dict[str, DataLoader]:
        loaders = {"train": self.train_loader}
        if self.valid_loader is not None:
            loaders["valid"] = self.valid_loader
        return loaders

    def _checkpoint_extra(self) -> dict[str, Any]:
        return {
            **_CHECKPOINT_EXTRA,
            "has_validation": self.valid_loader is not None,
        }

    def _prepare_checkpoint_directory(self) -> None:
        directory = Path(self.config.checkpoint.directory)

        def prepare() -> None:
            directory.mkdir(parents=True, exist_ok=True)
            if not directory.is_dir():
                raise NotADirectoryError(
                    f"checkpoint directory is not a directory: {directory}"
                )

        _distributed_call(
            self.context,
            "checkpoint directory creation",
            prepare,
        )

    def resume(self, path: str | Path) -> TrainingCheckpointLoadResult:
        def validate_resume_request() -> None:
            self._require_usable("resume")
            if self._resumed_from is not None:
                raise RuntimeError("this trainer has already resumed a checkpoint")
            if self._state.next_epoch != 0 or self._state.optimizer_step != 0:
                raise RuntimeError(
                    "resume is only allowed before this trainer has run"
                )
            if self._has_any_gradient():
                raise RuntimeError(
                    "resume requires every trainable parameter gradient to be "
                    "None before checkpoint validation"
                )

        _distributed_call(
            self.context,
            "resume request validation",
            validate_resume_request,
        )

        def validate_checkpoint_metadata(
            extra: Mapping[str, Any],
            state: TrainerState,
        ) -> None:
            if dict(extra) != self._checkpoint_extra():
                raise ValueError(
                    "checkpoint trainer metadata does not match this pretrainer"
                )
            if state.next_epoch > self.config.epochs:
                raise ValueError(
                    "checkpoint next_epoch exceeds configured training epochs"
                )
            if self.valid_loader is None:
                if state.best_epoch != -1:
                    raise ValueError(
                        "a trainer without validation cannot resume a best epoch"
                    )
                if state.bad_epochs != 0:
                    raise ValueError(
                        "a trainer without validation cannot resume bad epochs"
                    )
                if state.best_metric != 0.0:
                    raise ValueError(
                        "a trainer without validation cannot resume a best metric"
                    )

        try:
            result = load_training_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                config_fingerprint=self.config_fingerprint,
                loaders=self._loaders(),
                context=self.context,
                map_location=self.context.device,
                metadata_validator=validate_checkpoint_metadata,
            )
            _distributed_call(
                self.context,
                "post-resume gradient clearing",
                lambda: self.optimizer.zero_grad(set_to_none=True),
            )
        except BaseException:
            # Phase B may have partially changed model/optimizer/runtime state.
            # Without duplicating the full model, the only safe contract is to
            # make this trainer permanently unusable after any load failure.
            self._mark_unusable()
            raise
        self._state = result.state
        self._resumed_from = result.path
        return result

    def _loader_length(self, name: str, loader: DataLoader) -> int:
        def local_length() -> int:
            length = len(loader)
            return _strict_integer(f"len({name})", length, minimum=1)

        length = _distributed_call(
            self.context,
            f"{name} length validation",
            local_length,
        )
        return _require_matching_integer(self.context, f"len({name})", length)

    def _batch_tensor(
        self,
        batch: Mapping[str, Any],
        key: str,
        *,
        dtype: torch.dtype | None = None,
        ndim: int | None = None,
    ) -> Tensor:
        value = batch.get(key)
        if not isinstance(value, Tensor):
            raise TypeError(f"batch[{key!r}] must be a Tensor")
        if value.device != self.context.device:
            raise ValueError(
                f"batch[{key!r}] must be on {self.context.device}, got "
                f"{value.device}"
            )
        if dtype is not None and value.dtype != dtype:
            raise TypeError(
                f"batch[{key!r}] must use {dtype}, got {value.dtype}"
            )
        if ndim is not None and value.ndim != ndim:
            raise ValueError(
                f"batch[{key!r}] must have {ndim} dimensions, got "
                f"shape {tuple(value.shape)}"
            )
        return value

    @staticmethod
    def _nonempty_string_sequence(
        batch: Mapping[str, Any],
        key: str,
        *,
        length: int | None = None,
    ) -> tuple[str, ...]:
        value = batch.get(key)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"batch[{key!r}] must be a sequence of strings")
        normalized = tuple(value)
        if length is not None and len(normalized) != length:
            raise ValueError(
                f"batch[{key!r}] must contain {length} entries, got "
                f"{len(normalized)}"
            )
        if any(not isinstance(item, str) or not item for item in normalized):
            raise ValueError(
                f"batch[{key!r}] entries must be non-empty strings"
            )
        return normalized

    def _validate_batch_metadata(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[int, Tensor]:
        sample_ids = self._nonempty_string_sequence(batch, "sample_id")
        batch_size = len(sample_ids)
        if batch_size <= 0:
            raise ValueError("pretraining batches cannot be empty")
        self._nonempty_string_sequence(batch, "smiles", length=batch_size)

        source_index = self._batch_tensor(
            batch,
            "source_index",
            dtype=torch.long,
            ndim=1,
        )
        if source_index.shape != (batch_size,):
            raise ValueError("batch['source_index'] must have shape [batch]")
        if bool(torch.any(source_index < 0).item()):
            raise ValueError("batch['source_index'] cannot contain negative values")
        if "record_index" in batch:
            record_index = self._batch_tensor(
                batch,
                "record_index",
                dtype=torch.long,
                ndim=1,
            )
            if record_index.shape != (batch_size,):
                raise ValueError("batch['record_index'] must have shape [batch]")
            if bool(torch.any(record_index < 0).item()):
                raise ValueError(
                    "batch['record_index'] cannot contain negative values"
                )

        modality_mask = self._batch_tensor(
            batch,
            "modality_mask",
            dtype=torch.bool,
            ndim=2,
        )
        if modality_mask.shape != (batch_size, len(_MODALITY_ORDER)):
            raise ValueError(
                "batch['modality_mask'] must have shape "
                f"[{batch_size}, {len(_MODALITY_ORDER)}]"
            )

        enabled = set(self._semmol.modalities)
        for modality in _MODALITY_ORDER:
            column = _MODALITY_COLUMN[modality]
            present = bool(modality_mask[:, column].any().item())
            input_keys = _MODALITY_INPUT_KEYS[modality]
            target_keys = _MODALITY_TARGET_KEYS[modality]
            all_keys = (
                input_keys
                + target_keys
                + _MODALITY_OPTIONAL_KEYS[modality]
            )
            existing = tuple(key for key in all_keys if key in batch)
            if modality not in enabled:
                if present or existing:
                    raise ValueError(
                        f"disabled modality {modality!r} appears in the batch"
                    )
                continue
            if present:
                missing_inputs = tuple(
                    key for key in input_keys if key not in batch
                )
                missing_targets = tuple(
                    key for key in target_keys if key not in batch
                )
                if missing_inputs or missing_targets:
                    raise KeyError(
                        f"modality {modality!r} is present but its batch schema "
                        f"is incomplete; missing_inputs={missing_inputs}, "
                        f"missing_targets={missing_targets}"
                    )
            elif existing:
                raise ValueError(
                    f"modality {modality!r} has no present rows but batch keys "
                    f"were supplied: {existing}"
                )
        return batch_size, modality_mask

    def _validate_1d_batch(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
        expected_presence: Tensor,
    ) -> None:
        if not bool(expected_presence.any().item()):
            return
        input_ids = self._batch_tensor(
            batch,
            "input_ids",
            dtype=torch.long,
            ndim=2,
        )
        attention_mask = self._batch_tensor(
            batch,
            "attention_mask",
            dtype=torch.bool,
            ndim=2,
        )
        mlm_labels = self._batch_tensor(
            batch,
            "mlm_labels",
            dtype=torch.long,
            ndim=2,
        )
        if input_ids.shape[0] != batch_size or input_ids.shape[1] <= 0:
            raise ValueError(
                "batch['input_ids'] must have shape [batch, positive_length]"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids")
        if mlm_labels.shape != input_ids.shape:
            raise ValueError("mlm_labels shape must match input_ids")
        if not torch.equal(attention_mask.any(dim=1), expected_presence):
            raise ValueError(
                "attention_mask row presence disagrees with modality_mask[:, 0]"
            )

        encoder = self._semmol.encoders["1d"]
        vocab_size = getattr(encoder, "vocab_size", None)
        tokenizer = getattr(encoder, "espf_tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(vocab_size, Integral) or isinstance(vocab_size, bool):
            raise TypeError("the 1d encoder must expose an integer vocab_size")
        if not isinstance(pad_token_id, Integral) or isinstance(
            pad_token_id,
            bool,
        ):
            raise TypeError("the 1d encoder tokenizer must expose pad_token_id")
        vocabulary_size = int(vocab_size)
        padding_id = int(pad_token_id)
        transformer = getattr(encoder, "transformer", None)
        transformer_config = getattr(transformer, "config", None)
        transformer_padding_id = getattr(
            transformer_config,
            "pad_token_id",
            None,
        )
        transformer_embeddings = getattr(transformer, "embeddings", None)
        position_embeddings = getattr(
            transformer_embeddings,
            "position_embeddings",
            None,
        )
        position_count = getattr(position_embeddings, "num_embeddings", None)
        if not isinstance(transformer_padding_id, int) or isinstance(
            transformer_padding_id,
            bool,
        ):
            raise TypeError(
                "the 1d encoder transformer must expose an integer "
                "config.pad_token_id"
            )
        if not isinstance(position_count, Integral) or isinstance(
            position_count,
            bool,
        ):
            raise TypeError(
                "the 1d encoder transformer must expose an integer position "
                "embedding count"
            )
        encoder_mode = getattr(encoder, "mode", None)
        if encoder_mode == "scratch" and transformer_padding_id != padding_id:
            raise ValueError(
                "the 1d tokenizer and transformer pad-token identifiers differ"
            )
        maximum_position = transformer_padding_id + int(input_ids.shape[1])
        if maximum_position >= int(position_count):
            capacity = int(position_count) - transformer_padding_id - 1
            raise ValueError(
                f"input_ids sequence width {input_ids.shape[1]} exceeds the "
                f"1d encoder position capacity {capacity}"
            )
        if bool(torch.any(input_ids < 0).item()) or bool(
            torch.any(input_ids >= vocabulary_size).item()
        ):
            raise ValueError(
                f"input_ids must be within [0, {vocabulary_size - 1}]"
            )
        if bool(torch.any(input_ids[~attention_mask] != padding_id).item()):
            raise ValueError("inactive token positions must contain the pad token")
        if bool(torch.any(input_ids[attention_mask] == padding_id).item()):
            raise ValueError("active token positions cannot contain the pad token")
        if input_ids.shape[1] > 1 and bool(
            torch.any(attention_mask[:, 1:] & ~attention_mask[:, :-1]).item()
        ):
            raise ValueError("attention_mask must use contiguous right padding")
        cls_token_id = getattr(tokenizer, "cls_token_id", None)
        sep_token_id = getattr(tokenizer, "sep_token_id", None)
        if (
            not isinstance(cls_token_id, Integral)
            or isinstance(cls_token_id, bool)
            or not isinstance(sep_token_id, Integral)
            or isinstance(sep_token_id, bool)
        ):
            raise TypeError(
                "the 1d encoder tokenizer must expose integer CLS and SEP ids"
            )
        present_rows = torch.nonzero(
            expected_presence,
            as_tuple=False,
        ).flatten()
        if bool(
            torch.any(
                input_ids.index_select(0, present_rows)[:, 0]
                != int(cls_token_id)
            ).item()
        ):
            raise ValueError("every present 1d row must begin with the CLS token")
        active_lengths = attention_mask.sum(dim=1, dtype=torch.long)
        last_positions = active_lengths.index_select(0, present_rows) - 1
        last_tokens = input_ids[present_rows, last_positions]
        if bool(torch.any(last_tokens != int(sep_token_id)).item()):
            raise ValueError("every present 1d row must end with the SEP token")

        selected = mlm_labels != self.loss_fn.mlm.ignore_index
        if bool(torch.any(selected & ~attention_mask).item()):
            raise ValueError("MLM labels can only select active token positions")
        valid_labels = mlm_labels[selected]
        if valid_labels.numel() > 0 and (
            bool(torch.any(valid_labels < 0).item())
            or bool(torch.any(valid_labels >= vocabulary_size).item())
        ):
            raise ValueError(
                f"non-ignored MLM labels must be within [0, {vocabulary_size - 1}]"
            )
        if "token_spans" in batch:
            token_spans = self._batch_tensor(
                batch,
                "token_spans",
                dtype=torch.long,
                ndim=3,
            )
            if tuple(token_spans.shape) != (*tuple(input_ids.shape), 2):
                raise ValueError(
                    "token_spans must have shape [batch, length, 2]"
                )
            sentinel = torch.all(token_spans == -1, dim=-1)
            partially_negative = torch.any(token_spans < 0, dim=-1) & ~sentinel
            if bool(torch.any(partially_negative).item()):
                raise ValueError(
                    "negative token spans must use the complete (-1, -1) sentinel"
                )
            concrete = ~sentinel
            if bool(
                torch.any(
                    token_spans[..., 1][concrete]
                    < token_spans[..., 0][concrete]
                ).item()
            ):
                raise ValueError("token span ends cannot precede their starts")
            if bool(torch.any(~sentinel[~attention_mask]).item()):
                raise ValueError("inactive token positions must use sentinel spans")

    def _validate_categorical_matrix(
        self,
        name: str,
        values: Tensor,
        cardinalities: tuple[int, ...],
        *,
        allow_mask_token: bool,
    ) -> None:
        for column, cardinality in enumerate(cardinalities):
            field = values[:, column]
            upper_bound = cardinality if allow_mask_token else cardinality - 1
            if bool(torch.any(field < 0).item()) or bool(
                torch.any(field >= upper_bound).item()
            ):
                raise ValueError(
                    f"{name} column {column} must be within "
                    f"[0, {upper_bound - 1}]"
                )

    def _validate_masked_categorical_targets(
        self,
        *,
        name: str,
        corrupted: Tensor,
        mask: Tensor,
        labels: Tensor,
        cardinalities: tuple[int, ...],
    ) -> None:
        if mask.shape != (corrupted.shape[0],):
            raise ValueError(f"{name}_mask must have one value per row")
        if labels.shape != corrupted.shape:
            raise ValueError(f"{name}_labels shape must match corrupted features")
        ignore_index = self.loss_fn.graph.ignore_index
        if bool(torch.any(labels[~mask] != ignore_index).item()):
            raise ValueError(f"unmasked {name} labels must use ignore_index")
        if bool(torch.any(labels[mask] == ignore_index).item()):
            raise ValueError(f"masked {name} labels cannot use ignore_index")
        mask_tokens = corrupted.new_tensor(
            [cardinality - 1 for cardinality in cardinalities]
        )
        has_masked_rows = bool(mask.any().item())
        has_unmasked_rows = bool((~mask).any().item())
        if has_masked_rows and not torch.equal(
            corrupted[mask],
            mask_tokens.expand(int(mask.sum().item()), -1),
        ):
            raise ValueError(
                f"masked {name} features must contain the configured mask tokens"
            )
        if has_unmasked_rows:
            self._validate_categorical_matrix(
                f"unmasked {name} features",
                corrupted[~mask],
                cardinalities,
                allow_mask_token=False,
            )
        if has_masked_rows:
            self._validate_categorical_matrix(
                f"masked {name} labels",
                labels[mask],
                cardinalities,
                allow_mask_token=False,
            )

    def _validate_2d_batch(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
        expected_presence: Tensor,
    ) -> None:
        if not bool(expected_presence.any().item()):
            return
        graph = batch.get("graph")
        if not isinstance(graph, Batch):
            raise TypeError("batch['graph'] must be a PyG Batch")
        graph_sample_index = self._batch_tensor(
            batch,
            "graph_sample_index",
            dtype=torch.long,
            ndim=1,
        )
        expected_index = torch.nonzero(
            expected_presence,
            as_tuple=False,
        ).flatten()
        if not torch.equal(graph_sample_index, expected_index):
            raise ValueError(
                "graph_sample_index must exactly match modality_mask[:, 1]"
            )
        graph_count = int(graph_sample_index.numel())
        if int(graph.num_graphs) != graph_count:
            raise ValueError(
                "graph.num_graphs must equal graph_sample_index length"
            )

        graph_tensors: dict[str, Tensor] = {}
        for key, ndim, dtype in (
            ("x", 2, torch.long),
            ("edge_index", 2, torch.long),
            ("edge_attr", 2, torch.long),
            ("batch", 1, torch.long),
            ("ptr", 1, torch.long),
        ):
            value = getattr(graph, key, None)
            if not isinstance(value, Tensor):
                raise TypeError(f"batch['graph'].{key} must be a Tensor")
            if value.ndim != ndim or value.dtype != dtype:
                raise ValueError(
                    f"batch['graph'].{key} must be a {ndim}D {dtype} tensor"
                )
            if value.device != self.context.device:
                raise ValueError(
                    f"batch['graph'].{key} must be on {self.context.device}"
                )
            graph_tensors[key] = value
        node_features = graph_tensors["x"]
        edge_index = graph_tensors["edge_index"]
        edge_features = graph_tensors["edge_attr"]
        node_batch = graph_tensors["batch"]
        graph_ptr = graph_tensors["ptr"]

        graph_encoder = self._semmol.encoders["2d"]
        node_cardinalities = tuple(
            int(value)
            for value in graph_encoder.node_feature_cardinalities
        )
        edge_cardinalities = tuple(
            int(value)
            for value in graph_encoder.edge_feature_cardinalities
        )
        if node_features.shape[1] != len(node_cardinalities):
            raise ValueError("graph.x field count differs from the graph encoder")
        if edge_index.shape[0] != 2:
            raise ValueError("graph.edge_index must have shape [2, edges]")
        if edge_features.shape != (
            edge_index.shape[1],
            len(edge_cardinalities),
        ):
            raise ValueError(
                "graph.edge_attr must have one row per edge and the configured "
                "field count"
            )
        node_count = int(node_features.shape[0])
        edge_count = int(edge_index.shape[1])
        if graph.num_nodes is None or int(graph.num_nodes) != node_count:
            raise ValueError("graph.num_nodes must equal graph.x row count")
        if node_batch.shape != (node_count,):
            raise ValueError("graph.batch must have one entry per node")
        if graph_ptr.shape != (graph_count + 1,):
            raise ValueError("graph.ptr must have graph_count + 1 entries")
        if (
            int(graph_ptr[0].item()) != 0
            or int(graph_ptr[-1].item()) != node_count
        ):
            raise ValueError("graph.ptr must span all graph nodes")
        node_counts = graph_ptr[1:] - graph_ptr[:-1]
        if bool(torch.any(node_counts <= 0).item()):
            raise ValueError("every compact molecular graph must contain a node")
        expected_node_batch = torch.repeat_interleave(
            torch.arange(
                graph_count,
                dtype=torch.long,
                device=self.context.device,
            ),
            node_counts,
            output_size=node_count,
        )
        if not torch.equal(node_batch, expected_node_batch):
            raise ValueError("graph.batch is inconsistent with graph.ptr")
        if edge_count > 0:
            if bool(torch.any(edge_index < 0).item()) or bool(
                torch.any(edge_index >= node_count).item()
            ):
                raise ValueError("graph.edge_index contains invalid node indices")
            if bool(
                torch.any(
                    node_batch[edge_index[0]] != node_batch[edge_index[1]]
                ).item()
            ):
                raise ValueError("graph edges cannot connect different graphs")

        self._validate_categorical_matrix(
            "graph.x",
            node_features,
            node_cardinalities,
            allow_mask_token=True,
        )
        self._validate_categorical_matrix(
            "graph.edge_attr",
            edge_features,
            edge_cardinalities,
            allow_mask_token=True,
        )
        node_mask = self._batch_tensor(
            batch,
            "node_mask",
            dtype=torch.bool,
            ndim=1,
        )
        node_labels = self._batch_tensor(
            batch,
            "node_labels",
            dtype=torch.long,
            ndim=2,
        )
        edge_mask = self._batch_tensor(
            batch,
            "edge_mask",
            dtype=torch.bool,
            ndim=1,
        )
        edge_labels = self._batch_tensor(
            batch,
            "edge_labels",
            dtype=torch.long,
            ndim=2,
        )
        if edge_mask.shape != (edge_count,):
            raise ValueError("edge_mask must have one value per graph edge")
        if edge_labels.shape != edge_features.shape:
            raise ValueError("edge_labels shape must match graph.edge_attr")
        self._validate_masked_categorical_targets(
            name="node",
            corrupted=node_features,
            mask=node_mask,
            labels=node_labels,
            cardinalities=node_cardinalities,
        )
        self._validate_masked_categorical_targets(
            name="edge",
            corrupted=edge_features,
            mask=edge_mask,
            labels=edge_labels,
            cardinalities=edge_cardinalities,
        )

    def _validate_3d_batch(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
        expected_presence: Tensor,
    ) -> None:
        if not bool(expected_presence.any().item()):
            return
        atomic_numbers = self._batch_tensor(
            batch,
            "atomic_numbers",
            dtype=torch.long,
            ndim=2,
        )
        coords = self._batch_tensor(
            batch,
            "coords",
            dtype=torch.float32,
            ndim=4,
        )
        atom_mask = self._batch_tensor(
            batch,
            "atom_mask",
            dtype=torch.bool,
            ndim=2,
        )
        conformer_mask = self._batch_tensor(
            batch,
            "conformer_mask",
            dtype=torch.bool,
            ndim=2,
        )
        clean_coords = self._batch_tensor(
            batch,
            "clean_coords",
            dtype=torch.float32,
            ndim=4,
        )
        coord_noise = self._batch_tensor(
            batch,
            "coord_noise",
            dtype=torch.float32,
            ndim=4,
        )
        if atomic_numbers.shape[0] != batch_size or atomic_numbers.shape[1] <= 0:
            raise ValueError(
                "atomic_numbers must have shape [batch, positive_atom_count]"
            )
        atom_count = int(atomic_numbers.shape[1])
        if (
            coords.shape[0] != batch_size
            or coords.shape[1] <= 0
            or coords.shape[2:] != (atom_count, 3)
        ):
            raise ValueError(
                "coords must have shape [batch, positive_conformers, atoms, 3]"
            )
        conformer_count = int(coords.shape[1])
        if atom_mask.shape != (batch_size, atom_count):
            raise ValueError("atom_mask shape must match atomic_numbers")
        if conformer_mask.shape != (batch_size, conformer_count):
            raise ValueError("conformer_mask shape must match coords")
        if (
            clean_coords.shape != coords.shape
            or coord_noise.shape != coords.shape
        ):
            raise ValueError("clean_coords and coord_noise must match coords shape")
        if not torch.equal(atom_mask.any(dim=1), expected_presence):
            raise ValueError(
                "atom_mask presence disagrees with modality_mask[:, 2]"
            )
        if not torch.equal(conformer_mask.any(dim=1), expected_presence):
            raise ValueError(
                "conformer_mask presence disagrees with modality_mask[:, 2]"
            )
        if atom_count > 1 and bool(
            torch.any(atom_mask[:, 1:] & ~atom_mask[:, :-1]).item()
        ):
            raise ValueError("atom_mask must use contiguous right padding")
        if bool(torch.any(atomic_numbers[~atom_mask] != 0).item()):
            raise ValueError("padded atomic numbers must be zero")
        valid_atomic_numbers = atomic_numbers[atom_mask]
        if bool(torch.any(valid_atomic_numbers < 1).item()) or bool(
            torch.any(valid_atomic_numbers > 118).item()
        ):
            raise ValueError("valid atomic numbers must be within [1, 118]")
        for name, value in (
            ("coords", coords),
            ("clean_coords", clean_coords),
            ("coord_noise", coord_noise),
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must contain only finite values")
        padded_atoms = ~atom_mask[:, None, :].expand(
            batch_size,
            conformer_count,
            atom_count,
        )
        if bool(torch.any(coords[padded_atoms] != 0).item()):
            raise ValueError("coordinates for padded atoms must be zero")
        if bool(torch.any(clean_coords[padded_atoms] != 0).item()):
            raise ValueError("clean coordinates for padded atoms must be zero")
        valid_vectors = conformer_mask[:, :, None] & atom_mask[:, None, :]
        if bool(torch.any(coord_noise[~valid_vectors] != 0).item()):
            raise ValueError("coordinate noise outside valid atoms must be zero")
        if not torch.allclose(
            coords,
            clean_coords + coord_noise,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError("coords must equal clean_coords + coord_noise")

    def _validate_batch_schema(self, batch: Mapping[str, Any]) -> None:
        batch_size, modality_mask = self._validate_batch_metadata(batch)
        if "1d" in self._semmol.modalities:
            self._validate_1d_batch(
                batch,
                batch_size=batch_size,
                expected_presence=modality_mask[:, _MODALITY_COLUMN["1d"]],
            )
        if "2d" in self._semmol.modalities:
            self._validate_2d_batch(
                batch,
                batch_size=batch_size,
                expected_presence=modality_mask[:, _MODALITY_COLUMN["2d"]],
            )
        if "3d" in self._semmol.modalities:
            self._validate_3d_batch(
                batch,
                batch_size=batch_size,
                expected_presence=modality_mask[:, _MODALITY_COLUMN["3d"]],
            )

    def _prepare_batch(self, batch: object) -> Mapping[str, Any]:
        if not isinstance(batch, Mapping):
            raise TypeError("pretraining batches must be mappings")
        moved = move_batch_to_device(
            batch,
            self.context.device,
            non_blocking=self.config.non_blocking,
        )
        if not isinstance(moved, Mapping):
            raise TypeError("moved pretraining batch must remain a mapping")
        self._validate_batch_schema(moved)
        return moved

    @staticmethod
    def _validate_model_output(output: object) -> SemMolPretrainingOutput:
        if not isinstance(output, SemMolPretrainingOutput):
            raise TypeError(
                "model(batch, mode='pretrain') must return "
                "SemMolPretrainingOutput"
            )
        _strict_integer("outputs.batch_size", output.batch_size, minimum=1)
        return output

    def _compute_loss(
        self,
        batch: Mapping[str, Any],
        *,
        epoch: int,
        update_dcl: bool,
    ) -> tuple[SemMolPretrainingOutput, SemMolPretrainLossOutput]:
        outputs = self.model(
            batch,
            mode="pretrain",
            update_dcl=update_dcl,
        )
        prepared_outputs = _distributed_call(
            self.context,
            "pretraining model output validation",
            lambda: self._validate_model_output(outputs),
        )
        loss_output = self.loss_fn.compute(
            prepared_outputs,
            batch,
            epoch=epoch,
        )
        prepared_loss = _distributed_call(
            self.context,
            "pretraining finite-loss validation",
            lambda: _validate_loss_output(
                loss_output,
                device=self.context.device,
            ),
        )
        return prepared_outputs, prepared_loss

    def _set_epoch(self, name: str, loader: DataLoader, epoch: int) -> None:
        _distributed_call(
            self.context,
            f"{name} epoch update",
            lambda: set_dataloader_epoch(loader, epoch),
        )

    def _learning_rates(self) -> tuple[float, ...]:
        rates: list[float] = []
        for index, group in enumerate(self.optimizer.param_groups):
            if "lr" not in group:
                raise KeyError(f"optimizer.param_groups[{index}] has no lr")
            rates.append(
                _finite_float(
                    f"optimizer.param_groups[{index}].lr",
                    group["lr"],
                    minimum=0.0,
                )
            )
        if not rates:
            raise RuntimeError("optimizer must contain at least one parameter group")
        return tuple(rates)

    def _has_any_gradient(self) -> bool:
        return any(
            parameter.grad is not None
            for parameter in self.model.parameters()
        )

    def _gradients_are_finite(self) -> bool:
        for parameter in self.model.parameters():
            gradient = parameter.grad
            if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
                return False
        return True

    def _optimizer_step(self) -> bool:
        _distributed_call(
            self.context,
            "gradient unscale",
            lambda: PrecisionMode.unscale_for_clipping(
                self.scaler,
                self.optimizer,
            ),
        )
        _require_all_true(
            self.context,
            "no gradients were produced for an optimizer step",
            self._has_any_gradient(),
        )
        if self.config.max_grad_norm is not None:
            _distributed_call(
                self.context,
                "gradient clipping",
                lambda: nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.max_grad_norm,
                    error_if_nonfinite=False,
                ),
            )
        if self.scaler is None:
            _require_all_true(
                self.context,
                "non-finite gradients detected without GradScaler",
                self._gradients_are_finite(),
            )

        stepped = _distributed_call(
            self.context,
            "optimizer step",
            lambda: PrecisionMode.step_optimizer(self.scaler, self.optimizer),
        )
        stepped = _require_matching_bool(
            self.context,
            "optimizer step decision",
            stepped,
        )
        if stepped and self.scheduler is not None:
            _distributed_call(
                self.context,
                "scheduler step",
                self.scheduler.step,
            )
        _distributed_call(
            self.context,
            "post-step gradient clearing",
            lambda: self.optimizer.zero_grad(set_to_none=True),
        )
        return stepped

    def _run_train_epoch(
        self,
        epoch: int,
        *,
        progress_interval: int | None = None,
        progress_callbacks: Sequence[ProgressCallback] | None = None,
        commit_state: bool,
    ) -> tuple[PretrainTrainingResult, TrainerState]:
        def validate_epoch_request() -> tuple[
            int,
            int | None,
            tuple[ProgressCallback, ...],
        ]:
            self._require_usable("train an epoch")
            epoch_index = _strict_integer("epoch", epoch, minimum=0)
            if epoch_index != self._state.next_epoch:
                raise ValueError(
                    f"train_epoch expected epoch {self._state.next_epoch}, "
                    f"got {epoch_index}"
                )
            if self._state.micro_step != 0:
                raise RuntimeError(
                    "train_epoch requires an accumulation-free boundary"
                )
            if self._has_any_gradient():
                raise RuntimeError(
                    "train_epoch requires all parameter gradients to be None"
                )
            normalized_interval = (
                None
                if progress_interval is None
                else _strict_integer(
                    "progress_interval",
                    progress_interval,
                    minimum=1,
                )
            )
            normalized_callbacks = self._normalize_progress_callbacks(
                progress_callbacks
            )
            if normalized_interval is None and normalized_callbacks:
                raise ValueError(
                    "progress callbacks require a progress_interval"
                )
            return epoch_index, normalized_interval, normalized_callbacks

        (
            epoch_index,
            normalized_progress_interval,
            normalized_progress_callbacks,
        ) = _distributed_call(
            self.context,
            "train epoch request validation",
            validate_epoch_request,
        )
        _require_matching_integer(self.context, "train epoch", epoch_index)
        progress_enabled = _require_matching_bool(
            self.context,
            "progress interval presence",
            normalized_progress_interval is not None,
        )
        if progress_enabled:
            if normalized_progress_interval is None:
                raise RuntimeError("progress interval is missing on this rank")
            _require_matching_integer(
                self.context,
                "progress interval",
                normalized_progress_interval,
            )

        entry_state = self._state
        completed = False
        try:
            self._set_epoch("train_loader", self.train_loader, epoch_index)
            total_batches = self._loader_length(
                "train_loader",
                self.train_loader,
            )
            iterator = _distributed_call(
                self.context,
                "train_loader iterator creation",
                lambda: iter(self.train_loader),
            )
            self.model.train()
            self.loss_fn.train()
            _distributed_call(
                self.context,
                "initial gradient clearing",
                lambda: self.optimizer.zero_grad(set_to_none=True),
            )
            accumulator = _EpochLossAccumulator(
                self.loss_fn,
                self.context,
                epoch_index,
            )
            successful_steps = 0
            skipped_steps = 0

            for batch_index in range(total_batches):
                raw_batch = _coordinated_next(
                    self.context,
                    iterator,
                    operation=f"train_loader batch {batch_index}",
                    expect_item=True,
                )
                batch = _distributed_call(
                    self.context,
                    f"train batch {batch_index} preparation",
                    lambda raw_batch=raw_batch: self._prepare_batch(raw_batch),
                )
                with self.precision.autocast(self.context.device):
                    outputs, loss_output = self._compute_loss(
                        batch,
                        epoch=epoch_index,
                        update_dcl=True,
                    )
                    batch_loss = loss_output.total_loss
                if self.scaler is None:
                    batch_loss.backward()
                else:
                    self.scaler.scale(batch_loss).backward()

                self._state = replace(
                    self._state,
                    micro_step=1,
                )
                _distributed_call(
                    self.context,
                    "training loss accumulation",
                    lambda: accumulator.update(
                        loss_output,
                        batch_size=outputs.batch_size,
                    ),
                )
                stepped = self._optimizer_step()
                if stepped:
                    successful_steps += 1
                else:
                    skipped_steps += 1
                self._state = replace(
                    self._state,
                    micro_step=0,
                    optimizer_step=(
                        self._state.optimizer_step + int(stepped)
                    ),
                )

                completed_batches = batch_index + 1
                if (
                    normalized_progress_interval is not None
                    and (
                        completed_batches % normalized_progress_interval == 0
                        or completed_batches == total_batches
                    )
                ):
                    progress = PretrainProgressResult(
                        epoch=epoch_index,
                        completed_batches=completed_batches,
                        total_batches=total_batches,
                        optimizer_step=self._state.optimizer_step,
                    )
                    self._invoke_progress_callbacks(
                        normalized_progress_callbacks,
                        progress,
                    )

            _coordinated_next(
                self.context,
                iterator,
                operation="train_loader exhaustion check",
                expect_item=False,
            )
            if self._state.micro_step != 0:
                raise RuntimeError(
                    "train epoch ended with a pending optimizer step"
                )
            losses, microbatches, processed_samples = accumulator.finalize()
            learning_rates = _distributed_call(
                self.context,
                "optimizer learning-rate validation",
                self._learning_rates,
            )
            if self.context.distributed:
                gathered_rates: list[tuple[float, ...] | None] = [
                    None for _ in range(self.context.world_size)
                ]
                dist.all_gather_object(gathered_rates, learning_rates)
                if any(rates != learning_rates for rates in gathered_rates):
                    raise RuntimeError(
                        "optimizer learning rates differ across ranks: "
                        f"{gathered_rates}"
                    )
            result = _distributed_call(
                self.context,
                "training result validation",
                lambda: PretrainTrainingResult(
                    epoch=epoch_index,
                    losses=losses,
                    microbatches=microbatches,
                    processed_samples=processed_samples,
                    optimizer_steps=successful_steps,
                    skipped_optimizer_steps=skipped_steps,
                    learning_rates=learning_rates,
                ),
            )
            candidate_state = _distributed_call(
                self.context,
                "completed epoch state validation",
                lambda: replace(
                    self._state,
                    next_epoch=epoch_index + 1,
                ),
            )
            self._state = candidate_state if commit_state else entry_state
            completed = True
            return result, candidate_state
        finally:
            if not completed:
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                finally:
                    self._state = replace(entry_state, micro_step=0)

    def train_epoch(
        self,
        epoch: int,
        *,
        progress_interval: int | None = None,
        progress_callbacks: Sequence[ProgressCallback] | None = None,
    ) -> PretrainTrainingResult:
        """Train and immediately commit one epoch without validation or save.

        ``fit`` uses the private deferred-commit path so its public state only
        advances after validation and the mandatory latest checkpoint succeed.
        A standalone call commits ``next_epoch`` as soon as this training epoch
        finishes; callers own any subsequent validation/checkpoint workflow.
        """

        try:
            result, _ = self._run_train_epoch(
                epoch,
                progress_interval=progress_interval,
                progress_callbacks=progress_callbacks,
                commit_state=True,
            )
            return result
        except BaseException:
            self._mark_unusable()
            raise

    def _run_validation_epoch(self, epoch: int) -> PretrainValidationResult:
        def validate_epoch_request() -> int:
            self._require_usable("validate an epoch")
            epoch_index = _strict_integer("epoch", epoch, minimum=0)
            if self.valid_loader is None:
                raise RuntimeError("validate_epoch requires valid_loader")
            return epoch_index

        epoch_index = _distributed_call(
            self.context,
            "validation epoch request validation",
            validate_epoch_request,
        )
        _require_matching_integer(self.context, "validation epoch", epoch_index)
        if self.valid_loader is None:
            raise RuntimeError("validate_epoch requires valid_loader")
        self._set_epoch("valid_loader", self.valid_loader, epoch_index)
        total_batches = self._loader_length("valid_loader", self.valid_loader)
        iterator = _distributed_call(
            self.context,
            "valid_loader iterator creation",
            lambda: iter(self.valid_loader),
        )
        self.model.eval()
        self.loss_fn.eval()
        accumulator = _EpochLossAccumulator(
            self.loss_fn,
            self.context,
            epoch_index,
        )
        with torch.no_grad():
            for batch_index in range(total_batches):
                raw_batch = _coordinated_next(
                    self.context,
                    iterator,
                    operation=f"valid_loader batch {batch_index}",
                    expect_item=True,
                )
                batch = _distributed_call(
                    self.context,
                    f"validation batch {batch_index} preparation",
                    lambda raw_batch=raw_batch: self._prepare_batch(raw_batch),
                )
                with self.precision.autocast(self.context.device):
                    outputs, loss_output = self._compute_loss(
                        batch,
                        epoch=epoch_index,
                        update_dcl=False,
                    )
                _distributed_call(
                    self.context,
                    "validation loss accumulation",
                    lambda: accumulator.update(
                        loss_output,
                        batch_size=outputs.batch_size,
                    ),
                )
        _coordinated_next(
            self.context,
            iterator,
            operation="valid_loader exhaustion check",
            expect_item=False,
        )
        losses, microbatches, processed_samples = accumulator.finalize()
        return _distributed_call(
            self.context,
            "validation result validation",
            lambda: PretrainValidationResult(
                epoch=epoch_index,
                losses=losses,
                microbatches=microbatches,
                processed_samples=processed_samples,
            ),
        )

    def validate_epoch(self, epoch: int) -> PretrainValidationResult:
        """Validate one epoch, poisoning the trainer on any failure."""

        try:
            return self._run_validation_epoch(epoch)
        except BaseException:
            self._mark_unusable()
            raise

    def _save_checkpoint(
        self,
        path: Path,
        *,
        state: TrainerState | None = None,
    ) -> Path:
        self._require_usable("save a checkpoint")
        checkpoint_state = self._state if state is None else state
        try:
            save_training_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                state=checkpoint_state,
                config_fingerprint=self.config_fingerprint,
                loaders=self._loaders(),
                extra=self._checkpoint_extra(),
                context=self.context,
            )
        except BaseException:
            self._mark_unusable()
            raise
        return path.expanduser().resolve()

    def _invoke_callbacks(
        self,
        callbacks: tuple[EpochEndCallback, ...],
        result: PretrainEpochResult,
    ) -> None:
        def invoke() -> None:
            if not self.context.is_main_process:
                return
            for callback in callbacks:
                callback(result)

        _distributed_call(
            self.context,
            "rank-zero epoch callback",
            invoke,
        )

    def _invoke_progress_callbacks(
        self,
        callbacks: tuple[ProgressCallback, ...],
        result: PretrainProgressResult,
    ) -> None:
        def invoke() -> None:
            if not self.context.is_main_process:
                return
            for callback in callbacks:
                callback(result)

        _distributed_call(
            self.context,
            "rank-zero training progress callback",
            invoke,
        )

    @staticmethod
    def _normalize_callbacks(
        callbacks: Sequence[EpochEndCallback] | None,
    ) -> tuple[EpochEndCallback, ...]:
        if callbacks is None:
            return ()
        if isinstance(callbacks, (str, bytes)) or not isinstance(
            callbacks,
            Sequence,
        ):
            raise TypeError("callbacks must be a sequence of callables or None")
        normalized = tuple(callbacks)
        if any(not callable(callback) for callback in normalized):
            raise TypeError("every callback must be callable")
        return normalized

    @staticmethod
    def _normalize_progress_callbacks(
        callbacks: Sequence[ProgressCallback] | None,
    ) -> tuple[ProgressCallback, ...]:
        if callbacks is None:
            return ()
        if isinstance(callbacks, (str, bytes)) or not isinstance(
            callbacks,
            Sequence,
        ):
            raise TypeError(
                "progress_callbacks must be a sequence of callables or None"
            )
        normalized = tuple(callbacks)
        if any(not callable(callback) for callback in normalized):
            raise TypeError("every progress callback must be callable")
        return normalized

    def _run_fit(
        self,
        *,
        resume_from: str | Path | None = None,
        callbacks: Sequence[EpochEndCallback] | None = None,
        progress_interval: int | None = None,
        progress_callbacks: Sequence[ProgressCallback] | None = None,
    ) -> PretrainFitResult:
        normalized_callbacks = _distributed_call(
            self.context,
            "callback validation",
            lambda: self._normalize_callbacks(callbacks),
        )
        self._prepare_checkpoint_directory()
        resume_requested = _require_matching_bool(
            self.context,
            "resume request",
            resume_from is not None,
        )
        if resume_requested:
            if resume_from is None:
                raise RuntimeError("resume path is missing on this rank")
            self.resume(resume_from)

        epoch_results: list[PretrainEpochResult] = []
        periodic_checkpoints: list[Path] = []
        latest_checkpoint: Path | None = None
        best_checkpoint: Path | None = None
        start_epoch = self._state.next_epoch
        if start_epoch > self.config.epochs:
            raise ValueError("trainer state exceeds configured epochs")

        for epoch in range(start_epoch, self.config.epochs):
            training, candidate_state = self._run_train_epoch(
                epoch,
                progress_interval=progress_interval,
                progress_callbacks=progress_callbacks,
                commit_state=False,
            )
            validation = (
                None
                if self.valid_loader is None
                else self._run_validation_epoch(epoch)
            )
            improved = False
            if validation is not None:
                local_improved = (
                    candidate_state.best_epoch == -1
                    or validation.losses.total_loss
                    < candidate_state.best_metric
                )
                improved = broadcast_bool(
                    local_improved if self.context.is_main_process else False,
                    src=0,
                )
                if improved:
                    candidate_state = _distributed_call(
                        self.context,
                        "improved epoch state validation",
                        lambda: replace(
                            candidate_state,
                            best_metric=validation.losses.total_loss,
                            best_epoch=epoch,
                            bad_epochs=0,
                        ),
                    )
                else:
                    candidate_state = _distributed_call(
                        self.context,
                        "non-improved epoch state validation",
                        lambda: replace(
                            candidate_state,
                            bad_epochs=candidate_state.bad_epochs + 1,
                        ),
                    )

            latest_checkpoint = self._save_checkpoint(
                self.config.checkpoint.latest_path,
                state=candidate_state,
            )
            # This is the durable commit point: validation decisions and the
            # completed epoch already exist in the mandatory latest checkpoint.
            self._state = candidate_state
            completed_epoch = epoch + 1
            if (
                completed_epoch
                % self.config.checkpoint.save_every_n_epochs
                == 0
            ):
                periodic_checkpoints.append(
                    self._save_checkpoint(
                        self.config.checkpoint.periodic_path(completed_epoch),
                        state=candidate_state,
                    )
                )
            if improved:
                best_checkpoint = self._save_checkpoint(
                    self.config.checkpoint.best_path,
                    state=candidate_state,
                )

            epoch_result = _distributed_call(
                self.context,
                "epoch result validation",
                lambda: PretrainEpochResult(
                    epoch=epoch,
                    training=training,
                    validation=validation,
                    improved=improved,
                ),
            )
            epoch_results.append(epoch_result)
            self._invoke_callbacks(normalized_callbacks, epoch_result)

        return _distributed_call(
            self.context,
            "fit result validation",
            lambda: PretrainFitResult(
                state=self._state,
                epochs=tuple(epoch_results),
                resumed_from=self._resumed_from,
                latest_checkpoint=latest_checkpoint,
                best_checkpoint=best_checkpoint,
                periodic_checkpoints=tuple(periodic_checkpoints),
            ),
        )

    def fit(
        self,
        *,
        resume_from: str | Path | None = None,
        callbacks: Sequence[EpochEndCallback] | None = None,
        progress_interval: int | None = None,
        progress_callbacks: Sequence[ProgressCallback] | None = None,
    ) -> PretrainFitResult:
        """Execute the trainer's single fail-closed managed fit lifecycle."""

        def validate_fit_start() -> None:
            self._require_usable("start fit")
            if self._fit_invoked:
                raise RuntimeError("fit may be invoked only once per trainer")

        _distributed_call(
            self.context,
            "fit lifecycle validation",
            validate_fit_start,
        )
        self._fit_invoked = True
        try:
            return self._run_fit(
                resume_from=resume_from,
                callbacks=callbacks,
                progress_interval=progress_interval,
                progress_callbacks=progress_callbacks,
            )
        except BaseException:
            self._mark_unusable()
            raise


__all__ = [
    "EpochEndCallback",
    "ProgressCallback",
    "PretrainCheckpointConfig",
    "PretrainEpochResult",
    "PretrainFitResult",
    "PretrainLossSummary",
    "PretrainProgressResult",
    "PretrainTrainer",
    "PretrainTrainerConfig",
    "PretrainTrainingResult",
    "PretrainValidationResult",
]

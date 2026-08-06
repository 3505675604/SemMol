"""Strict single-run downstream finetuning for SemMol."""

from __future__ import annotations

import math
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, TypeAlias

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch

from src.datasets.collator import FinetuningDataCollator
from src.datasets.loader import set_dataloader_epoch
from src.evaluation.metrics import (
    ClassificationMetrics,
    IndexedPredictions,
    RegressionMetrics,
    evaluate_classification,
    evaluate_regression,
    gather_indexed_predictions,
)
from src.losses.downstream_loss import DownstreamTaskLoss
from src.losses.common import LossComponent
from src.molecular.espf_tokenizer import CLS_TOKEN_ID, PAD_TOKEN_ID
from src.models.semmol import SemMol, SemMolFinetuningOutput

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
    broadcast_float,
    move_batch_to_device,
    no_sync_context,
    unwrap_model,
)


MetricResult: TypeAlias = ClassificationMetrics | RegressionMetrics
EpochCallback: TypeAlias = Callable[["FinetuningEpochResult"], None]

_TASK_TYPES = frozenset({"classification", "regression"})
_DIRECTION_ALIASES = {
    "max": "maximize",
    "maximize": "maximize",
    "min": "minimize",
    "minimize": "minimize",
}
_METRIC_ALIASES = {
    "classification": {
        "roc_auc": "macro_roc_auc",
        "macro_roc_auc": "macro_roc_auc",
    },
    "regression": {
        "rmse": "macro_rmse",
        "macro_rmse": "macro_rmse",
        "mae": "macro_mae",
        "macro_mae": "macro_mae",
        "r2": "macro_r2",
        "macro_r2": "macro_r2",
    },
}
_EXPECTED_DIRECTIONS = {
    "macro_roc_auc": "maximize",
    "macro_rmse": "minimize",
    "macro_mae": "minimize",
    "macro_r2": "maximize",
}
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRAINER_NAME = "semmol_finetuning"
_TRAINER_SCHEMA_VERSION = 1
_FULL_BATCH_TENSOR_KEYS = (
    "source_index",
    "record_index",
    "modality_mask",
    "input_ids",
    "attention_mask",
    "atomic_numbers",
    "coords",
    "atom_mask",
    "conformer_mask",
    "qm_grid",
    "qm_mask",
    "labels",
    "label_mask",
)
_ANCHOR_COLUMNS = {"1d": 0, "2d": 1, "3d": 2}


def _type_identity(value: object | None) -> str | None:
    if value is None:
        return None
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _callable_identity(value: object | None) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    return _type_identity(value)


def _active_collective_device() -> torch.device:
    if (
        dist.is_available()
        and dist.is_initialized()
        and str(dist.get_backend()).lower() == "nccl"
    ):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _synchronize_constructor_preflight(
    operation: str,
    function: Callable[[], Any],
) -> Any:
    """Coordinate constructor-local failures using only the active group."""

    local_error: Exception | None = None
    result: Any = None
    try:
        result = function()
    except Exception as exc:
        local_error = exc

    active = dist.is_available() and dist.is_initialized()
    if not active:
        if local_error is not None:
            raise local_error
        return result

    world_size = dist.get_world_size()
    error_flag = torch.tensor(
        int(local_error is not None),
        dtype=torch.int32,
        device=_active_collective_device(),
    )
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    if int(error_flag.item()) == 0:
        return result

    descriptions: list[tuple[str, str] | None] = [
        None for _ in range(world_size)
    ]
    local_description = (
        None
        if local_error is None
        else (type(local_error).__name__, str(local_error))
    )
    dist.all_gather_object(descriptions, local_description)
    if local_error is not None:
        raise local_error
    failures = [
        f"rank {rank}: {description[0]}: {description[1]}"
        for rank, description in enumerate(descriptions)
        if description is not None
    ]
    raise RuntimeError(f"{operation} failed on another rank; " + "; ".join(failures))


def _control_value(value: Any, *, location: str) -> Any:
    """Convert control data into deterministic, safely comparable objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Tensor):
        if value.is_complex():
            raise TypeError(f"{location} cannot contain complex tensors")
        detached = value.detach().cpu()
        if detached.is_floating_point() and not bool(torch.isfinite(detached).all()):
            raise ValueError(f"{location} tensor must be finite")
        return {
            "dtype": str(detached.dtype),
            "shape": tuple(detached.shape),
            "values": detached.tolist(),
        }
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"{location} mapping keys must be strings")
            normalized[key] = _control_value(
                value[key],
                location=f"{location}.{key}",
            )
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(
            _control_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{location} has unsupported control value type "
        f"{type(value).__name__}"
    )


def _integer(name: str, value: object, *, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _finite_float(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _checkpoint_path(name: str, value: object) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a path string or pathlib.Path")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} cannot be empty")
    path = Path(value).expanduser()
    if not path.name:
        raise ValueError(f"{name} must identify a checkpoint file")
    return path


@dataclass(frozen=True)
class DownstreamTaskDefinition:
    """The output and model-selection contract for one downstream dataset."""

    task_type: str
    num_tasks: int
    task_names: Sequence[str]
    main_metric: str
    metric_direction: str

    def __post_init__(self) -> None:
        task_type = _nonempty_string("task_type", self.task_type).lower()
        if task_type not in _TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {sorted(_TASK_TYPES)}, got "
                f"{self.task_type!r}"
            )
        num_tasks = _integer("num_tasks", self.num_tasks, minimum=1)
        if isinstance(self.task_names, (str, bytes)) or not isinstance(
            self.task_names,
            Sequence,
        ):
            raise TypeError("task_names must be a sequence of strings")
        task_names = tuple(self.task_names)
        if len(task_names) != num_tasks:
            raise ValueError(
                f"task_names must contain {num_tasks} entries, got "
                f"{len(task_names)}"
            )
        if any(not isinstance(name, str) or not name.strip() for name in task_names):
            raise ValueError("task_names must contain only non-empty strings")
        task_names = tuple(name.strip() for name in task_names)
        if len(set(task_names)) != len(task_names):
            raise ValueError("task_names must be unique")

        requested_metric = _nonempty_string(
            "main_metric",
            self.main_metric,
        ).lower()
        metric_aliases = _METRIC_ALIASES[task_type]
        if requested_metric not in metric_aliases:
            raise ValueError(
                f"main_metric for {task_type} must be one of "
                f"{sorted(metric_aliases)}, got {self.main_metric!r}"
            )
        main_metric = metric_aliases[requested_metric]
        direction = _nonempty_string(
            "metric_direction",
            self.metric_direction,
        ).lower()
        if direction not in _DIRECTION_ALIASES:
            raise ValueError(
                "metric_direction must be one of 'max', 'maximize', 'min', "
                "or 'minimize'"
            )
        direction = _DIRECTION_ALIASES[direction]
        expected_direction = _EXPECTED_DIRECTIONS[main_metric]
        if direction != expected_direction:
            raise ValueError(
                f"{main_metric} must use metric_direction="
                f"{expected_direction!r}, got {direction!r}"
            )

        object.__setattr__(self, "task_type", task_type)
        object.__setattr__(self, "num_tasks", num_tasks)
        object.__setattr__(self, "task_names", task_names)
        object.__setattr__(self, "main_metric", main_metric)
        object.__setattr__(self, "metric_direction", direction)

    def as_dict(self) -> dict[str, object]:
        return {
            "task_type": self.task_type,
            "num_tasks": self.num_tasks,
            "task_names": tuple(self.task_names),
            "main_metric": self.main_metric,
            "metric_direction": self.metric_direction,
        }


@dataclass(frozen=True)
class FinetuningTrainerConfig:
    """Validated optimization and stopping policy for one seed."""

    max_epochs: int
    best_checkpoint_path: str | Path
    latest_checkpoint_path: str | Path | None = None
    gradient_accumulation_steps: int = 1
    precision: str = "none"
    gradient_clip_norm: float | None = 1.0
    early_stopping_patience: int | None = None
    min_improvement: float = 0.0
    non_blocking_transfer: bool = True

    def __post_init__(self) -> None:
        max_epochs = _integer("max_epochs", self.max_epochs, minimum=1)
        accumulation = _integer(
            "gradient_accumulation_steps",
            self.gradient_accumulation_steps,
            minimum=1,
        )
        precision = PrecisionMode(
            _nonempty_string("precision", self.precision).lower()
        ).mode
        clip_norm = self.gradient_clip_norm
        if clip_norm is not None:
            clip_norm = _finite_float(
                "gradient_clip_norm",
                clip_norm,
                strictly_positive=True,
            )
        patience = self.early_stopping_patience
        if patience is not None:
            patience = _integer(
                "early_stopping_patience",
                patience,
                minimum=1,
            )
        min_improvement = _finite_float(
            "min_improvement",
            self.min_improvement,
            minimum=0.0,
        )
        non_blocking = _strict_bool(
            "non_blocking_transfer",
            self.non_blocking_transfer,
        )
        best_path = _checkpoint_path(
            "best_checkpoint_path",
            self.best_checkpoint_path,
        )
        latest_path = (
            None
            if self.latest_checkpoint_path is None
            else _checkpoint_path(
                "latest_checkpoint_path",
                self.latest_checkpoint_path,
            )
        )
        if latest_path is not None and latest_path == best_path:
            raise ValueError(
                "latest_checkpoint_path must differ from best_checkpoint_path"
            )

        object.__setattr__(self, "max_epochs", max_epochs)
        object.__setattr__(self, "best_checkpoint_path", best_path)
        object.__setattr__(self, "latest_checkpoint_path", latest_path)
        object.__setattr__(self, "gradient_accumulation_steps", accumulation)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "gradient_clip_norm", clip_norm)
        object.__setattr__(self, "early_stopping_patience", patience)
        object.__setattr__(self, "min_improvement", min_improvement)
        object.__setattr__(self, "non_blocking_transfer", non_blocking)


@dataclass(frozen=True)
class EvaluationResult:
    """One globally aggregated validation or test evaluation."""

    split: str
    loss: float
    main_metric: float
    sample_count: int
    valid_label_count: int
    metrics: MetricResult | None

    def __post_init__(self) -> None:
        split = _nonempty_string("split", self.split).lower()
        if split not in {"validation", "test"}:
            raise ValueError("split must be 'validation' or 'test'")
        loss = _finite_float("loss", self.loss)
        metric = _finite_float("main_metric", self.main_metric)
        sample_count = _integer("sample_count", self.sample_count, minimum=1)
        valid_count = _integer(
            "valid_label_count",
            self.valid_label_count,
            minimum=1,
        )
        if self.metrics is not None and not isinstance(
            self.metrics,
            (ClassificationMetrics, RegressionMetrics),
        ):
            raise TypeError(
                "metrics must be ClassificationMetrics, RegressionMetrics, or None"
            )
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "main_metric", metric)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "valid_label_count", valid_count)


@dataclass(frozen=True)
class FinetuningEpochResult:
    """Training and validation outcome for one completed epoch."""

    epoch: int
    train_loss: float
    validation: EvaluationResult
    optimizer_steps: int
    learning_rates: tuple[float, ...]
    improved: bool
    bad_epochs: int

    def __post_init__(self) -> None:
        _integer("epoch", self.epoch, minimum=0)
        _finite_float("train_loss", self.train_loss)
        if not isinstance(self.validation, EvaluationResult):
            raise TypeError("validation must be EvaluationResult")
        if self.validation.split != "validation":
            raise ValueError("validation result must use split='validation'")
        _integer("optimizer_steps", self.optimizer_steps, minimum=0)
        if not isinstance(self.learning_rates, tuple) or not self.learning_rates:
            raise ValueError("learning_rates must be a non-empty tuple")
        for index, value in enumerate(self.learning_rates):
            _finite_float(f"learning_rates[{index}]", value, minimum=0.0)
        _strict_bool("improved", self.improved)
        _integer("bad_epochs", self.bad_epochs, minimum=0)


@dataclass(frozen=True)
class FinetuningRunResult:
    """Final state for one seed, including one best-model test evaluation."""

    terminal_state: TrainerState
    best_state: TrainerState
    epochs: tuple[FinetuningEpochResult, ...]
    test: EvaluationResult
    best_checkpoint_path: Path
    resumed_from: Path | None

    def __post_init__(self) -> None:
        if not isinstance(self.terminal_state, TrainerState):
            raise TypeError("terminal_state must be TrainerState")
        if not isinstance(self.best_state, TrainerState):
            raise TypeError("best_state must be TrainerState")
        if self.best_state.best_epoch < 0:
            raise ValueError("best_state must identify a validation-selected epoch")
        if not isinstance(self.epochs, tuple) or any(
            not isinstance(item, FinetuningEpochResult) for item in self.epochs
        ):
            raise TypeError("epochs must be a tuple of FinetuningEpochResult")
        if not isinstance(self.test, EvaluationResult) or self.test.split != "test":
            raise ValueError("test must be an EvaluationResult for split='test'")
        if not isinstance(self.best_checkpoint_path, Path):
            raise TypeError("best_checkpoint_path must be pathlib.Path")
        if self.resumed_from is not None and not isinstance(self.resumed_from, Path):
            raise TypeError("resumed_from must be pathlib.Path or None")


class FinetuningTrainer:
    """Train one SemMol downstream run and test its best validation epoch once."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: DownstreamTaskLoss,
        optimizer: Optimizer,
        scheduler: object | None,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        task: DownstreamTaskDefinition,
        config: FinetuningTrainerConfig,
        context: DistributedContext,
        config_fingerprint: str,
        epoch_callback: EpochCallback | None = None,
    ) -> None:
        def preflight() -> tuple[
            str,
            SemMol,
            DownstreamTaskLoss,
            DownstreamTaskLoss,
            PrecisionMode,
            torch.cuda.amp.GradScaler | None,
        ]:
            if not isinstance(model, nn.Module):
                raise TypeError("model must be a torch nn.Module")
            if not isinstance(loss_fn, DownstreamTaskLoss):
                raise TypeError("loss_fn must be DownstreamTaskLoss")
            if not isinstance(optimizer, Optimizer):
                raise TypeError("optimizer must be a torch Optimizer")
            if scheduler is not None:
                self._validate_scheduler(scheduler, optimizer)
            for name, loader in (
                ("train_loader", train_loader),
                ("valid_loader", valid_loader),
                ("test_loader", test_loader),
            ):
                if not isinstance(loader, DataLoader):
                    raise TypeError(
                        f"{name} must be a DataLoader; validation and test "
                        "loaders are mandatory so best selection cannot use "
                        "test data"
                    )
            if len({id(train_loader), id(valid_loader), id(test_loader)}) != 3:
                raise ValueError(
                    "train_loader, valid_loader, and test_loader must be "
                    "distinct DataLoader instances"
                )
            if valid_loader.drop_last or test_loader.drop_last:
                raise ValueError(
                    "validation and test DataLoaders must use drop_last=False"
                )
            if not isinstance(task, DownstreamTaskDefinition):
                raise TypeError("task must be DownstreamTaskDefinition")
            if not isinstance(config, FinetuningTrainerConfig):
                raise TypeError("config must be FinetuningTrainerConfig")
            if not isinstance(context, DistributedContext):
                raise TypeError("context must be DistributedContext")
            fingerprint = _nonempty_string(
                "config_fingerprint",
                config_fingerprint,
            )
            if _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
                raise ValueError(
                    "config_fingerprint must be a lowercase SHA-256 "
                    "hexadecimal string"
                )
            if epoch_callback is not None and not callable(epoch_callback):
                raise TypeError("epoch_callback must be callable or None")

            active = dist.is_available() and dist.is_initialized()
            active_world = dist.get_world_size() if active else 1
            active_rank = dist.get_rank() if active else 0
            active_backend = str(dist.get_backend()) if active else None
            if active_world != context.world_size or active_rank != context.rank:
                raise RuntimeError(
                    "DistributedContext does not match the active process group"
                )
            if context.distributed != (active_world > 1):
                raise RuntimeError(
                    "DistributedContext.distributed does not match the active job"
                )
            if active and context.backend != active_backend:
                raise RuntimeError(
                    "DistributedContext.backend does not match the active "
                    "process group"
                )
            if context.distributed and not isinstance(
                model,
                DistributedDataParallel,
            ):
                raise TypeError(
                    "distributed finetuning requires a "
                    "DistributedDataParallel model"
                )
            if not context.distributed and isinstance(
                model,
                DistributedDataParallel,
            ):
                raise ValueError(
                    "a single-process finetuning context must not receive a "
                    "DDP model"
                )
            if isinstance(model, DistributedDataParallel):
                if dist.get_world_size(model.process_group) != context.world_size:
                    raise ValueError(
                        "DDP model process-group world size differs from context"
                    )
                if dist.get_rank(model.process_group) != context.rank:
                    raise ValueError(
                        "DDP model process-group rank differs from context"
                    )
                if model.broadcast_buffers:
                    raise ValueError(
                        "SemMol DDP must use broadcast_buffers=False for DCL state"
                    )
                if context.device.type == "cuda" and model.device_ids != [
                    context.device.index
                ]:
                    raise ValueError(
                        "CUDA DDP model must use exactly context.device as "
                        "device_ids"
                    )
                if context.device.type == "cpu" and model.device_ids:
                    raise ValueError(
                        "CPU DDP model must not define CUDA device_ids"
                    )
            if (
                context.distributed
                and context.device.type == "cuda"
                and active_backend != "nccl"
            ):
                raise ValueError("distributed CUDA finetuning requires NCCL")

            base_model = unwrap_model(model)
            if not isinstance(base_model, SemMol):
                raise TypeError("model must be SemMol or DDP-wrapped SemMol")
            property_head = base_model.property_head
            if property_head is None:
                raise ValueError(
                    "SemMol must be built with a downstream property head"
                )
            if property_head.task_type != task.task_type:
                raise ValueError(
                    "model property-head task_type does not match task "
                    f"definition: {property_head.task_type!r} != "
                    f"{task.task_type!r}"
                )
            if property_head.num_tasks != task.num_tasks:
                raise ValueError(
                    "model property-head num_tasks does not match task "
                    f"definition: {property_head.num_tasks} != {task.num_tasks}"
                )
            if loss_fn.task_type != task.task_type:
                raise ValueError(
                    "DownstreamTaskLoss.task_type does not match task definition"
                )
            if context.distributed and not loss_fn.distributed_sync:
                raise ValueError(
                    "DownstreamTaskLoss.distributed_sync must be true for DDP "
                    "training"
                )

            self._validate_optimizer(model, optimizer)
            self._validate_loader("train_loader", train_loader, context)
            self._validate_loader("valid_loader", valid_loader, context)
            self._validate_loader("test_loader", test_loader, context)

            model_devices = {
                value.device
                for value in (
                    tuple(base_model.parameters()) + tuple(base_model.buffers())
                )
            }
            if model_devices and model_devices != {context.device}:
                raise ValueError(
                    "all model parameters and buffers must already be on "
                    f"context.device={context.device}; found "
                    f"{sorted(map(str, model_devices))}"
                )

            prepared_loss = loss_fn.to(context.device)
            if not isinstance(prepared_loss, DownstreamTaskLoss):
                raise TypeError("loss_fn.to() must return DownstreamTaskLoss")
            local_loss = DownstreamTaskLoss(
                task_type=prepared_loss.task_type,
                loss_type=prepared_loss.loss_type,
                huber_delta=prepared_loss.huber_delta,
                distributed_sync=False,
                validate_values=prepared_loss.validate_values,
            ).to(context.device)
            precision = PrecisionMode(config.precision)
            scaler = precision.create_grad_scaler(context.device)
            return (
                fingerprint,
                base_model,
                prepared_loss,
                local_loss,
                precision,
                scaler,
            )

        (
            fingerprint,
            base_model,
            prepared_loss,
            local_loss,
            precision,
            scaler,
        ) = _synchronize_constructor_preflight(
            "finetuning trainer constructor preflight",
            preflight,
        )

        self.model = model
        self.base_model = base_model
        self.loss_fn = prepared_loss
        self.local_loss_fn = local_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.task = task
        self.config = config
        self.context = context
        self.config_fingerprint = fingerprint
        self.epoch_callback = epoch_callback
        self.precision = precision
        self.scaler = scaler
        self._fit_started = False
        self._micro_step = 0
        self._validate_control_signature()

    @staticmethod
    def _validate_scheduler(scheduler: object, optimizer: Optimizer) -> None:
        for method_name in ("step", "state_dict", "load_state_dict"):
            if not callable(getattr(scheduler, method_name, None)):
                raise TypeError(
                    f"scheduler must implement callable {method_name}()"
                )
        if getattr(scheduler, "optimizer", None) is not optimizer:
            raise ValueError(
                "scheduler.optimizer must be the finetuning optimizer"
            )
        try:
            inspect.signature(scheduler.step).bind()
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "scheduler.step must accept a zero-argument call"
            ) from exc
        state = scheduler.state_dict()
        if not isinstance(state, Mapping):
            raise TypeError("scheduler.state_dict() must return a mapping")

    @staticmethod
    def _validate_optimizer(model: nn.Module, optimizer: Optimizer) -> None:
        model_parameters = {
            id(parameter): (name, parameter)
            for name, parameter in model.named_parameters()
        }
        trainable = {
            identity
            for identity, (_, parameter) in model_parameters.items()
            if parameter.requires_grad
        }
        if not trainable:
            raise ValueError("model must expose at least one trainable parameter")
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
                    raise ValueError(
                        "optimizer contains a parameter outside the finetuning model"
                    )
                if identity in optimized:
                    raise ValueError("optimizer contains a duplicate parameter")
                if not parameter.requires_grad:
                    name = model_parameters[identity][0]
                    raise ValueError(
                        f"optimizer contains frozen model parameter {name!r}"
                    )
                optimized.add(identity)
        missing = [
            model_parameters[identity][0]
            for identity in trainable - optimized
        ]
        if missing:
            missing.sort()
            preview = missing[:8]
            suffix = "..." if len(missing) > len(preview) else ""
            raise ValueError(
                "optimizer omits trainable model parameters: "
                f"{preview}{suffix}"
            )

    @staticmethod
    def _validate_loader(
        name: str,
        loader: DataLoader,
        context: DistributedContext,
    ) -> None:
        if not isinstance(loader.generator, torch.Generator):
            raise ValueError(
                f"{name} must have an explicit torch.Generator for exact resume"
            )
        sampler = loader.sampler
        if context.distributed and not isinstance(
            sampler,
            DistributedSampler,
        ):
            raise TypeError(f"{name} must use DistributedSampler under DDP")
        if isinstance(sampler, DistributedSampler):
            if sampler.num_replicas != context.world_size:
                raise ValueError(
                    f"{name} sampler num_replicas does not match world_size"
                )
            if sampler.rank != context.rank:
                raise ValueError(f"{name} sampler rank does not match context.rank")
            if name in {"valid_loader", "test_loader"} and sampler.drop_last:
                raise ValueError(
                    f"{name} DistributedSampler must use drop_last=False"
                )

    def _optimizer_control_signature(self) -> dict[str, Any]:
        parameter_names = {
            id(parameter): name
            for name, parameter in self.model.named_parameters()
        }
        groups: list[dict[str, Any]] = []
        for group_index, group in enumerate(self.optimizer.param_groups):
            parameters = group.get("params")
            if not isinstance(parameters, list):
                raise TypeError(
                    f"optimizer.param_groups[{group_index}]['params'] must be "
                    "a list"
                )
            names: list[str] = []
            for parameter in parameters:
                name = parameter_names.get(id(parameter))
                if name is None:
                    raise ValueError(
                        "optimizer control signature found a parameter outside "
                        "the model"
                    )
                names.append(name)
            hyperparameters = {
                key: value for key, value in group.items() if key != "params"
            }
            groups.append(
                {
                    "parameter_names": tuple(names),
                    "hyperparameters": _control_value(
                        hyperparameters,
                        location=f"optimizer.param_groups[{group_index}]",
                    ),
                }
            )
        return {
            "type": _type_identity(self.optimizer),
            "groups": tuple(groups),
        }

    @staticmethod
    def _loader_control_signature(
        name: str,
        loader: DataLoader,
    ) -> dict[str, Any]:
        sampler = loader.sampler
        generator = loader.generator
        if not isinstance(generator, torch.Generator):
            raise TypeError(f"{name}.generator must be torch.Generator")
        dataset_length = _integer(
            f"len({name}.dataset)",
            len(loader.dataset),
            minimum=1,
        )
        loader_length = _integer(
            f"len({name})",
            len(loader),
            minimum=1,
        )
        sampler_controls = {
            "type": _type_identity(sampler),
            "num_replicas": getattr(sampler, "num_replicas", None),
            "seed": getattr(sampler, "seed", None),
            "shuffle": getattr(sampler, "shuffle", None),
            "drop_last": getattr(sampler, "drop_last", None),
        }
        collator = loader.collate_fn
        collator_controls = {
            "type": _callable_identity(collator),
            "pad_token_id": (
                collator.pad_token_id
                if isinstance(collator, FinetuningDataCollator)
                else None
            ),
            "allow_partial_modalities": (
                collator.allow_partial_modalities
                if isinstance(collator, FinetuningDataCollator)
                else None
            ),
        }
        return {
            "type": _type_identity(loader),
            "batch_size": _control_value(
                loader.batch_size,
                location=f"{name}.batch_size",
            ),
            "drop_last": _strict_bool(f"{name}.drop_last", loader.drop_last),
            "length": loader_length,
            "dataset_type": _type_identity(loader.dataset),
            "dataset_length": dataset_length,
            "num_workers": _integer(
                f"{name}.num_workers",
                loader.num_workers,
                minimum=0,
            ),
            "pin_memory": _strict_bool(
                f"{name}.pin_memory",
                loader.pin_memory,
            ),
            "persistent_workers": _strict_bool(
                f"{name}.persistent_workers",
                loader.persistent_workers,
            ),
            "prefetch_factor": _control_value(
                loader.prefetch_factor,
                location=f"{name}.prefetch_factor",
            ),
            "generator_seed": _integer(
                f"{name}.generator.initial_seed()",
                generator.initial_seed(),
                minimum=0,
            ),
            "sampler": _control_value(
                sampler_controls,
                location=f"{name}.sampler",
            ),
            "batch_sampler_type": _type_identity(loader.batch_sampler),
            "collator": _control_value(
                collator_controls,
                location=f"{name}.collator",
            ),
        }

    def _control_signature(self) -> dict[str, Any]:
        loaders = {
            name: self._loader_control_signature(name, loader)
            for name, loader in (
                ("train_loader", self.train_loader),
                ("valid_loader", self.valid_loader),
                ("test_loader", self.test_loader),
            )
        }
        scheduler_signature = None
        if self.scheduler is not None:
            scheduler_state = self.scheduler.state_dict()
            if not isinstance(scheduler_state, Mapping):
                raise TypeError("scheduler.state_dict() must return a mapping")
            scheduler_signature = {
                "type": _type_identity(self.scheduler),
                "optimizer_bound": (
                    getattr(self.scheduler, "optimizer", None)
                    is self.optimizer
                ),
                "base_lrs": _control_value(
                    getattr(self.scheduler, "base_lrs", None),
                    location="scheduler.base_lrs",
                ),
                "progress": _control_value(
                    {
                        "last_epoch": scheduler_state.get("last_epoch"),
                        "_step_count": scheduler_state.get("_step_count"),
                    },
                    location="scheduler.progress",
                ),
            }
        return {
            "task": self.task.as_dict(),
            "max_epochs": self.config.max_epochs,
            "gradient_accumulation_steps": (
                self.config.gradient_accumulation_steps
            ),
            "precision": self.config.precision,
            "gradient_clip_norm": self.config.gradient_clip_norm,
            "early_stopping_patience": (
                self.config.early_stopping_patience
            ),
            "min_improvement": self.config.min_improvement,
            "non_blocking_transfer": self.config.non_blocking_transfer,
            "best_checkpoint_path": str(self.config.best_checkpoint_path),
            "latest_checkpoint_path": (
                None
                if self.config.latest_checkpoint_path is None
                else str(self.config.latest_checkpoint_path)
            ),
            "config_fingerprint": self.config_fingerprint,
            "loss": {
                "type": _type_identity(self.loss_fn),
                "task_type": self.loss_fn.task_type,
                "loss_type": self.loss_fn.loss_type,
                "huber_delta": self.loss_fn.huber_delta,
                "distributed_sync": self.loss_fn.distributed_sync,
                "validate_values": self.loss_fn.validate_values,
            },
            "optimizer": self._optimizer_control_signature(),
            "scheduler": scheduler_signature,
            "loaders": loaders,
            "scaler_present": self.scaler is not None,
        }

    def _validate_control_signature(self) -> None:
        if not self.context.distributed:
            return
        signature = _synchronize_constructor_preflight(
            "prepare finetuning control signature",
            self._control_signature,
        )
        signatures: list[dict[str, Any] | None] = [
            None for _ in range(self.context.world_size)
        ]
        dist.all_gather_object(signatures, signature)
        if any(candidate != signature for candidate in signatures):
            raise RuntimeError(
                "finetuning control configuration differs across ranks: "
                f"{signatures}"
            )

    @property
    def _loaders(self) -> dict[str, DataLoader]:
        return {
            "train": self.train_loader,
            "validation": self.valid_loader,
            "test": self.test_loader,
        }

    def _collective_device(self) -> torch.device:
        if self.context.distributed and self.context.backend == "nccl":
            return self.context.device
        return torch.device("cpu")

    def _synchronize_local_phase(
        self,
        operation: str,
        function: Callable[[], Any],
    ) -> Any:
        local_error: Exception | None = None
        result: Any = None
        try:
            result = function()
        except Exception as exc:
            local_error = exc
        if not self.context.distributed:
            if local_error is not None:
                raise local_error
            return result

        error_flag = torch.tensor(
            int(local_error is not None),
            dtype=torch.int32,
            device=self._collective_device(),
        )
        dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
        if int(error_flag.item()) == 0:
            return result
        descriptions: list[tuple[str, str] | None] = [
            None for _ in range(self.context.world_size)
        ]
        local_description = (
            None
            if local_error is None
            else (type(local_error).__name__, str(local_error))
        )
        dist.all_gather_object(descriptions, local_description)
        failures = [
            f"rank {rank}: {description[0]}: {description[1]}"
            for rank, description in enumerate(descriptions)
            if description is not None
        ]
        if local_error is not None:
            raise local_error
        raise RuntimeError(
            f"{operation} failed on another rank; " + "; ".join(failures)
        )

    def _coordinated_next(
        self,
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
        if self.context.distributed:
            status_tensor = torch.tensor(
                status,
                dtype=torch.int32,
                device=self._collective_device(),
            )
            minimum = status_tensor.clone()
            maximum = status_tensor.clone()
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            if int(maximum.item()) == 2:
                local_description = (
                    None
                    if local_error is None
                    else (type(local_error).__name__, str(local_error))
                )
                descriptions: list[tuple[str, str] | None] = [
                    None for _ in range(self.context.world_size)
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
            if int(minimum.item()) != int(maximum.item()):
                statuses: list[int | None] = [
                    None for _ in range(self.context.world_size)
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

    def _set_loader_epoch(
        self,
        name: str,
        loader: DataLoader,
        epoch: int,
    ) -> None:
        self._synchronize_local_phase(
            f"{name} epoch update",
            lambda: set_dataloader_epoch(loader, epoch),
        )

    def _create_loader_iterator(self, name: str, loader: DataLoader) -> Any:
        return self._synchronize_local_phase(
            f"{name} iterator creation",
            lambda: iter(loader),
        )

    def _prepare_checkpoint_directories(self) -> None:
        parents = {self.config.best_checkpoint_path.parent}
        if self.config.latest_checkpoint_path is not None:
            parents.add(self.config.latest_checkpoint_path.parent)

        def prepare() -> None:
            for directory in sorted(parents, key=str):
                directory.mkdir(parents=True, exist_ok=True)
                if not directory.is_dir():
                    raise NotADirectoryError(
                        f"checkpoint parent is not a directory: {directory}"
                    )

        self._synchronize_local_phase(
            "finetuning checkpoint-directory preparation",
            prepare,
        )

    def _matching_loader_length(self, name: str, loader: DataLoader) -> int:
        length = self._synchronize_local_phase(
            f"determine {name} length",
            lambda: _integer(f"len({name})", len(loader), minimum=1),
        )
        if not self.context.distributed:
            return length
        value = torch.tensor(
            length,
            dtype=torch.int64,
            device=self._collective_device(),
        )
        minimum = value.clone()
        maximum = value.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if int(minimum.item()) != int(maximum.item()):
            raise RuntimeError(
                f"{name} must yield the same number of batches on every rank"
            )
        return length

    def _any_rank_true(self, local_value: bool) -> bool:
        _strict_bool("local_value", local_value)
        if not self.context.distributed:
            return local_value
        flag = torch.tensor(
            int(local_value),
            dtype=torch.int32,
            device=self._collective_device(),
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(int(flag.item()))

    def _require_matching_bool(self, name: str, local_value: bool) -> bool:
        _strict_bool(name, local_value)
        if not self.context.distributed:
            return local_value
        value = torch.tensor(
            int(local_value),
            dtype=torch.int32,
            device=self._collective_device(),
        )
        minimum = value.clone()
        maximum = value.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if int(minimum.item()) != int(maximum.item()):
            raise RuntimeError(f"{name} differs across DDP ranks")
        return bool(int(minimum.item()))

    def _require_matching_object(self, name: str, value: Any) -> Any:
        if not self.context.distributed:
            return value
        gathered: list[Any] = [None for _ in range(self.context.world_size)]
        dist.all_gather_object(gathered, value)
        if any(item != value for item in gathered):
            raise RuntimeError(f"{name} differs across DDP ranks")
        return value

    def _broadcast_nonnegative_integer(self, value: int) -> int:
        local_value = (
            _integer("broadcast integer", value, minimum=0)
            if self.context.is_main_process
            else 0
        )
        tensor = torch.tensor(
            local_value,
            dtype=torch.int64,
            device=self._collective_device(),
        )
        if self.context.distributed:
            dist.broadcast(tensor, src=0)
        result = int(tensor.item())
        if result < 0:
            raise RuntimeError("received a negative integer from rank zero")
        return result

    def _raise_rank_zero_error(
        self,
        operation: str,
        error: Exception | None,
    ) -> None:
        if not self.context.distributed:
            if error is not None:
                raise error
            return
        status: list[tuple[str, str] | None] = [
            None
            if error is None
            else (type(error).__name__, str(error))
        ]
        dist.broadcast_object_list(status, src=0)
        description = status[0]
        if description is None:
            return
        if self.context.is_main_process and error is not None:
            raise error
        raise RuntimeError(
            f"rank zero failed during {operation} "
            f"({description[0]}): {description[1]}"
        )

    def _validate_batch_supervision(
        self,
        batch: Mapping[str, Any],
        *,
        require_source_index: bool,
        expected_device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if not isinstance(batch, Mapping):
            raise TypeError("each DataLoader batch must be a mapping")
        missing = [key for key in ("labels", "label_mask") if key not in batch]
        if missing:
            raise KeyError(f"finetuning batch is missing supervision keys {missing}")
        targets = batch["labels"]
        mask = batch["label_mask"]
        if not isinstance(targets, Tensor) or not isinstance(mask, Tensor):
            raise TypeError("batch labels and label_mask must be torch tensors")
        if targets.ndim != 2:
            raise ValueError(
                "batch labels must have shape [B, num_tasks], got "
                f"{tuple(targets.shape)}"
            )
        expected_shape = (targets.shape[0], self.task.num_tasks)
        if tuple(targets.shape) != expected_shape:
            raise ValueError(
                "batch labels must have shape [B, num_tasks], got "
                f"{tuple(targets.shape)}"
            )
        if not targets.is_floating_point():
            raise TypeError("batch labels must be floating point")
        if mask.dtype != torch.bool or mask.shape != targets.shape:
            raise ValueError(
                "batch label_mask must be boolean with the same shape as labels"
            )
        if targets.device != expected_device or mask.device != expected_device:
            raise ValueError(
                "labels and label_mask must be on the expected batch device "
                f"{expected_device}"
            )
        if bool(torch.isinf(targets).any()):
            raise ValueError("labels cannot contain positive or negative infinity")
        if not torch.equal(mask, torch.isfinite(targets)):
            raise ValueError(
                "label_mask must exactly identify finite label positions"
            )
        if self.task.task_type == "classification" and bool(
            torch.any((targets[mask] != 0) & (targets[mask] != 1))
        ):
            raise ValueError("classification labels must be exactly 0 or 1")

        source_index: Tensor | None = None
        if require_source_index:
            if "source_index" not in batch:
                raise KeyError("evaluation batch is missing source_index")
            source_index = batch["source_index"]
            if not isinstance(source_index, Tensor):
                raise TypeError("batch source_index must be a torch tensor")
            if source_index.dtype != torch.int64 or source_index.ndim != 1:
                raise ValueError("source_index must have shape [B] and dtype int64")
            if source_index.shape[0] != targets.shape[0]:
                raise ValueError("source_index and labels batch sizes must match")
            if source_index.device != expected_device:
                raise ValueError(
                    "source_index must be on the expected batch device "
                    f"{expected_device}"
                )
            if source_index.numel() and bool(torch.any(source_index < 0)):
                raise ValueError("source_index cannot contain negative values")
        return targets, mask, source_index

    def _validate_smiles_anchor(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
        expected_device: torch.device,
        modality_mask: Tensor,
    ) -> None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if not isinstance(input_ids, Tensor) or not isinstance(
            attention_mask,
            Tensor,
        ):
            raise TypeError("input_ids and attention_mask must be tensors")
        if input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
            raise ValueError("input_ids must have shape [B, sequence_length]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must have dtype torch.bool")
        if (
            input_ids.device != expected_device
            or attention_mask.device != expected_device
        ):
            raise ValueError("1d anchor tensors must be on the expected device")
        sequence_length = int(input_ids.shape[1])
        if sequence_length <= 0:
            raise ValueError("SMILES sequence length must be positive")

        encoder = self.base_model.encoders["1d"]
        vocab_size = getattr(encoder, "vocab_size", None)
        if not isinstance(vocab_size, Integral) or isinstance(vocab_size, bool):
            raise TypeError("SMILES encoder vocab_size must be an integer")
        if bool(torch.any(input_ids < 0)) or bool(
            torch.any(input_ids >= int(vocab_size))
        ):
            raise ValueError(
                f"input_ids must be in [0, {int(vocab_size) - 1}]"
            )
        if bool(
            torch.any(input_ids.masked_select(~attention_mask) != PAD_TOKEN_ID)
        ):
            raise ValueError(
                "positions outside attention_mask must contain ESPF [PAD]"
            )
        if bool(
            torch.any(input_ids.masked_select(attention_mask) == PAD_TOKEN_ID)
        ):
            raise ValueError("active SMILES positions cannot contain ESPF [PAD]")
        if sequence_length > 1 and bool(
            torch.any(attention_mask[:, 1:] & ~attention_mask[:, :-1])
        ):
            raise ValueError("attention_mask must use contiguous right padding")
        valid_rows = attention_mask.any(dim=1)
        if not bool(valid_rows.all()):
            raise ValueError(
                "finetuning requires a valid 1d anchor for every sample"
            )
        if not torch.equal(modality_mask[:, 0], valid_rows):
            raise ValueError(
                "modality_mask disagrees with 1d attention-mask presence"
            )
        pooling = getattr(encoder, "pooling", None)
        if pooling == "cls" and not bool(
            torch.all(input_ids[:, 0] == CLS_TOKEN_ID)
        ):
            raise ValueError(
                "CLS pooling requires ESPF [CLS] at the first active position"
            )

        transformer = getattr(encoder, "transformer", None)
        transformer_config = getattr(transformer, "config", None)
        padding_index = getattr(transformer_config, "pad_token_id", None)
        embeddings = getattr(transformer, "embeddings", None)
        position_embeddings = getattr(embeddings, "position_embeddings", None)
        position_count = getattr(position_embeddings, "num_embeddings", None)
        if not isinstance(padding_index, Integral) or isinstance(
            padding_index,
            bool,
        ):
            raise TypeError("SMILES transformer pad_token_id must be an integer")
        if not isinstance(position_count, Integral) or isinstance(
            position_count,
            bool,
        ):
            raise TypeError(
                "SMILES position-embedding capacity must be an integer"
            )
        maximum_position = int(padding_index) + sequence_length
        if maximum_position >= int(position_count):
            raise ValueError(
                f"SMILES sequence width {sequence_length} exceeds position "
                f"capacity {int(position_count) - int(padding_index) - 1}"
            )

    def _validate_graph_anchor(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
        expected_device: torch.device,
    ) -> None:
        graph = batch["graph"]
        sample_index = batch["graph_sample_index"]
        if not isinstance(graph, Batch):
            raise TypeError("graph must be a torch_geometric.data.Batch")
        if not isinstance(sample_index, Tensor):
            raise TypeError("graph_sample_index must be a tensor")
        if sample_index.ndim != 1 or sample_index.shape[0] != batch_size:
            raise ValueError("graph_sample_index must have shape [B]")
        if sample_index.dtype != torch.long:
            raise TypeError("graph_sample_index must have dtype torch.long")
        if sample_index.device != expected_device:
            raise ValueError("graph_sample_index must be on the expected device")
        expected_index = torch.arange(
            batch_size,
            dtype=torch.long,
            device=expected_device,
        )
        if not torch.equal(sample_index, expected_index):
            raise ValueError(
                "2d finetuning anchor must cover every sample exactly once in "
                "full-batch order"
            )

        node_features = getattr(graph, "x", None)
        edge_index = getattr(graph, "edge_index", None)
        edge_features = getattr(graph, "edge_attr", None)
        node_batch = getattr(graph, "batch", None)
        for name, value in (
            ("graph.x", node_features),
            ("graph.edge_index", edge_index),
            ("graph.edge_attr", edge_features),
            ("graph.batch", node_batch),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.device != expected_device:
                raise ValueError(f"{name} must be on the expected device")

        encoder = self.base_model.encoders["2d"]
        node_cardinalities = tuple(
            getattr(encoder, "node_feature_cardinalities", ())
        )
        edge_cardinalities = tuple(
            getattr(encoder, "edge_feature_cardinalities", ())
        )
        if not node_cardinalities or not edge_cardinalities:
            raise ValueError(
                "graph encoder must expose node and edge feature cardinalities"
            )
        if node_features.ndim != 2 or node_features.shape[1] != len(
            node_cardinalities
        ):
            raise ValueError(
                "graph.x has an incompatible categorical feature shape"
            )
        if node_features.dtype != torch.long:
            raise TypeError("graph.x must have dtype torch.long")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("graph.edge_index must have shape [2, edges]")
        if edge_index.dtype != torch.long:
            raise TypeError("graph.edge_index must have dtype torch.long")
        if edge_features.ndim != 2 or edge_features.shape[1] != len(
            edge_cardinalities
        ):
            raise ValueError(
                "graph.edge_attr has an incompatible categorical feature shape"
            )
        if edge_features.dtype != torch.long:
            raise TypeError("graph.edge_attr must have dtype torch.long")
        if edge_features.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "graph.edge_attr row count must equal graph.edge_index edge count"
            )

        node_count = int(node_features.shape[0])
        edge_count = int(edge_index.shape[1])
        if node_count <= 0:
            raise ValueError("every 2d anchor batch must contain graph nodes")
        if graph.num_nodes is None or int(graph.num_nodes) != node_count:
            raise ValueError("graph.num_nodes must equal graph.x row count")
        if int(graph.num_graphs) != batch_size:
            raise ValueError("graph.num_graphs must equal the full batch size")
        if node_batch.ndim != 1 or node_batch.shape[0] != node_count:
            raise ValueError("graph.batch must have one index per node")
        if node_batch.dtype != torch.long:
            raise TypeError("graph.batch must have dtype torch.long")
        if bool(torch.any(node_batch < 0)) or bool(
            torch.any(node_batch >= batch_size)
        ):
            raise ValueError("graph.batch contains an invalid graph index")
        if node_batch.numel() > 1 and bool(
            torch.any(node_batch[1:] < node_batch[:-1])
        ):
            raise ValueError("graph.batch must be grouped by graph index")
        graph_node_counts = torch.bincount(node_batch, minlength=batch_size)
        if graph_node_counts.shape[0] != batch_size or bool(
            torch.any(graph_node_counts == 0)
        ):
            raise ValueError("every full-batch graph must contain at least one node")

        if edge_count > 0:
            if bool(torch.any(edge_index < 0)) or bool(
                torch.any(edge_index >= node_count)
            ):
                raise ValueError("graph.edge_index contains an invalid node index")
            source, target = edge_index
            if bool(torch.any(node_batch[source] != node_batch[target])):
                raise ValueError("graph edges cannot connect different graphs")

        node_upper = node_features.new_tensor(node_cardinalities).unsqueeze(0)
        if bool(
            torch.any((node_features < 0) | (node_features >= node_upper))
        ):
            raise ValueError(
                "graph.x contains a category outside encoder cardinalities"
            )
        if edge_features.numel() > 0:
            edge_upper = edge_features.new_tensor(edge_cardinalities).unsqueeze(0)
            if bool(
                torch.any((edge_features < 0) | (edge_features >= edge_upper))
            ):
                raise ValueError(
                    "graph.edge_attr contains a category outside encoder "
                    "cardinalities"
                )

        graph_ptr = getattr(graph, "ptr", None)
        if graph_ptr is not None:
            if not isinstance(graph_ptr, Tensor):
                raise TypeError("graph.ptr must be a tensor when present")
            if (
                graph_ptr.ndim != 1
                or graph_ptr.dtype != torch.long
                or graph_ptr.device != expected_device
                or graph_ptr.shape[0] != batch_size + 1
            ):
                raise ValueError("graph.ptr must be int64 with shape [B + 1]")
            expected_ptr = torch.cat(
                (
                    graph_ptr.new_zeros((1,)),
                    graph_node_counts.cumsum(dim=0),
                )
            )
            if not torch.equal(graph_ptr, expected_ptr):
                raise ValueError("graph.ptr disagrees with graph.batch")

    def _validate_geometry_anchor(
        self,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
        expected_device: torch.device,
        modality_mask: Tensor,
    ) -> None:
        atomic_numbers = batch["atomic_numbers"]
        coords = batch["coords"]
        atom_mask = batch["atom_mask"]
        conformer_mask = batch["conformer_mask"]
        for name, value in (
            ("atomic_numbers", atomic_numbers),
            ("coords", coords),
            ("atom_mask", atom_mask),
            ("conformer_mask", conformer_mask),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.device != expected_device:
                raise ValueError(f"{name} must be on the expected device")
        if atomic_numbers.ndim != 2 or atomic_numbers.shape[0] != batch_size:
            raise ValueError("atomic_numbers must have shape [B, atoms]")
        if atomic_numbers.dtype != torch.long:
            raise TypeError("atomic_numbers must have dtype torch.long")
        if coords.ndim != 4 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape [B, conformers, atoms, 3]")
        if coords.dtype != torch.float32:
            raise TypeError("coords must have dtype torch.float32")
        if (
            coords.shape[0] != batch_size
            or coords.shape[2] != atomic_numbers.shape[1]
        ):
            raise ValueError(
                "coords batch and atom dimensions must match atomic_numbers"
            )
        if atom_mask.shape != atomic_numbers.shape or atom_mask.dtype != torch.bool:
            raise ValueError(
                "atom_mask must be boolean with the atomic_numbers shape"
            )
        expected_conformer_shape = (batch_size, coords.shape[1])
        if (
            tuple(conformer_mask.shape) != expected_conformer_shape
            or conformer_mask.dtype != torch.bool
        ):
            raise ValueError(
                "conformer_mask must be boolean with shape [B, conformers]"
            )
        if not bool(torch.isfinite(coords).all()):
            raise ValueError("coords must contain only finite values")
        if bool(torch.any(atomic_numbers[~atom_mask] != 0)):
            raise ValueError(
                "atomic_numbers outside atom_mask must use padding value zero"
            )
        valid_atomic_numbers = atomic_numbers[atom_mask]
        if valid_atomic_numbers.numel() and (
            bool(torch.any(valid_atomic_numbers < 1))
            or bool(torch.any(valid_atomic_numbers > 118))
        ):
            raise ValueError("valid atomic_numbers must be in [1, 118]")
        has_atoms = atom_mask.any(dim=1)
        has_conformers = conformer_mask.any(dim=1)
        if not torch.equal(has_atoms, has_conformers):
            raise ValueError(
                "each sample must have both atoms and a conformer, or neither"
            )
        if not bool(has_atoms.all()):
            raise ValueError(
                "finetuning requires a valid 3d anchor for every sample"
            )
        if not torch.equal(modality_mask[:, 2], has_conformers):
            raise ValueError(
                "modality_mask disagrees with 3d conformer presence"
            )

    def _validate_full_anchor_batch(
        self,
        batch: Mapping[str, Any],
        *,
        expected_device: torch.device,
    ) -> int:
        if not isinstance(batch, Mapping):
            raise TypeError("each finetuning batch must be a mapping")
        batch_size = self.base_model._batch_size(batch)
        for key in _FULL_BATCH_TENSOR_KEYS:
            if key not in batch:
                continue
            value = batch[key]
            if not isinstance(value, Tensor):
                raise TypeError(f"batch[{key!r}] must be a tensor")
            if value.device != expected_device:
                raise ValueError(
                    f"batch[{key!r}] must be on expected device "
                    f"{expected_device}"
                )
        modality_mask = batch.get("modality_mask")
        if not isinstance(modality_mask, Tensor):
            raise KeyError("finetuning batch requires tensor modality_mask")
        if modality_mask.shape != (batch_size, 4):
            raise ValueError("modality_mask must have shape [B, 4]")
        if modality_mask.dtype != torch.bool:
            raise TypeError("modality_mask must have dtype torch.bool")
        if modality_mask.device != expected_device:
            raise ValueError("modality_mask must be on the expected device")

        anchor = self.base_model.anchor_modality
        required_by_anchor = {
            "1d": ("input_ids", "attention_mask"),
            "2d": ("graph", "graph_sample_index"),
            "3d": (
                "atomic_numbers",
                "coords",
                "atom_mask",
                "conformer_mask",
            ),
        }
        if anchor not in required_by_anchor:
            raise ValueError(f"unsupported finetuning anchor modality {anchor!r}")
        required = required_by_anchor[anchor]
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                f"finetuning {anchor} anchor is missing inputs {missing}"
            )
        column = _ANCHOR_COLUMNS[anchor]
        if not bool(modality_mask[:, column].all()):
            raise ValueError(
                "finetuning requires the anchor modality for every sample"
            )
        if anchor == "1d":
            self._validate_smiles_anchor(
                batch,
                batch_size=batch_size,
                expected_device=expected_device,
                modality_mask=modality_mask,
            )
        elif anchor == "2d":
            self._validate_graph_anchor(
                batch,
                batch_size=batch_size,
                expected_device=expected_device,
            )
        else:
            self._validate_geometry_anchor(
                batch,
                batch_size=batch_size,
                expected_device=expected_device,
                modality_mask=modality_mask,
            )
        return batch_size

    def _validate_raw_training_batch(
        self,
        raw_batch: object,
    ) -> tuple[Mapping[str, Any], int]:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("each finetuning DataLoader batch must be a mapping")
        cpu_device = torch.device("cpu")
        _, mask, _ = self._validate_batch_supervision(
            raw_batch,
            require_source_index=False,
            expected_device=cpu_device,
        )
        self._validate_full_anchor_batch(
            raw_batch,
            expected_device=cpu_device,
        )
        return raw_batch, int(mask.sum().item())

    @staticmethod
    def _validate_valid_label_count(
        mask: Tensor,
        expected_count: int,
        *,
        operation: str,
    ) -> int:
        expected = _integer("expected valid-label count", expected_count, minimum=0)
        count = int(mask.sum().item())
        if count != expected:
            raise RuntimeError(
                f"{operation} changed the valid-label count: {count} != "
                f"{expected}"
            )
        return count

    @staticmethod
    def _validate_window_processed_count(
        processed_count: int,
        expected_count: int,
    ) -> None:
        processed = _integer("processed window label count", processed_count, minimum=0)
        expected = _integer("expected window label count", expected_count, minimum=0)
        if processed != expected:
            raise RuntimeError(
                "processed loss count differs from the prevalidated "
                f"accumulation-window count: {processed} != {expected}"
            )

    def _prepare_supervised_batch(
        self,
        raw_batch: object,
        *,
        require_source_index: bool,
    ) -> tuple[Mapping[str, Any], Tensor, Tensor, Tensor | None]:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("each finetuning DataLoader batch must be a mapping")
        moved = move_batch_to_device(
            raw_batch,
            self.context.device,
            non_blocking=self.config.non_blocking_transfer,
        )
        if not isinstance(moved, Mapping):
            raise TypeError("moved finetuning batch must remain a mapping")
        targets, mask, source_index = self._validate_batch_supervision(
            moved,
            require_source_index=require_source_index,
            expected_device=self.context.device,
        )
        batch_size = self._validate_full_anchor_batch(
            moved,
            expected_device=self.context.device,
        )
        if targets.shape[0] != batch_size:
            raise ValueError("supervision batch size differs from SemMol batch size")
        if require_source_index and source_index is None:
            raise RuntimeError("source_index validation returned None")
        return moved, targets, mask, source_index

    def _validate_finetuning_output(self, output: object) -> Tensor:
        if not isinstance(output, SemMolFinetuningOutput):
            raise TypeError(
                "SemMol finetune forward must return SemMolFinetuningOutput"
            )
        predictions = output.predictions
        if not isinstance(predictions, Tensor) or not predictions.is_floating_point():
            raise TypeError("SemMol finetune predictions must be floating point")
        if predictions.ndim != 2 or predictions.shape[1] != self.task.num_tasks:
            raise ValueError(
                "SemMol predictions must have shape [B, num_tasks], got "
                f"{tuple(predictions.shape)}"
            )
        if predictions.device != self.context.device:
            raise ValueError("SemMol predictions must be on context.device")
        return predictions

    @staticmethod
    def _validate_predictions_for_loss(
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor,
    ) -> None:
        if predictions.shape != targets.shape:
            raise ValueError("predictions and labels must have identical shapes")
        if predictions.device != targets.device or mask.device != targets.device:
            raise ValueError(
                "predictions, labels, and label_mask must share one device"
            )
        valid_predictions = predictions[mask]
        if valid_predictions.numel() and not bool(
            torch.isfinite(valid_predictions).all()
        ):
            raise FloatingPointError(
                "predictions contain non-finite values at valid labels"
            )

    def _gradient_values_are_finite(self) -> bool:
        for parameter in self.model.parameters():
            gradient = parameter.grad
            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                return False
        return True

    def _validate_loss_component(
        self,
        component: object,
        *,
        expected_local_count: int | None = None,
    ) -> LossComponent:
        if not isinstance(component, LossComponent):
            raise TypeError("DownstreamTaskLoss.compute must return LossComponent")
        for name, value in (
            ("loss", component.loss),
            ("numerator", component.numerator),
            ("local_count", component.local_count),
            ("global_count", component.global_count),
        ):
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise ValueError(f"loss component {name} must be a scalar Tensor")
            if value.device != self.context.device:
                raise ValueError(
                    f"loss component {name} must be on context.device"
                )
        if not component.loss.is_floating_point() or not (
            component.numerator.is_floating_point()
        ):
            raise TypeError("loss and numerator must be floating-point tensors")
        if component.local_count.dtype != torch.long or (
            component.global_count.dtype != torch.long
        ):
            raise TypeError("local_count and global_count must have dtype torch.long")
        local_count = int(component.local_count.item())
        global_count = int(component.global_count.item())
        if local_count < 0 or global_count < local_count:
            raise ValueError("loss component label counts are invalid")
        if expected_local_count is not None:
            expected = _integer(
                "expected_local_count",
                expected_local_count,
                minimum=0,
            )
            if local_count != expected:
                raise ValueError(
                    "DownstreamTaskLoss local_count disagrees with label_mask: "
                    f"{local_count} != {expected}"
                )
        if not bool(torch.isfinite(component.loss).all()) or not bool(
            torch.isfinite(component.numerator).all()
        ):
            raise FloatingPointError("downstream loss component is non-finite")
        return component

    @staticmethod
    def _accumulate_loss(
        local_numerator: Tensor,
        local_count: Tensor,
        component: LossComponent,
    ) -> None:
        local_numerator.add_(
            component.numerator.detach().to(dtype=torch.float64)
        )
        local_count.add_(component.local_count.detach().to(dtype=torch.int64))

    def _optimizer_update(self) -> bool:
        def prepare_gradients() -> None:
            self.precision.unscale_for_clipping(self.scaler, self.optimizer)
            if self.config.gradient_clip_norm is not None:
                clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clip_norm,
                    error_if_nonfinite=False,
                )

        self._synchronize_local_phase(
            "unscale and clip downstream gradients",
            prepare_gradients,
        )
        local_gradients_finite = self._synchronize_local_phase(
            "inspect downstream gradients",
            self._gradient_values_are_finite,
        )
        if self.scaler is None and self._any_rank_true(
            not local_gradients_finite
        ):
            self._synchronize_local_phase(
                "clear non-finite downstream gradients",
                lambda: self.optimizer.zero_grad(set_to_none=True),
            )
            raise FloatingPointError(
                "non-finite unscaled gradients were detected on at least one rank"
            )

        stepped = self._synchronize_local_phase(
            "downstream optimizer step",
            lambda: self.precision.step_optimizer(
                self.scaler,
                self.optimizer,
            ),
        )
        stepped = self._require_matching_bool("optimizer-step success", stepped)
        self._synchronize_local_phase(
            "clear gradients after downstream optimizer step",
            lambda: self.optimizer.zero_grad(set_to_none=True),
        )
        if stepped and self.scheduler is not None:
            self._synchronize_local_phase(
                "downstream scheduler step",
                self.scheduler.step,
            )
        return stepped

    def _global_loss(
        self,
        local_numerator: Tensor,
        local_count: Tensor,
        *,
        operation: str,
    ) -> tuple[float, int]:
        reduction_device = self._collective_device()
        prepared_numerator, prepared_count = self._synchronize_local_phase(
            f"prepare {operation} loss reductions",
            lambda: (
                local_numerator.detach().to(device=reduction_device),
                local_count.detach().to(device=reduction_device),
            ),
        )
        global_numerator = all_reduce_sum(prepared_numerator)
        global_count = all_reduce_sum(prepared_count)
        if not isinstance(global_numerator, Tensor) or not isinstance(
            global_count,
            Tensor,
        ):
            raise RuntimeError(f"{operation} reductions must return tensors")
        count = int(global_count.item())
        if count <= 0:
            raise ValueError(f"{operation} contains no valid labels")
        numerator = float(global_numerator.item())
        loss = numerator / count
        if not math.isfinite(loss):
            raise FloatingPointError(f"{operation} global loss is non-finite")
        return loss, count

    def _global_window_valid_label_count(
        self,
        local_count: int,
        *,
        operation: str,
    ) -> int:
        count = self._synchronize_local_phase(
            f"prepare {operation} valid-label count",
            lambda: torch.tensor(
                _integer("local window valid-label count", local_count, minimum=0),
                dtype=torch.int64,
                device=self._collective_device(),
            ),
        )
        if self.context.distributed:
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        global_count = int(count.item())
        if global_count < 0:
            raise RuntimeError(
                f"{operation} produced a negative global valid-label count"
            )
        return global_count

    def _clear_and_confirm_no_gradients(self, *, operation: str) -> None:
        self._synchronize_local_phase(
            f"clear gradients for {operation}",
            lambda: self.optimizer.zero_grad(set_to_none=True),
        )
        gradients_remain = self._synchronize_local_phase(
            f"inspect gradients for {operation}",
            lambda: any(
                parameter.grad is not None
                for parameter in self.model.parameters()
            ),
        )
        if self._any_rank_true(gradients_remain):
            raise RuntimeError(
                f"{operation} retained gradients after explicit clearing"
            )

    def _train_epoch(self, epoch: int, batch_count: int) -> tuple[float, int]:
        self._set_loader_epoch("train_loader", self.train_loader, epoch)
        iterator = self._create_loader_iterator("train_loader", self.train_loader)
        self._synchronize_local_phase(
            "enter downstream training mode",
            lambda: (
                self.model.train(),
                self.loss_fn.train(),
                self.local_loss_fn.train(),
            ),
        )
        self._synchronize_local_phase(
            "clear gradients before downstream epoch",
            lambda: self.optimizer.zero_grad(set_to_none=True),
        )
        local_numerator = torch.zeros(
            (),
            dtype=torch.float64,
            device=self.context.device,
        )
        local_count = torch.zeros(
            (),
            dtype=torch.int64,
            device=self.context.device,
        )
        successful_steps = 0
        accumulation = self.config.gradient_accumulation_steps
        completed = False
        try:
            if self._micro_step != 0:
                raise RuntimeError(
                    "training epoch requires an accumulation-free boundary"
                )
            for window_start in range(0, batch_count, accumulation):
                window_end = min(window_start + accumulation, batch_count)
                raw_window: list[tuple[int, Mapping[str, Any], int]] = []
                local_window_count = 0
                for batch_index in range(window_start, window_end):
                    raw_batch = self._coordinated_next(
                        iterator,
                        operation=f"train_loader batch {batch_index}",
                        expect_item=True,
                    )
                    validated_raw, batch_valid_count = (
                        self._synchronize_local_phase(
                            f"train batch {batch_index} CPU schema validation",
                            lambda raw_batch=raw_batch: (
                                self._validate_raw_training_batch(raw_batch)
                            ),
                        )
                    )
                    raw_window.append(
                        (batch_index, validated_raw, batch_valid_count)
                    )
                    local_window_count += batch_valid_count

                global_window_count = self._global_window_valid_label_count(
                    local_window_count,
                    operation=(
                        f"training accumulation window {window_start}:"
                        f"{window_end}"
                    ),
                )
                if global_window_count == 0:
                    self._micro_step = 0
                    self._clear_and_confirm_no_gradients(
                        operation=(
                            "zero-label training accumulation window "
                            f"{window_start}:{window_end}"
                        )
                    )
                    continue
                loss_scale = (
                    float(self.context.world_size) / float(global_window_count)
                )
                processed_local_count = 0
                for window_offset, (
                    batch_index,
                    raw_batch,
                    raw_valid_count,
                ) in enumerate(raw_window):
                    batch, targets, mask, _ = self._synchronize_local_phase(
                        f"train batch {batch_index} device preparation",
                        lambda raw_batch=raw_batch: (
                            self._prepare_supervised_batch(
                                raw_batch,
                                require_source_index=False,
                            )
                        ),
                    )
                    self._synchronize_local_phase(
                        f"train batch {batch_index} transferred-label check",
                        lambda: self._validate_valid_label_count(
                            mask,
                            raw_valid_count,
                            operation="device transfer",
                        ),
                    )
                    window_position = window_offset + 1
                    synchronize = batch_index + 1 == window_end

                    with no_sync_context(self.model, synchronize):
                        with self.precision.autocast(self.context.device):
                            output = self.model(batch, mode="finetune")
                            predictions = self._synchronize_local_phase(
                                f"train batch {batch_index} output validation",
                                lambda output=output: (
                                    self._validate_finetuning_output(output)
                                ),
                            )
                            self._synchronize_local_phase(
                                f"train batch {batch_index} prediction validation",
                                lambda: self._validate_predictions_for_loss(
                                    predictions,
                                    targets,
                                    mask,
                                ),
                            )
                            component = self._synchronize_local_phase(
                                f"train batch {batch_index} loss computation",
                                lambda: self._validate_loss_component(
                                    self.local_loss_fn.compute(
                                        predictions,
                                        targets,
                                        mask,
                                    ),
                                    expected_local_count=raw_valid_count,
                                ),
                            )
                            scaled_loss = component.numerator * loss_scale
                        if self.scaler is None:
                            scaled_loss.backward()
                        else:
                            self.scaler.scale(scaled_loss).backward()

                    self._micro_step = window_position
                    processed_local_count += int(component.local_count.item())
                    self._synchronize_local_phase(
                        f"train batch {batch_index} loss accumulation",
                        lambda: self._accumulate_loss(
                            local_numerator,
                            local_count,
                            component,
                        ),
                    )

                self._synchronize_local_phase(
                    f"training accumulation window {window_start} count check",
                    lambda: self._validate_window_processed_count(
                        processed_local_count,
                        local_window_count,
                    ),
                )
                if self._optimizer_update():
                    successful_steps += 1
                self._micro_step = 0

            self._coordinated_next(
                iterator,
                operation="train_loader exhaustion check",
                expect_item=False,
            )
            if self._micro_step != 0:
                raise RuntimeError(
                    "training epoch ended with an unfinished accumulation window"
                )
            gradients_remain = any(
                parameter.grad is not None for parameter in self.model.parameters()
            )
            if self._any_rank_true(gradients_remain):
                raise RuntimeError(
                    "training epoch ended with residual parameter gradients"
                )
            train_loss, _ = self._global_loss(
                local_numerator,
                local_count,
                operation="training epoch",
            )
            completed = True
            return train_loss, successful_steps
        finally:
            if not completed:
                self._micro_step = 0
                self.optimizer.zero_grad(set_to_none=True)

    def _rank_zero_metrics(
        self,
        gathered: IndexedPredictions | None,
    ) -> tuple[MetricResult | None, float, int, float, int]:
        metrics: MetricResult | None = None
        metric_value = 0.0
        sample_count = 0
        loss_value = 0.0
        valid_label_count = 0
        local_error: Exception | None = None
        if self.context.is_main_process:
            try:
                if gathered is None:
                    raise RuntimeError("rank zero did not receive gathered predictions")
                if not isinstance(gathered, IndexedPredictions):
                    raise TypeError(
                        "gathered predictions must be IndexedPredictions"
                    )
                sample_count = int(gathered.source_index.shape[0])
                if sample_count <= 0:
                    raise ValueError("evaluation contains no unique samples")
                if self.task.task_type == "classification":
                    metrics = evaluate_classification(
                        gathered.targets,
                        gathered.predictions,
                        gathered.mask,
                        task_names=self.task.task_names,
                    )
                else:
                    metrics = evaluate_regression(
                        gathered.targets,
                        gathered.predictions,
                        gathered.mask,
                        task_names=self.task.task_names,
                    )
                metric_value = float(getattr(metrics, self.task.main_metric))
                if not math.isfinite(metric_value):
                    raise ValueError(
                        f"validation main metric {self.task.main_metric} is "
                        "undefined or non-finite"
                    )
                deduplicated_loss = DownstreamTaskLoss(
                    task_type=self.loss_fn.task_type,
                    loss_type=self.loss_fn.loss_type,
                    huber_delta=self.loss_fn.huber_delta,
                    distributed_sync=False,
                    validate_values=self.loss_fn.validate_values,
                )
                component = deduplicated_loss.compute(
                    gathered.predictions,
                    gathered.targets,
                    gathered.mask,
                )
                if not isinstance(component, LossComponent):
                    raise TypeError(
                        "deduplicated DownstreamTaskLoss must return LossComponent"
                    )
                valid_label_count = int(component.local_count.item())
                if valid_label_count <= 0:
                    raise ValueError("evaluation contains no valid labels")
                if int(component.global_count.item()) != valid_label_count:
                    raise RuntimeError(
                        "non-distributed evaluation loss returned inconsistent "
                        "label counts"
                    )
                loss_value = float(component.loss.item())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(
                        "deduplicated evaluation loss is non-finite"
                    )
            except Exception as exc:
                local_error = exc
        self._raise_rank_zero_error(
            "deduplicated metric and loss evaluation",
            local_error,
        )
        metric_value = broadcast_float(metric_value, src=0)
        loss_value = broadcast_float(loss_value, src=0)
        sample_count = self._broadcast_nonnegative_integer(sample_count)
        valid_label_count = self._broadcast_nonnegative_integer(
            valid_label_count
        )
        return (
            metrics,
            metric_value,
            sample_count,
            loss_value,
            valid_label_count,
        )

    def _evaluate(
        self,
        loader: DataLoader,
        *,
        split: str,
        epoch: int,
        batch_count: int,
    ) -> EvaluationResult:
        self._set_loader_epoch(f"{split}_loader", loader, epoch)
        iterator = self._create_loader_iterator(f"{split}_loader", loader)
        was_training = self.model.training
        loss_was_training = self.loss_fn.training
        local_loss_was_training = self.local_loss_fn.training
        self._synchronize_local_phase(
            f"enter {split} evaluation mode",
            lambda: (
                self.model.eval(),
                self.loss_fn.eval(),
                self.local_loss_fn.eval(),
            ),
        )
        local_predictions: list[Tensor] = []
        local_targets: list[Tensor] = []
        local_masks: list[Tensor] = []
        local_indices: list[Tensor] = []
        try:
            with torch.no_grad():
                for batch_index in range(batch_count):
                    raw_batch = self._coordinated_next(
                        iterator,
                        operation=f"{split}_loader batch {batch_index}",
                        expect_item=True,
                    )
                    batch, targets, mask, source_index = (
                        self._synchronize_local_phase(
                            f"{split} batch {batch_index} preparation",
                            lambda raw_batch=raw_batch: (
                                self._prepare_supervised_batch(
                                    raw_batch,
                                    require_source_index=True,
                                )
                            ),
                        )
                    )
                    if source_index is None:
                        raise RuntimeError(
                            "prepared evaluation source_index cannot be None"
                        )
                    with self.precision.autocast(self.context.device):
                        output = self.model(batch, mode="finetune")
                        predictions = self._synchronize_local_phase(
                            f"{split} batch {batch_index} output validation",
                            lambda output=output: self._validate_finetuning_output(
                                output
                            ),
                        )
                        self._synchronize_local_phase(
                            f"{split} batch {batch_index} prediction validation",
                            lambda: self._validate_predictions_for_loss(
                                predictions,
                                targets,
                                mask,
                            ),
                        )
                    self._synchronize_local_phase(
                        f"{split} batch {batch_index} accumulation",
                        lambda: self._accumulate_evaluation_batch(
                            local_predictions=local_predictions,
                            local_targets=local_targets,
                            local_masks=local_masks,
                            local_indices=local_indices,
                            predictions=predictions,
                            targets=targets,
                            mask=mask,
                            source_index=source_index,
                        ),
                    )
            self._coordinated_next(
                iterator,
                operation=f"{split}_loader exhaustion check",
                expect_item=False,
            )
        finally:
            self.model.train(was_training)
            self.loss_fn.train(loss_was_training)
            self.local_loss_fn.train(local_loss_was_training)

        def concatenate() -> tuple[Tensor, Tensor, Tensor, Tensor]:
            if not local_predictions:
                return (
                    torch.empty(
                        (0, self.task.num_tasks),
                        dtype=torch.float32,
                        device=self.context.device,
                    ),
                    torch.empty(
                        (0, self.task.num_tasks),
                        dtype=torch.float32,
                        device=self.context.device,
                    ),
                    torch.empty(
                        (0, self.task.num_tasks),
                        dtype=torch.bool,
                        device=self.context.device,
                    ),
                    torch.empty(
                        (0,),
                        dtype=torch.int64,
                        device=self.context.device,
                    ),
                )
            return (
                torch.cat(local_predictions, dim=0),
                torch.cat(local_targets, dim=0),
                torch.cat(local_masks, dim=0),
                torch.cat(local_indices, dim=0),
            )

        predictions, targets, masks, indices = self._synchronize_local_phase(
            f"prepare {split} predictions",
            concatenate,
        )
        self._synchronize_local_phase(
            f"validate local {split} gather inputs",
            lambda: self._validate_gather_inputs(
                indices,
                predictions,
                targets,
                masks,
            ),
        )
        gathered = None
        gather_error: Exception | None = None
        try:
            gathered = gather_indexed_predictions(
                indices,
                predictions,
                targets,
                masks,
                dst=0,
            )
        except Exception as exc:
            gather_error = exc
        self._synchronize_local_phase(
            f"gather {split} predictions",
            lambda: self._raise_existing_error(gather_error),
        )

        (
            metrics,
            main_metric,
            sample_count,
            loss,
            valid_label_count,
        ) = self._rank_zero_metrics(
            gathered
        )
        return EvaluationResult(
            split=split,
            loss=loss,
            main_metric=main_metric,
            sample_count=sample_count,
            valid_label_count=valid_label_count,
            metrics=metrics,
        )

    @staticmethod
    def _accumulate_evaluation_batch(
        *,
        local_predictions: list[Tensor],
        local_targets: list[Tensor],
        local_masks: list[Tensor],
        local_indices: list[Tensor],
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor,
        source_index: Tensor,
    ) -> None:
        local_predictions.append(predictions.detach().to(torch.float32))
        local_targets.append(targets.detach())
        local_masks.append(mask.detach())
        local_indices.append(source_index.detach())

    def _validate_gather_inputs(
        self,
        source_index: Tensor,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor,
    ) -> None:
        if source_index.dtype != torch.int64 or source_index.ndim != 1:
            raise ValueError("gather source_index must be 1D int64")
        expected_shape = (source_index.shape[0], self.task.num_tasks)
        if predictions.shape != expected_shape or predictions.ndim != 2:
            raise ValueError(
                f"gather predictions must have shape {expected_shape}"
            )
        if targets.shape != expected_shape or mask.shape != expected_shape:
            raise ValueError(
                "gather targets and mask must exactly match prediction shape"
            )
        if not predictions.is_floating_point() or not targets.is_floating_point():
            raise TypeError("gather predictions and targets must be floating point")
        if mask.dtype != torch.bool:
            raise TypeError("gather mask must have dtype bool")
        if len(
            {
                source_index.device,
                predictions.device,
                targets.device,
                mask.device,
            }
        ) != 1:
            raise ValueError("all gather inputs must share one device")
        if predictions.device != self.context.device:
            raise ValueError("gather inputs must be on context.device")
        if source_index.numel() and bool(torch.any(source_index < 0)):
            raise ValueError("gather source_index cannot contain negative values")
        if bool(torch.isinf(targets).any()):
            raise ValueError("gather targets cannot contain infinity")
        if not torch.equal(mask, torch.isfinite(targets)):
            raise ValueError("gather mask must exactly identify finite targets")
        if bool(mask.any()) and not bool(torch.isfinite(predictions[mask]).all()):
            raise FloatingPointError(
                "gather predictions are non-finite at valid targets"
            )

    @staticmethod
    def _raise_existing_error(error: Exception | None) -> None:
        if error is not None:
            raise error

    def _is_improvement(self, metric: float, state: TrainerState) -> bool:
        if state.best_epoch < 0:
            return True
        if self.task.metric_direction == "maximize":
            return metric > state.best_metric + self.config.min_improvement
        return metric < state.best_metric - self.config.min_improvement

    def _learning_rates(self) -> tuple[float, ...]:
        rates: list[float] = []
        for index, group in enumerate(self.optimizer.param_groups):
            if "lr" not in group:
                raise KeyError(f"optimizer param group {index} is missing lr")
            rate = _finite_float(
                f"optimizer.param_groups[{index}].lr",
                group["lr"],
                minimum=0.0,
            )
            rates.append(rate)
        if not rates:
            raise ValueError("optimizer must contain at least one parameter group")
        return tuple(rates)

    def _checkpoint_extra(
        self,
        *,
        role: str,
        validation_metric: float,
    ) -> dict[str, object]:
        if role not in {"best", "latest"}:
            raise ValueError("checkpoint role must be 'best' or 'latest'")
        return {
            "trainer": _TRAINER_NAME,
            "trainer_schema_version": _TRAINER_SCHEMA_VERSION,
            "role": role,
            "task": self.task.as_dict(),
            "validation_main_metric": _finite_float(
                "validation_metric",
                validation_metric,
            ),
        }

    def _save_checkpoint(
        self,
        path: Path,
        *,
        state: TrainerState,
        role: str,
        validation_metric: float,
    ) -> None:
        if state.micro_step != 0 or self._micro_step != 0:
            raise RuntimeError("finetuning checkpoints require micro_step=0")
        save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            state=state,
            config_fingerprint=self.config_fingerprint,
            loaders=self._loaders,
            extra=self._checkpoint_extra(
                role=role,
                validation_metric=validation_metric,
            ),
            context=self.context,
        )

    def _validate_checkpoint_metadata(
        self,
        extra: Mapping[str, Any],
        state: TrainerState,
        *,
        require_role: str | None,
    ) -> None:
        if not isinstance(extra, Mapping):
            raise TypeError("finetuning checkpoint metadata must be a mapping")
        if not isinstance(state, TrainerState):
            raise TypeError("finetuning checkpoint state must be TrainerState")
        if require_role is not None and require_role not in {"best", "latest"}:
            raise ValueError("require_role must be 'best', 'latest', or None")
        expected_keys = {
            "trainer",
            "trainer_schema_version",
            "role",
            "task",
            "validation_main_metric",
        }
        if set(extra) != expected_keys:
            raise ValueError(
                "finetuning checkpoint metadata schema differs; "
                f"missing={sorted(expected_keys - set(extra))}, "
                f"extra={sorted(set(extra) - expected_keys)}"
            )
        if extra["trainer"] != _TRAINER_NAME:
            raise ValueError("checkpoint was not created by FinetuningTrainer")
        schema_version = extra["trainer_schema_version"]
        if (
            not isinstance(schema_version, Integral)
            or isinstance(schema_version, bool)
            or int(schema_version) != _TRAINER_SCHEMA_VERSION
        ):
            raise ValueError("unsupported finetuning trainer checkpoint schema")
        if extra["task"] != self.task.as_dict():
            raise ValueError("checkpoint downstream task definition does not match")
        role = extra["role"]
        if role not in {"best", "latest"}:
            raise ValueError("checkpoint metadata has an invalid role")
        if require_role is not None and role != require_role:
            raise ValueError(
                f"expected a {require_role!r} checkpoint, got {role!r}"
            )
        validation_metric = _finite_float(
            "checkpoint validation_main_metric",
            extra["validation_main_metric"],
        )
        if state.next_epoch > self.config.max_epochs:
            raise ValueError(
                "checkpoint next_epoch exceeds configured max_epochs"
            )
        if role == "best":
            if state.best_epoch < 0 or state.next_epoch <= 0:
                raise ValueError(
                    "best checkpoint state must identify a completed best epoch"
                )
            if state.best_epoch != state.next_epoch - 1:
                raise ValueError(
                    "best checkpoint state must end at its best epoch"
                )
            if state.bad_epochs != 0:
                raise ValueError("best checkpoint state must have bad_epochs=0")
            if validation_metric != state.best_metric:
                raise ValueError(
                    "best checkpoint metric does not match TrainerState.best_metric"
                )

    @staticmethod
    def _normalize_checkpoint_role_requirement(
        require_role: str | None,
    ) -> str | None:
        if require_role is not None and require_role not in {"best", "latest"}:
            raise ValueError("require_role must be 'best', 'latest', or None")
        return require_role

    @staticmethod
    def _normalize_expected_terminal_state(
        state: TrainerState | None,
    ) -> TrainerState | None:
        if state is not None and not isinstance(state, TrainerState):
            raise TypeError(
                "expected_terminal_state must be TrainerState or None"
            )
        return state

    @staticmethod
    def _validate_best_checkpoint_against_terminal_state(
        best_state: TrainerState,
        terminal_state: TrainerState,
    ) -> None:
        if not isinstance(best_state, TrainerState):
            raise TypeError("best checkpoint state must be TrainerState")
        if not isinstance(terminal_state, TrainerState):
            raise TypeError("terminal state must be TrainerState")
        if terminal_state.best_epoch < 0:
            raise ValueError("terminal state does not identify a best epoch")
        if best_state.best_epoch != terminal_state.best_epoch:
            raise ValueError(
                "best checkpoint best_epoch does not match terminal state: "
                f"{best_state.best_epoch} != {terminal_state.best_epoch}"
            )
        if best_state.best_metric != terminal_state.best_metric:
            raise ValueError(
                "best checkpoint best_metric does not match terminal state: "
                f"{best_state.best_metric} != {terminal_state.best_metric}"
            )
        expected_next_epoch = terminal_state.best_epoch + 1
        if best_state.next_epoch != expected_next_epoch:
            raise ValueError(
                "best checkpoint next_epoch must immediately follow its best "
                f"epoch: {best_state.next_epoch} != {expected_next_epoch}"
            )

    def _load_checkpoint(
        self,
        path: Path,
        *,
        require_role: str | None,
        expected_terminal_state: TrainerState | None = None,
    ) -> TrainingCheckpointLoadResult:
        normalized_role = self._synchronize_local_phase(
            "checkpoint role requirement validation",
            lambda: self._normalize_checkpoint_role_requirement(
                require_role
            ),
        )
        normalized_role = self._require_matching_object(
            "checkpoint role requirement",
            normalized_role,
        )
        expected_terminal = self._synchronize_local_phase(
            "expected terminal-state validation",
            lambda: self._normalize_expected_terminal_state(
                expected_terminal_state
            ),
        )
        expected_terminal_signature = (
            None
            if expected_terminal is None
            else expected_terminal.as_dict()
        )
        self._require_matching_object(
            "expected terminal state",
            expected_terminal_signature,
        )
        if expected_terminal is not None and normalized_role != "best":
            raise ValueError(
                "expected_terminal_state may only be used when loading the "
                "best checkpoint"
            )

        def validate_metadata(
            extra: Mapping[str, Any],
            state: TrainerState,
        ) -> None:
            self._validate_checkpoint_metadata(
                extra,
                state,
                require_role=normalized_role,
            )
            if expected_terminal is not None:
                self._validate_best_checkpoint_against_terminal_state(
                    state,
                    expected_terminal,
                )

        return load_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config_fingerprint=self.config_fingerprint,
            loaders=self._loaders,
            context=self.context,
            map_location=self.context.device,
            metadata_validator=validate_metadata,
        )

    def _run_callback(self, result: FinetuningEpochResult) -> None:
        local_error: Exception | None = None
        if self.context.is_main_process and self.epoch_callback is not None:
            try:
                self.epoch_callback(result)
            except Exception as exc:
                local_error = exc
        self._raise_rank_zero_error("epoch callback", local_error)

    def fit(
        self,
        *,
        resume_from: str | Path | None = None,
    ) -> FinetuningRunResult:
        """Fit, reload the validation-selected best checkpoint, then test once."""

        def validate_fit_request() -> None:
            if self._fit_started:
                raise RuntimeError(
                    "FinetuningTrainer.fit may be called only once; construct a "
                    "fresh trainer for another run or seed"
                )

        self._synchronize_local_phase(
            "finetuning fit request validation",
            validate_fit_request,
        )
        self._fit_started = True
        completed = False
        try:
            result = self._fit_once(resume_from=resume_from)
            completed = True
            return result
        finally:
            if not completed:
                self._micro_step = 0
                self.optimizer.zero_grad(set_to_none=True)

    def _fit_once(
        self,
        *,
        resume_from: str | Path | None,
    ) -> FinetuningRunResult:
        self._prepare_checkpoint_directories()
        train_batch_count = self._matching_loader_length(
            "train_loader",
            self.train_loader,
        )
        valid_batch_count = self._matching_loader_length(
            "valid_loader",
            self.valid_loader,
        )
        test_batch_count = self._matching_loader_length(
            "test_loader",
            self.test_loader,
        )
        self._synchronize_local_phase(
            "clear gradients before finetuning fit",
            lambda: self.optimizer.zero_grad(set_to_none=True),
        )
        self._micro_step = 0

        resumed_path = self._synchronize_local_phase(
            "resume request validation",
            lambda: (
                None
                if resume_from is None
                else _checkpoint_path("resume_from", resume_from)
            ),
        )
        resumed_path = self._require_matching_object(
            "resume request",
            resumed_path,
        )
        if resumed_path is None:
            state = TrainerState()
        else:
            loaded = self._load_checkpoint(
                resumed_path,
                require_role=None,
            )
            state = loaded.state
            if state.best_epoch >= 0:
                self._synchronize_local_phase(
                    "locate validation-best checkpoint",
                    lambda: self._require_existing_best_checkpoint(),
                )

        history: list[FinetuningEpochResult] = []
        terminal_state = state
        stopped_before_resume = (
            self.config.early_stopping_patience is not None
            and state.bad_epochs >= self.config.early_stopping_patience
        )
        first_epoch = (
            self.config.max_epochs if stopped_before_resume else state.next_epoch
        )
        for epoch in range(first_epoch, self.config.max_epochs):
            train_loss, successful_steps = self._train_epoch(
                epoch,
                train_batch_count,
            )
            learning_rates = self._synchronize_local_phase(
                "read downstream learning rates",
                self._learning_rates,
            )
            learning_rates = self._require_matching_object(
                "downstream learning rates",
                learning_rates,
            )
            validation = self._evaluate(
                self.valid_loader,
                split="validation",
                epoch=epoch,
                batch_count=valid_batch_count,
            )
            local_improved = (
                self._is_improvement(validation.main_metric, terminal_state)
                if self.context.is_main_process
                else False
            )
            improved = broadcast_bool(local_improved, src=0)
            if improved:
                best_metric = validation.main_metric
                best_epoch = epoch
                bad_epochs = 0
            else:
                best_metric = terminal_state.best_metric
                best_epoch = terminal_state.best_epoch
                bad_epochs = terminal_state.bad_epochs + 1

            terminal_state = TrainerState(
                next_epoch=epoch + 1,
                micro_step=0,
                optimizer_step=(
                    terminal_state.optimizer_step + successful_steps
                ),
                best_metric=best_metric,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
            )
            if improved:
                self._save_checkpoint(
                    self.config.best_checkpoint_path,
                    state=terminal_state,
                    role="best",
                    validation_metric=validation.main_metric,
                )
            if self.config.latest_checkpoint_path is not None:
                self._save_checkpoint(
                    self.config.latest_checkpoint_path,
                    state=terminal_state,
                    role="latest",
                    validation_metric=validation.main_metric,
                )

            epoch_result = FinetuningEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                validation=validation,
                optimizer_steps=successful_steps,
                learning_rates=learning_rates,
                improved=improved,
                bad_epochs=bad_epochs,
            )
            history.append(epoch_result)
            self._run_callback(epoch_result)

            should_stop = (
                self.config.early_stopping_patience is not None
                and bad_epochs >= self.config.early_stopping_patience
            )
            should_stop = broadcast_bool(
                should_stop if self.context.is_main_process else False,
                src=0,
            )
            if should_stop:
                break

        if terminal_state.best_epoch < 0:
            raise RuntimeError(
                "training produced no finite validation-selected best epoch"
            )
        best_loaded = self._load_checkpoint(
            self.config.best_checkpoint_path,
            require_role="best",
            expected_terminal_state=terminal_state,
        )
        self._synchronize_local_phase(
            "verify best checkpoint against terminal state before test",
            lambda: self._validate_best_checkpoint_against_terminal_state(
                best_loaded.state,
                terminal_state,
            ),
        )
        test = self._evaluate(
            self.test_loader,
            split="test",
            epoch=best_loaded.state.best_epoch,
            batch_count=test_batch_count,
        )
        self._synchronize_local_phase(
            "enter final finetuning evaluation mode",
            lambda: self.model.eval(),
        )
        return FinetuningRunResult(
            terminal_state=terminal_state,
            best_state=best_loaded.state,
            epochs=tuple(history),
            test=test,
            best_checkpoint_path=self.config.best_checkpoint_path,
            resumed_from=resumed_path,
        )

    def _require_existing_best_checkpoint(self) -> None:
        path = self.config.best_checkpoint_path
        if not path.is_file():
            raise FileNotFoundError(
                "resumed state references a best epoch, but the configured "
                f"best checkpoint does not exist: {path}"
            )


FinetuneTrainer = FinetuningTrainer
FinetuneTrainerConfig = FinetuningTrainerConfig


__all__ = [
    "DownstreamTaskDefinition",
    "EpochCallback",
    "EvaluationResult",
    "FinetuningEpochResult",
    "FinetuningRunResult",
    "FinetuningTrainer",
    "FinetuningTrainerConfig",
    "FinetuneTrainer",
    "FinetuneTrainerConfig",
    "MetricResult",
]

"""Ten-seed downstream evaluation with rank-consistent aggregation."""

from __future__ import annotations

import copy
import gc
import math
import re
import statistics
import traceback
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, TypeAlias

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.evaluation.metrics import ClassificationMetrics, RegressionMetrics
from src.losses.downstream_loss import DownstreamTaskLoss
from src.models.semmol import SemMol

from .common import DistributedContext, seed_everything
from .finetune_trainer import (
    DownstreamTaskDefinition,
    EpochCallback,
    FinetuningRunResult,
    FinetuningTrainer,
    FinetuningTrainerConfig,
)
from .runtime import (
    build_optimizer,
    build_scheduler,
    optimizer_steps_per_epoch,
    require_bool,
    require_real,
    require_string,
)


_MAX_SEED = 2**63 - 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_DISTRIBUTED_OPTION_KEYS = frozenset(
    {
        "broadcast_buffers",
        "find_unused_parameters",
        "sync_batchnorm",
        "sync_batch_norm",
    }
)


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _finite(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _optional_finite(name: str, value: object) -> float | None:
    if value is None:
        return None
    return _finite(name, value)


def _type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _callable_identity(value: Callable[..., Any]) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    return _type_identity(value)


def _copy_string_mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return copy.deepcopy(dict(value))


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        normalized = float(value)
        return normalized if math.isfinite(normalized) else None
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("metric dictionaries must use string keys")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(
        "metric payload contains a non-serializable value of type "
        f"{type(value).__name__}"
    )


def _json_safe_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON mappings must use string keys")
        return {key: _json_safe_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_copy(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_copy(item) for item in value]
    return _freeze_json(value)


def _validate_optimizer_options(options: Mapping[str, Any]) -> float:
    unknown = set(options) - {
        "type",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "amsgrad",
    }
    if unknown:
        raise ValueError(f"unsupported optimizer options: {sorted(unknown)}")
    optimizer_type = require_string(
        "optimizer.type",
        options.get("type", "adamw"),
    ).lower()
    if optimizer_type != "adamw":
        raise ValueError("only optimizer.type='adamw' is supported")
    learning_rate = require_real(
        "optimizer.lr",
        options.get("lr"),
        minimum=0.0,
        minimum_inclusive=False,
    )
    require_real(
        "optimizer.weight_decay",
        options.get("weight_decay", 0.0),
        minimum=0.0,
    )
    require_real(
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
    if any(beta >= 1.0 for beta in betas):
        raise ValueError("optimizer beta values must be smaller than 1")
    require_bool("optimizer.amsgrad", options.get("amsgrad", False))
    return learning_rate


def _validate_scheduler_options(
    options: Mapping[str, Any] | None,
    *,
    optimizer_learning_rate: float,
) -> None:
    if options is None:
        return
    unknown = set(options) - {"type", "warmup_ratio", "min_lr"}
    if unknown:
        raise ValueError(f"unsupported scheduler options: {sorted(unknown)}")
    scheduler_type = require_string(
        "scheduler.type",
        options.get("type", "cosine"),
    ).lower()
    if scheduler_type == "none":
        if set(options) - {"type"}:
            raise ValueError("scheduler.type='none' cannot have additional options")
        return
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
    if min_lr > optimizer_learning_rate:
        raise ValueError("scheduler.min_lr cannot exceed optimizer.lr")


@dataclass(frozen=True)
class PreparedFinetuningRun:
    """Purely local inputs from which the runner constructs one trainer."""

    seed: int
    model: nn.Module
    loss_fn: DownstreamTaskLoss
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    config: FinetuningTrainerConfig
    config_fingerprint: str
    optimizer_options: Mapping[str, Any]
    scheduler_options: Mapping[str, Any] | None = None
    distributed_options: Mapping[str, Any] = field(default_factory=dict)
    epoch_callback: EpochCallback | None = None

    def __post_init__(self) -> None:
        seed = _integer("seed", self.seed)
        if seed > _MAX_SEED:
            raise ValueError(f"seed must be at most {_MAX_SEED}")
        if not isinstance(self.model, SemMol):
            raise TypeError("model must be an unwrapped SemMol")
        if isinstance(self.model, DistributedDataParallel):
            raise TypeError("model must not be DistributedDataParallel-wrapped")
        if not isinstance(self.loss_fn, DownstreamTaskLoss):
            raise TypeError("loss_fn must be DownstreamTaskLoss")
        loaders = (self.train_loader, self.valid_loader, self.test_loader)
        if any(not isinstance(loader, DataLoader) for loader in loaders):
            raise TypeError("train, validation, and test loaders must be DataLoader")
        if len({id(loader) for loader in loaders}) != len(loaders):
            raise ValueError("train, validation, and test loaders must be distinct")
        if self.valid_loader.drop_last or self.test_loader.drop_last:
            raise ValueError("validation and test loaders must use drop_last=False")
        for name, loader in zip(("train", "validation", "test"), loaders):
            if not isinstance(loader.generator, torch.Generator):
                raise ValueError(
                    f"{name}_loader must have an explicit torch.Generator"
                )
            if int(loader.generator.initial_seed()) != seed:
                raise ValueError(
                    f"{name}_loader generator seed must equal Prepared seed"
                )
            if not callable(loader.collate_fn):
                raise TypeError(f"{name}_loader collate_fn must be callable")
            sampler = loader.sampler
            if isinstance(sampler, DistributedSampler) and int(sampler.seed) != seed:
                raise ValueError(
                    f"{name}_loader DistributedSampler seed must equal Prepared seed"
                )
        if not isinstance(self.config, FinetuningTrainerConfig):
            raise TypeError("config must be FinetuningTrainerConfig")
        if not isinstance(self.config_fingerprint, str) or _SHA256_PATTERN.fullmatch(
            self.config_fingerprint
        ) is None:
            raise ValueError(
                "config_fingerprint must be a lowercase SHA-256 hexadecimal string"
            )
        optimizer_options = _copy_string_mapping(
            "optimizer_options",
            self.optimizer_options,
        )
        scheduler_options = (
            None
            if self.scheduler_options is None
            else _copy_string_mapping("scheduler_options", self.scheduler_options)
        )
        optimizer_learning_rate = _validate_optimizer_options(optimizer_options)
        _validate_scheduler_options(
            scheduler_options,
            optimizer_learning_rate=optimizer_learning_rate,
        )
        distributed_options = _copy_string_mapping(
            "distributed_options",
            self.distributed_options,
        )
        unknown = set(distributed_options) - _DISTRIBUTED_OPTION_KEYS
        if unknown:
            raise ValueError(
                f"unsupported distributed options: {sorted(unknown)}"
            )
        if "sync_batchnorm" in distributed_options and "sync_batch_norm" in (
            distributed_options
        ):
            raise ValueError(
                "distributed options cannot define both sync_batchnorm and "
                "sync_batch_norm"
            )
        sync_batchnorm = distributed_options.pop(
            "sync_batch_norm",
            distributed_options.get("sync_batchnorm", False),
        )
        broadcast_buffers = distributed_options.get("broadcast_buffers", False)
        find_unused = distributed_options.get("find_unused_parameters", False)
        for name, value in (
            ("broadcast_buffers", broadcast_buffers),
            ("sync_batchnorm", sync_batchnorm),
            ("find_unused_parameters", find_unused),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"distributed_options.{name} must be bool")
        if broadcast_buffers:
            raise ValueError(
                "distributed_options.broadcast_buffers must be false for DCL"
            )
        distributed_options = {
            "broadcast_buffers": False,
            "sync_batchnorm": sync_batchnorm,
            "find_unused_parameters": find_unused,
        }
        if self.epoch_callback is not None and not callable(self.epoch_callback):
            raise TypeError("epoch_callback must be callable or None")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "optimizer_options", optimizer_options)
        object.__setattr__(self, "scheduler_options", scheduler_options)
        object.__setattr__(self, "distributed_options", distributed_options)


PreparedRunFactory: TypeAlias = Callable[[int], PreparedFinetuningRun]
TrainerFactory: TypeAlias = PreparedRunFactory


@dataclass(frozen=True)
class SeedRunSummary:
    """Primitive, serializable outcome for one independent seed."""

    seed: int
    best_epoch: int
    best_validation_metric: float
    test_loss: float
    test_main_metric: float
    test_sample_count: int
    test_valid_label_count: int
    macro_metrics: tuple[tuple[str, float | None], ...]
    metric_details: Mapping[str, Any]

    def __post_init__(self) -> None:
        seed = _integer("seed", self.seed)
        if seed > _MAX_SEED:
            raise ValueError(f"seed must be at most {_MAX_SEED}")
        best_epoch = _integer("best_epoch", self.best_epoch)
        best_metric = _finite(
            "best_validation_metric",
            self.best_validation_metric,
        )
        test_loss = _finite("test_loss", self.test_loss)
        test_metric = _finite("test_main_metric", self.test_main_metric)
        sample_count = _integer(
            "test_sample_count",
            self.test_sample_count,
            minimum=1,
        )
        label_count = _integer(
            "test_valid_label_count",
            self.test_valid_label_count,
            minimum=1,
        )
        if not isinstance(self.macro_metrics, tuple) or not self.macro_metrics:
            raise ValueError("macro_metrics must be a non-empty tuple")
        normalized_metrics: list[tuple[str, float | None]] = []
        seen: set[str] = set()
        for index, item in enumerate(self.macro_metrics):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    f"macro_metrics[{index}] must be a (name, value) tuple"
                )
            name, value = item
            if not isinstance(name, str) or not name:
                raise ValueError("macro metric names must be non-empty strings")
            if name in seen:
                raise ValueError(f"duplicate macro metric name {name!r}")
            seen.add(name)
            normalized_metrics.append(
                (name, _optional_finite(f"macro_metrics[{name}]", value))
            )
        if not isinstance(self.metric_details, Mapping):
            raise TypeError("metric_details must be a mapping")
        safe_details = _freeze_json(self.metric_details)
        if not isinstance(safe_details, Mapping):
            raise TypeError("metric_details normalization must produce a mapping")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "best_epoch", best_epoch)
        object.__setattr__(self, "best_validation_metric", best_metric)
        object.__setattr__(self, "test_loss", test_loss)
        object.__setattr__(self, "test_main_metric", test_metric)
        object.__setattr__(self, "test_sample_count", sample_count)
        object.__setattr__(self, "test_valid_label_count", label_count)
        object.__setattr__(self, "macro_metrics", tuple(normalized_metrics))
        object.__setattr__(self, "metric_details", safe_details)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "best_validation_metric": self.best_validation_metric,
            "test_loss": self.test_loss,
            "test_main_metric": self.test_main_metric,
            "test_sample_count": self.test_sample_count,
            "test_valid_label_count": self.test_valid_label_count,
            "macro_metrics": dict(self.macro_metrics),
            "metric_details": _json_safe_copy(self.metric_details),
        }


@dataclass(frozen=True)
class MetricAggregate:
    """Mean and sample standard deviation across eligible seed values."""

    name: str
    values: tuple[float | None, ...]
    eligible_seed_count: int
    mean: float | None
    sample_standard_deviation: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("aggregate name must be a non-empty string")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("aggregate values must be a non-empty tuple")
        values = tuple(
            _optional_finite(f"{self.name}.values[{index}]", value)
            for index, value in enumerate(self.values)
        )
        eligible = tuple(value for value in values if value is not None)
        eligible_count = _integer(
            "eligible_seed_count",
            self.eligible_seed_count,
        )
        if eligible_count != len(eligible):
            raise ValueError(
                "eligible_seed_count must equal the number of finite values"
            )
        expected_mean = statistics.fmean(eligible) if eligible else None
        expected_std = statistics.stdev(eligible) if len(eligible) >= 2 else None
        mean = _optional_finite("aggregate mean", self.mean)
        standard_deviation = _optional_finite(
            "aggregate sample_standard_deviation",
            self.sample_standard_deviation,
        )
        if mean != expected_mean:
            raise ValueError("aggregate mean does not match values")
        if standard_deviation != expected_std:
            raise ValueError(
                "aggregate sample standard deviation does not match values"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "eligible_seed_count", eligible_count)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(
            self,
            "sample_standard_deviation",
            standard_deviation,
        )

    @classmethod
    def from_values(
        cls,
        name: str,
        values: Sequence[float | None],
    ) -> "MetricAggregate":
        normalized = tuple(values)
        eligible = tuple(value for value in normalized if value is not None)
        return cls(
            name=name,
            values=normalized,
            eligible_seed_count=len(eligible),
            mean=statistics.fmean(eligible) if eligible else None,
            sample_standard_deviation=(
                statistics.stdev(eligible) if len(eligible) >= 2 else None
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "eligible_seed_count": self.eligible_seed_count,
            "mean": self.mean,
            "sample_standard_deviation": self.sample_standard_deviation,
        }


def _derive_aggregates(
    task: DownstreamTaskDefinition,
    runs: tuple[SeedRunSummary, ...],
) -> tuple[MetricAggregate, ...]:
    expected_names = (
        ("macro_roc_auc",)
        if task.task_type == "classification"
        else ("macro_rmse", "macro_mae", "macro_r2")
    )
    values_by_name: dict[str, tuple[float | None, ...]] = {}
    for run in runs:
        names = tuple(name for name, _ in run.macro_metrics)
        if names != expected_names:
            raise ValueError(
                "run macro metric schema does not match the downstream task"
            )
    for name in expected_names:
        values_by_name[name] = tuple(
            dict(run.macro_metrics)[name] for run in runs
        )
    main_values = values_by_name[task.main_metric]
    if any(
        macro_value != run.test_main_metric
        for macro_value, run in zip(main_values, runs)
    ):
        raise ValueError(
            "test_main_metric must equal the matching macro metric for every run"
        )
    return tuple(
        MetricAggregate.from_values(name, values_by_name[name])
        for name in expected_names
    )


@dataclass(frozen=True)
class TenSeedBenchmarkResult:
    task: DownstreamTaskDefinition
    runs: tuple[SeedRunSummary, ...]
    aggregates: tuple[MetricAggregate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task, DownstreamTaskDefinition):
            raise TypeError("task must be DownstreamTaskDefinition")
        if not isinstance(self.runs, tuple) or len(self.runs) != 10:
            raise ValueError("runs must contain exactly ten seed summaries")
        if any(not isinstance(run, SeedRunSummary) for run in self.runs):
            raise TypeError("runs must contain only SeedRunSummary values")
        seeds = tuple(run.seed for run in self.runs)
        if len(set(seeds)) != len(seeds):
            raise ValueError("run seeds must be unique")
        if not isinstance(self.aggregates, tuple) or not self.aggregates:
            raise ValueError("aggregates must be a non-empty tuple")
        if any(not isinstance(item, MetricAggregate) for item in self.aggregates):
            raise TypeError("aggregates must contain MetricAggregate values")
        expected_aggregates = _derive_aggregates(self.task, self.runs)
        if self.aggregates != expected_aggregates:
            raise ValueError(
                "aggregates must exactly match the values, means, and sample "
                "standard deviations derived from runs"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": _json_safe_copy(self.task.as_dict()),
            "seeds": [run.seed for run in self.runs],
            "runs": [run.as_dict() for run in self.runs],
            "aggregates": {
                aggregate.name: aggregate.as_dict()
                for aggregate in self.aggregates
            },
        }


class _LiveObjectRegistry:
    """Remember only objects that remain alive after a completed seed."""

    def __init__(self) -> None:
        self._weak: dict[
            int,
            tuple[int, str, weakref.ReferenceType[Any]],
        ] = {}
        self._strong: dict[int, tuple[int, str, object]] = {}

    def require_fresh(
        self,
        seed: int,
        named_objects: Sequence[tuple[str, object]],
    ) -> None:
        self._weak = {
            identity: entry
            for identity, entry in self._weak.items()
            if entry[2]() is not None
        }
        current: dict[int, tuple[str, object]] = {}
        for label, value in named_objects:
            current.setdefault(id(value), (label, value))
        for identity, (label, value) in current.items():
            weak_entry = self._weak.get(identity)
            if weak_entry is not None and weak_entry[2]() is value:
                raise ValueError(
                    f"{label} for seed {seed} reuses a still-live object from "
                    f"seed {weak_entry[0]} ({weak_entry[1]})"
                )
            strong_entry = self._strong.get(identity)
            if strong_entry is not None and strong_entry[2] is value:
                raise ValueError(
                    f"{label} for seed {seed} reuses a non-weak-referenceable "
                    f"object from seed {strong_entry[0]} ({strong_entry[1]})"
                )
        for identity, (label, value) in current.items():
            try:
                reference = weakref.ref(value)
            except TypeError:
                self._strong[identity] = (seed, label, value)
            else:
                self._weak[identity] = (seed, label, reference)


class TenSeedFinetuningRunner:
    """Build ten trainers from a factory that must perform local work only.

    The prepared-run factory must not initialize DDP, construct a trainer, or
    invoke any distributed collective.  The runner owns every collective phase.
    """

    def __init__(
        self,
        *,
        context: DistributedContext,
        task: DownstreamTaskDefinition,
        seeds: Sequence[int],
        prepared_run_factory: PreparedRunFactory,
        deterministic: bool = False,
        cudnn_benchmark: bool = False,
    ) -> None:
        active = dist.is_available() and dist.is_initialized()
        active_world = dist.get_world_size() if active else 1
        active_rank = dist.get_rank() if active else 0
        active_backend = str(dist.get_backend()) if active else None
        local_error: Exception | None = None
        normalized_seeds: tuple[int, ...] = ()
        try:
            self._validate_context_against_active_group(
                context,
                active=active,
                active_world=active_world,
                active_rank=active_rank,
                active_backend=active_backend,
            )
            if not isinstance(task, DownstreamTaskDefinition):
                raise TypeError("task must be DownstreamTaskDefinition")
            if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
                raise TypeError("seeds must be a sequence of integers")
            normalized_seeds = tuple(
                _integer(f"seeds[{index}]", seed)
                for index, seed in enumerate(seeds)
            )
            if len(normalized_seeds) != 10:
                raise ValueError("the paper protocol requires exactly ten seeds")
            if any(seed > _MAX_SEED for seed in normalized_seeds):
                raise ValueError(f"every seed must be at most {_MAX_SEED}")
            if len(set(normalized_seeds)) != len(normalized_seeds):
                raise ValueError("the ten seeds must be unique")
            if not callable(prepared_run_factory):
                raise TypeError("prepared_run_factory must be callable")
            if not isinstance(deterministic, bool):
                raise TypeError("deterministic must be bool")
            if not isinstance(cudnn_benchmark, bool):
                raise TypeError("cudnn_benchmark must be bool")
            if deterministic and cudnn_benchmark:
                raise ValueError(
                    "deterministic and cudnn_benchmark cannot both be true"
                )
        except Exception as exc:
            local_error = exc
        self._coordinate_active_preflight(
            active=active,
            world_size=active_world,
            operation="ten-seed runner construction",
            local_error=local_error,
        )

        self.context = context
        self.task = task
        self.seeds = normalized_seeds
        self.prepared_run_factory = prepared_run_factory
        self.deterministic = deterministic
        self.cudnn_benchmark = cudnn_benchmark
        self._live_objects = _LiveObjectRegistry()
        self._started = False
        self._require_matching_controls()

    @staticmethod
    def _validate_context_against_active_group(
        context: object,
        *,
        active: bool,
        active_world: int,
        active_rank: int,
        active_backend: str | None,
    ) -> None:
        if not isinstance(context, DistributedContext):
            raise TypeError("context must be DistributedContext")
        if context.world_size != active_world or context.rank != active_rank:
            raise RuntimeError("DistributedContext does not match the active job")
        if context.distributed != (active_world > 1):
            raise RuntimeError("DistributedContext.distributed is inconsistent")
        if active and context.backend != active_backend:
            raise RuntimeError(
                "DistributedContext.backend does not match the active process group"
            )
        if active_world > 1 and not active:
            raise RuntimeError("distributed runner requires an active process group")
        if active_world > 1 and context.device.type == "cuda":
            if active_backend != "nccl":
                raise RuntimeError("distributed CUDA finetuning requires NCCL")
            if context.device.index is None:
                raise ValueError("distributed CUDA context requires an explicit index")
            if torch.cuda.current_device() != context.device.index:
                raise RuntimeError(
                    "current CUDA device does not match DistributedContext.device"
                )
        if active_world > 1 and active_backend == "nccl" and (
            context.device.type != "cuda"
        ):
            raise RuntimeError("NCCL finetuning requires a CUDA context device")

    @staticmethod
    def _coordinate_active_preflight(
        *,
        active: bool,
        world_size: int,
        operation: str,
        local_error: Exception | None,
    ) -> None:
        if not active or world_size == 1:
            if local_error is not None:
                raise local_error
            return
        local_description = (
            None
            if local_error is None
            else (type(local_error).__name__, str(local_error))
        )
        descriptions: list[tuple[str, str] | None] = [None] * world_size
        dist.all_gather_object(descriptions, local_description)
        if all(description is None for description in descriptions):
            return
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

    def _require_matching_controls(self) -> None:
        if not self.context.distributed:
            return
        local = {
            "seeds": self.seeds,
            "task": self.task.as_dict(),
            "deterministic": self.deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
        }
        controls: list[dict[str, Any] | None] = [
            None for _ in range(self.context.world_size)
        ]
        dist.all_gather_object(controls, local)
        if any(control != controls[0] for control in controls):
            raise RuntimeError("ten-seed controls differ across DDP ranks")

    def _collective_device(self) -> torch.device:
        if self.context.distributed and str(dist.get_backend()) == "nccl":
            return self.context.device
        return torch.device("cpu")

    def _coordinated_local_call(
        self,
        operation: str,
        function: Callable[[], Any],
    ) -> Any:
        """Coordinate work that is guaranteed not to invoke a collective."""

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
        description = (
            None
            if local_error is None
            else (type(local_error).__name__, str(local_error))
        )
        descriptions: list[tuple[str, str] | None] = [
            None for _ in range(self.context.world_size)
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

    def _validate_prepared_run(
        self,
        value: object,
        *,
        seed: int,
    ) -> PreparedFinetuningRun:
        if not isinstance(value, PreparedFinetuningRun):
            raise TypeError(
                "prepared_run_factory must return PreparedFinetuningRun"
            )
        if value.seed != seed:
            raise ValueError(
                f"PreparedFinetuningRun.seed={value.seed} does not match {seed}"
            )
        property_head = value.model.property_head
        if property_head is None:
            raise ValueError("prepared SemMol must have a downstream property head")
        if property_head.task_type != self.task.task_type:
            raise ValueError("prepared model task_type differs from benchmark task")
        if property_head.num_tasks != self.task.num_tasks:
            raise ValueError("prepared model num_tasks differs from benchmark task")
        if value.loss_fn.task_type != self.task.task_type:
            raise ValueError("prepared loss task_type differs from benchmark task")
        if self.context.distributed and not value.loss_fn.distributed_sync:
            raise ValueError("distributed prepared loss must enable distributed_sync")
        sync_batchnorm = value.distributed_options["sync_batchnorm"]
        if sync_batchnorm and not self.context.distributed:
            raise ValueError("SyncBatchNorm requires a distributed process group")
        if sync_batchnorm and self.context.device.type != "cuda":
            raise ValueError("SyncBatchNorm finetuning requires CUDA")
        for name, loader in (
            ("train", value.train_loader),
            ("validation", value.valid_loader),
            ("test", value.test_loader),
        ):
            sampler = loader.sampler
            if self.context.distributed and not isinstance(
                sampler,
                DistributedSampler,
            ):
                raise TypeError(f"{name}_loader must use DistributedSampler")
            if isinstance(sampler, DistributedSampler):
                if sampler.num_replicas != self.context.world_size:
                    raise ValueError(
                        f"{name}_loader sampler world size differs from context"
                    )
                if sampler.rank != self.context.rank:
                    raise ValueError(
                        f"{name}_loader sampler rank differs from context"
                    )
                if name != "train" and sampler.drop_last:
                    raise ValueError(
                        f"{name}_loader sampler must use drop_last=False"
                    )
            collator_seed = getattr(loader.collate_fn, "seed", seed)
            if (
                not isinstance(collator_seed, Integral)
                or isinstance(collator_seed, bool)
                or int(collator_seed) != seed
            ):
                raise ValueError(
                    f"{name}_loader collator seed must equal Prepared seed"
                )
        if not any(parameter.requires_grad for parameter in value.model.parameters()):
            raise ValueError("prepared model has no trainable parameters")
        return value

    def _track_prepared_objects(self, prepared: PreparedFinetuningRun) -> None:
        named_objects: list[tuple[str, object]] = []
        for prefix, module in (
            ("model", prepared.model),
            ("loss_fn", prepared.loss_fn),
        ):
            named_objects.append((prefix, module))
            named_objects.extend(
                (
                    f"{prefix}.module[{qualified_name}]",
                    child,
                )
                for qualified_name, child in module.named_modules()
                if qualified_name
            )
            named_objects.extend(
                (
                    f"{prefix}.parameter[{qualified_name}]",
                    parameter,
                )
                for qualified_name, parameter in module.named_parameters()
            )
            named_objects.extend(
                (
                    f"{prefix}.buffer[{qualified_name}]",
                    buffer,
                )
                for qualified_name, buffer in module.named_buffers()
            )
        named_objects.extend(
            [
                ("train_loader", prepared.train_loader),
                ("valid_loader", prepared.valid_loader),
                ("test_loader", prepared.test_loader),
                ("train_loader.collate_fn", prepared.train_loader.collate_fn),
                ("valid_loader.collate_fn", prepared.valid_loader.collate_fn),
                ("test_loader.collate_fn", prepared.test_loader.collate_fn),
            ]
        )
        self._live_objects.require_fresh(prepared.seed, named_objects)

    @staticmethod
    def _loader_signature(name: str, loader: DataLoader) -> dict[str, Any]:
        batch_count = len(loader)
        dataset_count = len(loader.dataset)
        if batch_count <= 0:
            raise ValueError(f"{name}_loader must contain at least one batch")
        if dataset_count <= 0:
            raise ValueError(f"{name}_loader dataset must not be empty")
        sampler = loader.sampler
        sampler_signature: dict[str, Any] = {
            "type": _type_identity(sampler),
        }
        if isinstance(sampler, DistributedSampler):
            sampler_signature.update(
                {
                    "num_replicas": sampler.num_replicas,
                    "shuffle": sampler.shuffle,
                    "seed": sampler.seed,
                    "drop_last": sampler.drop_last,
                }
            )
        return {
            "type": _type_identity(loader),
            "batch_count": batch_count,
            "dataset_type": _type_identity(loader.dataset),
            "dataset_count": dataset_count,
            "batch_size": loader.batch_size,
            "drop_last": loader.drop_last,
            "generator_seed": loader.generator.initial_seed(),
            "sampler": sampler_signature,
            "collator_type": _callable_identity(loader.collate_fn),
            "collator_seed": getattr(loader.collate_fn, "seed", None),
        }

    def _prepared_signature(
        self,
        prepared: PreparedFinetuningRun,
    ) -> tuple[dict[str, Any], int]:
        parameter_schema = tuple(
            (
                name,
                tuple(int(dimension) for dimension in parameter.shape),
                str(parameter.dtype),
                parameter.requires_grad,
            )
            for name, parameter in prepared.model.named_parameters()
        )
        buffer_schema = tuple(
            (
                name,
                tuple(int(dimension) for dimension in buffer.shape),
                str(buffer.dtype),
            )
            for name, buffer in prepared.model.named_buffers()
        )
        loader_signatures = {
            name: self._loader_signature(name, loader)
            for name, loader in (
                ("train", prepared.train_loader),
                ("validation", prepared.valid_loader),
                ("test", prepared.test_loader),
            )
        }
        config = prepared.config
        signature = {
            "seed": prepared.seed,
            "fingerprint": prepared.config_fingerprint,
            "model_type": _type_identity(prepared.model),
            "parameter_schema": parameter_schema,
            "buffer_schema": buffer_schema,
            "loss": {
                "type": _type_identity(prepared.loss_fn),
                "task_type": prepared.loss_fn.task_type,
                "loss_type": prepared.loss_fn.loss_type,
                "huber_delta": prepared.loss_fn.huber_delta,
                "distributed_sync": prepared.loss_fn.distributed_sync,
                "validate_values": prepared.loss_fn.validate_values,
            },
            "trainer_config": {
                "max_epochs": config.max_epochs,
                "best_checkpoint_path": str(config.best_checkpoint_path),
                "latest_checkpoint_path": (
                    None
                    if config.latest_checkpoint_path is None
                    else str(config.latest_checkpoint_path)
                ),
                "gradient_accumulation_steps": (
                    config.gradient_accumulation_steps
                ),
                "precision": config.precision,
                "gradient_clip_norm": config.gradient_clip_norm,
                "early_stopping_patience": config.early_stopping_patience,
                "min_improvement": config.min_improvement,
                "non_blocking_transfer": config.non_blocking_transfer,
            },
            "optimizer_options": _json_safe_copy(prepared.optimizer_options),
            "scheduler_options": _json_safe_copy(prepared.scheduler_options),
            "distributed_options": _json_safe_copy(
                prepared.distributed_options
            ),
            "epoch_callback_type": (
                None
                if prepared.epoch_callback is None
                else _callable_identity(prepared.epoch_callback)
            ),
            "loaders": loader_signatures,
        }
        return signature, int(loader_signatures["train"]["batch_count"])

    def _require_matching_prepared_signature(
        self,
        signature: Mapping[str, Any],
    ) -> None:
        if not self.context.distributed:
            return
        signatures: list[Mapping[str, Any] | None] = [
            None for _ in range(self.context.world_size)
        ]
        dist.all_gather_object(signatures, dict(signature))
        if any(candidate != signatures[0] for candidate in signatures):
            raise RuntimeError("prepared finetuning controls differ across ranks")

    def _prepare_model_and_loss(
        self,
        prepared: PreparedFinetuningRun,
    ) -> tuple[SemMol, DownstreamTaskLoss]:
        model = self._coordinated_local_call(
            f"move seed {prepared.seed} model to device",
            lambda: prepared.model.to(self.context.device),
        )
        if prepared.distributed_options["sync_batchnorm"]:
            model = self._coordinated_local_call(
                f"convert seed {prepared.seed} model to SyncBatchNorm",
                lambda model=model: nn.SyncBatchNorm.convert_sync_batchnorm(model),
            )
        loss_fn = self._coordinated_local_call(
            f"move seed {prepared.seed} loss to device",
            lambda: prepared.loss_fn.to(self.context.device),
        )

        def validate_ready() -> None:
            if not isinstance(model, SemMol):
                raise TypeError("prepared model conversion must preserve SemMol")
            if not isinstance(loss_fn, DownstreamTaskLoss):
                raise TypeError("prepared loss conversion changed its type")
            model_devices = {
                value.device
                for value in tuple(model.parameters()) + tuple(model.buffers())
            }
            if model_devices and model_devices != {self.context.device}:
                raise ValueError("prepared model is not wholly on context.device")
            loss_devices = {
                value.device
                for value in tuple(loss_fn.parameters()) + tuple(loss_fn.buffers())
            }
            if loss_devices and loss_devices != {self.context.device}:
                raise ValueError("prepared loss is not wholly on context.device")

        self._coordinated_local_call(
            f"seed {prepared.seed} pre-DDP readiness",
            validate_ready,
        )
        return model, loss_fn

    def _wrap_model_collectively(
        self,
        model: SemMol,
        distributed_options: Mapping[str, Any],
    ) -> nn.Module:
        """Enter DDP only after every rank has passed the local preflight.

        DDP construction performs collectives itself, so this method must never
        be called through ``_coordinated_local_call``.  A rank-local failure
        inside the constructor (for example CUDA OOM) is deliberately allowed
        to propagate so torchrun's fail-fast supervision can terminate peers;
        attempting a second coordination collective from that rank could only
        create a mismatched collective.
        """

        if not self.context.distributed:
            return model
        find_unused = distributed_options["find_unused_parameters"]
        if self.context.device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[self.context.device.index],
                output_device=self.context.device.index,
                broadcast_buffers=False,
                find_unused_parameters=find_unused,
            )
        return DistributedDataParallel(
            model,
            device_ids=None,
            broadcast_buffers=False,
            find_unused_parameters=find_unused,
        )

    def _post_ddp_signature(
        self,
        model: nn.Module,
        prepared: PreparedFinetuningRun,
    ) -> dict[str, Any]:
        expected_find_unused = prepared.distributed_options[
            "find_unused_parameters"
        ]
        if self.context.distributed:
            if not isinstance(model, DistributedDataParallel):
                raise TypeError("distributed model construction did not return DDP")
            if model.module is not prepared.model:
                raise ValueError("DDP must wrap the exact prepared SemMol instance")
            if dist.get_world_size(model.process_group) != self.context.world_size:
                raise ValueError("post-DDP process-group world size differs")
            if dist.get_rank(model.process_group) != self.context.rank:
                raise ValueError("post-DDP process-group rank differs")
            if str(dist.get_backend(model.process_group)) != self.context.backend:
                raise ValueError("post-DDP process-group backend differs")
            if model.broadcast_buffers:
                raise ValueError("post-DDP broadcast_buffers must remain false")
            if model.find_unused_parameters != expected_find_unused:
                raise ValueError("post-DDP find_unused_parameters differs")
            if self.context.device.type == "cuda":
                if model.device_ids != [self.context.device.index]:
                    raise ValueError("post-DDP CUDA device_ids differ from context")
                if model.output_device != self.context.device.index:
                    raise ValueError("post-DDP CUDA output_device differs from context")
            elif model.device_ids:
                raise ValueError("post-DDP CPU model must not define device_ids")
            base_model = model.module
        else:
            if isinstance(model, DistributedDataParallel):
                raise ValueError("single-process model must not be DDP-wrapped")
            if model is not prepared.model:
                raise ValueError(
                    "single-process setup must preserve the prepared model"
                )
            base_model = model

        if not isinstance(base_model, SemMol):
            raise TypeError("post-DDP base model must be SemMol")
        model_devices = {
            value.device
            for value in tuple(base_model.parameters()) + tuple(base_model.buffers())
        }
        if model_devices and model_devices != {self.context.device}:
            raise ValueError("post-DDP model is not wholly on context.device")
        parameter_schema = tuple(
            (
                name,
                tuple(int(dimension) for dimension in parameter.shape),
                str(parameter.dtype),
                parameter.requires_grad,
            )
            for name, parameter in base_model.named_parameters()
        )
        buffer_schema = tuple(
            (
                name,
                tuple(int(dimension) for dimension in buffer.shape),
                str(buffer.dtype),
            )
            for name, buffer in base_model.named_buffers()
        )
        return {
            "wrapper_type": _type_identity(model),
            "base_model_type": _type_identity(base_model),
            "parameter_schema": parameter_schema,
            "buffer_schema": buffer_schema,
            "device_type": self.context.device.type,
            "distributed": self.context.distributed,
            "broadcast_buffers": (
                model.broadcast_buffers
                if isinstance(model, DistributedDataParallel)
                else None
            ),
            "find_unused_parameters": (
                model.find_unused_parameters
                if isinstance(model, DistributedDataParallel)
                else None
            ),
            "sync_batchnorm": prepared.distributed_options["sync_batchnorm"],
        }

    def _require_matching_post_ddp_signature(
        self,
        signature: Mapping[str, Any],
    ) -> None:
        if not self.context.distributed:
            return
        signatures: list[Mapping[str, Any] | None] = [
            None for _ in range(self.context.world_size)
        ]
        dist.all_gather_object(signatures, dict(signature))
        if any(candidate != signatures[0] for candidate in signatures):
            raise RuntimeError("post-DDP model controls differ across ranks")

    @staticmethod
    def _validate_run_result(value: object) -> FinetuningRunResult:
        if not isinstance(value, FinetuningRunResult):
            raise TypeError("FinetuningTrainer.fit returned an invalid result")
        return value

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
            None if error is None else (type(error).__name__, str(error))
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

    def _summarize_root(
        self,
        seed: int,
        result: FinetuningRunResult,
    ) -> dict[str, Any] | None:
        if not self.context.is_main_process:
            return None
        metrics = result.test.metrics
        if self.task.task_type == "classification":
            if not isinstance(metrics, ClassificationMetrics):
                raise TypeError(
                    "rank-zero classification test result lacks "
                    "ClassificationMetrics"
                )
            macro_metrics = (("macro_roc_auc", metrics.macro_roc_auc),)
        else:
            if not isinstance(metrics, RegressionMetrics):
                raise TypeError(
                    "rank-zero regression test result lacks RegressionMetrics"
                )
            macro_metrics = (
                ("macro_rmse", metrics.macro_rmse),
                ("macro_mae", metrics.macro_mae),
                (
                    "macro_r2",
                    metrics.macro_r2
                    if math.isfinite(metrics.macro_r2)
                    else None,
                ),
            )
        summary = SeedRunSummary(
            seed=seed,
            best_epoch=result.best_state.best_epoch,
            best_validation_metric=result.best_state.best_metric,
            test_loss=result.test.loss,
            test_main_metric=result.test.main_metric,
            test_sample_count=result.test.sample_count,
            test_valid_label_count=result.test.valid_label_count,
            macro_metrics=macro_metrics,
            metric_details=metrics.as_dict(),
        )
        return summary.as_dict()

    @staticmethod
    def _summary_from_payload(payload: Mapping[str, Any]) -> SeedRunSummary:
        required = {
            "seed",
            "best_epoch",
            "best_validation_metric",
            "test_loss",
            "test_main_metric",
            "test_sample_count",
            "test_valid_label_count",
            "macro_metrics",
            "metric_details",
        }
        if set(payload) != required:
            raise ValueError("broadcast seed summary has an invalid schema")
        raw_metrics = payload["macro_metrics"]
        if not isinstance(raw_metrics, Mapping):
            raise TypeError("broadcast macro_metrics must be a mapping")
        details = payload["metric_details"]
        if not isinstance(details, Mapping):
            raise TypeError("broadcast metric_details must be a mapping")
        return SeedRunSummary(
            seed=payload["seed"],
            best_epoch=payload["best_epoch"],
            best_validation_metric=payload["best_validation_metric"],
            test_loss=payload["test_loss"],
            test_main_metric=payload["test_main_metric"],
            test_sample_count=payload["test_sample_count"],
            test_valid_label_count=payload["test_valid_label_count"],
            macro_metrics=tuple(raw_metrics.items()),
            metric_details=dict(details),
        )

    def _broadcast_summary(
        self,
        seed: int,
        result: FinetuningRunResult,
    ) -> SeedRunSummary:
        payload: dict[str, Any] | None = None
        local_error: Exception | None = None
        if self.context.is_main_process:
            try:
                payload = self._summarize_root(seed, result)
            except Exception as exc:
                local_error = exc
        self._raise_rank_zero_error("seed-result serialization", local_error)
        container: list[dict[str, Any] | None] = [payload]
        if self.context.distributed:
            dist.broadcast_object_list(container, src=0)
        received = container[0]
        if not isinstance(received, Mapping):
            raise RuntimeError("rank zero did not broadcast a seed summary")
        summary = self._coordinated_local_call(
            "seed-result schema validation",
            lambda: self._summary_from_payload(received),
        )
        if not isinstance(summary, SeedRunSummary):
            raise TypeError("seed-result validation returned an invalid summary")
        return summary

    @staticmethod
    def _seed_loaders_for_cleanup(
        prepared: PreparedFinetuningRun | None,
        trainer: FinetuningTrainer | None,
    ) -> tuple[DataLoader, ...]:
        candidates: list[DataLoader] = []
        if prepared is not None:
            candidates.extend(
                [
                    prepared.train_loader,
                    prepared.valid_loader,
                    prepared.test_loader,
                ]
            )
        if trainer is not None:
            candidates.extend(
                [
                    trainer.train_loader,
                    trainer.valid_loader,
                    trainer.test_loader,
                ]
            )
        unique: dict[int, DataLoader] = {}
        for loader in candidates:
            unique.setdefault(id(loader), loader)
        return tuple(unique.values())

    @staticmethod
    def _shutdown_loader_workers(loader: DataLoader) -> None:
        """Stop a persistent DataLoader iterator without touching its dataset."""

        if not isinstance(loader, DataLoader):
            raise TypeError("seed cleanup requires DataLoader instances")
        iterator = getattr(loader, "_iterator", None)
        if iterator is None:
            return
        try:
            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if not callable(shutdown_workers):
                if loader.persistent_workers:
                    raise RuntimeError(
                        "persistent DataLoader iterator cannot shut down workers"
                    )
                return
            shutdown_workers()
        finally:
            loader._iterator = None

    @staticmethod
    def _attach_cleanup_failures(
        primary_error: BaseException,
        *,
        seed: int,
        cleanup_failures: Sequence[BaseException],
    ) -> None:
        if not cleanup_failures:
            return
        descriptions = "; ".join(
            f"{type(error).__name__}: {error}" for error in cleanup_failures
        )
        message = f"seed {seed} local cleanup also failed: {descriptions}"
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            try:
                add_note(message)
            except BaseException:
                note_added = False
            else:
                note_added = True
            if note_added:
                return
        try:
            existing = getattr(primary_error, "semmol_cleanup_notes", ())
            if not isinstance(existing, tuple):
                existing = (str(existing),)
            setattr(
                primary_error,
                "semmol_cleanup_notes",
                (*existing, message),
            )
        except BaseException:
            return
        try:
            cleanup_context = RuntimeError(message)
            cleanup_context.__context__ = primary_error.__context__
            primary_error.__context__ = cleanup_context
        except BaseException:
            return

    @staticmethod
    def _raise_cleanup_error(error: Exception | None) -> None:
        if error is not None:
            raise error

    def _release_seed_resources(self) -> None:
        gc.collect()
        if self.context.device.type == "cuda":
            torch.cuda.empty_cache()

    def run(self) -> TenSeedBenchmarkResult:
        def validate_request() -> None:
            if self._started:
                raise RuntimeError(
                    "TenSeedFinetuningRunner.run may be called only once"
                )

        self._coordinated_local_call(
            "ten-seed run request validation",
            validate_request,
        )
        self._started = True
        summaries: list[SeedRunSummary] = []
        for seed in self.seeds:
            prepared: PreparedFinetuningRun | None = None
            model: nn.Module | None = None
            loss_fn: DownstreamTaskLoss | None = None
            optimizer: Optimizer | None = None
            scheduler: Any = None
            trainer: FinetuningTrainer | None = None
            result: FinetuningRunResult | None = None
            seed_summary: SeedRunSummary | None = None
            primary_error: BaseException | None = None
            cleanup_error: BaseException | None = None
            try:
                self._coordinated_local_call(
                    f"seed global runtime for seed {seed}",
                    lambda seed=seed: seed_everything(
                        seed,
                        deterministic=self.deterministic,
                        cudnn_benchmark=self.cudnn_benchmark,
                    ),
                )
                prepared = self._coordinated_local_call(
                    f"prepare local finetuning inputs for seed {seed}",
                    lambda seed=seed: self.prepared_run_factory(seed),
                )
                prepared = self._coordinated_local_call(
                    f"validate prepared finetuning inputs for seed {seed}",
                    lambda prepared=prepared, seed=seed: (
                        self._validate_prepared_run(prepared, seed=seed)
                    ),
                )
                self._coordinated_local_call(
                    f"verify fresh resources for seed {seed}",
                    lambda prepared=prepared: self._track_prepared_objects(
                        prepared
                    ),
                )
                prepared_signature, train_batch_count = (
                    self._coordinated_local_call(
                        f"derive prepared controls for seed {seed}",
                        lambda prepared=prepared: self._prepared_signature(
                            prepared
                        ),
                    )
                )
                self._require_matching_prepared_signature(prepared_signature)
                model, loss_fn = self._prepare_model_and_loss(prepared)
                model = self._wrap_model_collectively(
                    model,
                    prepared.distributed_options,
                )
                post_ddp_signature = self._coordinated_local_call(
                    f"validate post-DDP model for seed {seed}",
                    lambda model=model, prepared=prepared: (
                        self._post_ddp_signature(model, prepared)
                    ),
                )
                self._require_matching_post_ddp_signature(post_ddp_signature)
                optimizer = self._coordinated_local_call(
                    f"build optimizer for seed {seed}",
                    lambda model=model, prepared=prepared: build_optimizer(
                        model,
                        prepared.optimizer_options,
                    ),
                )
                steps_per_epoch = optimizer_steps_per_epoch(
                    train_batch_count,
                    prepared.config.gradient_accumulation_steps,
                )
                total_optimizer_steps = (
                    steps_per_epoch * prepared.config.max_epochs
                )
                scheduler = self._coordinated_local_call(
                    f"build scheduler for seed {seed}",
                    lambda optimizer=optimizer, prepared=prepared: (
                        build_scheduler(
                            optimizer,
                            prepared.scheduler_options,
                            total_optimizer_steps=total_optimizer_steps,
                        )
                    ),
                )
                self._coordinated_local_call(
                    f"seed {seed} trainer-construction readiness",
                    lambda: None,
                )
                trainer = FinetuningTrainer(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loader=prepared.train_loader,
                    valid_loader=prepared.valid_loader,
                    test_loader=prepared.test_loader,
                    task=self.task,
                    config=prepared.config,
                    context=self.context,
                    config_fingerprint=prepared.config_fingerprint,
                    epoch_callback=prepared.epoch_callback,
                )
                result = trainer.fit()
                result = self._coordinated_local_call(
                    f"validate result for seed {seed}",
                    lambda result=result: self._validate_run_result(result),
                )
                seed_summary = self._broadcast_summary(seed, result)
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                # Keep failure cleanup strictly process-local.  In particular,
                # a rank that failed inside DDP construction must not enter a
                # new collective while torchrun is terminating the job.
                cleanup_failures: list[BaseException] = []
                try:
                    cleanup_loaders = self._seed_loaders_for_cleanup(
                        prepared,
                        trainer,
                    )
                except BaseException as exc:
                    cleanup_failures.append(exc)
                    cleanup_loaders = ()
                if optimizer is not None:
                    try:
                        optimizer.zero_grad(set_to_none=True)
                    except BaseException as exc:
                        cleanup_failures.append(exc)
                if model is not None:
                    try:
                        model.zero_grad(set_to_none=True)
                    except BaseException as exc:
                        cleanup_failures.append(exc)
                for loader in cleanup_loaders:
                    try:
                        self._shutdown_loader_workers(loader)
                    except BaseException as exc:
                        cleanup_failures.append(exc)

                result = None
                trainer = None
                scheduler = None
                optimizer = None
                model = None
                loss_fn = None
                prepared = None
                cleanup_loaders = ()

                errors_with_frames = tuple(cleanup_failures)
                if primary_error is not None:
                    errors_with_frames = (primary_error, *errors_with_frames)
                for error in errors_with_frames:
                    try:
                        traceback.clear_frames(error.__traceback__)
                    except BaseException as exc:
                        cleanup_failures.append(exc)
                try:
                    self._release_seed_resources()
                except BaseException as exc:
                    cleanup_failures.append(exc)

                if primary_error is not None:
                    self._attach_cleanup_failures(
                        primary_error,
                        seed=seed,
                        cleanup_failures=cleanup_failures,
                    )
                elif cleanup_failures:
                    cleanup_error = cleanup_failures[0]
                    self._attach_cleanup_failures(
                        cleanup_error,
                        seed=seed,
                        cleanup_failures=cleanup_failures[1:],
                    )

            if cleanup_error is not None and not isinstance(
                cleanup_error,
                Exception,
            ):
                raise cleanup_error
            coordinated_cleanup_error = (
                cleanup_error if isinstance(cleanup_error, Exception) else None
            )
            self._coordinated_local_call(
                f"verify local resource cleanup for seed {seed}",
                lambda error=coordinated_cleanup_error: (
                    self._raise_cleanup_error(error)
                ),
            )

            def commit_seed_summary() -> None:
                if not isinstance(seed_summary, SeedRunSummary):
                    raise RuntimeError("completed seed did not produce a summary")
                summaries.append(seed_summary)

            self._coordinated_local_call(
                f"commit summary for seed {seed}",
                commit_seed_summary,
            )
            if self.context.distributed:
                dist.barrier()
        runs = tuple(summaries)
        return TenSeedBenchmarkResult(
            task=self.task,
            runs=runs,
            aggregates=_derive_aggregates(self.task, runs),
        )


__all__ = [
    "MetricAggregate",
    "PreparedFinetuningRun",
    "PreparedRunFactory",
    "SeedRunSummary",
    "TenSeedBenchmarkResult",
    "TenSeedFinetuningRunner",
    "TrainerFactory",
]

"""Shared, strictly validated utilities for SemMol trainers."""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data.data import BaseData


_MAX_SEED = 2**63 - 1
_PRECISION_ALIASES = {
    "none": "none",
    "fp32": "none",
    "amp": "fp16",
    "fp16": "fp16",
    "bf16": "bf16",
}
_T = TypeVar("_T")


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _strict_integer(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return normalized


def _coerce_device(value: str | torch.device, *, name: str) -> torch.device:
    if not isinstance(value, (str, torch.device)):
        raise TypeError(f"{name} must be a device string or torch.device")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} cannot be empty")
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{name} must select a CPU or CUDA device")
    if device.type == "cpu" and device.index is not None:
        raise ValueError(f"{name} cannot give an index for a CPU device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{name} requests CUDA, but CUDA is unavailable")
        index = 0 if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"{name} CUDA index {index} is outside the visible device range"
            )
        device = torch.device("cuda", index)
    return device


@dataclass(frozen=True)
class PrecisionMode:
    """Validated autocast policy and gradient-scaling operations."""

    mode: str = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError("precision mode must be a string")
        if self.mode not in _PRECISION_ALIASES:
            raise ValueError(
                "precision mode must be one of 'none', 'fp32', 'amp', "
                "'fp16', or 'bf16'"
            )
        object.__setattr__(self, "mode", _PRECISION_ALIASES[self.mode])

    @property
    def dtype(self) -> torch.dtype | None:
        if self.mode == "fp16":
            return torch.float16
        if self.mode == "bf16":
            return torch.bfloat16
        return None

    def autocast(
        self,
        device: str | torch.device,
    ) -> AbstractContextManager[Any]:
        resolved = _coerce_device(device, name="autocast device")
        if self.mode == "none":
            return nullcontext()
        if self.mode == "fp16" and resolved.type != "cuda":
            raise ValueError("fp16/amp autocast requires a CUDA device")
        return torch.autocast(
            device_type=resolved.type,
            dtype=self.dtype,
            enabled=True,
        )

    def create_grad_scaler(
        self,
        device: str | torch.device,
    ) -> torch.cuda.amp.GradScaler | None:
        resolved = _coerce_device(device, name="GradScaler device")
        if self.mode == "fp16" and resolved.type != "cuda":
            raise ValueError("fp16/amp gradient scaling requires a CUDA device")
        if self.mode != "fp16":
            return None
        return torch.cuda.amp.GradScaler(enabled=True)

    @staticmethod
    def unscale_for_clipping(
        scaler: torch.cuda.amp.GradScaler | None,
        optimizer: Optimizer,
    ) -> None:
        if scaler is not None and not isinstance(
            scaler,
            torch.cuda.amp.GradScaler,
        ):
            raise TypeError("scaler must be a CUDA GradScaler or None")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)

    @staticmethod
    def step_optimizer(
        scaler: torch.cuda.amp.GradScaler | None,
        optimizer: Optimizer,
    ) -> bool:
        """Step and report whether an fp16 overflow skipped the optimizer."""

        if scaler is not None and not isinstance(
            scaler,
            torch.cuda.amp.GradScaler,
        ):
            raise TypeError("scaler must be a CUDA GradScaler or None")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        if scaler is None or not scaler.is_enabled():
            optimizer.step()
            return True
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        return float(scaler.get_scale()) >= scale_before


@dataclass(frozen=True)
class DistributedContext:
    """The process-local view of an initialized torchrun job."""

    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None
    initialized_here: bool

    def __post_init__(self) -> None:
        _strict_bool("distributed", self.distributed)
        _strict_integer("rank", self.rank, minimum=0)
        _strict_integer("local_rank", self.local_rank, minimum=0)
        _strict_integer("world_size", self.world_size, minimum=1)
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be torch.device")
        if self.backend is not None and (
            not isinstance(self.backend, str) or not self.backend
        ):
            raise ValueError("backend must be a non-empty string or None")
        _strict_bool("initialized_here", self.initialized_here)
        if self.distributed != (self.world_size > 1):
            raise ValueError("distributed must agree with world_size")

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if not self.distributed:
            return
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("the distributed process group is not initialized")
        if dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
            raise RuntimeError("the active process group no longer matches context")
        dist.barrier()

    def close(self) -> None:
        if not self.initialized_here or not dist.is_initialized():
            return
        if dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
            raise RuntimeError("refusing to close a different process group")
        if self.distributed:
            dist.barrier()
        dist.destroy_process_group()


def _environment_integer(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if not value or value.strip() != value:
        raise ValueError(f"environment variable {name} must be an integer")
    try:
        return int(value, 10)
    except ValueError as exc:
        raise ValueError(
            f"environment variable {name} must be an integer"
        ) from exc


def initialize_distributed(
    options: Mapping[str, Any],
    requested_device: str | torch.device | None = None,
) -> DistributedContext:
    """Initialize from torchrun variables without inventing a multi-rank job."""

    if not isinstance(options, Mapping):
        raise TypeError("distributed options must be a mapping")
    if any(not isinstance(key, str) for key in options):
        raise TypeError("distributed option keys must be strings")

    configured_world: int | None = None
    if "world_size" in options:
        configured_world = _strict_integer(
            "distributed.world_size",
            options["world_size"],
            minimum=1,
        )
    configured_backend: str | None = None
    if "backend" in options:
        raw_backend = options["backend"]
        if not isinstance(raw_backend, str) or not raw_backend:
            raise ValueError("distributed.backend must be a non-empty string")
        if raw_backend != raw_backend.strip().lower():
            raise ValueError("distributed.backend must be lowercase without spaces")
        configured_backend = raw_backend

    env_values = {
        name: _environment_integer(name)
        for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK")
    }
    present_count = sum(value is not None for value in env_values.values())
    if present_count not in {0, 3}:
        raise RuntimeError(
            "WORLD_SIZE, RANK, and LOCAL_RANK must be supplied together by "
            "torchrun"
        )
    under_torchrun = present_count == 3
    if under_torchrun:
        world_size = _strict_integer(
            "WORLD_SIZE",
            env_values["WORLD_SIZE"],
            minimum=1,
        )
        rank = _strict_integer("RANK", env_values["RANK"], minimum=0)
        local_rank = _strict_integer(
            "LOCAL_RANK",
            env_values["LOCAL_RANK"],
            minimum=0,
        )
        if rank >= world_size:
            raise ValueError("RANK must be smaller than WORLD_SIZE")
        if configured_world is not None and configured_world != world_size:
            raise RuntimeError(
                "distributed.world_size does not match torchrun WORLD_SIZE: "
                f"{configured_world} != {world_size}"
            )
    else:
        world_size = 1
        rank = 0
        local_rank = 0
        if configured_world is not None and configured_world > 1:
            raise RuntimeError(
                "distributed.world_size > 1 requires launch with torchrun; "
                "refusing to initialize a process group from one process"
            )

    distributed_job = world_size > 1
    if distributed_job and not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this build")

    if requested_device is None:
        if torch.cuda.is_available():
            selected_index = local_rank if distributed_job else 0
            device = _coerce_device(
                torch.device("cuda", selected_index),
                name="device",
            )
        else:
            device = torch.device("cpu")
    else:
        device = _coerce_device(requested_device, name="requested_device")

    backend = configured_backend
    if distributed_job and backend is None:
        backend = "nccl" if device.type == "cuda" else "gloo"
    if backend == "nccl":
        if device.type != "cuda":
            raise RuntimeError("the NCCL backend requires a CUDA device")
        if not dist.is_nccl_available():
            raise RuntimeError("the NCCL backend is unavailable")
    if distributed_job and device.type == "cuda":
        if device.index != local_rank:
            raise ValueError(
                "each torchrun process must use cuda:LOCAL_RANK; got "
                f"{device} for LOCAL_RANK={local_rank}"
            )
        torch.cuda.set_device(local_rank)

    initialized_here = False
    if dist.is_available() and dist.is_initialized():
        active_world = dist.get_world_size()
        active_rank = dist.get_rank()
        active_backend = str(dist.get_backend())
        if active_world != world_size or active_rank != rank:
            raise RuntimeError(
                "the existing process group does not match torchrun rank/world"
            )
        if backend is not None and active_backend != backend:
            raise RuntimeError(
                "the existing process group backend does not match the "
                f"configuration: {active_backend} != {backend}"
            )
        backend = active_backend
    elif distributed_job:
        if backend is None:
            raise RuntimeError("a distributed backend was not selected")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        initialized_here = True

    return DistributedContext(
        distributed=distributed_job,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
        initialized_here=initialized_here,
    )


def wrap_distributed_model(
    model: nn.Module,
    context: DistributedContext,
    options: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Move a model, optionally convert SyncBN, and wrap it with DDP."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(context, DistributedContext):
        raise TypeError("context must be DistributedContext")
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise TypeError("distributed options must be a mapping")
    if any(not isinstance(key, str) for key in options):
        raise TypeError("distributed option keys must be strings")

    broadcast_buffers = options.get("broadcast_buffers", False)
    _strict_bool("distributed.broadcast_buffers", broadcast_buffers)
    if broadcast_buffers:
        raise ValueError(
            "distributed.broadcast_buffers must be false for dynamic DCL buffers"
        )
    sync_batchnorm = options.get(
        "sync_batchnorm",
        options.get("sync_batch_norm", False),
    )
    _strict_bool("distributed.sync_batchnorm", sync_batchnorm)
    find_unused = options.get("find_unused_parameters", False)
    _strict_bool("distributed.find_unused_parameters", find_unused)

    prepared = model.to(context.device)
    if sync_batchnorm:
        if not context.distributed:
            raise ValueError("SyncBatchNorm requires a distributed process group")
        if context.device.type != "cuda":
            raise ValueError("SyncBatchNorm DDP requires CUDA")
        prepared = nn.SyncBatchNorm.convert_sync_batchnorm(prepared)
    if not context.distributed:
        return prepared
    if not dist.is_initialized():
        raise RuntimeError("cannot construct DDP before process-group initialization")

    if context.device.type == "cuda":
        return DistributedDataParallel(
            prepared,
            device_ids=[context.device.index],
            output_device=context.device.index,
            broadcast_buffers=False,
            find_unused_parameters=find_unused,
        )
    return DistributedDataParallel(
        prepared,
        device_ids=None,
        broadcast_buffers=False,
        find_unused_parameters=find_unused,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module or DistributedDataParallel")
    if isinstance(model, DistributedDataParallel):
        if not isinstance(model.module, nn.Module):
            raise TypeError("DistributedDataParallel.module must be an nn.Module")
        return model.module
    return model


def seed_everything(
    seed: int,
    *,
    deterministic: bool = False,
    cudnn_benchmark: bool = False,
) -> None:
    """Seed every process identically so DDP starts from identical weights."""

    normalized_seed = _strict_integer(
        "seed",
        seed,
        minimum=0,
        maximum=_MAX_SEED,
    )
    _strict_bool("deterministic", deterministic)
    _strict_bool("cudnn_benchmark", cudnn_benchmark)
    if deterministic and cudnn_benchmark:
        raise ValueError(
            "deterministic algorithms and cudnn_benchmark cannot both be enabled"
        )

    random.seed(normalized_seed)
    np.random.seed(normalized_seed % (2**32))
    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def move_batch_to_device(
    batch: Any,
    device: str | torch.device,
    *,
    non_blocking: bool = False,
) -> Any:
    """Recursively transfer tensor-bearing batches without touching metadata."""

    resolved = _coerce_device(device, name="batch device")
    _strict_bool("non_blocking", non_blocking)
    if isinstance(batch, Tensor):
        return batch.to(device=resolved, non_blocking=non_blocking)
    if isinstance(batch, BaseData):
        return batch.to(resolved, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        moved_items = [
            (
                key,
                move_batch_to_device(
                    value,
                    resolved,
                    non_blocking=non_blocking,
                ),
            )
            for key, value in batch.items()
        ]
        if isinstance(batch, defaultdict):
            result = type(batch)(batch.default_factory)
            result.update(moved_items)
            return result
        if type(batch) is dict:
            return dict(moved_items)
        try:
            return type(batch)(moved_items)
        except TypeError:
            return dict(moved_items)
    if isinstance(batch, list):
        return [
            move_batch_to_device(item, resolved, non_blocking=non_blocking)
            for item in batch
        ]
    if isinstance(batch, tuple):
        moved = tuple(
            move_batch_to_device(item, resolved, non_blocking=non_blocking)
            for item in batch
        )
        if hasattr(batch, "_fields"):
            return type(batch)(*moved)
        if type(batch) is tuple:
            return moved
        return type(batch)(moved)
    return batch


def is_distributed() -> bool:
    return (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def distributed_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def distributed_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def _collective_device() -> torch.device:
    if is_distributed() and str(dist.get_backend()) == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL collectives require CUDA")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _distributed_preflight(
    operation: str,
    validation: Callable[[], _T],
) -> _T:
    """Make ordinary local validation failures collective before payload I/O."""

    local_error: Exception | None = None
    result: Any = None
    try:
        result = validation()
    except Exception as exc:
        local_error = exc
    if not is_distributed():
        if local_error is not None:
            raise local_error
        return result

    error_flag = torch.tensor(
        int(local_error is not None),
        dtype=torch.int32,
        device=_collective_device(),
    )
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    if int(error_flag.item()) != 0:
        local_description = (
            None
            if local_error is None
            else (type(local_error).__name__, str(local_error))
        )
        descriptions: list[tuple[str, str] | None] = [
            None for _ in range(distributed_world_size())
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
            f"{operation} preflight failed on another rank; " + "; ".join(failures)
        )
    return result


def _validate_reduction_tensor(value: Tensor, *, mean: bool) -> None:
    if value.layout != torch.strided:
        raise TypeError("collective tensors must use strided layout")
    if value.dtype == torch.bool or value.is_complex():
        raise TypeError("collective tensors must have a real numeric dtype")
    if mean and not value.is_floating_point():
        raise TypeError("all_reduce_mean requires a floating-point tensor")
    expected = _collective_device()
    if is_distributed():
        if expected.type == "cuda" and value.device.type != "cuda":
            raise ValueError("NCCL collective tensors must be on CUDA")
        if expected.type == "cpu" and value.device.type != "cpu":
            raise ValueError("non-NCCL collective tensors must be on CPU")


def _assert_matching_metadata(metadata: tuple[Any, ...]) -> None:
    if not is_distributed():
        return
    gathered: list[tuple[Any, ...] | None] = [
        None for _ in range(distributed_world_size())
    ]
    dist.all_gather_object(gathered, metadata)
    if any(item != metadata for item in gathered):
        raise RuntimeError(
            "collective value kind, dtype, device type, and shape must match "
            "across ranks"
        )


def _prepare_reduction(
    value: Tensor | Real,
    *,
    mean: bool,
) -> tuple[Tensor, tuple[Any, ...], bool]:
    if isinstance(value, Tensor):
        _validate_reduction_tensor(value, mean=mean)
        reduced = value.detach().clone()
        metadata = (
            "all_reduce_mean" if mean else "all_reduce_sum",
            "tensor",
            str(value.dtype),
            value.device.type,
            tuple(value.shape),
        )
        return reduced, metadata, False
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError("collective scalars must be real numbers")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("collective scalars must be finite")
    reduced = torch.tensor(
        normalized,
        dtype=torch.float64,
        device=_collective_device(),
    )
    metadata = (
        "all_reduce_mean" if mean else "all_reduce_sum",
        "scalar",
        "float64",
        reduced.device.type,
        (),
    )
    return reduced, metadata, True


def _all_reduce(value: Tensor | Real, *, mean: bool) -> Tensor | float:
    operation = "all_reduce_mean" if mean else "all_reduce_sum"
    tensor, metadata, scalar_input = _distributed_preflight(
        operation,
        lambda: _prepare_reduction(value, mean=mean),
    )
    _assert_matching_metadata(metadata)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if mean:
            tensor.div_(distributed_world_size())
    if scalar_input:
        return float(tensor.item())
    return tensor


def all_reduce_sum(value: Tensor | Real) -> Tensor | float:
    return _all_reduce(value, mean=False)


def all_reduce_mean(value: Tensor | Real) -> Tensor | float:
    return _all_reduce(value, mean=True)


def _validate_source_rank(src: int) -> int:
    source = _strict_integer("src", src, minimum=0)
    if source >= distributed_world_size():
        raise ValueError("src must be smaller than the distributed world size")
    return source


def broadcast_float(value: Real, *, src: int = 0) -> float:
    def prepare() -> tuple[Tensor, int]:
        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
            raise TypeError("broadcast_float value must be a real number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("broadcast_float value must be finite")
        source_rank = _validate_source_rank(src)
        prepared = torch.tensor(
            normalized,
            dtype=torch.float64,
            device=_collective_device(),
        )
        return prepared, source_rank

    tensor, source = _distributed_preflight(
        "broadcast_float",
        prepare,
    )
    _assert_matching_metadata(
        ("broadcast_float", source, str(tensor.dtype), tensor.device.type, ())
    )
    if is_distributed():
        dist.broadcast(tensor, src=source)
    return float(tensor.item())


def broadcast_bool(value: bool, *, src: int = 0) -> bool:
    def prepare() -> tuple[Tensor, int]:
        _strict_bool("broadcast_bool value", value)
        source_rank = _validate_source_rank(src)
        prepared = torch.tensor(
            int(value),
            dtype=torch.uint8,
            device=_collective_device(),
        )
        return prepared, source_rank

    tensor, source = _distributed_preflight(
        "broadcast_bool",
        prepare,
    )
    _assert_matching_metadata(
        ("broadcast_bool", source, str(tensor.dtype), tensor.device.type, ())
    )
    if is_distributed():
        dist.broadcast(tensor, src=source)
    raw = int(tensor.item())
    if raw not in {0, 1}:
        raise RuntimeError("broadcast_bool received a value other than 0 or 1")
    return bool(raw)


def no_sync_context(
    model: nn.Module,
    synchronize: bool,
) -> AbstractContextManager[Any]:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    _strict_bool("synchronize", synchronize)
    if isinstance(model, DistributedDataParallel) and not synchronize:
        return model.no_sync()
    return nullcontext()


@dataclass(frozen=True)
class TrainerState:
    """Epoch-boundary resume cursor.

    ``micro_step`` is the current gradient-accumulation phase, not a global
    batch counter. Checkpoints are valid only when it is zero.
    """

    next_epoch: int = 0
    micro_step: int = 0
    optimizer_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = -1
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        for name in ("next_epoch", "micro_step", "optimizer_step", "bad_epochs"):
            value = getattr(self, name)
            _strict_integer(name, value, minimum=0)
        _strict_integer("best_epoch", self.best_epoch, minimum=-1)
        if self.best_epoch >= self.next_epoch and self.best_epoch != -1:
            raise ValueError("best_epoch must be earlier than next_epoch")
        if not isinstance(self.best_metric, Real) or isinstance(
            self.best_metric,
            (bool, np.bool_),
        ):
            raise TypeError("best_metric must be a real number")
        normalized_metric = float(self.best_metric)
        if not math.isfinite(normalized_metric):
            raise ValueError("best_metric must be finite")
        object.__setattr__(self, "best_metric", normalized_metric)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "next_epoch": self.next_epoch,
            "micro_step": self.micro_step,
            "optimizer_step": self.optimizer_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainerState":
        if not isinstance(payload, Mapping):
            raise TypeError("trainer state must be a mapping")
        expected = {
            "next_epoch",
            "micro_step",
            "optimizer_step",
            "best_metric",
            "best_epoch",
            "bad_epochs",
        }
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ValueError(
                f"trainer state keys do not match schema; missing={missing}, "
                f"extra={extra}"
            )
        return cls(**{key: payload[key] for key in expected})


def capture_dataloader_generator_state(loader: DataLoader) -> Tensor:
    if not isinstance(loader, DataLoader):
        raise TypeError("loader must be a torch DataLoader")
    generator = loader.generator
    if not isinstance(generator, torch.Generator):
        raise RuntimeError("loader must have an explicit torch.Generator")
    state = generator.get_state()
    if state.device.type != "cpu" or state.dtype != torch.uint8 or state.ndim != 1:
        raise RuntimeError("DataLoader generator returned an invalid RNG state")
    return state.clone()


def restore_dataloader_generator_state(
    loader: DataLoader,
    state: Tensor,
) -> None:
    if not isinstance(loader, DataLoader):
        raise TypeError("loader must be a torch DataLoader")
    if not isinstance(state, Tensor):
        raise TypeError("DataLoader generator state must be a tensor")
    if state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError("DataLoader generator state must be a 1D uint8 tensor")
    generator = loader.generator
    if not isinstance(generator, torch.Generator):
        raise RuntimeError("loader must have an explicit torch.Generator")
    generator.set_state(state.detach().cpu().contiguous())


def _validate_loaders(loaders: Mapping[str, DataLoader]) -> None:
    if not isinstance(loaders, Mapping):
        raise TypeError("loaders must be a name-to-DataLoader mapping")
    for name, loader in loaders.items():
        if not isinstance(name, str) or not name:
            raise ValueError("loader names must be non-empty strings")
        if not isinstance(loader, DataLoader):
            raise TypeError(f"loaders[{name!r}] must be a DataLoader")


def capture_dataloader_generator_states(
    loaders: Mapping[str, DataLoader],
) -> dict[str, Tensor]:
    _validate_loaders(loaders)
    return {
        name: capture_dataloader_generator_state(loader)
        for name, loader in loaders.items()
    }


def restore_dataloader_generator_states(
    loaders: Mapping[str, DataLoader],
    states: Mapping[str, Tensor],
) -> None:
    _validate_loaders(loaders)
    if not isinstance(states, Mapping):
        raise TypeError("loader generator states must be a mapping")
    if any(not isinstance(name, str) for name in states):
        raise TypeError("loader generator state names must be strings")
    if set(loaders) != set(states):
        missing = sorted(set(loaders) - set(states))
        extra = sorted(set(states) - set(loaders))
        raise ValueError(
            f"loader generator state names differ; missing={missing}, "
            f"extra={extra}"
        )
    for name, loader in loaders.items():
        restore_dataloader_generator_state(loader, states[name])


_SAMPLER_STATE_KEYS = {
    "type",
    "stateful",
    "state_dict",
    "distributed",
}
_DISTRIBUTED_SAMPLER_KEYS = {
    "epoch",
    "seed",
    "shuffle",
    "drop_last",
    "num_replicas",
    "rank",
}
_COLLATOR_STATE_KEYS = {
    "type",
    "epoch_visible",
    "set_epoch_available",
    "epoch",
}
_DATALOADER_STATE_KEYS = {"generator", "sampler", "collator"}


def _object_identity(value: object) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _distributed_sampler_state(
    sampler: DistributedSampler,
) -> dict[str, int | bool]:
    if not isinstance(sampler, DistributedSampler):
        raise TypeError("sampler must be DistributedSampler")
    epoch = _strict_integer("DistributedSampler.epoch", sampler.epoch, minimum=0)
    seed = _strict_integer("DistributedSampler.seed", sampler.seed, minimum=0)
    num_replicas = _strict_integer(
        "DistributedSampler.num_replicas",
        sampler.num_replicas,
        minimum=1,
    )
    rank = _strict_integer("DistributedSampler.rank", sampler.rank, minimum=0)
    if rank >= num_replicas:
        raise ValueError("DistributedSampler.rank must be below num_replicas")
    shuffle = _strict_bool("DistributedSampler.shuffle", sampler.shuffle)
    drop_last = _strict_bool("DistributedSampler.drop_last", sampler.drop_last)
    return {
        "epoch": epoch,
        "seed": seed,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_replicas": num_replicas,
        "rank": rank,
    }


def _capture_sampler_state(loader: DataLoader) -> dict[str, Any]:
    sampler = loader.sampler
    state_dict_method = getattr(sampler, "state_dict", None)
    load_state_dict_method = getattr(sampler, "load_state_dict", None)
    has_state_dict = callable(state_dict_method)
    has_load_state_dict = callable(load_state_dict_method)
    if has_state_dict != has_load_state_dict:
        raise TypeError(
            "sampler must provide both state_dict and load_state_dict, or neither"
        )
    serialized_state: dict[str, Any] | None = None
    if has_state_dict:
        raw_state = state_dict_method()
        if not isinstance(raw_state, Mapping):
            raise TypeError("sampler.state_dict() must return a mapping")
        if any(not isinstance(key, str) for key in raw_state):
            raise TypeError("sampler state_dict keys must be strings")
        serialized_state = dict(raw_state)
    distributed_state = (
        _distributed_sampler_state(sampler)
        if isinstance(sampler, DistributedSampler)
        else None
    )
    return {
        "type": _object_identity(type(sampler)),
        "stateful": has_state_dict,
        "state_dict": serialized_state,
        "distributed": distributed_state,
    }


def _capture_collator_state(loader: DataLoader) -> dict[str, Any]:
    collator = loader.collate_fn
    epoch_visible = hasattr(collator, "epoch")
    set_epoch = getattr(collator, "set_epoch", None)
    set_epoch_available = callable(set_epoch)
    epoch: int | None = None
    if epoch_visible:
        if not set_epoch_available:
            raise TypeError(
                "collator exposes epoch but not a callable set_epoch for restore"
            )
        epoch = _strict_integer("collator.epoch", collator.epoch, minimum=0)
    return {
        "type": _object_identity(collator),
        "epoch_visible": epoch_visible,
        "set_epoch_available": set_epoch_available,
        "epoch": epoch,
    }


def capture_dataloader_state(loader: DataLoader) -> dict[str, Any]:
    """Capture reproducible loader state visible to the main process.

    This covers the explicit generator, sampler state, and a collator's visible
    epoch. It deliberately makes no claim about opaque third-party worker RNGs;
    SemMol worker randomness is expected to derive from sample identity and
    epoch.
    """

    if not isinstance(loader, DataLoader):
        raise TypeError("loader must be a torch DataLoader")
    return {
        "generator": capture_dataloader_generator_state(loader),
        "sampler": _capture_sampler_state(loader),
        "collator": _capture_collator_state(loader),
    }


def _validate_sampler_state_compatibility(
    loader: DataLoader,
    state: Mapping[str, Any],
) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("sampler checkpoint state must be a mapping")
    if set(state) != _SAMPLER_STATE_KEYS:
        raise ValueError("sampler checkpoint state does not match its schema")
    sampler = loader.sampler
    expected_type = _object_identity(type(sampler))
    if state["type"] != expected_type:
        raise ValueError(
            f"sampler type changed: {state['type']!r} != {expected_type!r}"
        )
    stateful = _strict_bool("sampler stateful", state["stateful"])
    state_dict_method = getattr(sampler, "state_dict", None)
    load_state_dict_method = getattr(sampler, "load_state_dict", None)
    current_stateful = callable(state_dict_method) and callable(
        load_state_dict_method
    )
    if callable(state_dict_method) != callable(load_state_dict_method):
        raise TypeError(
            "sampler must provide both state_dict and load_state_dict, or neither"
        )
    if stateful != current_stateful:
        raise ValueError("sampler state_dict/load_state_dict presence changed")
    serialized_state = state["state_dict"]
    if stateful:
        if not isinstance(serialized_state, Mapping):
            raise TypeError("saved sampler state_dict must be a mapping")
        if any(not isinstance(key, str) for key in serialized_state):
            raise TypeError("saved sampler state_dict keys must be strings")
    elif serialized_state is not None:
        raise ValueError("stateless sampler must have a null saved state_dict")

    distributed_state = state["distributed"]
    current_distributed = isinstance(sampler, DistributedSampler)
    if (distributed_state is not None) != current_distributed:
        raise ValueError("DistributedSampler presence changed since checkpoint")
    if current_distributed:
        if not isinstance(distributed_state, Mapping):
            raise TypeError("DistributedSampler state must be a mapping")
        if set(distributed_state) != _DISTRIBUTED_SAMPLER_KEYS:
            raise ValueError("DistributedSampler state does not match its schema")
        current = _distributed_sampler_state(sampler)
        for field in (
            "seed",
            "shuffle",
            "drop_last",
            "num_replicas",
            "rank",
        ):
            if distributed_state[field] != current[field]:
                raise ValueError(
                    f"DistributedSampler.{field} changed: "
                    f"{distributed_state[field]!r} != {current[field]!r}"
                )

    if current_distributed:
        _strict_integer(
            "saved DistributedSampler.epoch",
            distributed_state["epoch"],
            minimum=0,
        )


def _restore_sampler_state(loader: DataLoader, state: Mapping[str, Any]) -> None:
    _validate_sampler_state_compatibility(loader, state)
    sampler = loader.sampler
    if state["stateful"]:
        sampler.load_state_dict(dict(state["state_dict"]))
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(int(state["distributed"]["epoch"]))


def _validate_collator_state_compatibility(
    loader: DataLoader,
    state: Mapping[str, Any],
) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("collator checkpoint state must be a mapping")
    if set(state) != _COLLATOR_STATE_KEYS:
        raise ValueError("collator checkpoint state does not match its schema")
    collator = loader.collate_fn
    expected_type = _object_identity(collator)
    if state["type"] != expected_type:
        raise ValueError(
            f"collator type changed: {state['type']!r} != {expected_type!r}"
        )
    epoch_visible = _strict_bool(
        "collator epoch_visible",
        state["epoch_visible"],
    )
    set_epoch_available = _strict_bool(
        "collator set_epoch_available",
        state["set_epoch_available"],
    )
    current_epoch_visible = hasattr(collator, "epoch")
    current_set_epoch = getattr(collator, "set_epoch", None)
    current_set_epoch_available = callable(current_set_epoch)
    if epoch_visible != current_epoch_visible:
        raise ValueError("collator epoch visibility changed since checkpoint")
    if set_epoch_available != current_set_epoch_available:
        raise ValueError("collator set_epoch availability changed since checkpoint")
    saved_epoch = state["epoch"]
    if epoch_visible:
        _strict_integer("saved collator.epoch", saved_epoch, minimum=0)
    elif saved_epoch is not None:
        raise ValueError("collator without visible epoch must save a null epoch")


def _restore_collator_state(loader: DataLoader, state: Mapping[str, Any]) -> None:
    _validate_collator_state_compatibility(loader, state)
    if state["epoch_visible"]:
        loader.collate_fn.set_epoch(int(state["epoch"]))


def validate_dataloader_state(
    loader: DataLoader,
    state: Mapping[str, Any],
) -> None:
    """Validate loader checkpoint compatibility without changing the loader."""

    if not isinstance(loader, DataLoader):
        raise TypeError("loader must be a torch DataLoader")
    if not isinstance(state, Mapping):
        raise TypeError("DataLoader checkpoint state must be a mapping")
    if set(state) != _DATALOADER_STATE_KEYS:
        raise ValueError("DataLoader checkpoint state does not match its schema")
    generator_state = state["generator"]
    if not isinstance(generator_state, Tensor):
        raise TypeError("saved DataLoader generator state must be a tensor")
    if generator_state.dtype != torch.uint8 or generator_state.ndim != 1:
        raise ValueError("saved generator state must be a 1D uint8 tensor")
    if not isinstance(loader.generator, torch.Generator):
        raise RuntimeError("loader must have an explicit torch.Generator")
    _validate_sampler_state_compatibility(loader, state["sampler"])
    _validate_collator_state_compatibility(loader, state["collator"])


def restore_dataloader_state(
    loader: DataLoader,
    state: Mapping[str, Any],
) -> None:
    validate_dataloader_state(loader, state)
    generator_state = state["generator"]
    restore_dataloader_generator_state(loader, generator_state)
    _restore_sampler_state(loader, state["sampler"])
    _restore_collator_state(loader, state["collator"])


def capture_dataloader_states(
    loaders: Mapping[str, DataLoader],
) -> dict[str, dict[str, Any]]:
    _validate_loaders(loaders)
    return {name: capture_dataloader_state(loader) for name, loader in loaders.items()}


def restore_dataloader_states(
    loaders: Mapping[str, DataLoader],
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    _validate_loaders(loaders)
    if not isinstance(states, Mapping):
        raise TypeError("DataLoader states must be a mapping")
    if any(not isinstance(name, str) for name in states):
        raise TypeError("DataLoader state names must be strings")
    if set(loaders) != set(states):
        missing = sorted(set(loaders) - set(states))
        extra = sorted(set(states) - set(loaders))
        raise ValueError(
            f"DataLoader state names differ; missing={missing}, extra={extra}"
        )
    for name, loader in loaders.items():
        restore_dataloader_state(loader, states[name])


def validate_dataloader_states(
    loaders: Mapping[str, DataLoader],
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    _validate_loaders(loaders)
    if not isinstance(states, Mapping):
        raise TypeError("DataLoader states must be a mapping")
    if any(not isinstance(name, str) for name in states):
        raise TypeError("DataLoader state names must be strings")
    if set(loaders) != set(states):
        missing = sorted(set(loaders) - set(states))
        extra = sorted(set(states) - set(loaders))
        raise ValueError(
            f"DataLoader state names differ; missing={missing}, extra={extra}"
        )
    for name, loader in loaders.items():
        validate_dataloader_state(loader, states[name])


__all__ = [
    "DistributedContext",
    "PrecisionMode",
    "TrainerState",
    "all_reduce_mean",
    "all_reduce_sum",
    "barrier",
    "broadcast_bool",
    "broadcast_float",
    "capture_dataloader_state",
    "capture_dataloader_states",
    "capture_dataloader_generator_state",
    "capture_dataloader_generator_states",
    "distributed_rank",
    "distributed_world_size",
    "initialize_distributed",
    "is_distributed",
    "move_batch_to_device",
    "no_sync_context",
    "restore_dataloader_state",
    "restore_dataloader_states",
    "restore_dataloader_generator_state",
    "restore_dataloader_generator_states",
    "seed_everything",
    "unwrap_model",
    "validate_dataloader_state",
    "validate_dataloader_states",
    "wrap_distributed_model",
]

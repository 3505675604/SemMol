"""Strict resume checkpoints and intentionally separate pretrained transfer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import random
import re
import secrets
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.models.semantic.dcl import DynamicCentralLibrary

from .common import (
    DistributedContext,
    TrainerState,
    capture_dataloader_states,
    restore_dataloader_states,
    unwrap_model,
    validate_dataloader_states,
)


_CHECKPOINT_VERSION = 2
_CHECKPOINT_KIND = "semmol_training_checkpoint"
_CHECKPOINT_KEYS = {
    "version",
    "kind",
    "checkpoint_id",
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "trainer_state",
    "config_fingerprint",
    "world_size",
    "per_rank_runtime",
    "extra",
}
_RUNTIME_KEYS = {"rank", "rng", "loaders"}
_RNG_KEYS = {"python", "numpy", "torch_cpu", "torch_cuda"}
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
_T = TypeVar("_T")
TrainingCheckpointMetadataValidator = Callable[
    [Mapping[str, Any], TrainerState],
    None,
]


@dataclass(frozen=True)
class TrainingCheckpointLoadResult:
    """Resume-only result, kept distinct from weight-transfer results."""

    state: TrainerState
    extra: dict[str, Any]
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.state, TrainerState):
            raise TypeError("state must be TrainerState")
        if not isinstance(self.extra, dict):
            raise TypeError("extra must be a dictionary")
        if any(not isinstance(key, str) for key in self.extra):
            raise TypeError("extra keys must be strings")
        if not isinstance(self.path, Path):
            raise TypeError("path must be pathlib.Path")


@dataclass(frozen=True)
class PretrainedTransferResult:
    """Weight-transfer result that cannot be mistaken for resumed state."""

    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    path: Path

    def __post_init__(self) -> None:
        for name, keys in (
            ("missing_keys", self.missing_keys),
            ("unexpected_keys", self.unexpected_keys),
        ):
            if not isinstance(keys, tuple) or any(
                not isinstance(key, str) for key in keys
            ):
                raise TypeError(f"{name} must be a tuple of strings")
        if not isinstance(self.path, Path):
            raise TypeError("path must be pathlib.Path")


def _normalize_configuration(value: Any, *, location: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{location} must not contain NaN or infinity")
        return normalized
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{location} contains a non-string mapping key: {key!r}"
                )
            normalized_mapping[key] = _normalize_configuration(
                item,
                location=f"{location}.{key}",
            )
        return normalized_mapping
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _normalize_configuration(
                item,
                location=f"{location}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{location} contains unsupported value type {type(value).__name__}"
    )


def configuration_fingerprint(configuration: Mapping[str, Any]) -> str:
    """Hash a canonical, finite JSON representation of a configuration."""

    if not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a mapping")
    normalized = _normalize_configuration(configuration, location="configuration")
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_fingerprint(value: object) -> str:
    if not isinstance(value, str) or _FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise ValueError("config_fingerprint must be a lowercase SHA-256 hex digest")
    return value


def _validate_checkpoint_id(value: object) -> str:
    if not isinstance(value, str) or _CHECKPOINT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("checkpoint_id must be a lowercase 256-bit hex identifier")
    return value


def capture_rng_state() -> dict[str, Any]:
    """Capture process-local Python, NumPy, CPU, and visible CUDA RNGs."""

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [
            state.clone() for state in torch.cuda.get_rng_state_all()
        ]
        if torch.cuda.is_available()
        else [],
    }


def _cpu_byte_rng_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError(f"{name} must be a 1D uint8 tensor")
    return value.detach().cpu().contiguous()


def _validated_rng_state(
    state: Mapping[str, Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...], Tensor, list[Tensor]]:
    if not isinstance(state, Mapping):
        raise TypeError("RNG state must be a mapping")
    if set(state) != _RNG_KEYS:
        missing = sorted(_RNG_KEYS - set(state))
        extra = sorted(set(state) - _RNG_KEYS)
        raise ValueError(
            f"RNG state keys do not match schema; missing={missing}, extra={extra}"
        )

    python_state = state["python"]
    if not isinstance(python_state, tuple):
        raise TypeError("Python RNG state must be a tuple")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise TypeError("NumPy RNG state must be a five-item tuple")
    if not isinstance(numpy_state[0], str):
        raise TypeError("NumPy RNG algorithm name must be a string")
    if not isinstance(numpy_state[1], np.ndarray):
        raise TypeError("NumPy RNG key state must be an ndarray")
    if not isinstance(numpy_state[2], Integral) or isinstance(
        numpy_state[2],
        (bool, np.bool_),
    ):
        raise TypeError("NumPy RNG position must be an integer")
    if not isinstance(numpy_state[3], Integral) or isinstance(
        numpy_state[3],
        (bool, np.bool_),
    ):
        raise TypeError("NumPy RNG Gaussian flag must be an integer")
    if not isinstance(numpy_state[4], Real) or isinstance(
        numpy_state[4],
        (bool, np.bool_),
    ):
        raise TypeError("NumPy RNG cached Gaussian must be real")

    cpu_state = _cpu_byte_rng_tensor("torch CPU RNG state", state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise TypeError("torch CUDA RNG states must be a list")
    expected_cuda_states = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states) != expected_cuda_states:
        raise RuntimeError(
            "checkpoint CUDA RNG state count does not match visible devices: "
            f"{len(cuda_states)} != {expected_cuda_states}"
        )
    prepared_cuda_states = [
        _cpu_byte_rng_tensor(f"torch CUDA RNG state {index}", item)
        for index, item in enumerate(cuda_states)
    ]
    normalized_numpy_state = (
        numpy_state[0],
        numpy_state[1],
        int(numpy_state[2]),
        int(numpy_state[3]),
        float(numpy_state[4]),
    )
    python_probe = random.Random()
    python_probe.setstate(python_state)
    numpy_probe = np.random.RandomState()
    numpy_probe.set_state(normalized_numpy_state)
    return (
        python_state,
        normalized_numpy_state,
        cpu_state,
        prepared_cuda_states,
    )


def validate_rng_state(state: Mapping[str, Any]) -> None:
    """Validate an RNG snapshot without changing process-global RNG state."""

    _validated_rng_state(state)


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a process-local RNG snapshot with an exact schema."""

    (
        python_state,
        numpy_state,
        cpu_state,
        prepared_cuda_states,
    ) = _validated_rng_state(state)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_state)
    if prepared_cuda_states:
        torch.cuda.set_rng_state_all(prepared_cuda_states)


def _checkpoint_path(path: str | Path, *, must_exist: bool) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("checkpoint path must be a string or pathlib.Path")
    if isinstance(path, str) and not path.strip():
        raise ValueError("checkpoint path cannot be empty")
    resolved = Path(path).expanduser().resolve()
    if must_exist:
        if not resolved.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
    elif not resolved.parent.is_dir():
        raise FileNotFoundError(
            f"checkpoint parent directory does not exist: {resolved.parent}"
        )
    return resolved


def _validate_model_and_optimizer(model: nn.Module, optimizer: Optimizer) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")


def _validate_stateful(name: str, value: object | None) -> None:
    if value is None:
        return
    if not callable(getattr(value, "state_dict", None)) or not callable(
        getattr(value, "load_state_dict", None)
    ):
        raise TypeError(f"{name} must provide state_dict/load_state_dict or be None")


def _validate_loaders(
    loaders: Mapping[str, DataLoader] | None,
) -> dict[str, DataLoader]:
    if loaders is None:
        return {}
    if not isinstance(loaders, Mapping):
        raise TypeError("loaders must be a name-to-DataLoader mapping or None")
    normalized: dict[str, DataLoader] = {}
    for name, loader in loaders.items():
        if not isinstance(name, str) or not name:
            raise ValueError("loader names must be non-empty strings")
        if not isinstance(loader, DataLoader):
            raise TypeError(f"loaders[{name!r}] must be a DataLoader")
        normalized[name] = loader
    return normalized


def _validate_extra(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    if extra is None:
        return {}
    if not isinstance(extra, Mapping):
        raise TypeError("extra must be a mapping or None")
    if any(not isinstance(key, str) for key in extra):
        raise TypeError("extra keys must be strings")
    return dict(extra)


def _active_rank_world() -> tuple[int, int]:
    active = dist.is_available() and dist.is_initialized()
    active_world = dist.get_world_size() if active else 1
    active_rank = dist.get_rank() if active else 0
    return active_rank, active_world


def _validate_context(
    context: DistributedContext | None,
    *,
    active_rank: int,
    active_world: int,
) -> None:
    if context is None:
        return
    if not isinstance(context, DistributedContext):
        raise TypeError("context must be DistributedContext or None")
    if context.rank != active_rank or context.world_size != active_world:
        raise RuntimeError("DistributedContext does not match the active process group")
    if context.distributed != (active_world > 1):
        raise RuntimeError("DistributedContext distributed flag is inconsistent")


def _rank_world(
    context: DistributedContext | None,
) -> tuple[int, int]:
    active_rank, active_world = _active_rank_world()
    _validate_context(
        context,
        active_rank=active_rank,
        active_world=active_world,
    )
    return active_rank, active_world


def _checkpoint_collective_device(world_size: int) -> torch.device:
    if world_size > 1 and str(dist.get_backend()) == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL checkpoint coordination requires CUDA")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _synchronize_checkpoint_preflight(
    operation: str,
    *,
    rank: int,
    world_size: int,
    validation: Callable[[], _T],
    failure_warning: str | None = None,
) -> _T:
    """Synchronize ordinary validation errors before barrier/gather calls."""

    local_error: Exception | None = None
    result: Any = None
    try:
        result = validation()
    except Exception as exc:
        local_error = exc
    if world_size == 1:
        if local_error is not None:
            if failure_warning is not None:
                raise RuntimeError(
                    f"{operation} failed: {failure_warning}"
                ) from local_error
            raise local_error
        return result

    error_flag = torch.tensor(
        int(local_error is not None),
        dtype=torch.int32,
        device=_checkpoint_collective_device(world_size),
    )
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    if int(error_flag.item()) != 0:
        local_description = (
            None
            if local_error is None
            else (type(local_error).__name__, str(local_error))
        )
        descriptions: list[tuple[str, str] | None] = [None] * world_size
        dist.all_gather_object(descriptions, local_description)
        if local_error is not None:
            if failure_warning is not None:
                raise RuntimeError(
                    f"{operation} failed on rank {rank}: {failure_warning}"
                ) from local_error
            raise local_error
        failures = [
            f"rank {failed_rank}: {description[0]}: {description[1]}"
            for failed_rank, description in enumerate(descriptions)
            if description is not None
        ]
        message = f"{operation} failed on another rank; " + "; ".join(failures)
        if failure_warning is not None:
            message += f"; {failure_warning}"
        raise RuntimeError(message)
    return result


def _type_identity(value: object | None) -> str | None:
    if value is None:
        return None
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _loader_control_signature(
    loader_states: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, ...]:
    controls: list[Any] = []
    for name in sorted(loader_states):
        state = loader_states[name]
        sampler = state["sampler"]
        collator = state["collator"]
        sampler_state_dict = sampler["state_dict"]
        state_dict_keys = (
            tuple(sorted(sampler_state_dict))
            if isinstance(sampler_state_dict, Mapping)
            else ()
        )
        distributed_state = sampler["distributed"]
        distributed_control = (
            tuple(
                sorted(
                    (key, value)
                    for key, value in distributed_state.items()
                    if key != "rank"
                )
            )
            if isinstance(distributed_state, Mapping)
            else None
        )
        controls.append(
            (
                name,
                sampler["type"],
                sampler["stateful"],
                state_dict_keys,
                distributed_control,
                tuple(sorted(collator.items())),
            )
        )
    return tuple(controls)


def _require_matching_control_signature(
    signature: Mapping[str, Any],
    *,
    world_size: int,
    operation: str,
) -> None:
    if world_size == 1:
        return
    signatures: list[Mapping[str, Any] | None] = [None] * world_size
    dist.all_gather_object(signatures, dict(signature))
    authoritative = signatures[0]
    if any(candidate != authoritative for candidate in signatures):
        raise RuntimeError(
            f"{operation} signature differs across ranks; refusing to continue"
        )


def _validate_epoch_boundary(
    state: TrainerState,
    *,
    model: nn.Module | None,
) -> None:
    if state.micro_step != 0:
        raise ValueError(
            "training checkpoints are allowed only at an epoch boundary with "
            "TrainerState.micro_step == 0"
        )
    if model is None:
        return
    pending_gradients = [
        name
        for name, parameter in unwrap_model(model).named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if pending_gradients:
        preview = pending_gradients[:8]
        suffix = "..." if len(pending_gradients) > len(preview) else ""
        raise RuntimeError(
            "checkpoint save requires zero_grad(set_to_none=True); trainable "
            f"parameters still have gradients: {preview}{suffix}"
        )


def _barrier(context: DistributedContext | None) -> None:
    _, world_size = _rank_world(context)
    if world_size > 1:
        dist.barrier()


def _capture_rank_runtime(
    rank: int,
    loaders: Mapping[str, DataLoader],
) -> dict[str, Any]:
    return {
        "rank": rank,
        "rng": capture_rng_state(),
        "loaders": capture_dataloader_states(loaders),
    }


def _validate_runtime_entry(
    entry: object,
    *,
    expected_rank: int,
) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise TypeError(f"per_rank_runtime[{expected_rank}] must be a mapping")
    if set(entry) != _RUNTIME_KEYS:
        missing = sorted(_RUNTIME_KEYS - set(entry))
        extra = sorted(set(entry) - _RUNTIME_KEYS)
        raise ValueError(
            f"per_rank_runtime[{expected_rank}] schema differs; "
            f"missing={missing}, extra={extra}"
        )
    rank = entry["rank"]
    if not isinstance(rank, Integral) or isinstance(rank, (bool, np.bool_)):
        raise TypeError("runtime rank must be an integer")
    if int(rank) != expected_rank:
        raise ValueError(
            f"runtime rank/index mismatch: {int(rank)} != {expected_rank}"
        )
    if not isinstance(entry["rng"], Mapping):
        raise TypeError("runtime RNG state must be a mapping")
    if not isinstance(entry["loaders"], Mapping):
        raise TypeError("runtime DataLoader state must be a mapping")
    return entry


def _remove_temporary_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        _remove_temporary_file(temporary_path)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: object | None,
    scaler: torch.cuda.amp.GradScaler | None,
    state: TrainerState,
    config_fingerprint: str,
    loaders: Mapping[str, DataLoader] | None = None,
    extra: Mapping[str, Any] | None = None,
    context: DistributedContext | None = None,
) -> None:
    """Atomically save a synchronized, accumulation-free epoch boundary.

    Every rank must call this function in the same order. The caller must first
    finish the optimizer step and call ``zero_grad(set_to_none=True)``.
    """

    rank, world_size = _active_rank_world()

    def prepare() -> tuple[
        Path,
        str,
        dict[str, DataLoader],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        _validate_context(
            context,
            active_rank=rank,
            active_world=world_size,
        )
        prepared_path = _checkpoint_path(path, must_exist=False)
        _validate_model_and_optimizer(model, optimizer)
        _validate_stateful("scheduler", scheduler)
        _validate_stateful("scaler", scaler)
        if not isinstance(state, TrainerState):
            raise TypeError("state must be TrainerState")
        _validate_epoch_boundary(state, model=model)
        prepared_fingerprint = _validate_fingerprint(config_fingerprint)
        prepared_loaders = _validate_loaders(loaders)
        prepared_extra = _validate_extra(extra)
        extra_fingerprint = configuration_fingerprint(prepared_extra)
        runtime = _capture_rank_runtime(rank, prepared_loaders)
        control_signature = {
            "path": str(prepared_path),
            "trainer_state": state.as_dict(),
            "config_fingerprint": prepared_fingerprint,
            "scheduler_present": scheduler is not None,
            "scaler_present": scaler is not None,
            "loader_names": tuple(sorted(prepared_loaders)),
            "loader_controls": _loader_control_signature(runtime["loaders"]),
            "extra_keys": tuple(sorted(prepared_extra)),
            "extra_fingerprint": extra_fingerprint,
            "model_type": _type_identity(unwrap_model(model)),
            "optimizer_type": _type_identity(optimizer),
            "scheduler_type": _type_identity(scheduler),
            "scaler_type": _type_identity(scaler),
        }
        pickle.dumps(runtime, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dumps(control_signature, protocol=pickle.HIGHEST_PROTOCOL)
        return (
            prepared_path,
            prepared_fingerprint,
            prepared_loaders,
            prepared_extra,
            runtime,
            control_signature,
        )

    (
        checkpoint_path,
        fingerprint,
        normalized_loaders,
        normalized_extra,
        local_runtime,
        control_signature,
    ) = _synchronize_checkpoint_preflight(
        "save_training_checkpoint",
        rank=rank,
        world_size=world_size,
        validation=prepare,
    )
    _require_matching_control_signature(
        control_signature,
        world_size=world_size,
        operation="save_training_checkpoint control",
    )

    _barrier(context)
    if world_size > 1:
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(
            gathered,
            local_runtime,
        )
        per_rank_runtime = gathered
    else:
        per_rank_runtime = [local_runtime]

    save_error: Exception | None = None
    error_description: tuple[str, str] | None = None
    if rank == 0:
        try:
            if not isinstance(per_rank_runtime, list) or len(
                per_rank_runtime
            ) != world_size:
                raise RuntimeError(
                    "rank-zero runtime gathering produced an invalid list"
                )
            for expected_rank, runtime in enumerate(per_rank_runtime):
                _validate_runtime_entry(runtime, expected_rank=expected_rank)
            checkpoint_id = _validate_checkpoint_id(secrets.token_hex(32))
            payload = {
                "version": _CHECKPOINT_VERSION,
                "kind": _CHECKPOINT_KIND,
                "checkpoint_id": checkpoint_id,
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (
                    None if scheduler is None else scheduler.state_dict()
                ),
                "scaler": None if scaler is None else scaler.state_dict(),
                "trainer_state": state.as_dict(),
                "config_fingerprint": fingerprint,
                "world_size": world_size,
                "per_rank_runtime": per_rank_runtime,
                "extra": normalized_extra,
            }
            _atomic_torch_save(payload, checkpoint_path)
        except Exception as exc:
            # The broad catch exists only to coordinate rank-zero failure. The
            # original exception is re-raised on rank zero after all ranks have
            # received the failure status and completed the final barrier.
            save_error = exc
            error_description = (type(exc).__name__, str(exc))

    if world_size > 1:
        status: list[tuple[str, str] | None] = [error_description]
        dist.broadcast_object_list(status, src=0)
        error_description = status[0]
    _barrier(context)
    if save_error is not None:
        raise save_error
    if error_description is not None:
        error_type, message = error_description
        raise RuntimeError(
            f"rank zero failed to save checkpoint ({error_type}): {message}"
        )


def _validate_training_payload(
    payload: object,
    *,
    expected_fingerprint: str,
    expected_world_size: int,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("training checkpoint must contain a mapping")
    if set(payload) != _CHECKPOINT_KEYS:
        missing = sorted(_CHECKPOINT_KEYS - set(payload))
        extra = sorted(set(payload) - _CHECKPOINT_KEYS)
        raise ValueError(
            f"checkpoint schema differs; missing={missing}, extra={extra}"
        )
    version = payload["version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise TypeError("checkpoint version must be an integer")
    if version != _CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {version}; "
            f"expected {_CHECKPOINT_VERSION}"
        )
    if payload["kind"] != _CHECKPOINT_KIND:
        raise ValueError("checkpoint kind is not a SemMol training resume payload")
    _validate_checkpoint_id(payload["checkpoint_id"])
    if _validate_fingerprint(payload["config_fingerprint"]) != expected_fingerprint:
        raise ValueError("checkpoint configuration fingerprint does not match")
    saved_world = payload["world_size"]
    if not isinstance(saved_world, int) or isinstance(saved_world, bool):
        raise TypeError("checkpoint world_size must be an integer")
    if saved_world != expected_world_size:
        raise ValueError(
            "checkpoint world_size does not match the current job: "
            f"{saved_world} != {expected_world_size}"
        )
    for name in ("model", "optimizer", "trainer_state", "extra"):
        if not isinstance(payload[name], Mapping):
            raise TypeError(f"checkpoint {name} must be a mapping")
    if any(not isinstance(key, str) for key in payload["model"]):
        raise TypeError("checkpoint model-state keys must be strings")
    if any(not isinstance(key, str) for key in payload["extra"]):
        raise TypeError("checkpoint extra keys must be strings")
    for name in ("scheduler", "scaler"):
        if payload[name] is not None and not isinstance(payload[name], Mapping):
            raise TypeError(f"checkpoint {name} must be a mapping or None")
    runtime = payload["per_rank_runtime"]
    if not isinstance(runtime, list) or len(runtime) != expected_world_size:
        raise ValueError(
            "per_rank_runtime must be a list with one entry per current rank"
        )
    for expected_rank, entry in enumerate(runtime):
        _validate_runtime_entry(entry, expected_rank=expected_rank)
    return payload


def _map_location_device(
    map_location: str | torch.device | None,
    context: DistributedContext | None,
) -> torch.device:
    if map_location is None:
        return context.device if context is not None else torch.device("cpu")
    if not isinstance(map_location, (str, torch.device)):
        raise TypeError("map_location must be a device string, torch.device, or None")
    if isinstance(map_location, str) and not map_location.strip():
        raise ValueError("map_location cannot be empty")
    device = torch.device(map_location)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("map_location must select a CPU or CUDA device")
    if device.type == "cpu" and device.index is not None:
        raise ValueError("map_location cannot index a CPU device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("map_location requests CUDA, but CUDA is unavailable")
    if device.type == "cuda" and device.index is None:
        index = (
            context.device.index
            if context is not None and context.device.type == "cuda"
            else torch.cuda.current_device()
        )
        device = torch.device("cuda", index)
    if context is not None and device != context.device:
        raise ValueError("explicit map_location must match context.device")
    return device


def _load_payload_and_content_digest(
    path: Path,
    *,
    map_location: torch.device,
) -> tuple[object, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        handle.seek(0)
        payload = torch.load(
            handle,
            map_location=map_location,
            weights_only=False,
        )
    return payload, digest.hexdigest()


def _validate_model_state_compatibility(
    model: nn.Module,
    saved_state: Mapping[str, Any],
) -> None:
    target = unwrap_model(model)
    current_state = target.state_dict()
    if set(saved_state) != set(current_state):
        missing = sorted(set(current_state) - set(saved_state))
        extra = sorted(set(saved_state) - set(current_state))
        raise ValueError(
            f"model state keys differ; missing={missing}, extra={extra}"
        )
    dynamic_samples: dict[str, DynamicCentralLibrary] = {}
    for name, module in target.named_modules():
        if isinstance(module, DynamicCentralLibrary):
            prefix = f"{name}." if name else ""
            dynamic_samples[prefix + "initialization_samples"] = module
    for key, current_value in current_state.items():
        saved_value = saved_state[key]
        if not isinstance(current_value, Tensor) or not isinstance(
            saved_value,
            Tensor,
        ):
            raise TypeError(f"model state {key!r} must contain tensors")
        if saved_value.dtype != current_value.dtype:
            raise ValueError(
                f"model state {key!r} dtype differs: "
                f"{saved_value.dtype} != {current_value.dtype}"
            )
        if key in dynamic_samples:
            library = dynamic_samples[key]
            if (
                saved_value.ndim != 2
                or saved_value.shape[1] != library.feature_dim
                or saved_value.shape[0] > library.init_max_samples
            ):
                raise ValueError(
                    f"model state {key!r} has an invalid dynamic DCL shape"
                )
        elif saved_value.shape != current_value.shape:
            raise ValueError(
                f"model state {key!r} shape differs: "
                f"{tuple(saved_value.shape)} != {tuple(current_value.shape)}"
            )


def _validate_optimizer_state_compatibility(
    optimizer: Optimizer,
    saved_state: Mapping[str, Any],
) -> None:
    if set(saved_state) != {"state", "param_groups"}:
        raise ValueError("optimizer state must contain exactly state/param_groups")
    if not isinstance(saved_state["state"], Mapping):
        raise TypeError("optimizer state entry must be a mapping")
    saved_groups = saved_state["param_groups"]
    if not isinstance(saved_groups, list):
        raise TypeError("optimizer param_groups must be a list")
    current_groups = optimizer.state_dict()["param_groups"]
    if len(saved_groups) != len(current_groups):
        raise ValueError("optimizer parameter-group count differs")
    saved_parameter_ids: list[int] = []
    for index, (saved_group, current_group) in enumerate(
        zip(saved_groups, current_groups)
    ):
        if not isinstance(saved_group, Mapping):
            raise TypeError(f"optimizer param_groups[{index}] must be a mapping")
        if set(saved_group) != set(current_group):
            raise ValueError(f"optimizer param_groups[{index}] keys differ")
        saved_parameters = saved_group.get("params")
        current_parameters = current_group.get("params")
        if not isinstance(saved_parameters, list) or not isinstance(
            current_parameters,
            list,
        ):
            raise TypeError("optimizer group params entries must be lists")
        if len(saved_parameters) != len(current_parameters):
            raise ValueError(
                f"optimizer param_groups[{index}] parameter count differs"
            )
        for parameter_id in saved_parameters:
            if not isinstance(parameter_id, int) or isinstance(parameter_id, bool):
                raise TypeError("optimizer saved parameter identifiers must be ints")
            saved_parameter_ids.append(parameter_id)
    if len(saved_parameter_ids) != len(set(saved_parameter_ids)):
        raise ValueError("optimizer saved parameter identifiers must be unique")
    unknown_state = set(saved_state["state"]) - set(saved_parameter_ids)
    if unknown_state:
        raise ValueError(
            f"optimizer state contains unknown parameter ids: {sorted(unknown_state)}"
        )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: object | None,
    scaler: torch.cuda.amp.GradScaler | None,
    config_fingerprint: str,
    loaders: Mapping[str, DataLoader] | None = None,
    context: DistributedContext | None = None,
    map_location: str | torch.device | None = None,
    metadata_validator: TrainingCheckpointMetadataValidator | None = None,
) -> TrainingCheckpointLoadResult:
    """Resume a strict epoch-boundary checkpoint with no partial accumulation."""

    rank, world_size = _active_rank_world()

    def prepare() -> tuple[
        Path,
        str,
        dict[str, DataLoader],
        torch.device,
        dict[str, Any],
    ]:
        _validate_context(
            context,
            active_rank=rank,
            active_world=world_size,
        )
        prepared_path = _checkpoint_path(path, must_exist=True)
        _validate_model_and_optimizer(model, optimizer)
        _validate_stateful("scheduler", scheduler)
        _validate_stateful("scaler", scaler)
        if metadata_validator is not None and not callable(metadata_validator):
            raise TypeError("metadata_validator must be callable or None")
        prepared_fingerprint = _validate_fingerprint(config_fingerprint)
        prepared_loaders = _validate_loaders(loaders)
        prepared_device = _map_location_device(map_location, context)
        load_control_signature = {
            "path": str(prepared_path),
            "config_fingerprint": prepared_fingerprint,
            "loader_names": tuple(sorted(prepared_loaders)),
            "model_type": _type_identity(unwrap_model(model)),
            "optimizer_type": _type_identity(optimizer),
            "scheduler_present": scheduler is not None,
            "scheduler_type": _type_identity(scheduler),
            "scaler_present": scaler is not None,
            "scaler_type": _type_identity(scaler),
            "map_location_type": prepared_device.type,
            "metadata_validator_present": metadata_validator is not None,
            "metadata_validator_type": _type_identity(metadata_validator),
        }
        pickle.dumps(
            load_control_signature,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return (
            prepared_path,
            prepared_fingerprint,
            prepared_loaders,
            prepared_device,
            load_control_signature,
        )

    (
        checkpoint_path,
        fingerprint,
        normalized_loaders,
        device,
        load_control_signature,
    ) = _synchronize_checkpoint_preflight(
        "load_training_checkpoint preparation",
        rank=rank,
        world_size=world_size,
        validation=prepare,
    )
    _require_matching_control_signature(
        load_control_signature,
        world_size=world_size,
        operation="load_training_checkpoint control",
    )

    def phase_a() -> tuple[
        Mapping[str, Any],
        TrainerState,
        Mapping[str, Any],
        dict[str, Any],
    ]:
        raw_payload, content_digest = _load_payload_and_content_digest(
            checkpoint_path,
            map_location=device,
        )
        payload = _validate_training_payload(
            raw_payload,
            expected_fingerprint=fingerprint,
            expected_world_size=world_size,
        )
        if (payload["scheduler"] is None) != (scheduler is None):
            raise ValueError(
                "scheduler presence differs between checkpoint and current trainer"
            )
        if (payload["scaler"] is None) != (scaler is None):
            raise ValueError(
                "GradScaler presence differs between checkpoint and current trainer"
            )
        trainer_state = TrainerState.from_dict(payload["trainer_state"])
        _validate_epoch_boundary(trainer_state, model=None)
        runtime = _validate_runtime_entry(
            payload["per_rank_runtime"][rank],
            expected_rank=rank,
        )
        _validate_model_state_compatibility(model, payload["model"])
        _validate_optimizer_state_compatibility(optimizer, payload["optimizer"])
        validate_rng_state(runtime["rng"])
        validate_dataloader_states(normalized_loaders, runtime["loaders"])
        if metadata_validator is not None:
            metadata_validator(copy.deepcopy(payload["extra"]), trainer_state)
        payload_identity_signature = {
            "content_sha256": content_digest,
            "checkpoint_id": _validate_checkpoint_id(payload["checkpoint_id"]),
            "trainer_state": trainer_state.as_dict(),
            "config_fingerprint": payload["config_fingerprint"],
            "world_size": payload["world_size"],
            "scheduler_present": payload["scheduler"] is not None,
            "scaler_present": payload["scaler"] is not None,
            "extra_keys": tuple(sorted(payload["extra"])),
        }
        pickle.dumps(
            payload_identity_signature,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return payload, trainer_state, runtime, payload_identity_signature

    (
        payload,
        trainer_state,
        runtime,
        payload_identity_signature,
    ) = _synchronize_checkpoint_preflight(
        "load_training_checkpoint phase A validation",
        rank=rank,
        world_size=world_size,
        validation=phase_a,
    )
    _require_matching_control_signature(
        payload_identity_signature,
        world_size=world_size,
        operation="load_training_checkpoint payload identity",
    )

    def phase_b() -> TrainingCheckpointLoadResult:
        unwrap_model(model).load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(payload["scaler"])
        restore_rng_state(runtime["rng"])
        restore_dataloader_states(
            normalized_loaders,
            runtime["loaders"],
        )
        return TrainingCheckpointLoadResult(
            state=trainer_state,
            extra=dict(payload["extra"]),
            path=checkpoint_path,
        )

    result = _synchronize_checkpoint_preflight(
        "load_training_checkpoint phase B restore",
        rank=rank,
        world_size=world_size,
        validation=phase_b,
        failure_warning=(
            "trainer objects may be partially changed and must be discarded "
            "and rebuilt before continuing"
        ),
    )
    _barrier(context)
    return result


def _extract_pretrained_state(payload: object) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("pretrained checkpoint must contain a mapping")
    candidate: object
    if "model" in payload:
        candidate = payload["model"]
    elif "model_state_dict" in payload:
        candidate = payload["model_state_dict"]
    elif "state_dict" in payload:
        candidate = payload["state_dict"]
    else:
        candidate = payload
    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError("pretrained model state must be a non-empty mapping")
    if any(not isinstance(key, str) or not key for key in candidate):
        raise TypeError("pretrained model-state keys must be non-empty strings")
    if any(not isinstance(value, Tensor) for value in candidate.values()):
        raise TypeError("pretrained model-state values must all be tensors")

    keys = tuple(candidate)
    prefixed = tuple(key.startswith("module.") for key in keys)
    if any(prefixed) and not all(prefixed):
        raise ValueError("pretrained state has a mixture of module-prefixed keys")
    if all(prefixed):
        stripped = {key[len("module.") :]: candidate[key] for key in keys}
        if any(not key for key in stripped):
            raise ValueError("module prefix removal produced an empty state key")
        if len(stripped) != len(candidate):
            raise ValueError("module prefix removal produced duplicate state keys")
        return stripped
    return candidate


def load_pretrained_semmol(
    path: str | Path,
    model: nn.Module,
    *,
    map_location: str | torch.device,
) -> PretrainedTransferResult:
    """Transfer pretrained SemMol weights without optimizer or RNG state."""

    checkpoint_path = _checkpoint_path(path, must_exist=True)
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    device = _map_location_device(map_location, context=None)
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = _extract_pretrained_state(payload)
    target = unwrap_model(model)
    incompatible = target.load_state_dict(state_dict, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    disallowed_missing = [
        key for key in missing if not key.startswith("property_head.")
    ]
    disallowed_unexpected = [
        key for key in unexpected if not key.startswith("pretraining_heads.")
    ]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "pretrained SemMol state is incompatible; "
            f"missing={disallowed_missing}, unexpected={disallowed_unexpected}"
        )

    libraries = [
        (name, module)
        for name, module in target.named_modules()
        if isinstance(module, DynamicCentralLibrary)
    ]
    if not libraries:
        raise RuntimeError("target model does not contain a DynamicCentralLibrary")
    uninitialized = [name for name, library in libraries if not library.is_initialized]
    if uninitialized:
        raise RuntimeError(
            "pretrained transfer did not provide initialized DCL state for "
            f"{uninitialized}"
        )
    return PretrainedTransferResult(
        missing_keys=missing,
        unexpected_keys=unexpected,
        path=checkpoint_path,
    )


__all__ = [
    "PretrainedTransferResult",
    "TrainingCheckpointMetadataValidator",
    "TrainingCheckpointLoadResult",
    "capture_rng_state",
    "configuration_fingerprint",
    "load_pretrained_semmol",
    "load_training_checkpoint",
    "restore_rng_state",
    "save_training_checkpoint",
    "validate_rng_state",
]

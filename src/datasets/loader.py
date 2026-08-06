"""Deterministic DataLoader factory shared by single-node and torch.distributed runs."""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def _seed_worker(worker_id: int) -> None:
    # The DataLoader generator has already derived torch.initial_seed for each worker.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    collate_fn: Callable,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = 2,
    seed: int = 3407,
) -> DataLoader:
    """Create a DataLoader without duplicate sampling for the current distributed state."""

    for name, value in (
        ("batch_size", batch_size),
        ("num_workers", num_workers),
        ("seed", seed),
    ):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    for name, value in (
        ("shuffle", shuffle),
        ("pin_memory", pin_memory),
        ("drop_last", drop_last),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a bool")
    if persistent_workers is not None and not isinstance(
        persistent_workers,
        (bool, np.bool_),
    ):
        raise TypeError("persistent_workers must be a bool or None")
    if prefetch_factor is not None and (
        not isinstance(prefetch_factor, (int, np.integer))
        or isinstance(prefetch_factor, bool)
    ):
        raise TypeError("prefetch_factor must be an integer or None")
    if not callable(collate_fn):
        raise TypeError("collate_fn must be callable")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if seed < 0 or seed > 2**63 - 1:
        raise ValueError("seed must be in [0, 2**63 - 1]")
    if prefetch_factor is not None and prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be a positive integer or None")
    batch_size = int(batch_size)
    num_workers = int(num_workers)
    seed = int(seed)
    prefetch_factor = (
        None if prefetch_factor is None else int(prefetch_factor)
    )
    shuffle = bool(shuffle)
    pin_memory = bool(pin_memory)
    drop_last = bool(drop_last)
    persistent_workers = (
        None if persistent_workers is None else bool(persistent_workers)
    )

    is_distributed = distributed.is_available() and distributed.is_initialized()
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=distributed.get_world_size(),
            rank=distributed.get_rank(),
            shuffle=shuffle,
            seed=int(seed),
            drop_last=drop_last,
        )
        if is_distributed
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "persistent_workers": bool(persistent_workers and num_workers > 0),
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def set_dataloader_epoch(loader: DataLoader, epoch: int) -> None:
    """Synchronize the epoch of the DDP sampler and dynamic-masking Collator."""

    if not isinstance(epoch, (int, np.integer)) or isinstance(epoch, bool):
        raise TypeError("epoch must be an integer")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    sampler = loader.sampler
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)
    collator = loader.collate_fn
    set_epoch = getattr(collator, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)

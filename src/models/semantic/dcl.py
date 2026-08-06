"""Distributed dynamic central library for semantic molecular prototypes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F


_INIT_METHODS: Final[frozenset[str]] = frozenset(
    {"kmeans_plus_plus", "first_batch", "random"}
)


@dataclass(frozen=True)
class DCLAssignment:
    """Cosine-space hard and soft assignments to semantic centers."""

    similarities: Tensor
    hard_assignments: Tensor
    soft_assignments: Tensor


@dataclass(frozen=True)
class DCLUpdate:
    """Observable state transition produced by one DCL update call."""

    initialized: bool
    just_initialized: bool
    updated: bool
    global_sample_count: int
    active_cluster_count: int
    step: int
    update_count: int


def _positive_integer(name: str, value: object, *, minimum: int = 1) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {normalized}")
    return normalized


def _finite_real(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


class DynamicCentralLibrary(nn.Module):
    """Maintain data-initialized semantic centers with distributed EMA updates.

    The default path never exposes random centers to ACSM. Projected features
    are gathered across ranks and accumulated until at least ``num_clusters``
    samples are available, after which rank zero performs deterministic
    spherical K-means initialization and broadcasts the result. Online updates
    all-reduce per-cluster sums and counts, so every rank evolves an identical
    center library.
    """

    def __init__(
        self,
        num_clusters: int = 256,
        feature_dim: int = 256,
        ema_momentum: float = 0.9,
        *,
        init_method: str = "kmeans_plus_plus",
        init_num_iters: int = 10,
        init_max_samples: int = 4096,
        init_seed: int = 3407,
        reassign_interval: int = 1,
        assignment_temperature: float = 0.5,
        center_l2_normalize: bool = True,
        distributed_sync: bool = True,
        eps: float = 1.0e-8,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        self.num_clusters = _positive_integer("num_clusters", num_clusters)
        self.feature_dim = _positive_integer("feature_dim", feature_dim)
        self.init_num_iters = _positive_integer(
            "init_num_iters", init_num_iters
        )
        self.init_max_samples = _positive_integer(
            "init_max_samples", init_max_samples
        )
        if self.init_max_samples < self.num_clusters:
            raise ValueError(
                "init_max_samples must be greater than or equal to "
                f"num_clusters ({self.num_clusters})"
            )
        self.init_seed = _positive_integer(
            "init_seed", init_seed, minimum=0
        )
        self.reassign_interval = _positive_integer(
            "reassign_interval", reassign_interval
        )

        momentum = _finite_real("ema_momentum", ema_momentum)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                f"ema_momentum must be in [0, 1), got {momentum}"
            )
        temperature = _finite_real(
            "assignment_temperature", assignment_temperature
        )
        if temperature <= 0.0:
            raise ValueError("assignment_temperature must be positive")
        epsilon = _finite_real("eps", eps)
        if epsilon <= 0.0:
            raise ValueError("eps must be positive")
        if not isinstance(init_method, str) or not init_method.strip():
            raise ValueError("init_method must be a non-empty string")
        normalized_init = init_method.strip().lower()
        if normalized_init not in _INIT_METHODS:
            raise ValueError(
                f"unsupported init_method={init_method!r}; expected one of "
                f"{sorted(_INIT_METHODS)}"
            )
        for name, value in (
            ("center_l2_normalize", center_l2_normalize),
            ("distributed_sync", distributed_sync),
            ("validate_values", validate_values),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")

        self.ema_momentum = momentum
        self.init_method = normalized_init
        self.assignment_temperature = temperature
        self.center_l2_normalize = center_l2_normalize
        self.distributed_sync = distributed_sync
        self.eps = epsilon
        self.validate_values = validate_values

        self.register_buffer(
            "centers",
            torch.zeros(
                self.num_clusters,
                self.feature_dim,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "initialized",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "update_count",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "cluster_counts",
            torch.zeros(self.num_clusters, dtype=torch.float32),
        )
        self.register_buffer(
            "initialization_samples",
            torch.empty(
                0,
                self.feature_dim,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "initialization_sample_count",
            torch.zeros((), dtype=torch.long),
        )

        if self.init_method == "random":
            self._initialize_random_ablation()

    @property
    def is_initialized(self) -> bool:
        """Whether the persistent center buffer contains usable prototypes."""

        return bool(self.initialized.item())

    def _distributed_active(self) -> bool:
        return (
            self.distributed_sync
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    @torch.no_grad()
    def synchronize_distributed_state(self) -> None:
        """Make rank zero authoritative before matching or center updates."""

        cached_count = int(self.initialization_sample_count.item())
        cached_rows = int(self.initialization_samples.shape[0])
        locally_valid = (
            cached_count == cached_rows
            and cached_rows <= self.init_max_samples
            and (not self.is_initialized or cached_rows == 0)
        )
        if not self._distributed_active():
            if not locally_valid:
                raise RuntimeError(
                    "DCL initialization cache state is inconsistent"
                )
            return

        state = self.step.new_tensor(
            (
                int(self.is_initialized),
                int(self.step.item()),
                int(self.update_count.item()),
                cached_count,
                cached_rows,
                int(locally_valid),
            )
        )
        dist.broadcast(state, src=0)
        if int(state[5].item()) != 1:
            raise RuntimeError(
                "rank zero has an inconsistent DCL initialization cache"
            )
        authoritative_initialized = bool(int(state[0].item()))
        authoritative_count = int(state[3].item())
        authoritative_rows = int(state[4].item())
        if authoritative_count != authoritative_rows:
            raise RuntimeError(
                "rank-zero DCL cache count and shape disagree"
            )

        self.initialized.fill_(authoritative_initialized)
        self.step.fill_(int(state[1].item()))
        self.update_count.fill_(int(state[2].item()))
        self.initialization_sample_count.fill_(authoritative_count)
        if int(self.initialization_samples.shape[0]) != authoritative_rows:
            self.initialization_samples = self.centers.new_empty(
                (authoritative_rows, self.feature_dim)
            )

        if authoritative_initialized:
            dist.broadcast(self.centers, src=0)
            dist.broadcast(self.cluster_counts, src=0)
        elif authoritative_rows > 0:
            dist.broadcast(self.initialization_samples, src=0)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        sample_key = prefix + "initialization_samples"
        saved_samples = state_dict.get(sample_key)
        if saved_samples is not None:
            if (
                not isinstance(saved_samples, Tensor)
                or saved_samples.ndim != 2
                or saved_samples.shape[1] != self.feature_dim
                or saved_samples.shape[0] > self.init_max_samples
            ):
                error_msgs.append(
                    f"{sample_key} must have shape [N, {self.feature_dim}] "
                    f"with N <= {self.init_max_samples}"
                )
            else:
                self.initialization_samples = (
                    self.initialization_samples.new_empty(
                        tuple(saved_samples.shape)
                    )
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        count_key = prefix + "initialization_sample_count"
        if sample_key in state_dict and count_key in state_dict:
            restored_count = int(
                self.initialization_sample_count.item()
            )
            if restored_count != int(
                self.initialization_samples.shape[0]
            ):
                error_msgs.append(
                    f"{count_key}={restored_count} does not match "
                    f"{sample_key} rows="
                    f"{self.initialization_samples.shape[0]}"
                )

    def _validate_features(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 2:
            raise ValueError(
                "features must have shape [num_samples, feature_dim], got "
                f"{tuple(features.shape)}"
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, "
                f"got {features.shape[1]}"
            )
        if not features.is_floating_point():
            raise TypeError(
                f"features must be floating point, got {features.dtype}"
            )
        if features.device != self.centers.device:
            raise ValueError(
                "features and DCL buffers must be on the same device: "
                f"{features.device} != {self.centers.device}"
            )
        prepared = features.float()
        if (
            self.validate_values
            and prepared.numel() > 0
            and not bool(torch.isfinite(prepared).all())
        ):
            raise ValueError("features contain NaN or infinite values")
        return F.normalize(
            prepared,
            p=2.0,
            dim=-1,
            eps=self.eps,
        )

    @torch.no_grad()
    def _initialize_random_ablation(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.init_seed)
        random_centers = torch.randn(
            self.num_clusters,
            self.feature_dim,
            generator=generator,
            dtype=torch.float32,
        )
        random_centers = F.normalize(
            random_centers,
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        self.centers.copy_(random_centers)
        self.initialized.fill_(True)

    def get_centers(self, *, require_initialized: bool = True) -> Tensor:
        """Return a detached copy of the current center matrix."""

        if not isinstance(require_initialized, bool):
            raise TypeError("require_initialized must be bool")
        if require_initialized and not self.is_initialized:
            raise RuntimeError(
                "DCL centers are not initialized; collect at least "
                f"{self.num_clusters} projected samples first"
            )
        centers = self.centers.float()
        if self.center_l2_normalize:
            centers = F.normalize(
                centers,
                p=2.0,
                dim=-1,
                eps=self.eps,
            )
        return centers.detach().clone()

    def snapshot_centers(
        self,
        *,
        require_initialized: bool = True,
    ) -> Tensor:
        """Clone centers for one forward pass before any in-place EMA update."""

        return self.get_centers(require_initialized=require_initialized)

    def _global_sample_count(self, local_count: int) -> int:
        count = self.centers.new_tensor(local_count, dtype=torch.long)
        if self._distributed_active():
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return int(count.item())

    @torch.no_grad()
    def _gather_variable_features(self, features: Tensor) -> Tensor:
        if not self._distributed_active():
            return features.contiguous()

        world_size = dist.get_world_size()
        local_size = torch.tensor(
            [features.shape[0]],
            dtype=torch.long,
            device=features.device,
        )
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size)
        sizes = [int(size.item()) for size in gathered_sizes]
        maximum_size = max(sizes, default=0)
        if maximum_size == 0:
            return features.new_empty((0, self.feature_dim))

        padded = features.new_zeros((maximum_size, self.feature_dim))
        if features.shape[0] > 0:
            padded[: features.shape[0]].copy_(features)
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded.contiguous())
        nonempty = [
            rank_features[:rank_size]
            for rank_features, rank_size in zip(gathered, sizes)
            if rank_size > 0
        ]
        if not nonempty:
            return features.new_empty((0, self.feature_dim))
        return torch.cat(nonempty, dim=0)

    @torch.no_grad()
    def _append_initialization_samples(self, features: Tensor) -> None:
        if features.shape[0] == 0:
            return
        start = int(self.initialization_samples.shape[0])
        remaining = self.init_max_samples - start
        if remaining <= 0:
            return
        take = min(remaining, int(features.shape[0]))
        self.initialization_samples = torch.cat(
            (
                self.initialization_samples,
                features[:take],
            ),
            dim=0,
        )
        self.initialization_sample_count.fill_(
            int(self.initialization_samples.shape[0])
        )

    @torch.no_grad()
    def _kmeans_plus_plus(self, features: Tensor) -> Tensor:
        sample_count = int(features.shape[0])
        if sample_count < self.num_clusters:
            raise ValueError(
                "K-means++ initialization requires at least num_clusters "
                f"samples, got {sample_count} < {self.num_clusters}"
            )
        generator = torch.Generator(device=features.device)
        generator.manual_seed(self.init_seed)
        selected = torch.zeros(
            sample_count,
            dtype=torch.bool,
            device=features.device,
        )
        centers = features.new_empty(
            (self.num_clusters, self.feature_dim)
        )

        first_index = torch.randint(
            sample_count,
            (1,),
            generator=generator,
            device=features.device,
        )[0]
        selected[first_index] = True
        centers[0].copy_(features[first_index])
        closest_distance = (
            2.0 - 2.0 * (features @ centers[0].unsqueeze(-1)).squeeze(-1)
        ).clamp_min_(0.0)

        for center_index in range(1, self.num_clusters):
            probabilities = closest_distance.masked_fill(selected, 0.0)
            probability_mass = probabilities.sum()
            if bool(probability_mass > self.eps):
                next_index = torch.multinomial(
                    probabilities / probability_mass,
                    num_samples=1,
                    generator=generator,
                )[0]
            else:
                available = torch.nonzero(~selected, as_tuple=False).flatten()
                next_index = available[0]
            selected[next_index] = True
            centers[center_index].copy_(features[next_index])
            distance = (
                2.0
                - 2.0
                * (features @ centers[center_index].unsqueeze(-1)).squeeze(-1)
            ).clamp_min_(0.0)
            closest_distance = torch.minimum(closest_distance, distance)
        return centers

    @torch.no_grad()
    def _spherical_lloyd(self, features: Tensor, centers: Tensor) -> Tensor:
        current = F.normalize(
            centers.float(),
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        for _ in range(self.init_num_iters):
            assignments = torch.argmax(features @ current.transpose(0, 1), dim=1)
            sums = features.new_zeros(
                (self.num_clusters, self.feature_dim)
            )
            sums.index_add_(0, assignments, features)
            counts = torch.bincount(
                assignments,
                minlength=self.num_clusters,
            ).to(dtype=torch.float32)
            means = sums / counts.clamp_min(1.0).unsqueeze(-1)
            proposed = torch.where(
                counts.unsqueeze(-1) > 0.0,
                means,
                current,
            )
            current = F.normalize(
                proposed,
                p=2.0,
                dim=-1,
                eps=self.eps,
            )
        return current

    @torch.no_grad()
    def _initialize_from_accumulator(self) -> int:
        sample_count = int(self.initialization_sample_count.item())
        if sample_count < self.num_clusters:
            raise RuntimeError(
                "cannot initialize DCL before enough samples are accumulated"
            )
        features = self.initialization_samples[:sample_count]
        rank = dist.get_rank() if self._distributed_active() else 0
        if rank == 0:
            if self.init_method == "first_batch":
                initial_centers = features[: self.num_clusters].clone()
            else:
                initial_centers = self._kmeans_plus_plus(features)
            initial_centers = self._spherical_lloyd(
                features,
                initial_centers,
            )
        else:
            initial_centers = features.new_zeros(
                (self.num_clusters, self.feature_dim)
            )

        if self._distributed_active():
            dist.broadcast(initial_centers, src=0)
        initial_centers = F.normalize(
            initial_centers.float(),
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        self.centers.copy_(initial_centers)
        assignments = torch.argmax(
            features @ initial_centers.transpose(0, 1),
            dim=1,
        )
        initial_counts = torch.bincount(
            assignments,
            minlength=self.num_clusters,
        ).to(dtype=torch.float32)
        self.cluster_counts.copy_(initial_counts)
        self.initialized.fill_(True)
        self.initialization_samples = self.centers.new_empty(
            (0, self.feature_dim)
        )
        self.initialization_sample_count.zero_()
        return int((initial_counts > 0.0).sum().item())

    @torch.no_grad()
    def update_centers(
        self,
        features: Tensor,
        *,
        synchronize_state: bool = True,
    ) -> DCLUpdate:
        """Accumulate initialization data or apply one synchronized EMA update."""

        if not isinstance(synchronize_state, bool):
            raise TypeError("synchronize_state must be bool")
        prepared = self._validate_features(features)
        if synchronize_state:
            self.synchronize_distributed_state()
        self.step.add_(1)
        step = int(self.step.item())
        global_sample_count = self._global_sample_count(
            int(prepared.shape[0])
        )

        if not self.is_initialized:
            gathered = self._gather_variable_features(prepared)
            self._append_initialization_samples(gathered)
            if int(self.initialization_sample_count.item()) < self.num_clusters:
                return DCLUpdate(
                    initialized=False,
                    just_initialized=False,
                    updated=False,
                    global_sample_count=global_sample_count,
                    active_cluster_count=0,
                    step=step,
                    update_count=int(self.update_count.item()),
                )
            active_cluster_count = self._initialize_from_accumulator()
            return DCLUpdate(
                initialized=True,
                just_initialized=True,
                updated=False,
                global_sample_count=global_sample_count,
                active_cluster_count=active_cluster_count,
                step=step,
                update_count=int(self.update_count.item()),
            )

        if (step - 1) % self.reassign_interval != 0:
            return DCLUpdate(
                initialized=True,
                just_initialized=False,
                updated=False,
                global_sample_count=global_sample_count,
                active_cluster_count=0,
                step=step,
                update_count=int(self.update_count.item()),
            )

        matching_centers = F.normalize(
            self.centers.float(),
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        similarities = prepared @ matching_centers.transpose(0, 1)
        assignments = torch.argmax(similarities, dim=1)
        cluster_sums = prepared.new_zeros(
            (self.num_clusters, self.feature_dim)
        )
        if assignments.numel() > 0:
            cluster_sums.index_add_(0, assignments, prepared)
        cluster_sizes = torch.bincount(
            assignments,
            minlength=self.num_clusters,
        ).to(dtype=torch.float32)

        if self._distributed_active():
            dist.all_reduce(cluster_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_sizes, op=dist.ReduceOp.SUM)

        active = cluster_sizes > 0.0
        means = cluster_sums / cluster_sizes.clamp_min(1.0).unsqueeze(-1)
        proposed = (
            self.ema_momentum * self.centers
            + (1.0 - self.ema_momentum) * means
        )
        if self.center_l2_normalize:
            proposed = F.normalize(
                proposed,
                p=2.0,
                dim=-1,
                eps=self.eps,
            )
        updated_centers = torch.where(
            active.unsqueeze(-1),
            proposed,
            self.centers,
        )
        self.centers.copy_(updated_centers)
        self.cluster_counts.add_(cluster_sizes)

        active_cluster_count = int(active.sum().item())
        updated = active_cluster_count > 0
        if updated:
            self.update_count.add_(1)
        return DCLUpdate(
            initialized=True,
            just_initialized=False,
            updated=updated,
            global_sample_count=global_sample_count,
            active_cluster_count=active_cluster_count,
            step=step,
            update_count=int(self.update_count.item()),
        )

    def assign(
        self,
        features: Tensor,
        *,
        temperature: float | None = None,
    ) -> DCLAssignment:
        """Assign features without mutating the library."""

        if not self.is_initialized:
            raise RuntimeError("cannot assign features before DCL initialization")
        prepared = self._validate_features(features)
        if temperature is None:
            normalized_temperature = self.assignment_temperature
        else:
            normalized_temperature = _finite_real("temperature", temperature)
            if normalized_temperature <= 0.0:
                raise ValueError("temperature must be positive")
        centers = F.normalize(
            self.centers.float(),
            p=2.0,
            dim=-1,
            eps=self.eps,
        )
        similarities = prepared @ centers.transpose(0, 1)
        hard_assignments = torch.argmax(similarities, dim=1)
        soft_assignments = F.softmax(
            similarities / normalized_temperature,
            dim=-1,
        )
        return DCLAssignment(
            similarities=similarities,
            hard_assignments=hard_assignments,
            soft_assignments=soft_assignments,
        )

    def forward(
        self,
        features: Tensor,
        *,
        temperature: float | None = None,
    ) -> DCLAssignment:
        return self.assign(features, temperature=temperature)


__all__ = [
    "DCLAssignment",
    "DCLUpdate",
    "DynamicCentralLibrary",
]

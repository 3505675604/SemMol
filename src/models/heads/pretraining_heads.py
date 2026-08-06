"""Trainable prediction heads for SemMol self-supervised objectives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor, nn
from torch_geometric.nn import radius_graph


@dataclass(frozen=True)
class MLMPrediction:
    """Compact masked-language-model prediction aligned by ``sample_index``."""

    logits: Tensor
    sample_index: Tensor
    token_mask: Tensor


@dataclass(frozen=True)
class GraphReconstructionPrediction:
    """Per-field graph logits and optional clean-view structural target."""

    node_logits: tuple[Tensor, ...]
    edge_logits: tuple[Tensor, ...]
    graph_sample_index: Tensor
    corrupted_embedding: Tensor
    clean_embedding: Tensor | None


@dataclass(frozen=True)
class GeometryDenoisingPrediction:
    """Full-batch equivariant prediction of the injected coordinate noise."""

    predicted_noise: Tensor
    valid_mask: Tensor


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive, got {normalized}")
    return normalized


def _positive_real(name: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return normalized


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _cardinalities(name: str, values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(
        _positive_integer(f"{name}[{index}]", value)
        for index, value in enumerate(values)
    )
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _require_floating_matrix(
    name: str,
    value: Tensor,
    feature_dim: int,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != feature_dim:
        raise ValueError(
            f"{name} must have shape [N, {feature_dim}], got "
            f"{tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")


class MaskedLanguageModelingHead(nn.Module):
    """Predict ESPF token identities from compact 1D token features."""

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        *,
        bias: bool = True,
        layer_norm_eps: float = 1.0e-5,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer("input_dim", input_dim)
        self.vocab_size = _positive_integer("vocab_size", vocab_size)
        if not isinstance(bias, bool):
            raise TypeError("bias must be bool")
        epsilon = _positive_real("layer_norm_eps", layer_norm_eps)
        self.validate_values = _bool(
            "validate_values",
            validate_values,
        )
        self.normalization = nn.LayerNorm(
            self.input_dim,
            eps=epsilon,
        )
        self.classifier = nn.Linear(
            self.input_dim,
            self.vocab_size,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.normalization.weight)
        nn.init.zeros_(self.normalization.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        tokens: Tensor,
        sample_index: Tensor,
        token_mask: Tensor,
    ) -> MLMPrediction:
        if not isinstance(tokens, Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.ndim != 3 or tokens.shape[2] != self.input_dim:
            raise ValueError(
                "tokens must have shape "
                f"[compact_batch, length, {self.input_dim}], got "
                f"{tuple(tokens.shape)}"
            )
        if not tokens.is_floating_point():
            raise TypeError("tokens must be floating point")
        if not isinstance(sample_index, Tensor):
            raise TypeError("sample_index must be a torch.Tensor")
        if (
            sample_index.ndim != 1
            or sample_index.dtype != torch.long
            or sample_index.shape[0] != tokens.shape[0]
        ):
            raise ValueError(
                "sample_index must be torch.long with shape [compact_batch]"
            )
        if not isinstance(token_mask, Tensor):
            raise TypeError("token_mask must be a torch.Tensor")
        if (
            token_mask.shape != tokens.shape[:2]
            or token_mask.dtype != torch.bool
        ):
            raise ValueError(
                "token_mask must be bool with shape "
                f"{tuple(tokens.shape[:2])}"
            )
        devices = {tokens.device, sample_index.device, token_mask.device}
        if len(devices) != 1:
            raise ValueError(
                "tokens, sample_index, and token_mask must share a device"
            )
        if (
            self.validate_values
            and tokens.numel() > 0
            and not bool(torch.isfinite(tokens).all())
        ):
            raise ValueError("tokens contain NaN or infinite values")

        logits = self.classifier(self.normalization(tokens))
        if (
            self.validate_values
            and logits.numel() > 0
            and not bool(torch.isfinite(logits).all())
        ):
            raise FloatingPointError(
                "MLM prediction head produced non-finite logits"
            )
        return MLMPrediction(
            logits=logits,
            sample_index=sample_index,
            token_mask=token_mask,
        )


class GraphReconstructionHead(nn.Module):
    """Predict every masked OGB atom and bond feature field independently."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        node_cardinalities: Sequence[int],
        edge_cardinalities: Sequence[int],
        *,
        bias: bool = True,
        layer_norm_eps: float = 1.0e-5,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer("input_dim", input_dim)
        self.embedding_dim = _positive_integer(
            "embedding_dim",
            embedding_dim,
        )
        self.node_cardinalities = _cardinalities(
            "node_cardinalities",
            node_cardinalities,
        )
        self.edge_cardinalities = _cardinalities(
            "edge_cardinalities",
            edge_cardinalities,
        )
        if not isinstance(bias, bool):
            raise TypeError("bias must be bool")
        epsilon = _positive_real("layer_norm_eps", layer_norm_eps)
        self.validate_values = _bool(
            "validate_values",
            validate_values,
        )

        self.node_normalization = nn.LayerNorm(
            self.input_dim,
            eps=epsilon,
        )
        self.edge_normalization = nn.LayerNorm(
            self.input_dim,
            eps=epsilon,
        )
        self.node_classifiers = nn.ModuleList(
            nn.Linear(self.input_dim, cardinality, bias=bias)
            for cardinality in self.node_cardinalities
        )
        self.edge_classifiers = nn.ModuleList(
            nn.Linear(self.input_dim, cardinality, bias=bias)
            for cardinality in self.edge_cardinalities
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for normalization in (
            self.node_normalization,
            self.edge_normalization,
        ):
            nn.init.ones_(normalization.weight)
            nn.init.zeros_(normalization.bias)
        for classifier in (
            *self.node_classifiers,
            *self.edge_classifiers,
        ):
            nn.init.xavier_uniform_(classifier.weight)
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)

    def forward(
        self,
        node_embedding: Tensor,
        edge_embedding: Tensor,
        graph_sample_index: Tensor,
        corrupted_embedding: Tensor,
        clean_embedding: Tensor | None = None,
    ) -> GraphReconstructionPrediction:
        _require_floating_matrix(
            "node_embedding",
            node_embedding,
            self.input_dim,
        )
        _require_floating_matrix(
            "edge_embedding",
            edge_embedding,
            self.input_dim,
        )
        _require_floating_matrix(
            "corrupted_embedding",
            corrupted_embedding,
            self.embedding_dim,
        )
        if clean_embedding is not None:
            _require_floating_matrix(
                "clean_embedding",
                clean_embedding,
                self.embedding_dim,
            )
            if clean_embedding.shape != corrupted_embedding.shape:
                raise ValueError(
                    "clean_embedding and corrupted_embedding must have "
                    "identical shapes"
                )
            if clean_embedding.device != corrupted_embedding.device:
                raise ValueError(
                    "clean and corrupted embeddings must share a device"
                )
        if not isinstance(graph_sample_index, Tensor):
            raise TypeError("graph_sample_index must be a torch.Tensor")
        if (
            graph_sample_index.ndim != 1
            or graph_sample_index.dtype != torch.long
            or graph_sample_index.shape[0] != corrupted_embedding.shape[0]
        ):
            raise ValueError(
                "graph_sample_index must be torch.long with shape "
                "[compact_graph_batch]"
            )
        devices = {
            node_embedding.device,
            edge_embedding.device,
            graph_sample_index.device,
            corrupted_embedding.device,
        }
        if len(devices) != 1:
            raise ValueError("all graph reconstruction inputs must share a device")
        if self.validate_values:
            for name, value in (
                ("node_embedding", node_embedding),
                ("edge_embedding", edge_embedding),
                ("corrupted_embedding", corrupted_embedding),
            ):
                if value.numel() > 0 and not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{name} contains non-finite values")
            if (
                clean_embedding is not None
                and clean_embedding.numel() > 0
                and not bool(torch.isfinite(clean_embedding).all())
            ):
                raise ValueError("clean_embedding contains non-finite values")

        normalized_nodes = self.node_normalization(node_embedding)
        normalized_edges = self.edge_normalization(edge_embedding)
        node_logits = tuple(
            classifier(normalized_nodes)
            for classifier in self.node_classifiers
        )
        edge_logits = tuple(
            classifier(normalized_edges)
            for classifier in self.edge_classifiers
        )
        return GraphReconstructionPrediction(
            node_logits=node_logits,
            edge_logits=edge_logits,
            graph_sample_index=graph_sample_index,
            corrupted_embedding=corrupted_embedding,
            clean_embedding=clean_embedding,
        )


class GeometryDenoisingHead(nn.Module):
    """Predict coordinate noise with an E(3)-equivariant relative-vector head.

    The head never regresses an absolute coordinate frame. It produces each
    atom's vector as a weighted sum of relative neighbor directions, with
    invariant weights conditioned on atom types and the DimeNet-derived
    molecule representation.
    """

    _MAX_ATOMIC_NUMBER = 118

    def __init__(
        self,
        context_dim: int,
        *,
        hidden_dim: int = 128,
        num_radial: int = 16,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        layer_norm_eps: float = 1.0e-5,
        eps: float = 1.0e-8,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        self.context_dim = _positive_integer("context_dim", context_dim)
        self.hidden_dim = _positive_integer("hidden_dim", hidden_dim)
        self.num_radial = _positive_integer("num_radial", num_radial)
        self.max_num_neighbors = _positive_integer(
            "max_num_neighbors",
            max_num_neighbors,
        )
        self.cutoff = _positive_real("cutoff", cutoff)
        self.eps = _positive_real("eps", eps)
        epsilon = _positive_real("layer_norm_eps", layer_norm_eps)
        self.validate_values = _bool(
            "validate_values",
            validate_values,
        )

        self.atom_embedding = nn.Embedding(
            self._MAX_ATOMIC_NUMBER + 1,
            self.hidden_dim,
            padding_idx=0,
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(self.context_dim, eps=epsilon),
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.SiLU(),
        )
        edge_input_dim = 2 * self.hidden_dim + self.num_radial
        self.edge_weight = nn.Sequential(
            nn.Linear(edge_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.node_scale = nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=epsilon),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Tanh(),
        )
        radial_centers = torch.linspace(
            0.0,
            self.cutoff,
            self.num_radial,
            dtype=torch.float32,
        )
        self.register_buffer("radial_centers", radial_centers)
        radial_step = (
            self.cutoff
            if self.num_radial == 1
            else self.cutoff / float(self.num_radial - 1)
        )
        self.register_buffer(
            "radial_gamma",
            torch.tensor(
                1.0 / max(radial_step * radial_step, self.eps),
                dtype=torch.float32,
            ),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.atom_embedding.weight,
            mean=0.0,
            std=self.hidden_dim**-0.5,
        )
        with torch.no_grad():
            self.atom_embedding.weight[0].zero_()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _parameter_zero(self, reference: Tensor) -> Tensor:
        zero = reference.reshape(-1)[:0].sum()
        for parameter in self.parameters():
            zero = zero + parameter.reshape(-1)[:1].sum() * 0.0
        return zero

    def _validate_inputs(
        self,
        atomic_numbers: Tensor,
        noisy_coords: Tensor,
        atom_mask: Tensor,
        conformer_mask: Tensor,
        context: Tensor,
        sample_index: Tensor,
    ) -> None:
        if not isinstance(atomic_numbers, Tensor):
            raise TypeError("atomic_numbers must be a torch.Tensor")
        if atomic_numbers.ndim != 2 or atomic_numbers.dtype != torch.long:
            raise ValueError(
                "atomic_numbers must be torch.long with shape [batch, atoms]"
            )
        if not isinstance(noisy_coords, Tensor):
            raise TypeError("noisy_coords must be a torch.Tensor")
        if (
            noisy_coords.ndim != 4
            or noisy_coords.shape[-1] != 3
            or not noisy_coords.is_floating_point()
        ):
            raise ValueError(
                "noisy_coords must be floating point with shape "
                "[batch, conformers, atoms, 3]"
            )
        batch_size, atom_count = atomic_numbers.shape
        if (
            noisy_coords.shape[0] != batch_size
            or noisy_coords.shape[2] != atom_count
        ):
            raise ValueError(
                "noisy_coords batch and atom dimensions must match "
                "atomic_numbers"
            )
        if (
            not isinstance(atom_mask, Tensor)
            or atom_mask.shape != atomic_numbers.shape
            or atom_mask.dtype != torch.bool
        ):
            raise ValueError(
                "atom_mask must be bool with shape [batch, atoms]"
            )
        if (
            not isinstance(conformer_mask, Tensor)
            or conformer_mask.shape
            != (batch_size, noisy_coords.shape[1])
            or conformer_mask.dtype != torch.bool
        ):
            raise ValueError(
                "conformer_mask must be bool with shape [batch, conformers]"
            )
        _require_floating_matrix("context", context, self.context_dim)
        if (
            not isinstance(sample_index, Tensor)
            or sample_index.ndim != 1
            or sample_index.dtype != torch.long
            or sample_index.shape[0] != context.shape[0]
        ):
            raise ValueError(
                "sample_index must be torch.long with shape [compact_batch]"
            )
        devices = {
            atomic_numbers.device,
            noisy_coords.device,
            atom_mask.device,
            conformer_mask.device,
            context.device,
            sample_index.device,
        }
        if len(devices) != 1:
            raise ValueError("all geometry denoising inputs must share a device")
        expected_sample_index = torch.nonzero(
            conformer_mask.any(dim=1),
            as_tuple=False,
        ).flatten()
        if not torch.equal(sample_index, expected_sample_index):
            raise ValueError(
                "sample_index must identify every row with a valid conformer"
            )
        if self.validate_values:
            if noisy_coords.numel() > 0 and not bool(
                torch.isfinite(noisy_coords).all()
            ):
                raise ValueError("noisy_coords contains non-finite values")
            if context.numel() > 0 and not bool(torch.isfinite(context).all()):
                raise ValueError("context contains non-finite values")
            valid_atomic_numbers = atomic_numbers[atom_mask]
            if valid_atomic_numbers.numel() > 0 and (
                bool(torch.any(valid_atomic_numbers < 1))
                or bool(
                    torch.any(
                        valid_atomic_numbers > self._MAX_ATOMIC_NUMBER
                    )
                )
            ):
                raise ValueError(
                    "valid atomic numbers must lie in [1, 118]"
                )

    def forward(
        self,
        atomic_numbers: Tensor,
        noisy_coords: Tensor,
        atom_mask: Tensor,
        conformer_mask: Tensor,
        context: Tensor,
        sample_index: Tensor,
    ) -> GeometryDenoisingPrediction:
        self._validate_inputs(
            atomic_numbers,
            noisy_coords,
            atom_mask,
            conformer_mask,
            context,
            sample_index,
        )
        batch_size, conformer_count, atom_count, _ = noisy_coords.shape
        valid_mask = (
            conformer_mask.unsqueeze(-1)
            & atom_mask.unsqueeze(1)
        )
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)
        parameter_zero = self._parameter_zero(noisy_coords)
        predicted_noise = torch.zeros_like(noisy_coords) + parameter_zero
        if valid_indices.shape[0] == 0:
            return GeometryDenoisingPrediction(
                predicted_noise=predicted_noise,
                valid_mask=valid_mask,
            )

        projected_context = self.context_projection(context)
        full_context = projected_context.new_zeros(
            (batch_size, self.hidden_dim)
        ).index_copy(
            0,
            sample_index,
            projected_context,
        )

        sample_ids = valid_indices[:, 0]
        conformer_ids = valid_indices[:, 1]
        atom_ids = valid_indices[:, 2]
        flat_coords = noisy_coords[
            sample_ids,
            conformer_ids,
            atom_ids,
        ]
        flat_atomic_numbers = atomic_numbers[sample_ids, atom_ids]
        atom_state = (
            self.atom_embedding(flat_atomic_numbers)
            + full_context.index_select(0, sample_ids)
        )
        flat_conformer_batch = (
            sample_ids * conformer_count + conformer_ids
        )
        edge_index = radius_graph(
            flat_coords,
            r=self.cutoff,
            batch=flat_conformer_batch,
            loop=False,
            max_num_neighbors=self.max_num_neighbors,
            flow="source_to_target",
        )
        flat_prediction = torch.zeros_like(flat_coords) + parameter_zero
        if edge_index.shape[1] > 0:
            source = edge_index[0]
            target = edge_index[1]
            relative = (
                flat_coords.index_select(0, source)
                - flat_coords.index_select(0, target)
            )
            distance = torch.linalg.vector_norm(
                relative.float(),
                dim=-1,
            )
            direction = relative / distance.to(
                dtype=relative.dtype
            ).clamp_min(self.eps).unsqueeze(-1)
            radial = torch.exp(
                -self.radial_gamma
                * (
                    distance.unsqueeze(-1)
                    - self.radial_centers.unsqueeze(0)
                ).square()
            ).to(dtype=atom_state.dtype)
            edge_features = torch.cat(
                (
                    atom_state.index_select(0, source),
                    atom_state.index_select(0, target),
                    radial,
                ),
                dim=-1,
            )
            coefficients = self.edge_weight(edge_features).squeeze(-1)
            messages = (
                coefficients.to(dtype=direction.dtype).unsqueeze(-1)
                * direction
            )
            flat_prediction = flat_prediction.index_add(
                0,
                target,
                messages,
            )
            neighbor_count = torch.bincount(
                target,
                minlength=flat_prediction.shape[0],
            ).to(dtype=flat_prediction.dtype)
            flat_prediction = flat_prediction / neighbor_count.clamp_min(
                1.0
            ).unsqueeze(-1)
            scale = 1.0 + self.node_scale(atom_state).to(
                dtype=flat_prediction.dtype
            )
            flat_prediction = flat_prediction * scale

        conformer_count_flat = batch_size * conformer_count
        conformer_atom_count = torch.bincount(
            flat_conformer_batch,
            minlength=conformer_count_flat,
        ).to(dtype=flat_prediction.dtype)
        conformer_prediction_sum = flat_prediction.new_zeros(
            (conformer_count_flat, 3)
        ).index_add(
            0,
            flat_conformer_batch,
            flat_prediction,
        )
        conformer_prediction_mean = (
            conformer_prediction_sum
            / conformer_atom_count.clamp_min(1.0).unsqueeze(-1)
        )
        flat_prediction = (
            flat_prediction
            - conformer_prediction_mean.index_select(
                0,
                flat_conformer_batch,
            )
        )
        predicted_noise[
            sample_ids,
            conformer_ids,
            atom_ids,
        ] = flat_prediction
        if (
            self.validate_values
            and predicted_noise.numel() > 0
            and not bool(torch.isfinite(predicted_noise).all())
        ):
            raise FloatingPointError(
                "geometry denoising head produced non-finite vectors"
            )
        return GeometryDenoisingPrediction(
            predicted_noise=predicted_noise,
            valid_mask=valid_mask,
        )


__all__ = [
    "GeometryDenoisingHead",
    "GeometryDenoisingPrediction",
    "GraphReconstructionHead",
    "GraphReconstructionPrediction",
    "MLMPrediction",
    "MaskedLanguageModelingHead",
]

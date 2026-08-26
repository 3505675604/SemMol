"""Sparse edge-aware GatedGCN, GINE, and GROVER-style graph encoders."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import softmax as segment_softmax
from torch_geometric.utils import to_dense_batch

from src.molecular.graph import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    OGB_ATOM_FEATURE_CARDINALITIES,
    OGB_BOND_FEATURE_CARDINALITIES,
)

from .common import (
    EncoderOutput,
    validate_encoder_output,
    validate_sample_index,
)


MASKED_OGB_ATOM_FEATURE_CARDINALITIES = tuple(
    cardinality + 1 for cardinality in OGB_ATOM_FEATURE_CARDINALITIES
)
MASKED_OGB_BOND_FEATURE_CARDINALITIES = tuple(
    cardinality + 1 for cardinality in OGB_BOND_FEATURE_CARDINALITIES
)


@dataclass(frozen=True)
class GraphEncoderOutput(EncoderOutput):
    """Graph output with flat features retained for reconstruction objectives."""

    node_embedding: Tensor
    edge_embedding: Tensor
    node_batch: Tensor
    edge_batch: Tensor


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _dropout_probability(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("dropout must be a real number")
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {result}")
    return result


def _cardinalities(
    values: Sequence[int],
    *,
    name: str,
    expected: tuple[int, ...],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc
    parsed: list[int] = []
    for column, value in enumerate(raw_values):
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"{name}[{column}] must be an integer")
        cardinality = int(value)
        if cardinality <= 1:
            raise ValueError(
                f"{name}[{column}] must contain at least two categories"
            )
        parsed.append(cardinality)
    result = tuple(parsed)
    if result != expected:
        raise ValueError(
            f"{name} must match the OGB categorical v1 schema with one reserved "
            f"mask category per field: expected {expected}, got {result}"
        )
    return result


def _require_boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _require_finite(tensor: Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains NaN or infinity")


def _scatter_sum(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum rank-two values by index without materializing dense adjacency."""

    if values.ndim != 2:
        raise ValueError(f"values must be rank two, got {tuple(values.shape)}")
    if index.ndim != 1 or index.numel() != values.shape[0]:
        raise ValueError("index must have one entry for every value row")
    if index.dtype != torch.long:
        raise TypeError(f"index must be torch.long, got {index.dtype}")
    output = values.new_zeros((dim_size, values.shape[1]))
    return output.index_add(0, index, values)


def _reset_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CategoricalFeatureEncoder(nn.Module):
    """Embed every categorical field independently and sum the field vectors."""

    def __init__(
        self,
        cardinalities: Sequence[int],
        hidden_size: int,
        *,
        feature_name: str,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(cardinalities, (str, bytes)):
            raise TypeError("cardinalities must be a sequence of integers")
        try:
            raw_cardinalities = tuple(cardinalities)
        except TypeError as exc:
            raise TypeError(
                "cardinalities must be a sequence of integers"
            ) from exc
        if not raw_cardinalities:
            raise ValueError("cardinalities cannot be empty")
        parsed: list[int] = []
        for column, value in enumerate(raw_cardinalities):
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"cardinalities[{column}] must be an integer")
            category_count = int(value)
            if category_count <= 1:
                raise ValueError(
                    f"cardinalities[{column}] must be greater than one"
                )
            parsed.append(category_count)
        if not isinstance(feature_name, str) or not feature_name:
            raise ValueError("feature_name must be a non-empty string")

        self.cardinalities = tuple(parsed)
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.feature_name = feature_name
        self.validate_values = _require_boolean(
            validate_values,
            "validate_values",
        )
        self.embeddings = nn.ModuleList(
            nn.Embedding(category_count, self.hidden_size)
            for category_count in self.cardinalities
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor):
            raise TypeError(f"{self.feature_name} must be a torch.Tensor")
        if features.ndim != 2:
            raise ValueError(
                f"{self.feature_name} must be rank two, got "
                f"{tuple(features.shape)}"
            )
        if features.shape[1] != len(self.cardinalities):
            raise ValueError(
                f"{self.feature_name} must have {len(self.cardinalities)} "
                f"fields, got {features.shape[1]}"
            )
        if features.dtype != torch.long:
            raise TypeError(
                f"{self.feature_name} must be torch.long, got {features.dtype}"
            )
        parameter_device = self.embeddings[0].weight.device
        if features.device != parameter_device:
            raise ValueError(
                f"{self.feature_name} is on {features.device}, but its "
                f"embeddings are on {parameter_device}"
            )

        encoded: Tensor | None = None
        if self.validate_values and features.numel() > 0:
            upper_bounds = features.new_tensor(self.cardinalities).unsqueeze(0)
            invalid = (features < 0) | (features >= upper_bounds)
            if bool(torch.any(invalid)):
                raise ValueError(
                    f"{self.feature_name} contains a category outside its "
                    "configured field cardinality"
                )

        for column, embedding in enumerate(self.embeddings):
            column_values = features[:, column]
            column_embedding = embedding(column_values)
            encoded = (
                column_embedding
                if encoded is None
                else encoded + column_embedding
            )

        if encoded is None:
            raise RuntimeError("categorical encoder has no configured fields")
        if self.validate_values:
            _require_finite(encoded, f"embedded {self.feature_name}")
        return encoded


class GatedGCNLayer(nn.Module):
    """Feature-wise gated message passing with learned edge-state updates."""

    def __init__(
        self,
        hidden_size: int,
        *,
        dropout: float,
        residual: bool,
        layer_norm: bool,
        gate_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.residual = _require_boolean(residual, "residual")
        use_layer_norm = _require_boolean(layer_norm, "layer_norm")
        if not isinstance(gate_epsilon, (int, float)) or isinstance(
            gate_epsilon, bool
        ):
            raise TypeError("gate_epsilon must be a real number")
        self.gate_epsilon = float(gate_epsilon)
        if self.gate_epsilon <= 0.0:
            raise ValueError("gate_epsilon must be positive")

        dropout_probability = _dropout_probability(dropout)
        self.node_self = nn.Linear(self.hidden_size, self.hidden_size)
        self.node_message = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )
        self.edge_self = nn.Linear(self.hidden_size, self.hidden_size)
        self.edge_source = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )
        self.edge_target = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )
        self.node_dropout = nn.Dropout(dropout_probability)
        self.edge_dropout = nn.Dropout(dropout_probability)
        self.node_norm = (
            nn.LayerNorm(self.hidden_size) if use_layer_norm else nn.Identity()
        )
        self.edge_norm = (
            nn.LayerNorm(self.hidden_size) if use_layer_norm else nn.Identity()
        )
        self.apply(_reset_linear)

    def forward(
        self,
        node_embedding: Tensor,
        edge_embedding: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        edge_logits = (
            self.edge_self(edge_embedding)
            + self.edge_source(node_embedding[source])
            + self.edge_target(node_embedding[target])
        )
        gates = torch.sigmoid(edge_logits)

        messages = self.node_message(node_embedding[source]) * gates
        message_sum = _scatter_sum(
            messages, target, node_embedding.shape[0]
        )
        gate_sum = _scatter_sum(gates, target, node_embedding.shape[0])
        neighbor_update = message_sum / gate_sum.clamp_min(self.gate_epsilon)
        node_candidate = F.silu(
            self.node_self(node_embedding) + neighbor_update
        )
        edge_candidate = F.silu(edge_logits)

        node_update = self.node_dropout(node_candidate)
        edge_update = self.edge_dropout(edge_candidate)
        if self.residual:
            node_update = node_embedding + node_update
            edge_update = edge_embedding + edge_update

        node_output = self.node_norm(node_update)
        edge_output = self.edge_norm(edge_update)
        return node_output, edge_output


class GINELayer(nn.Module):
    """Sparse GINE update whose messages always consume bond embeddings."""

    def __init__(
        self,
        hidden_size: int,
        *,
        dropout: float,
        residual: bool,
        layer_norm: bool,
        train_eps: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.residual = _require_boolean(residual, "residual")
        use_layer_norm = _require_boolean(layer_norm, "layer_norm")
        train_epsilon = _require_boolean(train_eps, "train_eps")
        dropout_probability = _dropout_probability(dropout)

        epsilon = torch.zeros(1)
        if train_epsilon:
            self.eps = nn.Parameter(epsilon)
        else:
            self.register_buffer("eps", epsilon)

        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.edge_self = nn.Linear(self.hidden_size, self.hidden_size)
        self.edge_source = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )
        self.edge_target = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )
        self.node_dropout = nn.Dropout(dropout_probability)
        self.edge_dropout = nn.Dropout(dropout_probability)
        self.node_norm = (
            nn.LayerNorm(self.hidden_size) if use_layer_norm else nn.Identity()
        )
        self.edge_norm = (
            nn.LayerNorm(self.hidden_size) if use_layer_norm else nn.Identity()
        )
        self.apply(_reset_linear)

    def forward(
        self,
        node_embedding: Tensor,
        edge_embedding: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        edge_candidate = F.silu(
            self.edge_self(edge_embedding)
            + self.edge_source(node_embedding[source])
            + self.edge_target(node_embedding[target])
        )
        edge_update = self.edge_dropout(edge_candidate)
        if self.residual:
            edge_update = edge_embedding + edge_update
        edge_output = self.edge_norm(edge_update)

        messages = F.relu(node_embedding[source] + edge_output)
        aggregated = _scatter_sum(
            messages, target, node_embedding.shape[0]
        )
        node_candidate = self.node_mlp(
            (1.0 + self.eps) * node_embedding + aggregated
        )
        node_update = self.node_dropout(node_candidate)
        if self.residual:
            node_update = node_embedding + node_update
        node_output = self.node_norm(node_update)
        return node_output, edge_output


class GroverLayer(nn.Module):
    """A lightweight GROVER-style message-passing transformer layer.

    This is a from-scratch graph transformer that uses multi-head attention
    over molecular edges and jointly updates node and edge embeddings.  It is
    intentionally self-contained so that the 2D encoder can be trained from
    scratch without loading external GROVER checkpoints, matching the
    manuscript's "all encoders randomly initialized and optimized from
    scratch" description.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 8,
        dropout: float,
        residual: bool,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.residual = _require_boolean(residual, "residual")
        use_layer_norm = _require_boolean(layer_norm, "layer_norm")
        self.num_heads = _positive_integer(num_heads, "num_heads")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        self.head_dim = self.hidden_size // self.num_heads
        dropout_probability = _dropout_probability(dropout)

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.edge_bias = nn.Linear(self.hidden_size, self.num_heads)
        self.edge_self = nn.Linear(self.hidden_size, self.hidden_size)
        self.edge_source = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )
        self.edge_target = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )
        self.node_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.node_attn_norm = (
            nn.LayerNorm(self.hidden_size) if use_layer_norm else nn.Identity()
        )
        self.node_ffn_norm = (
            nn.LayerNorm(self.hidden_size) if use_layer_norm else nn.Identity()
        )
        self.edge_norm = (
            nn.LayerNorm(self.hidden_size) if use_layer_norm else nn.Identity()
        )
        self.node_attn_dropout = nn.Dropout(dropout_probability)
        self.node_ffn_dropout = nn.Dropout(dropout_probability)
        self.edge_dropout = nn.Dropout(dropout_probability)
        self.apply(_reset_linear)

    def forward(
        self,
        node_embedding: Tensor,
        edge_embedding: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        node_count = int(node_embedding.shape[0])

        # Edge features conditioned on source/target node states.
        edge_candidate = F.silu(
            self.edge_self(edge_embedding)
            + self.edge_source(node_embedding[source])
            + self.edge_target(node_embedding[target])
        )
        edge_update = self.edge_dropout(edge_candidate)
        if self.residual:
            edge_update = edge_embedding + edge_update
        edge_output = self.edge_norm(edge_update)

        # Multi-head attention over directed edges.
        query = self.query(node_embedding[source]).view(
            -1, self.num_heads, self.head_dim
        )
        key = self.key(node_embedding[target]).view(
            -1, self.num_heads, self.head_dim
        )
        value = self.value(node_embedding[source]).view(
            -1, self.num_heads, self.head_dim
        )
        scores = (query * key).sum(dim=-1) / (self.head_dim ** 0.5)
        scores = scores + self.edge_bias(edge_output)
        if node_count == 0:
            attention = scores
        else:
            attention = segment_softmax(
                scores,
                target,
                num_nodes=node_count,
            )
        messages = value * attention.unsqueeze(-1)
        attention_output = _scatter_sum(
            messages.reshape(-1, self.hidden_size),
            target.repeat_interleave(self.num_heads),
            node_count,
        )
        attention_output = self.output_proj(attention_output)
        if self.residual:
            node_update = node_embedding + self.node_attn_dropout(attention_output)
        else:
            node_update = self.node_attn_dropout(attention_output)
        node_update = self.node_attn_norm(node_update)
        if self.residual:
            node_update = node_update + self.node_ffn_dropout(
                self.node_ffn(node_update)
            )
        else:
            node_update = self.node_ffn_dropout(self.node_ffn(node_update))
        node_update = self.node_ffn_norm(node_update)
        return node_update, edge_output


class AttentiveGraphReadout(nn.Module):
    """Normalize node attention independently inside each compact graph."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        dropout_probability = _dropout_probability(dropout)
        self.score = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Sequential(
            nn.Dropout(dropout_probability),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.apply(_reset_linear)

    def forward(
        self,
        node_embedding: Tensor,
        node_batch: Tensor,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor]:
        scores = self.score(node_embedding).squeeze(-1)
        values = self.value(node_embedding)
        if num_graphs == 0:
            attention = scores
        else:
            attention = segment_softmax(
                scores,
                node_batch,
                num_nodes=num_graphs,
            )
        pooled_values = _scatter_sum(
            values * attention.unsqueeze(-1),
            node_batch,
            num_graphs,
        )
        output = self.output_norm(
            pooled_values + self.output(pooled_values)
        )
        return output, attention


class GraphEncoder(nn.Module):
    """Encode a compact PyG molecular batch with GatedGCN, GINE, or GROVER."""

    SUPPORTED_ENCODERS = frozenset({"gatedgcn", "gine", "grover"})

    def __init__(
        self,
        *,
        encoder_type: str = "grover",
        node_feature_cardinalities: Sequence[
            int
        ] = MASKED_OGB_ATOM_FEATURE_CARDINALITIES,
        edge_feature_cardinalities: Sequence[
            int
        ] = MASKED_OGB_BOND_FEATURE_CARDINALITIES,
        hidden_size: int = 512,
        num_layers: int = 5,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual: bool = True,
        layer_norm: bool = True,
        train_eps: bool = True,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(encoder_type, str) or not encoder_type.strip():
            raise ValueError("encoder_type must be a non-empty string")
        normalized_type = encoder_type.strip().lower()
        if normalized_type == "gin":
            raise ValueError(
                "plain GIN ignores required bond features; use encoder_type='gine'"
            )
        if normalized_type not in self.SUPPORTED_ENCODERS:
            raise ValueError(
                f"unsupported encoder_type={encoder_type!r}; expected one of "
                f"{sorted(self.SUPPORTED_ENCODERS)}"
            )

        self.encoder_type = normalized_type
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        layer_count = _positive_integer(num_layers, "num_layers")
        head_count = _positive_integer(num_heads, "num_heads")
        if normalized_type == "grover" and self.hidden_size % head_count != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({head_count}) for encoder_type='grover'"
            )
        dropout_probability = _dropout_probability(dropout)
        use_residual = _require_boolean(residual, "residual")
        use_layer_norm = _require_boolean(layer_norm, "layer_norm")
        use_train_eps = _require_boolean(train_eps, "train_eps")
        self.validate_values = _require_boolean(
            validate_values,
            "validate_values",
        )
        self.node_feature_cardinalities = _cardinalities(
            node_feature_cardinalities,
            name="node_feature_cardinalities",
            expected=MASKED_OGB_ATOM_FEATURE_CARDINALITIES,
        )
        self.edge_feature_cardinalities = _cardinalities(
            edge_feature_cardinalities,
            name="edge_feature_cardinalities",
            expected=MASKED_OGB_BOND_FEATURE_CARDINALITIES,
        )

        self.node_features = CategoricalFeatureEncoder(
            self.node_feature_cardinalities,
            self.hidden_size,
            feature_name="graph.x",
            validate_values=self.validate_values,
        )
        self.edge_features = CategoricalFeatureEncoder(
            self.edge_feature_cardinalities,
            self.hidden_size,
            feature_name="graph.edge_attr",
            validate_values=self.validate_values,
        )
        self.node_input_norm = nn.LayerNorm(self.hidden_size)
        self.edge_input_norm = nn.LayerNorm(self.hidden_size)
        self.input_dropout = nn.Dropout(dropout_probability)

        if self.encoder_type == "gatedgcn":
            self.layers = nn.ModuleList(
                GatedGCNLayer(
                    self.hidden_size,
                    dropout=dropout_probability,
                    residual=use_residual,
                    layer_norm=use_layer_norm,
                )
                for _ in range(layer_count)
            )
        elif self.encoder_type == "grover":
            self.layers = nn.ModuleList(
                GroverLayer(
                    self.hidden_size,
                    num_heads=head_count,
                    dropout=dropout_probability,
                    residual=use_residual,
                    layer_norm=use_layer_norm,
                )
                for _ in range(layer_count)
            )
        else:
            self.layers = nn.ModuleList(
                GINELayer(
                    self.hidden_size,
                    dropout=dropout_probability,
                    residual=use_residual,
                    layer_norm=use_layer_norm,
                    train_eps=use_train_eps,
                )
                for _ in range(layer_count)
            )
        self.readout = AttentiveGraphReadout(
            self.hidden_size, dropout_probability
        )

    def _validate_inputs(
        self,
        graph: Batch,
        graph_sample_index: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        if not isinstance(graph, Batch):
            raise TypeError("graph must be a torch_geometric.data.Batch")
        if not isinstance(graph_sample_index, Tensor):
            raise TypeError("graph_sample_index must be a torch.Tensor")
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        batch_size = int(batch_size)
        if batch_size < 0:
            raise ValueError("batch_size cannot be negative")

        compact_size = int(graph_sample_index.numel())
        validate_sample_index(
            graph_sample_index,
            compact_size=compact_size,
            batch_size=batch_size,
            check_values=self.validate_values,
        )
        if compact_size == 0:
            graph_ptr = getattr(graph, "ptr", None)
            if graph_ptr is not None:
                if (
                    not isinstance(graph_ptr, Tensor)
                    or graph_ptr.ndim != 1
                    or graph_ptr.dtype != torch.long
                    or graph_ptr.numel() != 1
                    or int(graph_ptr[0]) != 0
                ):
                    raise ValueError(
                        "an empty compact batch may only use graph.ptr=[0]"
                    )
        parameter_device = self.node_features.embeddings[0].weight.device
        if graph_sample_index.device != parameter_device:
            raise ValueError(
                f"graph_sample_index is on {graph_sample_index.device}, but "
                f"the encoder is on {parameter_device}"
            )

        required_attributes = ("x", "edge_index", "edge_attr")
        missing_attributes = tuple(
            attribute
            for attribute in required_attributes
            if getattr(graph, attribute, None) is None
        )
        if compact_size == 0 and missing_attributes:
            declared_node_count = graph.num_nodes
            if (
                declared_node_count is not None
                and int(declared_node_count) != 0
            ):
                raise ValueError(
                    "an empty graph_sample_index requires graph.num_nodes == 0"
                )
            for attribute in required_attributes + ("batch",):
                value = getattr(graph, attribute, None)
                if value is not None:
                    if not isinstance(value, Tensor):
                        raise TypeError(
                            f"empty graph attribute {attribute!r} must be a tensor"
                        )
                    if value.numel() != 0:
                        raise ValueError(
                            "an empty graph_sample_index cannot accompany "
                            f"non-empty graph attribute {attribute!r}"
                        )
            node_features = torch.empty(
                (0, NODE_FEATURE_DIM),
                dtype=torch.long,
                device=parameter_device,
            )
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=parameter_device,
            )
            edge_features = torch.empty(
                (0, EDGE_FEATURE_DIM),
                dtype=torch.long,
                device=parameter_device,
            )
            node_batch = torch.empty(
                (0,),
                dtype=torch.long,
                device=parameter_device,
            )
            return (
                node_features,
                edge_index,
                edge_features,
                node_batch,
                compact_size,
            )
        if missing_attributes:
            raise ValueError(
                f"graph is missing required attributes {missing_attributes}"
            )

        node_features = graph.x
        edge_index = graph.edge_index
        edge_features = graph.edge_attr
        if not isinstance(node_features, Tensor):
            raise TypeError("graph.x must be a torch.Tensor")
        if not isinstance(edge_index, Tensor):
            raise TypeError("graph.edge_index must be a torch.Tensor")
        if not isinstance(edge_features, Tensor):
            raise TypeError("graph.edge_attr must be a torch.Tensor")

        if node_features.ndim != 2 or node_features.shape[1] != NODE_FEATURE_DIM:
            raise ValueError(
                f"graph.x must have shape [nodes, {NODE_FEATURE_DIM}], got "
                f"{tuple(node_features.shape)}"
            )
        if node_features.dtype != torch.long:
            raise TypeError(f"graph.x must be torch.long, got {node_features.dtype}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "graph.edge_index must have shape [2, edges], got "
                f"{tuple(edge_index.shape)}"
            )
        if edge_index.dtype != torch.long:
            raise TypeError(
                f"graph.edge_index must be torch.long, got {edge_index.dtype}"
            )
        if (
            edge_features.ndim != 2
            or edge_features.shape[1] != EDGE_FEATURE_DIM
        ):
            raise ValueError(
                f"graph.edge_attr must have shape [edges, {EDGE_FEATURE_DIM}], "
                f"got {tuple(edge_features.shape)}"
            )
        if edge_features.dtype != torch.long:
            raise TypeError(
                f"graph.edge_attr must be torch.long, got {edge_features.dtype}"
            )
        if edge_features.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "graph.edge_attr row count must equal graph.edge_index edge count"
            )

        devices = {
            node_features.device,
            edge_index.device,
            edge_features.device,
            graph_sample_index.device,
        }
        if len(devices) != 1:
            raise ValueError(
                "graph tensors and graph_sample_index must be on the same device"
            )
        if node_features.device != parameter_device:
            raise ValueError(
                f"graph tensors are on {node_features.device}, but the encoder "
                f"is on {parameter_device}"
            )

        node_count = int(node_features.shape[0])
        edge_count = int(edge_index.shape[1])
        if graph.num_nodes is None or int(graph.num_nodes) != node_count:
            raise ValueError(
                "graph.num_nodes must equal the number of rows in graph.x"
            )
        node_batch = getattr(graph, "batch", None)
        if node_batch is None:
            if node_count != 0:
                raise ValueError("graph.batch is required for a non-empty Batch")
            node_batch = edge_index.new_empty((0,))
        if not isinstance(node_batch, Tensor):
            raise TypeError("graph.batch must be a torch.Tensor")
        if node_batch.ndim != 1 or node_batch.numel() != node_count:
            raise ValueError("graph.batch must have one graph index per node")
        if node_batch.dtype != torch.long:
            raise TypeError(
                f"graph.batch must be torch.long, got {node_batch.dtype}"
            )
        if node_batch.device != node_features.device:
            raise ValueError("graph.batch and graph.x must be on the same device")

        if compact_size == 0:
            if node_count != 0 or edge_count != 0:
                raise ValueError(
                    "an empty graph_sample_index requires an empty graph Batch"
                )
            return (
                node_features,
                edge_index,
                edge_features,
                node_batch,
                compact_size,
            )

        if node_count == 0:
            raise ValueError("every compact molecular graph must contain an atom")
        if int(graph.num_graphs) != compact_size:
            raise ValueError(
                "graph.num_graphs must equal graph_sample_index length: "
                f"{int(graph.num_graphs)} != {compact_size}"
            )
        if self.validate_values:
            if bool(torch.any(node_batch < 0)) or bool(
                torch.any(node_batch >= compact_size)
            ):
                raise ValueError(
                    "graph.batch contains an invalid compact graph index"
                )
            if node_batch.numel() > 1 and bool(
                torch.any(node_batch[1:] < node_batch[:-1])
            ):
                raise ValueError("graph.batch must be grouped in graph-index order")
            graph_node_counts = torch.bincount(
                node_batch, minlength=compact_size
            )
            if bool(torch.any(graph_node_counts == 0)):
                raise ValueError(
                    "every compact graph must contain at least one node"
                )

            if edge_count > 0:
                if bool(torch.any(edge_index < 0)) or bool(
                    torch.any(edge_index >= node_count)
                ):
                    raise ValueError(
                        "graph.edge_index contains an invalid node index"
                    )
                source, target = edge_index
                if bool(torch.any(node_batch[source] != node_batch[target])):
                    raise ValueError(
                        "graph.edge_index cannot connect different graphs"
                    )
        return (
            node_features,
            edge_index,
            edge_features,
            node_batch,
            compact_size,
        )

    def forward(
        self,
        graph: Batch,
        graph_sample_index: Tensor,
        *,
        batch_size: int,
    ) -> GraphEncoderOutput:
        (
            node_features,
            edge_index,
            edge_features,
            node_batch,
            compact_size,
        ) = self._validate_inputs(graph, graph_sample_index, batch_size)

        node_embedding = self.input_dropout(
            self.node_input_norm(self.node_features(node_features))
        )
        edge_embedding = self.input_dropout(
            self.edge_input_norm(self.edge_features(edge_features))
        )
        for layer in self.layers:
            node_embedding, edge_embedding = layer(
                node_embedding,
                edge_embedding,
                edge_index,
            )
            if self.validate_values:
                _require_finite(node_embedding, "graph node embedding")
                _require_finite(edge_embedding, "graph edge embedding")

        global_embedding, node_attention = self.readout(
            node_embedding,
            node_batch,
            compact_size,
        )
        if self.validate_values:
            _require_finite(global_embedding, "graph global embedding")
            _require_finite(node_attention, "graph readout attention")

        if compact_size == 0:
            tokens = node_embedding.new_empty((0, 0, self.hidden_size))
            token_mask = torch.empty(
                (0, 0),
                dtype=torch.bool,
                device=node_embedding.device,
            )
        else:
            tokens, token_mask = to_dense_batch(
                node_embedding,
                node_batch,
                batch_size=compact_size,
            )
        edge_batch = node_batch[edge_index[0]]

        output = GraphEncoderOutput(
            global_embedding=global_embedding,
            sample_index=graph_sample_index,
            tokens=tokens,
            token_mask=token_mask,
            node_embedding=node_embedding,
            edge_embedding=edge_embedding,
            node_batch=node_batch,
            edge_batch=edge_batch,
        )
        validate_encoder_output(
            output,
            embedding_dim=self.hidden_size,
            batch_size=batch_size,
            check_values=self.validate_values,
        )
        if output.node_embedding.ndim != 2 or output.node_embedding.shape[1] != (
            self.hidden_size
        ):
            raise RuntimeError("node embedding shape invariant was violated")
        if output.edge_embedding.ndim != 2 or output.edge_embedding.shape[1] != (
            self.hidden_size
        ):
            raise RuntimeError("edge embedding shape invariant was violated")
        if output.node_batch.numel() != output.node_embedding.shape[0]:
            raise RuntimeError("node_batch length invariant was violated")
        if output.edge_batch.numel() != output.edge_embedding.shape[0]:
            raise RuntimeError("edge_batch length invariant was violated")
        return output

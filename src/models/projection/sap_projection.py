"""Token-aware semantic attention projection."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .unified_projection import ProjectionOutput, normalize_projection


class SemanticAttentionProjection(nn.Module):
    """Project real local features through learned semantic cross-attention queries."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        *,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_semantic_tokens: int = 16,
        dropout: float = 0.1,
        feedforward_multiplier: int = 4,
        layer_norm_eps: float = 1.0e-5,
        learnable_temperature: bool = True,
        temperature_init: float = 1.0,
        normalize_eps: float = 1.0e-8,
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if num_semantic_tokens <= 0:
            raise ValueError("num_semantic_tokens must be positive")
        if feedforward_multiplier <= 0:
            raise ValueError("feedforward_multiplier must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if layer_norm_eps <= 0.0 or normalize_eps <= 0.0:
            raise ValueError("normalization epsilons must be positive")
        if temperature_init <= 0.0:
            raise ValueError("temperature_init must be positive")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_semantic_tokens = num_semantic_tokens
        self.normalize_eps = normalize_eps
        self.learnable_temperature = learnable_temperature
        self.validate_values = validate_values

        self.semantic_queries = nn.Parameter(
            torch.empty(num_semantic_tokens, hidden_dim)
        )
        self.query_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.token_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=input_dim,
            vdim=input_dim,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(dropout)

        feedforward_dim = hidden_dim * feedforward_multiplier
        self.feedforward_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
        )
        self.feedforward_dropout = nn.Dropout(dropout)

        self.pool_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.output = nn.Linear(hidden_dim, output_dim)

        initial_log_temperature = math.log(temperature_init)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(initial_log_temperature, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(initial_log_temperature, dtype=torch.float32),
                persistent=True,
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.semantic_queries,
            mean=0.0,
            std=self.hidden_dim**-0.5,
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> ProjectionOutput:
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must have shape [compact_batch, length, input_dim], got {tuple(tokens.shape)}"
            )
        if tokens.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected token input_dim={self.input_dim}, got {tokens.shape[2]}"
            )
        if tuple(token_mask.shape) != tuple(tokens.shape[:2]):
            raise ValueError(
                f"token_mask must have shape {tuple(tokens.shape[:2])}, "
                f"got {tuple(token_mask.shape)}"
            )
        if token_mask.dtype != torch.bool:
            raise TypeError(f"token_mask must be bool, got {token_mask.dtype}")
        if tokens.device != token_mask.device:
            raise ValueError("tokens and token_mask must be on the same device")
        if not tokens.is_floating_point():
            raise TypeError(f"tokens must be floating point, got {tokens.dtype}")

        compact_size = tokens.shape[0]
        if compact_size == 0:
            raw = tokens.new_empty((0, self.output_dim))
            normalized = raw.float()
            return ProjectionOutput(raw=raw, normalized=normalized)
        if tokens.shape[1] == 0:
            raise ValueError("SAP requires at least one real token for every sample")
        if bool(torch.any(~token_mask.any(dim=1))):
            raise ValueError("SAP requires at least one real token for every sample")
        if self.validate_values:
            valid_tokens = tokens[token_mask]
            if valid_tokens.numel() > 0 and not bool(
                torch.isfinite(valid_tokens).all()
            ):
                raise ValueError("valid tokens contain NaN or infinite values")

        queries = self.semantic_queries.unsqueeze(0).expand(compact_size, -1, -1)
        normalized_queries = self.query_norm(queries)
        safe_tokens = tokens.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        normalized_tokens = self.token_norm(safe_tokens)
        attended, _ = self.cross_attention(
            query=normalized_queries,
            key=normalized_tokens,
            value=normalized_tokens,
            key_padding_mask=~token_mask,
            need_weights=False,
        )
        semantic_state = queries + self.cross_dropout(attended)
        semantic_state = semantic_state + self.feedforward_dropout(
            self.feedforward(self.feedforward_norm(semantic_state))
        )

        temperature = self.log_temperature.float().clamp(
            min=math.log(1.0e-4),
            max=math.log(1.0e4),
        ).exp()
        pool_logits = self.pool_score(self.pool_norm(semantic_state)).squeeze(-1)
        pool_weights = torch.softmax(pool_logits.float() / temperature, dim=1)
        pooled = torch.sum(
            pool_weights.to(dtype=semantic_state.dtype).unsqueeze(-1)
            * semantic_state,
            dim=1,
        )
        raw = self.output(pooled)
        normalized = normalize_projection(raw, self.normalize_eps)
        return ProjectionOutput(raw=raw, normalized=normalized)

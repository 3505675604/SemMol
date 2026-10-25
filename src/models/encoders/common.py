"""Shared contracts and validation helpers for modality encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EncoderOutput:
    """Compact encoder output containing only samples with a present modality.

    ``sample_index`` maps the compact first dimension back to the original
    multimodal batch. ``tokens`` are real local features consumed by semantic
    attention; padding positions are marked false in ``token_mask``.
    """

    global_embedding: Tensor
    sample_index: Tensor
    tokens: Tensor
    token_mask: Tensor


def compact_sample_index(valid_mask: Tensor) -> Tensor:
    """Return monotonically ordered source-row indices selected by a boolean mask."""

    if valid_mask.ndim != 1:
        raise ValueError(
            f"valid_mask must have shape [batch], got {tuple(valid_mask.shape)}"
        )
    if valid_mask.dtype != torch.bool:
        raise TypeError(f"valid_mask must be bool, got {valid_mask.dtype}")
    return torch.nonzero(valid_mask, as_tuple=False).flatten()


def validate_sample_index(
    sample_index: Tensor,
    *,
    compact_size: int,
    batch_size: int | None = None,
    check_values: bool = True,
) -> None:
    """Validate a compact-to-full batch index and its ordering."""

    if not isinstance(check_values, bool):
        raise TypeError("check_values must be bool")
    if sample_index.ndim != 1:
        raise ValueError(
            f"sample_index must have shape [compact_batch], got {tuple(sample_index.shape)}"
        )
    if sample_index.dtype != torch.long:
        raise TypeError(f"sample_index must be torch.long, got {sample_index.dtype}")
    if sample_index.numel() != compact_size:
        raise ValueError(
            "sample_index length must match compact batch size: "
            f"{sample_index.numel()} != {compact_size}"
        )
    if check_values:
        if sample_index.numel() > 1 and not bool(
            torch.all(sample_index[1:] > sample_index[:-1])
        ):
            raise ValueError("sample_index must be strictly increasing and unique")
        if sample_index.numel() > 0:
            if bool(torch.any(sample_index < 0)):
                raise ValueError("sample_index cannot contain negative indices")
            if batch_size is not None and bool(torch.any(sample_index >= batch_size)):
                raise ValueError(
                    f"sample_index contains an index outside batch_size={batch_size}"
                )


def validate_encoder_output(
    output: EncoderOutput,
    *,
    embedding_dim: int,
    batch_size: int | None = None,
    check_values: bool = True,
) -> None:
    """Check the common shape and dtype invariants of an encoder output."""

    if not isinstance(check_values, bool):
        raise TypeError("check_values must be bool")
    if output.global_embedding.ndim != 2:
        raise ValueError(
            "global_embedding must have shape [compact_batch, dim], got "
            f"{tuple(output.global_embedding.shape)}"
        )
    compact_size, actual_dim = output.global_embedding.shape
    if actual_dim != embedding_dim:
        raise ValueError(
            f"global_embedding dim must be {embedding_dim}, got {actual_dim}"
        )
    validate_sample_index(
        output.sample_index,
        compact_size=compact_size,
        batch_size=batch_size,
        check_values=check_values,
    )
    if output.tokens.ndim != 3:
        raise ValueError(
            f"tokens must have shape [compact_batch, length, dim], got {tuple(output.tokens.shape)}"
        )
    if output.tokens.shape[0] != compact_size or output.tokens.shape[2] != embedding_dim:
        raise ValueError(
            "tokens must agree with the compact batch and embedding dimensions; "
            f"got {tuple(output.tokens.shape)}"
        )
    expected_mask_shape = output.tokens.shape[:2]
    if tuple(output.token_mask.shape) != tuple(expected_mask_shape):
        raise ValueError(
            f"token_mask must have shape {tuple(expected_mask_shape)}, "
            f"got {tuple(output.token_mask.shape)}"
        )
    if output.token_mask.dtype != torch.bool:
        raise TypeError(f"token_mask must be bool, got {output.token_mask.dtype}")
    if (
        check_values
        and compact_size > 0
        and bool(torch.any(~output.token_mask.any(dim=1)))
    ):
        raise ValueError("every compact sample must contain at least one valid token")
    if output.global_embedding.device != output.sample_index.device:
        raise ValueError("global_embedding and sample_index must be on the same device")
    if output.tokens.device != output.global_embedding.device:
        raise ValueError("tokens and global_embedding must be on the same device")
    if output.token_mask.device != output.global_embedding.device:
        raise ValueError("token_mask and global_embedding must be on the same device")
    if not output.global_embedding.is_floating_point():
        raise TypeError("global_embedding must be floating point")
    if not output.tokens.is_floating_point():
        raise TypeError("tokens must be floating point")
    if output.tokens.dtype != output.global_embedding.dtype:
        raise TypeError("tokens and global_embedding must have the same dtype")
    if check_values:
        if output.global_embedding.numel() > 0 and not bool(
            torch.isfinite(output.global_embedding).all()
        ):
            raise ValueError("global_embedding contains NaN or infinite values")
        valid_tokens = output.tokens[output.token_mask]
        if valid_tokens.numel() > 0 and not bool(torch.isfinite(valid_tokens).all()):
            raise ValueError("valid tokens contain NaN or infinite values")

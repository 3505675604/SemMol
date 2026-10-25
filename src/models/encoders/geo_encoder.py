"""Sparse multi-conformer geometry encoder backed by PyG DimeNet."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.nn.models import DimeNet
from torch_geometric.utils import to_dense_batch

from .common import (
    EncoderOutput,
    compact_sample_index,
    validate_encoder_output,
)


@dataclass(frozen=True)
class GeometryEncoderOutput(EncoderOutput):
    """DimeNet features with the learned weight of each valid conformer."""

    conformer_weights: Tensor


class GeoEncoder(nn.Module):
    """Encode all valid conformers as independent sparse DimeNet graphs."""

    _MAX_ATOMIC_NUMBER = 118
    _NUM_ATOM_EMBEDDINGS = _MAX_ATOMIC_NUMBER + 1

    def __init__(
        self,
        hidden_size: int = 128,
        num_blocks: int = 6,
        num_bilinear: int = 8,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        target_dim: int = 512,
        dropout: float = 0.1,
        max_num_neighbors: int = 32,
        conformer_pooling: Literal["attention", "mean"] = "attention",
        validate_values: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or target_dim <= 0:
            raise ValueError("hidden_size and target_dim must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if max_num_neighbors <= 0:
            raise ValueError("max_num_neighbors must be positive")
        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("cutoff must be a positive finite value")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if conformer_pooling not in {"attention", "mean"}:
            raise ValueError(
                "conformer_pooling must be either 'attention' or 'mean'"
            )
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")

        self.hidden_size = int(hidden_size)
        self.target_dim = int(target_dim)
        self.conformer_pooling = conformer_pooling
        self.validate_values = validate_values

        self.dimenet = DimeNet(
            hidden_channels=self.hidden_size,
            out_channels=self.hidden_size,
            num_blocks=num_blocks,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=float(cutoff),
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act="swish",
            output_initializer="glorot_orthogonal",
        )
        self._expand_atomic_number_embedding()

        self.composition_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.SiLU(),
        )
        self.conformer_adapter = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.target_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        attention_hidden = max(self.target_dim // 2, 32)
        self.conformer_attention = nn.Sequential(
            nn.LayerNorm(self.target_dim),
            nn.Linear(self.target_dim, attention_hidden),
            nn.SiLU(),
            nn.Linear(attention_hidden, 1, bias=False),
        )

    def _expand_atomic_number_embedding(self) -> None:
        """Extend PyG 2.4 DimeNet's Z=0..94 table through Z=118."""

        embedding_block = getattr(self.dimenet, "emb", None)
        old_embedding = getattr(embedding_block, "emb", None)
        if not isinstance(old_embedding, nn.Embedding):
            raise RuntimeError(
                "PyG DimeNet atom embedding layout is incompatible with "
                "the pinned torch-geometric 2.4 implementation"
            )
        if old_embedding.embedding_dim != self.hidden_size:
            raise RuntimeError(
                "DimeNet atom embedding dimension does not match hidden_size"
            )
        if old_embedding.num_embeddings > self._NUM_ATOM_EMBEDDINGS:
            raise RuntimeError(
                "DimeNet unexpectedly provides more than 119 atom embeddings"
            )
        if old_embedding.num_embeddings == self._NUM_ATOM_EMBEDDINGS:
            return

        expanded = nn.Embedding(
            self._NUM_ATOM_EMBEDDINGS,
            self.hidden_size,
            device=old_embedding.weight.device,
            dtype=old_embedding.weight.dtype,
        )
        with torch.no_grad():
            expanded.weight.uniform_(-math.sqrt(3.0), math.sqrt(3.0))
            expanded.weight[: old_embedding.num_embeddings].copy_(
                old_embedding.weight
            )
        embedding_block.emb = expanded

    def _validate_inputs(
        self,
        atomic_numbers: Tensor,
        coords: Tensor,
        atom_mask: Tensor,
        conformer_mask: Tensor,
    ) -> None:
        if atomic_numbers.ndim != 2:
            raise ValueError(
                "atomic_numbers must have shape [batch, atoms], got "
                f"{tuple(atomic_numbers.shape)}"
            )
        if atomic_numbers.dtype != torch.long:
            raise TypeError(
                f"atomic_numbers must be torch.long, got {atomic_numbers.dtype}"
            )
        if coords.ndim != 4 or coords.shape[-1] != 3:
            raise ValueError(
                "coords must have shape [batch, conformers, atoms, 3], got "
                f"{tuple(coords.shape)}"
            )
        if coords.dtype != torch.float32:
            raise TypeError(f"coords must be torch.float32, got {coords.dtype}")
        batch_size, atom_count = atomic_numbers.shape
        if coords.shape[0] != batch_size or coords.shape[2] != atom_count:
            raise ValueError(
                "coords batch/atom dimensions must match atomic_numbers"
            )
        expected_atom_mask = (batch_size, atom_count)
        if tuple(atom_mask.shape) != expected_atom_mask:
            raise ValueError(
                f"atom_mask must have shape {expected_atom_mask}, got "
                f"{tuple(atom_mask.shape)}"
            )
        expected_conformer_mask = (batch_size, coords.shape[1])
        if tuple(conformer_mask.shape) != expected_conformer_mask:
            raise ValueError(
                f"conformer_mask must have shape {expected_conformer_mask}, got "
                f"{tuple(conformer_mask.shape)}"
            )
        if atom_mask.dtype != torch.bool:
            raise TypeError(f"atom_mask must be bool, got {atom_mask.dtype}")
        if conformer_mask.dtype != torch.bool:
            raise TypeError(
                f"conformer_mask must be bool, got {conformer_mask.dtype}"
            )
        devices = {
            atomic_numbers.device,
            coords.device,
            atom_mask.device,
            conformer_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("all geometry inputs must be on the same device")
        parameter_device = self.dimenet.emb.emb.weight.device
        if coords.device != parameter_device:
            raise ValueError(
                f"geometry inputs are on {coords.device}, but GeoEncoder is on "
                f"{parameter_device}"
            )
        if self.validate_values:
            if not bool(torch.isfinite(coords).all()):
                raise ValueError("coords must contain only finite values")
            if bool(torch.any(atomic_numbers[~atom_mask] != 0)):
                raise ValueError(
                    "atomic_numbers outside atom_mask must use padding value 0"
                )
            valid_atomic_numbers = atomic_numbers[atom_mask]
            if valid_atomic_numbers.numel() > 0 and (
                bool(torch.any(valid_atomic_numbers < 1))
                or bool(
                    torch.any(
                        valid_atomic_numbers > GeoEncoder._MAX_ATOMIC_NUMBER
                    )
                )
            ):
                raise ValueError("valid atomic_numbers must be in [1, 118]")
            has_atoms = atom_mask.any(dim=1)
            has_conformers = conformer_mask.any(dim=1)
            if not torch.equal(has_atoms, has_conformers):
                raise ValueError(
                    "a sample must have both valid atoms and at least one valid "
                    "conformer, or have neither"
                )

    def _empty_output(
        self,
        coords: Tensor,
        sample_index: Tensor,
        batch_size: int,
    ) -> GeometryEncoderOutput:
        output = GeometryEncoderOutput(
            global_embedding=coords.new_empty((0, self.target_dim)),
            sample_index=sample_index,
            tokens=coords.new_empty((0, 0, self.target_dim)),
            token_mask=torch.empty(
                (0, 0),
                dtype=torch.bool,
                device=coords.device,
            ),
            conformer_weights=coords.new_empty((0, 0)),
        )
        validate_encoder_output(
            output,
            embedding_dim=self.target_dim,
            batch_size=batch_size,
            check_values=self.validate_values,
        )
        return output

    def forward(
        self,
        atomic_numbers: Tensor,
        coords: Tensor,
        atom_mask: Tensor,
        conformer_mask: Tensor,
    ) -> GeometryEncoderOutput:
        """Return compact molecule features and real conformer token features."""

        self._validate_inputs(
            atomic_numbers,
            coords,
            atom_mask,
            conformer_mask,
        )
        batch_size = int(atomic_numbers.shape[0])
        present_mask = conformer_mask.any(dim=1)
        sample_index = compact_sample_index(present_mask)
        if sample_index.numel() == 0:
            return self._empty_output(coords, sample_index, batch_size)

        valid_pairs = torch.nonzero(conformer_mask, as_tuple=False)
        pair_sample_index = valid_pairs[:, 0]
        pair_conformer_index = valid_pairs[:, 1]
        pair_atom_mask = atom_mask.index_select(0, pair_sample_index)
        pair_atomic_numbers = atomic_numbers.index_select(
            0, pair_sample_index
        )
        pair_coords = coords[
            pair_sample_index,
            pair_conformer_index,
        ]

        flat_atomic_numbers = pair_atomic_numbers[pair_atom_mask]
        flat_coords = pair_coords[pair_atom_mask]
        atom_counts = pair_atom_mask.sum(dim=1)
        conformer_batch = torch.repeat_interleave(
            torch.arange(
                valid_pairs.shape[0],
                device=coords.device,
                dtype=torch.long,
            ),
            atom_counts,
            output_size=int(flat_atomic_numbers.shape[0]),
        )

        conformer_features = self.dimenet(
            flat_atomic_numbers,
            flat_coords,
            conformer_batch,
        )
        if conformer_features.ndim != 2 or conformer_features.shape != (
            valid_pairs.shape[0],
            self.hidden_size,
        ):
            raise RuntimeError(
                "DimeNet returned an unexpected conformer feature shape: "
                f"{tuple(conformer_features.shape)}"
            )
        atom_features = self.dimenet.emb.emb(flat_atomic_numbers)
        composition_sum = atom_features.new_zeros(
            (valid_pairs.shape[0], self.hidden_size)
        ).index_add(0, conformer_batch, atom_features)
        composition_mean = composition_sum / atom_counts.to(
            dtype=composition_sum.dtype
        ).unsqueeze(-1).clamp_min(1.0)
        conformer_features = conformer_features + self.composition_projection(
            composition_mean
        )
        conformer_features = self.conformer_adapter(conformer_features)

        conformer_counts = conformer_mask.index_select(
            0, sample_index
        ).sum(dim=1)
        compact_conformer_batch = torch.repeat_interleave(
            torch.arange(
                sample_index.numel(),
                device=coords.device,
                dtype=torch.long,
            ),
            conformer_counts,
            output_size=int(valid_pairs.shape[0]),
        )
        tokens, token_mask = to_dense_batch(
            conformer_features,
            compact_conformer_batch,
            batch_size=int(sample_index.numel()),
        )

        if self.conformer_pooling == "attention":
            scores = self.conformer_attention(tokens).squeeze(-1).float()
            scores = scores.masked_fill(
                ~token_mask,
                torch.finfo(scores.dtype).min,
            )
            conformer_weights = torch.softmax(scores, dim=1).to(tokens.dtype)
        else:
            conformer_weights = token_mask.to(tokens.dtype)
            conformer_weights = conformer_weights / conformer_weights.sum(
                dim=1,
                keepdim=True,
            )
        global_embedding = torch.sum(
            tokens.float() * conformer_weights.float().unsqueeze(-1),
            dim=1,
        ).to(dtype=tokens.dtype)

        output = GeometryEncoderOutput(
            global_embedding=global_embedding,
            sample_index=sample_index,
            tokens=tokens,
            token_mask=token_mask,
            conformer_weights=conformer_weights,
        )
        validate_encoder_output(
            output,
            embedding_dim=self.target_dim,
            batch_size=batch_size,
            check_values=self.validate_values,
        )
        return output

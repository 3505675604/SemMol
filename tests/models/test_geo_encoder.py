from __future__ import annotations

import pytest
import torch

from src.models.encoders.geo_encoder import GeoEncoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_attention_pooling_preserves_token_dtype_under_bfloat16_autocast() -> None:
    device = torch.device("cuda:0")
    encoder = GeoEncoder(
        hidden_size=16,
        num_blocks=1,
        num_bilinear=2,
        num_spherical=2,
        num_radial=2,
        num_before_skip=1,
        num_after_skip=1,
        num_output_layers=1,
        target_dim=16,
        dropout=0.0,
        max_num_neighbors=8,
        conformer_pooling="attention",
    ).to(device)
    atomic_numbers = torch.tensor([[6, 6, 8]], dtype=torch.long, device=device)
    coords = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]],
                [[0.0, 0.0, 0.1], [1.1, 0.1, 0.0], [0.1, 1.2, 0.0]],
            ],
        ],
        dtype=torch.float32,
        device=device,
    )
    atom_mask = torch.ones((1, 3), dtype=torch.bool, device=device)
    conformer_mask = torch.ones((1, 2), dtype=torch.bool, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = encoder(
            atomic_numbers,
            coords,
            atom_mask,
            conformer_mask,
        )

    assert output.tokens.dtype == torch.bfloat16
    assert output.global_embedding.dtype == output.tokens.dtype

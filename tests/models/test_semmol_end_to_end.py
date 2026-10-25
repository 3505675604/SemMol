"""A real tiny molecular batch exercises all default model and loss branches."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("transformers")
pytest.importorskip("rdkit")

from src.datasets.collator import MultimodalDataCollator, PretrainingDataCollator
from src.losses.total_loss import SemMolPretrainTotalLoss
from src.models.semmol import SemMol
from src.molecular.espf_tokenizer import ESPFTokenizer
from src.molecular.geometry import generate_conformers
from src.molecular.graph import smiles_to_pyg_graph


def test_real_tokenizer_graph_geometry_pretrain_and_downstream_backward(tmp_path):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    smiles = ["CCO", "CCN"]
    tokenizer = ESPFTokenizer.train(smiles, min_frequency=1, max_merges=2)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    samples = []
    for index, text in enumerate(smiles):
        geometry = generate_conformers(text, num_conformers=1, seed=17 + index, on_invalid="raise")
        encoded = tokenizer.encode_plus(geometry.canonical_smiles, max_length=None)
        samples.append({
            "sample_id": f"unit:{index}", "source_index": index,
            "smiles": geometry.canonical_smiles,
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "token_spans": torch.tensor(encoded["token_spans"], dtype=torch.long),
            "graph": smiles_to_pyg_graph(geometry.canonical_smiles, on_invalid="raise"),
            "atomic_numbers": torch.as_tensor(geometry.atomic_numbers.copy(), dtype=torch.long),
            "coords": torch.as_tensor(geometry.coords.copy(), dtype=torch.float32),
            "conformer_mask": torch.as_tensor(geometry.conformer_mask.copy(), dtype=torch.bool),
            "energies": torch.as_tensor(geometry.energies.copy(), dtype=torch.float32),
            "energy_mask": torch.as_tensor(geometry.conformer_mask.copy(), dtype=torch.bool),
            "heavy_atom_indices": torch.as_tensor(geometry.heavy_atom_indices.copy(), dtype=torch.long),
            "conformer_sources": tuple(str(value) for value in geometry.conformer_source),
        })
    collator = PretrainingDataCollator(
        pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size, special_token_ids=(0, 1, 2, 3, 4),
        smiles_mask_ratio=1.0, node_mask_ratio=1.0, edge_mask_ratio=1.0,
        geo_noise_std=0.05, graph_mask_mode="substructure", seed=17,
    )
    batch = collator(samples)
    options = dict(
        encoders={
            "shared_dim": 8,
            "smiles": {"tokenizer_dir": str(tokenizer_dir), "shared_dim": 8,
                       "scratch_hidden_size": 8, "scratch_num_hidden_layers": 1,
                       "scratch_num_attention_heads": 2, "scratch_intermediate_size": 16,
                       "scratch_max_position_embeddings": 64,
                       "scratch_hidden_dropout_prob": 0.0, "scratch_attention_probs_dropout_prob": 0.0},
            "graph": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0},
            "geo": {"hidden_size": 8, "target_dim": 8, "num_blocks": 1,
                    "num_bilinear": 2, "num_spherical": 2, "num_radial": 2,
                    "num_before_skip": 1, "num_after_skip": 1, "num_output_layers": 1,
                    "max_num_neighbors": 16, "conformer_pooling": "mean", "dropout": 0.0},
        },
        projection={"output_dim": 4, "mlp": {"hidden_dim": 8, "dropout": 0.0}},
        dcl={"num_clusters": 2, "init_max_samples": 8, "init_num_iters": 2,
             "distributed_sync": False},
        acsm={"num_retrieve": 1, "temperature": 0.2, "denoise_threshold": -1.0},
    )
    model = SemMol(**options, pretraining_heads={"geo": {"hidden_dim": 8, "num_radial": 4,
                                                         "max_num_neighbors": 16}})
    criterion = SemMolPretrainTotalLoss(temperature=0.2, warmup_epochs=0, distributed_sync=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_pretrain(batch)
        loss = criterion.compute(output, batch)
        assert torch.isfinite(loss.total_loss)
        assert loss.component_counts["mlm"].item() > 0
        assert loss.component_counts["graph_node"].item() > 0
        assert loss.component_counts["graph_edge"].item() > 0
        assert loss.component_counts["geo_mse"].item() > 0
        if step == 1:
            assert output.alignment_ready
            assert loss.component_counts["pseudo"].item() == 4
            expected = sum(match.positive_embedding for match in output.acsm_output.modality_matches.values())
            torch.testing.assert_close(output.h_pos_final, expected)
        loss.total_loss.backward()
        for modality in ("1d", "2d", "3d"):
            gradients = [p.grad for p in model.encoders[modality].parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
            assert any(torch.count_nonzero(grad).item() for grad in gradients)
        shared = model.projection.projections["1d"]
        assert any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in shared.parameters())
        optimizer.step()

    downstream = SemMol(**options, task={"type": "regression", "num_tasks": 1},
                        head={"hidden_dims": (8,), "dropout": 0.0})
    state = model.state_dict()
    compatible = {key: value for key, value in state.items() if not key.startswith("pretraining_heads.")}
    loaded = downstream.load_state_dict(compatible, strict=False)
    assert all(key.startswith("property_head.") for key in loaded.missing_keys)
    assert not loaded.unexpected_keys
    downstream.train()
    clean_batch = MultimodalDataCollator(pad_token_id=tokenizer.pad_token_id)(samples)
    result = downstream.forward_finetune(clean_batch)
    assert result.predictions.shape == (2, 1)
    torch.testing.assert_close(result.fused_features[:, :4], result.acsm_output.anchor_embedding)
    torch.testing.assert_close(result.fused_features[:, 4:], result.acsm_output.positive_embedding)
    target = torch.tensor([[0.0], [1.0]])
    (result.predictions - target).square().mean().backward()
    assert all(p.requires_grad for p in downstream.projection.projections["1d"].parameters())
    assert all(p.grad is None for p in downstream.encoders["2d"].parameters())

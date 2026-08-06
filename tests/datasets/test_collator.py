from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.datasets.collator import MultimodalDataCollator, PretrainingDataCollator


def _graph(num_nodes: int) -> Data:
    if num_nodes == 1:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)
    else:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)
    return Data(
        x=torch.zeros((num_nodes, 9), dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
    )


def _sample(sample_id: str, token_ids: list[int], atoms: int, conformers: int) -> dict:
    return {
        "sample_id": sample_id,
        "source_index": int(sample_id.rsplit(":", 1)[1]),
        "smiles": "CC",
        "input_ids": torch.tensor(token_ids, dtype=torch.long),
        "token_spans": torch.tensor([[0, 0]] + [[0, 1]] * (len(token_ids) - 2) + [[2, 2]]),
        "graph": _graph(atoms),
        "atomic_numbers": torch.full((atoms,), 6, dtype=torch.long),
        "coords": torch.zeros((conformers, atoms, 3), dtype=torch.float32),
        "conformer_mask": torch.ones(conformers, dtype=torch.bool),
        "energies": torch.arange(conformers, dtype=torch.float32),
        "energy_mask": torch.ones(conformers, dtype=torch.bool),
        "heavy_atom_indices": torch.arange(atoms, dtype=torch.long),
        "conformer_sources": tuple("unit" for _ in range(conformers)),
        "qm_grid": torch.ones((1, 4, 4, 4), dtype=torch.float32),
        "labels": torch.tensor([1.0, float("nan")]),
        "label_mask": torch.tensor([True, False]),
    }


def test_multimodal_collator_pads_tokens_atoms_and_conformers_with_masks() -> None:
    batch = [
        _sample("pcqm:1", [2, 8, 3], atoms=2, conformers=1),
        _sample("pcqm:2", [2, 9, 10, 3], atoms=3, conformers=2),
    ]

    out = MultimodalDataCollator(pad_token_id=0)(batch)

    assert out["input_ids"].shape == (2, 4)
    assert out["attention_mask"].tolist() == [[True, True, True, False], [True, True, True, True]]
    assert out["coords"].shape == (2, 2, 3, 3)
    assert out["atom_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert out["conformer_mask"].tolist() == [[True, False], [True, True]]
    assert out["graph"].num_graphs == 2
    assert out["qm_grid"].shape == (2, 1, 4, 4, 4)
    assert out["label_mask"].tolist() == [[True, False], [True, False]]


def test_multimodal_collator_marks_missing_optional_modality_without_fake_valid_data() -> None:
    complete = _sample("pcqm:1", [2, 8, 3], atoms=2, conformers=1)
    partial = {
        "sample_id": "pcqm:2",
        "source_index": 2,
        "smiles": "N",
        "input_ids": torch.tensor([2, 7, 3]),
        "token_spans": torch.tensor([[0, 0], [0, 1], [1, 1]]),
    }

    out = MultimodalDataCollator(pad_token_id=0, allow_partial_modalities=True)(
        [complete, partial]
    )

    assert out["modality_mask"].tolist() == [
        [True, True, True, True],
        [True, False, False, False],
    ]
    assert out["atom_mask"][1].sum().item() == 0
    assert out["qm_mask"].tolist() == [True, False]
    assert out["graph_sample_index"].tolist() == [0]


def test_pretraining_collator_is_deterministic_per_epoch_and_sample_id() -> None:
    batch = [
        _sample("pcqm:1", [2, 8, 9, 10, 11, 3], atoms=2, conformers=1),
        _sample("pcqm:2", [2, 12, 13, 14, 15, 3], atoms=3, conformers=1),
    ]
    collator = PretrainingDataCollator(
        pad_token_id=0,
        mask_token_id=4,
        vocab_size=32,
        special_token_ids=(0, 1, 2, 3, 4),
        smiles_mask_ratio=0.5,
        node_mask_ratio=0.5,
        edge_mask_ratio=0.5,
        geo_noise_std=0.2,
        seed=3407,
    )
    collator.set_epoch(7)

    first = collator(batch)
    second = collator(batch)

    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["mlm_labels"], second["mlm_labels"])
    assert torch.equal(first["coord_noise"], second["coord_noise"])
    assert torch.equal(first["node_mask"], second["node_mask"])
    assert torch.all(first["mlm_labels"][:, 0] == -100)
    assert torch.all(first["mlm_labels"][:, -1] == -100)


def test_graph_mask_uses_reserved_categories_not_valid_zero_categories() -> None:
    batch = [_sample("pcqm:1", [2, 8, 9, 3], atoms=2, conformers=1)]
    collator = PretrainingDataCollator(
        pad_token_id=0,
        mask_token_id=4,
        vocab_size=32,
        special_token_ids=(0, 1, 2, 3, 4),
        smiles_mask_ratio=0.0,
        node_mask_ratio=1.0,
        edge_mask_ratio=1.0,
        geo_noise_std=0.0,
        seed=3407,
    )

    output = collator(batch)

    assert output["node_mask"].all()
    assert output["graph"].x.tolist() == [
        [119, 5, 12, 12, 10, 6, 6, 2, 2],
        [119, 5, 12, 12, 10, 6, 6, 2, 2],
    ]
    assert output["edge_mask"].all()
    assert output["graph"].edge_attr.tolist() == [[5, 6, 2], [5, 6, 2]]
    assert output["node_labels"].tolist() == [[0] * 9, [0] * 9]
    assert output["edge_labels"].tolist() == [[0] * 3, [0] * 3]

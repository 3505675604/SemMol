import pytest
import torch
from rdkit import Chem

from src.datasets.collator import CollationError, PretrainingDataCollator
from src.molecular.graph import mol_to_pyg_graph
from src.molecular.substructure_alignment import align_token_spans_to_atoms


def sample():
    smiles = "CC(=O)N"
    return {
        "sample_id": "guided:0", "source_index": 0, "smiles": smiles,
        "input_ids": torch.tensor([2, 8, 9, 10, 3]),
        "token_spans": torch.tensor([[-1, -1], [0, 2], [2, 6], [6, 7], [-1, -1]]),
        "graph": mol_to_pyg_graph(Chem.MolFromSmiles(smiles)),
    }


def collator():
    return PretrainingDataCollator(
        pad_token_id=0, mask_token_id=4, vocab_size=32,
        special_token_ids=(0, 1, 2, 3, 4), smiles_mask_ratio=1/3,
        node_mask_ratio=1/3, edge_mask_ratio=1/3, geo_noise_std=0.0,
    )


def test_equal_ratios_share_fragments_and_mask_incident_bidirectional_bonds():
    record = sample()
    batch = collator()([record])
    selected = torch.nonzero(batch["mlm_labels"][0] != -100).flatten().tolist()
    alignment = align_token_spans_to_atoms(record["smiles"], record["token_spans"].tolist())
    atoms = set(alignment.atoms_for_tokens(selected))
    assert batch["node_mask"].tolist() == [i in atoms for i in range(4)]
    original_edges = record["graph"].edge_index.t().tolist()
    assert batch["edge_mask"].tolist() == [a in atoms or b in atoms for a, b in original_edges]
    masks = batch["substructure_token_masks"]
    assert torch.equal(masks["mlm"], masks["node"])
    assert torch.equal(masks["mlm"], masks["edge"])
    assert "_fragment_memberships" not in batch


def test_guided_masks_are_invariant_to_batch_order():
    first = sample()
    second = sample()
    second["sample_id"] = "guided:1"
    second["source_index"] = 1
    fn = collator()
    a, b = fn([first, second]), fn([second, first])
    assert torch.equal(a["substructure_token_masks"]["node"][0], b["substructure_token_masks"]["node"][1])


def test_guided_masks_fail_instead_of_guessing_missing_alignment():
    record = sample()
    record.pop("token_spans")
    with pytest.raises(CollationError, match="requires aligned ESPF"):
        collator()([record])

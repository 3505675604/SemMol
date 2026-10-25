import pytest
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem

from src.molecular.graph import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GraphConstructionError,
    mol_to_pyg_graph,
    smiles_to_pyg_graph,
)


def test_graph_uses_ogb_categorical_feature_schema():
    mol = Chem.MolFromSmiles("C=O")
    graph = mol_to_pyg_graph(mol)

    assert graph.x.shape == (2, 9)
    assert graph.edge_index.shape == (2, 2)
    assert graph.edge_attr.shape == (2, 3)
    assert graph.x.dtype == torch.long
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_attr.dtype == torch.long
    assert graph.x[0].tolist() == atom_to_feature_vector(mol.GetAtomWithIdx(0))
    expected_bond = bond_to_feature_vector(mol.GetBondWithIdx(0))
    assert graph.edge_attr[0].tolist() == expected_bond
    assert graph.edge_attr[1].tolist() == expected_bond
    assert NODE_FEATURE_DIM == 9
    assert EDGE_FEATURE_DIM == 3


def test_single_atom_graph_has_a_valid_empty_edge_schema():
    graph = smiles_to_pyg_graph("[Na+]")

    assert graph is not None
    assert graph.num_nodes == 1
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 3)
    assert graph.edge_attr.dtype == torch.long


def test_invalid_smiles_has_one_explicit_failure_policy():
    assert smiles_to_pyg_graph("not a smiles") is None
    with pytest.raises(GraphConstructionError):
        smiles_to_pyg_graph("not a smiles", on_invalid="raise")


def test_noncanonical_smiles_is_reparsed_in_canonical_atom_order():
    noncanonical = smiles_to_pyg_graph("OC")
    canonical = smiles_to_pyg_graph("CO")

    assert noncanonical.canonical_smiles == "CO"
    assert canonical.canonical_smiles == "CO"
    assert torch.equal(noncanonical.x, canonical.x)
    assert torch.equal(noncanonical.edge_index, canonical.edge_index)
    assert torch.equal(noncanonical.edge_attr, canonical.edge_attr)

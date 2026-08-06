"""Build PyG molecular graphs with the canonical OGB categorical schema.

Schema:
    ``x``: ``torch.long`` with shape ``(num_atoms, 9)``
    ``edge_index``: ``torch.long`` with shape ``(2, 2 * num_bonds)``
    ``edge_attr``: ``torch.long`` with shape ``(2 * num_bonds, 3)``

Each RDKit bond is emitted in both directions. Molecules with no bonds have
empty edge tensors with the same rank and dtype as bonded molecules.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from ogb.utils.features import (
    atom_to_feature_vector,
    bond_to_feature_vector,
    get_atom_feature_dims,
    get_bond_feature_dims,
)
from rdkit import Chem
from torch_geometric.data import Data

from .rdkit_utils import (
    InvalidSmilesError,
    canonicalize_smiles,
    smiles_to_canonical_mol,
    smiles_to_mol,
)

NODE_FEATURE_DIM = 9
EDGE_FEATURE_DIM = 3
OGB_ATOM_FEATURE_CARDINALITIES = tuple(int(v) for v in get_atom_feature_dims())
OGB_BOND_FEATURE_CARDINALITIES = tuple(int(v) for v in get_bond_feature_dims())

if len(OGB_ATOM_FEATURE_CARDINALITIES) != NODE_FEATURE_DIM:
    raise RuntimeError("installed OGB atom feature schema is not the expected v1 schema")
if len(OGB_BOND_FEATURE_CARDINALITIES) != EDGE_FEATURE_DIM:
    raise RuntimeError("installed OGB bond feature schema is not the expected v1 schema")


class GraphConstructionError(ValueError):
    """Raised when a valid molecular graph cannot be constructed."""


def atom_features(atom: Chem.Atom) -> list[int]:
    """Return the nine OGB categorical indices for one atom."""
    return atom_to_feature_vector(atom)


def bond_features(bond: Chem.Bond) -> list[int]:
    """Return the three OGB categorical indices for one bond."""
    return bond_to_feature_vector(bond)


def mol_to_pyg_graph(
    mol: Chem.Mol,
    y: Optional[torch.Tensor] = None,
) -> Data:
    """Convert a non-empty RDKit molecule into a PyG ``Data`` object."""
    if not isinstance(mol, Chem.Mol) or mol.GetNumAtoms() == 0:
        raise GraphConstructionError("expected a non-empty RDKit molecule")
    if y is not None and not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor or None")

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.long,
    ).reshape(mol.GetNumAtoms(), NODE_FEATURE_DIM)

    directed_edges: list[tuple[int, int]] = []
    directed_features: list[list[int]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feature = bond_features(bond)
        directed_edges.extend(((begin, end), (end, begin)))
        directed_features.extend((feature, feature))

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(directed_features, dtype=torch.long).reshape(
            -1, EDGE_FEATURE_DIM
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=mol.GetNumAtoms(),
    )
    if y is not None:
        data.y = y
    return data


def smiles_to_pyg_graph(
    smiles: str,
    y: Optional[torch.Tensor] = None,
    on_invalid: Literal["none", "raise"] = "none",
    canonicalize_atoms: bool = True,
) -> Optional[Data]:
    """Convert SMILES to PyG; invalid input returns ``None`` or raises explicitly."""
    if on_invalid not in {"none", "raise"}:
        raise ValueError("on_invalid must be 'none' or 'raise'")
    if not isinstance(canonicalize_atoms, bool):
        raise TypeError("canonicalize_atoms must be boolean")
    try:
        canonical_smiles = canonicalize_smiles(smiles, on_invalid="raise")
        mol = (
            smiles_to_canonical_mol(smiles, on_invalid="raise")
            if canonicalize_atoms
            else smiles_to_mol(smiles, on_invalid="raise")
        )
    except InvalidSmilesError as exc:
        if on_invalid == "none":
            return None
        if on_invalid == "raise":
            raise GraphConstructionError(str(exc)) from exc
        raise ValueError("on_invalid must be 'none' or 'raise'") from exc
    data = mol_to_pyg_graph(mol, y=y)
    data.canonical_smiles = canonical_smiles
    data.canonical_atom_order = bool(canonicalize_atoms)
    return data

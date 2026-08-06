import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from src.molecular.geometry import (
    GeometryInputError,
    GeometryMismatchError,
    GeometryRecord,
    extract_sdf_geometry,
    generate_conformers,
    geometry_to_safe_mapping,
)


def _embedded_mol(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    return mol


def test_sdf_geometry_reorders_heavy_atoms_to_canonical_smiles():
    original = _embedded_mol("CO")
    renumbered = Chem.RenumberAtoms(
        original,
        list(reversed(range(original.GetNumAtoms()))),
    )

    record = extract_sdf_geometry(renumbered, canonical_smiles="CO")

    assert record.atomic_numbers.tolist()[:2] == [6, 8]
    assert record.heavy_atom_indices.tolist() == [0, 1]
    assert record.heavy_atom_mapping.shape == (2,)
    assert sorted(record.heavy_atom_mapping.tolist()) == sorted(
        atom.GetIdx() for atom in renumbered.GetAtoms() if atom.GetAtomicNum() != 1
    )
    assert record.coords.shape == (1, original.GetNumAtoms(), 3)
    assert record.conformer_source.tolist() == ["pcqm_dft_sdf"]


def test_generated_conformer_count_is_expressed_by_mask_not_padding():
    record = generate_conformers("C", num_conformers=8, seed=7)

    assert record is not None
    assert record.coords.shape[0] == record.conformer_mask.shape[0]
    assert record.coords.shape[0] <= 8
    assert np.all(record.conformer_mask)
    assert not np.any(record.conformer_source == "padding")
    assert np.all(np.isfinite(record.energies) | np.isnan(record.energies))


def test_invalid_geometry_input_is_explicit():
    assert generate_conformers("not a smiles", on_invalid="none") is None
    with pytest.raises(GeometryInputError):
        generate_conformers("not a smiles", on_invalid="raise")


def test_geometry_storage_contract_contains_no_object_arrays():
    record = generate_conformers("CC", num_conformers=1, seed=11)
    assert record is not None

    stored = record.to_storage_dict()

    expected = {
        "atomic_numbers",
        "coords",
        "energies",
        "energy_mask",
        "conformer_mask",
        "conformer_source",
        "sources",
        "heavy_atom_indices",
        "heavy_atom_mapping",
        "canonical_smiles",
        "reason",
    }
    assert set(stored) == expected
    assert all(value.dtype != object for value in stored.values())

    builder_mapping = geometry_to_safe_mapping(record)
    assert set(builder_mapping) == {
        "atomic_numbers",
        "coords",
        "conformer_mask",
        "energies",
        "energy_mask",
        "heavy_atom_indices",
        "sources",
    }
    assert builder_mapping["energy_mask"].tolist() == [
        bool(np.isfinite(record.energies[0]))
    ]
    assert isinstance(builder_mapping["sources"], list)
    assert all(isinstance(source, str) for source in builder_mapping["sources"])
    assert record["sources"] == builder_mapping["sources"]


def test_generated_geometry_uses_same_canonical_heavy_atom_order_as_2d():
    record = generate_conformers("OC", num_conformers=1, seed=23)

    assert record is not None
    assert record.canonical_smiles == "CO"
    assert record.atomic_numbers[record.heavy_atom_indices].tolist() == [6, 8]


def test_long_sdf_source_is_not_truncated_to_one_character():
    record = extract_sdf_geometry(
        _embedded_mol("CO"),
        canonical_smiles="CO",
        source="pcqm_dft_sdf",
    )

    assert record.conformer_source.tolist() == ["pcqm_dft_sdf"]
    assert geometry_to_safe_mapping(record)["sources"] == ["pcqm_dft_sdf"]


def test_sdf_stereochemistry_must_match_canonical_smiles():
    opposite_stereo = _embedded_mol(r"F/C=C\F")

    with pytest.raises(GeometryMismatchError):
        extract_sdf_geometry(
            opposite_stereo,
            canonical_smiles="F/C=C/F",
        )


@pytest.mark.parametrize(
    ("heavy_indices", "heavy_mapping"),
    [
        (np.array([0, 1]), np.array([0, 0])),
        (np.array([0]), np.array([0])),
    ],
)
def test_heavy_atom_mapping_must_be_unique_and_cover_all_heavy_atoms(
    heavy_indices,
    heavy_mapping,
):
    with pytest.raises(GeometryInputError):
        GeometryRecord(
            atomic_numbers=np.array([6, 8], dtype=np.int64),
            coords=np.zeros((1, 2, 3), dtype=np.float32),
            energies=np.array([np.nan], dtype=np.float32),
            conformer_mask=np.array([True]),
            conformer_source=np.array(["pcqm_dft_sdf"]),
            heavy_atom_indices=heavy_indices,
            heavy_atom_mapping=heavy_mapping,
            canonical_smiles="CO",
        )

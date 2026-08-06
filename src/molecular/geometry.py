"""Deterministic 3D geometry extraction and conformer generation.

``GeometryRecord`` keeps full-atom coordinates for density construction while
also carrying the canonical-heavy-atom mapping needed to align official PCQM
SDF geometries with SMILES-derived 2D graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Optional, TypedDict, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .rdkit_utils import (
    InvalidSmilesError,
    canonicalize_smiles,
    smiles_to_canonical_mol,
    smiles_to_mol,
)

MoleculeInput = Union[str, Chem.Mol]


class GeometrySafeMapping(TypedDict):
    """Core builder contract: numeric arrays plus JSON-safe source strings."""

    atomic_numbers: np.ndarray
    coords: np.ndarray
    conformer_mask: np.ndarray
    energies: np.ndarray
    energy_mask: np.ndarray
    heavy_atom_indices: np.ndarray
    sources: list[str]


class GeometryError(ValueError):
    """Base class for molecular geometry failures."""


class GeometryInputError(GeometryError):
    """Raised for an invalid SMILES, molecule, or geometry shape."""


class GeometryMismatchError(GeometryError):
    """Raised when an SDF heavy-atom graph does not match canonical SMILES."""


class GeometryEmbeddingError(GeometryError):
    """Raised when RDKit cannot embed any conformer."""


@dataclass(frozen=True)
class GeometryRecord:
    """A NumPy-only, safe-to-serialize molecular geometry record.

    ``heavy_atom_mapping[i]`` is the original SDF atom index corresponding to
    canonical heavy atom ``i``. ``heavy_atom_indices`` are the positions of
    those canonical heavy atoms in the stored/reordered all-atom arrays.
    """

    atomic_numbers: np.ndarray
    coords: np.ndarray
    energies: np.ndarray
    conformer_mask: np.ndarray
    conformer_source: np.ndarray
    heavy_atom_indices: np.ndarray
    heavy_atom_mapping: np.ndarray
    canonical_smiles: str
    reason: str = ""

    def __post_init__(self) -> None:
        raw_atomic_numbers = np.asarray(self.atomic_numbers)
        raw_coords = np.asarray(self.coords)
        raw_energies = np.asarray(self.energies)
        raw_conformer_mask = np.asarray(self.conformer_mask)
        raw_conformer_source = np.asarray(self.conformer_source)
        raw_heavy_atom_indices = np.asarray(self.heavy_atom_indices)
        raw_heavy_atom_mapping = np.asarray(self.heavy_atom_mapping)
        for name, array in (
            ("atomic_numbers", raw_atomic_numbers),
            ("heavy_atom_indices", raw_heavy_atom_indices),
            ("heavy_atom_mapping", raw_heavy_atom_mapping),
        ):
            if array.dtype.kind not in {"i", "u"}:
                raise GeometryInputError(
                    f"{name} must use an integer dtype; implicit truncation is forbidden"
                )
            if (
                array.dtype.kind == "u"
                and array.size
                and int(array.max()) > np.iinfo(np.int64).max
            ):
                raise GeometryInputError(f"{name} exceeds the int64 range")
        if raw_coords.dtype.kind not in {"i", "u", "f"}:
            raise GeometryInputError("coords must use a real numeric dtype")
        if raw_energies.dtype.kind not in {"i", "u", "f"}:
            raise GeometryInputError("energies must use a real numeric dtype")
        if raw_conformer_mask.dtype.kind != "b":
            raise GeometryInputError("conformer_mask must use a boolean dtype")
        if raw_conformer_source.dtype.kind != "U":
            raise GeometryInputError("conformer_source must use a Unicode dtype")
        if not isinstance(self.reason, str):
            raise GeometryInputError("reason must be a string")

        atomic_numbers = raw_atomic_numbers.astype(np.int64, copy=False)
        coords = raw_coords.astype(np.float32, copy=False)
        energies = raw_energies.astype(np.float32, copy=False)
        conformer_mask = raw_conformer_mask.astype(np.bool_, copy=False)
        conformer_source = raw_conformer_source.astype(np.str_, copy=False)
        heavy_atom_indices = raw_heavy_atom_indices.astype(np.int64, copy=False)
        heavy_atom_mapping = raw_heavy_atom_mapping.astype(np.int64, copy=False)

        if atomic_numbers.ndim != 1 or atomic_numbers.size == 0:
            raise GeometryInputError("atomic_numbers must be a non-empty 1D array")
        if coords.ndim != 3 or coords.shape[1:] != (atomic_numbers.size, 3):
            raise GeometryInputError("coords must have shape (C, A, 3)")
        count = coords.shape[0]
        if count == 0:
            raise GeometryInputError("at least one conformer is required")
        if energies.shape != (count,):
            raise GeometryInputError("energies must have shape (C,)")
        if np.any(np.isinf(energies)):
            raise GeometryInputError("energies may be finite or NaN, but not infinite")
        if conformer_mask.shape != (count,):
            raise GeometryInputError("conformer_mask must have shape (C,)")
        if not np.any(conformer_mask):
            raise GeometryInputError("at least one conformer must be valid")
        if np.any(np.isfinite(energies) & ~conformer_mask):
            raise GeometryInputError("invalid conformers cannot carry finite energies")
        if conformer_source.shape != (count,):
            raise GeometryInputError("conformer_source must have shape (C,)")
        if heavy_atom_indices.ndim != 1:
            raise GeometryInputError("heavy_atom_indices must be one-dimensional")
        if heavy_atom_mapping.shape != heavy_atom_indices.shape:
            raise GeometryInputError(
                "heavy_atom_mapping and heavy_atom_indices must have equal shape"
            )
        if np.any(heavy_atom_indices < 0) or np.any(
            heavy_atom_indices >= atomic_numbers.size
        ):
            raise GeometryInputError("heavy_atom_indices are out of range")
        expected_heavy = np.flatnonzero(atomic_numbers != 1).astype(np.int64)
        if not np.array_equal(heavy_atom_indices, expected_heavy):
            raise GeometryInputError(
                "heavy_atom_indices must canonically cover every stored heavy atom"
            )
        if (
            np.any(heavy_atom_mapping < 0)
            or np.any(heavy_atom_mapping >= atomic_numbers.size)
            or np.unique(heavy_atom_mapping).size != heavy_atom_mapping.size
        ):
            raise GeometryInputError(
                "heavy_atom_mapping must contain unique in-range source indices"
            )
        if np.any(atomic_numbers <= 0) or np.any(atomic_numbers > 118):
            raise GeometryInputError("atomic numbers must be in the range 1..118")
        if np.any(atomic_numbers[heavy_atom_indices] == 1):
            raise GeometryInputError("heavy_atom_indices must not point to hydrogen")
        if not np.all(conformer_source != ""):
            raise GeometryInputError("every conformer must have a non-empty source")
        if not np.all(np.isfinite(coords)):
            raise GeometryInputError("coords must contain only finite values")
        if not isinstance(self.canonical_smiles, str) or not self.canonical_smiles:
            raise GeometryInputError("canonical_smiles must be non-empty")

        object.__setattr__(self, "atomic_numbers", atomic_numbers)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "conformer_mask", conformer_mask)
        object.__setattr__(self, "conformer_source", conformer_source)
        object.__setattr__(self, "heavy_atom_indices", heavy_atom_indices)
        object.__setattr__(self, "heavy_atom_mapping", heavy_atom_mapping)

    def __getitem__(self, key: str) -> Any:
        """Provide backward-compatible dictionary-style field access."""
        if key in {"energy_mask", "sources"}:
            return getattr(self, key)
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    @property
    def energy_mask(self) -> np.ndarray:
        """Indicate which conformers have a real force-field/DFT energy."""
        return np.isfinite(self.energies)

    @property
    def sources(self) -> list[str]:
        """Stable plural alias used by the downstream storage contract."""
        return self.conformer_source.tolist()

    def to_storage_dict(self) -> dict[str, np.ndarray]:
        """Return only non-object NumPy arrays accepted by safe NPZ loading."""
        safe = geometry_to_safe_mapping(self)
        return {
            "atomic_numbers": safe["atomic_numbers"],
            "coords": safe["coords"],
            "conformer_mask": safe["conformer_mask"],
            "energies": safe["energies"],
            "energy_mask": safe["energy_mask"],
            "heavy_atom_indices": safe["heavy_atom_indices"],
            "sources": np.asarray(safe["sources"], dtype=np.str_),
            "conformer_source": self.conformer_source.astype(np.str_),
            "heavy_atom_mapping": self.heavy_atom_mapping,
            "canonical_smiles": np.asarray(self.canonical_smiles, dtype=np.str_),
            "reason": np.asarray(self.reason, dtype=np.str_),
        }

    @classmethod
    def from_storage_dict(
        cls,
        data: Mapping[str, np.ndarray],
        prefix: str = "",
    ) -> "GeometryRecord":
        """Restore a record from an NPZ-like mapping and optional key prefix."""
        def value(name: str) -> np.ndarray:
            return np.asarray(data[f"{prefix}{name}"])

        conformer_source_key = f"{prefix}conformer_source"
        available_keys = (
            set(data.files)
            if hasattr(data, "files")
            else set(data.keys())
        )
        source_field = (
            value("conformer_source")
            if conformer_source_key in available_keys
            else value("sources")
        )

        return cls(
            atomic_numbers=value("atomic_numbers"),
            coords=value("coords"),
            energies=value("energies"),
            conformer_mask=value("conformer_mask"),
            conformer_source=source_field,
            heavy_atom_indices=value("heavy_atom_indices"),
            heavy_atom_mapping=value("heavy_atom_mapping"),
            canonical_smiles=str(value("canonical_smiles").item()),
            reason=str(value("reason").item()),
        )


def geometry_to_safe_mapping(record: GeometryRecord) -> GeometrySafeMapping:
    """Convert a geometry record to the stable builder-facing array mapping."""
    if not isinstance(record, GeometryRecord):
        raise TypeError("record must be a GeometryRecord")
    return {
        "atomic_numbers": record.atomic_numbers,
        "coords": record.coords,
        "conformer_mask": record.conformer_mask,
        "energies": record.energies,
        "energy_mask": record.energy_mask.astype(np.bool_, copy=False),
        "heavy_atom_indices": record.heavy_atom_indices,
        "sources": list(record.sources),
    }


def _canonical_molecule(molecule: MoleculeInput) -> tuple[Chem.Mol, str]:
    if isinstance(molecule, str):
        try:
            canonical = canonicalize_smiles(molecule, on_invalid="raise")
            mol = smiles_to_canonical_mol(molecule, on_invalid="raise")
        except InvalidSmilesError as exc:
            raise GeometryInputError(str(exc)) from exc
        return mol, canonical
    if not isinstance(molecule, Chem.Mol) or molecule.GetNumAtoms() == 0:
        raise GeometryInputError("expected a non-empty SMILES or RDKit Mol")
    try:
        canonical = Chem.MolToSmiles(
            Chem.RemoveHs(Chem.Mol(molecule)),
            canonical=True,
            isomericSmiles=True,
        )
        mol = Chem.MolFromSmiles(canonical)
    except (RuntimeError, ValueError) as exc:
        raise GeometryInputError(f"cannot canonicalize RDKit Mol: {exc}") from exc
    if mol is None or not canonical:
        raise GeometryInputError("cannot canonicalize RDKit Mol")
    return mol, canonical


def _force_field_result(
    mol: Chem.Mol,
    conf_id: int,
    optimize: bool,
) -> tuple[float, str, str]:
    if not optimize:
        return np.nan, "etkdg_unoptimized", "optimization_disabled"

    if AllChem.MMFFHasAllMoleculeParams(mol):
        try:
            properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
            status = AllChem.MMFFOptimizeMolecule(
                mol,
                mmffVariant="MMFF94",
                confId=conf_id,
                maxIters=500,
            )
            force_field = AllChem.MMFFGetMoleculeForceField(
                mol,
                properties,
                confId=conf_id,
            )
            if force_field is None:
                return np.nan, "etkdg_unoptimized", "mmff_force_field_unavailable"
            source = "etkdg_mmff" if status == 0 else "etkdg_mmff_not_converged"
            reason = "" if status == 0 else f"mmff_status_{status}"
            return float(force_field.CalcEnergy()), source, reason
        except (RuntimeError, ValueError) as exc:
            mmff_reason = f"mmff_failed:{type(exc).__name__}"
    else:
        mmff_reason = "mmff_parameters_unavailable"

    if AllChem.UFFHasAllMoleculeParams(mol):
        try:
            status = AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
            force_field = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if force_field is None:
                return np.nan, "etkdg_unoptimized", (
                    f"{mmff_reason};uff_force_field_unavailable"
                )
            source = "etkdg_uff" if status == 0 else "etkdg_uff_not_converged"
            reason = mmff_reason if status == 0 else (
                f"{mmff_reason};uff_status_{status}"
            )
            return float(force_field.CalcEnergy()), source, reason
        except (RuntimeError, ValueError) as exc:
            return np.nan, "etkdg_unoptimized", (
                f"{mmff_reason};uff_failed:{type(exc).__name__}"
            )
    return np.nan, "etkdg_unoptimized", (
        f"{mmff_reason};uff_parameters_unavailable"
    )


def generate_conformers(
    molecule: MoleculeInput,
    num_conformers: int = 5,
    prune_rms_thresh: float = 0.5,
    seed: int = 42,
    optimize: bool = True,
    on_invalid: Literal["none", "raise"] = "none",
) -> Optional[GeometryRecord]:
    """Generate up to ``num_conformers`` full-atom ETKDGv2 conformers."""
    if on_invalid not in {"none", "raise"}:
        raise ValueError("on_invalid must be 'none' or 'raise'")
    if (
        not isinstance(num_conformers, (int, np.integer))
        or isinstance(num_conformers, (bool, np.bool_))
    ):
        raise TypeError("num_conformers must be an integer")
    if int(num_conformers) <= 0:
        raise ValueError("num_conformers must be positive")
    if (
        not isinstance(prune_rms_thresh, (int, float, np.integer, np.floating))
        or isinstance(prune_rms_thresh, (bool, np.bool_))
    ):
        raise TypeError("prune_rms_thresh must be numeric")
    if not np.isfinite(float(prune_rms_thresh)) or float(prune_rms_thresh) < 0:
        raise ValueError("prune_rms_thresh must be finite and non-negative")
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
    ):
        raise TypeError("seed must be an integer")
    if not 0 <= int(seed) <= 0x7FFFFFFF:
        raise ValueError("seed must be within RDKit's signed 32-bit range")
    if not isinstance(optimize, (bool, np.bool_)):
        raise TypeError("optimize must be boolean")
    num_conformers = int(num_conformers)
    prune_rms_thresh = float(prune_rms_thresh)
    seed = int(seed)
    optimize = bool(optimize)
    try:
        base_mol, canonical = _canonical_molecule(molecule)
    except GeometryInputError:
        if on_invalid == "none":
            return None
        if on_invalid == "raise":
            raise
        raise ValueError("on_invalid must be 'none' or 'raise'")

    mol = Chem.AddHs(base_mol)
    params = AllChem.ETKDGv2()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = float(prune_rms_thresh)
    params.numThreads = 1
    params.clearConfs = True
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=max(num_conformers * 2, num_conformers),
            params=params,
        )
    )
    if not conformer_ids:
        if on_invalid == "none":
            return None
        raise GeometryEmbeddingError(
            f"ETKDGv2 failed to embed any conformer for {canonical!r}"
        )

    optimized: list[tuple[int, float, str, str]] = []
    for conformer_id in conformer_ids:
        energy, source, reason = _force_field_result(
            mol,
            int(conformer_id),
            optimize=optimize,
        )
        optimized.append((int(conformer_id), energy, source, reason))

    optimized.sort(
        key=lambda item: (
            not np.isfinite(item[1]),
            item[1] if np.isfinite(item[1]) else np.inf,
            item[0],
        )
    )
    selected = optimized[:num_conformers]
    coords = np.stack(
        [
            np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=np.float32)
            for conf_id, _, _, _ in selected
        ],
        axis=0,
    )
    atomic_numbers = np.fromiter(
        (atom.GetAtomicNum() for atom in mol.GetAtoms()),
        dtype=np.int64,
        count=mol.GetNumAtoms(),
    )
    heavy_indices = np.flatnonzero(atomic_numbers != 1).astype(np.int64)
    reasons = sorted({item[3] for item in selected if item[3]})
    if len(selected) < num_conformers:
        reasons.append(
            f"requested_{num_conformers}_conformers_generated_{len(selected)}"
        )
    return GeometryRecord(
        atomic_numbers=atomic_numbers,
        coords=coords,
        energies=np.asarray([item[1] for item in selected], dtype=np.float32),
        conformer_mask=np.ones(len(selected), dtype=np.bool_),
        conformer_source=np.asarray([item[2] for item in selected], dtype=np.str_),
        heavy_atom_indices=heavy_indices,
        heavy_atom_mapping=heavy_indices.copy(),
        canonical_smiles=canonical,
        reason=";".join(reasons),
    )


def generate_single_conformer(
    molecule: MoleculeInput,
    seed: int = 42,
    optimize: bool = True,
    on_invalid: Literal["none", "raise"] = "none",
) -> Optional[GeometryRecord]:
    """Generate one deterministic full-atom conformer."""
    return generate_conformers(
        molecule,
        num_conformers=1,
        seed=seed,
        optimize=optimize,
        on_invalid=on_invalid,
    )


def _heavy_atom_mapping(
    sdf_mol: Chem.Mol,
    canonical_smiles: str,
) -> tuple[np.ndarray, Chem.Mol]:
    target = smiles_to_mol(canonical_smiles, on_invalid="raise")
    target_heavy = Chem.RemoveHs(Chem.Mol(target))
    source_heavy = Chem.RemoveHs(Chem.Mol(sdf_mol))
    if source_heavy.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(
            source_heavy,
            confId=0,
            replaceExistingTags=False,
        )
    Chem.AssignStereochemistry(target_heavy, cleanIt=True, force=True)
    Chem.AssignStereochemistry(source_heavy, cleanIt=True, force=True)
    if (
        source_heavy.GetNumAtoms() != target_heavy.GetNumAtoms()
        or source_heavy.GetNumBonds() != target_heavy.GetNumBonds()
    ):
        raise GeometryMismatchError(
            "SDF and canonical SMILES have different heavy-atom graph sizes"
        )
    match = source_heavy.GetSubstructMatch(target_heavy, useChirality=True)
    reverse = target_heavy.GetSubstructMatch(source_heavy, useChirality=True)
    if len(match) != target_heavy.GetNumAtoms() or len(reverse) != source_heavy.GetNumAtoms():
        raise GeometryMismatchError(
            "SDF heavy-atom graph/stereochemistry does not match canonical SMILES"
        )
    source_isomeric = Chem.MolToSmiles(
        source_heavy,
        canonical=True,
        isomericSmiles=True,
    )
    target_isomeric = Chem.MolToSmiles(
        target_heavy,
        canonical=True,
        isomericSmiles=True,
    )
    if source_isomeric != target_isomeric:
        raise GeometryMismatchError(
            "SDF stereochemistry does not match canonical SMILES"
        )
    source_full_heavy = np.fromiter(
        (
            atom.GetIdx()
            for atom in sdf_mol.GetAtoms()
            if atom.GetAtomicNum() != 1
        ),
        dtype=np.int64,
        count=source_heavy.GetNumAtoms(),
    )
    mapped = source_full_heavy[np.asarray(match, dtype=np.int64)]
    if (
        np.unique(mapped).size != mapped.size
        or set(mapped.tolist()) != set(source_full_heavy.tolist())
    ):
        raise GeometryMismatchError(
            "SDF heavy-atom mapping is not a unique complete cover"
        )
    return mapped, target_heavy


def extract_sdf_geometry(
    sdf_mol: Chem.Mol,
    canonical_smiles: str,
    energy_property: Optional[str] = None,
    source: str = "pcqm_dft_sdf",
) -> GeometryRecord:
    """Validate, align, and extract all conformers from an official SDF molecule."""
    if sdf_mol is None or sdf_mol.GetNumAtoms() == 0:
        raise GeometryInputError("SDF molecule is empty")
    canonical = canonicalize_smiles(canonical_smiles, on_invalid="raise")
    mapping, _ = _heavy_atom_mapping(sdf_mol, canonical)
    if sdf_mol.GetNumConformers() == 0:
        raise GeometryInputError("SDF molecule contains no conformer coordinates")

    mapped_set = set(mapping.tolist())
    remaining = [
        atom.GetIdx()
        for atom in sdf_mol.GetAtoms()
        if atom.GetIdx() not in mapped_set
    ]
    atom_order = np.asarray(mapping.tolist() + remaining, dtype=np.int64)
    atomic_numbers_original = np.fromiter(
        (atom.GetAtomicNum() for atom in sdf_mol.GetAtoms()),
        dtype=np.int64,
        count=sdf_mol.GetNumAtoms(),
    )
    coords = np.stack(
        [
            np.asarray(conf.GetPositions(), dtype=np.float32)[atom_order]
            for conf in sdf_mol.GetConformers()
        ],
        axis=0,
    )
    count = coords.shape[0]
    energies = np.full(count, np.nan, dtype=np.float32)
    reason = "sdf_energy_not_provided"
    if energy_property:
        if not sdf_mol.HasProp(energy_property):
            reason = f"missing_sdf_energy_property:{energy_property}"
        else:
            try:
                energies[:] = float(sdf_mol.GetProp(energy_property))
                reason = ""
            except ValueError as exc:
                raise GeometryInputError(
                    f"SDF energy property {energy_property!r} is not numeric"
                ) from exc

    return GeometryRecord(
        atomic_numbers=atomic_numbers_original[atom_order],
        coords=coords,
        energies=energies,
        conformer_mask=np.ones(count, dtype=np.bool_),
        conformer_source=np.asarray([source for _ in range(count)], dtype=np.str_),
        heavy_atom_indices=np.arange(mapping.size, dtype=np.int64),
        heavy_atom_mapping=mapping,
        canonical_smiles=canonical,
        reason=reason,
    )


def geometry_from_sdf_or_fallback(
    sdf_mol: Optional[Chem.Mol],
    smiles: str,
    num_conformers: int = 1,
    seed: int = 42,
    energy_property: Optional[str] = None,
    prune_rms_thresh: float = 0.5,
    optimize: bool = True,
    sdf_unavailable_reason: str = "sdf_record_missing",
) -> GeometryRecord:
    """Prefer validated SDF DFT geometry, recording any ETKDG fallback reason."""
    fallback_reason = sdf_unavailable_reason
    if sdf_mol is not None:
        try:
            return extract_sdf_geometry(
                sdf_mol,
                canonical_smiles=smiles,
                energy_property=energy_property,
            )
        except GeometryError as exc:
            fallback_reason = f"sdf_fallback:{type(exc).__name__}:{exc}"
        except InvalidSmilesError as exc:
            fallback_reason = f"sdf_fallback:{type(exc).__name__}:{exc}"
    generated = generate_conformers(
        smiles,
        num_conformers=num_conformers,
        prune_rms_thresh=prune_rms_thresh,
        seed=seed,
        optimize=optimize,
        on_invalid="raise",
    )
    return replace(
        generated,
        reason=";".join(part for part in (fallback_reason, generated.reason) if part),
    )


def iter_sdf_molecules(
    path: Union[str, Path],
    sanitize: bool = True,
    remove_hs: bool = False,
) -> Iterator[tuple[int, Optional[Chem.Mol]]]:
    """Stream SDF records without loading the complete official file."""
    supplier = Chem.ForwardSDMolSupplier(
        str(path),
        sanitize=sanitize,
        removeHs=remove_hs,
        strictParsing=True,
    )
    for sdf_ordinal, mol in enumerate(supplier):
        yield sdf_ordinal, mol


def conformers_to_dict(
    molecule: MoleculeInput,
    num_conformers: int = 5,
    seed: int = 42,
) -> Optional[dict[str, np.ndarray]]:
    """Compatibility helper returning the safe NumPy storage mapping."""
    record = generate_conformers(
        molecule,
        num_conformers=num_conformers,
        seed=seed,
    )
    return None if record is None else record.to_storage_dict()

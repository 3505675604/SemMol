"""Small, consistent RDKit parsing and canonicalization helpers."""

from __future__ import annotations

import hashlib
from typing import Literal, Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

InvalidPolicy = Literal["none", "raise"]


class InvalidSmilesError(ValueError):
    """Raised when a SMILES string cannot be parsed and sanitized."""


def _invalid_smiles(
    smiles: object,
    reason: str,
    on_invalid: InvalidPolicy,
) -> None:
    if on_invalid == "raise":
        raise InvalidSmilesError(f"Invalid SMILES {smiles!r}: {reason}")
    if on_invalid != "none":
        raise ValueError("on_invalid must be 'none' or 'raise'")
    return None


def smiles_to_mol(
    smiles: str,
    sanitize: bool = True,
    on_invalid: InvalidPolicy = "none",
) -> Optional[Chem.Mol]:
    """Parse ``smiles`` using the uniform ``None``/exception failure policy."""
    if on_invalid not in {"none", "raise"}:
        raise ValueError("on_invalid must be 'none' or 'raise'")
    if not isinstance(sanitize, bool):
        raise TypeError("sanitize must be boolean")
    if not isinstance(smiles, str) or not smiles.strip():
        return _invalid_smiles(smiles, "expected a non-empty string", on_invalid)
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    except (RuntimeError, ValueError) as exc:
        return _invalid_smiles(smiles, str(exc), on_invalid)
    if mol is None:
        return _invalid_smiles(smiles, "RDKit parser returned no molecule", on_invalid)
    if mol.GetNumAtoms() == 0:
        return _invalid_smiles(smiles, "molecule contains no atoms", on_invalid)
    return mol


def canonicalize_smiles(
    smiles: str,
    on_invalid: InvalidPolicy = "none",
) -> Optional[str]:
    """Return an isomeric canonical SMILES under the shared failure policy."""
    mol = smiles_to_mol(smiles, on_invalid=on_invalid)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except (RuntimeError, ValueError) as exc:
        return _invalid_smiles(smiles, str(exc), on_invalid)


def smiles_to_canonical_mol(
    smiles: str,
    on_invalid: InvalidPolicy = "none",
) -> Optional[Chem.Mol]:
    """Canonicalize SMILES and reparse it to obtain canonical atom ordering."""
    canonical = canonicalize_smiles(smiles, on_invalid=on_invalid)
    if canonical is None:
        return None
    return smiles_to_mol(canonical, on_invalid=on_invalid)


def mol_to_smiles(
    mol: Chem.Mol,
    on_invalid: InvalidPolicy = "none",
) -> Optional[str]:
    """Return an isomeric canonical SMILES for an RDKit molecule."""
    if on_invalid not in {"none", "raise"}:
        raise ValueError("on_invalid must be 'none' or 'raise'")
    if not isinstance(mol, Chem.Mol) or mol.GetNumAtoms() == 0:
        return _invalid_smiles(mol, "expected a non-empty RDKit molecule", on_invalid)
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except (RuntimeError, ValueError) as exc:
        return _invalid_smiles(mol, str(exc), on_invalid)


def smiles_hash(smiles: str, length: int = 16) -> str:
    """Return the legacy-compatible MD5 prefix used as a cache identifier."""
    if not isinstance(smiles, str):
        raise TypeError("smiles must be a string")
    if not isinstance(length, int) or isinstance(length, bool):
        raise TypeError("length must be an integer")
    if not 8 <= length <= 32:
        raise ValueError("length must be between 8 and 32")
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()[:length]

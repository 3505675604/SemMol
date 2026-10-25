"""Lossless ESPF-fragment to canonical graph atom alignment.

SMILES graph builders parse the canonical SMILES in lexical atom order.  BPE
fragments span whole lexical symbols; branches and ring-closure symbols do not
create atoms.  This module checks that contract before cross-modal masking.
It never guesses an atom order after a mismatch or splits a bracket atom.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

from .espf_tokenizer import ESPFTokenizer


class SubstructureAlignmentError(ValueError):
    """Token characters and graph atom ordering cannot be aligned safely."""


@dataclass(frozen=True)
class SubstructureAlignment:
    """Token-to-atom membership in the canonical SMILES graph order."""

    token_atoms: tuple[tuple[int, ...], ...]
    atom_spans: tuple[tuple[int, int], ...]

    @property
    def atom_count(self) -> int:
        return len(self.atom_spans)

    @property
    def chemical_token_indices(self) -> tuple[int, ...]:
        return tuple(i for i, atoms in enumerate(self.token_atoms) if atoms)

    def atoms_for_tokens(self, token_indices: Sequence[int]) -> tuple[int, ...]:
        atoms: set[int] = set()
        for token_index in token_indices:
            if not isinstance(token_index, Integral) or isinstance(token_index, bool):
                raise SubstructureAlignmentError("token indices must be integers")
            if not 0 <= int(token_index) < len(self.token_atoms):
                raise SubstructureAlignmentError("token index outside alignment")
            atoms.update(self.token_atoms[int(token_index)])
        return tuple(sorted(atoms))


def align_token_spans_to_atoms(
    smiles: str,
    token_spans: Sequence[Sequence[int]],
    *,
    expected_atoms: int | None = None,
) -> SubstructureAlignment:
    """Validate complete fragment spans and assign each atom to one fragment.

    Explicit sentinel spans ``(-1, -1)`` are reserved for special/pad tokens.
    Ordinary spans must form a non-overlapping, complete partition of SMILES.
    The expected atom count detects stale/reordered graph data rather than
    silently masking unrelated atoms. The caller must use graphs parsed from
    this exact canonical SMILES; graph isomorphism is not an atom mapping.
    """

    symbols, lexical_spans = ESPFTokenizer.atom_tokenize_with_spans(smiles)
    atom_spans = tuple(
        span for symbol, span in zip(symbols, lexical_spans)
        if symbol.startswith("[") or symbol in {"Br", "Cl", "*"}
        or (len(symbol) == 1 and symbol in "BCNOPSFIbcnops")
    )
    if not atom_spans:
        raise SubstructureAlignmentError("SMILES must contain at least one atom")
    if expected_atoms is not None:
        if (not isinstance(expected_atoms, Integral) or isinstance(expected_atoms, bool)
                or expected_atoms <= 0):
            raise SubstructureAlignmentError("expected_atoms must be positive")
        if len(atom_spans) != int(expected_atoms):
            raise SubstructureAlignmentError(
                f"SMILES has {len(atom_spans)} lexical atoms but graph has "
                f"{expected_atoms}; rebuild canonical graph/token data together"
            )
    token_atoms: list[tuple[int, ...]] = []
    cursor = 0
    memberships = [0] * len(atom_spans)
    for token_index, raw_span in enumerate(token_spans):
        if len(raw_span) != 2 or any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in raw_span
        ):
            raise SubstructureAlignmentError("token spans must contain integer pairs")
        start, end = map(int, raw_span)
        if (start, end) == (-1, -1):
            token_atoms.append(())
            continue
        if start != cursor or not start < end <= len(smiles):
            raise SubstructureAlignmentError(
                f"token {token_index} span {(start, end)} does not continue "
                f"the lossless SMILES partition at {cursor}"
            )
        members = []
        for atom_index, (atom_start, atom_end) in enumerate(atom_spans):
            if start < atom_end and atom_start < end:
                if not (start <= atom_start and atom_end <= end):
                    raise SubstructureAlignmentError("fragment splits an atom symbol")
                members.append(atom_index)
                memberships[atom_index] += 1
        token_atoms.append(tuple(members))
        cursor = end
    if cursor != len(smiles) or any(count != 1 for count in memberships):
        raise SubstructureAlignmentError(
            "fragments must cover the complete SMILES and every atom exactly once"
        )
    return SubstructureAlignment(tuple(token_atoms), atom_spans)

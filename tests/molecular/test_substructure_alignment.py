import pytest

from src.molecular.substructure_alignment import (
    SubstructureAlignmentError,
    align_token_spans_to_atoms,
)


def test_fragment_alignment_handles_brackets_branches_and_ring_closures():
    smiles = "ClC1CC([NH3+])CC1"
    # Non-overlapping BPE fragments; the ring digit creates no new atom.
    fragments = ["ClC1", "CC(", "[NH3+]", ")CC1"]
    spans = [(-1, -1)]
    position = 0
    for fragment in fragments:
        spans.append((position, position + len(fragment)))
        position += len(fragment)
    spans.append((-1, -1))
    alignment = align_token_spans_to_atoms(smiles, spans, expected_atoms=7)
    assert alignment.token_atoms == ((), (0, 1), (2, 3), (4,), (5, 6), ())
    assert alignment.atom_count == 7


def test_fragment_alignment_rejects_stale_graph_atom_count():
    with pytest.raises(SubstructureAlignmentError, match="graph has"):
        align_token_spans_to_atoms("CCO", [(0, 2), (2, 3)], expected_atoms=2)


def test_fragment_alignment_rejects_partial_bracket_atom():
    with pytest.raises(SubstructureAlignmentError, match="splits an atom"):
        align_token_spans_to_atoms("C[NH3+]", [(0, 3), (3, 7)], expected_atoms=2)


def test_fragment_alignment_rejects_missing_or_overlapping_characters():
    for spans in ([(0, 1), (2, 3)], [(0, 2), (1, 3)]):
        with pytest.raises(SubstructureAlignmentError, match="partition"):
            align_token_spans_to_atoms("CCO", spans)


def test_syntax_only_fragment_is_not_a_chemical_mask_candidate():
    alignment = align_token_spans_to_atoms("C(C)O", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert alignment.chemical_token_indices == (0, 2, 4)

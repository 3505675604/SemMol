import json
import pytest

from src.datasets.scaffold_split import generate_scaffold, save_scaffold_split, scaffold_split


def test_all_acyclic_molecules_share_the_empty_murcko_group():
    molecules = ["C", "CC", "CCO", "CCCCN", "C.C", "CC(C)C"]
    assert {generate_scaffold(smiles) for smiles in molecules} == {"murcko:empty"}
    train, valid, test, report = scaffold_split(
        molecules + ["c1ccccc1", "c1ccncc1", "C1CCOCC1"],
        return_report=True, seed=19,
    )
    splits = [set(train), set(valid), set(test)]
    assert sum(bool(set(range(6)) & split) for split in splits) == 1
    assert report["scaffold_definition"] == "bemis_murcko"
    assert report["optimization"]["uses_labels"] is False


def test_modified_acyclic_definition_is_only_explicit():
    assert generate_scaffold("CC") == generate_scaffold("CCCO")
    assert generate_scaffold("CC", definition="bemis_murcko_with_acyclic_connectivity") != generate_scaffold(
        "CCCO", definition="bemis_murcko_with_acyclic_connectivity")
    with pytest.raises(ValueError, match="definition"):
        generate_scaffold("CC", definition="unregistered")


def test_saved_split_identifies_definition_and_allocator(tmp_path):
    train, valid, test, report = scaffold_split(
        ["CC", "CCC", "c1ccccc1", "c1ccncc1"], return_report=True)
    path = tmp_path / "split.json"
    save_scaffold_split(path, train, valid, test,
                        scaffold_groups=report["scaffold_groups"], optimization=report["optimization"])
    saved = json.loads(path.read_text())
    assert saved["scaffold_definition"] == "bemis_murcko"
    assert saved["allocation_strategy"] == "random_groups"

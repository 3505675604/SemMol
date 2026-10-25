from __future__ import annotations

import json

import pytest

from src.molecular.espf_tokenizer import (
    CLS_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    SEP_TOKEN_ID,
    UNK_TOKEN_ID,
    ESPFTokenizer,
    TokenizationError,
    TokenizerFormatError,
)


def test_atom_tokenization_is_lossless_for_smiles_syntax() -> None:
    smiles = r"Cl[C@@H](Br)/C=C\N%12.[13CH2-:7]"

    tokens = ESPFTokenizer.atom_tokenize(smiles)

    assert tokens == [
        "Cl",
        "[C@@H]",
        "(",
        "Br",
        ")",
        "/",
        "C",
        "=",
        "C",
        "\\",
        "N",
        "%12",
        ".",
        "[13CH2-:7]",
    ]
    assert "".join(tokens) == smiles


def test_stereochemical_markers_outside_brackets_are_not_dropped() -> None:
    assert ESPFTokenizer.atom_tokenize("C@@C") == ["C", "@", "@", "C"]


@pytest.mark.parametrize(
    ("smiles", "offset"),
    [
        ("C C", 1),
        ("Cé", 1),
        ("C[Na", 1),
        ("C%1", 1),
        ("C[PAD]", 1),
    ],
)
def test_atom_tokenization_rejects_uncovered_or_incomplete_input(
    smiles: str, offset: int
) -> None:
    with pytest.raises(TokenizationError, match=rf"offset {offset}\b"):
        ESPFTokenizer.atom_tokenize(smiles)


def test_training_applies_pairs_in_learned_order() -> None:
    tokenizer = ESPFTokenizer.train(
        ["CNC", "CNC", "CN"],
        min_frequency=2,
        max_merges=2,
    )

    assert tokenizer.merges == [("C", "N"), ("CN", "C")]
    assert tokenizer.tokenize("CNC") == ["CNC"]
    assert tokenizer.tokenize("CN") == ["CN"]
    assert tokenizer.decode(tokenizer.encode("CNC", max_len=4)) == "CNC"


def test_equal_frequency_pairs_use_lexicographic_tie_breaking() -> None:
    first = ESPFTokenizer.train(
        ["CO", "CN"],
        min_frequency=1,
        max_merges=1,
    )
    second = ESPFTokenizer.train(
        iter(["CO", "CN"]),
        min_frequency=1,
        max_merges=1,
    )

    assert first.merges == [("C", "N")]
    assert second.merges == first.merges
    assert second.vocab == first.vocab


def test_training_consumes_single_pass_iterable_once_and_records_rust_backend() -> None:
    class SinglePassCorpus:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("source corpus was iterated more than once")
            yield from ("CCO", "CCO", "CCN")

    corpus = SinglePassCorpus()

    tokenizer = ESPFTokenizer.train(
        corpus,
        min_frequency=2,
        max_merges=2,
    )

    assert corpus.iterations == 1
    assert tokenizer.metadata["training_backend"] == "tokenizers-rust-bpe"
    assert tokenizer.metadata["encoding_algorithm"] == "pair-rank-heap"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"min_frequency": True}, TypeError),
        ({"max_merges": 1.5}, TypeError),
        ({"vocab_size": "8"}, TypeError),
        ({"max_merges": None, "vocab_size": None}, ValueError),
    ],
)
def test_training_parameters_are_strictly_validated(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ESPFTokenizer.train(["CC"], **kwargs)


def test_training_supports_vocab_size_without_max_merges() -> None:
    tokenizer = ESPFTokenizer.train(
        ["CCO", "CCO"],
        min_frequency=1,
        max_merges=None,
        vocab_size=8,
    )

    assert tokenizer.vocab_size == 8
    assert len(tokenizer.merges) == 1


def test_encode_plus_aligns_attention_mask_and_character_spans() -> None:
    tokenizer = ESPFTokenizer.train(
        ["ClCBr"],
        min_frequency=2,
        max_merges=0,
    )

    encoded = tokenizer.encode_plus("ClCBr", max_len=7)

    assert encoded["input_ids"][0] == CLS_TOKEN_ID
    assert encoded["input_ids"][4] == SEP_TOKEN_ID
    assert encoded["input_ids"][5:] == [PAD_TOKEN_ID, PAD_TOKEN_ID]
    assert encoded["attention_mask"] == [1, 1, 1, 1, 1, 0, 0]
    assert encoded["token_spans"] == [
        (-1, -1),
        (0, 2),
        (2, 3),
        (3, 5),
        (-1, -1),
        (-1, -1),
        (-1, -1),
    ]
    assert tokenizer.decode(encoded["input_ids"]) == "ClCBr"


def test_truncation_always_keeps_sep() -> None:
    tokenizer = ESPFTokenizer.train(
        ["CCCC"],
        min_frequency=99,
        max_merges=0,
    )

    encoded = tokenizer.encode_plus("CCCC", max_len=3)
    one_token = tokenizer.encode_plus("CCCC", max_len=1)

    assert encoded["input_ids"][-1] == SEP_TOKEN_ID
    assert encoded["token_spans"] == [(-1, -1), (0, 1), (-1, -1)]
    assert one_token["input_ids"] == [SEP_TOKEN_ID]
    assert one_token["attention_mask"] == [1]


def test_max_length_alias_matches_dataset_compatible_max_len() -> None:
    tokenizer = ESPFTokenizer.train(["CC"], min_frequency=99, max_merges=0)

    assert tokenizer.encode_plus("CC", max_length=4) == tokenizer.encode_plus(
        "CC", max_len=4
    )


def test_unbounded_standard_encode_plus_is_unpadded() -> None:
    tokenizer = ESPFTokenizer.train(["CC"], min_frequency=99, max_merges=0)

    encoded = tokenizer.encode_plus(
        "CC",
        max_length=None,
        add_special_tokens=True,
    )

    assert len(encoded["input_ids"]) == 4
    assert encoded["attention_mask"] == [1, 1, 1, 1]
    assert encoded["token_spans"] == [
        (-1, -1),
        (0, 1),
        (1, 2),
        (-1, -1),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_length": True},
        {"max_len": 2.5},
        {"add_special_tokens": 1},
        {"padding": 1},
    ],
)
def test_encode_plus_rejects_non_exact_parameter_types(
    kwargs: dict[str, object],
) -> None:
    tokenizer = ESPFTokenizer.train(["CC"], min_frequency=99, max_merges=0)

    with pytest.raises(TypeError):
        tokenizer.encode_plus("CC", **kwargs)


def test_encode_plus_rejects_conflicting_aliases() -> None:
    tokenizer = ESPFTokenizer.train(["CC"], min_frequency=99, max_merges=0)

    with pytest.raises(ValueError, match="max_length and max_len disagree"):
        tokenizer.encode_plus("CC", max_length=3, max_len=4)
    with pytest.raises(
        ValueError, match="add_special_tokens and add_special disagree"
    ):
        tokenizer.encode_plus(
            "CC",
            add_special_tokens=False,
            add_special=True,
        )
    with pytest.raises(
        ValueError, match="add_special_tokens and add_special disagree"
    ):
        tokenizer.encode_plus(
            "CC",
            add_special_tokens=True,
            add_special=False,
        )
    with pytest.raises(ValueError, match="max_length and max_len disagree"):
        tokenizer.encode_plus("CC", max_length=None, max_len=4)


def test_encode_plus_keeps_non_conflicting_legacy_aliases() -> None:
    tokenizer = ESPFTokenizer.train(["CC"], min_frequency=99, max_merges=0)

    encoded = tokenizer.encode_plus(
        "CC",
        max_len=3,
        add_special=False,
    )

    assert encoded["attention_mask"] == [1, 1, 0]
    assert encoded["token_spans"] == [(0, 1), (1, 2), (-1, -1)]


def test_unseen_atom_uses_unk_without_losing_alignment() -> None:
    tokenizer = ESPFTokenizer.train(["CC"], min_frequency=2, max_merges=0)

    encoded = tokenizer.encode_plus("N", max_len=3)

    assert encoded["input_ids"] == [CLS_TOKEN_ID, UNK_TOKEN_ID, SEP_TOKEN_ID]
    assert encoded["attention_mask"] == [1, 1, 1]
    assert encoded["token_spans"] == [(-1, -1), (0, 1), (-1, -1)]


def test_vocab_size_is_a_total_cap_including_special_and_atom_tokens() -> None:
    tokenizer = ESPFTokenizer.train(
        ["CCO", "CCO"],
        min_frequency=1,
        max_merges=99,
        vocab_size=8,
    )

    assert tokenizer.vocab_size == 8
    assert len(tokenizer.merges) == 1


def test_special_token_ids_are_fixed() -> None:
    tokenizer = ESPFTokenizer()

    assert tokenizer.pad_token_id == PAD_TOKEN_ID == 0
    assert tokenizer.unk_token_id == UNK_TOKEN_ID == 1
    assert tokenizer.cls_token_id == CLS_TOKEN_ID == 2
    assert tokenizer.sep_token_id == SEP_TOKEN_ID == 3
    assert tokenizer.mask_token_id == MASK_TOKEN_ID == 4


def test_save_and_load_preserve_encoding_and_metadata(tmp_path) -> None:
    trained = ESPFTokenizer.train(
        ["CCO", "CCO", "CCN"],
        min_frequency=2,
        max_merges=3,
    )
    output_dir = tmp_path / "tokenizer"

    saved = trained.save_pretrained(output_dir)
    loaded = ESPFTokenizer.from_pretrained(output_dir)
    loaded_via_alias = ESPFTokenizer.load(output_dir)
    loaded_from_manifest = ESPFTokenizer.load(saved["tokenizer_manifest"])

    assert set(saved) == {
        "tokenizer_manifest",
        "artifact_manifest",
        "vocab",
        "merges",
        "tokenizer_config",
        "metadata",
    }
    assert all(path.is_file() for path in saved.values())
    root_manifest = json.loads(
        saved["tokenizer_manifest"].read_text(encoding="utf-8")
    )
    assert len(root_manifest["generation_id"]) == 64
    assert set(root_manifest["artifacts"]) == {
        "artifact_manifest.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "metadata.json",
    }
    assert loaded.vocab == trained.vocab
    assert loaded.merges == trained.merges
    assert loaded.metadata == trained.metadata
    assert loaded_via_alias.corpus_sha256 == trained.metadata["corpus_sha256"]
    assert loaded_from_manifest.generation_id == loaded.generation_id
    assert loaded.encode_plus("CCO", max_len=8) == trained.encode_plus(
        "CCO", max_len=8
    )


def test_loading_generation_rejects_artifact_hash_mismatch(tmp_path) -> None:
    tokenizer = ESPFTokenizer.train(
        ["CCO", "CCO"],
        min_frequency=2,
        max_merges=2,
    )
    output_dir = tmp_path / "tokenizer"
    saved = tokenizer.save_pretrained(output_dir)
    saved["vocab"].write_text(
        saved["vocab"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TokenizerFormatError, match="hash mismatch"):
        ESPFTokenizer.from_pretrained(output_dir)


def test_new_generation_switches_manifest_without_mixing_artifacts(
    tmp_path,
) -> None:
    output_dir = tmp_path / "tokenizer"
    first = ESPFTokenizer.train(
        ["CCO", "CCO"],
        min_frequency=2,
        max_merges=1,
    )
    second = ESPFTokenizer.train(
        ["NNN", "NNN"],
        min_frequency=2,
        max_merges=1,
    )
    first.save_pretrained(output_dir)
    first_generation = first.generation_id

    second_paths = second.save_pretrained(output_dir)
    loaded = ESPFTokenizer.from_pretrained(output_dir)

    assert first_generation != second.generation_id
    assert (output_dir / "generations" / str(first_generation)).is_dir()
    assert loaded.generation_id == second.generation_id
    assert loaded.vocab == second.vocab
    assert all(
        str(second.generation_id) in str(path)
        for name, path in second_paths.items()
        if name != "tokenizer_manifest"
    )


def test_inference_respects_rank_and_left_to_right_overlap() -> None:
    tokenizer = ESPFTokenizer.train(
        ["CCCC", "CCCC"],
        min_frequency=1,
        max_merges=2,
    )

    assert tokenizer.tokenize("CCCC") == ["CCCC"]
    assert tokenizer.decode(
        tokenizer.encode_plus("CCCC", max_length=None)["input_ids"]
    ) == "CCCC"


def test_loading_rejects_merge_with_unknown_component(tmp_path) -> None:
    output_dir = tmp_path / "broken"
    output_dir.mkdir()
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "C": 5,
        "CX": 6,
    }
    (output_dir / "vocab.json").write_text(
        json.dumps(vocab), encoding="utf-8"
    )
    (output_dir / "merges.txt").write_text(
        "#version: semmol-espf-bpe-v1\nC X\n", encoding="utf-8"
    )
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_frequency": 1,
                "max_merges": 1,
                "vocab_size": len(vocab),
                "corpus_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TokenizerFormatError, match="unknown right token"):
        ESPFTokenizer.from_pretrained(output_dir)


def test_loading_rejects_duplicate_or_malformed_merge(tmp_path) -> None:
    output_dir = tmp_path / "broken"
    output_dir.mkdir()
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "C": 5,
        "CC": 6,
    }
    (output_dir / "vocab.json").write_text(
        json.dumps(vocab), encoding="utf-8"
    )
    (output_dir / "merges.txt").write_text(
        "#version: semmol-espf-bpe-v1\nC C\nC C\n",
        encoding="utf-8",
    )
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "min_frequency": 1,
                "max_merges": 2,
                "vocab_size": len(vocab),
                "corpus_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TokenizerFormatError, match="duplicate merge"):
        ESPFTokenizer.from_pretrained(output_dir)

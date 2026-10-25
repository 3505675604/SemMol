"""Deterministic atom-wise BPE/ESPF tokenizer for SMILES.

The tokenizer first performs a lossless lexical split at SMILES atom/syntax
boundaries and then applies learned, ordered byte-pair merges.  It has no
network dependency and fixes the five model special-token IDs:

``[PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4``.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import tempfile
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

SPECIAL_TOKENS: List[str] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
CLS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
MASK_TOKEN_ID = 4

SCHEMA_VERSION = 1
MERGES_HEADER = "#version: semmol-espf-bpe-v1"
ROOT_MANIFEST_NAME = "tokenizer_manifest.json"
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"

PathLike = Union[str, os.PathLike[str]]
Pair = Tuple[str, str]
Span = Tuple[int, int]

_TWO_CHARACTER_ATOMS = frozenset(("Br", "Cl"))
_ONE_CHARACTER_ATOMS = frozenset("BCNOPSFIbcnops*")
_ONE_CHARACTER_SYNTAX = frozenset("-=#$:/\\.()~?<>")


class TokenizationError(ValueError):
    """Raised when a SMILES string cannot be split without dropping input."""


class TokenizerFormatError(ValueError):
    """Raised when persisted tokenizer artifacts are inconsistent or corrupt."""


class EncodedSmiles(TypedDict):
    """Dataset-ready encoding and its alignment to the source SMILES."""

    input_ids: List[int]
    attention_mask: List[int]
    token_spans: List[Span]


class _OmittedDefault:
    """Sentinel whose repr keeps public signatures human-readable."""

    def __init__(self, displayed_value: Any) -> None:
        self.displayed_value = displayed_value

    def __repr__(self) -> str:
        return repr(self.displayed_value)


_OMITTED_NONE: Any = _OmittedDefault(None)
_OMITTED_TRUE: Any = _OmittedDefault(True)
_OMITTED_128: Any = _OmittedDefault(128)


def _require_int(
    name: str,
    value: Any,
    *,
    minimum: int,
    allow_none: bool,
) -> Optional[int]:
    if value is None and allow_none:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        expected = "an integer or None" if allow_none else "an integer"
        raise TypeError(f"{name} must be {expected}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _sha256_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _private_use_symbols(count: int) -> List[str]:
    """Return deterministic Unicode scalars unavailable in accepted SMILES."""
    if count < 0:
        raise ValueError("private-use symbol count must be non-negative")
    if count == 0:
        return []
    ranges = (
        (0xE000, 0xF8FF),
        (0xF0000, 0xFFFFD),
        (0x100000, 0x10FFFD),
    )
    symbols: List[str] = []
    for start, end in ranges:
        for codepoint in range(start, end + 1):
            symbols.append(chr(codepoint))
            if len(symbols) == count:
                return symbols
    raise ValueError(
        f"atom alphabet of {count} exceeds private-use symbol capacity"
    )


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text through a sibling temporary file and atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_text_sync(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _artifact_descriptor(path: Path) -> Dict[str, Any]:
    sha256, size = _sha256_file(path)
    return {"sha256": sha256, "size": size}


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _apply_ranked_merges(
    tokens: Sequence[str],
    spans: Sequence[Span],
    merge_ranks: Mapping[Pair, int],
) -> Tuple[List[str], List[Span]]:
    """Apply BPE with a pair-rank heap and mutable token adjacency."""
    if not tokens:
        return [], []
    token_values = list(tokens)
    span_values = list(spans)
    token_count = len(token_values)
    previous = [index - 1 for index in range(token_count)]
    following = [
        index + 1 if index + 1 < token_count else -1
        for index in range(token_count)
    ]
    alive = [True] * token_count
    candidates: List[Tuple[int, int, int, str, str]] = []

    def push_candidate(left_index: int) -> None:
        if left_index < 0 or not alive[left_index]:
            return
        right_index = following[left_index]
        if right_index < 0 or not alive[right_index]:
            return
        left_token = token_values[left_index]
        right_token = token_values[right_index]
        rank = merge_ranks.get((left_token, right_token))
        if rank is not None:
            heapq.heappush(
                candidates,
                (rank, left_index, right_index, left_token, right_token),
            )

    for index in range(token_count - 1):
        push_candidate(index)

    while candidates:
        rank, left_index, right_index, left_token, right_token = heapq.heappop(
            candidates
        )
        if (
            not alive[left_index]
            or not alive[right_index]
            or following[left_index] != right_index
            or token_values[left_index] != left_token
            or token_values[right_index] != right_token
            or merge_ranks.get((left_token, right_token)) != rank
        ):
            continue

        token_values[left_index] = left_token + right_token
        span_values[left_index] = (
            span_values[left_index][0],
            span_values[right_index][1],
        )
        alive[right_index] = False
        new_right = following[right_index]
        following[left_index] = new_right
        if new_right >= 0:
            previous[new_right] = left_index
        push_candidate(previous[left_index])
        push_candidate(left_index)

    merged_tokens: List[str] = []
    merged_spans: List[Span] = []
    index = 0
    while index >= 0:
        if alive[index]:
            merged_tokens.append(token_values[index])
            merged_spans.append(span_values[index])
        index = following[index]
    return merged_tokens, merged_spans


class ESPFTokenizer:
    """SMILES tokenizer with an in-project deterministic atom-wise BPE backend.

    ``vocab_path`` may be a tokenizer directory or a legacy ``vocab.json``.
    A legacy vocabulary without sibling merge/config files remains loadable and
    simply performs atom tokenization with no BPE merges.
    """

    def __init__(self, vocab_path: Optional[PathLike] = None) -> None:
        self.special_tokens = list(SPECIAL_TOKENS)
        self.vocab: Dict[str, int] = {
            token: index for index, token in enumerate(self.special_tokens)
        }
        self.inv_vocab: Dict[int, str] = {
            index: token for token, index in self.vocab.items()
        }
        self.merges: List[Pair] = []
        self._merge_ranks: Dict[Pair, int] = {}
        self.metadata: Dict[str, Any] = self._default_metadata()
        self.generation_id: Optional[str] = None
        self.artifact_hashes: Dict[str, str] = {}
        if vocab_path is not None:
            self._load_path(Path(vocab_path))

    @staticmethod
    def _default_metadata() -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "atom-wise-bpe",
            "min_frequency": None,
            "max_merges": 0,
            "target_vocab_size": None,
            "vocab_size": len(SPECIAL_TOKENS),
            "num_merges": 0,
            "corpus_sha256": None,
            "corpus_size": 0,
            "atom_token_count": 0,
            "tie_breaking": "frequency-desc,rust-token-id-ascending",
            "training_backend": None,
            "encoding_algorithm": "pair-rank-heap",
        }

    # ------------------------------------------------------------------
    # Lossless SMILES lexical split
    # ------------------------------------------------------------------
    @staticmethod
    def atom_tokenize_with_spans(smiles: str) -> Tuple[List[str], List[Span]]:
        """Split a SMILES string without losing any character.

        Bracket atoms are kept intact.  ``Br``/``Cl``, atoms, bonds, ring
        closures, branches, dot separators, and stereochemical markers are
        recognized explicitly.  Whitespace, non-ASCII input, incomplete
        brackets/ring markers, and all other uncovered characters raise
        :class:`TokenizationError` with the exact character offset.
        """
        if not isinstance(smiles, str):
            raise TypeError(f"smiles must be str, got {type(smiles).__name__}")

        tokens: List[str] = []
        spans: List[Span] = []
        offset = 0
        while offset < len(smiles):
            character = smiles[offset]

            if character == "[":
                closing = smiles.find("]", offset + 1)
                if closing < 0:
                    raise TokenizationError(
                        f"unterminated bracket atom at offset {offset}"
                    )
                content = smiles[offset + 1 : closing]
                invalid_index = next(
                    (
                        index
                        for index, item in enumerate(content)
                        if item.isspace()
                        or ord(item) < 33
                        or ord(item) > 126
                        or item in "[]"
                    ),
                    None,
                )
                if not content:
                    raise TokenizationError(
                        f"empty bracket atom at offset {offset}"
                    )
                if invalid_index is not None:
                    absolute = offset + 1 + invalid_index
                    raise TokenizationError(
                        f"invalid bracket-atom character at offset {absolute}"
                    )
                end = closing + 1
                bracket_token = smiles[offset:end]
                if bracket_token in SPECIAL_TOKENS:
                    raise TokenizationError(
                        f"reserved special token at offset {offset}"
                    )
                tokens.append(bracket_token)
                spans.append((offset, end))
                offset = end
                continue

            two_characters = smiles[offset : offset + 2]
            if two_characters in _TWO_CHARACTER_ATOMS:
                tokens.append(two_characters)
                spans.append((offset, offset + 2))
                offset += 2
                continue

            if character in _ONE_CHARACTER_ATOMS:
                tokens.append(character)
                spans.append((offset, offset + 1))
                offset += 1
                continue

            if character == "@":
                tokens.append(character)
                spans.append((offset, offset + 1))
                offset += 1
                continue

            if character.isdigit() and character.isascii():
                tokens.append(character)
                spans.append((offset, offset + 1))
                offset += 1
                continue

            if character == "%":
                if (
                    offset + 2 < len(smiles)
                    and smiles[offset + 1 : offset + 3].isdigit()
                    and smiles[offset + 1 : offset + 3].isascii()
                ):
                    end = offset + 3
                elif offset + 4 < len(smiles) and smiles[offset + 1] == "(":
                    closing = smiles.find(")", offset + 2)
                    digits = (
                        smiles[offset + 2 : closing] if closing >= 0 else ""
                    )
                    if (
                        closing < 0
                        or len(digits) < 3
                        or not digits.isdigit()
                        or not digits.isascii()
                    ):
                        raise TokenizationError(
                            f"invalid extended ring number at offset {offset}"
                        )
                    end = closing + 1
                else:
                    raise TokenizationError(
                        f"incomplete ring number at offset {offset}"
                    )
                tokens.append(smiles[offset:end])
                spans.append((offset, end))
                offset = end
                continue

            if character in _ONE_CHARACTER_SYNTAX:
                tokens.append(character)
                spans.append((offset, offset + 1))
                offset += 1
                continue

            raise TokenizationError(
                f"unrecognized SMILES character {character!r} at offset {offset}"
            )

        if "".join(tokens) != smiles:
            raise TokenizationError("internal lossless-tokenization invariant failed")
        return tokens, spans

    @staticmethod
    def atom_tokenize(smiles: str) -> List[str]:
        """Return only the atom/syntax tokens from the lossless lexical split."""
        tokens, _ = ESPFTokenizer.atom_tokenize_with_spans(smiles)
        return tokens

    # Compatibility aliases for callers that use pre-tokenizer terminology.
    pre_tokenize = atom_tokenize
    tokenize_atoms = atom_tokenize

    def _bpe_tokenize_with_spans(self, smiles: str) -> Tuple[List[str], List[Span]]:
        tokens, spans = self.atom_tokenize_with_spans(smiles)
        tokens, spans = _apply_ranked_merges(
            tokens,
            spans,
            self._merge_ranks,
        )
        if "".join(tokens) != smiles:
            raise TokenizationError("BPE merges did not reconstruct the input SMILES")
        return tokens, spans

    def tokenize(self, smiles: str) -> List[str]:
        """Apply the lossless lexical split followed by ordered BPE merges."""
        tokens, _ = self._bpe_tokenize_with_spans(smiles)
        return tokens

    # ------------------------------------------------------------------
    # Streaming deterministic BPE training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        smiles_corpus: Iterable[str],
        min_frequency: int = 2,
        max_merges: Optional[int] = 1000,
        vocab_size: Optional[int] = None,
        temp_dir: Optional[PathLike] = None,
    ) -> "ESPFTokenizer":
        """Train with the high-performance Rust ``tokenizers`` BPE trainer.

        The Python source iterable is consumed exactly once into a temporary
        atom-token JSONL spool while collecting the finite atom alphabet.
        Each atom token is then mapped, in lexical order, to one private-use
        Unicode scalar.  Rust BPE therefore sees one character per atom and
        cannot split across the lossless pre-tokenization boundary.  Its
        incremental occurrence index and priority queue avoid a Python
        full-corpus scan/rewrite for every merge.
        """
        if isinstance(smiles_corpus, (str, bytes)):
            raise TypeError("smiles_corpus must be an iterable of SMILES strings")
        if temp_dir is not None and not isinstance(
            temp_dir, (str, os.PathLike)
        ):
            raise TypeError("temp_dir must be path-like or None")
        validated_min_frequency = _require_int(
            "min_frequency",
            min_frequency,
            minimum=1,
            allow_none=False,
        )
        validated_max_merges = _require_int(
            "max_merges",
            max_merges,
            minimum=0,
            allow_none=True,
        )
        validated_vocab_size = _require_int(
            "vocab_size",
            vocab_size,
            minimum=len(SPECIAL_TOKENS),
            allow_none=True,
        )
        if validated_min_frequency is None:
            raise TypeError("min_frequency must be an integer")
        if max_merges is None and vocab_size is None:
            raise ValueError("at least one of max_merges or vocab_size is required")

        temporary_parent = str(temp_dir) if temp_dir is not None else None
        with tempfile.TemporaryDirectory(
            prefix="semmol_espf_", dir=temporary_parent
        ) as temporary_directory:
            corpus_path = Path(temporary_directory) / "atoms.jsonl"
            atom_alphabet: set[str] = set()
            corpus_hash = hashlib.sha256()
            corpus_size = 0
            atom_token_count = 0

            with corpus_path.open("w", encoding="utf-8", newline="\n") as handle:
                for smiles in smiles_corpus:
                    tokens = cls.atom_tokenize(smiles)
                    handle.write(
                        json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))
                    )
                    handle.write("\n")
                    atom_alphabet.update(tokens)
                    atom_token_count += len(tokens)
                    corpus_size += 1
                    corpus_hash.update(smiles.encode("utf-8"))
                    corpus_hash.update(b"\n")

            base_tokens = sorted(atom_alphabet)
            base_vocab_size = len(SPECIAL_TOKENS) + len(base_tokens)
            if (
                validated_vocab_size is not None
                and validated_vocab_size < base_vocab_size
            ):
                raise ValueError(
                    "vocab_size is smaller than the lossless corpus alphabet: "
                    f"requested {validated_vocab_size}, "
                    f"need at least {base_vocab_size}"
                )

            merge_budget = validated_max_merges
            if validated_vocab_size is not None:
                vocab_budget = validated_vocab_size - base_vocab_size
                merge_budget = (
                    vocab_budget
                    if merge_budget is None
                    else min(merge_budget, vocab_budget)
                )
            if merge_budget is None:
                raise ValueError("could not derive a finite merge budget")

            merges, backend_version = cls._train_rust_bpe(
                corpus_path=corpus_path,
                base_tokens=base_tokens,
                corpus_size=corpus_size,
                min_frequency=validated_min_frequency,
                max_merges=merge_budget,
                temporary_directory=Path(temporary_directory),
            )

        tokenizer = cls()
        ordered_tokens = list(SPECIAL_TOKENS)
        ordered_tokens.extend(base_tokens)
        known_tokens = set(ordered_tokens)
        for left, right in merges:
            merged = left + right
            if merged not in known_tokens:
                ordered_tokens.append(merged)
                known_tokens.add(merged)
        tokenizer.vocab = {
            token: index for index, token in enumerate(ordered_tokens)
        }
        tokenizer.inv_vocab = {
            index: token for token, index in tokenizer.vocab.items()
        }
        tokenizer.merges = merges
        tokenizer._merge_ranks = {
            pair: rank for rank, pair in enumerate(merges)
        }
        tokenizer.metadata = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "atom-wise-bpe",
            "min_frequency": validated_min_frequency,
            "max_merges": validated_max_merges,
            "target_vocab_size": validated_vocab_size,
            "vocab_size": len(tokenizer.vocab),
            "num_merges": len(merges),
            "corpus_sha256": corpus_hash.hexdigest(),
            "corpus_size": corpus_size,
            "atom_token_count": atom_token_count,
            "tie_breaking": (
                "frequency-desc,rust-token-id-ascending"
            ),
            "training_backend": "tokenizers-rust-bpe",
            "training_backend_version": backend_version,
            "encoding_algorithm": "pair-rank-heap",
        }
        tokenizer._validate_state()
        return tokenizer

    @staticmethod
    def _train_rust_bpe(
        corpus_path: Path,
        base_tokens: Sequence[str],
        corpus_size: int,
        min_frequency: int,
        max_merges: int,
        temporary_directory: Path,
    ) -> Tuple[List[Pair], str]:
        if max_merges == 0 or not base_tokens:
            try:
                import tokenizers
            except ImportError as error:
                raise RuntimeError(
                    "ESPF BPE training requires the project's tokenizers "
                    "Rust dependency"
                ) from error
            return [], str(getattr(tokenizers, "__version__", "unknown"))

        try:
            import tokenizers
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
        except ImportError as error:
            raise RuntimeError(
                "ESPF BPE training requires the project's tokenizers "
                "Rust dependency"
            ) from error

        symbols = _private_use_symbols(len(base_tokens))
        atom_to_symbol = dict(zip(base_tokens, symbols))
        symbol_to_atom = dict(zip(symbols, base_tokens))

        def encoded_corpus() -> Iterable[str]:
            with corpus_path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    try:
                        raw_tokens = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise TokenizerFormatError(
                            "invalid atom spool JSON at "
                            f"line {line_number}: {error}"
                        ) from error
                    if not isinstance(raw_tokens, list) or not all(
                        isinstance(token, str) for token in raw_tokens
                    ):
                        raise TokenizerFormatError(
                            f"invalid atom spool at line {line_number}"
                        )
                    yield "".join(atom_to_symbol[token] for token in raw_tokens)

        non_special_vocab_size = len(base_tokens) + max_merges
        backend = Tokenizer(BPE())
        trainer = BpeTrainer(
            vocab_size=non_special_vocab_size,
            min_frequency=min_frequency,
            show_progress=False,
            special_tokens=[],
            initial_alphabet=symbols,
        )
        backend.train_from_iterator(
            encoded_corpus(),
            trainer=trainer,
            length=corpus_size,
        )

        model_directory = temporary_directory / "rust_model"
        model_directory.mkdir()
        saved_files = backend.model.save(str(model_directory), "espf")
        if not isinstance(saved_files, list) or not all(
            isinstance(saved_file, str) for saved_file in saved_files
        ):
            raise TokenizerFormatError(
                "Rust BPE backend returned an invalid saved-file list"
            )
        merges_candidates: List[Path] = []
        for saved_file in saved_files:
            candidate = Path(saved_file)
            if not candidate.is_absolute() and not candidate.is_file():
                candidate = model_directory / candidate.name
            if str(candidate).endswith("merges.txt"):
                merges_candidates.append(candidate)
        if len(merges_candidates) != 1:
            raise TokenizerFormatError(
                "Rust BPE backend did not produce exactly one merges.txt"
            )

        def decode_symbol_token(symbol_token: str) -> str:
            try:
                return "".join(
                    symbol_to_atom[character] for character in symbol_token
                )
            except KeyError as error:
                raise TokenizerFormatError(
                    "Rust BPE emitted a token outside the atom symbol alphabet"
                ) from error

        merges: List[Pair] = []
        for line_number, line in enumerate(
            merges_candidates[0].read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise TokenizerFormatError(
                    "malformed Rust merge at "
                    f"line {line_number}: {line!r}"
                )
            merges.append(
                (
                    decode_symbol_token(fields[0]),
                    decode_symbol_token(fields[1]),
                )
            )
        return (
            merges[:max_merges],
            str(getattr(tokenizers, "__version__", "unknown")),
        )

    def build_vocab(
        self,
        smiles_list: Iterable[str],
        min_freq: int = 1,
        max_merges: Optional[int] = 1000,
        vocab_size: Optional[int] = None,
        temp_dir: Optional[PathLike] = None,
    ) -> Dict[str, int]:
        """Compatibility mutator that trains this instance and returns its vocab."""
        trained = type(self).train(
            smiles_list,
            min_frequency=min_freq,
            max_merges=max_merges,
            vocab_size=vocab_size,
            temp_dir=temp_dir,
        )
        self.vocab = trained.vocab
        self.inv_vocab = trained.inv_vocab
        self.merges = trained.merges
        self._merge_ranks = trained._merge_ranks
        self.metadata = trained.metadata
        self.generation_id = None
        self.artifact_hashes = {}
        return self.vocab

    # ------------------------------------------------------------------
    # Encoding and decoding
    # ------------------------------------------------------------------
    def encode_plus(
        self,
        smiles: str,
        max_length: Optional[int] = _OMITTED_NONE,
        add_special_tokens: bool = _OMITTED_TRUE,
        padding: Optional[bool] = None,
        *,
        max_len: Optional[int] = _OMITTED_NONE,
        add_special: Optional[bool] = _OMITTED_NONE,
    ) -> EncodedSmiles:
        """Encode SMILES into IDs, attention mask, and source character spans.

        Special and padding positions use the sentinel span ``(-1, -1)``.
        Truncation removes only content tokens and always retains ``[SEP]``;
        with a length limit of one the sole output token is therefore
        ``[SEP]``.  ``max_length=None`` returns an unpadded, untruncated
        sequence.  The ``max_len`` and ``add_special`` keywords preserve the
        legacy SemMol API; specifying ``max_len`` enables padding by default.
        """
        max_length_supplied = max_length is not _OMITTED_NONE
        max_len_supplied = max_len is not _OMITTED_NONE
        add_special_tokens_supplied = (
            add_special_tokens is not _OMITTED_TRUE
        )
        add_special_supplied = add_special is not _OMITTED_NONE
        raw_max_length = max_length if max_length_supplied else None
        raw_max_len = max_len if max_len_supplied else None
        raw_add_special_tokens = (
            add_special_tokens if add_special_tokens_supplied else True
        )
        raw_add_special = add_special if add_special_supplied else None

        validated_max_length = _require_int(
            "max_length",
            raw_max_length,
            minimum=1,
            allow_none=True,
        )
        validated_max_len = _require_int(
            "max_len",
            raw_max_len,
            minimum=1,
            allow_none=True,
        )
        if not isinstance(raw_add_special_tokens, bool):
            raise TypeError("add_special_tokens must be bool")
        if raw_add_special is not None and not isinstance(
            raw_add_special, bool
        ):
            raise TypeError("add_special must be bool or None")
        if padding is not None and not isinstance(padding, bool):
            raise TypeError("padding must be bool or None")
        if (
            max_length_supplied
            and max_len_supplied
            and validated_max_length != validated_max_len
        ):
            raise ValueError("max_length and max_len disagree")
        if max_length_supplied:
            length_limit = validated_max_length
        elif max_len_supplied:
            length_limit = validated_max_len
        else:
            length_limit = None
        if (
            add_special_tokens_supplied
            and add_special_supplied
            and raw_add_special_tokens != raw_add_special
        ):
            raise ValueError(
                "add_special_tokens and add_special disagree"
            )
        resolved_add_special = (
            raw_add_special_tokens
            if add_special_tokens_supplied
            else raw_add_special
            if add_special_supplied
            else True
        )
        if padding is None:
            padding = length_limit is not None
        if padding and length_limit is None:
            raise ValueError("padding requires max_length or max_len")
        tokens, spans = self._bpe_tokenize_with_spans(smiles)

        if resolved_add_special:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            spans = [(-1, -1)] + spans + [(-1, -1)]

        if length_limit is not None and len(tokens) > length_limit:
            if resolved_add_special:
                if length_limit == 1:
                    tokens = ["[SEP]"]
                    spans = [(-1, -1)]
                else:
                    content_budget = length_limit - 2
                    tokens = ["[CLS]"] + tokens[1:-1][:content_budget] + ["[SEP]"]
                    spans = [(-1, -1)] + spans[1:-1][:content_budget] + [
                        (-1, -1)
                    ]
            else:
                tokens = tokens[:length_limit]
                spans = spans[:length_limit]

        input_ids = [
            self.vocab.get(token, UNK_TOKEN_ID) for token in tokens
        ]
        attention_mask = [1] * len(input_ids)
        if (
            padding
            and length_limit is not None
            and len(input_ids) < length_limit
        ):
            padding_length = length_limit - len(input_ids)
            input_ids.extend([PAD_TOKEN_ID] * padding_length)
            attention_mask.extend([0] * padding_length)
            spans.extend([(-1, -1)] * padding_length)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_spans": spans,
        }

    def encode(
        self,
        smiles: str,
        max_len: Optional[int] = _OMITTED_128,
        add_special: bool = _OMITTED_TRUE,
        *,
        max_length: Optional[int] = _OMITTED_NONE,
        add_special_tokens: Optional[bool] = _OMITTED_NONE,
    ) -> List[int]:
        """Dataset-compatible shorthand returning padded token IDs only."""
        max_len_supplied = max_len is not _OMITTED_128
        max_length_supplied = max_length is not _OMITTED_NONE
        add_special_supplied = add_special is not _OMITTED_TRUE
        add_special_tokens_supplied = (
            add_special_tokens is not _OMITTED_NONE
        )
        raw_max_len = max_len if max_len_supplied else 128
        raw_max_length = max_length if max_length_supplied else None
        raw_add_special = add_special if add_special_supplied else True
        raw_add_special_tokens = (
            add_special_tokens if add_special_tokens_supplied else None
        )
        validated_max_len = _require_int(
            "max_len",
            raw_max_len,
            minimum=1,
            allow_none=True,
        )
        validated_max_length = _require_int(
            "max_length",
            raw_max_length,
            minimum=1,
            allow_none=True,
        )
        if not isinstance(raw_add_special, bool):
            raise TypeError("add_special must be bool")
        if raw_add_special_tokens is not None and not isinstance(
            raw_add_special_tokens, bool
        ):
            raise TypeError("add_special_tokens must be bool or None")
        if (
            max_len_supplied
            and max_length_supplied
            and validated_max_len != validated_max_length
        ):
            raise ValueError("max_len and max_length disagree")
        if (
            add_special_supplied
            and add_special_tokens_supplied
            and raw_add_special != raw_add_special_tokens
        ):
            raise ValueError(
                "add_special and add_special_tokens disagree"
            )
        effective_max_length: Optional[int] = (
            validated_max_length
            if max_length_supplied
            else validated_max_len
        )
        effective_add_special = (
            raw_add_special_tokens
            if add_special_tokens_supplied
            else raw_add_special
        )
        if not isinstance(effective_add_special, bool):
            raise TypeError("resolved special-token flag must be bool")
        return self.encode_plus(
            smiles,
            max_length=effective_max_length,
            add_special_tokens=effective_add_special,
            padding=effective_max_length is not None,
        )["input_ids"]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """Join token surfaces; known non-special sequences decode reversibly."""
        tokens: List[str] = []
        for raw_id in ids:
            token = self.inv_vocab.get(int(raw_id), "[UNK]")
            if skip_special and token in self.special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_pretrained(self, output_dir: PathLike) -> Dict[str, Path]:
        """Publish one immutable generation and atomically switch its manifest.

        Readers resolve only ``tokenizer_manifest.json``.  That manifest names
        one generation and carries SHA-256 plus byte size for every artifact,
        so interruption cannot expose a mixture of old and new files.
        """
        self._validate_state()
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        generations_directory = directory / "generations"
        generations_directory.mkdir(parents=True, exist_ok=True)

        normalized_metadata = dict(self.metadata)
        normalized_metadata.update(
            {
                "schema_version": SCHEMA_VERSION,
                "vocab_size": len(self.vocab),
                "num_merges": len(self.merges),
            }
        )
        self.metadata = normalized_metadata
        merge_lines = [MERGES_HEADER]
        merge_lines.extend(f"{left} {right}" for left, right in self.merges)
        artifact_text = {
            "vocab.json": _json_text(self.vocab),
            "merges.txt": "\n".join(merge_lines) + "\n",
            "tokenizer_config.json": _json_text(normalized_metadata),
            "metadata.json": _json_text(normalized_metadata),
        }

        generation_id: Optional[str] = None
        with tempfile.TemporaryDirectory(
            prefix=".staging-",
            dir=generations_directory,
        ) as staging_string:
            staging_directory = Path(staging_string)
            for name, text in artifact_text.items():
                _write_text_sync(staging_directory / name, text)

            artifact_descriptors = {
                name: _artifact_descriptor(staging_directory / name)
                for name in sorted(artifact_text)
            }
            generation_manifest = {
                "schema_version": SCHEMA_VERSION,
                "artifacts": artifact_descriptors,
            }
            generation_manifest_text = _json_text(generation_manifest)
            generation_manifest_path = (
                staging_directory / ARTIFACT_MANIFEST_NAME
            )
            _write_text_sync(
                generation_manifest_path,
                generation_manifest_text,
            )
            generation_id = hashlib.sha256(
                generation_manifest_text.encode("utf-8")
            ).hexdigest()
            final_generation = generations_directory / generation_id

            if final_generation.exists():
                if not final_generation.is_dir():
                    raise TokenizerFormatError(
                        "content-addressed generation path is not a directory"
                    )
                expected_entries = set(artifact_text) | {
                    ARTIFACT_MANIFEST_NAME
                }
                actual_entries = {
                    entry.name for entry in final_generation.iterdir()
                }
                if actual_entries != expected_entries:
                    raise TokenizerFormatError(
                        "existing generation contains missing or unknown "
                        "artifacts"
                    )
                existing_manifest = final_generation / ARTIFACT_MANIFEST_NAME
                existing_hash, _ = _sha256_file(existing_manifest)
                if existing_hash != generation_id:
                    raise TokenizerFormatError(
                        "existing content-addressed generation is corrupt"
                    )
                for name, descriptor in artifact_descriptors.items():
                    existing_descriptor = _artifact_descriptor(
                        final_generation / name
                    )
                    if existing_descriptor != descriptor:
                        raise TokenizerFormatError(
                            "existing generation artifact differs from "
                            f"content-addressed save: {name}"
                        )
            else:
                os.replace(staging_directory, final_generation)
                _fsync_directory(generations_directory)

        if generation_id is None:
            raise TokenizerFormatError("failed to derive tokenizer generation")
        generation_directory = generations_directory / generation_id
        generation_names = sorted(
            list(artifact_text) + [ARTIFACT_MANIFEST_NAME]
        )
        root_artifacts: Dict[str, Dict[str, Any]] = {}
        for name in generation_names:
            descriptor = _artifact_descriptor(generation_directory / name)
            descriptor["path"] = (
                Path("generations") / generation_id / name
            ).as_posix()
            root_artifacts[name] = descriptor
        root_manifest = {
            "schema_version": SCHEMA_VERSION,
            "generation_id": generation_id,
            "generation_sha256": generation_id,
            "artifacts": root_artifacts,
        }
        root_manifest_path = directory / ROOT_MANIFEST_NAME
        _atomic_write_text(root_manifest_path, _json_text(root_manifest))

        self.generation_id = generation_id
        self.artifact_hashes = {
            name: descriptor["sha256"]
            for name, descriptor in root_artifacts.items()
        }
        return {
            "tokenizer_manifest": root_manifest_path,
            "artifact_manifest": (
                generation_directory / ARTIFACT_MANIFEST_NAME
            ),
            "vocab": generation_directory / "vocab.json",
            "merges": generation_directory / "merges.txt",
            "tokenizer_config": (
                generation_directory / "tokenizer_config.json"
            ),
            "metadata": generation_directory / "metadata.json",
        }

    save = save_pretrained

    @classmethod
    def from_pretrained(cls, path: PathLike) -> "ESPFTokenizer":
        tokenizer = cls()
        tokenizer._load_path(Path(path))
        return tokenizer

    load = from_pretrained

    def _load_path(self, path: Path) -> None:
        generation_id: Optional[str] = None
        artifact_hashes: Dict[str, str] = {}
        if path.is_dir() and (path / ROOT_MANIFEST_NAME).is_file():
            (
                vocab_path,
                merges_path,
                config_path,
                metadata_path,
                generation_id,
                artifact_hashes,
            ) = self._resolve_generation(path, path / ROOT_MANIFEST_NAME)
            vocab = self._read_vocab(vocab_path)
            merges = self._read_merges(merges_path)
            metadata = self._read_metadata(config_path)
            duplicate_metadata = self._read_metadata(metadata_path)
            if duplicate_metadata != metadata:
                raise TokenizerFormatError(
                    "metadata.json does not match tokenizer_config.json"
                )
        elif path.is_file() and path.name == ROOT_MANIFEST_NAME:
            root = path.parent
            (
                vocab_path,
                merges_path,
                config_path,
                metadata_path,
                generation_id,
                artifact_hashes,
            ) = self._resolve_generation(root, path)
            vocab = self._read_vocab(vocab_path)
            merges = self._read_merges(merges_path)
            metadata = self._read_metadata(config_path)
            duplicate_metadata = self._read_metadata(metadata_path)
            if duplicate_metadata != metadata:
                raise TokenizerFormatError(
                    "metadata.json does not match tokenizer_config.json"
                )
        elif path.is_dir():
            vocab_path = path / "vocab.json"
            merges_path = path / "merges.txt"
            config_path = path / "tokenizer_config.json"
            metadata_path = path / "metadata.json"
            missing = [
                item.name
                for item in (vocab_path, merges_path, config_path)
                if not item.is_file()
            ]
            if missing:
                raise TokenizerFormatError(
                    f"tokenizer directory is missing: {', '.join(missing)}"
                )
            vocab = self._read_vocab(vocab_path)
            merges = self._read_merges(merges_path)
            metadata = self._read_metadata(config_path)
            if metadata_path.is_file():
                duplicate_metadata = self._read_metadata(metadata_path)
                if duplicate_metadata != metadata:
                    raise TokenizerFormatError(
                        "metadata.json does not match tokenizer_config.json"
                    )
        else:
            vocab_path = path
            vocab = self._read_vocab(vocab_path)
            sibling_merges = vocab_path.with_name("merges.txt")
            sibling_config = vocab_path.with_name("tokenizer_config.json")
            has_merges = sibling_merges.is_file()
            has_config = sibling_config.is_file()
            if has_merges != has_config:
                raise TokenizerFormatError(
                    "vocab has only one of sibling merges.txt and "
                    "tokenizer_config.json"
                )
            if has_merges:
                merges = self._read_merges(sibling_merges)
                metadata = self._read_metadata(sibling_config)
            else:
                merges = []
                metadata = self._default_metadata()
                metadata["vocab_size"] = len(vocab)

        self.vocab = vocab
        self.inv_vocab = {index: token for token, index in vocab.items()}
        self.merges = merges
        self._merge_ranks = {
            pair: rank for rank, pair in enumerate(merges)
        }
        self.metadata = metadata
        self.generation_id = generation_id
        self.artifact_hashes = artifact_hashes
        self._validate_state()

    @staticmethod
    def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TokenizerFormatError(
                f"cannot read {label} {path}: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise TokenizerFormatError(f"{label} must contain a JSON object")
        return dict(raw)

    @classmethod
    def _resolve_generation(
        cls,
        root: Path,
        manifest_path: Path,
    ) -> Tuple[Path, Path, Path, Path, str, Dict[str, str]]:
        manifest = cls._read_json_object(
            manifest_path,
            "tokenizer manifest",
        )
        if set(manifest) != {
            "schema_version",
            "generation_id",
            "generation_sha256",
            "artifacts",
        }:
            raise TokenizerFormatError(
                "tokenizer manifest has missing or unknown fields"
            )
        manifest_schema = manifest.get("schema_version")
        if (
            not isinstance(manifest_schema, int)
            or isinstance(manifest_schema, bool)
            or manifest_schema != SCHEMA_VERSION
        ):
            raise TokenizerFormatError(
                "unsupported tokenizer manifest schema_version"
            )
        generation_id = manifest.get("generation_id")
        generation_sha256 = manifest.get("generation_sha256")
        if not _valid_sha256(generation_id):
            raise TokenizerFormatError(
                "tokenizer manifest generation_id is not SHA-256"
            )
        if generation_sha256 != generation_id:
            raise TokenizerFormatError(
                "tokenizer manifest generation hash is inconsistent"
            )
        expected_names = {
            ARTIFACT_MANIFEST_NAME,
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "metadata.json",
        }
        raw_artifacts = manifest.get("artifacts")
        if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != expected_names:
            raise TokenizerFormatError(
                "tokenizer manifest must describe every required artifact"
            )
        generation_directory = root / "generations" / generation_id
        if not generation_directory.is_dir():
            raise TokenizerFormatError("tokenizer generation directory is missing")
        actual_entries = {entry.name for entry in generation_directory.iterdir()}
        if actual_entries != expected_names:
            raise TokenizerFormatError(
                "tokenizer generation contains missing or unknown artifacts"
            )

        resolved: Dict[str, Path] = {}
        artifact_hashes: Dict[str, str] = {}
        for name in sorted(expected_names):
            descriptor = raw_artifacts[name]
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"path", "sha256", "size"}
            ):
                raise TokenizerFormatError(
                    f"invalid artifact descriptor for {name}"
                )
            expected_relative = (
                Path("generations") / generation_id / name
            ).as_posix()
            if descriptor.get("path") != expected_relative:
                raise TokenizerFormatError(
                    f"artifact path escapes its generation: {name}"
                )
            expected_hash = descriptor.get("sha256")
            expected_size = descriptor.get("size")
            if not _valid_sha256(expected_hash):
                raise TokenizerFormatError(
                    f"invalid artifact SHA-256 for {name}"
                )
            if (
                not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size < 0
            ):
                raise TokenizerFormatError(
                    f"invalid artifact byte size for {name}"
                )
            artifact_path = root / Path(expected_relative)
            if not artifact_path.is_file():
                raise TokenizerFormatError(
                    f"generation artifact is missing: {name}"
                )
            actual_hash, actual_size = _sha256_file(artifact_path)
            if actual_hash != expected_hash:
                raise TokenizerFormatError(
                    f"artifact hash mismatch: {name}"
                )
            if actual_size != expected_size:
                raise TokenizerFormatError(
                    f"artifact size mismatch: {name}"
                )
            resolved[name] = artifact_path
            artifact_hashes[name] = actual_hash

        artifact_manifest_path = resolved[ARTIFACT_MANIFEST_NAME]
        if artifact_hashes[ARTIFACT_MANIFEST_NAME] != generation_id:
            raise TokenizerFormatError(
                "artifact manifest hash does not match generation_id"
            )
        generation_manifest = cls._read_json_object(
            artifact_manifest_path,
            "artifact manifest",
        )
        if set(generation_manifest) != {"schema_version", "artifacts"}:
            raise TokenizerFormatError(
                "artifact manifest has missing or unknown fields"
            )
        generation_schema = generation_manifest.get("schema_version")
        if (
            not isinstance(generation_schema, int)
            or isinstance(generation_schema, bool)
            or generation_schema != SCHEMA_VERSION
        ):
            raise TokenizerFormatError(
                "unsupported artifact manifest schema_version"
            )
        generation_artifacts = generation_manifest.get("artifacts")
        expected_generation_artifacts = {
            name: {
                "sha256": raw_artifacts[name]["sha256"],
                "size": raw_artifacts[name]["size"],
            }
            for name in expected_names
            if name != ARTIFACT_MANIFEST_NAME
        }
        if generation_artifacts != expected_generation_artifacts:
            raise TokenizerFormatError(
                "artifact manifest does not match tokenizer manifest"
            )
        return (
            resolved["vocab.json"],
            resolved["merges.txt"],
            resolved["tokenizer_config.json"],
            resolved["metadata.json"],
            generation_id,
            artifact_hashes,
        )

    @staticmethod
    def _read_vocab(path: Path) -> Dict[str, int]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TokenizerFormatError(f"cannot read vocab {path}: {error}") from error
        if not isinstance(raw, dict):
            raise TokenizerFormatError("vocab.json must contain a JSON object")
        vocab: Dict[str, int] = {}
        for token, token_id in raw.items():
            if not isinstance(token, str) or not token:
                raise TokenizerFormatError("vocab token keys must be non-empty strings")
            if (
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
            ):
                raise TokenizerFormatError(
                    f"invalid id for vocab token {token!r}: {token_id!r}"
                )
            vocab[token] = token_id
        return vocab

    @staticmethod
    def _read_merges(path: Path) -> List[Pair]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise TokenizerFormatError(f"cannot read merges {path}: {error}") from error
        if not lines or lines[0] != MERGES_HEADER:
            raise TokenizerFormatError(
                f"merges.txt must start with {MERGES_HEADER!r}"
            )
        merges: List[Pair] = []
        for line_number, line in enumerate(lines[1:], start=2):
            if not line:
                raise TokenizerFormatError(
                    f"blank merge at merges.txt line {line_number}"
                )
            fields = line.split()
            if len(fields) != 2:
                raise TokenizerFormatError(
                    f"malformed merge at merges.txt line {line_number}"
                )
            merges.append((fields[0], fields[1]))
        return merges

    @staticmethod
    def _read_metadata(path: Path) -> Dict[str, Any]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TokenizerFormatError(
                f"cannot read tokenizer config {path}: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise TokenizerFormatError(
                "tokenizer_config.json must contain a JSON object"
            )
        required = {
            "schema_version",
            "min_frequency",
            "max_merges",
            "vocab_size",
            "corpus_sha256",
        }
        missing = sorted(required.difference(raw))
        if missing:
            raise TokenizerFormatError(
                "tokenizer config is missing: " + ", ".join(missing)
            )
        if (
            not isinstance(raw["schema_version"], int)
            or isinstance(raw["schema_version"], bool)
            or raw["schema_version"] != SCHEMA_VERSION
        ):
            raise TokenizerFormatError(
                f"unsupported schema_version {raw['schema_version']!r}"
            )
        min_frequency = raw["min_frequency"]
        if min_frequency is not None and (
            not isinstance(min_frequency, int)
            or isinstance(min_frequency, bool)
            or min_frequency < 1
        ):
            raise TokenizerFormatError("min_frequency must be a positive integer")
        max_merges = raw["max_merges"]
        if max_merges is not None and (
            not isinstance(max_merges, int)
            or isinstance(max_merges, bool)
            or max_merges < 0
        ):
            raise TokenizerFormatError("max_merges must be non-negative or null")
        vocab_size = raw["vocab_size"]
        if (
            not isinstance(vocab_size, int)
            or isinstance(vocab_size, bool)
            or vocab_size < len(SPECIAL_TOKENS)
        ):
            raise TokenizerFormatError("vocab_size must be an integer of at least 5")
        corpus_sha256 = raw["corpus_sha256"]
        if corpus_sha256 is not None and (
            not isinstance(corpus_sha256, str)
            or len(corpus_sha256) != 64
            or any(character not in "0123456789abcdef" for character in corpus_sha256)
        ):
            raise TokenizerFormatError(
                "corpus_sha256 must be null or 64 lowercase hexadecimal characters"
            )
        return dict(raw)

    def load_vocab(self, vocab_path: PathLike) -> None:
        """Load a legacy vocab file or its complete sibling artifact set."""
        self._load_path(Path(vocab_path))

    def save_vocab(self, vocab_path: PathLike) -> None:
        """Atomically write only ``vocab.json`` for legacy callers."""
        self._validate_vocab()
        _atomic_write_text(Path(vocab_path), _json_text(self.vocab))

    def _validate_vocab(self) -> None:
        expected_specials = {
            token: index for index, token in enumerate(SPECIAL_TOKENS)
        }
        for token, expected_id in expected_specials.items():
            actual_id = self.vocab.get(token)
            if actual_id != expected_id:
                raise TokenizerFormatError(
                    f"{token} must have id {expected_id}, got {actual_id!r}"
                )
        ids = list(self.vocab.values())
        if len(ids) != len(set(ids)):
            raise TokenizerFormatError("vocab IDs must be unique")
        if sorted(ids) != list(range(len(ids))):
            raise TokenizerFormatError("vocab IDs must be contiguous from zero")

    def _validate_state(self) -> None:
        self._validate_vocab()
        expected_ranks = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        if self._merge_ranks != expected_ranks:
            raise TokenizerFormatError(
                "in-memory merge rank index does not match merges"
            )
        merge_results = [left + right for left, right in self.merges]
        result_set = set(merge_results)
        known_tokens = set(self.vocab).difference(result_set)
        known_tokens.difference_update(SPECIAL_TOKENS)
        seen_pairs: set[Pair] = set()
        for merge_index, pair in enumerate(self.merges):
            if pair in seen_pairs:
                raise TokenizerFormatError(
                    f"duplicate merge {pair!r} at index {merge_index}"
                )
            seen_pairs.add(pair)
            left, right = pair
            if left not in known_tokens:
                raise TokenizerFormatError(
                    f"merge {merge_index} references unknown left token {left!r}"
                )
            if right not in known_tokens:
                raise TokenizerFormatError(
                    f"merge {merge_index} references unknown right token {right!r}"
                )
            merged = left + right
            if merged not in self.vocab:
                raise TokenizerFormatError(
                    f"merge {merge_index} result {merged!r} is absent from vocab"
                )
            known_tokens.add(merged)

        configured_vocab_size = self.metadata.get("vocab_size")
        if configured_vocab_size not in (None, len(self.vocab)):
            raise TokenizerFormatError(
                "tokenizer config vocab_size does not match vocab.json"
            )
        configured_num_merges = self.metadata.get("num_merges")
        if configured_num_merges not in (None, len(self.merges)):
            raise TokenizerFormatError(
                "tokenizer config num_merges does not match merges.txt"
            )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return PAD_TOKEN_ID

    @property
    def mask_token_id(self) -> int:
        return MASK_TOKEN_ID

    @property
    def unk_token_id(self) -> int:
        return UNK_TOKEN_ID

    @property
    def cls_token_id(self) -> int:
        return CLS_TOKEN_ID

    @property
    def sep_token_id(self) -> int:
        return SEP_TOKEN_ID

    @property
    def corpus_sha256(self) -> Optional[str]:
        """SHA-256 of the ordered training corpus, or ``None`` if untrained."""
        value = self.metadata.get("corpus_sha256")
        return value if isinstance(value, str) else None

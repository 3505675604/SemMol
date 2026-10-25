"""ChemBERTa and from-scratch RoBERTa encoders for ESPF-tokenized SMILES."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    RobertaConfig,
    RobertaModel,
)

from src.models.encoders.common import (
    EncoderOutput,
    compact_sample_index,
    validate_encoder_output,
)
from src.molecular.espf_tokenizer import (
    CLS_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    SEP_TOKEN_ID,
    SPECIAL_TOKENS,
    UNK_TOKEN_ID,
    ESPFTokenizer,
)


PathLike = Union[str, Path]

PRETRAINED_ADAPTER_MODE = "pretrained_adapter"
SCRATCH_MODE = "scratch"
_SUPPORTED_MODES = frozenset((PRETRAINED_ADAPTER_MODE, SCRATCH_MODE))
_SUPPORTED_POOLING = frozenset(("cls", "mean"))

_ESPF_SPECIAL_TOKEN_IDS: Mapping[str, int] = {
    "[PAD]": PAD_TOKEN_ID,
    "[UNK]": UNK_TOKEN_ID,
    "[CLS]": CLS_TOKEN_ID,
    "[SEP]": SEP_TOKEN_ID,
    "[MASK]": MASK_TOKEN_ID,
}


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint is not a strictly compatible RoBERTa encoder."""


def _require_positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


def _require_probability(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1), got {value}")
    return result


def _normalize_path(value: Optional[PathLike]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError("model and cache paths must be strings or pathlib.Path")
    normalized = str(value)
    if not normalized:
        raise ValueError("model and cache paths cannot be empty")
    return normalized


def _native_special_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, int]:
    candidates: Mapping[str, Sequence[str]] = {
        "[PAD]": ("pad_token_id",),
        "[UNK]": ("unk_token_id",),
        "[CLS]": ("cls_token_id", "bos_token_id"),
        "[SEP]": ("sep_token_id", "eos_token_id"),
        "[MASK]": ("mask_token_id",),
    }
    resolved: Dict[str, int] = {}
    for espf_token, attributes in candidates.items():
        native_id: Optional[int] = None
        for attribute in attributes:
            value = getattr(tokenizer, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool):
                native_id = int(value)
                break
        if native_id is None:
            joined = " or ".join(attributes)
            raise CheckpointCompatibilityError(
                f"native tokenizer does not define {joined} for {espf_token}"
            )
        resolved[espf_token] = native_id
    return resolved


def _is_intentionally_discarded_head_key(key: str) -> bool:
    normalized = key.removeprefix("module.")
    return normalized.startswith(
        (
            "lm_head.",
            "cls.predictions.",
            "cls.seq_relationship.",
        )
    )


def _validate_loading_info(loading_info: Mapping[str, Any]) -> None:
    mismatched = tuple(loading_info.get("mismatched_keys", ()))
    missing = tuple(loading_info.get("missing_keys", ()))
    errors = tuple(loading_info.get("error_msgs", ()))
    unexpected = tuple(loading_info.get("unexpected_keys", ()))
    unsupported_unexpected = tuple(
        key
        for key in unexpected
        if not _is_intentionally_discarded_head_key(str(key))
    )

    problems = []
    if mismatched:
        problems.append(f"mismatched keys: {mismatched}")
    if missing:
        problems.append(f"missing keys: {missing}")
    if unsupported_unexpected:
        problems.append(f"unexpected keys: {unsupported_unexpected}")
    if errors:
        problems.append(f"loader errors: {errors}")
    if problems:
        raise CheckpointCompatibilityError(
            "checkpoint is not strictly compatible with RobertaModel; "
            + "; ".join(problems)
        )

    discarded = tuple(
        key for key in unexpected if _is_intentionally_discarded_head_key(str(key))
    )
    if discarded:
        warnings.warn(
            "Discarding the checkpoint's task-specific MLM head while loading "
            f"the encoder backbone: {discarded}",
            UserWarning,
            stacklevel=3,
        )


class SMILESEncoder(nn.Module):
    """Encode ESPF token IDs without conflating ESPF and native vocabularies.

    In ``pretrained_adapter`` mode, a dedicated ESPF embedding table is
    initialized by decomposing every ESPF token with the checkpoint's native
    tokenizer and averaging its pretrained word embeddings. The backbone then
    receives those vectors through ``inputs_embeds``. In ``scratch`` mode, the
    RoBERTa vocabulary is ESPF from the outset and its normal input embedding is
    trained directly.
    """

    def __init__(
        self,
        *,
        tokenizer_dir: PathLike,
        mode: str = SCRATCH_MODE,
        model_name_or_path: Optional[PathLike] = None,
        native_tokenizer_name_or_path: Optional[PathLike] = None,
        shared_dim: int = 512,
        pooling: str = "cls",
        freeze_layers: int = 0,
        validate_values: bool = False,
        local_files_only: bool = True,
        cache_dir: Optional[PathLike] = None,
        revision: str = "main",
        scratch_hidden_size: int = 768,
        scratch_num_hidden_layers: int = 12,
        scratch_num_attention_heads: int = 12,
        scratch_intermediate_size: int = 3072,
        scratch_max_position_embeddings: int = 258,
        scratch_hidden_act: str = "gelu",
        scratch_hidden_dropout_prob: float = 0.1,
        scratch_attention_probs_dropout_prob: float = 0.1,
        scratch_layer_norm_eps: float = 1.0e-12,
    ) -> None:
        super().__init__()
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_SUPPORTED_MODES)}, got {mode!r}"
            )
        if pooling not in _SUPPORTED_POOLING:
            raise ValueError(
                f"pooling must be one of {sorted(_SUPPORTED_POOLING)}, "
                f"got {pooling!r}"
            )
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be bool")
        if not isinstance(validate_values, bool):
            raise TypeError("validate_values must be bool")
        if not isinstance(revision, str) or not revision:
            raise ValueError("revision must be a non-empty string")

        self.mode = mode
        self.pooling = pooling
        self.validate_values = validate_values
        self.shared_dim = _require_positive_int("shared_dim", shared_dim)
        normalized_tokenizer_dir = _normalize_path(tokenizer_dir)
        if normalized_tokenizer_dir is None:
            raise ValueError("tokenizer_dir is required")
        self.espf_tokenizer = ESPFTokenizer(normalized_tokenizer_dir)
        self._validate_espf_special_tokens()
        self.vocab_size = self.espf_tokenizer.vocab_size

        normalized_model_path = _normalize_path(model_name_or_path)
        normalized_native_tokenizer_path = _normalize_path(
            native_tokenizer_name_or_path
        )
        normalized_cache_dir = _normalize_path(cache_dir)

        if mode == PRETRAINED_ADAPTER_MODE:
            if normalized_model_path is None:
                raise ValueError(
                    "model_name_or_path is required in pretrained_adapter mode"
                )
            (
                self.transformer,
                self.espf_embeddings,
                self.native_special_token_ids,
            ) = self._build_pretrained_adapter(
                model_name_or_path=normalized_model_path,
                native_tokenizer_name_or_path=(
                    normalized_native_tokenizer_path or normalized_model_path
                ),
                local_files_only=local_files_only,
                cache_dir=normalized_cache_dir,
                revision=revision,
            )
        else:
            if normalized_model_path is not None:
                raise ValueError(
                    "model_name_or_path must be omitted in scratch mode; "
                    "scratch mode never loads pretrained weights"
                )
            if normalized_native_tokenizer_path is not None:
                raise ValueError(
                    "native_tokenizer_name_or_path is invalid in scratch mode"
                )
            self.transformer = self._build_scratch_transformer(
                hidden_size=scratch_hidden_size,
                num_hidden_layers=scratch_num_hidden_layers,
                num_attention_heads=scratch_num_attention_heads,
                intermediate_size=scratch_intermediate_size,
                max_position_embeddings=scratch_max_position_embeddings,
                hidden_act=scratch_hidden_act,
                hidden_dropout_prob=scratch_hidden_dropout_prob,
                attention_probs_dropout_prob=(
                    scratch_attention_probs_dropout_prob
                ),
                layer_norm_eps=scratch_layer_norm_eps,
            )
            self.espf_embeddings = None
            self.native_special_token_ids = {}

        self.hidden_size = int(self.transformer.config.hidden_size)
        self.output_adapter = nn.Sequential(
            nn.Linear(self.hidden_size, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
        )
        self._freeze_first_encoder_layers(freeze_layers)

    @classmethod
    def from_pretrained_adapter(
        cls,
        *,
        tokenizer_dir: PathLike,
        model_name_or_path: PathLike,
        **kwargs: Any,
    ) -> "SMILESEncoder":
        if "mode" in kwargs:
            raise TypeError("mode is fixed by from_pretrained_adapter")
        return cls(
            tokenizer_dir=tokenizer_dir,
            mode=PRETRAINED_ADAPTER_MODE,
            model_name_or_path=model_name_or_path,
            **kwargs,
        )

    @classmethod
    def from_scratch(
        cls,
        *,
        tokenizer_dir: PathLike,
        **kwargs: Any,
    ) -> "SMILESEncoder":
        if "mode" in kwargs:
            raise TypeError("mode is fixed by from_scratch")
        if "model_name_or_path" in kwargs:
            raise TypeError("from_scratch does not accept model_name_or_path")
        return cls(
            tokenizer_dir=tokenizer_dir,
            mode=SCRATCH_MODE,
            **kwargs,
        )

    def _validate_espf_special_tokens(self) -> None:
        actual_special_tokens = tuple(
            self.espf_tokenizer.inv_vocab.get(index)
            for index in range(len(SPECIAL_TOKENS))
        )
        if actual_special_tokens != tuple(SPECIAL_TOKENS):
            raise ValueError(
                "ESPF tokenizer special-token order differs from the model contract"
            )
        for token, expected_id in _ESPF_SPECIAL_TOKEN_IDS.items():
            actual_id = self.espf_tokenizer.vocab.get(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"ESPF token {token} must have id {expected_id}, "
                    f"got {actual_id!r}"
                )

    @staticmethod
    def _load_roberta_checkpoint(
        *,
        model_name_or_path: str,
        local_files_only: bool,
        cache_dir: Optional[str],
        revision: str,
    ) -> RobertaModel:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            trust_remote_code=False,
        )
        if not isinstance(config, RobertaConfig) or config.model_type != "roberta":
            raise CheckpointCompatibilityError(
                "ChemBERTa adapter requires a RoBERTa checkpoint, got "
                f"{type(config).__name__} with model_type={config.model_type!r}"
            )
        if bool(config.is_decoder) or bool(config.add_cross_attention):
            raise CheckpointCompatibilityError(
                "ChemBERTa adapter requires an encoder-only RoBERTa checkpoint"
            )
        loaded = RobertaModel.from_pretrained(
            model_name_or_path,
            config=config,
            add_pooling_layer=False,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            ignore_mismatched_sizes=False,
            output_loading_info=True,
        )
        transformer, loading_info = loaded
        _validate_loading_info(loading_info)
        return transformer

    def _build_pretrained_adapter(
        self,
        *,
        model_name_or_path: str,
        native_tokenizer_name_or_path: str,
        local_files_only: bool,
        cache_dir: Optional[str],
        revision: str,
    ) -> Tuple[RobertaModel, nn.Embedding, Dict[str, int]]:
        transformer = self._load_roberta_checkpoint(
            model_name_or_path=model_name_or_path,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            revision=revision,
        )
        native_tokenizer = AutoTokenizer.from_pretrained(
            native_tokenizer_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            trust_remote_code=False,
            use_fast=True,
        )
        native_special_ids = _native_special_token_ids(native_tokenizer)
        native_word_embeddings = transformer.get_input_embeddings()
        if not isinstance(native_word_embeddings, nn.Embedding):
            raise CheckpointCompatibilityError(
                "RoBERTa checkpoint does not expose nn.Embedding word embeddings"
            )
        if native_word_embeddings.embedding_dim != transformer.config.hidden_size:
            raise CheckpointCompatibilityError(
                "checkpoint word-embedding width does not match config.hidden_size"
            )
        native_vocabulary = native_tokenizer.get_vocab()
        native_vocabulary_ids = tuple(native_vocabulary.values())
        if not native_vocabulary_ids or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in native_vocabulary_ids
        ):
            raise CheckpointCompatibilityError(
                "native tokenizer vocabulary must contain integer token IDs"
            )
        if len(set(native_vocabulary_ids)) != len(native_vocabulary_ids):
            raise CheckpointCompatibilityError(
                "native tokenizer vocabulary IDs must be unique"
            )
        if min(native_vocabulary_ids) < 0 or max(
            native_vocabulary_ids
        ) >= native_word_embeddings.num_embeddings:
            raise CheckpointCompatibilityError(
                "native tokenizer contains a token ID outside the checkpoint "
                f"embedding table with {native_word_embeddings.num_embeddings} rows"
            )

        for token, native_id in native_special_ids.items():
            if not 0 <= native_id < native_word_embeddings.num_embeddings:
                raise CheckpointCompatibilityError(
                    f"native special token {token} has out-of-range id {native_id}"
                )
        espf_embeddings = nn.Embedding(
            self.vocab_size,
            int(transformer.config.hidden_size),
            padding_idx=PAD_TOKEN_ID,
        )
        self._initialize_espf_embeddings(
            target=espf_embeddings,
            native_tokenizer=native_tokenizer,
            native_embeddings=native_word_embeddings,
            native_special_ids=native_special_ids,
        )

        for parameter in native_word_embeddings.parameters():
            parameter.requires_grad_(False)
        return transformer, espf_embeddings, native_special_ids

    def _initialize_espf_embeddings(
        self,
        *,
        target: nn.Embedding,
        native_tokenizer: PreTrainedTokenizerBase,
        native_embeddings: nn.Embedding,
        native_special_ids: Mapping[str, int],
    ) -> None:
        native_weight = native_embeddings.weight.detach()
        if not bool(torch.isfinite(native_weight).all()):
            raise CheckpointCompatibilityError(
                "checkpoint word embeddings contain non-finite values"
            )
        ordered_tokens = sorted(
            self.espf_tokenizer.vocab.items(),
            key=lambda item: item[1],
        )
        with torch.no_grad():
            for espf_token, espf_id in ordered_tokens:
                if espf_token in native_special_ids:
                    native_ids = [native_special_ids[espf_token]]
                else:
                    encoded = native_tokenizer.encode(
                        espf_token,
                        add_special_tokens=False,
                    )
                    native_ids = [int(token_id) for token_id in encoded]
                    if not native_ids:
                        native_ids = [native_special_ids["[UNK]"]]
                if any(
                    token_id < 0
                    or token_id >= native_embeddings.num_embeddings
                    for token_id in native_ids
                ):
                    raise CheckpointCompatibilityError(
                        f"native tokenizer decomposed ESPF token {espf_token!r} "
                        f"to an out-of-range id sequence {native_ids}"
                    )
                index = torch.tensor(
                    native_ids,
                    dtype=torch.long,
                    device=native_weight.device,
                )
                initialized = native_weight.index_select(0, index).mean(dim=0)
                target.weight[espf_id].copy_(
                    initialized.to(
                        device=target.weight.device,
                        dtype=target.weight.dtype,
                    )
                )

    def _build_scratch_transformer(
        self,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        max_position_embeddings: int,
        hidden_act: str,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        layer_norm_eps: float,
    ) -> RobertaModel:
        hidden_size = _require_positive_int("scratch_hidden_size", hidden_size)
        num_hidden_layers = _require_positive_int(
            "scratch_num_hidden_layers", num_hidden_layers
        )
        num_attention_heads = _require_positive_int(
            "scratch_num_attention_heads", num_attention_heads
        )
        intermediate_size = _require_positive_int(
            "scratch_intermediate_size", intermediate_size
        )
        max_position_embeddings = _require_positive_int(
            "scratch_max_position_embeddings",
            max_position_embeddings,
            minimum=PAD_TOKEN_ID + 3,
        )
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "scratch_hidden_size must be divisible by "
                "scratch_num_attention_heads"
            )
        if not isinstance(hidden_act, str) or not hidden_act:
            raise ValueError("scratch_hidden_act must be a non-empty string")
        if not isinstance(layer_norm_eps, (int, float)) or isinstance(
            layer_norm_eps, bool
        ):
            raise TypeError("scratch_layer_norm_eps must be a real number")
        layer_norm_eps = float(layer_norm_eps)
        if not math.isfinite(layer_norm_eps) or layer_norm_eps <= 0.0:
            raise ValueError("scratch_layer_norm_eps must be finite and positive")

        config = RobertaConfig(
            vocab_size=self.vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            hidden_dropout_prob=_require_probability(
                "scratch_hidden_dropout_prob", hidden_dropout_prob
            ),
            attention_probs_dropout_prob=_require_probability(
                "scratch_attention_probs_dropout_prob",
                attention_probs_dropout_prob,
            ),
            layer_norm_eps=layer_norm_eps,
            pad_token_id=PAD_TOKEN_ID,
            bos_token_id=CLS_TOKEN_ID,
            eos_token_id=SEP_TOKEN_ID,
            type_vocab_size=1,
        )
        return RobertaModel(config, add_pooling_layer=False)

    def _freeze_first_encoder_layers(self, freeze_layers: int) -> None:
        if not isinstance(freeze_layers, int) or isinstance(freeze_layers, bool):
            raise TypeError("freeze_layers must be an integer")
        layer_count = len(self.transformer.encoder.layer)
        if not 0 <= freeze_layers <= layer_count:
            raise ValueError(
                f"freeze_layers must be in [0, {layer_count}], "
                f"got {freeze_layers}"
            )
        for layer in self.transformer.encoder.layer[:freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)

    def _input_embedding_module(self) -> nn.Embedding:
        if self.espf_embeddings is not None:
            return self.espf_embeddings
        embedding = self.transformer.get_input_embeddings()
        if not isinstance(embedding, nn.Embedding):
            raise RuntimeError("scratch RoBERTa input embedding is not nn.Embedding")
        return embedding

    def _validate_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        modality_mask: Optional[Tensor],
    ) -> Tensor:
        if not isinstance(input_ids, Tensor) or not isinstance(
            attention_mask, Tensor
        ):
            raise TypeError("input_ids and attention_mask must be torch.Tensor")
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, length], got {input_ids.shape}"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids, got "
                f"{attention_mask.shape} and {input_ids.shape}"
            )
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")
        if attention_mask.dtype != torch.bool:
            raise TypeError(
                f"attention_mask must be torch.bool, got {attention_mask.dtype}"
            )
        if input_ids.device != attention_mask.device:
            raise ValueError("input_ids and attention_mask must share a device")
        embedding_device = self._input_embedding_module().weight.device
        if input_ids.device != embedding_device:
            raise ValueError(
                "input_ids and encoder parameters must share a device, got "
                f"{input_ids.device} and {embedding_device}"
            )
        if input_ids.shape[1] == 0:
            raise ValueError("SMILES sequence length cannot be zero")
        if self.validate_values:
            if bool(torch.any(input_ids < 0)) or bool(
                torch.any(input_ids >= self.vocab_size)
            ):
                raise ValueError(
                    f"input_ids must be in [0, {self.vocab_size - 1}]"
                )
            if bool(
                torch.any(
                    input_ids.masked_select(~attention_mask) != PAD_TOKEN_ID
                )
            ):
                raise ValueError(
                    "positions masked out by attention_mask must contain "
                    "ESPF [PAD]"
                )
            if bool(
                torch.any(
                    input_ids.masked_select(attention_mask) == PAD_TOKEN_ID
                )
            ):
                raise ValueError(
                    "active token positions cannot contain ESPF [PAD]"
                )
            if input_ids.shape[1] > 1 and bool(
                torch.any(attention_mask[:, 1:] & ~attention_mask[:, :-1])
            ):
                raise ValueError(
                    "attention_mask must use contiguous right padding"
                )

        valid_rows = attention_mask.any(dim=1)
        if modality_mask is not None:
            if not isinstance(modality_mask, Tensor):
                raise TypeError("modality_mask must be torch.Tensor or None")
            if modality_mask.shape != (input_ids.shape[0],):
                raise ValueError(
                    "modality_mask must have shape [batch], got "
                    f"{modality_mask.shape}"
                )
            if modality_mask.dtype != torch.bool:
                raise TypeError(
                    f"modality_mask must be torch.bool, got {modality_mask.dtype}"
                )
            if modality_mask.device != input_ids.device:
                raise ValueError(
                    "modality_mask and input_ids must share a device"
                )
            if self.validate_values and not bool(
                torch.equal(modality_mask, valid_rows)
            ):
                raise ValueError(
                    "modality_mask disagrees with rows present in attention_mask"
                )
        return valid_rows

    def _position_ids(self, compact_mask: Tensor) -> Tensor:
        padding_index = self.transformer.config.pad_token_id
        if not isinstance(padding_index, int):
            raise RuntimeError("RoBERTa config.pad_token_id must be an integer")
        mask_as_long = compact_mask.to(dtype=torch.long)
        position_ids = mask_as_long.cumsum(dim=1) * mask_as_long
        position_ids = position_ids + padding_index
        position_count = int(
            self.transformer.embeddings.position_embeddings.num_embeddings
        )
        maximum_position = padding_index + int(compact_mask.shape[1])
        if maximum_position >= position_count:
            raise ValueError(
                f"SMILES sequence width {compact_mask.shape[1]} exceeds the checkpoint's "
                f"position capacity {position_count - padding_index - 1}"
            )
        return position_ids

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        modality_mask: Optional[Tensor] = None,
    ) -> EncoderOutput:
        valid_rows = self._validate_inputs(
            input_ids,
            attention_mask,
            modality_mask,
        )
        sample_index = compact_sample_index(valid_rows)
        compact_size = int(sample_index.numel())
        sequence_length = int(input_ids.shape[1])
        if compact_size == 0:
            reference = self.output_adapter[0].weight
            output = EncoderOutput(
                global_embedding=torch.empty(
                    (0, self.shared_dim),
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                sample_index=sample_index,
                tokens=torch.empty(
                    (0, sequence_length, self.shared_dim),
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                token_mask=attention_mask.new_empty(
                    (0, sequence_length),
                    dtype=torch.bool,
                ),
            )
            validate_encoder_output(
                output,
                embedding_dim=self.shared_dim,
                batch_size=int(input_ids.shape[0]),
                check_values=self.validate_values,
            )
            return output

        compact_ids = input_ids.index_select(0, sample_index)
        compact_mask = attention_mask.index_select(0, sample_index)
        if self.validate_values and self.pooling == "cls" and not bool(
            torch.all(compact_ids[:, 0] == CLS_TOKEN_ID)
        ):
            raise ValueError(
                "CLS pooling requires ESPF [CLS] at the first active position"
            )
        position_ids = self._position_ids(compact_mask)

        transformer_arguments: Dict[str, Tensor] = {}
        if self.mode == PRETRAINED_ADAPTER_MODE:
            if self.espf_embeddings is None:
                raise RuntimeError(
                    "pretrained_adapter mode is missing ESPF embeddings"
                )
            transformer_arguments["inputs_embeds"] = self.espf_embeddings(
                compact_ids
            )
        else:
            transformer_arguments["input_ids"] = compact_ids

        transformer_output = self.transformer(
            attention_mask=compact_mask,
            position_ids=position_ids,
            return_dict=True,
            **transformer_arguments,
        )
        hidden = transformer_output.last_hidden_state
        if hidden.shape != (
            compact_size,
            sequence_length,
            self.hidden_size,
        ):
            raise RuntimeError(
                "RoBERTa returned an unexpected hidden-state shape: "
                f"{tuple(hidden.shape)}"
            )
        if self.validate_values and not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError(
                "RoBERTa produced non-finite token representations"
            )

        tokens = self.output_adapter(hidden)
        tokens = tokens.masked_fill(~compact_mask.unsqueeze(-1), 0.0)
        if self.pooling == "cls":
            global_embedding = tokens[:, 0]
        else:
            weights = compact_mask.unsqueeze(-1).to(dtype=tokens.dtype)
            global_embedding = (tokens * weights).sum(dim=1)
            global_embedding = global_embedding / weights.sum(dim=1).clamp_min(1.0)

        if self.validate_values and (
            not bool(torch.isfinite(tokens).all())
            or not bool(torch.isfinite(global_embedding).all())
        ):
            raise FloatingPointError(
                "SMILES output adapter produced non-finite representations"
            )
        output = EncoderOutput(
            global_embedding=global_embedding,
            sample_index=sample_index,
            tokens=tokens,
            token_mask=compact_mask,
        )
        validate_encoder_output(
            output,
            embedding_dim=self.shared_dim,
            batch_size=int(input_ids.shape[0]),
            check_values=self.validate_values,
        )
        return output


SmilesEncoder = SMILESEncoder


__all__ = [
    "CheckpointCompatibilityError",
    "PRETRAINED_ADAPTER_MODE",
    "SCRATCH_MODE",
    "SMILESEncoder",
    "SmilesEncoder",
]

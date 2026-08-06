from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.losses.common import LossComponent
from src.losses.total_loss import SemMolPretrainLossOutput
from src.models.encoders.smiles_encoder import (
    PRETRAINED_ADAPTER_MODE,
    SCRATCH_MODE,
)
from src.trainers.pretrain_trainer import PretrainTrainer, _validate_loss_output


def _trainer_with_padding_ids(
    *,
    mode: str,
    transformer_pad_token_id: int,
) -> PretrainTrainer:
    trainer = object.__new__(PretrainTrainer)
    trainer.context = SimpleNamespace(device=torch.device("cpu"))
    trainer._semmol = SimpleNamespace(
        encoders={
            "1d": SimpleNamespace(
                mode=mode,
                vocab_size=32,
                espf_tokenizer=SimpleNamespace(
                    pad_token_id=0,
                    cls_token_id=2,
                    sep_token_id=3,
                ),
                transformer=SimpleNamespace(
                    config=SimpleNamespace(
                        pad_token_id=transformer_pad_token_id,
                    ),
                    embeddings=SimpleNamespace(
                        position_embeddings=SimpleNamespace(
                            num_embeddings=514,
                        ),
                    ),
                ),
            ),
        },
    )
    trainer.loss_fn = SimpleNamespace(
        mlm=SimpleNamespace(ignore_index=-100),
    )
    return trainer


def _valid_1d_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[2, 5, 3, 0]], dtype=torch.long),
        "attention_mask": torch.tensor(
            [[True, True, True, False]],
            dtype=torch.bool,
        ),
        "mlm_labels": torch.full((1, 4), -100, dtype=torch.long),
    }


def test_pretrained_adapter_accepts_distinct_espf_and_roberta_padding_ids() -> None:
    trainer = _trainer_with_padding_ids(
        mode=PRETRAINED_ADAPTER_MODE,
        transformer_pad_token_id=1,
    )

    trainer._validate_1d_batch(
        _valid_1d_batch(),
        batch_size=1,
        expected_presence=torch.tensor([True]),
    )


def test_scratch_encoder_rejects_distinct_tokenizer_and_transformer_padding_ids() -> None:
    trainer = _trainer_with_padding_ids(
        mode=SCRATCH_MODE,
        transformer_pad_token_id=1,
    )

    with pytest.raises(
        ValueError,
        match="tokenizer and transformer pad-token identifiers differ",
    ):
        trainer._validate_1d_batch(
            _valid_1d_batch(),
            batch_size=1,
            expected_presence=torch.tensor([True]),
        )


def _loss_component(numerator: float) -> LossComponent:
    value = torch.tensor(numerator)
    count = torch.tensor(1, dtype=torch.long)
    return LossComponent(
        loss=value,
        numerator=value,
        local_count=count,
        global_count=count.clone(),
    )


def test_loss_validation_names_nonfinite_component_before_total() -> None:
    finite = _loss_component(1.0)
    output = SemMolPretrainLossOutput(
        total_loss=torch.tensor(float("nan")),
        mlm_loss=_loss_component(float("nan")),
        graph_loss=SimpleNamespace(
            node=finite,
            edge=finite,
            structure=finite,
        ),
        geo_loss=SimpleNamespace(mse=finite, direction=finite),
        pseudo_loss=finite,
        alignment_loss=finite,
        acsm=SimpleNamespace(pseudo_scale=torch.tensor(0.0)),
        component_counts={},
        weighted_losses={},
    )

    with pytest.raises(FloatingPointError, match="non-finite mlm numerator"):
        _validate_loss_output(output, device=torch.device("cpu"))

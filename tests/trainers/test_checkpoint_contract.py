"""Weight transfer must reject incompatible contracts before mutating tensors."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.semantic.dcl import DynamicCentralLibrary
from src.trainers.checkpointing import (
    _validate_training_payload, load_pretrained_semmol, model_contract_signature,
)


class TinyTransferModel(nn.Module):
    def __init__(self, *, shared: bool = True, property_head: bool = False):
        super().__init__()
        self.modalities = ("1d", "2d")
        self.anchor_modality = "1d"
        self.encoder_dim = self.feature_dim = 4
        self.body = nn.Linear(4, 4)
        self.projection = nn.Linear(4, 4)
        self.projection.share_mlp_weights = shared
        self.dcls = nn.ModuleDict({"2d": DynamicCentralLibrary(
            num_clusters=2, feature_dim=4, init_max_samples=8,
            init_num_iters=2, distributed_sync=False,
        )})
        self.dcls["2d"].update_centers(torch.randn(8, 4))
        if property_head:
            self.property_head = nn.Linear(8, 1)


def _snapshot(model):
    return {key: tensor.clone() for key, tensor in model.state_dict().items()}


def test_legacy_checkpoint_is_rejected_without_mutating_model(tmp_path):
    source, target = TinyTransferModel(shared=False), TinyTransferModel(property_head=True)
    path = tmp_path / "legacy.pt"
    torch.save(source.state_dict(), path)
    before = _snapshot(target)
    with pytest.raises(RuntimeError, match="signature.*Legacy|signature",):
        load_pretrained_semmol(path, target, map_location="cpu")
    for key, tensor in before.items():
        assert torch.equal(target.state_dict()[key], tensor)


def test_valid_signed_body_transfers_but_preserves_fresh_property_head(tmp_path):
    source, target = TinyTransferModel(), TinyTransferModel(property_head=True)
    path = tmp_path / "current.pt"
    torch.save({"model": source.state_dict(), "model_signature": model_contract_signature(source)}, path)
    head_before = target.property_head.weight.clone()
    transfer = load_pretrained_semmol(path, target, map_location="cpu")
    assert set(transfer.missing_keys) == {"property_head.weight", "property_head.bias"}
    assert torch.equal(target.body.weight, source.body.weight)
    assert torch.equal(target.property_head.weight, head_before)


def test_shared_projection_signature_mismatch_requires_retraining(tmp_path):
    source, target = TinyTransferModel(shared=False), TinyTransferModel(shared=True)
    path = tmp_path / "incompatible.pt"
    torch.save({"model": source.state_dict(), "model_signature": model_contract_signature(source)}, path)
    before = _snapshot(target)
    with pytest.raises(RuntimeError, match="retrain"):
        load_pretrained_semmol(path, target, map_location="cpu")
    assert all(torch.equal(target.state_dict()[key], value) for key, value in before.items())


def test_state_shape_is_checked_before_any_partial_load(tmp_path):
    source, target = TinyTransferModel(), TinyTransferModel()
    state = source.state_dict()
    state["body.weight"] = torch.ones(3, 4)
    path = tmp_path / "bad-shape.pt"
    torch.save({"model": state, "model_signature": model_contract_signature(source)}, path)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="shape differs"):
        load_pretrained_semmol(path, target, map_location="cpu")
    assert all(torch.equal(target.state_dict()[key], value) for key, value in before.items())


def test_old_resume_schema_gives_actionable_error():
    with pytest.raises(ValueError, match="retrain.*version 3"):
        _validate_training_payload({"version": 2}, expected_fingerprint="0" * 64, expected_world_size=1)


@pytest.mark.parametrize("field,value", [("center_l2_normalize", True), ("ema_momentum", 0.5)])
def test_same_shape_dcl_semantic_change_is_rejected_before_loading(tmp_path, field, value):
    source, target = TinyTransferModel(), TinyTransferModel()
    setattr(target.dcls["2d"], field, value)
    path = tmp_path / "different-dcl-algorithm.pt"
    torch.save({"model": source.state_dict(), "model_signature": model_contract_signature(source)}, path)
    before = _snapshot(target)
    with pytest.raises(RuntimeError, match="DCL geometry/EMA"):
        load_pretrained_semmol(path, target, map_location="cpu")
    assert all(torch.equal(target.state_dict()[key], tensor) for key, tensor in before.items())

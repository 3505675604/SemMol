"""Regression checks for the manuscript-aligned raw semantic representation."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("transformers")
pytest.importorskip("rdkit")

from src.models.alignment.acsm import AnchorCentersSoftMatching
from src.models.encoders.common import EncoderOutput
from src.models.projection.bank import ProjectionBank
from src.models.semantic.dcl import DynamicCentralLibrary
from src.models.semmol import SemMol, SemMolPretrainingOutput


def _encoded(features):
    return EncoderOutput(
        global_embedding=features,
        sample_index=torch.arange(features.shape[0]),
        tokens=features.unsqueeze(1),
        token_mask=torch.ones((features.shape[0], 1), dtype=torch.bool),
    )


def test_default_molecular_projectors_share_one_registered_mlp():
    bank = ProjectionBank(
        input_dim=4, output_dim=2, modalities=("1d", "2d", "3d"),
        mlp={"hidden_dim": 4, "dropout": 0.0},
    )
    assert bank.normalize_eps == 1.0e-8
    assert bank.projections["1d"] is bank.projections["2d"]
    assert bank.projections["1d"] is bank.projections["3d"]
    assert set(bank.projectors) == {"shared_mlp"}
    assert all(key.startswith("projectors.shared_mlp.") for key in bank.state_dict())
    encoded = _encoded(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    torch.testing.assert_close(bank("1d", encoded).raw, bank("3d", encoded).raw)


def test_independent_mlp_and_local_token_sap_are_explicit_variants():
    independent = ProjectionBank(
        input_dim=4, output_dim=2, modalities=("1d", "2d"),
        share_mlp_weights=False, mlp={"hidden_dim": 4, "dropout": 0.0},
    )
    assert independent.projections["1d"] is not independent.projections["2d"]
    sap = ProjectionBank(
        input_dim=4, output_dim=2, modalities=("1d", "3d"),
        projection_types={"3d": "sap"},
        mlp={"hidden_dim": 4, "dropout": 0.0},
        sap={"hidden_dim": 4, "num_heads": 2, "num_semantic_tokens": 2, "dropout": 0.0},
    )
    assert sap.projections["1d"] is not sap.projections["3d"]
    assert sap("3d", _encoded(torch.ones((2, 4)))).raw.shape == (2, 2)


def test_projector_epsilon_cannot_silently_disagree_with_cosine_epsilon():
    with pytest.raises(ValueError, match="normalize_eps"):
        ProjectionBank(modalities=("1d",), mlp={"normalize_eps": 1.0e-5})


def test_euclidean_kmeans_initialization_preserves_raw_center_magnitude():
    library = DynamicCentralLibrary(
        num_clusters=1, feature_dim=2, init_num_iters=2, init_max_samples=8,
        distributed_sync=False,
    )
    library.update_centers(torch.tensor([[2.0, 0.0], [4.0, 0.0]]))
    torch.testing.assert_close(library.centers, torch.tensor([[3.0, 0.0]]))
    assert not library.centers.requires_grad
    spherical = DynamicCentralLibrary(
        num_clusters=1, feature_dim=2, init_num_iters=2, init_max_samples=8,
        center_l2_normalize=True, distributed_sync=False,
    )
    spherical.update_centers(torch.tensor([[2.0, 0.0], [4.0, 0.0]]))
    torch.testing.assert_close(spherical.centers, torch.tensor([[1.0, 0.0]]))


def test_ordinary_ema_does_not_normalize_and_keeps_empty_clusters():
    library = DynamicCentralLibrary(
        num_clusters=2, feature_dim=2, ema_momentum=0.5,
        init_max_samples=8, distributed_sync=False,
    )
    library.initialized.fill_(True)
    library.centers.copy_(torch.tensor([[1.0, 0.0], [-10.0, 0.0]]))
    library.update_centers(torch.tensor([[0.0, 1.0]], requires_grad=True))
    torch.testing.assert_close(library.centers, torch.tensor([[0.5, 0.5], [-10.0, 0.0]]))
    library.update_centers(torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(library.centers, torch.tensor([[0.75, 0.25], [-10.0, 0.0]]))
    assert library.centers.grad_fn is None


def test_clustering_metric_distinguishes_default_euclidean_and_spherical_variant():
    options = dict(num_clusters=2, feature_dim=2, init_max_samples=8, distributed_sync=False)
    euclidean = DynamicCentralLibrary(**options)
    spherical = DynamicCentralLibrary(**options, center_l2_normalize=True)
    for library in (euclidean, spherical):
        library.initialized.fill_(True)
        library.centers.copy_(torch.tensor([[10.0, 0.0], [0.0, 1.0]]))
    feature = torch.tensor([[1.0, 0.0]])
    assert euclidean.assign(feature).hard_assignments.item() == 1
    assert spherical.assign(feature).hard_assignments.item() == 0


def test_acsm_keeps_raw_anchor_and_raw_center_sum_and_includes_self_in_softmax():
    matching = AnchorCentersSoftMatching(feature_dim=2, num_retrieve=1)
    anchor = torch.tensor([[2.0, 0.0]], requires_grad=True)
    output = matching(anchor, {"2d": torch.tensor([[3.0, 0.0]]),
                               "3d": torch.tensor([[0.0, 4.0]])}, anchor_modality="1d")
    torch.testing.assert_close(output.anchor_embedding, anchor)
    torch.testing.assert_close(output.positive_embedding, torch.tensor([[3.0, 4.0]]))
    two = AnchorCentersSoftMatching(feature_dim=2, num_retrieve=2, temperature=0.07)
    weights = two._positive_weights(torch.tensor([[0.8, 0.6]]))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(weights, torch.softmax(torch.tensor([[0.8, 0.6]]) / 0.07, -1))


def test_hard_negative_threshold_keeps_every_eligible_center_by_default():
    matching = AnchorCentersSoftMatching(feature_dim=2, num_retrieve=1, denoise_threshold=0.5)
    x = torch.linspace(0.51, 0.99, 20)
    centers = torch.stack((x, torch.sqrt(1 - x.square())), dim=-1)
    result = matching.match_modality(torch.tensor([[1.0, 0.0]]), centers, modality="2d")
    assert result.negative_count.item() == 19  # Exceeds the former silent cap of 10.
    assert torch.all(result.negative_similarities[result.negative_mask] >= 0.5)


def test_pretrain_output_and_dcl_input_use_raw_projection():
    raw = torch.tensor([[3.0, 4.0]])
    representation = SimpleNamespace(projection_output=SimpleNamespace(raw=raw, normalized=raw / 5))
    output = object.__new__(SemMolPretrainingOutput)
    object.__setattr__(output, "representations", {"2d": representation})
    assert output.projected("2d") is raw
    assert SemMol._features_for_dcl(SimpleNamespace(), {"2d": representation}, "2d") is raw


def test_freezing_target_alias_does_not_freeze_shared_anchor_projection():
    model = object.__new__(SemMol)
    torch.nn.Module.__init__(model)
    model.modalities = ("1d", "2d", "3d")
    model.anchor_modality = "1d"
    model.encoders = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in model.modalities})
    model.projection = ProjectionBank(input_dim=2, output_dim=2, modalities=model.modalities,
                                     mlp={"hidden_dim": 2, "dropout": 0.0})
    model.property_head = torch.nn.Linear(4, 1)
    model.set_encoders_frozen(False)
    model.train(True)
    shared = model.projection.projections["1d"]
    assert shared.training
    assert all(parameter.requires_grad for parameter in shared.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoders["2d"].parameters())
    assert not model.encoders["2d"].training

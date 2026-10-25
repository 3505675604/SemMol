"""InfoNCE and raw alignment have independent, explicit normalization rules."""

from datetime import timedelta
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("transformers")
pytest.importorskip("rdkit")

from src.losses.acsm_loss import ACSMContrastiveLoss
from src.losses.total_loss import SemMolPretrainTotalLoss
from src.models.alignment.acsm import AnchorCentersSoftMatching
from src.models.semmol import SemMolPretrainingOutput


def test_infonce_uses_cosine_divided_by_temperature_and_alignment_uses_raw_scale():
    matching = AnchorCentersSoftMatching(feature_dim=2, num_retrieve=1,
                                        temperature=0.2, denoise_threshold=-1.0)
    anchor = torch.tensor([[2.0, 0.0]], requires_grad=True)
    centers = torch.tensor([[3.0, 0.0], [0.0, 4.0]], requires_grad=True)
    output = matching(anchor, {"2d": centers}, anchor_modality="1d")
    criterion = ACSMContrastiveLoss(temperature=0.2, warmup_epochs=0,
                                   pseudo_weight=1.0, alignment_weight=1.0,
                                   distributed_sync=False)
    result = criterion.compute(output)
    expected_info = torch.log1p(torch.exp(torch.tensor(-5.0)))
    torch.testing.assert_close(result.pseudo.loss, expected_info)
    torch.testing.assert_close(result.alignment.loss, torch.tensor(0.5))
    result.loss.backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    assert centers.grad is None  # EMA centers are state, not optimizer parameters.


def test_missing_negatives_do_not_dilute_valid_pair_mean_and_alignment_remains_active():
    matching = AnchorCentersSoftMatching(feature_dim=2, num_retrieve=1, denoise_threshold=0.5)
    anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    output = matching(anchor, {"2d": torch.tensor([[2.0, 0.0], [0.0, 2.0]])}, anchor_modality="1d")
    result = ACSMContrastiveLoss(warmup_epochs=0, distributed_sync=False).compute(output)
    assert result.pseudo.global_count.item() == 0
    torch.testing.assert_close(result.pseudo.loss, torch.tensor(0.0))
    torch.testing.assert_close(result.alignment.loss, torch.tensor(0.5))
    result.loss.backward()
    assert torch.isfinite(anchor.grad).all()


def test_warmup_scales_pseudo_only_and_keeps_disclosed_alignment_regularizer():
    matching = AnchorCentersSoftMatching(feature_dim=2, num_retrieve=1, denoise_threshold=-1.0)
    output = matching(torch.tensor([[2.0, 0.0]], requires_grad=True),
                      {"2d": torch.tensor([[3.0, 0.0], [0.0, 4.0]])}, anchor_modality="1d")
    criterion = ACSMContrastiveLoss(warmup_epochs=5, pseudo_weight=0.1,
                                   alignment_weight=0.01, distributed_sync=False)
    result = criterion.compute(output, epoch=0)
    torch.testing.assert_close(result.loss, 0.01 * result.alignment.loss)
    assert result.pseudo_scale.item() == 0


def test_total_objective_counts_semantic_terms_once_and_is_a_batch_mean():
    matching = AnchorCentersSoftMatching(feature_dim=2, num_retrieve=1,
                                        temperature=0.2, denoise_threshold=-1.0)
    criterion = SemMolPretrainTotalLoss(temperature=0.2, warmup_epochs=0,
                                       pseudo_weight=0.1, alignment_weight=0.01,
                                       contrastive_reduction="weighted_mean",
                                       distributed_sync=False)
    means = []
    for batch_size in (1, 3):
        anchor = torch.tensor([[2.0, 0.0]]).repeat(batch_size, 1).requires_grad_()
        acsm = matching(anchor, {"2d": torch.tensor([[3.0, 0.0], [0.0, 4.0]])},
                        anchor_modality="1d")
        output = SemMolPretrainingOutput(
            batch_size=batch_size, representations={}, anchor_modality="1d",
            anchor_sample_index=torch.arange(batch_size), anchor_embedding=anchor,
            dcl_assignments={}, acsm_output=acsm, dcl_updates={},
            dcl_initialized=True, alignment_ready=True, warmup_active=False,
        )
        result = criterion.compute(output, {})
        expected = 0.1 * result.pseudo_loss.loss + 0.01 * result.alignment_loss.loss
        torch.testing.assert_close(result.total_loss, expected)
        means.append(result.total_loss)
    torch.testing.assert_close(means[0], means[1])


@pytest.mark.parametrize("reduction,factor", [("sum", 3.0), ("weighted_mean", 1.0)])
def test_repeating_anchors_scales_paper_sum_but_preserves_weighted_mean(reduction, factor):
    matching = AnchorCentersSoftMatching(feature_dim=2, num_retrieve=1,
                                        temperature=0.2, denoise_threshold=-1.0)
    criterion = ACSMContrastiveLoss(temperature=0.2, reduction=reduction, warmup_epochs=0,
                                   distributed_sync=False)
    results = []
    for count in (1, 3):
        anchors = torch.tensor([[2.0, 0.0]]).repeat(count, 1).requires_grad_()
        output = matching(anchors, {"2d": torch.tensor([[3.0, 0.0], [0.0, 4.0]])},
                          anchor_modality="1d")
        results.append(criterion.compute(output).pseudo.loss)
    torch.testing.assert_close(results[1], factor * results[0])


def test_paper_sum_preserves_absolute_modality_weight_scale():
    matching = AnchorCentersSoftMatching(feature_dim=2, num_retrieve=1,
                                        temperature=0.2, denoise_threshold=-1.0)
    output = matching(torch.tensor([[2.0, 0.0]], requires_grad=True),
                      {name: torch.tensor([[3.0, 0.0], [0.0, 4.0]]) for name in ("2d", "3d")},
                      anchor_modality="1d")
    options = dict(temperature=0.2, warmup_epochs=0, distributed_sync=False)
    small = ACSMContrastiveLoss(**options, modality_weights={"2d": 2.0, "3d": 3.0}).compute(output)
    large = ACSMContrastiveLoss(**options, modality_weights={"2d": 20.0, "3d": 30.0}).compute(output)
    expected = sum(weight * small.modality_losses[name].loss
                   for name, weight in (("2d", 2.0), ("3d", 3.0)))
    torch.testing.assert_close(small.pseudo.loss, expected)
    torch.testing.assert_close(large.pseudo.loss, 10 * small.pseudo.loss)
    means = [ACSMContrastiveLoss(**options, reduction="weighted_mean", modality_weights=weights)
             .compute(output).pseudo.loss for weights in ({"2d": 2.0, "3d": 3.0},
                                                          {"2d": 20.0, "3d": 30.0})]
    torch.testing.assert_close(means[0], means[1])


def test_reduction_rejects_unknown_or_ambiguous_mean_alias():
    with pytest.raises(ValueError, match="reduction"):
        ACSMContrastiveLoss(reduction="mean")


def _ddp_sum_worker(rank, init_uri, result_path):
    """Compare actual DDP with uneven local batches against one global batch."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=init_uri,
                            timeout=timedelta(seconds=60))
    try:
        linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.eye(2))
        model = DistributedDataParallel(linear)
        local_inputs = [torch.tensor([[1.0, 0.2]]), torch.tensor([[0.3, 1.1], [0.9, 0.4]])]
        centers = {"2d": torch.tensor([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])}
        matching = AnchorCentersSoftMatching(feature_dim=2, num_retrieve=1,
                                            temperature=0.2, denoise_threshold=-1.0)
        options = dict(temperature=0.2, modality_weights={"2d": 2.5},
                       pseudo_weight=1.0, alignment_weight=0.0, warmup_epochs=0)
        local = ACSMContrastiveLoss(**options, distributed_sync=True).compute(
            matching(model(local_inputs[rank]), centers, anchor_modality="1d"))
        local.loss.backward()

        reference = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            reference.weight.copy_(torch.eye(2))
        global_output = matching(reference(torch.cat(local_inputs)), centers, anchor_modality="1d")
        expected = ACSMContrastiveLoss(**options, distributed_sync=False).compute(global_output)
        expected.loss.backward()
        torch.testing.assert_close(model.module.weight.grad, reference.weight.grad, rtol=2e-5, atol=2e-6)
        reported_sum = local.pseudo.reporting_numerator.clone()
        dist.all_reduce(reported_sum)
        torch.testing.assert_close(reported_sum, expected.pseudo.loss.double(), rtol=2e-5, atol=2e-6)
        if rank == 0:
            Path(result_path).write_text(json.dumps({"world_size": 2, "global_pairs": 3,
                                                     "global_sum": reported_sum.item()}), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="CPU Gloo distributed support is required")
def test_paper_sum_ddp_gradient_matches_global_sum_with_uneven_rank_batches(tmp_path):
    result_path = tmp_path / "sum_result.json"
    torch.multiprocessing.spawn(_ddp_sum_worker,
                                args=((tmp_path / "gloo_init").as_uri(), str(result_path)),
                                nprocs=2, join=True)
    assert json.loads(result_path.read_text(encoding="utf-8"))["global_pairs"] == 3

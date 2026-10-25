"""Exercise configuration-to-objective wiring and real shared-parameter gradients."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune.run_finetune import _parse_configuration as parse_finetune
from scripts.pretrain.diagnose_alignment import alignment_batch_diagnostic
from scripts.pretrain.run_pretrain import _parse_configuration, build_pretraining_criterion
from src.losses.acsm_loss import ACSMContrastiveLoss
from src.models import resolve_semmol_config
from src.models.alignment.acsm import AnchorCentersSoftMatching
from src.trainers.runtime import load_experiment_configuration


def test_temperature_epsilon_weights_warmup_and_reduction_reach_actual_loss():
    configuration = load_experiment_configuration(ROOT / "configs/pretrain/pretrain_1m.yaml", expected_mode="pretrain")
    configuration.values["objective"].update(
        warmup_epochs=7, modality_weights={"2d": 2.0, "3d": 0.5}, contrastive_reduction="weighted_mean",
    )
    resolved = resolve_semmol_config(configuration.values, project_root=ROOT)
    resolved.model_options["acsm"].update(temperature=0.13, eps=1.0e-7)
    options = _parse_configuration(configuration, resume_override=None)
    criterion = build_pretraining_criterion(options, resolved.model_options, distributed_sync=False)
    assert criterion.acsm.temperature == 0.13
    assert criterion.acsm.eps == 1.0e-7
    assert criterion.acsm.warmup_epochs == 7
    assert criterion.acsm.modality_weights == {"2d": 2.0, "3d": 0.5}
    assert criterion.acsm.reduction == "weighted_mean"


def test_paper_budget_and_all_anchor_configurations_validate():
    for path in (ROOT / "configs/pretrain").glob("pretrain_[13]m*.yaml"):
        configuration = load_experiment_configuration(path, expected_mode="pretrain")
        options = _parse_configuration(configuration, resume_override=None)
        assert options.epochs == 10 and options.distributed["world_size"] == 8
        assert options.objective["contrastive_reduction"] == "sum"
    for path in (ROOT / "configs/finetune").glob("*.yaml"):
        configuration = load_experiment_configuration(path, expected_mode="finetune")
        options = parse_finetune(configuration)
        assert options.data_modalities == (options.model["anchor_modality"],)
        assert len(options.seeds) == 10
        assert options.distributed["world_size"] == 8


def test_every_published_pretraining_configuration_uses_eight_gpus():
    for path in (ROOT / "configs/pretrain").glob("*.yaml"):
        configuration = load_experiment_configuration(path, expected_mode="pretrain")
        options = _parse_configuration(configuration, resume_override=None)
        assert options.distributed["world_size"] == 8, path.name


def _diagnostic(threshold):
    torch.manual_seed(42)
    model = nn.Module()
    model.projection = nn.Module()
    model.projection.share_mlp_weights = True
    model.projection.projectors = nn.ModuleDict({"shared_mlp": nn.Linear(4, 4)})
    anchor = model.projection.projectors["shared_mlp"](torch.randn(6, 4))
    acsm = AnchorCentersSoftMatching(feature_dim=4, num_retrieve=1,
                                     temperature=0.1, denoise_threshold=threshold)
    output = SimpleNamespace(acsm_output=acsm(
        anchor, {"2d": torch.randn(8, 4), "3d": torch.randn(8, 4)}, anchor_modality="1d",
    ))
    criterion = ACSMContrastiveLoss(temperature=0.1, warmup_epochs=0, distributed_sync=False)
    return model, alignment_batch_diagnostic(model, output, criterion)


def test_gradient_diagnostic_uses_shared_parameters_and_reports_coverage():
    model, result = _diagnostic(-1.0)
    assert result["parameter_scope"] == "actual_shared_mlp_parameters"
    assert result["gradient_conflicts"]["eligible_pair_count"] == 1
    assert all(row["negative_coverage"] == 1.0 for row in result["negative_coverage"].values())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_zero_negative_coverage_is_ineligible_instead_of_zero_conflict():
    _, result = _diagnostic(1.0)
    assert result["gradient_conflicts"]["eligible_pair_count"] == 0
    assert all(row["negative_coverage"] == 0.0 for row in result["negative_coverage"].values())


def test_epoch_logs_separate_summed_objective_from_per_pair_mean():
    from src.losses.common import LossComponent
    from src.losses.total_loss import SemMolPretrainLossOutput, SemMolPretrainTotalLoss
    from src.trainers.pretrain_trainer import _EpochLossAccumulator

    criterion = SemMolPretrainTotalLoss(
        mlm_weight=0, graph_weight=0, geo_weight=0, pseudo_weight=0.1,
        alignment_weight=0, warmup_epochs=0, contrastive_reduction="sum", distributed_sync=False,
    )
    context = SimpleNamespace(device=torch.device("cpu"), distributed=False, world_size=1)
    accumulator = _EpochLossAccumulator(criterion, context, epoch=0)
    empty = LossComponent(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0), torch.tensor(0))
    for numerator in (10.0, 20.0):
        pseudo = LossComponent(
            loss=torch.tensor(numerator), numerator=torch.tensor(numerator),
            local_count=torch.tensor(2), global_count=torch.tensor(2),
            reporting_numerator=torch.tensor(numerator), reporting_denominator=torch.tensor(2.0),
        )
        output = SemMolPretrainLossOutput(
            total_loss=torch.tensor(numerator * 0.1), mlm_loss=empty,
            graph_loss=SimpleNamespace(node=empty, edge=empty, structure=empty),
            geo_loss=SimpleNamespace(mse=empty, direction=empty),
            pseudo_loss=pseudo, alignment_loss=empty,
            acsm=SimpleNamespace(pseudo_scale=torch.tensor(1.0)), component_counts={}, weighted_losses={},
        )
        accumulator.update(output, batch_size=2)
    summary, _, _ = accumulator.finalize()
    assert summary.total_loss == 1.5
    assert summary.epoch_objective_sum == 3.0
    assert summary.pseudo_objective_epoch_sum == 30.0
    assert summary.pseudo_per_pair_mean == 7.5
    assert summary.contrastive_reduction == "sum"

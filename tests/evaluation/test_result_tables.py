import math
import pytest
import torch
from torch import nn

from src.evaluation.result_tables import RunMeasurement, paired_relative_improvements, summarize_measurements
from src.evaluation.shap_analysis import SemanticFusionPredictor


def row(model="new", dataset="solv", unit="kcal/mol", seed=1, value=2.0, digest="a" * 64):
    return RunMeasurement.from_mapping(dict(model=model, dataset=dataset, metric="rmse", unit=unit,
                                           seed=seed, value=value, split_sha256=digest))


def test_table_uses_actual_seeds_sample_std_and_rejects_duplicates():
    observations = [row(seed=seed, value=value) for seed, value in enumerate([1., 2., 3.])]
    cell = summarize_measurements(observations, expected_seeds=3)[0]
    assert cell["mean"] == 2.0
    assert cell["sample_std"] == 1.0
    with pytest.raises(ValueError, match="duplicate seed"):
        summarize_measurements([*observations[:2], observations[0]], expected_seeds=3)


def test_relative_macro_is_invariant_to_a_tasks_unit_conversion():
    original = [row(model=model, dataset=task, unit=unit, seed=seed, value=value)
                for task, unit, new, old in [("solv", "kcal/mol", 2., 4.), ("esol", "logM", .9, 1.)]
                for model, value in [("new", new), ("old", old)] for seed in (1, 2)]
    converted = [row(model=x.model, dataset=x.dataset,
                     unit="kJ/mol" if x.dataset == "solv" else x.unit, seed=x.seed,
                     value=x.value * 4.184 if x.dataset == "solv" else x.value) for x in original]
    a = paired_relative_improvements(original, model="new", baseline="old")
    b = paired_relative_improvements(converted, model="new", baseline="old")
    assert a["macro_relative_improvement"] == pytest.approx(.3)
    assert b["macro_relative_improvement"] == pytest.approx(a["macro_relative_improvement"])


def test_comparison_rejects_different_outer_split():
    with pytest.raises(ValueError, match="split mismatch"):
        paired_relative_improvements([row(), row(model="old", digest="b" * 64)],
                                     model="new", baseline="old")


class SumHead(nn.Module):
    input_dim = 4
    num_tasks = 1
    task_type = "regression"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 1))

    def forward(self, features):
        return features @ self.weight


def test_shap_predictor_preserves_raw_semantic_fusion_scale():
    predictor = SemanticFusionPredictor(SumHead(), ["1d_anchor", "retrieved_2d_centers", "retrieved_3d_centers"], eps=1e-8)
    blocks = torch.tensor([[[1., 2.], [3., 0.], [0., 4.]]])
    assert predictor(blocks).item() == 10.0
    normalized = SemanticFusionPredictor(SumHead(), predictor.group_names, eps=1e-8, normalize_fusion=True)
    assert normalized(blocks).item() == pytest.approx(4.4)

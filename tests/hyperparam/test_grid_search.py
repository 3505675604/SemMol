"""Unit tests for the hyperparameter grid search module."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.hyperparam import GridAxis, GridDefinition, TrialResult, TrialSpec
from scripts.hyperparam.run_grid_search import (
    _apply_constraints,
    _deep_merge,
    _evaluate_constraint,
    _expand_grid,
    _flatten_overrides,
    _parse_grid_definition,
    _resolve_model_references,
    _set_nested,
    synthesize_trial_yaml,
)
from scripts.hyperparam.sensitivity import (
    SensitivityReport,
    compute_sensitivity_scores,
    generate_markdown_report,
    generate_sensitivity_csv,
    rank_results,
    write_best_config,
)
from scripts.hyperparam.trial_runner import (
    TrialRun,
    _detect_oom,
    _parse_final_metrics,
    _parse_train_loss,
    launch_trial,
)


# ---------------------------------------------------------------------------
# GridAxis
# ---------------------------------------------------------------------------


class TestGridAxis:
    def test_valid_axis(self) -> None:
        axis = GridAxis(path="model.dcl.num_clusters", values=[32, 64, 128])
        assert axis.path == "model.dcl.num_clusters"
        assert axis.values == [32, 64, 128]
        assert axis.value_type == "auto"

    def test_empty_path_raises(self) -> None:
        with pytest.raises(ValueError):
            GridAxis(path="", values=[1])

    def test_empty_values_raises(self) -> None:
        with pytest.raises(ValueError):
            GridAxis(path="x", values=[])

    def test_invalid_value_type_raises(self) -> None:
        with pytest.raises(ValueError):
            GridAxis(path="x", values=[1], value_type="unknown_type")


# ---------------------------------------------------------------------------
# GridDefinition
# ---------------------------------------------------------------------------


class TestGridDefinition:
    def test_valid_grid_definition(self) -> None:
        axis = GridAxis(path="lr", values=[1e-4, 3e-4])
        gd = GridDefinition(
            name="test",
            description="test grid",
            axes=(axis,),
        )
        assert gd.name == "test"
        assert gd.fast_epochs == 10
        assert gd.direction == "minimize"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            GridDefinition(name="   ", description="", axes=())

    def test_bad_strategy_raises(self) -> None:
        axis = GridAxis(path="x", values=[1])
        with pytest.raises(ValueError):
            GridDefinition(
                name="t", description="", axes=(axis,),
                search_strategy="invalid",
            )

    def test_bad_direction_raises(self) -> None:
        axis = GridAxis(path="x", values=[1])
        with pytest.raises(ValueError):
            GridDefinition(
                name="t", description="", axes=(axis,),
                direction="sideways",
            )

    def test_direction_aliases(self) -> None:
        axis = GridAxis(path="x", values=[1])
        gd_min = GridDefinition(
            name="t", description="", axes=(axis,), direction="min",
        )
        assert gd_min.direction == "minimize"
        gd_max = GridDefinition(
            name="t", description="", axes=(axis,), direction="max",
        )
        assert gd_max.direction == "maximize"


# ---------------------------------------------------------------------------
# TrialResult
# ---------------------------------------------------------------------------


class TestTrialResult:
    def test_valid_result(self) -> None:
        result = TrialResult(
            trial_index=0,
            grid_values={"lr": 1e-4},
            status="completed",
            metrics={"train_loss": 0.5},
            best_epoch=3,
            wall_time_seconds=120.0,
            error_message=None,
            config_path=Path("/tmp/c.yaml"),
            output_dir=Path("/tmp/o"),
        )
        assert result.trial_index == 0
        assert result.metrics["train_loss"] == 0.5


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


class TestExpandGrid:
    def test_empty_axes(self) -> None:
        result = _expand_grid(())
        assert result == [{}]

    def test_single_axis(self) -> None:
        axis = GridAxis(path="lr", values=[1e-4, 3e-4])
        result = _expand_grid((axis,))
        assert len(result) == 2
        assert result[0] == {"lr": 1e-4}
        assert result[1] == {"lr": 3e-4}

    def test_two_axes_cartesian(self) -> None:
        axis1 = GridAxis(path="K", values=[64, 128])
        axis2 = GridAxis(path="beta", values=[0.9, 0.99])
        result = _expand_grid((axis1, axis2))
        assert len(result) == 4
        expected = [
            {"K": 64, "beta": 0.9},
            {"K": 64, "beta": 0.99},
            {"K": 128, "beta": 0.9},
            {"K": 128, "beta": 0.99},
        ]
        assert result == expected

    def test_three_axes(self) -> None:
        axes = (
            GridAxis(path="a", values=[1, 2]),
            GridAxis(path="b", values=[10]),
            GridAxis(path="c", values=[100, 200, 300]),
        )
        result = _expand_grid(axes)
        assert len(result) == 2 * 1 * 3


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_greater_equal(self) -> None:
        assert _evaluate_constraint("a >= b", {"a": 5, "b": 3})
        assert not _evaluate_constraint("a >= b", {"a": 3, "b": 5})

    def test_less_equal(self) -> None:
        assert _evaluate_constraint("a <= b", {"a": 3, "b": 5})
        assert not _evaluate_constraint("a <= b", {"a": 5, "b": 3})

    def test_equal(self) -> None:
        assert _evaluate_constraint("a == b", {"a": 5, "b": 5})
        assert not _evaluate_constraint("a == b", {"a": 5, "b": 3})

    def test_not_equal(self) -> None:
        assert _evaluate_constraint("a != b", {"a": 5, "b": 3})
        assert not _evaluate_constraint("a != b", {"a": 5, "b": 5})

    def test_greater(self) -> None:
        assert _evaluate_constraint("a > b", {"a": 5, "b": 3})
        assert not _evaluate_constraint("a > b", {"a": 3, "b": 5})

    def test_less(self) -> None:
        assert _evaluate_constraint("a < b", {"a": 3, "b": 5})
        assert not _evaluate_constraint("a < b", {"a": 5, "b": 3})

    def test_literal_constant(self) -> None:
        assert _evaluate_constraint("a >= 5", {"a": 10})
        assert not _evaluate_constraint("a >= 5", {"a": 3})
        assert _evaluate_constraint("a > 0", {"a": 1})
        assert not _evaluate_constraint("a > 0", {"a": 0})

    def test_apply_constraints(self) -> None:
        trials = [
            {"M": 16, "K": 128},
            {"M": 64, "K": 32},
            {"M": 8, "K": 256},
        ]
        constraints = (
            {"expression": "M < K", "message": "M must be less than K"},
        )
        filtered = _apply_constraints(trials, constraints)
        assert len(filtered) == 2
        assert {"M": 64, "K": 32} not in filtered

    def test_apply_constraints_empty(self) -> None:
        trials = [{"a": 1}]
        assert _apply_constraints(trials, ()) == trials


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_top_level_override(self) -> None:
        base = {"a": 1, "b": 2}
        overrides = {"a": 10}
        merged = _deep_merge(base, overrides)
        assert merged == {"a": 10, "b": 2}

    def test_nested_override(self) -> None:
        base = {"train": {"lr": 1e-4, "epochs": 100}}
        overrides = {"train": {"lr": 3e-4}}
        merged = _deep_merge(base, overrides)
        assert merged["train"]["lr"] == 3e-4
        assert merged["train"]["epochs"] == 100

    def test_new_key(self) -> None:
        base = {"a": 1}
        overrides = {"b": 2}
        merged = _deep_merge(base, overrides)
        assert merged == {"a": 1, "b": 2}

    def test_does_not_mutate_original(self) -> None:
        base = {"train": {"lr": 1e-4}}
        overrides = {"train": {"lr": 3e-4}}
        _deep_merge(base, overrides)
        assert base["train"]["lr"] == 1e-4


class TestSetNested:
    def test_top_level(self) -> None:
        d: dict[str, Any] = {}
        _set_nested(d, "lr", 1e-4)
        assert d == {"lr": 1e-4}

    def test_two_levels(self) -> None:
        d: dict[str, Any] = {}
        _set_nested(d, "train.optimizer.lr", 1e-4)
        assert d == {"train": {"optimizer": {"lr": 1e-4}}}

    def test_three_levels(self) -> None:
        d: dict[str, Any] = {"model": {"dcl": {"num_clusters": 256}}}
        _set_nested(d, "model.dcl.ema_momentum", 0.99)
        assert d["model"]["dcl"]["ema_momentum"] == 0.99
        assert d["model"]["dcl"]["num_clusters"] == 256


class TestFlattenOverrides:
    def test_simple(self) -> None:
        overrides = {"model.dcl.num_clusters": 128}
        result = _flatten_overrides(overrides)
        assert result == {"model": {"dcl": {"num_clusters": 128}}}

    def test_multiple_paths(self) -> None:
        overrides = {
            "model.dcl.num_clusters": 128,
            "model.dcl.ema_momentum": 0.99,
            "train.optimizer.lr": 3e-4,
        }
        result = _flatten_overrides(overrides)
        assert result["model"]["dcl"]["num_clusters"] == 128
        assert result["model"]["dcl"]["ema_momentum"] == 0.99
        assert result["train"]["optimizer"]["lr"] == 3e-4


# ---------------------------------------------------------------------------
# YAML synthesis
# ---------------------------------------------------------------------------


class TestSynthesizeTrialYaml:
    def test_writes_valid_yaml(self, tmp_path: Path) -> None:
        base = {
            "experiment": {"name": "test", "mode": "pretrain", "seed": 42},
            "model": {
                "encoders": {"shared_dim": 512},
                "dcl": {"num_clusters": 256, "ema_momentum": 0.9},
                "acsm": {"num_retrieve": 16, "temperature": 0.07},
                "projection": {"input_dim": 512, "output_dim": 256},
                "modalities": ["1d", "2d", "3d"],
                "anchor_modality": "1d",
            },
            "train": {
                "optimizer": {"type": "adamw", "lr": 1e-4, "weight_decay": 0.05},
                "epochs": 100,
            },
            "output": {
                "checkpoint_dir": "checkpoints",
                "log_dir": "logs",
                "save_every_n_epochs": 10,
                "tensorboard": False,
                "wandb": False,
                "resume": None,
            },
        }
        overrides = {"model.dcl.num_clusters": 128}
        output_path = tmp_path / "config.yaml"
        project_root = tmp_path

        result = synthesize_trial_yaml(
            base_config=base,
            overrides=overrides,
            output_path=output_path,
            fast_epochs=5,
            project_root=project_root,
        )
        assert result == output_path.resolve()
        assert output_path.exists()

        loaded = yaml.safe_load(output_path.read_text())
        assert loaded["model"]["dcl"]["num_clusters"] == 128
        assert loaded["train"]["epochs"] == 5
        assert loaded["output"]["tensorboard"] is False
        assert loaded["output"]["wandb"] is False


# ---------------------------------------------------------------------------
# Grid definition parsing
# ---------------------------------------------------------------------------


class TestParseGridDefinition:
    def test_valid_grid_yaml(self, tmp_path: Path) -> None:
        grid_yaml = tmp_path / "grid.yaml"
        grid_yaml.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "description": "test grid",
                    "search_strategy": "grid",
                    "axes": [
                        {"path": "lr", "values": [1e-4, 3e-4]},
                        {"path": "K", "values": [64, 128]},
                    ],
                    "evaluation": {
                        "mode": "pretrain",
                        "fast_epochs": 5,
                        "metrics": ["train_loss"],
                        "direction": "minimize",
                    },
                }
            )
        )
        gd = _parse_grid_definition(grid_yaml)
        assert gd.name == "test"
        assert len(gd.axes) == 2
        assert gd.axes[0].path == "lr"
        assert gd.fast_epochs == 5

    def test_missing_axes(self, tmp_path: Path) -> None:
        grid_yaml = tmp_path / "grid.yaml"
        grid_yaml.write_text(yaml.dump({"name": "test", "evaluation": {"mode": "pretrain"}}))
        with pytest.raises(ValueError, match="non-empty 'axes'"):
            _parse_grid_definition(grid_yaml)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


class TestSensitivityScores:
    def _make_result(self, idx: int, K: int, loss: float) -> TrialResult:
        return TrialResult(
            trial_index=idx,
            grid_values={"K": K},
            status="completed",
            metrics={"train_loss": loss},
            best_epoch=None,
            wall_time_seconds=0.0,
            error_message=None,
            config_path=Path(f"/tmp/c{idx}.yaml"),
            output_dir=Path(f"/tmp/o{idx}"),
        )

    def test_single_axis(self) -> None:
        axis = GridAxis(path="K", values=[64, 128, 256])
        results = [
            self._make_result(0, 64, 0.5),
            self._make_result(1, 128, 0.3),
            self._make_result(2, 256, 0.4),
        ]
        scores = compute_sensitivity_scores(results, (axis,))
        assert "K" in scores
        assert scores["K"] > 0.0

    def test_no_variation(self) -> None:
        axis = GridAxis(path="K", values=[128])
        results = [
            self._make_result(0, 128, 0.5),
            self._make_result(1, 128, 0.5),
        ]
        scores = compute_sensitivity_scores(results, (axis,))
        assert scores["K"] == 0.0

    def test_skips_failed(self) -> None:
        axis = GridAxis(path="K", values=[64, 128])
        results = [
            self._make_result(0, 64, 0.5),
            TrialResult(
                trial_index=1,
                grid_values={"K": 128},
                status="failed",
                metrics={},
                best_epoch=None,
                wall_time_seconds=0.0,
                error_message="oom",
                config_path=Path("/tmp/c.yaml"),
                output_dir=Path("/tmp/o"),
            ),
        ]
        scores = compute_sensitivity_scores(results, (axis,))
        assert scores["K"] == 0.0


class TestRankResults:
    def _make_result(
        self, idx: int, loss: float, status: str = "completed"
    ) -> TrialResult:
        return TrialResult(
            trial_index=idx,
            grid_values={},
            status=status,
            metrics={"train_loss": loss},
            best_epoch=None,
            wall_time_seconds=0.0,
            error_message=None,
            config_path=Path(f"/tmp/c{idx}.yaml"),
            output_dir=Path(f"/tmp/o{idx}"),
        )

    def test_minimize_ranks_ascending(self) -> None:
        results = [
            self._make_result(0, 0.8),
            self._make_result(1, 0.3),
            self._make_result(2, 0.5),
        ]
        ranked = rank_results(results, "minimize", "train_loss")
        assert ranked[0].metrics["train_loss"] == 0.3
        assert ranked[1].metrics["train_loss"] == 0.5
        assert ranked[2].metrics["train_loss"] == 0.8

    def test_maximize_ranks_descending(self) -> None:
        results = [
            self._make_result(0, 0.8),
            self._make_result(1, 0.3),
            self._make_result(2, 0.5),
        ]
        ranked = rank_results(results, "maximize", "train_loss")
        assert ranked[0].metrics["train_loss"] == 0.8
        assert ranked[2].metrics["train_loss"] == 0.3


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    def _make_axis(self) -> GridAxis:
        return GridAxis(path="K", values=[64, 128, 256])

    def _make_report(self) -> SensitivityReport:
        axis = self._make_axis()
        results = [
            TrialResult(
                trial_index=0,
                grid_values={"K": 64},
                status="completed",
                metrics={"train_loss": 0.5},
                best_epoch=3,
                wall_time_seconds=100.0,
                error_message=None,
                config_path=Path("/tmp/c.yaml"),
                output_dir=Path("/tmp/o"),
            ),
            TrialResult(
                trial_index=1,
                grid_values={"K": 128},
                status="completed",
                metrics={"train_loss": 0.3},
                best_epoch=5,
                wall_time_seconds=110.0,
                error_message=None,
                config_path=Path("/tmp/c2.yaml"),
                output_dir=Path("/tmp/o2"),
            ),
        ]
        return SensitivityReport(
            grid_name="test",
            grid_description="a test",
            direction="minimize",
            primary_metric="train_loss",
            metrics=("train_loss",),
            axes=(axis,),
            all_results=results,
            ranked_results=rank_results(results, "minimize", "train_loss"),
            sensitivity_scores={"K": 0.5},
            best_config={"K": 128},
            best_metrics={"train_loss": 0.3},
            total_trials=2,
            completed_trials=2,
            failed_trials=0,
        )

    def test_generates_markdown(self) -> None:
        report = self._make_report()
        md = generate_markdown_report(report)
        assert "# test" in md
        assert "Sensitivity Analysis" in md
        assert "K" in md
        assert "0.5" in md


class TestSensitivityCSV:
    def test_writes_csv(self, tmp_path: Path) -> None:
        axis = GridAxis(path="K", values=[64, 128, 256])
        results = [
            TrialResult(
                trial_index=0,
                grid_values={"K": 64},
                status="completed",
                metrics={"train_loss": 0.5},
                best_epoch=None,
                wall_time_seconds=0.0,
                error_message=None,
                config_path=Path("/tmp/c.yaml"),
                output_dir=Path("/tmp/o"),
            ),
            TrialResult(
                trial_index=1,
                grid_values={"K": 128},
                status="completed",
                metrics={"train_loss": 0.3},
                best_epoch=None,
                wall_time_seconds=0.0,
                error_message=None,
                config_path=Path("/tmp/c2.yaml"),
                output_dir=Path("/tmp/o2"),
            ),
        ]
        report = SensitivityReport(
            grid_name="test",
            grid_description="",
            direction="minimize",
            primary_metric="train_loss",
            metrics=("train_loss",),
            axes=(axis,),
            all_results=results,
            ranked_results=results,
            sensitivity_scores={"K": 0.5},
            best_config={"K": 128},
            best_metrics={"train_loss": 0.3},
            total_trials=2,
            completed_trials=2,
            failed_trials=0,
        )
        csv_path = tmp_path / "sensitivity.csv"
        generate_sensitivity_csv(report, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "K" in content


class TestBestConfig:
    def test_writes_best_config(self, tmp_path: Path) -> None:
        axis = GridAxis(path="K", values=[64, 128])
        report = SensitivityReport(
            grid_name="test",
            grid_description="",
            direction="minimize",
            primary_metric="train_loss",
            metrics=("train_loss",),
            axes=(axis,),
            all_results=[],
            ranked_results=[],
            sensitivity_scores={},
            best_config={"K": 128, "lr": 1e-4},
            best_metrics={"train_loss": 0.3},
            total_trials=0,
            completed_trials=0,
            failed_trials=0,
        )
        best_path = tmp_path / "best_config.yaml"
        write_best_config(report, best_path)
        assert best_path.exists()
        content = best_path.read_text()
        assert "K" in content
        assert "128" in content


# ---------------------------------------------------------------------------
# Trial runner — parsing
# ---------------------------------------------------------------------------


class TestParseFinalMetrics:
    def test_json_line(self) -> None:
        stdout = 'some output\n{"train_loss": 0.5, "val_loss": 0.6}\nmore output'
        metrics = _parse_final_metrics(stdout)
        assert metrics == {"train_loss": 0.5, "val_loss": 0.6}

    def test_no_json(self) -> None:
        stdout = "just text, no json at all"
        metrics = _parse_final_metrics(stdout)
        assert metrics == {}

    def test_json_with_non_numeric(self) -> None:
        stdout = '{"train_loss": 0.5, "name": "experiment"}'
        metrics = _parse_final_metrics(stdout)
        assert metrics == {"train_loss": 0.5}


class TestParseTrainLoss:
    def test_extracts_loss(self) -> None:
        stdout = "train_loss: 0.5432\nother stuff"
        loss = _parse_train_loss(stdout)
        assert loss == pytest.approx(0.5432)

    def test_no_loss(self) -> None:
        stdout = "no loss here"
        assert _parse_train_loss(stdout) is None

    def test_scientific_notation(self) -> None:
        stdout = "train_loss: 1.5e-3"
        loss = _parse_train_loss(stdout)
        assert loss == pytest.approx(0.0015)


class TestDetectOOM:
    def test_cuda_oom(self) -> None:
        assert _detect_oom("RuntimeError: CUDA out of memory")
        assert _detect_oom("RuntimeError: CUDA error: out of memory")
        assert _detect_oom("CUDA error: an illegal memory access was encountered")

    def test_no_oom(self) -> None:
        assert not _detect_oom("normal error message")
        assert not _detect_oom("")
        assert not _detect_oom("ValueError: invalid input")


# ---------------------------------------------------------------------------
# Trial runner — launch_trial (dry-run / non-GPU)
# ---------------------------------------------------------------------------


class TestLaunchTrial:
    def test_launch_with_invalid_config_triggers_error(
        self, tmp_path: Path
    ) -> None:
        """When the config is garbage, the subprocess should fail."""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("not: valid: yaml: [[[")

        output_dir = tmp_path / "output"
        project_root = tmp_path

        run = launch_trial(
            config_path=config_path,
            output_dir=output_dir,
            mode="pretrain",
            project_root=project_root,
            device="cpu",
            timeout=30,
        )

        assert run.status in ("failed", "error", "oom")

    def test_launch_with_fast_debug_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthesize a minimal valid config and verify the subprocess
        can at least start (it will fail because there is no real data,
        but the trial runner should handle the failure gracefully)."""
        config = {
            "experiment": {
                "name": "grid_test",
                "mode": "pretrain",
                "seed": 42,
                "deterministic": False,
                "cudnn_benchmark": False,
            },
            "model": {
                "encoders": {
                    "shared_dim": 128,
                    "smiles": {
                        "mode": "pretrained_adapter",
                        "tokenizer_dir": str(tmp_path / "tokenizer"),
                        "model_name_or_path": "DeepChem/ChemBERTa-77M-MLM",
                        "native_tokenizer_name_or_path": "DeepChem/ChemBERTa-77M-MLM",
                        "shared_dim": 128,
                        "pooling": "cls",
                        "freeze_layers": 0,
                        "validate_values": False,
                        "local_files_only": True,
                        "cache_dir": str(tmp_path / "hf_cache"),
                    },
                    "graph": {
                        "encoder_type": "gatedgcn",
                        "node_feature_cardinalities": [5, 5],
                        "edge_feature_cardinalities": [5, 5],
                        "num_layers": 2,
                        "hidden_size": 128,
                        "dropout": 0.1,
                        "residual": True,
                        "layer_norm": True,
                        "train_eps": True,
                        "validate_values": False,
                    },
                    "geo": {
                        "hidden_size": 64,
                        "num_blocks": 2,
                        "num_bilinear": 4,
                        "num_spherical": 3,
                        "num_radial": 3,
                        "cutoff": 5.0,
                        "envelope_exponent": 5,
                        "num_before_skip": 1,
                        "num_after_skip": 2,
                        "num_output_layers": 2,
                        "target_dim": 128,
                        "dropout": 0.0,
                        "max_num_neighbors": 16,
                        "conformer_pooling": "mean",
                        "validate_values": False,
                    },
                },
                "dcl": {
                    "feature_dim": 128,
                    "num_clusters": 32,
                    "ema_momentum": 0.9,
                    "init_method": "random",
                    "init_num_iters": 5,
                    "init_max_samples": 128,
                    "init_seed": 42,
                    "reassign_interval": 1,
                    "assignment_temperature": 0.5,
                    "center_l2_normalize": True,
                    "distributed_sync": True,
                    "warmup_steps": 10,
                    "eps": 1e-8,
                    "validate_values": False,
                },
                "acsm": {
                    "feature_dim": 128,
                    "num_retrieve": 8,
                    "temperature": 0.07,
                    "learnable_temperature": False,
                    "weighting": "softmax",
                    "denoise_threshold": 0.5,
                    "negative_selection": "hard",
                    "max_negatives": 10,
                    "eps": 1e-8,
                    "validate_values": False,
                },
                "projection": {
                    "input_dim": 128,
                    "output_dim": 128,
                    "projection_types": {"1d": "mlp", "2d": "mlp", "3d": "mlp"},
                    "normalize_eps": 1e-8,
                    "validate_values": False,
                    "mlp": {
                        "hidden_dim": 128,
                        "num_layers": 1,
                        "activation": "relu",
                        "dropout": 0.0,
                        "bias": True,
                        "layer_norm_eps": 1e-5,
                    },
                },
                "pretraining_heads": {
                    "mlm": {"hidden_dim": 128},
                    "graph": {"hidden_dim": 128},
                    "geo": {"hidden_dim": 128},
                },
                "modalities": ["1d", "2d", "3d"],
                "anchor_modality": "1d",
                "freeze_encoders": False,
            },
            "data": {
                "store_dir": str(tmp_path / "store"),
                "manifest_path": str(tmp_path / "manifest.npz"),
                "tokenizer_dir": str(tmp_path / "tokenizer"),
                "modalities": ["1d", "2d", "3d"],
                "strict_modalities": False,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
            },
            "train": {
                "batch_size": 2,
                "epochs": 1,
                "accum_steps": 1,
                "grad_clip": 1.0,
                "mixed_precision": "none",
                "optimizer": {
                    "type": "adamw",
                    "lr": 1e-4,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                },
                "scheduler": {
                    "type": "cosine",
                    "warmup_ratio": 0.0,
                    "min_lr": 1e-6,
                },
            },
            "loss": {
                "mlm": 1.0,
                "graph": 1.0,
                "geo": 0.5,
                "pseudo": 0.1,
                "alignment": 0.01,
            },
            "mask": {
                "smiles_ratio": 0.15,
                "node_ratio": 0.15,
                "edge_ratio": 0.15,
                "geo_noise_std": 0.1,
            },
            "distributed": {
                "backend": "gloo",
                "world_size": 1,
                "broadcast_buffers": False,
                "find_unused_parameters": True,
                "sampler": "sequential",
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
                "save_every_n_epochs": 1,
                "resume": None,
                "log_interval": 10,
                "tensorboard": False,
                "wandb": False,
            },
        }
        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config, f)

        output_dir = tmp_path / "output"
        project_root = tmp_path

        run = launch_trial(
            config_path=config_path,
            output_dir=output_dir,
            mode="pretrain",
            project_root=project_root,
            device="cpu",
            timeout=60,
        )

        # The run will fail because there is no real data / tokenizer,
        # but the trial runner should NOT crash — it should return a
        # TrialRun with a non-error status field.
        assert run.status in ("completed", "failed", "error", "oom", "timeout")
        assert isinstance(run.return_code, int)
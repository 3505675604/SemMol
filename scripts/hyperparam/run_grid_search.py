"""SemMol hyperparameter grid search orchestrator.

Supports both grid search and random search across pretraining and finetuning
hyperparameters.  Each trial runs as an independent subprocess to avoid
state leakage from the fail-closed ``PretrainTrainer``.

Usage::

    python scripts/hyperparam/run_grid_search.py \\
        --base-config configs/pretrain/debug.yaml \\
        --grid configs/hyperparam/dcl_sensitivity.yaml \\
        --output-dir outputs/hyperparam/dcl_sensitivity \\
        --mode pretrain \\
        --max-trials 50 \\
        --epochs 5
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

# Support both relative imports (when run as a package) and absolute imports
# (when run directly as a script or from tests).
try:
    from . import GridAxis, GridDefinition, TrialResult, TrialSpec
    from .sensitivity import (
        SensitivityReport,
        compute_sensitivity_scores,
        generate_markdown_report,
        generate_sensitivity_csv,
        rank_results,
        write_best_config,
    )
    from .trial_runner import launch_trial
except ImportError:
    import sys as _sys
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_PROJECT_ROOT))
    from scripts.hyperparam import (
        GridAxis, GridDefinition, TrialResult, TrialSpec,
    )
    from scripts.hyperparam.sensitivity import (
        SensitivityReport,
        compute_sensitivity_scores,
        generate_markdown_report,
        generate_sensitivity_csv,
        rank_results,
        write_best_config,
    )
    from scripts.hyperparam.trial_runner import launch_trial

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

_GRID_SCHEMA_KEYS = frozenset(
    {
        "name",
        "description",
        "search_strategy",
        "axes",
        "constraints",
        "evaluation",
        "seed",
        "max_trials",
        "random_trials",
    }
)
_AXIS_SCHEMA_KEYS = frozenset({"path", "values", "value_type"})
_EVALUATION_SCHEMA_KEYS = frozenset(
    {"mode", "fast_epochs", "metrics", "direction"}
)
_DIRECTION_ALIASES = {
    "min": "minimize", "minimize": "minimize",
    "max": "maximize", "maximize": "maximize",
}


def _load_yaml(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a YAML mapping")
    return dict(payload)


def _deep_merge(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge *overrides* into *base* and return a new dict."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_nested(dct: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set ``dct[key1][key2]... = value`` given ``key1.key2``."""
    parts = dotted_path.split(".")
    cursor = dct
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], Mapping):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _flatten_overrides(
    overrides: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Convert dotted-key overrides into a nested dict for YAML output."""
    result: dict[str, Any] = {}
    for key, value in overrides.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            result[key] = _flatten_overrides(
                dict(value),
                prefix=full,
            )
        else:
            _set_nested(result, key, value)
    return result


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def _expand_grid(axes: tuple[GridAxis, ...]) -> list[dict[str, Any]]:
    """Cartesian product of all axis values."""
    if not axes:
        return [{}]
    names = [axis.path for axis in axes]
    value_lists = [axis.values for axis in axes]
    trials: list[dict[str, Any]] = []
    for combination in itertools.product(*value_lists):
        trials.append(dict(zip(names, combination)))
    return trials


def _sample_random(
    axes: tuple[GridAxis, ...],
    n_trials: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Randomly sample *n_trials* combinations."""
    samples: list[dict[str, Any]] = []
    for _ in range(n_trials):
        entry: dict[str, Any] = {}
        for axis in axes:
            entry[axis.path] = rng.choice(axis.values)
        samples.append(entry)
    return samples


def _resolve_dotted(raw: str, base_values: dict[str, Any]) -> Any:
    """Walk a dotted path like ``model.dcl.num_clusters`` through a nested dict."""
    parts = raw.split(".")
    cursor: Any = base_values
    for part in parts:
        if isinstance(cursor, Mapping) and part in cursor:
            cursor = cursor[part]
        else:
            return None
    return cursor


def _evaluate_constraint(
    expression: str,
    overrides: dict[str, Any],
    base_values: dict[str, Any] | None = None,
) -> bool:
    """Evaluate a simple comparison expression against the overrides dict.

    Supported forms::

        a.b >= c.d
        a.b > c.d
        a.b == c.d
        a.b != c.d
        a.b < c.d
        a.b <= c.d
        a.b >= 5
        a.b > 0

    If *base_values* is provided, dotted-path keys not found in
    *overrides* are resolved from *base_values* first.
    """
    expression = expression.strip()
    for op in (">=", "<=", "!=", "==", ">", "<"):
        if op in expression:
            left_raw, right_raw = expression.split(op, 1)
            break
    else:
        raise ValueError(f"unsupported constraint expression: {expression!r}")

    left_raw = left_raw.strip()
    right_raw = right_raw.strip()

    def _resolve(raw: str) -> Any:
        raw = raw.strip()
        # 1) flat key in overrides
        if raw in overrides:
            return overrides[raw]
        # 2) dotted-path walk through base_values (e.g. model.dcl.num_clusters)
        if base_values is not None and "." in raw:
            resolved = _resolve_dotted(raw, base_values)
            if resolved is not None and not isinstance(resolved, Mapping):
                return resolved
        # 3) flat key in base_values (top-level lookup)
        if base_values is not None and raw in base_values:
            return base_values[raw]
        # 4) try literal numeric
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except (ValueError, TypeError):
            pass
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
        return raw

    left = _resolve(left_raw)
    right = _resolve(right_raw)

    try:
        if op == ">=":
            return left >= right
        if op == "<=":
            return left <= right
        if op == "!=":
            return left != right
        if op == "==":
            return left == right
        if op == ">":
            return left > right
        if op == "<":
            return left < right
    except TypeError:
        return False
    return False


def _apply_constraints(
    trials: list[dict[str, Any]],
    constraints: tuple[dict[str, str], ...],
    base_values: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Filter out trials that violate any constraint.

    *base_values* provides fallback resolution for dotted-path keys
    referenced in constraints but not present in the trial overrides
    (e.g., ``model.dcl.num_clusters`` when the overrides only contain
    ACSM parameters).
    """
    if not constraints:
        return trials
    filtered: list[dict[str, Any]] = []
    for trial in trials:
        ok = True
        for constraint in constraints:
            expression = constraint.get("expression", "")
            if not expression:
                continue
            if not _evaluate_constraint(expression, trial, base_values):
                ok = False
                break
        if ok:
            filtered.append(trial)
    return filtered


# ---------------------------------------------------------------------------
# Trial YAML synthesis
# ---------------------------------------------------------------------------


def _resolve_model_references(
    base_config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Inline all model YAML references so the trial config is self-contained."""
    config = copy.deepcopy(base_config)
    model = config.get("model")
    if not isinstance(model, Mapping):
        return config

    reference_keys = ("encoders", "projection", "dcl", "acsm")
    optional_keys = ("pretraining_heads",)

    for key in reference_keys:
        value = model.get(key)
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.is_absolute():
                path = project_root / path
            model[key] = _load_yaml(path, name=f"model.{key}")

    for key in optional_keys:
        value = model.get(key)
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.is_absolute():
                path = project_root / path
            model[key] = _load_yaml(path, name=f"model.{key}")

    return config


def synthesize_trial_yaml(
    base_config: dict[str, Any],
    overrides: dict[str, Any],
    output_path: Path,
    *,
    fast_epochs: int,
    project_root: Path,
) -> Path:
    """Merge *overrides* into *base_config* and write the result to *output_path*.

    Returns the absolute path to the written file.
    """
    resolved = _resolve_model_references(base_config, project_root)

    nested_overrides = _flatten_overrides(overrides)

    merged = _deep_merge(resolved, nested_overrides)

    if "train" in merged and isinstance(merged["train"], Mapping):
        merged["train"]["epochs"] = fast_epochs

    if "output" in merged and isinstance(merged["output"], Mapping):
        output_dir = output_path.parent
        merged["output"]["checkpoint_dir"] = str(
            output_dir / "checkpoints"
        )
        merged["output"]["log_dir"] = str(output_dir / "logs")
        merged["output"]["save_every_n_epochs"] = fast_epochs
        merged["output"]["tensorboard"] = False
        merged["output"]["wandb"] = False
        merged["output"]["resume"] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, default_flow_style=False, sort_keys=False)

    return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def _parse_grid_definition(path: Path) -> GridDefinition:
    raw = _load_yaml(path, name="grid definition")

    unknown = set(raw) - _GRID_SCHEMA_KEYS
    if unknown:
        raise ValueError(
            f"unknown grid definition keys: {sorted(unknown)}"
        )

    axes_raw = raw.get("axes")
    if not isinstance(axes_raw, list) or not axes_raw:
        raise ValueError("grid definition must contain a non-empty 'axes' list")

    axes: list[GridAxis] = []
    for i, axis_raw in enumerate(axes_raw):
        if not isinstance(axis_raw, Mapping):
            raise TypeError(f"axes[{i}] must be a mapping")
        unknown_axis = set(axis_raw) - _AXIS_SCHEMA_KEYS
        if unknown_axis:
            raise ValueError(
                f"axes[{i}] unknown keys: {sorted(unknown_axis)}"
            )
        axes.append(
            GridAxis(
                path=axis_raw["path"],
                values=list(axis_raw["values"]),
                value_type=axis_raw.get("value_type", "auto"),
            )
        )

    constraints_raw = raw.get("constraints", [])
    if not isinstance(constraints_raw, list):
        raise TypeError("constraints must be a list")
    constraints: list[dict[str, str]] = []
    for i, c in enumerate(constraints_raw):
        if not isinstance(c, Mapping):
            raise TypeError(f"constraints[{i}] must be a mapping")
        constraints.append(
            {
                "expression": str(c.get("expression", "")),
                "message": str(c.get("message", "")),
            }
        )

    evaluation = raw.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping")
    unknown_eval = set(evaluation) - _EVALUATION_SCHEMA_KEYS
    if unknown_eval:
        raise ValueError(
            f"evaluation unknown keys: {sorted(unknown_eval)}"
        )

    mode = str(evaluation.get("mode", "pretrain"))
    fast_epochs = int(evaluation.get("fast_epochs", 10))
    metrics = tuple(
        str(m)
        for m in evaluation.get("metrics", ["train_loss"])
    )
    direction = str(evaluation.get("direction", "minimize"))

    return GridDefinition(
        name=str(raw.get("name", path.stem)),
        description=str(raw.get("description", "")),
        axes=tuple(axes),
        search_strategy=str(raw.get("search_strategy", "grid")),
        constraints=tuple(constraints),
        evaluation_mode=mode,
        fast_epochs=fast_epochs,
        metrics=metrics,
        direction=direction,
        seed=int(raw.get("seed", 3407)),
        max_trials=(
            int(raw["max_trials"])
            if raw.get("max_trials") is not None
            else None
        ),
        random_trials=int(raw.get("random_trials", 50)),
    )


def _generate_trials(
    grid: GridDefinition,
    rng: random.Random,
    base_values: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate the list of trial override dicts."""
    if grid.search_strategy == "random":
        trials = _sample_random(grid.axes, grid.random_trials, rng)
    else:
        trials = _expand_grid(grid.axes)

    trials = _apply_constraints(trials, grid.constraints, base_values)

    if grid.search_strategy == "random":
        trials = trials[:grid.random_trials]

    if grid.max_trials is not None and len(trials) > grid.max_trials:
        if grid.search_strategy == "grid":
            trials = trials[:grid.max_trials]
        else:
            rng.shuffle(trials)
            trials = trials[:grid.max_trials]

    return trials


def _find_project_root() -> Path:
    """Find the SemMol project root relative to this script."""
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return script_dir.parent.parent


def _run_trial(
    trial_spec: TrialSpec,
    *,
    mode: str,
    project_root: Path,
    device: str | None,
    timeout_per_trial: int,
    trial_script: Path | None,
) -> TrialResult:
    """Execute one trial and return its result."""
    start = time.monotonic()

    try:
        run = launch_trial(
            config_path=trial_spec.config_path,
            output_dir=trial_spec.output_dir,
            mode=mode,
            project_root=project_root,
            device=device,
            timeout=timeout_per_trial,
            trial_script=trial_script,
        )

        elapsed = time.monotonic() - start

        return TrialResult(
            trial_index=trial_spec.trial_index,
            grid_values=trial_spec.overrides,
            status=run.status,
            metrics=run.metrics,
            best_epoch=run.best_epoch,
            wall_time_seconds=elapsed,
            error_message=run.error_message,
            config_path=trial_spec.config_path,
            output_dir=trial_spec.output_dir,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return TrialResult(
            trial_index=trial_spec.trial_index,
            grid_values=trial_spec.overrides,
            status="error",
            metrics={},
            best_epoch=None,
            wall_time_seconds=elapsed,
            error_message=str(exc),
            config_path=trial_spec.config_path,
            output_dir=trial_spec.output_dir,
        )


def _write_results_json(results: list[TrialResult], path: Path) -> None:
    """Write all trial results to a JSON file."""
    serializable: list[dict[str, Any]] = []
    for r in results:
        serializable.append(
            {
                "trial_index": r.trial_index,
                "grid_values": r.grid_values,
                "status": r.status,
                "metrics": r.metrics,
                "best_epoch": r.best_epoch,
                "wall_time_seconds": r.wall_time_seconds,
                "error_message": r.error_message,
                "config_path": str(r.config_path),
                "output_dir": str(r.output_dir),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SemMol hyperparameter grid search",
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument(
        "--grid",
        required=True,
        help="Path to the grid definition YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for trial outputs and results.",
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="Training mode (default: pretrain).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum number of trials to run (overrides grid definition).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of fast evaluation epochs (overrides grid definition).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g., cuda:0, cpu).",
    )
    parser.add_argument(
        "--timeout-per-trial",
        type=int,
        default=86400,
        help="Maximum seconds per trial (default: 86400 = 24 hours).",
    )
    parser.add_argument(
        "--trial-script",
        default=None,
        help="Path to a custom trial runner script (advanced).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print trials without running them.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)

    project_root = _find_project_root()

    base_config_path = Path(args.base_config).expanduser()
    if not base_config_path.is_absolute():
        base_config_path = project_root / base_config_path
    base_config = _load_yaml(base_config_path, name="base config")
    base_config = _resolve_model_references(base_config, project_root)

    grid_path = Path(args.grid).expanduser()
    if not grid_path.is_absolute():
        grid_path = project_root / grid_path
    grid = _parse_grid_definition(grid_path)

    if args.max_trials is not None:
        object.__setattr__(grid, "max_trials", args.max_trials)
    if args.epochs is not None:
        object.__setattr__(grid, "fast_epochs", args.epochs)

    if grid.evaluation_mode != args.mode:
        print(
            f"[WARNING] grid evaluation mode is '{grid.evaluation_mode}' "
            f"but --mode is '{args.mode}'; using grid setting"
        )
        args.mode = grid.evaluation_mode

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(grid.seed)

    # Provide the full resolved config for constraint resolution so that
    # dotted paths like "model.dcl.num_clusters" can be walked.
    constraint_base = base_config
    trials = _generate_trials(grid, rng, constraint_base)
    print(f"Generated {len(trials)} trial(s) "
          f"({'random' if grid.search_strategy == 'random' else 'grid'} search)")

    if not trials:
        print("No trials to run after applying constraints.")
        return 1

    trial_script = (
        Path(args.trial_script).expanduser().resolve()
        if args.trial_script
        else None
    )

    results: list[TrialResult] = []

    for i, trial_overrides in enumerate(trials):
        trial_output_dir = output_dir / "trials" / f"trial_{i:04d}"
        trial_config_path = trial_output_dir / "config.yaml"

        print(
            f"\n{'=' * 70}\n"
            f"Trial {i + 1}/{len(trials)}: "
            f"{json.dumps(trial_overrides, sort_keys=True)}\n"
            f"{'=' * 70}"
        )

        config_path = synthesize_trial_yaml(
            base_config=base_config,
            overrides=trial_overrides,
            output_path=trial_config_path,
            fast_epochs=grid.fast_epochs,
            project_root=project_root,
        )

        spec = TrialSpec(
            trial_index=i,
            overrides=trial_overrides,
            output_dir=trial_output_dir,
            config_path=config_path,
        )

        if args.dry_run:
            print(f"  [DRY RUN] config: {config_path}")
            results.append(
                TrialResult(
                    trial_index=i,
                    grid_values=trial_overrides,
                    status="dry_run",
                    metrics={},
                    best_epoch=None,
                    wall_time_seconds=0.0,
                    error_message=None,
                    config_path=config_path,
                    output_dir=trial_output_dir,
                )
            )
            continue

        result = _run_trial(
            spec,
            mode=args.mode,
            project_root=project_root,
            device=args.device,
            timeout_per_trial=args.timeout_per_trial,
            trial_script=trial_script,
        )

        results.append(result)

        status_icon = {
            "completed": "[OK]",
            "failed": "[FAIL]",
            "oom": "[OOM]",
            "timeout": "[TIMEOUT]",
            "error": "[ERROR]",
        }.get(result.status, "[?]")

        print(
            f"  {status_icon} status={result.status} "
            f"metrics={result.metrics} "
            f"time={result.wall_time_seconds:.0f}s"
        )
        if result.error_message:
            print(f"  error: {result.error_message[:200]}")

    results_path = output_dir / "results.json"
    _write_results_json(results, results_path)
    print(f"\nResults written to {results_path}")

    completed = [r for r in results if r.status == "completed"]
    if completed:
        ranked = rank_results(completed, grid.direction, grid.metrics[0])
        sensitivity_scores = compute_sensitivity_scores(completed, grid.axes)
        report = SensitivityReport(
            grid_name=grid.name,
            grid_description=grid.description,
            direction=grid.direction,
            primary_metric=grid.metrics[0],
            metrics=grid.metrics,
            axes=grid.axes,
            all_results=results,
            ranked_results=ranked,
            sensitivity_scores=sensitivity_scores,
            best_config=ranked[0].grid_values if ranked else {},
            best_metrics=ranked[0].metrics if ranked else {},
            total_trials=len(results),
            completed_trials=len(completed),
            failed_trials=len(results) - len(completed),
        )

        report_path = output_dir / "report.md"
        md = generate_markdown_report(report)
        report_path.write_text(md, encoding="utf-8")
        print(f"Report written to {report_path}")

        csv_path = output_dir / "sensitivity.csv"
        generate_sensitivity_csv(report, csv_path)
        print(f"Sensitivity CSV written to {csv_path}")

        best_path = output_dir / "best_config.yaml"
        write_best_config(report, best_path)
        print(f"Best config written to {best_path}")

    failures = len(results) - len(completed)
    if args.dry_run:
        print(f"\nDry run completed. {len(results)} trial config(s) generated.")
        return 0
    if failures:
        print(f"\n{failures} trial(s) failed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Sensitivity analysis and reporting for hyperparameter grid search results.

Computes per-parameter sensitivity scores (range / mean), ranks
configurations, and generates Markdown/CSV reports.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from . import GridAxis, TrialResult
except ImportError:
    from scripts.hyperparam import GridAxis, TrialResult


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityReport:
    """Complete results and analysis of a hyperparameter search."""

    grid_name: str
    grid_description: str
    direction: str
    primary_metric: str
    metrics: tuple[str, ...]
    axes: tuple[GridAxis, ...]
    all_results: list[TrialResult]
    ranked_results: list[TrialResult]
    sensitivity_scores: dict[str, float]
    best_config: dict[str, Any]
    best_metrics: dict[str, float]
    total_trials: int
    completed_trials: int
    failed_trials: int


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _metric_value(result: TrialResult, metric_name: str) -> float:
    """Extract a metric value from a trial result.  Missing → inf for
    minimization or -inf for maximization (so it sorts last)."""
    value = result.metrics.get(metric_name)
    if value is not None and math.isfinite(value):
        return float(value)
    return float("inf")


def rank_results(
    results: list[TrialResult],
    direction: str,
    primary_metric: str,
) -> list[TrialResult]:
    """Sort completed trials by *primary_metric* according to *direction*.

    Parameters
    ----------
    results:
        Completed trial results.
    direction:
        ``"minimize"`` or ``"maximize"``.
    primary_metric:
        Name of the metric to sort by (e.g., ``"train_loss"``).
    """
    reverse = direction == "maximize"
    return sorted(
        results,
        key=lambda r: _metric_value(r, primary_metric),
        reverse=reverse,
    )


# ---------------------------------------------------------------------------
# Sensitivity scoring
# ---------------------------------------------------------------------------


def _group_by_axis(
    results: list[TrialResult],
    axis_path: str,
) -> dict[Any, list[float]]:
    """Group metric values by the value of a single axis.

    For each distinct value of *axis_path*, collect all
    *primary_metric* values from the results.
    """
    groups: dict[Any, list[float]] = defaultdict(list)
    for r in results:
        if r.status != "completed":
            continue
        axis_value = r.grid_values.get(axis_path)
        if axis_value is None:
            continue
        metric_value = r.metrics.get("train_loss")
        if metric_value is not None and math.isfinite(metric_value):
            groups[axis_value].append(metric_value)
    return dict(groups)


def compute_sensitivity_scores(
    results: list[TrialResult],
    axes: tuple[GridAxis, ...],
) -> dict[str, float]:
    """Compute sensitivity score for each axis.

    Score = (max_of_means - min_of_means) / grand_mean_of_means.

    A higher score indicates the parameter has a larger impact on the
    primary metric.
    """
    scores: dict[str, float] = {}
    for axis in axes:
        groups = _group_by_axis(results, axis.path)
        if len(groups) < 2:
            scores[axis.path] = 0.0
            continue

        means = [
            statistics.mean(values) if values else 0.0
            for values in groups.values()
        ]
        if not means:
            scores[axis.path] = 0.0
            continue

        range_val = max(means) - min(means)
        grand_mean = statistics.mean(means)
        if grand_mean == 0.0:
            scores[axis.path] = 0.0
        else:
            scores[axis.path] = range_val / abs(grand_mean)

    return scores


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) < 1e-4 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def generate_markdown_report(report: SensitivityReport) -> str:
    """Generate a comprehensive Markdown report for the grid search."""
    lines: list[str] = []

    lines.append(f"# {report.grid_name}")
    lines.append("")
    if report.grid_description:
        lines.append(f"_{report.grid_description}_")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Trials**: {report.total_trials} total, "
                 f"{report.completed_trials} completed, "
                 f"{report.failed_trials} failed")
    lines.append(f"- **Primary metric**: `{report.primary_metric}` "
                 f"({report.direction})")
    lines.append("")

    if report.best_metrics:
        lines.append("### Best Configuration")
        lines.append("")
        lines.append("```yaml")
        for key, value in sorted(report.best_config.items()):
            lines.append(f"{key}: {_format_value(value)}")
        lines.append("```")
        lines.append("")
        lines.append("**Best metrics**:")
        for key, value in sorted(report.best_metrics.items()):
            lines.append(f"- `{key}`: {_format_value(value)}")
        lines.append("")

    if report.sensitivity_scores:
        lines.append("## Sensitivity Analysis")
        lines.append("")
        lines.append("Parameters ranked by sensitivity score "
                     "(higher = more impact on performance):")
        lines.append("")
        lines.append("| Rank | Parameter | Sensitivity Score |")
        lines.append("|------|-----------|-------------------|")
        sorted_scores = sorted(
            report.sensitivity_scores.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        for rank, (name, score) in enumerate(sorted_scores, start=1):
            lines.append(f"| {rank} | `{name}` | {score:.4f} |")
        lines.append("")

    lines.append("## Trial Results (Ranked)")
    lines.append("")
    if report.ranked_results:
        header = (
            "| Rank | Trial | Status | "
            + " | ".join(
                axis.path.rsplit(".", 1)[-1] for axis in report.axes
            )
            + f" | {report.primary_metric} | Time (s) |"
        )
        lines.append(header)
        lines.append(
            "|------|-------|--------|"
            + "|".join("---" for _ in report.axes)
            + "|------|------|"
        )

        for rank, result in enumerate(report.ranked_results[:50], start=1):
            values = [
                _format_value(result.grid_values.get(axis.path, "?"))
                for axis in report.axes
            ]
            metric = _format_value(
                result.metrics.get(report.primary_metric, "N/A")
            )
            lines.append(
                f"| {rank} | {result.trial_index} | {result.status} | "
                + " | ".join(values)
                + f" | {metric} | {result.wall_time_seconds:.0f} |"
            )

        if len(report.ranked_results) > 50:
            lines.append(f"")
            lines.append(
                f"_... and {len(report.ranked_results) - 50} more trials_"
            )
    lines.append("")

    failures = [r for r in report.all_results if r.status != "completed"]
    if failures:
        lines.append("## Failed Trials")
        lines.append("")
        for r in failures:
            lines.append(
                f"- **Trial {r.trial_index}** ({r.status}): "
                f"{r.error_message or 'unknown error'}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def generate_sensitivity_csv(
    report: SensitivityReport,
    output_path: Path,
) -> None:
    """Write a CSV file with sensitivity scores and per-axis-value means."""
    rows: list[dict[str, Any]] = []

    if report.sensitivity_scores:
        for axis in report.axes:
            groups = _group_by_axis(report.all_results, axis.path)
            for value, metric_values in groups.items():
                rows.append(
                    {
                        "parameter": axis.path,
                        "value": value,
                        "mean_metric": (
                            statistics.mean(metric_values)
                            if metric_values
                            else None
                        ),
                        "std_metric": (
                            statistics.stdev(metric_values)
                            if len(metric_values) >= 2
                            else None
                        ),
                        "n_samples": len(metric_values),
                        "sensitivity_score": report.sensitivity_scores.get(
                            axis.path, 0.0
                        ),
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["parameter", "value", "mean_metric", "sensitivity_score"])
            writer.writerow([])
        return

    fieldnames = [
        "parameter",
        "value",
        "mean_metric",
        "std_metric",
        "n_samples",
        "sensitivity_score",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Best config export
# ---------------------------------------------------------------------------


def write_best_config(
    report: SensitivityReport,
    output_path: Path,
) -> None:
    """Write the best trial's configuration as a standalone YAML snippet."""
    import yaml

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"# Best configuration from grid search: {report.grid_name}\n"
            f"# Primary metric ({report.primary_metric}): "
            f"{_format_value(report.best_metrics.get(report.primary_metric, 'N/A'))}\n"
        )
        for key, value in sorted(report.best_metrics.items()):
            handle.write(f"#   {key}: {_format_value(value)}\n")
        handle.write("\n")
        yaml.safe_dump(
            dict(sorted(report.best_config.items())),
            handle,
            default_flow_style=False,
            sort_keys=False,
        )


__all__ = [
    "SensitivityReport",
    "compute_sensitivity_scores",
    "generate_markdown_report",
    "generate_sensitivity_csv",
    "rank_results",
    "write_best_config",
]
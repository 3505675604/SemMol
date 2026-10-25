"""Auditable table aggregation from measured per-seed results.

No reference scores are embedded here. Cross-task summaries use dimensionless
relative improvements; raw RMSE values in different units are never added.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RunMeasurement:
    model: str
    dataset: str
    metric: str
    unit: str
    seed: int
    value: float
    split_sha256: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "RunMeasurement":
        fields = ("model", "dataset", "metric", "unit", "split_sha256")
        for key in fields:
            if not isinstance(row.get(key), str) or not str(row[key]).strip():
                raise ValueError(f"measurement.{key} must be a nonempty string")
        seed, value = row.get("seed"), row.get("value")
        if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
            raise ValueError("measurement.seed must be a non-negative integer")
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("measurement.value must be finite")
        digest = str(row["split_sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("measurement.split_sha256 must be a SHA256 digest")
        return cls(*(str(row[key]).strip() for key in fields[:4]), int(seed), float(value), digest)


def summarize_measurements(rows: Iterable[RunMeasurement], *, expected_seeds: int = 10) -> list[dict[str, object]]:
    """Mean/sample-SD/count with an explicit seed and frozen-split contract."""
    if not isinstance(expected_seeds, int) or isinstance(expected_seeds, bool) or expected_seeds < 2:
        raise ValueError("expected_seeds must be an integer >=2")
    groups: dict[tuple[str, str, str, str], list[RunMeasurement]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, RunMeasurement):
            raise TypeError("rows must contain validated RunMeasurement objects")
        groups[(row.model, row.dataset, row.metric, row.unit)].append(row)
    if not groups:
        raise ValueError("no measurements supplied")
    output = []
    for (model, dataset, metric, unit), observations in sorted(groups.items()):
        seeds = [row.seed for row in observations]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed for {model}/{dataset}/{metric}")
        if len(seeds) != expected_seeds:
            raise ValueError(f"expected {expected_seeds} seeds for {model}/{dataset}/{metric}, got {len(seeds)}")
        digests = {row.split_sha256 for row in observations}
        if len(digests) != 1:
            raise ValueError("seed repeats must use one frozen outer split")
        values = [row.value for row in observations]
        output.append(dict(model=model, dataset=dataset, metric=metric, unit=unit,
                           n=len(values), mean=statistics.mean(values),
                           sample_std=statistics.stdev(values), seeds=sorted(seeds),
                           split_sha256=next(iter(digests))))
    return output


def paired_relative_improvements(
    rows: Iterable[RunMeasurement], *, model: str, baseline: str,
    metric: str = "rmse", lower_is_better: bool = True,
) -> dict[str, object]:
    """Pair seeds/splits within tasks before a dimensionless task macro average."""
    selected: dict[tuple[str, str, int], RunMeasurement] = {}
    for row in rows:
        if row.metric != metric or row.model not in {model, baseline}:
            continue
        key = row.model, row.dataset, row.seed
        if key in selected:
            raise ValueError(f"duplicate paired measurement {key}")
        selected[key] = row
    datasets = sorted({key[1] for key in selected})
    if not datasets:
        raise ValueError("no comparable measurements")
    task_results = []
    for dataset in datasets:
        current_seeds = {seed for name, task, seed in selected if name == model and task == dataset}
        reference_seeds = {seed for name, task, seed in selected if name == baseline and task == dataset}
        if not current_seeds or current_seeds != reference_seeds:
            raise ValueError(f"paired seed sets differ for {dataset}")
        relative = []
        for seed in sorted(current_seeds):
            current, reference = selected[model, dataset, seed], selected[baseline, dataset, seed]
            if current.unit != reference.unit or current.split_sha256 != reference.split_sha256:
                raise ValueError(f"unit or split mismatch for {dataset}/seed{seed}")
            if reference.value <= 0:
                raise ValueError("relative improvement requires a positive baseline")
            sign = 1.0 if lower_is_better else -1.0
            relative.append(sign * (reference.value - current.value) / reference.value)
        task_results.append(dict(dataset=dataset, seed_count=len(relative),
                                 mean_relative_improvement=statistics.mean(relative),
                                 paired_seed_values=relative))
    return dict(metric=metric, comparison=f"{model} versus {baseline}",
                aggregation="equal_weight_mean_of_task_mean_relative_improvements",
                task_count=len(task_results), per_task=task_results,
                macro_relative_improvement=statistics.mean(
                    task["mean_relative_improvement"] for task in task_results))

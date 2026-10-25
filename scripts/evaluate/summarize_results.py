"""Build publication table cells from measured per-seed CSV results."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.result_tables import RunMeasurement, paired_relative_improvements, summarize_measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV columns: model,dataset,metric,unit,seed,value,split_sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--model")
    parser.add_argument("--baseline")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    for row in raw:
        row["seed"], row["value"] = int(row["seed"]), float(row["value"])
    measurements = [RunMeasurement.from_mapping(row) for row in raw]
    result = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "source": str(args.input.resolve()), "statistical_unit": "training_seed",
              "table_cells": summarize_measurements(measurements, expected_seeds=args.expected_seeds)}
    if bool(args.model) != bool(args.baseline):
        parser.error("--model and --baseline must be supplied together")
    if args.model:
        result["rmse_relative_comparison"] = paired_relative_improvements(
            measurements, model=args.model, baseline=args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

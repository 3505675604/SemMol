"""Train-only semantic alignment from scratch, then ten-seed supervised fitting.

This entry point does not use an external molecular pretraining checkpoint.
Alignment uses the registered outer training view, never validation or test.
Use --prepare-only to validate provenance and write reviewable run configs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import subprocess
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.hyperparam.protocol import file_sha256


def _path(value: str | Path, root: Path = _ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _load(path: Path) -> dict:
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return result


def prepare_two_stage(
    finetune_config: Path,
    alignment_template: Path,
    output_dir: Path,
    *,
    alignment_epochs: int = 10,
    alignment_world_size: int = 8,
) -> tuple[Path, Path, dict]:
    """Construct and validate both actual stages without launching training."""
    if alignment_epochs < 1 or alignment_world_size < 1:
        raise ValueError("alignment epochs and world size must be positive")
    final = _load(finetune_config)
    if final.get("experiment", {}).get("mode") != "finetune":
        raise ValueError("the supplied configuration must be a finetuning experiment")
    data = final["data"]
    store_dir = _path(data["store_dir"])
    parent_manifest = store_dir / "build-manifest.json"
    build = json.loads(parent_manifest.read_text(encoding="utf-8"))
    descriptor = build["views"]["train"]
    train_manifest = _path(data["train_manifest"])
    if train_manifest != (store_dir / descriptor["path"]).resolve():
        raise ValueError("semantic alignment must use the registered outer training view")
    if file_sha256(train_manifest) != descriptor["sha256"]:
        raise ValueError("registered training manifest checksum differs")
    if len(final["eval"]["seeds"]) != 10 or len(set(final["eval"]["seeds"])) != 10:
        raise ValueError("the supervised benchmark requires ten distinct seeds")
    model = copy.deepcopy(final["model"])
    modalities = [m for m in model["modalities"] if m not in final.get("drop_modalities", [])]
    if "qm" in modalities:
        raise ValueError("this protocol supports structural modalities only")
    model["modalities"] = modalities
    model["freeze_encoders"] = False
    encoder_options = model["encoders"]
    if isinstance(encoder_options, str):
        encoder_options = _load(_path(encoder_options))
    if encoder_options.get("smiles", {}).get("mode", "scratch") != "scratch":
        raise ValueError("without external pretraining requires a from-scratch SMILES encoder")
    model["pretraining_heads"] = {m: {"enabled": False} for m in ("mlm", "graph", "geo")}
    alignment = _load(alignment_template)
    name = final["experiment"]["name"] + "_train_alignment"
    alignment["experiment"].update(name=name, seed=final["experiment"]["seed"])
    alignment["model"] = model
    alignment["data"] = {
        key: copy.deepcopy(data[key])
        for key in ("store_dir", "tokenizer_dir", "strict_modalities", "num_workers",
                    "pin_memory", "persistent_workers", "prefetch_factor")
    }
    alignment["data"].update(manifest_path=str(train_manifest), modalities=modalities)
    alignment["train"]["epochs"] = alignment_epochs
    alignment["train"]["batch_size"] = final["train"]["batch_size"]
    alignment["train"]["mixed_precision"] = final["train"]["mixed_precision"]
    alignment["loss"] = {"mlm": 0.0, "graph": 0.0, "geo": 0.0, "pseudo": 1.0, "alignment": 0.0}
    alignment["objective"] = {
        "warmup_epochs": 0, "alignment_metric": "mse", "contrastive_reduction": "sum",
        "modality_weights": {m: 1.0 for m in modalities if m != model["anchor_modality"]},
    }
    alignment["mask"] = {
        "smiles_ratio": 0.0, "node_ratio": 0.0, "edge_ratio": 0.0,
        "geo_noise_std": 0.0, "graph_mask_mode": "substructure",
    }
    alignment["distributed"]["world_size"] = alignment_world_size
    alignment["distributed"]["backend"] = final["distributed"]["backend"]
    alignment["distributed"]["sync_batchnorm"] = False
    output_dir = output_dir.resolve()
    alignment["output"].update(
        checkpoint_dir=str(output_dir / "alignment_checkpoints"),
        log_dir=str(output_dir / "alignment_logs"),
        save_every_n_epochs=alignment_epochs, resume=None,
    )
    final["pretrained_ckpt"] = str(output_dir / "alignment_checkpoints" / f"{name}_latest.pt")
    final["output"].update(
        checkpoint_dir=str(output_dir / "supervised_checkpoints"),
        log_dir=str(output_dir / "supervised_logs"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_dir / "alignment.yaml", output_dir / "supervised.yaml"
    for path, values in zip(paths, (alignment, final)):
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    plan = {
        "schema": "semmol.two_stage_downstream.v1",
        "external_pretraining": False,
        "alignment_pool": "registered_outer_training_split_only",
        "alignment_manifest": str(train_manifest),
        "alignment_manifest_sha256": file_sha256(train_manifest),
        "store_manifest_sha256": file_sha256(parent_manifest),
        "alignment_seed": alignment["experiment"]["seed"],
        "alignment_epochs": alignment_epochs,
        "alignment_world_size": alignment_world_size,
        "supervised_world_size": final["distributed"]["world_size"],
        "alignment_objective": "Eq10/11 semantic contrastive alignment only",
        "supervised_seeds": final["eval"]["seeds"],
        "note": "One fixed train-only self-supervised initialization, followed by ten independently seeded supervised fits. Outer validation/test never update DCL.",
        "status": "prepared_not_trained",
    }
    (output_dir / "protocol.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return *paths, plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="A normal ten-seed finetuning YAML")
    parser.add_argument("--alignment-template", default="configs/pretrain/pretrain_1m.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alignment-epochs", type=int, default=10)
    parser.add_argument("--alignment-world-size", type=int, default=8,
                        help="Alignment process/GPU count (default: 8)")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    alignment, supervised, plan = prepare_two_stage(
        _path(args.config), _path(args.alignment_template), _path(args.output_dir),
        alignment_epochs=args.alignment_epochs, alignment_world_size=args.alignment_world_size,
    )
    print(json.dumps(plan, sort_keys=True), flush=True)
    if args.prepare_only:
        return 0
    for stage, config in (("pretrain", alignment), ("finetune", supervised)):
        world_size = _load(config)["distributed"]["world_size"]
        script = _ROOT / "scripts" / stage / f"run_{stage}.py"
        command = [sys.executable]
        if world_size > 1:
            if args.device is not None:
                raise ValueError("omit --device with multiple GPUs so LOCAL_RANK selects devices")
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world_size}"]
        command += [str(script), str(config)]
        if args.device:
            command += ["--device", args.device]
        started = time.perf_counter()
        environment = {**os.environ, "PYTHONPATH": str(_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        try:
            subprocess.run(command, cwd=_ROOT, env=environment, check=True)
        except subprocess.CalledProcessError:
            plan.update(status="failed", failed_stage=stage)
            (_path(args.output_dir) / "protocol.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            raise
        plan.setdefault("stages", []).append({"stage": stage, "wall_clock_seconds": time.perf_counter() - started})
    plan["status"] = "completed"
    plan["alignment_checkpoint_sha256"] = file_sha256(_path(_load(supervised)["pretrained_ckpt"]))
    (_path(args.output_dir) / "protocol.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

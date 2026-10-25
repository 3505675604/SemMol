"""Real tiny LMDB, model, checkpoint and subprocess protocol integration."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune.run_two_stage import prepare_two_stage
from scripts.hyperparam.protocol import build_training_folds, file_sha256
from src.datasets import tokenizer_artifact_sha256
from src.datasets.storage import LmdbShardWriter, StoreMetadata, write_store_metadata
from src.models import resolve_semmol_config
from src.molecular.espf_tokenizer import ESPFTokenizer
from src.molecular.graph import smiles_to_pyg_graph
from src.trainers.checkpointing import model_contract_signature


def _fixture(directory: Path):
    texts = ["C1" + "C" * length + "1" for length in range(2, 22)]
    tokenizer = ESPFTokenizer.train(texts, min_frequency=1, max_merges=2)
    tokenizer_dir = directory / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    tokenizer_hash = tokenizer_artifact_sha256(tokenizer_dir)
    store = directory / "store"
    records = []
    for index, text in enumerate(texts):
        encoded = tokenizer.encode_plus(text, max_length=None)
        graph = smiles_to_pyg_graph(text, on_invalid="raise")
        records.append({
            "sample_id": f"bace:{index}", "source_index": index, "smiles": text,
            "input_ids": np.asarray(encoded["input_ids"], dtype=np.int32),
            "token_spans": np.asarray(encoded["token_spans"], dtype=np.int32),
            "graph": {"node_feat": graph.x.numpy(), "edge_index": graph.edge_index.numpy(),
                      "edge_feat": graph.edge_attr.numpy(), "num_nodes": graph.num_nodes},
            "labels": np.asarray([index % 2], dtype=np.float32), "label_mask": np.asarray([True]),
            "dataset_name": "bace", "task_type": "classification", "label_columns": ["Class"],
        })
    with LmdbShardWriter(store_dir=store, shard_id=0, start_index=0,
                         expected_records=20, map_size=16 * 1024 * 1024) as writer:
        for index, record in enumerate(records):
            writer.put(index, record)
    write_store_metadata(store, StoreMetadata(
        schema_version=1, record_count=20, records_per_shard=20, modalities=("1d", "2d"),
        tokenizer_sha256=tokenizer_hash, tokenizer_vocab_size=tokenizer.vocab_size,
        shards=("shard-000000.lmdb",),
    ))
    split_indices = {"train": list(range(16)), "valid": [16, 17], "test": [18, 19]}
    views = {}
    for split, indices in split_indices.items():
        path = store / f"{split}.npz"
        np.savez(path, record_index=np.asarray(indices), source_index=np.asarray(indices))
        views[split] = {"path": path.name, "record_count": len(indices), "sha256": file_sha256(path)}
    (store / "build-manifest.json").write_text(json.dumps({
        "schema": "semmol.moleculenet_store_build.v1", "status": "complete", "dataset_name": "bace",
        "record_count": 20, "tokenizer": {"artifact_sha256": tokenizer_hash, "vocab_size": tokenizer.vocab_size},
        "source": {"label_columns": ["Class"]},
        "task": {"type": "classification", "num_tasks": 1, "main_metric": "roc_auc"}, "views": views,
    }))
    scaffold_path = directory / "scaffolds.json"
    scaffold_path.write_text(json.dumps({"indices": split_indices,
                                         "scaffold_members": {f"ring-{i + 3}": [i] for i in range(20)}}))
    config = yaml.safe_load((ROOT / "configs/finetune/bace.yaml").read_text())
    config["experiment"]["name"] = "tiny_protocol"
    config["model"].update(
        modalities=["1d", "2d"],
        encoders={"shared_dim": 8,
                  "smiles": {"mode": "scratch", "tokenizer_dir": str(tokenizer_dir), "shared_dim": 8,
                             "scratch_hidden_size": 8, "scratch_num_hidden_layers": 1,
                             "scratch_num_attention_heads": 2, "scratch_intermediate_size": 16,
                             "scratch_max_position_embeddings": 64,
                             "scratch_hidden_dropout_prob": 0.0, "scratch_attention_probs_dropout_prob": 0.0},
                  "graph": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0}},
        projection={"output_dim": 4, "mlp": {"hidden_dim": 8, "dropout": 0.0}},
        dcl={"num_clusters": 2, "init_max_samples": 8, "init_num_iters": 2,
             "distributed_sync": False, "warmup_steps": 0},
        acsm={"num_retrieve": 1, "temperature": 0.2, "denoise_threshold": -1.0},
    )
    config["data"].update(store_dir=str(store), tokenizer_dir=str(tokenizer_dir),
                           train_manifest=str(store / "train.npz"), valid_manifest=str(store / "valid.npz"),
                           test_manifest=str(store / "test.npz"), num_workers=0, persistent_workers=False,
                           pin_memory=False)
    config["train"].update(batch_size=4, epochs=1, mixed_precision="none")
    config["output"].update(checkpoint_dir=str(directory / "checkpoints"), log_dir=str(directory / "logs"),
                             tensorboard=False, wandb=False)
    config["distributed"].update(world_size=1, backend="gloo", sync_batchnorm=False)
    source = resolve_semmol_config(config, project_root=ROOT).build()
    source.dcls["2d"].update_centers(torch.randn(8, 4))
    checkpoint = directory / "tiny.pt"
    # A body checkpoint excludes the task head, as real pretraining does.
    state = {name: value for name, value in source.state_dict().items() if not name.startswith("property_head.")}
    torch.save({"model": state, "model_signature": model_contract_signature(source)}, checkpoint)
    config["pretrained_ckpt"] = str(checkpoint)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path, config, scaffold_path


def _execute(script: str, arguments: list[str], timeout: int = 120):
    process = subprocess.run([sys.executable, str(ROOT / script), *arguments], cwd=ROOT,
                             env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                             capture_output=True, text=True, timeout=timeout)
    assert process.returncode == 0, process.stdout + "\n" + process.stderr
    return process


def test_cv_fold_real_training_uses_no_outer_validation_or_test_files(tmp_path):
    _, config, scaffold = _fixture(tmp_path)
    folds = build_training_folds(Path(config["data"]["train_manifest"]), scaffold, tmp_path / "folds")
    train, valid = folds[0]
    config["data"].update(train_manifest=str(train), valid_manifest=str(valid))
    config["cross_validation"] = {"sidecar": str(valid.with_suffix(".json"))}
    # A correct development-only entry point must not need these files.
    (tmp_path / "store/valid.npz").unlink()
    (tmp_path / "store/test.npz").unlink()
    path = tmp_path / "cv.yaml"
    path.write_text(yaml.safe_dump(config))
    result = _execute("scripts/hyperparam/run_cv_fold.py", [str(path), "--device", "cpu"])
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["purpose"] == "development-only"
    assert summary["test_evaluated"] is False
    assert "test_loss" not in summary
    assert np.isfinite(summary["val_loss"])


def test_two_stage_real_alignment_then_ten_supervised_seeds(tmp_path):
    config_path, config, _ = _fixture(tmp_path)
    result = _execute("scripts/finetune/run_two_stage.py", [
        str(config_path), "--output-dir", str(tmp_path / "two_stage"),
        "--alignment-epochs", "1", "--alignment-world-size", "1", "--device", "cpu",
    ])
    result_path = tmp_path / "two_stage/supervised_logs/tiny_protocol/ten_seed_results.json"
    summary = json.loads(result_path.read_text())
    assert len(summary["seeds"]) == 10
    assert all(np.isfinite(run["test_loss"]) for run in summary["runs"])
    alignment = torch.load(tmp_path / "two_stage/alignment_checkpoints/tiny_protocol_train_alignment_latest.pt", weights_only=False)
    assert alignment["version"] == 3
    assert alignment["model_signature"]["shared_mlp_projection"] is True


def test_two_stage_defaults_to_eight_gpu_alignment_and_supervision(tmp_path):
    config_path, config, _ = _fixture(tmp_path)
    config["distributed"].update(world_size=8, backend="nccl")
    config_path.write_text(yaml.safe_dump(config))
    alignment, supervised, plan = prepare_two_stage(
        config_path, ROOT / "configs/pretrain/pretrain_1m.yaml", tmp_path / "prepared",
    )
    assert plan["alignment_world_size"] == 8
    assert plan["supervised_world_size"] == 8
    assert yaml.safe_load(alignment.read_text())["distributed"]["world_size"] == 8
    assert yaml.safe_load(supervised.read_text())["distributed"]["world_size"] == 8
    _execute("scripts/finetune/run_two_stage.py", [
        str(config_path), "--output-dir", str(tmp_path / "prepared_cli"), "--prepare-only",
    ])
    cli_plan = json.loads((tmp_path / "prepared_cli/protocol.json").read_text())
    assert cli_plan["alignment_world_size"] == cli_plan["supervised_world_size"] == 8

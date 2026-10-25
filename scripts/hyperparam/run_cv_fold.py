"""Fit one inner scaffold fold; never load or report an outer test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run(config_path: str, device: str | None = None) -> int:
    from scripts.finetune.run_finetune import (
        _parse_configuration, _rank_zero_stage, _synchronized_local_stage,
    )
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel
    from scripts.hyperparam.protocol import file_sha256, validate_fold_sidecar
    from src.datasets import (
        FinetuningDataCollator, MoleculeNetDataset, create_dataloader,
        tokenizer_artifact_sha256,
    )
    from src.losses import DownstreamTaskLoss
    from src.models import resolve_semmol_config
    from src.molecular.espf_tokenizer import ESPFTokenizer
    from src.trainers.checkpointing import configuration_fingerprint, load_pretrained_semmol
    from src.trainers.common import initialize_distributed, seed_everything
    from src.trainers.finetune_trainer import (
        DownstreamTaskDefinition, FinetuningTrainer, FinetuningTrainerConfig,
        FinetuningValidationResult,
    )
    from src.trainers.runtime import (
        LoadedExperimentConfiguration, build_optimizer, build_scheduler,
        load_experiment_configuration, optimizer_steps_per_epoch,
    )

    started = time.perf_counter()
    loaded = load_experiment_configuration(config_path, expected_mode="finetune")
    values = dict(loaded.values)
    cv = values.pop("cross_validation")
    sidecar = Path(cv["sidecar"]).resolve()
    provenance = validate_fold_sidecar(sidecar)
    configuration = LoadedExperimentConfiguration(loaded.path, loaded.project_root, values)
    options = _parse_configuration(configuration)
    if options.train_manifest != Path(provenance["train_manifest"]):
        raise ValueError("configured training view differs from verified fold")
    if options.valid_manifest != Path(provenance["valid_manifest"]):
        raise ValueError("configured validation view differs from verified fold")
    # The final-test path in the original config is deliberately never opened.
    context = initialize_distributed(options.distributed, requested_device=device)
    opened = []
    try:
        seed_everything(options.experiment_seed, deterministic=True, cudnn_benchmark=False)
        tokenizer = ESPFTokenizer.from_pretrained(options.tokenizer_dir)
        tokenizer_hash = tokenizer_artifact_sha256(options.tokenizer_dir)
        datasets = []
        for manifest in (options.train_manifest, options.valid_manifest):
            dataset = MoleculeNetDataset(
                dataset_name=options.dataset_name, store_dir=options.store_dir,
                manifest_path=manifest, modalities=options.data_modalities,
                strict=True, expected_tokenizer_sha256=tokenizer_hash,
            )
            opened.append(dataset)
            datasets.append(dataset)
        if datasets[0].label_columns != datasets[1].label_columns:
            raise ValueError("inner train/validation label definitions differ")
        collator = FinetuningDataCollator(
            pad_token_id=tokenizer.pad_token_id, allow_partial_modalities=False,
        )
        loaders = [create_dataloader(
            dataset, batch_size=options.batch_size, collate_fn=collator,
            shuffle=index == 0, num_workers=options.num_workers,
            pin_memory=options.pin_memory, drop_last=False,
            persistent_workers=options.persistent_workers,
            prefetch_factor=options.prefetch_factor, seed=options.experiment_seed,
        ) for index, dataset in enumerate(datasets)]
        def prepare_model():
            resolved = resolve_semmol_config(values, project_root=loaded.project_root)
            model = resolved.build()
            load_pretrained_semmol(options.pretrained_checkpoint, model, map_location="cpu")
            model.to(context.device)
            if options.distributed["sync_batchnorm"]:
                if not context.distributed or context.device.type != "cuda":
                    raise ValueError("SyncBatchNorm CV requires a distributed CUDA job")
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            return model

        model = _synchronized_local_stage(context, "inner-CV model and checkpoint preparation", prepare_model)
        if context.distributed:
            # Enter the DDP collective only after all ranks pass local setup.
            # EMA state is frozen and must not be broadcast as ordinary BN buffers.
            model = DistributedDataParallel(
                model,
                device_ids=[context.device.index] if context.device.type == "cuda" else None,
                output_device=context.device.index if context.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=options.distributed["find_unused_parameters"],
            )
        optimizer = build_optimizer(model, options.optimizer)
        scheduler = build_scheduler(
            optimizer, options.scheduler,
            total_optimizer_steps=optimizer_steps_per_epoch(len(loaders[0]), options.accumulation_steps) * options.epochs,
        )
        output_dir = options.checkpoint_dir / options.experiment_name
        fingerprint = configuration_fingerprint({
            "config": values, "fold": provenance, "tokenizer": tokenizer_hash,
            "pretrained_checkpoint_sha256": file_sha256(options.pretrained_checkpoint),
            "purpose": "development-only", "selection": "validation_loss",
        })
        trainer = FinetuningTrainer(
            model=model,
            loss_fn=DownstreamTaskLoss(task_type=options.task_type, loss_type="mse", distributed_sync=context.distributed),
            optimizer=optimizer, scheduler=scheduler,
            train_loader=loaders[0], valid_loader=loaders[1], test_loader=None,
            task=DownstreamTaskDefinition(
                task_type=options.task_type, num_tasks=options.num_tasks,
                task_names=datasets[0].label_columns, main_metric=options.main_metric,
                metric_direction=options.early_stopping_mode,
            ),
            config=FinetuningTrainerConfig(
                max_epochs=options.epochs, best_checkpoint_path=output_dir / "best.pt",
                latest_checkpoint_path=output_dir / "latest.pt",
                gradient_accumulation_steps=options.accumulation_steps,
                precision=options.precision, gradient_clip_norm=options.gradient_clip_norm,
                early_stopping_patience=options.early_stopping_patience if options.early_stopping_enabled else None,
            ),
            context=context, config_fingerprint=fingerprint, validation_only=True,
        )
        result = trainer.fit()
        if not isinstance(result, FinetuningValidationResult):
            raise RuntimeError("CV returned a final-test result")
        summary = {
            "status": "completed", "purpose": "development-only",
            "cv_folds": provenance["n_folds"], "fold_index": provenance["fold_index"],
            "inner_seed": options.experiment_seed, "inner_seeds_per_fold": 1,
            "world_size": context.world_size, "device_mapping": "LOCAL_RANK" if context.distributed else str(context.device),
            "final_benchmark_seeds": len(options.seeds),
            "val_loss": result.validation.loss, "selection_metric": "validation_loss",
            "best_epoch": result.best_state.best_epoch,
            "test_evaluated": False, "wall_clock_seconds": time.perf_counter() - started,
            "provenance_sha256": file_sha256(sidecar),
        }
        def publish_result():
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "cv_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, sort_keys=True), flush=True)
        _rank_zero_stage(context, "inner-CV result publication", publish_result)
        return 0
    finally:
        for dataset in reversed(opened):
            dataset.close()
        context.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.device))

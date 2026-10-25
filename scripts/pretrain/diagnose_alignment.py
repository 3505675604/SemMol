"""Measure ACSM gradient conflicts on real batches and shared MLP parameters.

Requires a compatible trained checkpoint and processed data. This diagnostic
does not train, update EMA centers, or recreate any historical paper figure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def alignment_batch_diagnostic(model, output, acsm_criterion, accumulator=None) -> dict:
    from src.evaluation.diagnostics import compute_gradient_conflicts

    if not model.projection.share_mlp_weights or "shared_mlp" not in model.projection.projectors:
        raise ValueError("gradient conflicts require the actual shared MLP projector")
    if output.acsm_output is None:
        raise ValueError("alignment is not ready: DCL initialization/warmup has not completed")
    loss_output = acsm_criterion.compute(output.acsm_output, epoch=acsm_criterion.warmup_epochs)
    losses = {name: component.loss for name, component in loss_output.modality_losses.items()}
    if len(losses) < 2:
        raise ValueError("at least two ACSM target losses are needed for an inter-modal conflict")
    parameters = tuple(
        ("projection.shared_mlp." + name, parameter)
        for name, parameter in model.projection.projectors["shared_mlp"].named_parameters()
        if parameter.requires_grad
    )
    conflicts = compute_gradient_conflicts(losses, parameters, distributed=False)
    if accumulator is not None:
        accumulator.update(conflicts)
    coverage = {}
    for name, match in output.acsm_output.modality_matches.items():
        rows = int(match.valid_negative_rows.numel())
        eligible = int(match.valid_negative_rows.sum().item())
        coverage[name] = {
            "anchor_count": rows, "eligible_anchor_count": eligible,
            "negative_coverage": eligible / rows if rows else None,
            "total_retained_negatives": int(match.negative_count.sum().item()),
        }
    return {
        "gradient_conflicts": conflicts.as_dict(), "negative_coverage": coverage,
        "parameter_scope": "actual_shared_mlp_parameters",
        "parameter_count": sum(parameter.numel() for _, parameter in parameters),
    }


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def main() -> int:
    import torch
    from scripts.pretrain.run_pretrain import _parse_configuration, build_pretraining_criterion
    from src.datasets import PCQMMultimodalDataset, PretrainingDataCollator, create_dataloader, tokenizer_artifact_sha256
    from src.evaluation.diagnostics import GradientConflictAccumulator
    from src.models import resolve_semmol_config
    from src.molecular.espf_tokenizer import ESPFTokenizer
    from src.trainers.checkpointing import load_pretrained_semmol
    from src.trainers.common import move_batch_to_device, seed_everything
    from src.trainers.runtime import load_experiment_configuration
    from src.utils.io import sha256_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_batches < 1:
        parser.error("--max-batches must be positive")
    config = load_experiment_configuration(args.config, expected_mode="pretrain")
    options = _parse_configuration(config, resume_override=None)
    seed_everything(options.seed, deterministic=True, cudnn_benchmark=False)
    resolved = resolve_semmol_config(config.values, project_root=config.project_root)
    resolved.model_options["pretraining_heads"] = {name: {"enabled": False} for name in ("mlm", "graph", "geo")}
    model = resolved.build()
    load_pretrained_semmol(args.checkpoint, model, map_location="cpu")
    model.to(torch.device(args.device)).eval()
    criterion = build_pretraining_criterion(options, resolved.model_options, distributed_sync=False)
    tokenizer = ESPFTokenizer.from_pretrained(options.tokenizer_dir)
    dataset = PCQMMultimodalDataset(
        store_dir=options.store_dir, manifest_path=options.manifest_path,
        modalities=options.modalities, strict=True,
        expected_tokenizer_sha256=tokenizer_artifact_sha256(options.tokenizer_dir),
    )
    collator = PretrainingDataCollator(
        pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size,
        special_token_ids=tuple(tokenizer.vocab[token] for token in tokenizer.special_tokens),
        smiles_mask_ratio=0.0, node_mask_ratio=0.0, edge_mask_ratio=0.0,
        geo_noise_std=0.0, graph_mask_mode=options.graph_mask_mode,
        seed=options.seed, allow_partial_modalities=False,
    )
    loader = create_dataloader(dataset, batch_size=options.batch_size, collate_fn=collator,
                               shuffle=False, num_workers=0, pin_memory=False, seed=options.seed)
    before = {name: value.clone() for name, value in model.state_dict().items() if name.startswith("dcls.")}
    batches = []
    accumulator = GradientConflictAccumulator()
    try:
        for index, batch in enumerate(loader):
            if index >= args.max_batches:
                break
            batch = move_batch_to_device(batch, torch.device(args.device), non_blocking=False)
            output = model.forward_pretrain(batch, update_dcl=False)
            result = alignment_batch_diagnostic(model, output, criterion.acsm, accumulator)
            batches.append({"batch_index": index, **result})
        if not batches:
            raise ValueError("the selected diagnostic dataset is empty")
        if any(not torch.equal(value, model.state_dict()[name]) for name, value in before.items()):
            raise RuntimeError("diagnostic unexpectedly changed DCL state")
        result = _finite_json({
            "schema": "semmol.shared_gradient_diagnostic.v1", "checkpoint_sha256": sha256_file(args.checkpoint),
            "manifest_sha256": sha256_file(options.manifest_path), "seed": options.seed,
            "dcl_updated": False, "optimizer_steps": 0, "historical_figure_reproduction": False,
            "summary": accumulator.as_dict(), "batches": batches,
        })
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"output": str(output_path), "batches": len(batches), "dcl_updated": False}))
        return 0
    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())

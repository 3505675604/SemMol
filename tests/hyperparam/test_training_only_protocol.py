"""Protocol tests use temporary manifests and never launch model training."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune.run_two_stage import prepare_two_stage
from scripts.hyperparam.protocol import (
    assign_scaffold_folds, build_training_folds, file_sha256, validate_fold_sidecar,
)


class TrainingOnlyProtocolTests(unittest.TestCase):
    def test_groups_are_disjoint_deterministic_and_cover_training_pool(self):
        members = {str(i): [i * 2, i * 2 + 1] for i in range(15)}
        sources = list(range(30))
        folds = assign_scaffold_folds(sources, members, n_folds=10, seed=42)
        self.assertEqual(folds, assign_scaffold_folds(sources, members, n_folds=10, seed=42))
        self.assertEqual(sorted(x for fold in folds for x in fold), sources)
        locations = {x: i for i, fold in enumerate(folds) for x in fold}
        for group in members.values():
            self.assertEqual(len({locations[x] for x in group}), 1)

    def test_missing_scaffolds_and_too_few_groups_fail(self):
        with self.assertRaisesRegex(ValueError, "cover"):
            assign_scaffold_folds([1, 2, 3], {"a": [1, 2]}, n_folds=2)
        with self.assertRaisesRegex(ValueError, "enough"):
            assign_scaffold_folds([1, 2, 3], {"a": [1, 2, 3]}, n_folds=2)

    def test_npz_folds_exclude_outer_validation_test_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.arange(30, dtype=np.int64)
            np.savez(root / "train.npz", record_index=source, source_index=source)
            metadata = {"indices": {"train": list(range(30)), "valid": [30], "test": [31]},
                        "scaffold_members": {str(i): [i * 2, i * 2 + 1] for i in range(16)}}
            (root / "split.json").write_text(json.dumps(metadata))
            folds = build_training_folds(root / "train.npz", root / "split.json", root / "folds")
            self.assertEqual(len(folds), 10)
            for train, valid in folds:
                info = validate_fold_sidecar(valid.with_suffix(".json"))
                self.assertEqual(info["purpose"], "development-only")
                with np.load(valid) as view:
                    self.assertTrue(set(view["source_index"]).isdisjoint({30, 31}))
            first_train, first_valid = folds[0]
            np.savez(first_train, record_index=np.array([31]), source_index=np.array([31]))
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_fold_sidecar(first_valid.with_suffix(".json"))

    def test_outer_validation_molecule_cannot_become_cv_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "train.npz", record_index=np.arange(12), source_index=np.arange(12))
            metadata = {"indices": {"train": list(range(11))},
                        "scaffold_members": {str(i): [i] for i in range(12)}}
            (root / "split.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "outside"):
                build_training_folds(root / "train.npz", root / "split.json", root / "folds")

    def test_two_stage_uses_registered_train_and_never_requires_heldout_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "train.npz", record_index=np.arange(12), source_index=np.arange(12))
            build = {"views": {"train": {"path": "train.npz", "sha256": file_sha256(root / "train.npz"), "record_count": 12}}}
            (root / "build-manifest.json").write_text(json.dumps(build))
            config = yaml.safe_load((ROOT / "configs/finetune/bace.yaml").read_text())
            config["data"].update(store_dir=str(root), train_manifest=str(root / "train.npz"),
                                  valid_manifest=str(root / "missing-valid.npz"), test_manifest=str(root / "missing-test.npz"))
            config_path = root / "final.yaml"
            config_path.write_text(yaml.safe_dump(config))
            alignment_path, final_path, plan = prepare_two_stage(
                config_path, ROOT / "configs/pretrain/pretrain_1m.yaml", root / "run",
            )
            alignment = yaml.safe_load(alignment_path.read_text())
            final = yaml.safe_load(final_path.read_text())
            self.assertEqual(alignment["data"]["manifest_path"], str(root / "train.npz"))
            self.assertNotIn("valid_manifest", alignment["data"])
            self.assertNotIn("test_manifest", alignment["data"])
            self.assertEqual(alignment["loss"]["pseudo"], 1.0)
            self.assertTrue(all(alignment["loss"][key] == 0 for key in ("mlm", "graph", "geo", "alignment")))
            self.assertEqual(len(final["eval"]["seeds"]), 10)
            self.assertFalse(plan["external_pretraining"])
            self.assertEqual(plan["status"], "prepared_not_trained")
            self.assertTrue(final["pretrained_ckpt"].endswith("_train_alignment_latest.pt"))


if __name__ == "__main__":
    unittest.main()

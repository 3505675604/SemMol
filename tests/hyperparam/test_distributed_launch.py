"""Preserve declared GPU/process counts across grid and fold subprocesses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hyperparam.run_grid_search import synthesize_trial_yaml
from scripts.hyperparam.trial_runner import build_trial_command, launch_trial


def test_grid_synthesis_preserves_eight_processes_and_sync_batchnorm(tmp_path):
    base = yaml.safe_load((ROOT / "configs/pretrain/pretrain_1m.yaml").read_text())
    base["distributed"].update(world_size=8, sync_batchnorm=True)
    result = synthesize_trial_yaml(base, {}, tmp_path / "trial.yaml", fast_epochs=1, project_root=ROOT)
    trial = yaml.safe_load(result.read_text())
    assert trial["distributed"]["world_size"] == 8
    assert trial["distributed"]["sync_batchnorm"] is True


def test_eight_process_configuration_uses_torchrun_without_fixed_device():
    command = build_trial_command("python", "fold.py", "config.yaml", world_size=8, environment={})
    assert command[:3] == ["python", "-m", "torch.distributed.run"]
    assert "--nproc-per-node=8" in command
    assert "--nnodes=1" in command and "--standalone" in command
    assert "--device" not in command


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cpu"])
def test_fixed_device_cannot_silently_downgrade_or_pin_an_eight_process_job(device):
    with pytest.raises(ValueError, match="LOCAL_RANK"):
        build_trial_command("python", "fold.py", "config.yaml", world_size=8, device=device, environment={})


def test_explicit_single_process_cpu_configuration_remains_supported():
    assert build_trial_command("python", "fold.py", "config.yaml", world_size=1, device="cpu") == [
        "python", "fold.py", "config.yaml", "--device", "cpu",
    ]


def test_multi_node_process_product_must_match_the_configured_total():
    command = build_trial_command("python", "fold.py", "config.yaml", world_size=8, environment={
        "NNODES": "2", "NPROC_PER_NODE": "4", "NODE_RANK": "1", "MASTER_ADDR": "10.0.0.1",
    })
    assert "--nproc-per-node=4" in command and "--nnodes=2" in command
    with pytest.raises(ValueError, match="must equal"):
        build_trial_command("python", "fold.py", "config.yaml", world_size=8,
                            environment={"NNODES": "1", "NPROC_PER_NODE": "1"})


def test_real_torchrun_starts_all_eight_lightweight_probe_workers(tmp_path, monkeypatch):
    """No GPU, dataset, model, or optimizer is used by the probe workers."""
    monkeypatch.setattr("scripts.hyperparam.trial_runner._find_python", lambda: sys.executable)
    for name in ("NNODES", "NPROC_PER_NODE", "NODE_RANK"):
        monkeypatch.delenv(name, raising=False)
    ranks = tmp_path / "ranks"
    ranks.mkdir()
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "config = json.loads(Path(sys.argv[1]).read_text())\n"
        "rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))\n"
        "assert world == 8 and rank == local and 0 <= rank < 8\n"
        "Path(config['probe_dir'], f'rank_{rank}.json').write_text(json.dumps({'rank':rank,'local_rank':local,'world_size':world}))\n"
        "if rank == 0: print(json.dumps({'observed_world_size':world}), flush=True)\n",
        encoding="utf-8",
    )
    config = tmp_path / "probe.yaml"
    config.write_text(json.dumps({"distributed": {"world_size": 8}, "probe_dir": str(ranks)}))
    result = launch_trial(config, tmp_path / "launch", mode="finetune", project_root=ROOT,
                          trial_script=probe, timeout=60)
    assert result.status == "completed", result.stderr
    observations = [json.loads(path.read_text()) for path in ranks.glob("rank_*.json")]
    assert {row["rank"] for row in observations} == set(range(8))
    assert all(row["world_size"] == 8 for row in observations)
    launch = json.loads((tmp_path / "launch/launch.json").read_text())
    assert launch["configured_world_size"] == 8 and launch["launcher"] == "torchrun"

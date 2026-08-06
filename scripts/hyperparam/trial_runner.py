"""Single-trial subprocess executor for the hyperparameter grid search.

Each trial runs as a separate subprocess to avoid state leakage from the
fail-closed ``PretrainTrainer``.  Supports both pretraining and finetuning
modes, with automatic detection of CUDA OOM and timeout handling.

For fast evaluation, trials are run in single-process mode (no torchrun)
by passing ``--device`` directly to the training script.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_OOM_PATTERNS = (
    re.compile(r"out\s+of\s+memory", re.IGNORECASE),
    re.compile(r"CUDA\s+error", re.IGNORECASE),
    re.compile(r"CUDNN\s+status", re.IGNORECASE),
    re.compile(r"RuntimeError.*memory", re.IGNORECASE),
)

_JSON_LINE_PATTERN = re.compile(r'^\s*\{.*\}\s*$')


@dataclass(frozen=True)
class TrialRun:
    """Result of executing one trial."""

    status: str
    metrics: dict[str, float]
    best_epoch: int | None
    error_message: str | None
    stdout: str
    stderr: str
    return_code: int


def _find_python() -> str:
    """Find a usable Python executable, preferring the project conda env."""
    candidates = [
        os.environ.get("PYTHON_BIN"),
        os.environ.get("PY"),
        sys.executable,
        "python",
        "python3",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            result = subprocess.run(
                [candidate, "-c", "import torch; print(torch.__version__)"],
                capture_output=True,
                text=True,
                timeout=15,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            if result.returncode == 0 and result.stdout.strip():
                return candidate
        except Exception:
            continue
    return sys.executable


def _parse_final_metrics(stdout: str) -> dict[str, float]:
    """Try to extract a JSON metrics line from the training script's stdout.

    The training scripts write a JSON summary line containing key metrics
    at the end of a successful run.  We scan the last few lines for it.
    """
    lines = stdout.strip().splitlines()
    candidate_lines = lines[-50:] if len(lines) > 50 else lines

    for line in reversed(candidate_lines):
        if not _JSON_LINE_PATTERN.match(line.strip()):
            continue
        try:
            payload = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        metrics: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = float(value)
        if metrics:
            return metrics

    return {}


def _parse_best_epoch(stdout: str) -> int | None:
    """Try to extract the best epoch from the training output."""
    pattern = re.compile(
        r"best[_\s-]epoch[:\s=]+(\d+)",
        re.IGNORECASE,
    )
    match = pattern.search(stdout)
    if match:
        return int(match.group(1))
    return None


def _parse_train_loss(stdout: str) -> float | None:
    """Fallback: extract the last reported training loss."""
    pattern = re.compile(r"train[_\s-]loss[:\s=]+([\d.]+(?:e[+-]?\d+)?)", re.IGNORECASE)
    matches = pattern.findall(stdout)
    if matches:
        return float(matches[-1])
    return None


def _detect_oom(stderr: str) -> bool:
    """Check if the error output indicates a CUDA out-of-memory error."""
    for pattern in _OOM_PATTERNS:
        if pattern.search(stderr):
            return True
    return False


def _detect_timeout(return_code: int) -> bool:
    """Check if the process was killed by a timeout signal."""
    if os.name == "nt":
        return False
    return return_code == -9 or return_code == 124 or return_code == 137


def launch_trial(
    config_path: Path,
    output_dir: Path,
    *,
    mode: str,
    project_root: Path,
    device: str | None = None,
    timeout: int = 86400,
    trial_script: Path | None = None,
) -> TrialRun:
    """Execute one trial as a subprocess.

    Parameters
    ----------
    config_path:
        Absolute path to the synthesized trial YAML.
    output_dir:
        Directory where trial artifacts (logs, checkpoints) are written.
    mode:
        ``"pretrain"`` or ``"finetune"``.
    project_root:
        SemMol project root directory.
    device:
        Device override for single-process execution (e.g., ``"cuda:0"``
        or ``"cpu"``).  When set, the training script is invoked directly
        with ``--device`` rather than through ``torchrun``.
    timeout:
        Maximum wall-clock seconds for the trial subprocess.
    trial_script:
        Optional custom Python script to run instead of the default
        pretrain/finetune entry points.

    Returns
    -------
    TrialRun:
        Aggregated result of the trial execution.
    """
    python_bin = _find_python()
    output_dir.mkdir(parents=True, exist_ok=True)

    if trial_script is not None:
        script_path = str(trial_script)
    elif mode == "pretrain":
        script_path = str(
            project_root / "scripts" / "pretrain" / "run_pretrain.py"
        )
    elif mode == "finetune":
        script_path = str(
            project_root / "scripts" / "finetune" / "run_finetune.py"
        )
    else:
        raise ValueError(f"unsupported mode: {mode!r}")

    cmd = [python_bin, script_path, str(config_path)]

    if device is not None:
        cmd.extend(["--device", device])

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(project_root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    log_path = output_dir / "trial_output.log"
    err_path = output_dir / "trial_error.log"

    try:
        with log_path.open("w", encoding="utf-8") as stdout_handle, \
             err_path.open("w", encoding="utf-8") as stderr_handle:

            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=timeout,
            )

        return_code = result.returncode
        stdout_text = log_path.read_text(encoding="utf-8")
        stderr_text = err_path.read_text(encoding="utf-8")

    except subprocess.TimeoutExpired:
        return TrialRun(
            status="timeout",
            metrics={},
            best_epoch=None,
            error_message=f"trial exceeded timeout of {timeout}s",
            stdout="",
            stderr="",
            return_code=-1,
        )

    if return_code == 0:
        metrics = _parse_final_metrics(stdout_text)
        if not metrics:
            loss = _parse_train_loss(stdout_text)
            if loss is not None:
                metrics = {"train_loss": loss}

        return TrialRun(
            status="completed",
            metrics=metrics,
            best_epoch=_parse_best_epoch(stdout_text),
            error_message=None,
            stdout=stdout_text,
            stderr=stderr_text,
            return_code=return_code,
        )

    if _detect_oom(stderr_text):
        status = "oom"
    elif _detect_timeout(return_code):
        status = "timeout"
    else:
        status = "failed"

    error_lines = stderr_text.strip().splitlines()
    error_message = (
        error_lines[-1][:500] if error_lines else f"exit code {return_code}"
    )

    return TrialRun(
        status=status,
        metrics={},
        best_epoch=None,
        error_message=error_message,
        stdout=stdout_text,
        stderr=stderr_text,
        return_code=return_code,
    )


__all__ = [
    "TrialRun",
    "launch_trial",
]
#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 CONFIG_YAML [--device DEVICE]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "${script_dir}/../.." && pwd -P)"
config_argument="$1"
shift

if [[ "${config_argument}" == /* ]]; then
    config_path="${config_argument}"
else
    config_path="${PWD}/${config_argument}"
fi
if [[ ! -f "${config_path}" ]]; then
    echo "Configuration file does not exist: ${config_path}" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python executable is unavailable: ${python_bin}" >&2
    exit 127
fi

nnodes="${NNODES:-1}"
master_port="${MASTER_PORT:-29500}"

if [[ ! "${nnodes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NNODES must be a positive integer: ${nnodes}" >&2
    exit 2
fi
if [[ -n "${NODE_RANK+x}" ]]; then
    node_rank="${NODE_RANK}"
elif (( nnodes == 1 )); then
    node_rank="0"
else
    echo "NODE_RANK must be explicitly set when NNODES > 1" >&2
    exit 2
fi
if [[ ! "${node_rank}" =~ ^(0|[1-9][0-9]*)$ ]] \
    || (( node_rank >= nnodes )); then
    echo "NODE_RANK must be an integer in [0, NNODES): ${node_rank}" >&2
    exit 2
fi
if [[ -n "${MASTER_ADDR+x}" ]]; then
    master_addr="${MASTER_ADDR}"
elif (( nnodes == 1 )); then
    master_addr="127.0.0.1"
else
    echo "MASTER_ADDR must be explicitly set when NNODES > 1" >&2
    exit 2
fi
if [[ -z "${master_addr}" ]]; then
    echo "MASTER_ADDR cannot be empty" >&2
    exit 2
fi
if [[ ! "${master_port}" =~ ^[1-9][0-9]*$ ]] \
    || (( master_port < 1 || master_port > 65535 )); then
    echo "MASTER_PORT must be an integer in [1, 65535]: ${master_port}" >&2
    exit 2
fi

configured_world_size="$(
    "${python_bin}" - "${config_path}" <<'PY'
import pathlib
import sys

import yaml

configuration_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
with configuration_path.open("r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
if not isinstance(payload, dict):
    raise SystemExit("configuration root must be a mapping")
distributed = payload.get("distributed")
if not isinstance(distributed, dict):
    raise SystemExit("configuration.distributed must be a mapping")
world_size = distributed.get("world_size")
if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
    raise SystemExit("distributed.world_size must be a positive integer")
print(world_size)
PY
)"

if [[ ! "${configured_world_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Configured world size is invalid: ${configured_world_size}" >&2
    exit 2
fi

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    nproc_per_node="${NPROC_PER_NODE}"
else
    if (( configured_world_size % nnodes != 0 )); then
        echo "distributed.world_size=${configured_world_size} is not divisible by NNODES=${nnodes}" \
            >&2
        exit 2
    fi
    nproc_per_node="$((configured_world_size / nnodes))"
fi

if [[ ! "${nproc_per_node}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer: ${nproc_per_node}" >&2
    exit 2
fi
if (( nproc_per_node * nnodes != configured_world_size )); then
    echo "NNODES * NPROC_PER_NODE must equal distributed.world_size: " \
        "${nnodes} * ${nproc_per_node} != ${configured_world_size}" >&2
    exit 2
fi

cd -- "${project_root}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -m torch.distributed.run \
    --nnodes="${nnodes}" \
    --nproc-per-node="${nproc_per_node}" \
    --node-rank="${node_rank}" \
    --master-addr="${master_addr}" \
    --master-port="${master_port}" \
    "${project_root}/scripts/finetune/run_finetune.py" \
    "${config_path}" \
    "$@"

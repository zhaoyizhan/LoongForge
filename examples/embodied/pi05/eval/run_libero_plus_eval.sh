#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# LIBERO-Plus evaluation entry point.
#
# LIBERO-Plus is a drop-in replacement for the `libero` package (same import
# name and task interface), so evaluation reuses the standard LIBERO
# orchestrator unchanged. The only LIBERO-Plus-specific switch is
# `benchmark.variant: plus` in the YAML, which enables the perturbation-level
# summary (libero_plus_summary.csv).
#
# Prerequisites:
#   * BENCHMARK_PYTHON is the python of an env with LIBERO-Plus installed in
#     place of stock LIBERO, and its assets unpacked under
#     <LIBERO-plus>/libero/libero/assets
#   * the YAML's env.libero_config_path points at a config whose
#     benchmark_root / assets resolve to that checkout
#   * the YAML's server.python is an env that can import LoongForge
#
# Usage:
#   BENCHMARK_PYTHON=/path/to/libero_plus/bin/python \
#     bash run_libero_plus_eval.sh [CONFIG]
#
# Default CONFIG is configs/libero/libero_plus.yaml (all 4 suites via `sweep`).
#
# Useful overrides:
#   CONFIG=...                    single-suite / shard YAML
#   CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1   # pick another GPU
#   HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0        # if weights are remote
#   DRY_RUN=1                     print env/command only

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}
CONFIG=${1:-${CONFIG:-${SCRIPT_DIR}/configs/libero/libero_plus.yaml}}
BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-/path/to/libero_plus/bin/python}

if [[ ! -x "${BENCHMARK_PYTHON}" ]]; then
  echo "ERROR: BENCHMARK_PYTHON is not executable: ${BENCHMARK_PYTHON}" >&2
  echo "Set it to the python of the env that has LIBERO-Plus installed." >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi

# These only affect the benchmark client; the policy server is launched by the
# orchestrator from the YAML's server.python with its own environment.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# robosuite requires MUJOCO_EGL_DEVICE_ID to be present in CUDA_VISIBLE_DEVICES.
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # .pruned_init pickles under torch>=2.6
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
# Checkpoints and tokenizers in the YAML are local; keep the run offline.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

CMD=("${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run --config "${CONFIG}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "CONFIG=${CONFIG}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
  printf 'CMD:'
  printf ' %q' "${CMD[@]}"
  echo
  exit 0
fi

exec "${CMD[@]}"

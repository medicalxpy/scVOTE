#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Positional args (optional):
#   1) RESULTS_ROOT (default: results)
#   2) OUT_CSV      (default: results/evaluation/all_runs_metrics.csv)
RESULTS_ROOT="${1:-results}"
OUT_CSV="${2:-results/evaluation/all_runs_metrics.csv}"

# Optional env vars:
#   LABEL_KEY=cell_type
#   INCLUDE_TUNING=1
#   NO_JSON=1
#   DRY_RUN=1
#   RES_MIN=0.0 RES_MAX=2.0 RES_STEP=0.1 SEED=0
LABEL_KEY="${LABEL_KEY:-}"
INCLUDE_TUNING="${INCLUDE_TUNING:-0}"
NO_JSON="${NO_JSON:-0}"
DRY_RUN="${DRY_RUN:-0}"

RES_MIN="${RES_MIN:-0.0}"
RES_MAX="${RES_MAX:-2.0}"
RES_STEP="${RES_STEP:-0.1}"
SEED="${SEED:-0}"

args=(
  --results_root "${RESULTS_ROOT}"
  --out_csv "${OUT_CSV}"
  --res_min "${RES_MIN}"
  --res_max "${RES_MAX}"
  --res_step "${RES_STEP}"
  --seed "${SEED}"
)

if [[ -n "${LABEL_KEY}" ]]; then
  args+=(--label_key "${LABEL_KEY}")
fi
if [[ "${INCLUDE_TUNING}" == "1" ]]; then
  args+=(--include_tuning)
fi
if [[ "${NO_JSON}" == "1" ]]; then
  args+=(--no_json)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry_run)
fi

echo "[eval_all_results] repo_root=${ROOT_DIR}"
echo "[eval_all_results] results_root=${RESULTS_ROOT}"
echo "[eval_all_results] out_csv=${OUT_CSV}"
echo "[eval_all_results] cmd: ${PYTHON_BIN} test/eval_results_to_csv.py ${args[*]}"

exec "${PYTHON_BIN}" test/eval_results_to_csv.py "${args[@]}"


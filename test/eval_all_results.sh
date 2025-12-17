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
#   DRY_RUN=1
#   LOG_DIR=logs
#   RES_MIN=0.0 RES_MAX=2.0 RES_STEP=0.1 SEED=0
LABEL_KEY="${LABEL_KEY:-cell_type}"
INCLUDE_TUNING="${INCLUDE_TUNING:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs}"

RES_MIN="${RES_MIN:-0.0}"
RES_MAX="${RES_MAX:-2.0}"
RES_STEP="${RES_STEP:-0.1}"
SEED="${SEED:-0}"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${OUT_CSV}")"

_is_run_dir() {
  local name="$1"
  [[ "${name}" =~ _K[0-9]+$ ]]
}

_infer_method() {
  local name="$1"
  if [[ "${name}" == *"_structure_contrastive_"* ]]; then
    echo "structure_contrastive"
  elif [[ "${name}" == *"_structure_"* ]]; then
    echo "structure"
  elif [[ "${name}" == *"_contrastive_"* ]]; then
    echo "contrastive"
  elif [[ "${name}" == *"_baseline_"* ]]; then
    echo "baseline"
  else
    echo ""
  fi
}

_infer_base_dataset() {
  local name="$1"
  local base="$name"
  base="${base%%_structure_contrastive_K*}"
  base="${base%%_structure_K*}"
  base="${base%%_contrastive_K*}"
  base="${base%%_baseline_K*}"
  echo "$base"
}

_resolve_adata_path() {
  local base_dataset="$1"
  local direct="data/${base_dataset}.h5ad"
  if [[ -f "${direct}" ]]; then
    echo "${direct}"
    return 0
  fi
  local want
  want="$(echo "${base_dataset}" | tr '[:upper:]' '[:lower:]')"
  shopt -s nullglob
  for f in data/*.h5ad; do
    local stem
    stem="$(basename "${f}" .h5ad | tr '[:upper:]' '[:lower:]')"
    if [[ "${stem}" == "${want}" ]]; then
      echo "${f}"
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

_find_dataset_id() {
  local run_dir="$1"
  local k="$2"
  shopt -s nullglob
  local matches=("${run_dir}/cell_topic/"*_cell_topic_matrix_"${k}".pkl)
  shopt -u nullglob
  if [[ "${#matches[@]}" -eq 0 ]]; then
    return 1
  fi
  local f="${matches[0]}"
  local bn
  bn="$(basename "${f}")"
  bn="${bn%_cell_topic_matrix_${k}.pkl}"
  echo "${bn}"
}

_extract_json_path_from_log() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    return 1
  fi
  # Some environments may inject NUL bytes into logs; strip them for parsing.
  tr -d '\000' < "${log_path}" | sed -n 's/.*Saved evaluation metrics to //p' | tail -n 1
}

echo "[eval_all_results] repo_root=${ROOT_DIR}"
echo "[eval_all_results] results_root=${RESULTS_ROOT}"
echo "[eval_all_results] out_csv=${OUT_CSV}"
echo "[eval_all_results] log_dir=${LOG_DIR}"

manifest="$(mktemp)"
trap 'rm -f "${manifest}"' EXIT
printf "run_dir\trun_name\tbase_dataset\tmethod\texpected_k\tdataset_id\tadata_path\tlog_path\tjson_path\texit_code\n" > "${manifest}"

run_dirs=()
shopt -s nullglob
for d in "${RESULTS_ROOT}"/*; do
  [[ -d "${d}" ]] || continue
  name="$(basename "${d}")"
  case "${name}" in
    cell_embedding|cell_topic|topic_gene|gene_embedding|topic_embedding|gpu_stats|eval_cache|evaluation|incremental_eval|visualization|tuning)
      continue
      ;;
  esac
  if _is_run_dir "${name}"; then
    run_dirs+=("${d}")
  fi
done
if [[ "${INCLUDE_TUNING}" == "1" && -d "${RESULTS_ROOT}/tuning" ]]; then
  for d in "${RESULTS_ROOT}/tuning"/*; do
    [[ -d "${d}" ]] || continue
    run_dirs+=("${d}")
  done
fi
shopt -u nullglob

if [[ "${#run_dirs[@]}" -eq 0 ]]; then
  echo "[eval_all_results] No run dirs found under: ${RESULTS_ROOT}"
  exit 0
fi

total="${#run_dirs[@]}"
idx=0
for run_dir in "${run_dirs[@]}"; do
  idx=$((idx + 1))
  run_name="$(basename "${run_dir}")"
  k="${run_name##*_K}"
  if [[ ! "${k}" =~ ^[0-9]+$ ]]; then
    echo "[eval_all_results] skip (cannot parse K): ${run_name}"
    continue
  fi

  method="$(_infer_method "${run_name}")"
  base_dataset="$(_infer_base_dataset "${run_name}")"
  if ! adata_path="$(_resolve_adata_path "${base_dataset}")"; then
    echo "[eval_all_results] skip (missing .h5ad): ${run_name} (base_dataset=${base_dataset})"
    continue
  fi

  if ! dataset_id="$(_find_dataset_id "${run_dir}" "${k}")"; then
    echo "[eval_all_results] skip (missing cell_topic file): ${run_name}"
    continue
  fi

  out_dir="${run_dir}/evaluation"
  mkdir -p "${out_dir}"
  log_path="${LOG_DIR}/eval_${run_name}.log"

  echo "[eval_all_results] (${idx}/${total}) evaluating: ${run_dir}"
  cmd=(
    "${PYTHON_BIN}" evaluation.py
    --adata_path "${adata_path}"
    --results_dir "${run_dir}"
    --dataset "${dataset_id}"
    --n_topics "${k}"
    --label_key "${LABEL_KEY}"
    --res_min "${RES_MIN}"
    --res_max "${RES_MAX}"
    --res_step "${RES_STEP}"
    --out_dir "${out_dir}"
    --tag "${run_name}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[eval_all_results] DRY_RUN: %q ' "${cmd[@]}"
    echo
    continue
  fi

  set +e
  "${cmd[@]}" > "${log_path}" 2>&1
  exit_code=$?
  set -e

  json_path="$(_extract_json_path_from_log "${log_path}" || true)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_dir}" "${run_name}" "${base_dataset}" "${method}" "${k}" "${dataset_id}" \
    "${adata_path}" "${log_path}" "${json_path}" "${exit_code}" >> "${manifest}"
done

echo "[eval_all_results] Aggregating metrics into CSV: ${OUT_CSV}"
export _SCVOTE_MANIFEST_PATH="${manifest}"
export _SCVOTE_OUT_CSV="${OUT_CSV}"
exec "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

import pandas as pd

manifest = Path(os.environ["_SCVOTE_MANIFEST_PATH"])
out_csv = Path(os.environ["_SCVOTE_OUT_CSV"])

rows = []
with manifest.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        base = dict(r)
        json_path = r.get("json_path", "").strip()
        exit_code = int(r.get("exit_code", "0") or 0)
        status = "ok" if exit_code == 0 else "error"
        base["status"] = status
        base["error"] = ""
        if json_path:
            jp = Path(json_path)
            if jp.exists():
                try:
                    with jp.open("r", encoding="utf-8") as jf:
                        metrics = json.load(jf)
                    for k, v in metrics.items():
                        base[k] = v
                except Exception as e:  # noqa: BLE001
                    base["status"] = "error"
                    base["error"] = f"json_load_failed: {type(e).__name__}: {e}"
            else:
                base["status"] = "error"
                base["error"] = f"missing_json: {json_path}"
        else:
            if exit_code != 0:
                base["error"] = "evaluation_failed (see log_path)"
            else:
                base["error"] = "no_json_path_found_in_log"
        rows.append(base)

df = pd.DataFrame(rows)
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"[eval_all_results] wrote: {out_csv}")
if "status" in df.columns:
    print(df["status"].value_counts(dropna=False).to_string())
PY

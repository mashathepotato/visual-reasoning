#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${VR_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
WORKERS="${VLM_WORKERS:-16}"

cd "${REPO_ROOT}"

run_status=0
if "${PYTHON_BIN}" scripts/evaluate_sota_vlm_baselines.py \
    --model gpt-5.6-sol \
    --reasoning-effort high \
    --workers "${WORKERS}"; then
  if ! "${PYTHON_BIN}" scripts/evaluate_sota_vlm_baselines.py \
      --model gpt-5.6-sol \
      --reasoning-effort high \
      --workers "${WORKERS}" \
      --tasks sat_v2 \
      --sat-circular \
      --result-tag sat_circular; then
    run_status=1
    "${PYTHON_BIN}" scripts/evaluate_sota_vlm_baselines.py \
      --model gpt-5.6-sol --reasoning-effort high \
      --tasks sat_v2 --sat-circular --result-tag sat_circular --summarize-cache
  fi
else
  run_status=1
  "${PYTHON_BIN}" scripts/evaluate_sota_vlm_baselines.py \
    --model gpt-5.6-sol --reasoning-effort high \
    --result-tag partial --summarize-cache
fi

"${PYTHON_BIN}" scripts/compile_sota_comparison.py

exit "${run_status}"

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing ${python_bin}. Create the environment first; see manual.md." >&2
  exit 1
fi

if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu "${python_bin}" "${repo_root}/scripts/run_paper_mps_suite.py" "$@"
fi

exec "${python_bin}" "${repo_root}/scripts/run_paper_mps_suite.py" "$@"

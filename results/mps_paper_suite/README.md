# MPS paper-suite results

This directory is populated by `scripts/run_paper_mps_suite.sh`. Generated
results are not scientific until the status is complete and each run's
`preliminary` field is false.

Expected audit files:

- `status.json`
- `audit.json`
- `metrics.csv`
- `REPORT.md`

Raw checkpoints, logs, and per-example predictions live under
`models/runs/mps_paper_suite/` and are intentionally not committed by default.

## Training / eval results

This folder stores lightweight, commit-friendly summaries of runs (JSON) so we can
track progress without committing large checkpoints.

- `*.json`: run metadata + key metrics + the exact command used.

The MPS overnight runner stores its lightweight outputs under `mps_baselines/`:

- `overnight_status.json`: live progress, failures, durations, and output paths.
- `cnn_seeds0-1-2.json`: aggregate CNN metrics across completed seeds.
- `vit_seeds0-1-2.json`: aggregate ViT metrics across completed seeds.

Checkpoints, per-example predictions, resolved configs, and console logs remain
under ignored `models/runs/mps_baselines/` so large artifacts are not committed.

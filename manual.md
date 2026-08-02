# Manual

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA is optional for smoke tests and required in practice for the planned
multi-seed training matrix. Scripts choose CUDA, then Apple MPS, then CPU when a
configuration uses `"device": "auto"`. Set an explicit device in the config for
portable smoke checks.

## Verify the repository

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q utils scripts benchmarks
```

## Recreate fixed rotation manifests

The committed files contain the exact item IDs, render seeds, angles, and labels
used by every matched method. Regeneration should produce the same hashes.

```bash
.venv/bin/python scripts/generate_rotation_manifests.py
```

## Run the matched supervised baselines

### One-command Apple MPS overnight run

This runs the full colored-rotation CNN and ViT for seeds 0, 1, and 2
sequentially, prevents macOS sleep with `caffeinate`, and resumes by skipping
completed runs:

```bash
./scripts/run_mps_overnight.sh
```

Large checkpoints, predictions, resolved configs, and console logs are saved to
`models/runs/mps_baselines/{cnn,vit}/seedN/`. Lightweight aggregate results and
the live/final status file are saved to `results/mps_baselines/`. A failed run is
recorded and the remaining runs continue. Re-run the same command to retry only
incomplete runs, or use `./scripts/run_mps_overnight.sh --force` to rerun all six.

Before committing to the overnight run, verify the wrapper on tiny subsets:

```bash
./scripts/run_mps_overnight.sh --smoke --seeds 0 --force
```

Keep the Mac connected to power. You can follow progress in another terminal:

```bash
tail -f models/runs/mps_baselines/vit/seed0/console.log
```

Tiny end-to-end ViT smoke test (10 train examples and 8 examples per evaluation
split; pipeline validation only):

```bash
.venv/bin/python scripts/fot/train_supervised_baseline.py \
  --config configs/baselines/rotation_vit_smoke.json \
  --output-dir results/runs/colored_rotation_vit_smoke_seed0
```

Full colored-shape ViT runs:

```bash
.venv/bin/python scripts/fot/train_supervised_baseline.py \
  --config configs/baselines/rotation_vit_colored.json \
  --seed 0 --output-dir results/runs/colored_rotation_vit_seed0
.venv/bin/python scripts/fot/train_supervised_baseline.py \
  --config configs/baselines/rotation_vit_colored.json \
  --seed 1 --output-dir results/runs/colored_rotation_vit_seed1
.venv/bin/python scripts/fot/train_supervised_baseline.py \
  --config configs/baselines/rotation_vit_colored.json \
  --seed 2 --output-dir results/runs/colored_rotation_vit_seed2
```

Use `configs/baselines/rotation_cnn_colored.json` with the same commands for the
matched CNN. Aggregate only compatible configurations:

```bash
.venv/bin/python scripts/aggregate_results.py \
  results/runs/colored_rotation_vit_seed0 \
  results/runs/colored_rotation_vit_seed1 \
  results/runs/colored_rotation_vit_seed2 \
  --output results/aggregates/colored_rotation_vit_seeds0-2.json
```

The aggregator lists incomplete run directories in `missing_or_failed_runs` and
raises on incompatible configurations or duplicate seeds.

## Run notebooks

```bash
jupyter lab
```

Install Jupyter separately if notebook exploration is needed; notebooks are not
the canonical experiment entry points.

Notebooks live in:
- `notebooks/` (main pipelines)
- `benchmarks/` (eval + baselines)
- `examples/` (small demos)

## Generate figures / documents

- DINOv3 rotation baseline plots (downloads weights on first run) → `diagrams/`
```bash
python3 scripts/make_dinov3_rotation_baseline_plots.py
```

- Flow-matching training loss curve → `diagrams/`
```bash
python3 scripts/plot_fm_training_curve.py
```

- PPO training curves (expects a Stable-Baselines3 `progress.csv`) → `diagrams/`
```bash
python3 scripts/plot_ppo_training_curves.py
```

- Export rollout GIFs into labeled paper figures → `examples/paper_figures/`
```bash
python3 scripts/export_gifs_to_paper_frames.py --gifs-dir gifs --out-dir examples/paper_figures
```

## Where to find outputs

- Plots/figures: `diagrams/` (`.pdf` + `.png`)
- Rollout GIFs: `gifs/`
- Paper-ready PNG grids: `examples/paper_figures/`
- Training logs/checkpoints: `logs/`, `models/`

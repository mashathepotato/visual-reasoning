# Manual

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter pillow imageio
```

## Run notebooks

```bash
jupyter lab
```

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

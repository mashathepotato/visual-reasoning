# Visual Reasoning / Flow-of-Thought (FoT) — Context

Last updated: 2026-04-20 • repo: `visual-reasoning` • python: `3.12.12` (see `.venv/`)

Quickstart: `source .venv/bin/activate` (or run commands via `.venv/bin/python`)

## Goal / motivation (1-liner)
Add a **visual thinking layer** to VLMs: instead of “answer in one shot”, learn/execute *visual intermediate steps* (sketch/transform) and benchmark vs frozen-vision + LLM baselines.

## Core idea implemented here (FoT)
- `utils/fot/`: “Flow-of-Thought” building blocks extracted from notebooks.
- Use **Flow Matching (FM)** models as *dynamics operators* in image space (rotation / sketch progression).
- Use **PPO** to learn *control policies* that steer those dynamics (e.g., rotate until aligned, then commit SAME/DIFFERENT).
- Conditioning:
  - **Tetris rotation FM**: 1-channel images in `[-1,1]` + **frozen DINOv3** embedding.
  - **Colors rotation FM**: 3-channel images in `[0,1]` + **learned encoder** (DINO optional elsewhere).

## Tasks / datasets (synthetic + small 3D set)
- Rotation “same vs mirrored”:
  - Synthetic: Tetris shapes + colored rectangles (generated in `utils/llm_baselines.py`).
  - “3D blocks” pairs: preprocessed Ganis–Kievit stimuli in `data/train_pairs.npy` + `data/test_balanced.npy` (balanced 39/39).
- Mazes:
  - `maze-trace`: classify if a highlighted path is valid (`YES/NO`).
  - `maze-solve`: output a valid shortest-path-like move string (`UDLR...`).

Prompts used for eval are defined verbatim in `utils/llm_baselines.py` (`ROTATION_PROMPT`, `MAZE_TRACE_PROMPT`, `MAZE_SOLVE_PROMPT`).

## Where the “main stuff” lives
- Notebooks (end-to-end pipelines):
  - `notebooks/pipeline.ipynb` (colors FM + PPO rotation)
  - `notebooks/pipeline_tetris.ipynb` (tetris FM + PPO rotation)
  - `notebooks/pipeline_maze.ipynb` (maze FM sketcher + PPO progress control)
  - `notebooks/pipeline_3d.ipynb` (zero-shot 3D eval / model fusion)
- Reusable code:
  - `utils/fot/*` (envs, models, integrators, ops)
  - `utils/llm_baselines.py` (dataset builders + eval loops + caching)
- Script entrypoints:
  - FM training: `scripts/fot/train_fm_tetris.py`, `scripts/fot/train_fm_maze_sketcher.py`
  - PPO training: `scripts/fot/train_ppo_tetris_fm.py`, `scripts/fot/train_ppo_colors_fm.py`, `scripts/fot/train_ppo_maze_progress.py`
  - Baselines: `benchmarks/vipergpt_tests.py`
  - Plotting: `scripts/plot_fm_training_curve.py`, `scripts/plot_ppo_training_curves.py`, `scripts/make_dinov3_rotation_baseline_plots.py`

## Artifacts / outputs
- Figures: `diagrams/` (FM/PPO templates + baseline plots + training curves)
- Checkpoints: `models/`
- Training logs: `logs/*/progress.csv` (SB3 PPO)
- Rollout GIFs: `gifs/` and `examples/*.gif`; paper-ready frames: `examples/paper_figures/`

## Concrete results captured in-repo (so far)
### ViperGPT-style *tools baseline* (no learning)
From `benchmarks/vipergpt_results_seed0.json`:
- tetris rotation acc: **0.82** (n=400)
- colored shapes rotation acc: **0.8675** (n=400)
- 3d blocks rotation acc: **0.6538** (n=78)
- maze trace acc: **1.0** (n=300)
- maze solve success: **1.0** (n=60)

Re-run:
`.venv/bin/python benchmarks/vipergpt_tests.py --seed 0 --n-rotation 400 --n-maze-trace 300 --n-maze-solve 60 --tasks tetris,colors,3d,maze-trace,maze-solve --out benchmarks/vipergpt_results_seed0.json`

## External benchmark training (SAT-v2, V*Bench)
FoT “heatmap sketcher” MCQ training scripts:
- SAT-v2: `scripts/fot/train_sat_v2_fot_vqa.py`
  - For quick runs: `--streaming --max-train N --max-val M`
- V*Bench (V-STaR w/ bbox): `scripts/fot/train_vstar_fot_vqa.py`

Notes:
- Requires the `datasets` Python package and access to HuggingFace dataset files.
- If you hit network/DNS restrictions, run on a machine with internet (or pre-download HF caches).

## Training log (append-only)
2026-04-20:
- Pending: run FoT training scripts and record metrics into `results/*.json`.

### PPO training snapshots (from last row of each `logs/*/progress.csv`)
- `ppo_tetris`: steps=51200, ep_rew_mean=61.55, ep_len_mean=9.20
- `ppo_rotation`: steps=51200, ep_rew_mean=9.61, ep_len_mean=71.95
- `ppo_colors_fm`: steps=51200, ep_rew_mean=-0.33, ep_len_mean=15.24
- `ppo_tetris_fm`: steps=61440, ep_rew_mean=11.05, ep_len_mean=30.71
- `ppo_tetris_fm_consistency`: steps=51200, ep_rew_mean=45.95, ep_len_mean=13.98
- `ppo_maze_fm`: steps=100352, ep_rew_mean=66.34, ep_len_mean=180.00
- `ppo_maze_fm_actions`: steps=100352, ep_rew_mean=-5.04, ep_len_mean=179.75
- `ppo_maze_fm_progress`: steps=100352, ep_rew_mean=6.40, ep_len_mean=9.46

## Baselines (LLM + frozen vision)
- LLM notebooks: `benchmarks/gpt_baselines.ipynb`, `benchmarks/claude_baselines.ipynb`
  - Cached calls: `benchmarks/llm_baselines_cache.jsonl`, `benchmarks/claude_baselines_cache.jsonl`
- Frozen DINOv3 baselines: `benchmarks/dinov3_*_baseline.ipynb`
- DINOv3 rotation separation plots already exported: `diagrams/*baseline_separation.(png|pdf)`

## Gotchas / assumptions (fast)
- `requirements.txt` is **not exhaustive** for all scripts/notebooks (common missing deps: `gymnasium`, `stable-baselines3`, `kornia`, `Pillow`, `imageio`, `jupyter`, `tensorboard`).
- Image conventions matter:
  - tetris FM expects grayscale `[-1,1]` FM tensors; colors uses RGB `[0,1]`.
- 3D “blocks” data used by eval is **already** in `data/test_balanced.npy`; regenerating from raw jpgs requires `data/ganis_kievit_data/stimuli_jpgs/` + `scripts/prep_ganis_kievit.py`.

## Next useful steps (suggested)
- Add a single “eval harness” that reports *comparable metrics* for learned PPO/FoT agents (not just reward curves).
- Consolidate deps into a reproducible env file (or split `requirements.txt` into core vs RL vs notebooks).
- Tighten the 3D story: standardize whether 3D uses raw jpgs vs only preprocessed `.npy`, and keep scripts consistent.

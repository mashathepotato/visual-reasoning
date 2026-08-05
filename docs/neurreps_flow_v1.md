# NeurReps flow rebuild (v1)

This experiment family isolates flow matching from PPO. It does not run, train,
or evaluate a policy. Its purpose is to establish whether the continuous visual
dynamics themselves learn sharp, meaningful intermediate states before a
decision mechanism is added.

## Dynamics

All tasks use `TrajectoryFlowField`, a coordinate-aware, time-conditioned
spatial U-Net with GroupNorm, SiLU activations, and FiLM conditioning. The model
is conditioned on the current state, the initial/task condition, continuous time,
and a three-dimensional action representation.

- Rotation uses a learned dense transport field. The training target is the
  tangent of the rendered rotation-group orbit, expressed as a spatial velocity
  field. ODE integration advects the image through that learned field, which
  preserves object identity and sharpness better than independently adding RGB
  pixel velocities. No exact rotator is called during model rollout.
- Maze uses an additive neural-state field. Its ground-truth process is a
  cumulative shortest-path trace, and its objective combines a foreground-
  balanced local tangent loss with differentiable multi-step BCE and Dice
  rollout losses.

Rotation sources are randomly pre-rotated and mirrored during training, so the
conditioning distribution matches inference. Every ground-truth image is
rendered once from a common base with black padding; rotations are never
accumulated through repeated resampling.

For Tetris, the flow is trained on J/L/S/Z and validated on held-out F/P shapes.
This makes the flow-quality gate an operator-transfer test rather than a
memorization test over all six objects. Colored validation uses a disjoint fixed
procedural seed range.

## Quality control and model selection

Each run saves:

- `audit_best.png`: fixed validation examples at t = 0, 0.25, 0.5, and 1,
  showing ground truth, generated state, and absolute error;
- `audits/epoch_*.png`: periodic grids using the same fixed examples;
- `quality_metrics.json`: endpoint and complete-trajectory metrics;
- `epoch_metrics.json`: all training and validation history;
- `resolved_config.json` and `run_metadata.json`: configuration, Git, package,
  device, and command provenance;
- `best_checkpoint.pt` and `summary.json`.

Rotation checkpoints are selected by validation silhouette IoU. Maze checkpoints
are selected by validation path IoU, not background-dominated pixel MSE. Maze QC
also reports path precision, path recall, goal-reached rate, and obstacle-
violation rate. Multi-seed aggregates include Student-t 95% confidence intervals;
a single smoke seed correctly reports an undefined interval.

## Running on Apple MPS

Run all three tasks and three seeds:

```bash
.venv/bin/python scripts/run_neurreps_flow_suite.py --profile overnight --device mps
```

The runner is resumable. A completed `summary.json` causes that stage to be
skipped, and aggregate JSON, CSV, Markdown, and status files are refreshed under
`results/neurreps_flow_v1/overnight`. Smoke and overnight checkpoints also use
separate profile directories, so a smoke test can never make the overnight run
incorrectly resume. Use `--rerun` only when intentionally replacing a completed
stage within the selected profile.

For a short installation/backend check:

```bash
.venv/bin/python scripts/run_neurreps_flow_suite.py --profile smoke --seeds 0 --device mps --rerun
```

Smoke results are marked preliminary and must not be reported as paper results.

## Preserved earlier baselines

The completed 65-stage baseline suite is frozen at
`results/baseline_archives/mps_paper_suite_2026-08-03`. Its full checkpoint tree
is preserved read-only under `models/archives/mps_paper_suite_2026-08-03` using
an APFS copy-on-write clone. Both trees have SHA-256 manifests. New flow-v1 runs
write only under `models/runs/neurreps_flow_v1`, so they cannot overwrite the
old suite.

Hypothesis-conditioned competing flows and the replacement for PPO are
deliberately out of scope for v1. They should be designed only after these flow
quality gates pass across seeds.

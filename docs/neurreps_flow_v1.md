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

## Frozen-checkpoint post-hoc audit

The original rotation rollout repeatedly sampled the previous generated image.
That numerical renderer accumulated interpolation blur even when the learned
transport field was accurate. `integrate_deformation_times` now integrates a
backward characteristic map while retaining the checkpoint's native recurrent
state, then samples the original source exactly once at every requested time.
It supports arbitrary sorted times in `[0, 1]`, not only the training grid.

All nine completed checkpoints were re-evaluated without retraining or model
selection:

| Task | Metric | Original rollout | Single-source rollout |
|---|---|---:|---:|
| Tetris | silhouette IoU | 0.792 | 0.874 |
| Tetris | PSNR | 19.88 dB | 23.58 dB |
| Tetris | sharpness ratio | 0.816 | 1.003 |
| Colored shapes | silhouette IoU | 0.671 | 0.825 |
| Colored shapes | PSNR | 18.35 dB | 21.54 dB |
| Colored shapes | sharpness ratio | 0.841 | 1.137 |

The arbitrary-time `t=0.37` MSE was 0.00185 for Tetris and 0.00455 for colored
shapes. This establishes that the learned field can be decoded continuously and
that much of the earlier apparent rotation failure was a rendering artifact.
Full per-seed values, confidence intervals, and image grids are in
`results/neurreps_flow_v1/posthoc_v2`.

The same audit adds temporal-causality metrics for the maze flow. Its final path
remains excellent (IoU 0.975, goal reached 1.0, obstacle violations 0), and mean
intermediate prefix IoU is 0.842. However, 31.7% of future-path pixels cross the
activation threshold before their ground-truth step; mean future-path intensity
is 0.164. The current model therefore learns a mostly monotone final route but
does not yet support a strong claim of strictly causal, step-by-step search.

Reproduce the frozen audit on MPS with:

```bash
.venv/bin/python scripts/evaluate_neurreps_flow_posthoc.py --device mps
```

## Zero-shot Ganis-Kievit diagnostic

After the renderer gate passed, the six frozen 2-D rotation checkpoints were
applied to all 78 balanced Ganis-Kievit 3-D block pairs. No 3-D image or label
was used for training, validation, threshold fitting, or checkpoint selection.
For each pair, the evaluator compares reconstruction error under an original-
source rotation hypothesis and a horizontally reflected-source hypothesis. It
reports both the supplied angular disparity (including both signs) and a
label-free full-angle marginalization.

| 2-D training domain | Supplied-angle accuracy | AUC | Marginalized accuracy | AUC |
|---|---:|---:|---:|---:|
| Tetris | 0.585 [0.537, 0.634] | 0.639 | 0.564 [0.509, 0.619] | 0.645 |
| Colored shapes | 0.615 [0.505, 0.726] | 0.647 | 0.581 [0.469, 0.693] | 0.662 |

These are weak, angle-dependent zero-shot signals rather than a solved 3-D
reasoning result. Performance is strong at 0 degrees, generally informative at
50/100 degrees, and reverses or degrades at 150 degrees. The generated states
are coherent planar rotations, but planar reflection is not a physical mirrored
3-D transformation. In addition, the legacy dataset preparation has object-
identity overlap and supplies no ground-truth intermediate views. Treat this as
a transparent transfer diagnostic and motivation for a future 3-D-aware
hypothesis-conditioned flow, not as unseen-object OOD evidence.

Reproduce it with:

```bash
.venv/bin/python scripts/evaluate_neurreps_flow_3d_zero_shot.py --device mps
```

The complete predictions, per-angle breakdowns, checkpoint/data hashes,
confidence intervals, and visual trajectories are in
`results/neurreps_flow_v1/ganis3d_zero_shot`.

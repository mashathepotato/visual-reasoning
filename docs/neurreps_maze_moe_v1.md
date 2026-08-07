# Maze frozen-expert mixture (v1)

This experiment tests whether spatial representations learned by the Tetris and
colored-shape rotation flows transfer to progressive maze tracing. It is a dense,
pixelwise two-expert mixture: both frozen rotation encoders are evaluated for
every maze, a trainable router blends their full-resolution hidden features, and
a maze-specific additive flow decoder predicts the trace dynamics. PPO is not
used.

## Fairness and scope

The learned components have 898,643 parameters, close to the scratch maze
flow's 887,905 trainable parameters (+1.2%). The mixture additionally uses
447,364 frozen pretrained parameters. Both methods use 3,000 training mazes,
400 fixed disjoint validation mazes, 30 epochs, eight integration steps, the
same temporal targets and losses, three seeds, and endpoint-IoU checkpoint
selection.

This is representation transfer, not zero-shot task transfer: the frozen
rotation experts never update, but the router, feature projections, and maze
decoder are trained on maze trajectories.

## Three-seed result

| Method | Endpoint IoU | Trajectory MSE | Prefix IoU | Premature activation |
|---|---:|---:|---:|---:|
| Scratch maze flow | 0.9748 | 0.00942 | 0.8423 | 0.3165 |
| Learned expert mixture | 0.9786 | 0.00944 | 0.8377 | 0.3429 |
| Forced uniform gate | 0.9771 | 0.00965 | 0.8359 | 0.3349 |
| Forced Tetris-only gate | 0.5741 | 0.05499 | 0.4364 | 0.1749 |
| Forced colored-only gate | 0.5608 | 0.04776 | 0.5591 | 0.1389 |

The learned mixture matches the scratch flow within the uncertainty from three
seeds. It does not establish an adaptive-routing improvement: learned versus
uniform endpoint differences are +0.00544, -0.00030, and -0.00070 across the
three seeds. Mean router entropy is 0.6828, close to the two-expert maximum of
0.6931, and the saved router maps are spatially low-contrast. The current
mechanism therefore behaves like dense feature fusion rather than distinct
local expert specialization. Because the decoder also sees the raw maze
condition, matching scratch does not demonstrate that the frozen rotation
features improve maze reasoning.

The forced single-expert interventions are much worse on average, but they are
not independently trained baselines. They show that the jointly trained decoder
depends on its mixture regime; they do not prove that each pretrained expert
contributes unique information or that rotation pretraining is useful. The next
causal ablation should train fixed uniform, Tetris-only, colored-only, zeroed
feature, randomly initialized frozen-expert, and scratch variants from
initialization under the same parameter and data budget.

Endpoint selection also hides a process-quality trade-off. A descriptive,
post-hoc scan of saved validation histories found that all three seeds had their
lowest premature activation among epochs within 0.01 endpoint IoU of the best
at epoch 24. Those states average 0.9708 endpoint IoU, 0.8330 prefix IoU, 0.01046
trajectory MSE, and 0.2766 premature activation. They were not used for the
primary result and were not retained as checkpoints. A process-aware selection
criterion should be declared before the next experiment.

## Reproduction and audit

Run or resume the three-seed suite on Apple MPS:

```bash
.venv/bin/python scripts/run_neurreps_maze_moe_suite.py --profile overnight --device mps
```

Add `--rerun` only to intentionally replace completed seed outputs. The result
package under `results/neurreps_maze_moe_v1/overnight` contains the aggregate
JSON and CSV, paired deltas, raw and display confidence intervals, per-seed
summaries, histories, resolved configurations, run metadata, checkpoint hashes,
activation/router audit images, and a SHA-256 artifact manifest. Checkpoints remain under
`models/runs/neurreps_maze_moe_v1/overnight` and are not committed to Git.

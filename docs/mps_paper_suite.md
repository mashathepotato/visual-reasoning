# Comprehensive MPS paper suite

The paper suite is a resumable, single-command experiment matrix for the native
Tetris rotation, colored-shape rotation, Ganis-Kievit 3-D blocks, and maze tasks,
plus the external SAT-v2 dynamic spatial-reasoning benchmark.

## Why SAT-v2

SAT-v2 was selected because it adds dynamic camera/object-motion and perspective
questions with public training supervision. This is closer to the paper's claim
about explicit visual state transitions than V*Bench, which primarily tests
high-resolution visual search, or the broader BLINK collection. SAT reports 175K
synthetic question-answer pairs, 20K scenes, and a 150-question real-image test
set, and shows transfer from simulated spatial training to real-image benchmarks:

- SAT paper and protocol: <https://arxiv.org/abs/2412.07755>
- BLINK comparison benchmark: <https://arxiv.org/abs/2404.12390>
- SRBench's broader spatial taxonomy: <https://arxiv.org/abs/2503.19707>

The SAT-v2 loader tiles all images associated with a question into one visual
canvas; it never silently discards all but the first frame. The default overnight
profile streams 10,000 train and 2,000 validation questions per seed, then
evaluates the complete 150-question real-image test split, to avoid the roughly
5 GB full training download. The first run requires network access and may
pause while Hugging Face fetches its first data shard. Subsequent runs reuse the
local cache.

## Experiment matrix

The overnight profile runs seeds 0, 1, and 2 for:

- a cached program-of-vision-tools baseline on all four native datasets;
- exact rendered-rotation search under both rotation and mirror hypotheses;
- matched CNN and from-scratch ViT classifiers on both fixed rotation datasets;
- a frozen DINOv3 ViT-S/16 pair encoder with a trained nonlinear relation head on
  Tetris, colored shapes, and zero-shot Ganis-Kievit transfer;
- matched no-trace and visual-heatmap-trace models on SAT-v2;
- independently retrained Tetris and colored rotation-orbit flow models;
- independently retrained maze trace sketchers;
- PPO controllers trained on each corresponding frozen flow model;
- validation-selected controller thresholds evaluated on fixed ID and held-out-
  angle manifests;
- zero-shot transfer of both 2-D flow models to Ganis-Kievit 3-D blocks.

The `extended` profile additionally partially fine-tunes the final two DINOv3
transformer blocks. DINOv3 ViT-S/16 is a roughly 21M-parameter externally
pretrained model, so its LVD-1689M pretraining and trainable parameter count are
recorded explicitly. Official model details and weights are documented by Meta:
<https://github.com/facebookresearch/dinov3>.

Partial DINO fine-tuning freezes LayerNorm affine parameters because PyTorch 2.9
on MPS can produce non-finite gradients for those parameters. Attention and MLP
weights in the final blocks remain trainable. The smoke test exercises this path.

## Statistics and audit trail

Every compatible classification run records:

- resolved arguments, command, dependency versions, hardware, git commit, and
  dirty-worktree state;
- per-epoch training and validation metrics;
- validation-selected checkpoints and thresholds;
- per-example labels, probabilities, predictions, angles, and task metadata;
- accuracy, balanced accuracy, AUROC, Brier score, log loss, confusion counts,
  runtime, parameter counts, and task-specific controller/trajectory statistics;
- a 95% Wilson interval over test items inside each run;
- mean, sample standard deviation, and a 95% Student-t interval over independent
  training seeds in the final aggregate.

SAT-v2 additionally reports accuracy by question type. Failed and missing stages
remain in `status.json` and `audit.json`; the aggregator never silently drops them.

The generated audit artifacts are:

- `results/mps_paper_suite/REPORT.md`: human-readable table;
- `results/mps_paper_suite/metrics.csv`: paper/table analysis input;
- `results/mps_paper_suite/audit.json`: complete machine-readable aggregation,
  warnings, source files, and stage status;
- `results/mps_paper_suite/status.json`: commands, return codes, durations, and
  expected outputs for every stage;
- `models/runs/mps_paper_suite/`: checkpoints, logs, predictions, and raw summaries.

## Commands

Validate the complete pipeline with tiny, explicitly preliminary runs:

```bash
scripts/run_paper_mps_suite.sh --profile smoke --seeds 0 --keep-going
```

Run the paper-oriented overnight matrix:

```bash
scripts/run_paper_mps_suite.sh --profile overnight --seeds 0 1 2 --keep-going
```

Run the additional partial-DINO experiments after the main matrix completes:

```bash
scripts/run_paper_mps_suite.sh --profile extended --seeds 0 1 2 --keep-going
```

The same command is safe to rerun: completed stages are skipped. Use `--force`
only to intentionally replace completed runs. Use `--categories dino sat`, for
example, to resume selected independent categories.

## Interpretation constraints

- No experiment suite can guarantee NeurIPS acceptance. This matrix targets the
  central reviewer risks: unfair baselines, missing external pretraining,
  single-seed results, missing uncertainty, absent component controls, and weak
  provenance.
- The Ganis-Kievit test contains only 78 pairs and the legacy preprocessing has
  object-identity overlap. It is a small transfer diagnostic, not unseen-object
  OOD evidence.
- ID and held-out-angle manifests contain different rendered scenes. Compare
  models within each split; do not infer an OOD benefit from OOD accuracy being
  numerically higher than ID accuracy.
- SAT-v2 uses a finite streamed subset in the overnight profile. Report this
  exact budget and do not compare its values directly with full-data published
  results.
- The program-of-tools row is a handcrafted CV/BFS baseline, not ViperGPT unless
  a language model actually synthesizes new programs for that run.

# External spatial-reasoning benchmark protocol

## Selection

The immediate strict zero-shot benchmark is **BLINK Multi-view Reasoning**. It
contains pairs of real video frames and asks whether the camera moved clockwise
(left/A) or counter-clockwise (right/B). The fixed two-answer semantics let us
evaluate the existing frozen, action-conditioned rotation flows without adding
or fitting a language head.

This is a stronger transfer test than another synthetic 2-D rotation set: the
images contain real objects, backgrounds, perspective, occlusion, and parallax.
It is also deliberately difficult for the present model because camera orbit is
a 3-D transformation, not an in-plane rotation.

## Frozen-transfer protocol

1. Load the pinned BLINK validation split and the already-trained Tetris and
   colored-rotation flow checkpoints for seeds 0, 1, and 2.
2. Freeze every checkpoint. Do not train, calibrate, choose a threshold, or
   select a checkpoint using BLINK. The development-time sign convention is
   disclosed below.
3. Center-crop each frame, resize to 64 by 64, and use either fixed luminance
   (Tetris flow) or RGB (colored flow).
4. Render signed rotation hypotheses from -170 to -10 degrees and +10 to +170
   degrees in ten-degree increments. The flow action describes object motion,
   while BLINK asks for camera motion, so the standard inverse-view convention
   maps a positive/counter-clockwise apparent object rotation to the
   clockwise/left camera hypothesis, and a negative/clockwise object rotation to
   the counter-clockwise/right camera hypothesis.
5. Predict the direction whose best hypothesis has lower pixel reconstruction
   error. The decision boundary is exactly zero.
6. Report accuracy with a Wilson 95% interval, balanced accuracy, ROC AUC,
   left/right recall, confusion counts, all per-item margins, seed variation,
   and a mean-margin seed ensemble.
7. Run the same scan with an exact image-plane rotation operator. This is a
   hypothesis-class control, not a trainable baseline: it separates limitations
   of the learned flow from limitations of treating camera motion as planar
   image rotation.

The evaluation writes numerical predictions and provenance to
`results/neurreps_flow_v1/blink_multiview_zero_shot`. Audit trajectory grids are
saved locally but ignored by Git because BLINK aggregates third-party images.

Run the full evaluation on Apple Silicon with:

```bash
./.venv/bin/python scripts/evaluate_neurreps_flow_blink_multiview.py --device mps
```

The command is resumable by default. Add `--rerun` only to deliberately replace
compatible cached per-seed results.

Development disclosure: the inverse camera/object sign convention was finalized
after a two-item evaluator smoke. No checkpoint, threshold, or reconstruction
metric was fit on BLINK, but the reported public-validation result is exploratory
and must be confirmed on the hidden test or another locked split before it is a
headline generalization claim.

## Completed result

In the exploratory run on all 133 labeled validation pairs, the colored flows
average 74.7% accuracy over three seeds (individual runs 72.2–75.9%) and 0.809
AUC. The Tetris flows average 71.7% accuracy and 0.793 AUC. Mean-margin
ensembles obtain 72.2% and 71.4% accuracy, respectively. All are clearly above
the 50% random baseline.

The exact in-plane rotation control reaches 70.7% accuracy and 0.763 AUC. No
individual flow or seed ensemble significantly beats that control in a paired
exact McNemar test (all p values are at least 0.167). Most selected hypotheses
are at the smallest tested magnitude, 10 degrees. This supports transfer of a
signed visual-transformation prior, but it does **not** yet prove that the learned
flow provides general 3-D reasoning beyond the hand-specified planar hypothesis
class.

For context, the official BLINK paper reports 92.48% human, 58.65% GPT-4V, and
41.35% Gemini Pro accuracy on the validation task. Its pretrained LoFTR
specialist reaches 90.22% on the paper's separate dev/test table. The split and
interface differences mean these are contextual references rather than paired
comparisons.

## Why not use SAT-v2 as the strict frozen test?

SAT-v2 is highly relevant contextual evidence and the repository already has
completed supervised direct and FoT results. However, SAT questions combine
language, multi-image layouts, task-specific answer choices, dynamic 3-D scenes,
and several distinct action semantics. Applying a frozen image rotation flow
requires a new language/action interface; fitting that interface on SAT would no
longer be the same strict zero-target-training protocol as the Ganis-Kievit 3-D
test.

## Other candidates reviewed

- **Spatial Reasoning with Denoising Models (SRM)** is the closest methodological
  comparison to flow reasoning. Its Sudoku and polygon-counting models are
  trained on each target task, so it belongs in a future task-trained reasoning
  experiment rather than this frozen-transfer claim.
- **3DSRBench** covers viewpoint and 3-D relational VQA, but it requires a
  benchmark-specific visual-language answer interface.
- Other BLINK categories are useful for multimodal foundation models, but do not
  expose a fixed signed transformation that the current flows can score without
  target-specific machinery.

Primary sources: [BLINK paper](https://arxiv.org/abs/2404.12390),
[BLINK repository](https://github.com/zeyofu/BLINK_Benchmark),
[SAT paper](https://arxiv.org/abs/2412.07755),
[SRM paper](https://arxiv.org/abs/2502.21075),
[SRM repository](https://github.com/Chrixtar/SRM), and
[3DSRBench](https://3dsrbench.github.io/).

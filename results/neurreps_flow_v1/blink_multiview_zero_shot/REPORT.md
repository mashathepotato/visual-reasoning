# BLINK Multi-view frozen-flow transfer

No BLINK example was used to train, calibrate, threshold, or select a checkpoint. The inverse camera/object sign convention was finalized after a two-item evaluator smoke, so this public-validation result is exploratory and requires confirmation on the hidden test or another locked split.

## Frozen-flow results

| 2-D source | Metric | Mean over seeds | 95% t CI |
|---|---|---:|---:|
| colored | accuracy | 0.747 | [0.693, 0.801] |
| colored | balanced_accuracy | 0.742 | [0.664, 0.820] |
| colored | auc | 0.809 | [0.797, 0.822] |
| colored | left_recall | 0.695 | [0.403, 0.987] |
| colored | right_recall | 0.788 | [0.653, 0.924] |
| tetris | accuracy | 0.717 | [0.670, 0.764] |
| tetris | balanced_accuracy | 0.704 | [0.624, 0.783] |
| tetris | auc | 0.793 | [0.781, 0.805] |
| tetris | left_recall | 0.588 | [0.210, 0.965] |
| tetris | right_recall | 0.820 | [0.596, 1.000] |

## Seed ensembles and controls

| Method | Accuracy (Wilson 95% CI) | Balanced accuracy | AUC | Exact p vs 50% |
|---|---:|---:|---:|---:|
| colored seed ensemble | 0.722 [0.640, 0.791] | 0.714 | 0.814 | 1.6e-07 |
| tetris seed ensemble | 0.714 [0.632, 0.784] | 0.700 | 0.800 | 4.12e-07 |
| exact in-plane rotation control | 0.707 [0.624, 0.777] | 0.709 | 0.763 | 1.03e-06 |
| random | 0.500 | 0.500 | 0.500 | 1.000 |

Published validation context from the official BLINK paper: human 92.48%, GPT-4V direct 58.65%, GPT-4V with concatenated images 57.89%, and Gemini Pro direct 41.35%. The paper also reports 90.22% for its pretrained LoFTR specialist on its dev/test table (a different split). These are contextual reference values, not confidence-interval-matched comparisons.

## Paired comparison with exact planar geometry

| Flow | Seed | Accuracy delta | Flow wins | Flow losses | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| tetris | 0 | +0.030 | 12 | 8 | 0.503 |
| tetris | 1 | -0.008 | 18 | 19 | 1 |
| tetris | 2 | +0.008 | 19 | 18 | 1 |
| colored | 0 | +0.053 | 13 | 6 | 0.167 |
| colored | 1 | +0.015 | 17 | 15 | 0.86 |
| colored | 2 | +0.053 | 13 | 6 | 0.167 |
| colored | ensemble | +0.015 | 13 | 11 | 0.839 |
| tetris | ensemble | +0.008 | 16 | 15 | 1 |

None of the frozen-flow versus exact-control accuracy differences reaches p < 0.05. The result therefore supports signed synthetic-to-real transfer, but does not yet establish that the learned flow contributes accuracy beyond the predeclared planar-rotation hypothesis class.

## Selected-angle diagnostics

| Method | Mean absolute angle | Median | Fraction at smallest grid angle |
|---|---:|---:|---:|
| tetris seed 0 | 21.7° | 10.0° | 0.842 |
| tetris seed 1 | 20.9° | 10.0° | 0.835 |
| tetris seed 2 | 21.3° | 10.0° | 0.835 |
| colored seed 0 | 21.4° | 10.0° | 0.835 |
| colored seed 1 | 20.6° | 10.0° | 0.812 |
| colored seed 2 | 21.2° | 10.0° | 0.835 |
| exact in-plane control | 48.8° | 10.0° | 0.549 |

## What this tests

A result above chance would support transfer of a signed visual-transformation prior from synthetic 2-D rotations to real camera motion. Failure of both the learned flows and the exact control means the in-plane rotation hypothesis class is insufficient for 3-D viewpoint change; failure of only the learned flows localizes the problem to flow learning or rendering quality.

The audit PNGs are intentionally kept local because BLINK aggregates images from external sources. Numerical predictions contain IDs and scores only.

Sources: [BLINK paper](https://arxiv.org/abs/2404.12390), [official repository](https://github.com/zeyofu/BLINK_Benchmark), [dataset](https://huggingface.co/datasets/BLINK-Benchmark/BLINK).

# Frozen-flow zero-shot Ganis-Kievit audit

All 2-D flow checkpoints were frozen before this evaluation. No 3-D labels were used for training, calibration, checkpoint selection, or threshold selection.

| 2-D source | Protocol | Metric | Mean over seeds | 95% t CI |
|---|---|---|---:|---:|
| colored | provided_angle | accuracy | 0.615 | [0.505, 0.726] |
| colored | provided_angle | balanced_accuracy | 0.615 | [0.505, 0.726] |
| colored | provided_angle | auc | 0.647 | [0.625, 0.668] |
| colored | provided_angle | positive_recall | 0.675 | [0.578, 0.773] |
| colored | provided_angle | negative_recall | 0.556 | [0.423, 0.688] |
| colored | angle_marginalized | accuracy | 0.581 | [0.469, 0.693] |
| colored | angle_marginalized | balanced_accuracy | 0.581 | [0.469, 0.693] |
| colored | angle_marginalized | auc | 0.662 | [0.627, 0.696] |
| colored | angle_marginalized | positive_recall | 0.521 | [0.448, 0.595] |
| colored | angle_marginalized | negative_recall | 0.641 | [0.472, 0.810] |
| tetris | provided_angle | accuracy | 0.585 | [0.537, 0.634] |
| tetris | provided_angle | balanced_accuracy | 0.585 | [0.537, 0.634] |
| tetris | provided_angle | auc | 0.639 | [0.615, 0.663] |
| tetris | provided_angle | positive_recall | 0.667 | [0.603, 0.730] |
| tetris | provided_angle | negative_recall | 0.504 | [0.431, 0.578] |
| tetris | angle_marginalized | accuracy | 0.564 | [0.509, 0.619] |
| tetris | angle_marginalized | balanced_accuracy | 0.564 | [0.509, 0.619] |
| tetris | angle_marginalized | auc | 0.645 | [0.582, 0.707] |
| tetris | angle_marginalized | positive_recall | 0.530 | [0.493, 0.567] |
| tetris | angle_marginalized | negative_recall | 0.598 | [0.466, 0.731] |

## Seed ensembles

| 2-D source | Protocol | Accuracy (Wilson 95% CI) | AUC |
|---|---|---:|---:|
| colored | provided_angle | 0.590 [0.479, 0.692] | 0.657 |
| colored | angle_marginalized | 0.577 [0.466, 0.680] | 0.666 |
| tetris | provided_angle | 0.564 [0.454, 0.669] | 0.648 |
| tetris | angle_marginalized | 0.590 [0.479, 0.692] | 0.650 |

## Interpretation constraints

This is a small zero-shot domain-transfer diagnostic, not a validated unseen-object benchmark. A horizontal image reflection is not a physical 3-D mirror transformation, and the stimulus set provides no ground-truth intermediate rotations. The audit images therefore test whether the frozen flow produces coherent continuous states, while classification tests only whether its reconstruction margin separates same from mirrored pairs.

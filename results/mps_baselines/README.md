# Matched colored-rotation baselines on Apple MPS

Run completed: 2026-08-02

These are full, validation-selected results for the task-trained CNN and
from-scratch ViT baselines. Each method used the same committed 5,000-example
training manifest, 1,000-example validation split, 1,000-example ID test split,
and 1,000-example held-out-angle test split. Models trained for 50 epochs with
seeds 0, 1, and 2. Checkpoints were selected by validation accuracy only.

## Aggregate results

Values are mean ± standard deviation across three training seeds.

| Model | Parameters | ID accuracy | ID AUC | Held-out-angle accuracy | Held-out-angle AUC |
|---|---:|---:|---:|---:|---:|
| CNN | 288,578 | 65.93 ± 0.78% | 0.7231 ± 0.0070 | 67.67 ± 0.15% | 0.7476 ± 0.0041 |
| ViT | 2,756,546 | 56.40 ± 2.01% | 0.5969 ± 0.0277 | 58.97 ± 1.46% | 0.6256 ± 0.0273 |

The smaller CNN is the stronger direct classifier under this protocol: it leads
the ViT by 9.53 percentage points on the ID split and 8.70 points on the
held-out-angle split. Both models are above chance, but neither result yet answers
whether explicit visual transition traces help because FoT has not been evaluated
on these exact manifests.

## Runtime

All six runs completed successfully on Apple MPS in 4,312.6 seconds (71.9
minutes) including orchestration and evaluation. The three CNN trainings used
694.9 seconds in total; the three ViT trainings used 3,604.4 seconds in total.

## Important OOD caveat

The ID and held-out-angle splits currently use different procedural scenes and
random, uneven angle frequencies. The slightly higher held-out-angle scores must
not be interpreted as improved OOD generalization. A decisive angle study should
pair the same base scenes across angle sets, balance every angle and label, report
macro-averages by angle, and bootstrap by base scene.

## Files

- `cnn_seeds0-1-2.json`: complete aggregate CNN metrics.
- `vit_seeds0-1-2.json`: complete aggregate ViT metrics.
- `overnight_status.json`: commands, environment, dependency versions, per-run
  durations, completion state, and artifact locations.

Large checkpoints, predictions, resolved configurations, and console logs remain
under ignored `models/runs/mps_baselines/`.

# NeurReps trajectory-flow results

Profile: `overnight`. PPO is not used.

| Task | Metric | Seeds | Mean | 95% CI |
|---|---|---:|---:|---:|
| colored | cycle_mse | 3 | 0.034053 | [0.030010, 0.038096] |
| colored | endpoint_mse | 3 | 0.023030 | [0.020668, 0.025391] |
| colored | endpoint_psnr_db | 3 | 18.345910 | [16.970819, 19.721002] |
| colored | silhouette_iou | 3 | 0.670828 | [0.642783, 0.698873] |
| colored | trajectory_mse | 3 | 0.012506 | [0.011286, 0.013726] |
| maze | endpoint_iou | 3 | 0.974833 | [0.946173, 1.003492] |
| maze | endpoint_mse | 3 | 0.006393 | [0.000469, 0.012317] |
| maze | goal_reached | 3 | 1.000000 | [1.000000, 1.000000] |
| maze | obstacle_violation_rate | 3 | 0.000000 | [0.000000, 0.000000] |
| maze | path_precision | 3 | 0.986767 | [0.971955, 1.001580] |
| maze | path_recall | 3 | 0.987661 | [0.973161, 1.002161] |
| maze | trajectory_mse | 3 | 0.009424 | [0.007779, 0.011068] |
| tetris | cycle_mse | 3 | 0.023978 | [0.021438, 0.026518] |
| tetris | endpoint_mse | 3 | 0.014038 | [0.012717, 0.015359] |
| tetris | endpoint_psnr_db | 3 | 19.880948 | [19.217728, 20.544167] |
| tetris | silhouette_iou | 3 | 0.791705 | [0.791093, 0.792317] |
| tetris | trajectory_mse | 3 | 0.006954 | [0.006306, 0.007602] |

Each run directory contains `audit_best.png`, `quality_metrics.json`, the complete epoch history, resolved configuration, provenance, and the best checkpoint.

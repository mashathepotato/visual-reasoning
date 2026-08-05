# NeurReps trajectory-flow results

Profile: `smoke`. PPO is not used.

| Task | Metric | Seeds | Mean | 95% CI |
|---|---|---:|---:|---:|
| colored | cycle_mse | 1 | 0.004457 | undefined (n<2) |
| colored | endpoint_mse | 1 | 0.112883 | undefined (n<2) |
| colored | endpoint_psnr_db | 1 | 10.196983 | undefined (n<2) |
| colored | silhouette_iou | 1 | 0.260116 | undefined (n<2) |
| colored | trajectory_mse | 1 | 0.078770 | undefined (n<2) |
| maze | endpoint_iou | 1 | 0.011353 | undefined (n<2) |
| maze | endpoint_mse | 1 | 0.202676 | undefined (n<2) |
| maze | goal_reached | 1 | 0.000000 | undefined (n<2) |
| maze | obstacle_violation_rate | 1 | 0.000000 | undefined (n<2) |
| maze | path_precision | 1 | 1.000000 | undefined (n<2) |
| maze | path_recall | 1 | 0.011353 | undefined (n<2) |
| maze | trajectory_mse | 1 | 0.102218 | undefined (n<2) |
| tetris | cycle_mse | 1 | 0.004347 | undefined (n<2) |
| tetris | endpoint_mse | 1 | 0.058996 | undefined (n<2) |
| tetris | endpoint_psnr_db | 1 | 12.599340 | undefined (n<2) |
| tetris | silhouette_iou | 1 | 0.482710 | undefined (n<2) |
| tetris | trajectory_mse | 1 | 0.040062 | undefined (n<2) |

Each run directory contains `audit_best.png`, `quality_metrics.json`, the complete epoch history, resolved configuration, provenance, and the best checkpoint.

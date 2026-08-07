# Maze mixture-of-experts results

Profile: `overnight`. Frozen seed-matched Tetris and colored rotation flows provide spatial features; PPO is not used.

The MoE has 898,643 trainable and 447,364 frozen parameters. The scratch maze reference has 887,905 trainable parameters, so the trainable budgets differ by 1.2%; the MoE additionally benefits from its frozen pretrained feature bank.

| Model / routing | Metric | Seeds | Mean | 95% t CI |
|---|---|---:|---:|---:|
| scratch maze reference | endpoint_iou | 3 | 0.974833 | [0.946173, 1.000000] |
| MoE / colored_only | endpoint_iou | 3 | 0.560773 | [0.223655, 0.897890] |
| MoE / learned | endpoint_iou | 3 | 0.978597 | [0.965343, 0.991851] |
| MoE / tetris_only | endpoint_iou | 3 | 0.574117 | [0.000000, 1.000000] |
| MoE / uniform | endpoint_iou | 3 | 0.977119 | [0.955323, 0.998915] |
| scratch maze reference | trajectory_mse | 3 | 0.009424 | [0.007779, 0.011068] |
| MoE / colored_only | trajectory_mse | 3 | 0.047764 | [0.000000, 0.116720] |
| MoE / learned | trajectory_mse | 3 | 0.009437 | [0.007561, 0.011312] |
| MoE / tetris_only | trajectory_mse | 3 | 0.054993 | [0.000000, 0.145569] |
| MoE / uniform | trajectory_mse | 3 | 0.009650 | [0.006607, 0.012693] |
| scratch maze reference | intermediate_prefix_iou | 3 | 0.842305 | [0.822026, 0.862584] |
| MoE / colored_only | intermediate_prefix_iou | 3 | 0.559118 | [0.000000, 1.000000] |
| MoE / learned | intermediate_prefix_iou | 3 | 0.837696 | [0.830253, 0.845138] |
| MoE / tetris_only | intermediate_prefix_iou | 3 | 0.436419 | [0.000000, 1.000000] |
| MoE / uniform | intermediate_prefix_iou | 3 | 0.835916 | [0.818774, 0.853059] |
| scratch maze reference | premature_activation_rate | 3 | 0.316510 | [0.251054, 0.381966] |
| MoE / colored_only | premature_activation_rate | 3 | 0.138931 | [0.000000, 0.412384] |
| MoE / learned | premature_activation_rate | 3 | 0.342880 | [0.247395, 0.438365] |
| MoE / tetris_only | premature_activation_rate | 3 | 0.174882 | [0.075023, 0.274741] |
| MoE / uniform | premature_activation_rate | 3 | 0.334910 | [0.232354, 0.437467] |
| scratch maze reference | future_path_mean_intensity | 3 | 0.164398 | [0.146227, 0.182570] |
| MoE / colored_only | future_path_mean_intensity | 3 | 0.131039 | [0.013806, 0.248272] |
| MoE / learned | future_path_mean_intensity | 3 | 0.189408 | [0.137460, 0.241355] |
| MoE / tetris_only | future_path_mean_intensity | 3 | 0.147848 | [0.093048, 0.202648] |
| MoE / uniform | future_path_mean_intensity | 3 | 0.185035 | [0.126555, 0.243515] |
| scratch maze reference | obstacle_violation_rate | 3 | 0.000000 | [0.000000, 0.000000] |
| MoE / colored_only | obstacle_violation_rate | 3 | 0.020531 | [0.000000, 0.107761] |
| MoE / learned | obstacle_violation_rate | 3 | 0.000000 | [0.000000, 0.000000] |
| MoE / tetris_only | obstacle_violation_rate | 3 | 0.000011 | [0.000000, 0.000035] |
| MoE / uniform | obstacle_violation_rate | 3 | 0.000000 | [0.000000, 0.000000] |
| MoE / colored_only | router_tetris_weight | 3 | 0.000000 | [0.000000, 0.000000] |
| MoE / learned | router_tetris_weight | 3 | 0.528465 | [0.369333, 0.687597] |
| MoE / tetris_only | router_tetris_weight | 3 | 1.000000 | [1.000000, 1.000000] |
| MoE / uniform | router_tetris_weight | 3 | 0.500000 | [0.500000, 0.500000] |
| MoE / colored_only | router_tetris_weight_on_path | 3 | 0.000000 | [0.000000, 0.000000] |
| MoE / learned | router_tetris_weight_on_path | 3 | 0.535032 | [0.385091, 0.684974] |
| MoE / tetris_only | router_tetris_weight_on_path | 3 | 1.000000 | [1.000000, 1.000000] |
| MoE / uniform | router_tetris_weight_on_path | 3 | 0.500000 | [0.500000, 0.500000] |
| MoE / colored_only | router_entropy | 3 | 0.000000 | [0.000000, 0.000000] |
| MoE / learned | router_entropy | 3 | 0.682808 | [0.653092, 0.693147] |
| MoE / tetris_only | router_entropy | 3 | 0.000000 | [0.000000, 0.000000] |
| MoE / uniform | router_entropy | 3 | 0.693147 | [0.693147, 0.693147] |

Display intervals are clipped to each metric's natural range; raw Student-t intervals are retained in JSON and CSV.

## Paired primary comparisons

Every delta is learned minus comparator on the same seed. Positive is better for IoU and negative is better for MSE or premature activation.

| Comparator | Metric | Mean delta | 95% t CI | Seedwise deltas |
|---|---|---:|---:|---:|
| uniform | endpoint_iou | +0.001478 | [-0.007064, 0.010020] | +0.005442, -0.000304, -0.000703 |
| uniform | trajectory_mse | -0.000214 | [-0.001428, 0.001000] | -0.000775, +0.000113, +0.000022 |
| uniform | intermediate_prefix_iou | +0.001779 | [-0.008223, 0.011782] | +0.006108, -0.001854, +0.001084 |
| uniform | premature_activation_rate | +0.007970 | [-0.013704, 0.029644] | +0.006012, +0.017507, +0.000391 |
| scratch | endpoint_iou | +0.003765 | [-0.036659, 0.044188] | -0.013127, +0.005086, +0.019335 |
| scratch | trajectory_mse | +0.000013 | [-0.003366, 0.003391] | +0.001557, -0.001004, -0.000515 |
| scratch | intermediate_prefix_iou | -0.004609 | [-0.026330, 0.017111] | -0.013685, -0.003901, +0.003758 |
| scratch | premature_activation_rate | +0.026370 | [-0.047926, 0.100665] | -0.006292, +0.052408, +0.032994 |

## Interpretation

The learned mixture matches the scratch maze flow within three-seed uncertainty, but learned routing does not consistently improve over a forced 50/50 gate. Router entropy is close to its two-expert maximum and audit maps are low-contrast, so the mechanism behaves like dense feature fusion rather than adaptive expert specialization.

Because the trainable decoder also receives the raw maze condition, parity with scratch does not establish that rotation pretraining adds useful information. Forcing either single gate after training substantially degrades the aggregate endpoint result, but those are interventions on a jointly trained decoder—not separately trained controls. They establish sensitivity to the learned mixture regime, not transfer benefit or unique causal value for either expert.

Primary checkpoints are selected only by endpoint IoU. A descriptive post-hoc history check found that all three seeds reached their cleanest near-best process point at epoch 24: endpoint IoU 0.970793, prefix IoU 0.832962, trajectory MSE 0.010457, and premature activation 0.276599. These states are not used as primary results and were not retained as final checkpoints; they motivate preregistering a process-aware selection rule in the next run.

The learned-router row is the trained model. Uniform, Tetris-only, and colored-only rows force the gate after training and therefore measure reliance on each expert; they are not independently trained baselines.

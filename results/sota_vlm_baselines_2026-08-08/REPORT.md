# Cross-dataset spatial baseline comparison

The exact-rotation operator is retained only as a diagnostic and is not the target comparator. The headline compares learned FoT/trajectory-flow systems with direct frontier VLMs and published spatial systems. Values are accuracy percentages unless stated otherwise; bracketed intervals are 95% confidence intervals.

| Dataset | Our method | GPT-5.6 Sol (direct, high) | Prior direct VLM | Strong specialist / SOTA |
|---|---:|---:|---:|---:|
| Tetris unseen angles | 56.2 [49.9, 62.4] | 72.0 [62.5, 79.9] | GPT-4o-mini 52.4† | No published matched SOTA |
| Colored unseen angles | 53.3 [44.3, 62.3] | 98.0 [93.0, 99.4] | Claude Opus 4.6 51.6† | No published matched SOTA |
| Ganis-Kievit 3-D | 61.5 [50.5, 72.6] | — | Claude Opus 4.6 53.8 | Program-of-tools 65.4 |
| SAT-Real | 54.4 [51.9, 57.0] | — | GPT-4V 50.7‡ | SpatialDreamer 93.9‡ |
| BLINK Multi-view | 74.7 [69.3, 80.1] | — | GPT-4V 58.7 | P2+Gemini 63.9 / LoFTR 90.2‡ |

## Maze and intermediate-process metrics

The complete GPT-5.6 maze run was not obtained before provider credit was exhausted. The deterministic program-of-tools baseline is 100% on both trace validity and path solving. These are not the same interface as the trajectory flow, which reaches 100% goal activation, 97.5% endpoint IoU, and 84.2% intermediate-prefix IoU over three seeds.

## Reading the table

- † The historical direct VLM used the same procedural distribution but a different random item set from the committed unseen-angle manifests.
- ‡ Published SAT values use circular answer-order evaluation. The GPT-5.6 circular pass could not run after provider credit was exhausted; published circular results remain protocol-labelled references. LoFTR's 90.2% is from the BLINK paper's separate dev/test table.
- The rebuilt trajectory flow currently has no same/different head on the 2-D rotation manifests. The Tetris/colored accuracy rows therefore use the legacy flow+PPO classifier; current flow quality is audited separately (single-source silhouette IoU: 87.4% Tetris, 82.5% colored).
- OpenAI credit was exhausted after 279/711 GPT-5.6 responses. Only tasks marked complete are used in headline comparisons; partial and missing rows remain in the complete audit table for coverage accounting.
- Claude Fable 5 was selected and probed, but the Anthropic account returned `credit balance is too low`; no value is reported. Cached Claude Opus 4.6 results remain included.

## Main findings

- On Tetris unseen angles, the legacy FoT classifier is 15.8 percentage points below GPT-5.6 Sol; on colored unseen angles it is 44.7 percentage points below GPT-5.6 Sol. It narrowly exceeds the old cached direct VLMs but does not beat the frontier model. These rows do not yet test the rebuilt auditable flow.
- On Ganis-Kievit 3-D, the frozen colored-source trajectory flow reaches 61.5%: 7.7 points above cached Claude Opus 4.6 and 11.5 above cached GPT-4o-mini, but 3.8 below the program-of-tools baseline. GPT-5.6 reached 57.3% on 75/78 cached items; that incomplete result is excluded from the headline.
- On SAT-Real, FoT reaches 54.4%. This is 3.7 points above published GPT-4V, but 19.6 below GPT-4.1 and 39.5 below SpatialDreamer. Those published values use circular answer-order evaluation; our GPT-5.6 circular result is not completed (provider credit exhausted).
- BLINK Multi-view is the strongest result for the paper's generalization claim: the frozen colored-source flow reaches 74.7% on all 133 validation items, 16.0 points above GPT-4V and 10.8 above P2 + Gemini 2.5 Pro. It remains 15.5 points below the LoFTR specialist and 17.8 below humans.
- Maze endpoint and intermediate-state metrics support process auditability, but they are not accuracy-equivalent to direct path-text generation. They are kept separate rather than averaged into an artificial cross-task score.

## Complete audit table

This table contains every collected learned, tool, VLM, specialist, and human result used in the audit. `Comparability` identifies rows whose item set, answer-order protocol, split, or interface differs.

| Dataset | Method | Category | Metric | Value (95% CI) | n | CI method | Comparability | Protocol |
|---|---|---|---|---:|---:|---|---|---|
| Tetris rotation | FoT flow+PPO (legacy) | ours | accuracy | 56.2 [49.9, 62.4] | 1000/seed × 3 seeds | student_t_over_independent_seeds | matched | test_ood_angle; current GPT fixed balanced n=100 subset |
| Tetris rotation | CNN trained on task | trained local | accuracy | 87.0 [77.7, 96.3] | 1000/seed × 3 seeds | student_t_over_independent_seeds | matched | test_ood_angle; current GPT fixed balanced n=100 subset |
| Tetris rotation | ViT trained on task | trained local | accuracy | 56.8 [26.7, 86.9] | 1000/seed × 3 seeds | student_t_over_independent_seeds | matched | test_ood_angle; current GPT fixed balanced n=100 subset |
| Tetris rotation | DINOv3 frozen + trained head | trained local | accuracy | 61.2 [56.0, 66.4] | 1000/seed × 3 seeds | student_t_over_independent_seeds | matched | test_ood_angle; current GPT fixed balanced n=100 subset |
| Tetris rotation | GPT-5.6 Sol (high) | direct frontier VLM | accuracy | 72.0 [62.5, 79.9] | 100 | wilson_over_items | matched | test_ood_angle; current GPT fixed balanced n=100 subset |
| Tetris rotation | gpt-4o-mini | direct historical VLM | accuracy | 52.4 [48.0, 56.7] | 500 | wilson_over_items | same task distribution, different item set | procedural random sample, seed 0 |
| Tetris rotation | claude-opus-4-6 | direct historical VLM | accuracy | 46.2 [41.9, 50.6] | 500 | wilson_over_items | same task distribution, different item set | procedural random sample, seed 0 |
| Colored rotation | FoT flow+PPO (legacy) | ours | accuracy | 53.3 [44.3, 62.3] | 1000/seed × 3 seeds | student_t_over_independent_seeds | matched | test_ood_angle; current GPT fixed balanced n=100 subset |
| Colored rotation | CNN trained on task | trained local | accuracy | 67.7 [67.3, 68.0] | 1000/seed × 3 seeds | student_t_over_independent_seeds | matched | test_ood_angle; current GPT fixed balanced n=100 subset |
| Colored rotation | ViT trained on task | trained local | accuracy | 59.0 [55.3, 62.6] | 1000/seed × 3 seeds | student_t_over_independent_seeds | matched | test_ood_angle; current GPT fixed balanced n=100 subset |
| Colored rotation | DINOv3 frozen + trained head | trained local | accuracy | 50.8 [49.2, 52.4] | 1000/seed × 3 seeds | student_t_over_independent_seeds | matched | test_ood_angle; current GPT fixed balanced n=100 subset |
| Colored rotation | GPT-5.6 Sol (high) | direct frontier VLM | accuracy | 98.0 [93.0, 99.4] | 100 | wilson_over_items | matched | test_ood_angle; current GPT fixed balanced n=100 subset |
| Colored rotation | gpt-4o-mini | direct historical VLM | accuracy | 50.0 [45.6, 54.4] | 500 | wilson_over_items | same task distribution, different item set | procedural random sample, seed 0 |
| Colored rotation | claude-opus-4-6 | direct historical VLM | accuracy | 51.6 [47.2, 56.0] | 500 | wilson_over_items | same task distribution, different item set | procedural random sample, seed 0 |
| Ganis-Kievit 3-D | Frozen trajectory flow (colored source) | ours | accuracy | 61.5 [50.5, 72.6] | 78/seed × 3 seeds | student_t_over_independent_seeds | matched | full n=78 |
| Ganis-Kievit 3-D | DINOv3 frozen + trained head | trained local | accuracy | 50.4 [48.6, 52.3] | 78/seed × 3 seeds | student_t_over_independent_seeds | matched | full n=78 |
| Ganis-Kievit 3-D | ViperGPT program-of-tools | tool baseline | accuracy | 65.4 [65.4, 65.4] | 78/seed × 3 seeds | student_t_over_independent_seeds | matched | full n=78 |
| Ganis-Kievit 3-D | GPT-5.6 Sol (high; partial) | direct frontier VLM | accuracy | 57.3 [46.1, 67.9] | 75/78 cached | wilson_over_items | incomplete due provider credit exhaustion; excluded from headline claims | full n=78 |
| Ganis-Kievit 3-D | gpt-4o-mini | direct historical VLM | accuracy | 50.0 [39.2, 60.8] | 78 | wilson_over_items | matched | full n=78; shuffled order |
| Ganis-Kievit 3-D | claude-opus-4-6 | direct historical VLM | accuracy | 53.8 [42.9, 64.5] | 78 | wilson_over_items | matched | full n=78; shuffled order |
| Maze generation | Trajectory flow (no PPO) | ours | goal_reached | 100.0 [100.0, 100.0] | 3 seeds | student_t_over_independent_seeds | matched | held-out procedural validation, 3 seeds |
| Maze generation | Trajectory flow (no PPO) | ours | endpoint_iou | 97.5 [94.6, 100.0] | 3 seeds | student_t_over_independent_seeds | matched | held-out procedural validation, 3 seeds |
| Maze generation | Trajectory flow (no PPO) | ours | intermediate_prefix_iou | 84.2 [82.2, 86.3] | 3 seeds | student_t_over_independent_seeds | matched | held-out procedural validation, 3 seeds |
| Maze trace validity | GPT-5.6 Sol (high; partial) | direct frontier VLM | accuracy | 75.0 [30.1, 95.4] | 4/100 cached | wilson_over_items | incomplete due provider credit exhaustion; excluded from headline claims | fixed current subset n=100 |
| Maze trace validity | gpt-4o-mini | direct historical VLM | accuracy | 50.6 [46.2, 55.0] | 500 | wilson_over_items | same task distribution, different item set | procedural random sample, seed 0 |
| Maze trace validity | claude-opus-4-6 | direct historical VLM | accuracy | 50.0 [45.6, 54.4] | 500 | wilson_over_items | same task distribution, different item set | procedural random sample, seed 0 |
| Maze trace validity | ViperGPT program-of-tools | tool baseline | accuracy | 100.0 [100.0, 100.0] |  | student_t_over_independent_seeds | matched | procedural random samples across 3 seeds |
| Maze path solving | GPT-5.6 Sol (high; partial) | direct frontier VLM | success_rate | — | 0/50 cached |  | incomplete due provider credit exhaustion; excluded from headline claims | fixed current subset n=50 |
| Maze path solving | gpt-4o-mini | direct historical VLM | success_rate | 0.0 [0.0, 0.8] | 500 | wilson_over_items | same task distribution, different item set | procedural random sample, seed 0 |
| Maze path solving | claude-opus-4-6 | direct historical VLM | success_rate | 0.0 [0.0, 0.8] | 500 | wilson_over_items | same task distribution, different item set | procedural random sample, seed 0 |
| Maze path solving | ViperGPT program-of-tools | tool baseline | success_rate | 100.0 [100.0, 100.0] |  | student_t_over_independent_seeds | matched | procedural random samples across 3 seeds |
| Maze trace validity | FoT trace controller (legacy) | ours | accuracy | 49.2 [43.6, 54.8] | 200/seed × 3 seeds | student_t_over_independent_seeds | matched | procedural random samples across 3 seeds |
| SAT-v2 / SAT-Real | FoT heatmap MCQ | ours | accuracy | 54.4 [51.9, 57.0] | 150/seed × 3 seeds | student_t_over_independent_seeds | matched | first/all 150 test items; single answer ordering |
| SAT-v2 / SAT-Real | Matched direct MCQ | trained local | accuracy | 51.6 [32.1, 71.0] | 150/seed × 3 seeds | student_t_over_independent_seeds | matched | first/all 150 test items; single answer ordering |
| SAT-v2 / SAT-Real | GPT-5.6 Sol (high; partial) | direct frontier VLM | accuracy | — | 0/150 cached |  | incomplete due provider credit exhaustion; excluded from headline claims | all 150 test items; single answer ordering |
| SAT-v2 / SAT-Real | GPT-4V (paper) | published general VLM | accuracy | 50.7 | 150 |  | published protocol; circular answer-order evaluation | paper circular evaluation |
| SAT-v2 / SAT-Real | GPT-4o (paper) | published general VLM | accuracy | 60.3 | 150 |  | published protocol; circular answer-order evaluation | paper circular evaluation |
| SAT-v2 / SAT-Real | GPT-4.1 (paper) | published general VLM | accuracy | 74.0 | 150 |  | published protocol; circular answer-order evaluation | paper circular evaluation |
| SAT-v2 / SAT-Real | SpatialDreamer | published spatial SOTA | accuracy | 93.9 | 150 |  | published protocol; circular answer-order evaluation | paper circular evaluation |
| BLINK Multi-view | Frozen trajectory flow (colored source) | ours | accuracy | 74.7 [69.3, 80.1] | 133/seed × 3 seeds | student_t_over_independent_seeds | matched | full validation n=133 |
| BLINK Multi-view | GPT-5.6 Sol (high; partial) | direct frontier VLM | accuracy | — | 0/133 cached |  | incomplete due provider credit exhaustion; excluded from headline claims | full validation n=133 |
| BLINK Multi-view | GPT-4V direct | published general VLM | accuracy | 58.7 | 133 |  | matched public validation | full validation n=133 |
| BLINK Multi-view | Gemini Pro direct | published general VLM | accuracy | 41.3 | 133 |  | matched public validation | full validation n=133 |
| BLINK Multi-view | P2 + Gemini 2.5 Pro | published tool VLM | accuracy | 63.9 | 133 |  | matched public validation | full validation n=133 |
| BLINK Multi-view | DR-MV3D | published spatial model | accuracy | 56.4 |  |  | published reference; split/protocol differs | BLINK Multi-view |
| BLINK Multi-view | Human | human | accuracy | 92.5 | 133 |  | matched public validation | full validation n=133 |
| BLINK Multi-view | LoFTR specialist | published CV specialist | accuracy | 90.2 |  |  | published reference; split/protocol differs | paper dev/test table |

## Sources

[OpenAI GPT-5.6 Sol documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol), [Anthropic Claude Fable 5](https://www.anthropic.com/claude/fable), [SAT dataset card](https://huggingface.co/datasets/array/SAT), [SAT paper](https://arxiv.org/abs/2412.07755), [SpatialDreamer](https://arxiv.org/abs/2512.07733), [BLINK paper](https://arxiv.org/abs/2404.12390), [P2 / Perception Programs](https://openaccess.thecvf.com/content/CVPR2026/html/Janjua_Dont_Show_Pixels_Show_Cues_Unlocking_Visual_Tool_Reasoning_in_CVPR_2026_paper.html), [DR-MV3D](https://dr-mv3d.github.io/).

# Visual Reasoning


## Motivation
Chain-of-thought has been taking off in LLMs, but there has not been as much of an emphasis on visual reasoning. Many tasks are difficult to solve in one shot, but a human may be able to reduce the complexity by sketching out steps to the solution, either on paper or in the mind's eye. This research aims to recreate this process by adding an extra visual thinking layer at the end of a VLM and benchmarking it against existing VLMs and LLMs.

## Comprehensive Apple MPS experiments

The reproducible paper suite, benchmark rationale, statistical protocol, output
schema, and runtime profiles are documented in
[`docs/mps_paper_suite.md`](docs/mps_paper_suite.md). Launch the main three-seed
matrix with:

```bash
scripts/run_paper_mps_suite.sh --profile overnight --seeds 0 1 2 --keep-going
```

## PPO-free trajectory flow rebuild

The NeurReps-oriented flow-only method, generated-image QC, frozen baseline
archive, and MPS commands are documented in
[`docs/neurreps_flow_v1.md`](docs/neurreps_flow_v1.md). Run its resumable
three-task, three-seed overnight suite with:

```bash
.venv/bin/python scripts/run_neurreps_flow_suite.py --profile overnight --device mps
```

## Frontier VLM and published-SOTA audit

The resumable direct-VLM evaluation covers both rotation tasks, Ganis-Kievit
3-D, maze trace/path generation, the complete SAT-Real test set, and the
complete BLINK Multi-view validation set. It also performs SAT's official
original/reversed answer-order evaluation and compiles all archived learned
baselines and published references into one protocol-labelled table. With
`OPENAI_API_KEY` in the environment or repository `.env`, run:

```bash
scripts/run_sota_vlm_audit.sh
```

The auditable report and prediction-level JSON are written under
[`results/sota_vlm_baselines_2026-08-08`](results/sota_vlm_baselines_2026-08-08).

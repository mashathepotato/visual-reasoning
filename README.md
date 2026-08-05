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

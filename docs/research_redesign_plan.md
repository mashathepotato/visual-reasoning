# Flow-of-Thought research redesign plan

Last updated: 2026-08-02

This is the living checklist for the workshop and archival redesign. It records
verified repository facts, planned falsification tests, implementation state,
experiment state, and the paper sections affected. Empty result fields mean that
the experiment has not been run; they are not placeholders for expected results.

## Scientific decision

The defensible question is:

> Do learned, explicit visual transition traces improve spatial reasoning,
> generalization, or data efficiency over equally trained non-visual models,
> simpler transition mechanisms, and simpler controllers?

The recommended workshop target is **BENTO**, with colored-shape rotation as the
main task and Tetris as a controlled diagnostic. The current repository does not
support a NeurReps submission: it supervises a learned velocity field along a
rotation orbit, but it neither implements nor evaluates SO(2)-equivariance. Maze,
3D transfer, neuroscience, external benchmarks, and multi-agent framing are not
workshop-critical and must not delay the matched rotation study.

Claims are conditional on results:

- Retain a visual-trace claim only if exact or learned visual traces beat the
  matched no-trace classifier.
- Retain a learned-flow claim only if it beats deterministic rotation, direct
  prediction, and a matched linear/rectified-flow path.
- Use “geodesic” only if the rotation-orbit path wins a controlled path ablation;
  do not claim equivariance without a separate equivariance test.
- Retain PPO only if it beats greedy and supervised controllers at a matched
  observation and inference budget.
- Report a simpler surviving method or a negative result if these tests fail.

## Verified audit findings

### Reproducibility and protocol blockers

- There is no `AGENTS.md` in this repository.
- There are no experiment configuration files, package lock, or automated tests.
  `requirements.txt` uses lower bounds rather than a verified environment lock.
- Synthetic datasets are mutable RNG streams rather than stateless samples keyed
  by `(split, seed, item_id)`. `FastTetrisDataset` and `MazeTraceDataset` ignore
  the requested item index, so samples change with access order and worker count.
- Tetris, colored shapes, and the main 3D learned evaluations live in notebooks.
  There is no shared held-out evaluation CLI for learned methods.
- Training scripts generally save final weights without optimizer state,
  validation-based selection, resolved configuration, environment metadata, git
  commit, runtime, or test metrics.
- DINO and FoT rows use different image pipelines, item sets, and model-selection
  rules. Several notebook thresholds are selected on the evaluation set.
- The current handcrafted image-processing/BFS baseline in
  `benchmarks/vipergpt_results_seed0.json` reports 82% Tetris, 86.75% colored
  shapes, and 100% on both maze tasks. It is not ViperGPT and should be labeled a
  handcrafted CV/BFS baseline. Its performance rules out claims that the current
  tasks inherently require FoT.
- Paper result provenance is incomplete. In particular, the 74.5% Tetris table
  entry has no corresponding machine-readable result; the checked-in notebook's
  small final evaluation does not reproduce it.

### Dataset integrity

- `scripts/prep_ganis_kievit.py` shuffles without a seed and splits images rather
  than object identities.
- The checked-in 3D arrays contain 153 same-only training rows and 78 test rows
  (39 same, 39 different). All 41 object identities represented in the test set
  also occur in training. Therefore the present split cannot establish unseen-
  object generalization.
- Tetris FM train/test streams use the same six canonical shapes. Colored-shape
  and maze tasks have no immutable train/validation/test manifests.
- Tetris transformations differ between OpenCV and Kornia code paths, and colored
  shapes use arbitrary integer angles in one path versus a 5-degree grid in
  another. A canonical generator is required before matched comparisons.

### Method and implementation mismatches

- The live Tetris FM script trains finite-difference velocities along Kornia
  rotations with MSE and no OneCycle schedule. The paper describes MAE plus
  OneCycle and calls the method geodesic flow matching.
- The code does not enforce or measure SO(2)-equivariance. The precise current
  description is “rotation-orbit path supervision,” pending a controlled test.
- The three rotation environments duplicate nearly identical action and reward
  logic. Their default actions are `{-30,+30,-15,+15,-2,+2,commit_same,
  commit_different,+180}`; default maximum steps are 180, with a 360-degree
  rotation budget; rotation reward is negative MSE with no constant step penalty.
- Tetris notebook and script settings conflict (shape subsets, maximum steps,
  error threshold, and training budget). Some evaluations classify uncommitted
  PPO episodes from the best image-error threshold, conflating policy and search.
- The maze controller has privileged access to the BFS-derived trajectory length.
  Its actions advance the sketch by `{1,2,4,0}` oracle-indexed steps, and reward
  and termination use oracle progress rather than generated-trace validity.
- Tetris FM and maze “FM” are materially different: the former regresses a local
  finite-difference velocity; the latter regresses the next trace-segment delta.

### Paper integrity and scope

- No reviewer-directed prompt-injection text was found in the tracked TeX source
  or other searchable repository text. No submitted paper PDF is present, so the
  submitted artifact itself cannot be certified from this checkout.
- `paper/neurips.tex` and `paper/dissertation.tex` are tracked, despite the broad
  `paper/` ignore rule. However, required build inputs such as `main.bib`, the
  NeurIPS style, checklist, and referenced `paper/figures/` are absent, so the
  paper cannot be rebuilt from a clean checkout.
- The abstract, introduction, discussion, and conclusion currently claim broad
  generalization, biological mimicry, state-of-the-art superiority, and necessity
  of continuous visual reasoning without matched evidence.
- The neuroscience wording treats classic reaction-time findings as proof of a
  literal continuous brain simulation. At most, angular-disparity scaling offers
  a testable behavioral analogy; it is not evidence that this architecture is
  biologically plausible.
- Major related-work and implementation citations cannot be fully audited because
  the bibliography is missing. Citation verification is required before rewrite.

## Prioritized experiment matrix

| Priority | Reviewer concern | Planned response | Metrics | Implementation status | Experiment status | Result | Paper section | Venue |
|---|---|---|---|---|---|---|---|---|
| P0 | Unfair zero-shot baselines | Train small CNN and ViT on identical manifests, pixels, updates, and validation protocol | Accuracy, AUC, calibration, parameters, runtime | Implemented for colored rotation | Seeds 0/1/2 complete | ID accuracy: CNN 65.93 ± 0.78%, ViT 56.40 ± 2.01%; held-out-angle comparison has an unpaired-split caveat | Experiments | BENTO |
| P0 | Explicit traces may not help | Compare direct pair classifier, scalar/no-image controller, exact rendered trace, and learned trace | Accuracy/AUC, latency, transition calls | Not started | Not run | — | Main hypothesis | BENTO |
| P0 | Flow may be unnecessary | Hold controller fixed across deterministic rotation, direct learned predictor, linear/rectified flow, and rotation-orbit flow | Task metrics and trace fidelity | Not started | Not run | — | Method ablations | BENTO |
| P0 | Intermediate states may be blurry fades | Match predicted frames to exact rotations | Angular error, mask/edge IoU, SSIM, area drift, connected components, rollout drift | Not started | Not run | — | Trajectory evaluation | BENTO |
| P0 | PPO may be unnecessary | Compare greedy, supervised/BC, PPO, and random with common observations and budgets | Accuracy, success, actions, reward, calls, latency | Not started | Not run | — | Controller ablations | BENTO |
| P0 | Test leakage and unstable estimates | Lock validation-selected rules; run seeds 0/1/2; paired bootstrap over fixed test items | Mean, SD, 95% CI, n | Manifests, validation selection, and seed aggregation implemented | Full runs not launched | — | Protocol/statistics | BENTO |
| P0 | “Generalization” is vague | Predeclare ID and OOD-angle splits | Accuracy and trace error by angle | ID/OOD-angle manifests implemented | Baseline smoke only | — | Generalization | BENTO |
| P1 | Data efficiency may justify complexity | Use identical nested 1/5/10/25/50/100% subsets; screen 10/100% first | Accuracy vs examples and compute | Not started | Not run | — | Data efficiency | BENTO/ICLR |
| P1 | 3D split is leaky and tiny | Regenerate identity-disjoint splits or grouped cross-validation; remove test-tuned fusion | Macro accuracy/AUC and identity-grouped CI | Not started | Invalid current result | — | Transfer | ICLR |
| P1 | Maze uses oracle structure | Redesign reward/termination around trace validity and goal reach | Validity, reach, collision, connectivity, path ratio | Not started | Invalid current controller evidence | — | Additional tasks | ICLR |
| P1 | No external benchmark | Select one compatible benchmark only if core claim survives | Official metric plus trace metrics | Interface prototypes only | Not run | — | External validity | ICLR |
| P2 | SO(2)/equivariance claim | Implement and measure an actually equivariant alternative if path results justify it | Equivariance defect, OOD-angle error | Not started | Not run | — | Geometry | ICLR/NeurReps |
| P2 | Neuroscience framing | Test compute/steps versus angular disparity; otherwise reduce to motivation/limitations | Slope and uncertainty versus angular disparity | Not started | Not run | — | Motivation/limitations | ICLR optional |
| P2 | Multi-agent framing | Only implement explicit proposer/verifier communication after core baselines | Reliability at equal rounds/calls | Not started | Not run | — | Extension | Optional |

## Execution order

### Workshop-critical

1. Add a pinned, clean setup and record run metadata.
2. Implement stateless canonical pair generation and fixed ID/validation/OOD-angle
   manifests shared by every method.
3. Add fast unit and end-to-end smoke tests.
4. Add matched small CNN and ViT classifiers.
5. Build the common transition/controller evaluator and trajectory metrics.
6. Run the transition and controller factorial for three seeds.
7. Aggregate without silently dropping failed runs; produce confidence intervals,
   data/compute curves, and rule-based success/failure examples.
8. Rewrite the workshop paper only after the component decisions are known.

### ICLR-critical

1. Extend primary comparisons to five seeds and full data-efficiency curves.
2. Add stronger OOD identity/style/horizon tests.
3. Repair and reevaluate either identity-safe 3D transfer or one established
   external benchmark.
4. Complete controller search ablations and rollout-stability analysis.
5. Add true equivariance only if the rotation-orbit ablation supports pursuing it.
6. Verify venue overlap and dual-submission rules immediately before submission.

### Optional agentic extension

Defer a proposer/verifier system until the central trace claim survives. A passive
flow-network call is a learned transition model, not a second agent. Any later
extension must compare single versus multiple proposals, no/random/greedy/PPO
verifiers, communication rounds, and equalized inference calls.

## Compute and schedule

Estimates assume 64×64 inputs on one modern 24 GB GPU and must be recalibrated
from smoke benchmarks on the actual machine.

- Matched CNN/ViT baselines: 4–10 GPU-hours for the initial matrix.
- Transition/path ablations: 12–30 GPU-hours.
- Trajectory evaluation: 2–6 GPU-hours.
- Greedy/BC/PPO controller comparisons: 8–20 GPU-hours.
- Minimum three-seed workshop package: approximately 25–60 GPU-hours plus 2–4
  CPU-days for evaluation/aggregation, with 30–50% contingency.
- Identity-safe 3D or external benchmark work: reserve 20–100+ additional
  GPU-hours for ICLR, depending on the selected benchmark.

Local capability on 2026-08-02: Apple Silicon with MPS, no CUDA. The existing tiny
SAT smoke configuration completed in about 7 seconds, which validates only the
training loop—not scientific performance. Full primary runs should use a CUDA
host after local smoke validation.

Proposed dates:

- Aug 2–7: manifests, configuration, provenance, tests, and matched classifiers.
- Aug 8–14: transition ablations and trajectory metrics.
- Aug 15–20: controller comparisons and three-seed runs.
- Aug 21–24: OOD evaluation, statistics, and failure analysis.
- Aug 25–28: BENTO paper, figures, and clean-clone reproduction.

## Session log

### 2026-08-02

- [x] Read repository documentation and verified no applicable `AGENTS.md`.
- [x] Created branch `codex/research-reproducibility` from clean `main`.
- [x] Mapped data, notebooks, live scripts, environments, checkpoints, results,
  benchmark code, and paper sources.
- [x] Compiled all Python under `utils/`, `scripts/`, and `benchmarks/`.
- [x] Verified CLI help for all live FoT training/evaluation scripts.
- [x] Ran the smallest existing synthetic training smoke test (SAT-v2 smoke):
  one epoch, 8 train / 4 validation examples, seed 0, MPS, approximately 7 seconds.
  It produced `val_loss=1.3901`, `val_acc=0.0`; this is a pipeline check only.
- [x] Confirmed 3D object-identity overlap in the checked-in arrays.
- [x] Implement the reproducibility foundation: pinned direct dependencies,
  immutable manifests, stateless datasets, strict configs, run metadata, and
  aggregation that marks missing runs.
- [x] Implement matched small CNN/ViT baselines with validation-only checkpoint
  selection and machine-readable predictions.
- [x] Smoke-test the ViT pipeline on CPU: 10 train examples, 8 per evaluation
  split, 58,434 parameters, approximately 1.4 seconds end to end. Accuracy was
  0.5 on ID and OOD-angle smoke splits; this is explicitly non-scientific.
- [x] Track the legacy paper sources, remove reviewer template boilerplate, and
  remove unsupported legacy result claims from the NeurIPS draft.
- [x] Run full CNN/ViT seeds 0, 1, and 2 on Apple MPS and commit the aggregate
  results. All six runs completed in 71.9 minutes.
- [ ] Implement the no-trace/transition/controller factorial and trajectory metrics.

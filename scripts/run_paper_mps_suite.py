from __future__ import annotations

"""Resumable, one-command MPS benchmark and methodology suite."""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from utils.fot.reproducibility import collect_run_metadata, write_json


@dataclass(frozen=True)
class Stage:
    name: str
    command: List[str]
    outputs: List[Path]
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the comprehensive paper benchmark suite on Apple MPS.")
    parser.add_argument("--profile", choices=("smoke", "overnight", "extended"), default="overnight")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "models" / "runs" / "mps_paper_suite")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "mps_paper_suite")
    parser.add_argument("--categories", nargs="+", choices=("classical", "supervised", "dino", "sat", "fot"), default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue independent stages after a failure.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def py(*parts: str) -> List[str]:
    return [sys.executable, *parts]


def stage_plan(args: argparse.Namespace) -> List[Stage]:
    run_root = args.run_root.resolve(); seeds = list(dict.fromkeys(args.seeds))
    smoke = args.profile == "smoke"
    extended = args.profile == "extended"
    if smoke:
        seeds = seeds[:1]
    scratch_epochs = 1 if smoke else 50
    dino_epochs = 1 if smoke else 20
    dino_partial_epochs = 1 if smoke else (12 if extended else 8)
    sat_epochs = 1 if smoke else (8 if extended else 5)
    sat_train = 32 if smoke else (25000 if extended else 10000)
    sat_validation = 16 if smoke else 2000
    flow_epochs = 1 if smoke else (25 if extended else 15)
    flow_samples = 32 if smoke else (5000 if extended else 3000)
    ppo_steps = 64 if smoke else (50000 if extended else 30000)
    ppo_n_steps = 32 if smoke else 1024
    ppo_batch = 16 if smoke else 64
    eval_n = 8 if smoke else (500 if extended else 200)
    fm_steps = 2 if smoke else (10 if extended else 6)
    angle_step = 90 if smoke else (15 if extended else 30)
    stages: List[Stage] = []

    for seed in seeds:
        out = run_root / "classical" / f"seed{seed}" / "results.json"
        stages.append(Stage(f"classical_seed{seed}", py(str(REPO_ROOT / "benchmarks" / "vipergpt_pipeline_tests.py"),
            "--seed", str(seed), "--no-api", "--n-rotation", str(8 if smoke else 300),
            "--n-maze-trace", str(8 if smoke else 300), "--n-maze-solve", str(4 if smoke else 75),
            "--out", str(out)), [out], "classical"))

    for task in ("tetris", "colored"):
        directory = run_root / "classical" / "exact_rotation_search" / task
        command = py(str(REPO_ROOT / "scripts" / "fot" / "eval_exact_rotation_search.py"), "--task", task,
            "--output-dir", str(directory), "--device", "mps", "--angle-step", str(90 if smoke else 2))
        if smoke:
            command += ["--max-eval", "8", "--preliminary"]
        stages.append(Stage(f"exact_rotation_search_{task}", command, [directory / "summary.json"], "classical"))

    configs = {
        ("tetris", "cnn"): REPO_ROOT / "configs" / "baselines" / "rotation_cnn_tetris.json",
        ("tetris", "vit"): REPO_ROOT / "configs" / "baselines" / "rotation_vit_tetris.json",
        ("colored", "cnn"): REPO_ROOT / "configs" / "baselines" / "rotation_cnn_colored.json",
        ("colored", "vit"): REPO_ROOT / "configs" / "baselines" / "rotation_vit_colored.json",
    }
    for (task, model), config in configs.items():
        for seed in seeds:
            directory = run_root / "supervised" / task / model / f"seed{seed}"
            command = py(str(REPO_ROOT / "scripts" / "fot" / "train_supervised_baseline.py"),
                "--config", str(config), "--output-dir", str(directory), "--seed", str(seed),
                "--device", "mps", "--num-workers", str(args.num_workers), "--epochs", str(scratch_epochs))
            if smoke:
                command += ["--train-fraction", "0.004", "--max-eval-samples", "8", "--preliminary"]
            stages.append(Stage(f"supervised_{task}_{model}_seed{seed}", command, [directory / "summary.json"], "supervised"))

    for task in ("tetris", "colored", "ganis3d"):
        for seed in seeds:
            directory = run_root / "dino" / task / "frozen" / f"seed{seed}"
            command = py(str(REPO_ROOT / "scripts" / "fot" / "train_dinov3_pair_baseline.py"),
                "--dataset", task, "--mode", "frozen", "--output-dir", str(directory), "--seed", str(seed),
                "--epochs", str(dino_epochs), "--device", "mps", "--num-workers", str(args.num_workers))
            if smoke:
                command += ["--max-train", "32", "--max-eval", "8", "--preliminary"]
            stages.append(Stage(f"dino_frozen_{task}_seed{seed}", command, [directory / "summary.json"], "dino"))
    # Partial fine-tuning is useful but dominates MPS runtime (~20 additional
    # hours for three seeds), so it belongs to the extended profile.  Smoke
    # always exercises it to guard the MPS-specific training path.
    if extended or smoke:
        for task in ("tetris", "colored"):
            for seed in seeds:
                directory = run_root / "dino" / task / "partial" / f"seed{seed}"
                command = py(str(REPO_ROOT / "scripts" / "fot" / "train_dinov3_pair_baseline.py"),
                    "--dataset", task, "--mode", "partial", "--output-dir", str(directory), "--seed", str(seed),
                    "--epochs", str(dino_partial_epochs), "--batch-size", str(8 if smoke else 16),
                    "--device", "mps", "--num-workers", str(args.num_workers))
                if smoke:
                    command += ["--max-train", "16", "--max-eval", "8", "--preliminary"]
                stages.append(Stage(f"dino_partial_{task}_seed{seed}", command, [directory / "summary.json"], "dino"))

    for model in ("direct", "fot"):
        for seed in seeds:
            directory = run_root / "sat_v2" / model / f"seed{seed}"
            command = py(str(REPO_ROOT / "scripts" / "fot" / "train_sat_v2_benchmark.py"), "--model", model,
                "--output-dir", str(directory), "--seed", str(seed), "--device", "mps",
                "--epochs", str(sat_epochs), "--max-train", str(sat_train), "--max-validation", str(sat_validation),
                "--max-test", str(16 if smoke else 150),
                "--num-workers", str(args.num_workers), "--image-size", str(64 if smoke else 128),
                "--batch-size", str(8 if smoke else 32), "--streaming")
            if smoke:
                command += ["--smoke", "--preliminary"]
            stages.append(Stage(f"sat_v2_{model}_seed{seed}", command, [directory / "summary.json"], "sat"))

    for seed in seeds:
        tetris = run_root / "fot" / "tetris" / f"seed{seed}"
        tetris_flow = tetris / "flow"; rotator = tetris_flow / "rotator.pth"; corrector = tetris_flow / "corrector.pth"; controller = tetris / "ppo_controller"
        stages.append(Stage(f"fot_tetris_flow_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "train_fm_tetris.py"),
            "--out-rotator", str(rotator), "--out-corrector", str(corrector), "--seed", str(seed),
            "--train-samples", str(flow_samples), "--test-samples", str(max(16, flow_samples // 10)),
            "--epochs", str(flow_epochs), "--batch-size", str(8 if smoke else 32),
            "--summary-out", str(tetris_flow / "summary.json"), "--history-out", str(tetris_flow / "epoch_metrics.json"))
            + (["--preliminary"] if smoke else []), [rotator, corrector, tetris_flow / "summary.json"], "fot"))
        stages.append(Stage(f"fot_tetris_ppo_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "train_ppo_tetris_fm.py"),
            "--rotator", str(rotator), "--corrector", str(corrector), "--out", str(controller),
            "--log-dir", str(tetris / "ppo_logs"), "--seed", str(seed), "--fm-steps", str(fm_steps),
            "--total-timesteps", str(ppo_steps), "--n-steps", str(ppo_n_steps), "--batch-size", str(ppo_batch)),
            [controller.with_suffix(".zip")], "fot"))
        eval_dir = tetris / "evaluation"
        stages.append(Stage(f"fot_tetris_eval_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "eval_rotation_fot.py"),
            "--task", "tetris", "--flow-checkpoint", str(rotator), "--corrector", str(corrector),
            "--controller", str(controller.with_suffix('.zip')), "--output-dir", str(eval_dir), "--seed", str(seed),
            "--device", "mps", "--fm-steps", str(fm_steps), "--max-eval", str(eval_n),
            "--max-episode-steps", str(20 if smoke else 80)) + (["--preliminary"] if smoke else []),
            [eval_dir / "summary.json"], "fot"))
        transfer = tetris / "ganis3d_transfer"
        stages.append(Stage(f"fot_tetris_3d_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "eval_3d_fot_transfer.py"),
            "--source-model", "tetris", "--flow-checkpoint", str(rotator), "--corrector", str(corrector),
            "--output-dir", str(transfer), "--seed", str(seed), "--device", "mps", "--fm-steps", str(fm_steps),
            "--angle-step", str(angle_step), "--max-eval", str(8 if smoke else 78)) + (["--preliminary"] if smoke else []),
            [transfer / "summary.json"], "fot"))

        colored = run_root / "fot" / "colored" / f"seed{seed}"; colored_flow = colored / "flow" / "best_checkpoint.pt"
        stages.append(Stage(f"fot_colored_flow_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "train_fm_colored.py"),
            "--output-dir", str(colored / "flow"), "--seed", str(seed), "--device", "mps",
            "--train-samples", str(flow_samples), "--validation-samples", str(max(16, flow_samples // 10)),
            "--epochs", str(flow_epochs), "--batch-size", str(8 if smoke else 32), "--num-workers", str(args.num_workers))
            + (["--preliminary"] if smoke else []), [colored_flow], "fot"))
        colored_controller = colored / "ppo_controller"
        stages.append(Stage(f"fot_colored_ppo_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "train_ppo_colors_fm.py"),
            "--ckpt", str(colored_flow), "--out", str(colored_controller), "--log-dir", str(colored / "ppo_logs"),
            "--seed", str(seed), "--fm-steps", str(fm_steps), "--total-timesteps", str(ppo_steps),
            "--n-steps", str(ppo_n_steps), "--batch-size", str(ppo_batch)), [colored_controller.with_suffix('.zip')], "fot"))
        colored_eval = colored / "evaluation"
        stages.append(Stage(f"fot_colored_eval_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "eval_rotation_fot.py"),
            "--task", "colored", "--flow-checkpoint", str(colored_flow), "--controller", str(colored_controller.with_suffix('.zip')),
            "--output-dir", str(colored_eval), "--seed", str(seed), "--device", "mps", "--fm-steps", str(fm_steps),
            "--max-eval", str(eval_n), "--max-episode-steps", str(20 if smoke else 80)) + (["--preliminary"] if smoke else []),
            [colored_eval / "summary.json"], "fot"))
        colored_transfer = colored / "ganis3d_transfer"
        stages.append(Stage(f"fot_colored_3d_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "eval_3d_fot_transfer.py"),
            "--source-model", "colored", "--flow-checkpoint", str(colored_flow), "--output-dir", str(colored_transfer),
            "--seed", str(seed), "--device", "mps", "--fm-steps", str(fm_steps), "--angle-step", str(angle_step),
            "--max-eval", str(8 if smoke else 78)) + (["--preliminary"] if smoke else []),
            [colored_transfer / "summary.json"], "fot"))

        maze = run_root / "fot" / "maze" / f"seed{seed}"; maze_flow = maze / "flow"; sketcher = maze_flow / "sketcher.pth"; maze_controller = maze / "ppo_controller"
        stages.append(Stage(f"fot_maze_flow_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "train_fm_maze_sketcher.py"),
            "--out", str(sketcher), "--seed", str(seed), "--train-samples", str(flow_samples),
            "--validation-samples", str(max(16, flow_samples // 10)), "--epochs", str(flow_epochs),
            "--batch-size", str(8 if smoke else 32), "--summary-out", str(maze_flow / "summary.json"),
            "--history-out", str(maze_flow / "epoch_metrics.json")) + (["--preliminary"] if smoke else []),
            [sketcher, maze_flow / "summary.json"], "fot"))
        stages.append(Stage(f"fot_maze_ppo_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "train_ppo_maze_progress.py"),
            "--sketcher", str(sketcher), "--out", str(maze_controller), "--log-dir", str(maze / "ppo_logs"),
            "--seed", str(seed), "--total-timesteps", str(ppo_steps), "--n-steps", str(ppo_n_steps),
            "--batch-size", str(ppo_batch)), [maze_controller.with_suffix('.zip')], "fot"))
        maze_eval = maze / "evaluation"; maze_metrics = maze_eval / "metrics.json"; maze_summary = maze_eval / "summary.json"
        stages.append(Stage(f"fot_maze_eval_seed{seed}", py(str(REPO_ROOT / "scripts" / "fot" / "eval_maze_trace_validity.py"),
            "--seed", str(seed), "--n", str(eval_n), "--method", "fot", "--rollout", "controller",
            "--calibrate-seed", str(seed + 1000), "--calibrate-n", str(eval_n), "--sketcher", str(sketcher),
            "--controller", str(maze_controller.with_suffix('.zip')), "--device", "mps", "--out", str(maze_metrics),
            "--summary-out", str(maze_summary), "--save-results"), [maze_metrics, maze_summary], "fot"))

    if args.categories:
        stages = [stage for stage in stages if stage.category in set(args.categories)]
    return stages


def run_stage(stage: Stage, environment: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(stage.command, cwd=REPO_ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line); sys.stdout.flush(); handle.write(line); handle.flush()
        return int(process.wait())


def main() -> None:
    args = parse_args(); args.run_root = args.run_root.resolve(); args.results_dir = args.results_dir.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True); args.results_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Apple MPS is unavailable in this Python environment")
    stages = stage_plan(args); status_path = args.results_dir / "status.json"
    state: Dict[str, Any] = {"schema_version": 1, "profile": args.profile, "seeds": args.seeds,
        "started_at_utc": utc_now(), "finished_at_utc": None, "status": "running", "run_root": str(args.run_root),
        "results_dir": str(args.results_dir), "metadata": collect_run_metadata(repo_root=REPO_ROOT), "stages": []}
    write_json(status_path, state)
    environment = os.environ.copy(); environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"; environment["PYTHONUNBUFFERED"] = "1"
    failed = False; started = time.perf_counter()
    for index, stage in enumerate(stages, start=1):
        record: Dict[str, Any] = {"name": stage.name, "category": stage.category, "command": stage.command,
                                  "outputs": [str(path) for path in stage.outputs], "started_at_utc": utc_now()}
        complete = all(path.exists() for path in stage.outputs)
        print(f"[{index}/{len(stages)}] {stage.name}")
        if complete and not args.force:
            record.update({"status": "skipped_complete", "returncode": 0, "elapsed_seconds": 0.0})
        elif args.dry_run:
            print(" ".join(stage.command)); record.update({"status": "dry_run", "returncode": None, "elapsed_seconds": 0.0})
        else:
            stage_started = time.perf_counter(); code = run_stage(stage, environment, args.run_root / "logs" / f"{stage.name}.log")
            okay = code == 0 and all(path.exists() for path in stage.outputs)
            record.update({"status": "complete" if okay else "failed", "returncode": code,
                           "elapsed_seconds": time.perf_counter() - stage_started})
            failed = failed or not okay
        state["stages"].append(record); write_json(status_path, state)
        if record["status"] == "failed" and not args.keep_going:
            break
    state["finished_at_utc"] = utc_now(); state["elapsed_seconds"] = time.perf_counter() - started
    state["status"] = "dry_run" if args.dry_run else ("failed" if failed else "complete"); write_json(status_path, state)
    if not args.dry_run:
        aggregate = py(str(REPO_ROOT / "scripts" / "aggregate_benchmark_suite.py"), "--run-root", str(args.run_root),
                       "--results-dir", str(args.results_dir), "--status", str(status_path))
        aggregate_code = subprocess.call(aggregate, cwd=REPO_ROOT, env=environment)
        failed = failed or aggregate_code != 0
        if failed and state["status"] != "failed":
            state["status"] = "failed"; write_json(status_path, state)
    print(f"Suite status: {status_path}")
    print(f"Audit report: {args.results_dir / 'REPORT.md'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

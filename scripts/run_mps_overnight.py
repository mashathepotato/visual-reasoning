from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from utils.fot.aggregation import aggregate_run_directories
from utils.fot.reproducibility import collect_run_metadata, write_json


CONFIGS = {
    "cnn": REPO_ROOT / "configs" / "baselines" / "rotation_cnn_colored.json",
    "vit": REPO_ROOT / "configs" / "baselines" / "rotation_vit_colored.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all implemented colored-rotation baselines sequentially on Apple MPS."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", choices=sorted(CONFIGS), default=["cnn", "vit"])
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "models" / "runs" / "mps_baselines")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "mps_baselines")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Rerun completed runs instead of skipping them.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without launching training.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run both models for one epoch on tiny subsets; outputs are marked preliminary.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(command: List[str], *, log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return int(process.wait())


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    models = list(dict.fromkeys(str(model) for model in args.models))
    if not seeds:
        raise ValueError("At least one seed is required")
    if not args.dry_run and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Apple MPS is not available in this Python environment")

    output_root = args.output_root.resolve()
    results_dir = args.results_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    state_path = results_dir / ("overnight_smoke_status.json" if args.smoke else "overnight_status.json")
    started = time.perf_counter()
    state: Dict[str, Any] = {
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "device": "mps",
        "smoke": bool(args.smoke),
        "seeds": seeds,
        "models": models,
        "output_root": str(output_root),
        "results_dir": str(results_dir),
        "runs": [],
        "metadata": collect_run_metadata(repo_root=REPO_ROOT),
    }
    write_json(state_path, state)

    environment = os.environ.copy()
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    failed = False
    for model_name in models:
        for seed in seeds:
            suffix = "_smoke" if args.smoke else ""
            run_dir = output_root / f"{model_name}{suffix}" / f"seed{seed}"
            summary_path = run_dir / "summary.json"
            record: Dict[str, Any] = {
                "model": model_name,
                "seed": seed,
                "run_directory": str(run_dir),
                "started_at_utc": utc_now(),
            }
            if summary_path.exists() and not args.force:
                record.update({"status": "skipped_complete", "returncode": 0, "elapsed_seconds": 0.0})
                state["runs"].append(record)
                write_json(state_path, state)
                print(f"Skipping completed run: {run_dir}")
                continue

            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "fot" / "train_supervised_baseline.py"),
                "--config",
                str(CONFIGS[model_name]),
                "--output-dir",
                str(run_dir),
                "--seed",
                str(seed),
                "--device",
                "mps",
                "--num-workers",
                str(args.num_workers),
            ]
            if args.smoke:
                command.extend(
                    [
                        "--epochs",
                        "1",
                        "--train-fraction",
                        "0.002",
                        "--max-eval-samples",
                        "8",
                        "--preliminary",
                    ]
                )
            print("Launching:", " ".join(command))
            if args.dry_run:
                record.update({"status": "dry_run", "returncode": None, "elapsed_seconds": 0.0})
            else:
                run_started = time.perf_counter()
                returncode = run_command(command, log_path=run_dir / "console.log", env=environment)
                record.update(
                    {
                        "status": "complete" if returncode == 0 and summary_path.exists() else "failed",
                        "returncode": returncode,
                        "elapsed_seconds": time.perf_counter() - run_started,
                    }
                )
                failed = failed or record["status"] == "failed"
            state["runs"].append(record)
            write_json(state_path, state)

    if not args.dry_run:
        for model_name in models:
            suffix = "_smoke" if args.smoke else ""
            run_directories = [output_root / f"{model_name}{suffix}" / f"seed{seed}" for seed in seeds]
            try:
                aggregate = aggregate_run_directories(run_directories)
            except ValueError as error:
                state.setdefault("aggregation_errors", {})[model_name] = str(error)
                failed = True
            else:
                aggregate_name = f"{model_name}{suffix}_seeds{'-'.join(map(str, seeds))}.json"
                aggregate_path = results_dir / aggregate_name
                write_json(aggregate_path, aggregate)
                state.setdefault("aggregates", {})[model_name] = str(aggregate_path)

    state["finished_at_utc"] = utc_now()
    state["elapsed_seconds"] = time.perf_counter() - started
    state["status"] = "dry_run" if args.dry_run else ("failed" if failed else "complete")
    write_json(state_path, state)
    print(f"Overnight status: {state_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

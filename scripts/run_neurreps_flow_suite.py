#!/usr/bin/env python3
"""Run the PPO-free NeurReps flow rebuild on MPS with resumable stages."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts/fot/train_trajectory_flow.py"
sys.path.insert(0, str(REPO_ROOT))

from utils.fot.metrics import mean_t_ci  # noqa: E402


PROFILES = {
    "smoke": {"train_samples": 24, "validation_samples": 12, "epochs": 1, "batch_size": 4,
              "width": 8, "context_dim": 32, "integration_steps": 4, "rollout_batch": 2,
              "audit_every": 1, "validation_every": 1},
    "overnight": {"train_samples": 3000, "validation_samples": 400, "epochs": 30, "batch_size": 20,
                  "width": 16, "context_dim": 64, "integration_steps": 12, "rollout_batch": 4,
                  "rollout_every": 10, "audit_every": 5, "validation_every": 5,
                  "learning_rate": 1e-3},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="overnight")
    parser.add_argument("--tasks", nargs="+", choices=("tetris", "colored", "maze"),
                        default=["tetris", "colored", "maze"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "models/runs/neurreps_flow_v1")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results/neurreps_flow_v1")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def confidence_interval(values: List[float]) -> tuple[Optional[float], Optional[float]]:
    if len(values) < 2:
        return None, None
    return mean_t_ci(values)


def aggregate(summaries: List[dict], results_dir: Path, profile: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for summary in summaries:
        for metric, value in summary["metrics"]["validation"].items():
            metric_rows.append({"task": summary["task"], "seed": summary["seed"], "metric": metric,
                                "value": value, "preliminary": summary["preliminary"]})
    with (results_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task", "seed", "metric", "value", "preliminary"),
            lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(metric_rows)
    grouped: Dict[tuple, List[float]] = {}
    for row in metric_rows:
        grouped.setdefault((row["task"], row["metric"]), []).append(float(row["value"]))
    aggregates = []
    for (task, metric), values in sorted(grouped.items()):
        low, high = confidence_interval(values)
        aggregates.append({"task": task, "metric": metric, "n_seeds": len(values), "mean": mean(values),
                           "ci95_low": low, "ci95_high": high, "values": values})
    payload = {"schema_version": 1, "profile": profile, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "method": "trajectory_supervised_spatial_flow_no_ppo", "summaries": summaries,
               "aggregates": aggregates}
    (results_dir / "flow_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# NeurReps trajectory-flow results", "", f"Profile: `{profile}`. PPO is not used.", "",
             "| Task | Metric | Seeds | Mean | 95% CI |", "|---|---|---:|---:|---:|"]
    for row in aggregates:
        interval = (
            "undefined (n<2)" if row["ci95_low"] is None
            else f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]"
        )
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['n_seeds']} | {row['mean']:.6f} | {interval} |"
        )
    lines += ["", "Each run directory contains `audit_best.png`, `quality_metrics.json`, the complete epoch history, "
              "resolved configuration, provenance, and the best checkpoint.", ""]
    (results_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = PROFILES[args.profile]
    profile_run_root = args.run_root / args.profile
    profile_results_dir = args.results_dir / args.profile
    profile_run_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures = []
    for task in args.tasks:
        for seed in args.seeds:
            output = profile_run_root / task / f"seed{seed}"
            summary_path = output / "summary.json"
            if summary_path.exists() and not args.rerun:
                print(f"[resume] {task} seed {seed}: {summary_path}", flush=True)
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue
            command = [
                sys.executable, str(TRAIN_SCRIPT), "--task", task, "--output-dir", str(output),
                "--device", args.device, "--seed", str(seed), "--num-workers", str(args.num_workers),
            ]
            stage_config = dict(config)
            if args.profile == "overnight" and task == "maze":
                stage_config.update({"width": 32, "context_dim": 128, "integration_steps": 8,
                                     "rollout_every": 2, "validation_every": 2,
                                     "learning_rate": 2e-4})
            for key, value in stage_config.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            if args.profile == "smoke":
                command.append("--preliminary")
            print(f"[run] {task} seed {seed}", flush=True)
            completed = subprocess.run(command, cwd=REPO_ROOT)
            if completed.returncode != 0 or not summary_path.exists():
                failures.append({"task": task, "seed": seed, "returncode": completed.returncode})
                continue
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            aggregate(summaries, profile_results_dir, args.profile)
    aggregate(summaries, profile_results_dir, args.profile)
    status = {"profile": args.profile, "completed": len(summaries), "failed": failures,
              "expected": len(args.tasks) * len(args.seeds)}
    (profile_results_dir / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

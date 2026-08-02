from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.fot.metrics import mean_t_ci
from utils.fot.reproducibility import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate every completed MPS paper-suite run.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, default=None)
    return parser.parse_args()


def flatten_metrics(summary: Dict[str, Any]) -> Iterable[Tuple[str, str, float]]:
    def walk(prefix: str, value: Dict[str, Any]):
        for key, child in value.items():
            if isinstance(child, (int, float)) and math.isfinite(float(child)):
                yield prefix, str(key), float(child)
            elif isinstance(child, dict):
                yield from walk(f"{prefix}/{key}", child)
    for split, metrics in summary.get("metrics", {}).items():
        if not isinstance(metrics, dict):
            continue
        yield from walk(str(split), metrics)


def classical_summaries(root: Path) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path in root.glob("classical/seed*/results.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = {"tetris": ("accuracy", "tetris_rotation"), "colored_shapes": ("accuracy", "colored_rotation"),
                   "3d_blocks": ("accuracy", "ganis3d"), "maze_trace": ("accuracy", "maze_trace"),
                   "maze_solve": ("success_rate", "maze_solve")}
        for key, (metric, task) in mapping.items():
            if isinstance(payload.get(key), dict) and isinstance(payload[key].get(metric), (int, float)):
                summaries.append({"experiment_name": f"classical_cv_bfs_{task}", "task": task,
                    "model": "classical_cv_bfs", "seed": int(payload["seed"]),
                    "metrics": {"test": {metric: float(payload[key][metric])}},
                    "source_file": str(path)})
    return summaries


def main() -> None:
    args = parse_args(); args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in args.run_root.rglob("summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["source_file"] = str(path); summaries.append(payload)
    summaries.extend(classical_summaries(args.run_root))
    grouped: Dict[Tuple[str, str, str, str], List[Tuple[int, float]]] = defaultdict(list)
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        for split, metric, value in flatten_metrics(summary):
            key = (str(summary.get("experiment_name")), str(summary.get("task")), split, metric)
            grouped[key].append((int(summary.get("seed", 0)), value))
    for (experiment, task, split, metric), values in sorted(grouped.items()):
        ordered = sorted(values); numeric = [value for _, value in ordered]
        low, high = mean_t_ci(numeric)
        rows.append({"experiment": experiment, "task": task, "split": split, "metric": metric,
            "mean": statistics.fmean(numeric), "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
            "ci95_low": low, "ci95_high": high,
            "ci_method": "student_t_over_independent_seeds" if len(numeric) > 1 else "undefined_single_seed",
            "n_seeds": len(numeric), "seeds": [seed for seed, _ in ordered], "values": numeric})
    status = None
    if args.status and args.status.exists():
        status = json.loads(args.status.read_text(encoding="utf-8"))
    audit = {"schema_version": 1, "statistical_unit": "independent training seed for aggregate intervals; test item for within-run Wilson accuracy intervals",
             "completed_summary_files": [summary.get("source_file") for summary in summaries],
             "aggregate_rows": rows, "suite_status": status,
             "warnings": ["Ganis-Kievit has 78 test pairs and legacy identity overlap; do not describe it as unseen-object OOD.",
                          "Held-out-angle and ID manifests contain different scenes, so compare methods within each split, not ID versus OOD accuracy directly.",
                          "SAT-v2 selects checkpoints on a finite streamed synthetic validation subset and reports the complete 150-question real-image test split."]}
    write_json(args.results_dir / "audit.json", audit)
    with (args.results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "task", "split", "metric", "mean", "std", "ci95_low", "ci95_high", "ci_method", "n_seeds", "seeds", "values"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "seeds": json.dumps(row["seeds"]), "values": json.dumps(row["values"])})
    lines = ["# Comprehensive MPS benchmark audit", "", "Generated from completed machine-readable runs. Missing or failed stages remain visible in `audit.json`.", "",
             "| Experiment | Task | Split | Metric | Mean | SD | 95% CI | Seeds |", "|---|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        interval = "—" if math.isnan(row["ci95_low"]) else f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        lines.append(f"| {row['experiment']} | {row['task']} | {row['split']} | {row['metric']} | {row['mean']:.4f} | {row['std']:.4f} | {interval} | {row['n_seeds']} |")
    lines.extend(["", "## Audit warnings", "", *[f"- {warning}" for warning in audit["warnings"]]])
    (args.results_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.results_dir / 'audit.json'}")
    print(f"Wrote {args.results_dir / 'metrics.csv'}")
    print(f"Wrote {args.results_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()

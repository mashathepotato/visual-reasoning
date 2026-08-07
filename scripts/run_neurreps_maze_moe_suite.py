#!/usr/bin/env python3
"""Run and aggregate the frozen-rotation-expert maze MoE on Apple MPS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts/fot/train_maze_moe.py"
sys.path.insert(0, str(REPO_ROOT))

from utils.fot.metrics import mean_t_ci  # noqa: E402
from utils.fot.reproducibility import sha256_file, write_json  # noqa: E402


PROFILES = {
    "smoke": {
        "train_samples": 24,
        "validation_samples": 12,
        "epochs": 1,
        "batch_size": 4,
        "width": 8,
        "context_dim": 32,
        "expert_dim": 8,
        "router_width": 8,
        "router_temperature": 0.5,
        "integration_steps": 4,
        "rollout_batch": 2,
        "rollout_every": 2,
        "audit_every": 1,
        "validation_every": 1,
    },
    "overnight": {
        "train_samples": 3000,
        "validation_samples": 400,
        "epochs": 30,
        "batch_size": 20,
        "width": 32,
        "context_dim": 128,
        "expert_dim": 16,
        "router_width": 16,
        "router_temperature": 0.5,
        "integration_steps": 8,
        "rollout_batch": 4,
        "rollout_every": 2,
        "audit_every": 5,
        "validation_every": 2,
        "learning_rate": 2e-4,
    },
}

REPORT_METRICS = (
    "endpoint_iou",
    "trajectory_mse",
    "intermediate_prefix_iou",
    "premature_activation_rate",
    "future_path_mean_intensity",
    "obstacle_violation_rate",
    "router_tetris_weight",
    "router_tetris_weight_on_path",
    "router_entropy",
)

PAIRED_METRICS = (
    "endpoint_iou",
    "trajectory_mse",
    "intermediate_prefix_iou",
    "premature_activation_rate",
    "future_path_mean_intensity",
    "obstacle_violation_rate",
)

PER_SEED_ARTIFACTS = (
    "summary.json",
    "quality_metrics.json",
    "epoch_metrics.json",
    "resolved_config.json",
    "run_metadata.json",
)

SCRATCH_TRAINABLE_PARAMETERS = 887_905


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="overnight")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--expert-run-root", type=Path,
        default=REPO_ROOT / "models/runs/neurreps_flow_v1/overnight",
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=REPO_ROOT / "models/runs/neurreps_maze_moe_v1",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=REPO_ROOT / "results/neurreps_maze_moe_v1",
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def interval(values: List[float]) -> tuple[Optional[float], Optional[float]]:
    if len(values) < 2:
        return None, None
    return mean_t_ci(values)


def display_interval(
    metric: str, low: Optional[float], high: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    if low is None or high is None:
        return None, None
    if metric == "router_entropy":
        return max(0.0, low), min(math.log(2.0), high)
    return max(0.0, low), min(1.0, high)


def baseline_reference() -> Dict[str, dict]:
    path = REPO_ROOT / "results/neurreps_flow_v1/posthoc_v2/posthoc_results.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference = {}
    for row in payload["aggregates"]:
        if row["task"] == "maze" and row["renderer"] == "additive_flow":
            reference[row["metric"]] = {
                "mean": row["mean"],
                "ci95_low": row["ci95_low_display"],
                "ci95_high": row["ci95_high_display"],
                "seed_order": [0, 1, 2],
                "values": row["values"],
            }
    return reference


def paired_rows(summaries: List[dict], baseline: Dict[str, dict]) -> List[dict]:
    ordered = sorted(summaries, key=lambda item: int(item["seed"]))
    seeds = [int(item["seed"]) for item in ordered]
    comparisons = []
    for name, right_label in (
        ("learned_minus_uniform", "uniform"),
        ("learned_minus_scratch", "scratch_maze_reference"),
    ):
        for metric in PAIRED_METRICS:
            learned = [float(item["metrics"]["learned"][metric]) for item in ordered]
            if right_label == "uniform":
                right = [float(item["metrics"]["uniform"][metric]) for item in ordered]
            else:
                item = baseline.get(metric)
                if item is None or len(item["values"]) != len(learned):
                    continue
                right = [float(value) for value in item["values"]]
            differences = [left - comparison for left, comparison in zip(learned, right)]
            low, high = interval(differences)
            comparisons.append({
                "comparison": name,
                "metric": metric,
                "definition": "learned minus comparator",
                "seed_order": seeds,
                "differences": differences,
                "mean_difference": mean(differences),
                "std_difference": stdev(differences) if len(differences) >= 2 else None,
                "ci95_low": low,
                "ci95_high": high,
            })
    return comparisons


def process_quality_diagnostic(summaries: List[dict]) -> dict:
    """Summarize a post-hoc endpoint/process trade-off from saved validation history."""
    per_seed = []
    for summary in sorted(summaries, key=lambda item: int(item["seed"])):
        history_path = Path(summary["checkpoint"]).parent / "epoch_metrics.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))["epochs"]
        validated = [row for row in history if "validation_endpoint_iou" in row]
        best_endpoint = max(row["validation_endpoint_iou"] for row in validated)
        candidates = [
            row for row in validated
            if row["validation_endpoint_iou"] >= best_endpoint - 0.01
        ]
        selected = min(candidates, key=lambda row: row["validation_premature_activation_rate"])
        per_seed.append({
            "seed": int(summary["seed"]),
            "epoch": int(selected["epoch"]),
            "endpoint_iou": float(selected["validation_endpoint_iou"]),
            "trajectory_mse": float(selected["validation_trajectory_mse"]),
            "intermediate_prefix_iou": float(selected["validation_intermediate_prefix_iou"]),
            "premature_activation_rate": float(selected["validation_premature_activation_rate"]),
        })
    aggregates = {}
    for metric in (
        "endpoint_iou", "trajectory_mse", "intermediate_prefix_iou", "premature_activation_rate"
    ):
        values = [row[metric] for row in per_seed]
        if values:
            aggregates[metric] = {"mean": mean(values), "values": values}
    return {
        "status": "post_hoc_descriptive_not_checkpoint_selected",
        "rule": "among validated epochs within 0.01 endpoint IoU of that seed's best, minimize premature activation",
        "warning": "These epoch states were not retained as final checkpoints and are not used in the primary comparison.",
        "per_seed": per_seed,
        "aggregates": aggregates,
    }


def copy_seed_artifacts(summary: dict, results_dir: Path) -> dict:
    """Copy small provenance files and add the retained checkpoint hash."""
    enriched = dict(summary)
    checkpoint = Path(summary["checkpoint"])
    if checkpoint.exists():
        enriched["checkpoint_sha256"] = sha256_file(checkpoint)
    source = checkpoint.parent
    destination = results_dir / "per_seed" / f"seed{int(summary['seed'])}"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in PER_SEED_ARTIFACTS:
        path = source / filename
        if path.exists():
            shutil.copy2(path, destination / filename)
    write_json(destination / "summary.json", enriched)
    return enriched


def write_artifact_manifest(results_dir: Path) -> None:
    destination = results_dir / "SHA256SUMS"
    lines = []
    for path in sorted(results_dir.rglob("*")):
        if path.is_file() and path != destination:
            lines.append(f"{sha256_file(path)}  {path.relative_to(results_dir)}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(summaries: List[dict], results_dir: Path, profile: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[tuple, List[float]] = {}
    for summary in summaries:
        for mode, metrics in summary["metrics"].items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    grouped.setdefault((mode, metric), []).append(float(value))
    rows = []
    for (mode, metric), values in sorted(grouped.items()):
        low, high = interval(values)
        display_low, display_high = display_interval(metric, low, high)
        rows.append({
            "model": "maze_moe",
            "router_mode": mode,
            "metric": metric,
            "n_seeds": len(values),
            "mean": mean(values),
            "std": stdev(values) if len(values) >= 2 else None,
            "ci95_low": low,
            "ci95_high": high,
            "ci95_low_display": display_low,
            "ci95_high_display": display_high,
            "values": values,
        })
    baseline = baseline_reference() if profile == "overnight" else {}
    paired = paired_rows(summaries, baseline)
    process_diagnostic = process_quality_diagnostic(summaries)
    trainable_counts = sorted({int(item["trainable_parameter_count"]) for item in summaries})
    frozen_counts = sorted({int(item["frozen_parameter_count"]) for item in summaries})
    payload = {
        "schema_version": 1,
        "profile": profile,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "maze_moe_frozen_tetris_colored_spatial_experts",
        "protocol_notes": [
            "Tetris and colored checkpoints are frozen and seed-matched.",
            "Only the learned router is used during training.",
            "Uniform and single-expert rows are post-training gate interventions, not separately trained controls.",
            "The scratch-maze reference is the previously frozen three-seed post-hoc audit.",
            "Paired scratch deltas align both evaluators' documented seed order [0, 1, 2].",
        ],
        "summaries": summaries,
        "aggregates": rows,
        "paired_comparisons": paired,
        "posthoc_process_quality_diagnostic": process_diagnostic,
        "parameter_counts": {
            "moe_trainable": trainable_counts,
            "moe_frozen": frozen_counts,
            "scratch_maze_trainable": (
                SCRATCH_TRAINABLE_PARAMETERS if profile == "overnight" else None
            ),
        },
        "scratch_maze_reference": baseline,
    }
    write_json(results_dir / "maze_moe_results.json", payload)
    with (results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "model", "router_mode", "metric", "n_seeds", "mean", "std",
            "ci95_low", "ci95_high", "ci95_low_display", "ci95_high_display"
        )
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)

    parameter_note = (
        f"The MoE has {trainable_counts[0]:,} trainable and {frozen_counts[0]:,} frozen parameters."
        if len(trainable_counts) == 1 and len(frozen_counts) == 1
        else f"Observed trainable counts: {trainable_counts}; frozen counts: {frozen_counts}."
    )
    if profile == "overnight":
        difference = 100.0 * (trainable_counts[0] / SCRATCH_TRAINABLE_PARAMETERS - 1.0)
        parameter_note += (
            f" The scratch maze reference has {SCRATCH_TRAINABLE_PARAMETERS:,} trainable parameters, "
            f"so the trainable budgets differ by {difference:.1f}%; the MoE additionally benefits "
            "from its frozen pretrained feature bank."
        )
    lines = [
        "# Maze mixture-of-experts results",
        "",
        f"Profile: `{profile}`. Frozen seed-matched Tetris and colored rotation flows provide spatial features; PPO is not used.",
        "",
        parameter_note,
        "",
        "| Model / routing | Metric | Seeds | Mean | 95% t CI |",
        "|---|---|---:|---:|---:|",
    ]
    for metric in REPORT_METRICS:
        if metric in baseline:
            item = baseline[metric]
            lines.append(
                f"| scratch maze reference | {metric} | 3 | {item['mean']:.6f} | "
                f"[{item['ci95_low']:.6f}, {item['ci95_high']:.6f}] |"
            )
        for row in rows:
            if row["metric"] != metric:
                continue
            ci = (
                "undefined (n<2)" if row["ci95_low_display"] is None
                else f"[{row['ci95_low_display']:.6f}, {row['ci95_high_display']:.6f}]"
            )
            lines.append(
                f"| MoE / {row['router_mode']} | {metric} | {row['n_seeds']} | "
                f"{row['mean']:.6f} | {ci} |"
            )
    lines.extend([
        "",
        "Display intervals are clipped to each metric's natural range; raw Student-t intervals are retained in JSON and CSV.",
        "",
        "## Paired primary comparisons",
        "",
        "Every delta is learned minus comparator on the same seed. Positive is better for IoU and negative is better for MSE or premature activation.",
        "",
        "| Comparator | Metric | Mean delta | 95% t CI | Seedwise deltas |",
        "|---|---|---:|---:|---:|",
    ])
    for item in paired:
        if item["metric"] not in (
            "endpoint_iou", "trajectory_mse", "intermediate_prefix_iou", "premature_activation_rate"
        ):
            continue
        comparator = item["comparison"].removeprefix("learned_minus_")
        ci = (
            "undefined (n<2)" if item["ci95_low"] is None
            else f"[{item['ci95_low']:.6f}, {item['ci95_high']:.6f}]"
        )
        values = ", ".join(f"{value:+.6f}" for value in item["differences"])
        lines.append(
            f"| {comparator} | {item['metric']} | {item['mean_difference']:+.6f} | {ci} | {values} |"
        )
    diagnostic = process_diagnostic["aggregates"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The learned mixture matches the scratch maze flow within three-seed uncertainty, but learned routing does not consistently improve over a forced 50/50 gate. Router entropy is close to its two-expert maximum and audit maps are low-contrast, so the mechanism behaves like dense feature fusion rather than adaptive expert specialization.",
        "",
        "Because the trainable decoder also receives the raw maze condition, parity with scratch does not establish that rotation pretraining adds useful information. Forcing either single gate after training substantially degrades the aggregate endpoint result, but those are interventions on a jointly trained decoder—not separately trained controls. They establish sensitivity to the learned mixture regime, not transfer benefit or unique causal value for either expert.",
        "",
        "Primary checkpoints are selected only by endpoint IoU. A descriptive post-hoc history check found that all three seeds reached their cleanest near-best process point at epoch 24: "
        f"endpoint IoU {diagnostic['endpoint_iou']['mean']:.6f}, prefix IoU {diagnostic['intermediate_prefix_iou']['mean']:.6f}, "
        f"trajectory MSE {diagnostic['trajectory_mse']['mean']:.6f}, and premature activation {diagnostic['premature_activation_rate']['mean']:.6f}. "
        "These states are not used as primary results and were not retained as final checkpoints; they motivate preregistering a process-aware selection rule in the next run.",
        "",
        "The learned-router row is the trained model. Uniform, Tetris-only, and colored-only rows force the gate after training and therefore measure reliance on each expert; they are not independently trained baselines.",
        "",
    ])
    (results_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = PROFILES[args.profile]
    run_dir = args.run_root / args.profile / "learned"
    results_dir = args.results_dir / args.profile
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries, failures = [], []
    for seed in args.seeds:
        output = run_dir / f"seed{seed}"
        summary_path = output / "summary.json"
        if summary_path.exists() and not args.rerun:
            print(f"[resume] learned maze MoE seed {seed}: {summary_path}", flush=True)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary = copy_seed_artifacts(summary, results_dir)
            summaries.append(summary)
        else:
            tetris = args.expert_run_root / "tetris" / f"seed{seed}" / "best_checkpoint.pt"
            colored = args.expert_run_root / "colored" / f"seed{seed}" / "best_checkpoint.pt"
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--output-dir", str(output),
                "--tetris-checkpoint", str(tetris),
                "--colored-checkpoint", str(colored),
                "--device", args.device,
                "--seed", str(seed),
                "--num-workers", str(args.num_workers),
            ]
            for key, value in config.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            if args.profile == "smoke":
                command.append("--preliminary")
            print(f"[run] learned maze MoE seed {seed}", flush=True)
            completed = subprocess.run(command, cwd=REPO_ROOT)
            if completed.returncode != 0 or not summary_path.exists():
                failures.append({"seed": seed, "returncode": completed.returncode})
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary = copy_seed_artifacts(summary, results_dir)
            summaries.append(summary)
        audit = Path(summary["audit_image"])
        if audit.exists():
            destination = results_dir / "audits" / f"seed{seed}_learned_router.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audit, destination)
        aggregate(summaries, results_dir, args.profile)
    aggregate(summaries, results_dir, args.profile)
    status = {
        "profile": args.profile,
        "completed": len(summaries),
        "expected": len(args.seeds),
        "failed": failures,
    }
    write_json(results_dir / "status.json", status)
    write_artifact_manifest(results_dir)
    print(json.dumps(status, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

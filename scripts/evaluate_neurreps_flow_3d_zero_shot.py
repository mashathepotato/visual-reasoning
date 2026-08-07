#!/usr/bin/env python3
"""Frozen, label-free 2-D-flow transfer to Ganis-Kievit 3-D block pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from utils.fot.metrics import binary_auc, mean_t_ci, wilson_accuracy_ci
from utils.fot.reproducibility import collect_run_metadata, sha256_file, write_json
from utils.fot.trajectory_flow import (
    TrajectoryFlowField,
    integrate_deformation_times,
    rotation_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=REPO_ROOT / "models/runs/neurreps_flow_v1/overnight",
    )
    parser.add_argument(
        "--data-path", type=Path, default=REPO_ROOT / "data/test_balanced.npy"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "results/neurreps_flow_v1/ganis3d_zero_shot",
    )
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--source-models", nargs="+", choices=("tetris", "colored"),
        default=["tetris", "colored"],
    )
    parser.add_argument("--angle-step", type=int, default=10)
    parser.add_argument("--item-batch-size", type=int, default=6)
    parser.add_argument("--hypothesis-batch-size", type=int, default=48)
    parser.add_argument("--max-eval", type=int, default=0, help="0 evaluates all items")
    parser.add_argument("--audit-examples", type=int, default=6)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> Tuple[TrajectoryFlowField, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TrajectoryFlowField(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def image_tensor(item: dict, key: str, channels: int, device: torch.device) -> torch.Tensor:
    image = torch.as_tensor(item[key], dtype=torch.float32, device=device)
    image = ((image + 1.0) * 0.5).clamp(0.0, 1.0)
    return image.repeat(channels, 1, 1) if channels > 1 else image


def classification_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, Any]:
    """Threshold-free ranking plus a predeclared zero-margin decision rule."""
    y = np.asarray(labels, dtype=np.int64)
    margins = np.asarray(scores, dtype=np.float64)
    predictions = (margins >= 0.0).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (y == 1)))
    tn = int(np.sum((predictions == 0) & (y == 0)))
    fp = int(np.sum((predictions == 1) & (y == 0)))
    fn = int(np.sum((predictions == 0) & (y == 1)))
    correct = int(np.sum(predictions == y))
    low, high = wilson_accuracy_ci(correct, len(y))
    positive = margins[y == 1]
    negative = margins[y == 0]
    positive_recall = tp / (tp + fn) if tp + fn else None
    negative_recall = tn / (tn + fp) if tn + fp else None
    auc = binary_auc(y.tolist(), margins.tolist())
    mean_positive = float(np.mean(positive)) if len(positive) else None
    mean_negative = float(np.mean(negative)) if len(negative) else None
    effect = None
    if len(positive) >= 2 and len(negative) >= 2:
        pooled_variance = (
            ((len(positive) - 1) * float(np.var(positive, ddof=1)))
            + ((len(negative) - 1) * float(np.var(negative, ddof=1)))
        ) / (len(positive) + len(negative) - 2)
        effect = (mean_positive - mean_negative) / max(
            math.sqrt(max(0.0, pooled_variance)), 1e-12
        )
    return {
        "n": int(len(y)),
        "accuracy": correct / len(y),
        "accuracy_ci95_low": low,
        "accuracy_ci95_high": high,
        "accuracy_ci_method": "wilson_test_items",
        "balanced_accuracy": (
            0.5 * (positive_recall + negative_recall)
            if positive_recall is not None and negative_recall is not None else None
        ),
        "auc": float(auc) if math.isfinite(auc) else None,
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "decision_rule": "same iff mirror_error - same_error >= 0",
        "mean_margin_same": mean_positive,
        "mean_margin_different": mean_negative,
        "margin_cohens_d": effect,
    }


def metrics_by_angle(predictions: Sequence[dict]) -> Dict[str, dict]:
    output: Dict[str, dict] = {}
    for angle in sorted({int(row["provided_angle_degrees"]) for row in predictions}):
        selected = [row for row in predictions if int(row["provided_angle_degrees"]) == angle]
        output[str(angle)] = classification_metrics(
            [int(row["label"]) for row in selected],
            [float(row["score"]) for row in selected],
        )
    return output


def angle_indices(angles: Sequence[int], provided: int) -> List[int]:
    candidates = {int(provided) % 360, (-int(provided)) % 360}
    indices = [index for index, value in enumerate(angles) if value in candidates]
    if not indices:
        raise ValueError(
            f"Provided angle {provided} is absent from the scan grid; choose a compatible --angle-step"
        )
    return indices


@torch.no_grad()
def evaluate_model(
    model: TrajectoryFlowField,
    rows: Sequence[dict],
    *,
    steps: int,
    angle_step: int,
    item_batch_size: int,
    hypothesis_batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    angles = list(range(0, 360, angle_step))
    channels = model.state_channels
    all_errors: List[torch.Tensor] = []
    for offset in range(0, len(rows), item_batch_size):
        items = rows[offset:offset + item_batch_size]
        source = torch.stack([image_tensor(item, "x0", channels, device) for item in items])
        target = torch.stack([image_tensor(item, "x1", channels, device) for item in items])
        variants = torch.stack([source, torch.flip(source, dims=[-1])], dim=1)
        expanded_source = variants[:, :, None].expand(-1, -1, len(angles), -1, -1, -1)
        expanded_target = target[:, None, None].expand_as(expanded_source)
        expanded_angles = torch.tensor(angles, device=device, dtype=source.dtype)[None, None]
        expanded_angles = expanded_angles.expand(len(items), 2, -1)
        flat_source = expanded_source.reshape(-1, channels, source.shape[-2], source.shape[-1])
        flat_target = expanded_target.reshape_as(flat_source)
        flat_angles = expanded_angles.reshape(-1)
        chunk_errors = []
        for start in range(0, flat_source.shape[0], hypothesis_batch_size):
            stop = start + hypothesis_batch_size
            source_chunk = flat_source[start:stop]
            prediction = integrate_deformation_times(
                model,
                source_chunk,
                source_chunk,
                rotation_action(flat_angles[start:stop]),
                [1.0],
                max_step=1.0 / steps,
                clamp=(0.0, 1.0),
            )[0]
            chunk_errors.append(
                ((prediction - flat_target[start:stop]) ** 2).mean(dim=(1, 2, 3)).cpu()
            )
        all_errors.append(torch.cat(chunk_errors).reshape(len(items), 2, len(angles)))

    errors = torch.cat(all_errors)
    protocols: Dict[str, List[dict]] = {"provided_angle": [], "angle_marginalized": []}
    for index, (item, item_errors) in enumerate(zip(rows, errors)):
        label = 1 if item.get("label") == "same" else 0
        subsets = {
            "provided_angle": angle_indices(angles, int(item["angle"])),
            "angle_marginalized": list(range(len(angles))),
        }
        for protocol, indices in subsets.items():
            same_subset = item_errors[0, indices]
            mirror_subset = item_errors[1, indices]
            same_local = int(torch.argmin(same_subset))
            mirror_local = int(torch.argmin(mirror_subset))
            same_index, mirror_index = indices[same_local], indices[mirror_local]
            same_error = float(item_errors[0, same_index])
            mirror_error = float(item_errors[1, mirror_index])
            protocols[protocol].append({
                "sample_id": f"ganis3d-{index:03d}-{item.get('name', '')}",
                "name": str(item.get("name", "")),
                "label": label,
                "prediction": int(same_error <= mirror_error),
                "score": mirror_error - same_error,
                "same_error": same_error,
                "mirror_error": mirror_error,
                "best_same_angle_degrees": int(angles[same_index]),
                "best_mirror_angle_degrees": int(angles[mirror_index]),
                "provided_angle_degrees": int(item["angle"]),
            })

    summaries: Dict[str, dict] = {}
    for protocol, predictions in protocols.items():
        summaries[protocol] = classification_metrics(
            [int(row["label"]) for row in predictions],
            [float(row["score"]) for row in predictions],
        )
        summaries[protocol]["by_provided_angle"] = metrics_by_angle(predictions)
    return summaries, protocols


def configure_matplotlib(output: Path) -> None:
    cache = output / ".mpl-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def show_tensor(axis, tensor: torch.Tensor, title: str, *, error: bool = False) -> None:
    image = tensor.detach().cpu().permute(1, 2, 0)
    if image.shape[-1] == 1:
        axis.imshow(image.squeeze(-1), cmap="magma" if error else "gray", vmin=0, vmax=1)
    else:
        axis.imshow(image.clamp(0, 1))
    axis.set_title(title)
    axis.axis("off")


@torch.no_grad()
def make_audit(
    model: TrajectoryFlowField,
    rows: Sequence[dict],
    predictions: Sequence[dict],
    *,
    steps: int,
    examples: int,
    device: torch.device,
    destination: Path,
) -> None:
    configure_matplotlib(destination.parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    same_indices = [index for index, row in enumerate(rows) if row.get("label") == "same"]
    diff_indices = [index for index, row in enumerate(rows) if row.get("label") != "same"]
    half = max(1, examples // 2)
    selected = (same_indices[:half] + diff_indices[:examples - half])[:examples]
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    figure, axes = plt.subplots(len(selected), 10, figsize=(19, 2.2 * len(selected)), squeeze=False)
    for plot_row, item_index in enumerate(selected):
        item, result = rows[item_index], predictions[item_index]
        source = image_tensor(item, "x0", model.state_channels, device).unsqueeze(0)
        target = image_tensor(item, "x1", model.state_channels, device).unsqueeze(0)
        mirror = torch.flip(source, dims=[-1])
        same_endpoint = integrate_deformation_times(
            model, source, source,
            rotation_action(torch.tensor([result["best_same_angle_degrees"]], device=device)),
            [1.0], max_step=1.0 / steps, clamp=(0.0, 1.0),
        )[0]
        mirror_endpoint = integrate_deformation_times(
            model, mirror, mirror,
            rotation_action(torch.tensor([result["best_mirror_angle_degrees"]], device=device)),
            [1.0], max_step=1.0 / steps, clamp=(0.0, 1.0),
        )[0]
        use_same = bool(result["prediction"])
        selected_source = source if use_same else mirror
        selected_angle = (
            result["best_same_angle_degrees"] if use_same else result["best_mirror_angle_degrees"]
        )
        trajectory = integrate_deformation_times(
            model, selected_source, selected_source,
            rotation_action(torch.tensor([selected_angle], device=device)),
            times, max_step=1.0 / steps, clamp=(0.0, 1.0),
        )
        show_tensor(axes[plot_row, 0], source[0], f"source\ntrue={item['label']}")
        show_tensor(axes[plot_row, 1], target[0], f"target\npred={'same' if use_same else 'different'}")
        show_tensor(axes[plot_row, 2], same_endpoint[0], f"H_same\n{result['best_same_angle_degrees']}°")
        show_tensor(axes[plot_row, 3], mirror_endpoint[0], f"H_mirror\n{result['best_mirror_angle_degrees']}°")
        for time_index, frame in enumerate(trajectory):
            show_tensor(axes[plot_row, 4 + time_index], frame[0], f"selected\nt={times[time_index]:g}")
        selected_endpoint = trajectory[-1]
        show_tensor(
            axes[plot_row, 9], (selected_endpoint - target).abs()[0],
            f"|error|\nmargin={result['score']:.3g}", error=True,
        )
    figure.suptitle(
        "Frozen 2-D flow on Ganis-Kievit 3-D blocks (no intermediate 3-D ground truth)"
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate(per_seed: Sequence[dict], output_dir: Path) -> dict:
    metric_names = ("accuracy", "balanced_accuracy", "auc", "positive_recall", "negative_recall")
    aggregate_rows: List[dict] = []
    for source_model in sorted({row["source_model"] for row in per_seed}):
        for protocol in ("provided_angle", "angle_marginalized"):
            relevant = [row for row in per_seed if row["source_model"] == source_model]
            for metric in metric_names:
                values = [float(row["metrics"][protocol][metric]) for row in relevant]
                low, high = mean_t_ci(values)
                aggregate_rows.append({
                    "source_model": source_model,
                    "protocol": protocol,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": mean(values),
                    "std": stdev(values) if len(values) >= 2 else None,
                    "ci95_low": max(0.0, low) if math.isfinite(low) else None,
                    "ci95_high": min(1.0, high) if math.isfinite(high) else None,
                    "ci95_low_raw": low if math.isfinite(low) else None,
                    "ci95_high_raw": high if math.isfinite(high) else None,
                    "values": values,
                })

    ensembles = []
    for source_model in sorted({row["source_model"] for row in per_seed}):
        relevant = [row for row in per_seed if row["source_model"] == source_model]
        for protocol in ("provided_angle", "angle_marginalized"):
            per_run = [row["predictions"][protocol] for row in relevant]
            labels = [int(item["label"]) for item in per_run[0]]
            scores = [mean(float(run[index]["score"]) for run in per_run) for index in range(len(labels))]
            ensembles.append({
                "source_model": source_model,
                "protocol": protocol,
                "method": "mean reconstruction margin across frozen seeds",
                "metrics": classification_metrics(labels, scores),
            })

    payload = {
        "schema_version": 1,
        "protocol": {
            "checkpoint_use": "frozen; no retraining, 3-D validation, or checkpoint selection",
            "decision": "compare reconstruction error under original-source and horizontal-reflection hypotheses",
            "provided_angle": "minimum over the signed pair of the supplied angular disparity",
            "angle_marginalized": "minimum over a predeclared full in-plane angle grid",
            "limitations": [
                "Horizontal 2-D reflection is only a proxy for a mirrored 3-D object.",
                "The dataset has no ground-truth intermediate images; trajectories are qualitative only.",
                "The checked-in split has 78 balanced items and legacy object-identity overlap.",
            ],
        },
        "per_seed": [
            {key: value for key, value in row.items() if key != "predictions"} for row in per_seed
        ],
        "aggregates": aggregate_rows,
        "ensembles": ensembles,
    }
    write_json(output_dir / "zero_shot_results.json", payload)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("source_model", "protocol", "metric", "n_seeds", "mean", "std", "ci95_low", "ci95_high")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in aggregate_rows)

    lines = [
        "# Frozen-flow zero-shot Ganis-Kievit audit",
        "",
        "All 2-D flow checkpoints were frozen before this evaluation. No 3-D labels were used for training, calibration, checkpoint selection, or threshold selection.",
        "",
        "| 2-D source | Protocol | Metric | Mean over seeds | 95% t CI |",
        "|---|---|---|---:|---:|",
    ]
    for row in aggregate_rows:
        ci = "undefined (n<2)" if row["ci95_low"] is None else f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
        lines.append(
            f"| {row['source_model']} | {row['protocol']} | {row['metric']} | {row['mean']:.3f} | {ci} |"
        )
    lines.extend(["", "## Seed ensembles", "", "| 2-D source | Protocol | Accuracy (Wilson 95% CI) | AUC |", "|---|---|---:|---:|"])
    for row in ensembles:
        metrics = row["metrics"]
        lines.append(
            f"| {row['source_model']} | {row['protocol']} | {metrics['accuracy']:.3f} "
            f"[{metrics['accuracy_ci95_low']:.3f}, {metrics['accuracy_ci95_high']:.3f}] | {metrics['auc']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation constraints",
        "",
        "This is a small zero-shot domain-transfer diagnostic, not a validated unseen-object benchmark. A horizontal image reflection is not a physical 3-D mirror transformation, and the stimulus set provides no ground-truth intermediate rotations. The audit images therefore test whether the frozen flow produces coherent continuous states, while classification tests only whether its reconstruction margin separates same from mirrored pairs.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    if args.angle_step <= 0 or 360 % args.angle_step:
        raise SystemExit("--angle-step must be a positive divisor of 360")
    if args.item_batch_size <= 0 or args.hypothesis_batch_size <= 0:
        raise SystemExit("Batch sizes must be positive")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(np.load(args.data_path, allow_pickle=True))
    if args.max_eval > 0:
        rows = rows[:args.max_eval]
    if not rows:
        raise SystemExit("No evaluation rows loaded")
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    write_json(args.output_dir / "resolved_protocol.json", {
        "schema_version": 1,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "data": {
            "path": str(args.data_path.resolve()),
            "sha256": sha256_file(args.data_path),
            "n_items": len(rows),
            "labels": {
                "same": sum(item.get("label") == "same" for item in rows),
                "different": sum(item.get("label") != "same" for item in rows),
            },
        },
    })

    per_seed = []
    for source_model in args.source_models:
        for seed in args.seeds:
            checkpoint_path = args.run_root / source_model / f"seed{seed}" / "best_checkpoint.pt"
            if not checkpoint_path.exists():
                raise SystemExit(f"Missing frozen checkpoint: {checkpoint_path}")
            model, checkpoint = load_model(checkpoint_path, device)
            started = time.perf_counter()
            steps = int(checkpoint["training_arguments"]["integration_steps"])
            metrics, predictions = evaluate_model(
                model, rows, steps=steps, angle_step=args.angle_step,
                item_batch_size=args.item_batch_size,
                hypothesis_batch_size=args.hypothesis_batch_size, device=device,
            )
            result = {
                "source_model": source_model,
                "seed": seed,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "best_epoch": int(checkpoint["best_epoch"]),
                "integration_steps": steps,
                "elapsed_seconds": time.perf_counter() - started,
                "metrics": metrics,
                "predictions": predictions,
            }
            per_seed.append(result)
            write_json(
                args.output_dir / "per_seed" / f"{source_model}_seed{seed}.json", result
            )
            if seed == args.seeds[0]:
                make_audit(
                    model, rows, predictions["angle_marginalized"], steps=steps,
                    examples=args.audit_examples, device=device,
                    destination=args.output_dir / "audits" / f"{source_model}_seed{seed}_trajectories.png",
                )
            aggregate(per_seed, args.output_dir)
            printable = {key: value for key, value in result.items() if key != "predictions"}
            print(json.dumps(printable, sort_keys=True), flush=True)
    aggregate(per_seed, args.output_dir)


if __name__ == "__main__":
    main()

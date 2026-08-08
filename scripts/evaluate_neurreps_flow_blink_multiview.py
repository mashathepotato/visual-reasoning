#!/usr/bin/env python3
"""Strict frozen-flow transfer to BLINK Multi-view Reasoning.

The evaluator never fits, calibrates, or selects a checkpoint on BLINK.  A
frozen 2-D object-rotation flow renders clockwise and counter-clockwise
hypotheses for the second video frame.  Camera motion is mapped to the opposite
apparent object motion by the standard camera/object duality.  The lower minimum
reconstruction error determines the fixed left/right answer.  An exact in-plane
rotation scan is reported as a hypothesis-class control, not as a learned model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from utils.fot.metrics import binary_auc, mean_t_ci, wilson_accuracy_ci
from utils.fot.reproducibility import collect_run_metadata, sha256_file, write_json
from utils.fot.rotation_ops import rotate_tensor
from utils.fot.trajectory_flow import (
    TrajectoryFlowField,
    integrate_deformation_times,
    rotation_action,
)


DATASET_ID = "BLINK-Benchmark/BLINK"
DATASET_CONFIG = "Multi-view_Reasoning"
# Pin the data so a rerun cannot silently evaluate different examples.
DEFAULT_REVISION = "a3666eb249237ba3d5eca8db21176cc47967e040"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=REPO_ROOT / "models/runs/neurreps_flow_v1/overnight",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "results/neurreps_flow_v1/blink_multiview_zero_shot",
    )
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--source-models", nargs="+", choices=("tetris", "colored"),
        default=["tetris", "colored"],
    )
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="val")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--angle-step", type=int, default=10)
    parser.add_argument("--max-angle", type=int, default=170)
    parser.add_argument("--item-batch-size", type=int, default=4)
    parser.add_argument("--hypothesis-batch-size", type=int, default=48)
    parser.add_argument("--max-eval", type=int, default=0, help="0 evaluates all items")
    parser.add_argument("--audit-examples", type=int, default=6)
    parser.add_argument("--skip-exact-control", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Ignore compatible cached per-seed results")
    return parser.parse_args()


def load_blink_rows(args: argparse.Namespace) -> List[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise SystemExit("Missing dependency: install requirements.txt") from exc
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=args.split,
        revision=args.dataset_revision,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    if args.max_eval > 0:
        dataset = dataset.select(range(min(args.max_eval, len(dataset))))
    rows = list(dataset)
    if not rows:
        raise SystemExit("No BLINK examples loaded")
    return rows


def parse_label(answer: str) -> int:
    normalized = str(answer).strip().upper()
    if normalized in {"(A)", "A", "LEFT"}:
        return 0
    if normalized in {"(B)", "B", "RIGHT"}:
        return 1
    raise ValueError(f"Unexpected BLINK answer: {answer!r}")


def angle_grid(step: int, maximum: int) -> List[int]:
    if step <= 0 or maximum <= 0 or maximum > 180:
        raise ValueError("Require 0 < --angle-step and 0 < --max-angle <= 180")
    magnitudes = list(range(step, maximum + 1, step))
    if not magnitudes or magnitudes[-1] != maximum:
        raise ValueError("--max-angle must be divisible by --angle-step")
    # rotation_action contains the signed raw delta/180 feature.  Staying in
    # [-180, 180] avoids an action representation absent from flow training.
    return [-value for value in magnitudes] + magnitudes


def center_crop_resize(image: Any, size: int) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    side = min(height, width)
    y0, x0 = (height - side) // 2, (width - side) // 2
    tensor = torch.from_numpy(array[y0:y0 + side, x0:x0 + side].copy())
    tensor = tensor.permute(2, 0, 1).float().div_(255.0)
    return F.interpolate(
        tensor.unsqueeze(0), size=(size, size), mode="bilinear",
        align_corners=False, antialias=True,
    )[0].clamp(0.0, 1.0)


def model_channels(rgb: torch.Tensor, channels: int) -> torch.Tensor:
    if channels == 3:
        return rgb
    if channels == 1:
        weights = rgb.new_tensor([0.2989, 0.5870, 0.1140])[:, None, None]
        return (rgb * weights).sum(dim=0, keepdim=True)
    raise ValueError(f"Unsupported frozen flow channel count: {channels}")


def prepared_pairs(
    rows: Sequence[dict], size: int, channels: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    source_rgb = torch.stack([center_crop_resize(row["image_1"], size) for row in rows])
    target_rgb = torch.stack([center_crop_resize(row["image_2"], size) for row in rows])
    source = torch.stack([model_channels(image, channels) for image in source_rgb]).to(device)
    target = torch.stack([model_channels(image, channels) for image in target_rgb]).to(device)
    return source, target


def dataset_fingerprint(rows: Sequence[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["idx"]).encode("utf-8"))
        digest.update(b"\0" + str(row["answer"]).encode("utf-8") + b"\0")
        for key in ("image_1", "image_2"):
            image = row[key].convert("RGB")
            digest.update(f"{image.width}x{image.height}".encode("ascii"))
            digest.update(image.tobytes())
    return digest.hexdigest()


def load_model(path: Path, device: torch.device) -> Tuple[TrajectoryFlowField, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TrajectoryFlowField(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if model.dynamics_mode != "transport":
        raise ValueError("BLINK evaluation requires a transport flow checkpoint")
    return model, checkpoint


def classification_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, Any]:
    """Score the fixed rule: positive margin means answer B/right."""
    y = np.asarray(labels, dtype=np.int64)
    margins = np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or y.shape != margins.shape or not np.all(np.isfinite(margins)):
        raise ValueError("Expected non-empty aligned labels and finite scores")
    predictions = (margins >= 0.0).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (y == 1)))
    tn = int(np.sum((predictions == 0) & (y == 0)))
    fp = int(np.sum((predictions == 1) & (y == 0)))
    fn = int(np.sum((predictions == 0) & (y == 1)))
    correct = int(np.sum(predictions == y))
    low, high = wilson_accuracy_ci(correct, len(y))
    chance_p = sum(
        math.comb(len(y), value) for value in range(correct, len(y) + 1)
    ) / (2 ** len(y))
    right_recall = tp / (tp + fn) if tp + fn else None
    left_recall = tn / (tn + fp) if tn + fp else None
    auc = binary_auc(y.tolist(), margins.tolist())
    balanced_accuracy = (
        0.5 * (right_recall + left_recall)
        if right_recall is not None and left_recall is not None else None
    )
    return {
        "n": int(len(y)),
        "accuracy": correct / len(y),
        "accuracy_ci95_low": low,
        "accuracy_ci95_high": high,
        "accuracy_ci_method": "wilson_test_items",
        "chance_accuracy_p_value_one_sided_exact_binomial": chance_p,
        "balanced_accuracy": balanced_accuracy,
        "auc": float(auc) if math.isfinite(auc) else None,
        "left_recall": left_recall,
        "right_recall": right_recall,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "decision_rule": "right/B iff left-camera error minus right-camera error >= 0",
        "mean_margin": float(np.mean(margins)),
        "mean_abs_margin": float(np.mean(np.abs(margins))),
        "mean_left_error": None,
        "mean_right_error": None,
    }


def predictions_from_errors(
    rows: Sequence[dict], errors: torch.Tensor, angles: Sequence[int]
) -> List[dict]:
    negative = [index for index, angle in enumerate(angles) if angle < 0]
    positive = [index for index, angle in enumerate(angles) if angle > 0]
    predictions: List[dict] = []
    for row, item_errors in zip(rows, errors):
        # The checkpoint action rotates an object, whereas BLINK asks for camera
        # orbit. A clockwise/left camera orbit induces the opposite apparent
        # object motion (counter-clockwise/positive), and vice versa.
        left_local = int(torch.argmin(item_errors[positive]))
        right_local = int(torch.argmin(item_errors[negative]))
        left_index, right_index = positive[left_local], negative[right_local]
        left_error = float(item_errors[left_index])
        right_error = float(item_errors[right_index])
        score = left_error - right_error
        predictions.append({
            "sample_id": str(row["idx"]),
            "label": parse_label(row["answer"]),
            "prediction": int(score >= 0.0),
            "score": score,
            "left_camera_error": left_error,
            "right_camera_error": right_error,
            "best_left_camera_object_angle_degrees": int(angles[left_index]),
            "best_right_camera_object_angle_degrees": int(angles[right_index]),
        })
    return predictions


def summarize_predictions(predictions: Sequence[dict]) -> dict:
    metrics = classification_metrics(
        [int(row["label"]) for row in predictions],
        [float(row["score"]) for row in predictions],
    )
    metrics["mean_left_error"] = mean(float(row["left_camera_error"]) for row in predictions)
    metrics["mean_right_error"] = mean(float(row["right_camera_error"]) for row in predictions)
    selected_angles = [
        int(row[
            "best_right_camera_object_angle_degrees"
            if row["prediction"] else "best_left_camera_object_angle_degrees"
        ])
        for row in predictions
    ]
    smallest_magnitude = min(abs(angle) for angle in selected_angles)
    metrics["selected_angle_diagnostics"] = {
        "mean_absolute_degrees": mean(abs(angle) for angle in selected_angles),
        "median_absolute_degrees": float(np.median(np.abs(selected_angles))),
        "smallest_grid_magnitude_degrees": smallest_magnitude,
        "fraction_at_smallest_grid_magnitude": sum(
            abs(angle) == smallest_magnitude for angle in selected_angles
        ) / len(selected_angles),
        "counts": {
            str(angle): selected_angles.count(angle) for angle in sorted(set(selected_angles))
        },
    }
    return metrics


def paired_exact_comparison(candidate: Sequence[dict], exact: Sequence[dict]) -> dict:
    """Paired accuracy delta and two-sided exact McNemar test."""
    if [row["sample_id"] for row in candidate] != [row["sample_id"] for row in exact]:
        raise ValueError("Paired predictions must contain aligned sample IDs")
    candidate_correct = [row["prediction"] == row["label"] for row in candidate]
    exact_correct = [row["prediction"] == row["label"] for row in exact]
    wins = sum(left and not right for left, right in zip(candidate_correct, exact_correct))
    losses = sum(not left and right for left, right in zip(candidate_correct, exact_correct))
    discordant = wins + losses
    if discordant:
        tail = sum(math.comb(discordant, value) for value in range(min(wins, losses) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    else:
        p_value = 1.0
    return {
        "candidate_correct_exact_wrong": wins,
        "candidate_wrong_exact_correct": losses,
        "discordant_items": discordant,
        "accuracy_delta": (sum(candidate_correct) - sum(exact_correct)) / len(candidate_correct),
        "p_value_two_sided_exact_mcnemar": p_value,
    }


@torch.no_grad()
def evaluate_flow(
    model: TrajectoryFlowField,
    rows: Sequence[dict],
    *,
    angles: Sequence[int],
    image_size: int,
    steps: int,
    item_batch_size: int,
    hypothesis_batch_size: int,
    device: torch.device,
) -> Tuple[dict, List[dict]]:
    all_errors: List[torch.Tensor] = []
    angle_tensor = torch.tensor(angles, device=device, dtype=torch.float32)
    for offset in range(0, len(rows), item_batch_size):
        items = rows[offset:offset + item_batch_size]
        source, target = prepared_pairs(items, image_size, model.state_channels, device)
        flat_source = source[:, None].expand(-1, len(angles), -1, -1, -1).reshape(
            -1, model.state_channels, image_size, image_size
        )
        flat_target = target[:, None].expand_as(
            source[:, None].expand(-1, len(angles), -1, -1, -1)
        ).reshape_as(flat_source)
        flat_angles = angle_tensor[None].expand(len(items), -1).reshape(-1)
        chunk_errors = []
        for start in range(0, len(flat_source), hypothesis_batch_size):
            stop = min(start + hypothesis_batch_size, len(flat_source))
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
        all_errors.append(torch.cat(chunk_errors).reshape(len(items), len(angles)))
    predictions = predictions_from_errors(rows, torch.cat(all_errors), angles)
    return summarize_predictions(predictions), predictions


@torch.no_grad()
def evaluate_exact_rotation(
    rows: Sequence[dict],
    *,
    angles: Sequence[int],
    image_size: int,
    item_batch_size: int,
    device: torch.device,
) -> Tuple[dict, List[dict]]:
    """Test the planar-rotation hypothesis class without learned-flow error."""
    all_errors: List[torch.Tensor] = []
    angle_tensor = torch.tensor(angles, device=device, dtype=torch.float32)
    for offset in range(0, len(rows), item_batch_size):
        items = rows[offset:offset + item_batch_size]
        source, target = prepared_pairs(items, image_size, 3, device)
        flat_source = source[:, None].expand(-1, len(angles), -1, -1, -1).reshape(
            -1, 3, image_size, image_size
        )
        flat_target = target[:, None].expand(-1, len(angles), -1, -1, -1).reshape_as(flat_source)
        flat_angles = angle_tensor[None].expand(len(items), -1).reshape(-1)
        prediction = rotate_tensor(flat_source, flat_angles, pad_to_diag=False)
        all_errors.append(
            ((prediction - flat_target) ** 2).mean(dim=(1, 2, 3)).cpu().reshape(len(items), len(angles))
        )
    predictions = predictions_from_errors(rows, torch.cat(all_errors), angles)
    return summarize_predictions(predictions), predictions


def configure_matplotlib(output: Path) -> None:
    cache = output / ".mpl-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def show_tensor(axis, tensor: torch.Tensor, title: str, *, error: bool = False) -> None:
    image = tensor.detach().cpu().permute(1, 2, 0)
    if image.shape[-1] == 1:
        axis.imshow(image.squeeze(-1), cmap="magma" if error else "gray", vmin=0, vmax=1)
    elif error:
        axis.imshow(image.mean(dim=-1), cmap="magma", vmin=0, vmax=1)
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
    image_size: int,
    steps: int,
    examples: int,
    device: torch.device,
    destination: Path,
) -> None:
    if examples <= 0:
        return
    configure_matplotlib(destination.parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mistakes = [index for index, row in enumerate(predictions) if row["prediction"] != row["label"]]
    correct = [index for index, row in enumerate(predictions) if row["prediction"] == row["label"]]
    selected = (mistakes[: max(1, examples // 2)] + correct)[:examples]
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    figure, axes = plt.subplots(len(selected), 9, figsize=(18, 2.25 * len(selected)), squeeze=False)
    for plot_row, item_index in enumerate(selected):
        item, result = rows[item_index], predictions[item_index]
        source, target = prepared_pairs([item], image_size, model.state_channels, device)
        left_angle = torch.tensor(
            [result["best_left_camera_object_angle_degrees"]], device=device, dtype=source.dtype
        )
        right_angle = torch.tensor(
            [result["best_right_camera_object_angle_degrees"]], device=device, dtype=source.dtype
        )
        left_endpoint = integrate_deformation_times(
            model, source, source, rotation_action(left_angle), [1.0],
            max_step=1.0 / steps, clamp=(0.0, 1.0),
        )[0]
        right_endpoint = integrate_deformation_times(
            model, source, source, rotation_action(right_angle), [1.0],
            max_step=1.0 / steps, clamp=(0.0, 1.0),
        )[0]
        selected_angle = right_angle if result["prediction"] else left_angle
        trajectory = integrate_deformation_times(
            model, source, source, rotation_action(selected_angle), times,
            max_step=1.0 / steps, clamp=(0.0, 1.0),
        )
        truth = "right" if result["label"] else "left"
        predicted = "right" if result["prediction"] else "left"
        show_tensor(axes[plot_row, 0], source[0], f"source\ntrue={truth}")
        show_tensor(axes[plot_row, 1], target[0], f"target\npred={predicted}")
        show_tensor(axes[plot_row, 2], left_endpoint[0], f"best left\n{int(left_angle.item())}°")
        show_tensor(axes[plot_row, 3], right_endpoint[0], f"best right\n{int(right_angle.item())}°")
        for time_index, frame in enumerate(trajectory):
            show_tensor(axes[plot_row, 4 + time_index], frame[0], f"selected t={times[time_index]:g}")
    figure.suptitle("Frozen 2-D rotation flow on BLINK Multi-view (qualitative; no target fitting)")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def aggregate(per_seed: Sequence[dict], exact_control: dict | None, output_dir: Path) -> dict:
    metric_names = ("accuracy", "balanced_accuracy", "auc", "left_recall", "right_recall")
    aggregates: List[dict] = []
    ensembles: List[dict] = []
    for source_model in sorted({row["source_model"] for row in per_seed}):
        relevant = sorted(
            (row for row in per_seed if row["source_model"] == source_model),
            key=lambda row: row["seed"],
        )
        for metric in metric_names:
            values = [float(row["metrics"][metric]) for row in relevant]
            low, high = mean_t_ci(values)
            aggregates.append({
                "source_model": source_model,
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
        labels = [int(item["label"]) for item in relevant[0]["predictions"]]
        scores = [
            mean(float(run["predictions"][index]["score"]) for run in relevant)
            for index in range(len(labels))
        ]
        ensemble_predictions = []
        for index, (label, score) in enumerate(zip(labels, scores)):
            ensemble_predictions.append({
                "sample_id": relevant[0]["predictions"][index]["sample_id"],
                "label": label,
                "prediction": int(score >= 0.0),
                "score": score,
            })
        ensembles.append({
            "source_model": source_model,
            "method": "mean signed reconstruction margin across frozen seeds",
            "metrics": classification_metrics(labels, scores),
            "predictions": ensemble_predictions,
        })

    paired_comparisons = []
    if exact_control:
        exact_predictions = exact_control["predictions"]
        for row in per_seed:
            paired_comparisons.append({
                "source_model": row["source_model"],
                "seed": row["seed"],
                "method": "frozen flow versus exact in-plane rotation control",
                **paired_exact_comparison(row["predictions"], exact_predictions),
            })
        for row in ensembles:
            paired_comparisons.append({
                "source_model": row["source_model"],
                "seed": "ensemble",
                "method": "mean-margin seed ensemble versus exact in-plane rotation control",
                **paired_exact_comparison(row["predictions"], exact_predictions),
            })

    payload = {
        "schema_version": 1,
        "protocol": {
            "benchmark": "BLINK Multi-view Reasoning validation split",
            "checkpoint_use": "frozen; no BLINK training, calibration, threshold selection, or checkpoint selection",
            "development_disclosure": "the inverse camera/object sign convention was finalized after a two-item evaluator smoke; the full validation result is exploratory and requires hidden-test confirmation",
            "hypotheses": "object motion is inverse to camera orbit: positive object angles mean clockwise/left camera/A; negative object angles mean counter-clockwise/right camera/B",
            "decision": "right/B iff minimum left-camera error minus minimum right-camera error is nonnegative",
            "preprocessing": "center square crop, antialiased resize, RGB for colored flow and fixed luminance for Tetris flow",
            "exact_control": "same fixed decision using an exact image-plane rotation operator; not a learned baseline",
            "limitations": [
                "BLINK depicts 3-D camera orbit and parallax, whereas the frozen flows learned 2-D in-plane rotation.",
                "The fixed reconstruction metric is sensitive to real-video lighting, background, and correspondence changes.",
                "The validation split has 133 labeled pairs; official hidden-test numbers are contextual, not paired estimates.",
            ],
        },
        "external_reference_values": [
            {"method": "random", "split": "validation", "accuracy": 0.5},
            {"method": "human", "split": "validation", "accuracy": 0.9248},
            {"method": "GPT-4V direct", "split": "validation", "accuracy": 0.5865},
            {"method": "GPT-4V concatenated images", "split": "validation", "accuracy": 0.5789},
            {"method": "Gemini Pro direct", "split": "validation", "accuracy": 0.4135},
            {"method": "LoFTR specialist", "split": "paper dev/test table; not local validation", "accuracy": 0.9022},
        ],
        "per_seed": [
            {key: value for key, value in row.items() if key != "predictions"} for row in per_seed
        ],
        "aggregates": aggregates,
        "ensembles": ensembles,
        "exact_in_plane_rotation_control": exact_control,
        "paired_comparisons_to_exact_control": paired_comparisons,
    }
    write_json(output_dir / "zero_shot_results.json", payload)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("source_model", "metric", "n_seeds", "mean", "std", "ci95_low", "ci95_high")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in aggregates)

    lines = [
        "# BLINK Multi-view frozen-flow transfer",
        "",
        "No BLINK example was used to train, calibrate, threshold, or select a checkpoint. The inverse camera/object sign convention was finalized after a two-item evaluator smoke, so this public-validation result is exploratory and requires confirmation on the hidden test or another locked split.",
        "",
        "## Frozen-flow results",
        "",
        "| 2-D source | Metric | Mean over seeds | 95% t CI |",
        "|---|---|---:|---:|",
    ]
    for row in aggregates:
        ci = "undefined (n<2)" if row["ci95_low"] is None else f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
        lines.append(f"| {row['source_model']} | {row['metric']} | {row['mean']:.3f} | {ci} |")
    lines.extend([
        "",
        "## Seed ensembles and controls",
        "",
        "| Method | Accuracy (Wilson 95% CI) | Balanced accuracy | AUC | Exact p vs 50% |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in ensembles:
        metrics = row["metrics"]
        lines.append(
            f"| {row['source_model']} seed ensemble | {metrics['accuracy']:.3f} "
            f"[{metrics['accuracy_ci95_low']:.3f}, {metrics['accuracy_ci95_high']:.3f}] | "
            f"{metrics['balanced_accuracy']:.3f} | {metrics['auc']:.3f} | "
            f"{metrics['chance_accuracy_p_value_one_sided_exact_binomial']:.3g} |"
        )
    if exact_control:
        metrics = exact_control["metrics"]
        lines.append(
            f"| exact in-plane rotation control | {metrics['accuracy']:.3f} "
            f"[{metrics['accuracy_ci95_low']:.3f}, {metrics['accuracy_ci95_high']:.3f}] | "
            f"{metrics['balanced_accuracy']:.3f} | {metrics['auc']:.3f} | "
            f"{metrics['chance_accuracy_p_value_one_sided_exact_binomial']:.3g} |"
        )
    lines.extend([
        "| random | 0.500 | 0.500 | 0.500 | 1.000 |",
        "",
        "Published validation context from the official BLINK paper: human 92.48%, GPT-4V direct 58.65%, GPT-4V with concatenated images 57.89%, and Gemini Pro direct 41.35%. The paper also reports 90.22% for its pretrained LoFTR specialist on its dev/test table (a different split). These are contextual reference values, not confidence-interval-matched comparisons.",
        "",
        "## Paired comparison with exact planar geometry",
        "",
        "| Flow | Seed | Accuracy delta | Flow wins | Flow losses | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in paired_comparisons:
        lines.append(
            f"| {row['source_model']} | {row['seed']} | {row['accuracy_delta']:+.3f} | "
            f"{row['candidate_correct_exact_wrong']} | {row['candidate_wrong_exact_correct']} | "
            f"{row['p_value_two_sided_exact_mcnemar']:.3g} |"
        )
    lines.extend([
        "",
        "None of the frozen-flow versus exact-control accuracy differences reaches p < 0.05. The result therefore supports signed synthetic-to-real transfer, but does not yet establish that the learned flow contributes accuracy beyond the predeclared planar-rotation hypothesis class.",
        "",
        "## Selected-angle diagnostics",
        "",
        "| Method | Mean absolute angle | Median | Fraction at smallest grid angle |",
        "|---|---:|---:|---:|",
    ])
    for row in per_seed:
        diagnostics = row["metrics"]["selected_angle_diagnostics"]
        lines.append(
            f"| {row['source_model']} seed {row['seed']} | "
            f"{diagnostics['mean_absolute_degrees']:.1f}° | "
            f"{diagnostics['median_absolute_degrees']:.1f}° | "
            f"{diagnostics['fraction_at_smallest_grid_magnitude']:.3f} |"
        )
    if exact_control:
        diagnostics = exact_control["metrics"]["selected_angle_diagnostics"]
        lines.append(
            f"| exact in-plane control | {diagnostics['mean_absolute_degrees']:.1f}° | "
            f"{diagnostics['median_absolute_degrees']:.1f}° | "
            f"{diagnostics['fraction_at_smallest_grid_magnitude']:.3f} |"
        )
    lines.extend([
        "",
        "## What this tests",
        "",
        "A result above chance would support transfer of a signed visual-transformation prior from synthetic 2-D rotations to real camera motion. Failure of both the learned flows and the exact control means the in-plane rotation hypothesis class is insufficient for 3-D viewpoint change; failure of only the learned flows localizes the problem to flow learning or rendering quality.",
        "",
        "The audit PNGs are intentionally kept local because BLINK aggregates images from external sources. Numerical predictions contain IDs and scores only.",
        "",
        "Sources: [BLINK paper](https://arxiv.org/abs/2404.12390), [official repository](https://github.com/zeyofu/BLINK_Benchmark), [dataset](https://huggingface.co/datasets/BLINK-Benchmark/BLINK).",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def compatible_cached(path: Path, signature: str, checkpoint_sha256: str) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("protocol_signature") != signature:
        return None
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        return None
    if not isinstance(payload.get("predictions"), list):
        return None
    # Derived diagnostics can grow without invalidating expensive predictions
    # or changing the predeclared benchmark protocol.
    payload["metrics"] = summarize_predictions(payload["predictions"])
    return payload


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.item_batch_size <= 0 or args.hypothesis_batch_size <= 0:
        raise SystemExit("Image and batch sizes must be positive")
    angles = angle_grid(args.angle_step, args.max_angle)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_blink_rows(args)
    labels = [parse_label(row["answer"]) for row in rows]
    fingerprint = dataset_fingerprint(rows)
    protocol_core = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "dataset_fingerprint_sha256": fingerprint,
        "image_size": args.image_size,
        "angles_degrees": angles,
        "preprocessing": "center_square_crop_antialiased_resize",
        "decision": "right camera iff min_positive_object_angle_error - min_negative_object_angle_error >= 0",
    }
    signature = hashlib.sha256(
        json.dumps(protocol_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    write_json(args.output_dir / "resolved_protocol.json", {
        "schema_version": 1,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "protocol": protocol_core,
        "protocol_signature": signature,
        "data": {
            "n_items": len(rows),
            "label_counts": {"left_A": labels.count(0), "right_B": labels.count(1)},
            "raw_images_redistributed": False,
            "evaluation_status": "exploratory public validation; sign convention finalized after a two-item smoke",
        },
    })

    exact_control = None
    exact_path = args.output_dir / "exact_in_plane_rotation_control.json"
    if not args.skip_exact_control:
        if exact_path.exists() and not args.rerun:
            candidate = json.loads(exact_path.read_text(encoding="utf-8"))
            if candidate.get("protocol_signature") == signature:
                candidate["metrics"] = summarize_predictions(candidate["predictions"])
                exact_control = candidate
                write_json(exact_path, exact_control)
                print("Reusing exact in-plane rotation control", flush=True)
        if exact_control is None:
            started = time.perf_counter()
            metrics, predictions = evaluate_exact_rotation(
                rows, angles=angles, image_size=args.image_size,
                item_batch_size=args.item_batch_size, device=device,
            )
            exact_control = {
                "method": "exact in-plane rotation hypothesis-class control",
                "protocol_signature": signature,
                "elapsed_seconds": time.perf_counter() - started,
                "metrics": metrics,
                "predictions": predictions,
            }
            write_json(exact_path, exact_control)
            print(json.dumps({"exact_control": metrics}, sort_keys=True), flush=True)

    per_seed = []
    for source_model in args.source_models:
        for seed in args.seeds:
            checkpoint_path = args.run_root / source_model / f"seed{seed}" / "best_checkpoint.pt"
            if not checkpoint_path.exists():
                raise SystemExit(f"Missing frozen checkpoint: {checkpoint_path}")
            checkpoint_sha256 = sha256_file(checkpoint_path)
            result_path = args.output_dir / "per_seed" / f"{source_model}_seed{seed}.json"
            cached = None if args.rerun else compatible_cached(result_path, signature, checkpoint_sha256)
            if cached is not None:
                per_seed.append(cached)
                write_json(result_path, cached)
                print(f"Reusing {source_model} seed {seed}", flush=True)
                aggregate(per_seed, exact_control, args.output_dir)
                continue
            model, checkpoint = load_model(checkpoint_path, device)
            steps = int(checkpoint["training_arguments"]["integration_steps"])
            started = time.perf_counter()
            metrics, predictions = evaluate_flow(
                model, rows, angles=angles, image_size=args.image_size, steps=steps,
                item_batch_size=args.item_batch_size,
                hypothesis_batch_size=args.hypothesis_batch_size, device=device,
            )
            result = {
                "source_model": source_model,
                "seed": seed,
                "protocol_signature": signature,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "best_epoch": int(checkpoint["best_epoch"]),
                "integration_steps": steps,
                "elapsed_seconds": time.perf_counter() - started,
                "metrics": metrics,
                "predictions": predictions,
            }
            per_seed.append(result)
            write_json(result_path, result)
            if seed == args.seeds[0]:
                make_audit(
                    model, rows, predictions, image_size=args.image_size, steps=steps,
                    examples=args.audit_examples, device=device,
                    destination=args.output_dir / "audits" / f"{source_model}_seed{seed}_trajectories.png",
                )
            aggregate(per_seed, exact_control, args.output_dir)
            print(json.dumps({
                "source_model": source_model,
                "seed": seed,
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": metrics,
            }, sort_keys=True), flush=True)
    aggregate(per_seed, exact_control, args.output_dir)


if __name__ == "__main__":
    main()

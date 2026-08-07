#!/usr/bin/env python3
"""Post-hoc renderer and temporal audit of frozen NeurReps flow checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from utils.fot.metrics import mean_t_ci
from utils.fot.reproducibility import collect_run_metadata, write_json
from utils.fot.trajectory_datasets import (
    MazeTrajectoryDataset,
    RotationTrajectoryDataset,
    render_rotation_frames,
    render_rotation_state,
)
from utils.fot.trajectory_flow import (
    TrajectoryFlowField,
    integrate_deformation_times,
    integrate_trajectory,
    rotation_action,
)
from utils.fot.trajectory_metrics import maze_temporal_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "models/runs/neurreps_flow_v1/overnight")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/neurreps_flow_v1/posthoc_v2")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--tasks", nargs="+", choices=("tetris", "colored", "maze"),
                        default=["tetris", "colored", "maze"])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--validation-samples", type=int, default=400)
    parser.add_argument("--audit-examples", type=int, default=4)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> Tuple[TrajectoryFlowField, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TrajectoryFlowField(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def validation_dataset(task: str, seed: int, checkpoint: dict, maximum: int):
    arguments = checkpoint["training_arguments"]
    image_size = int(arguments.get("image_size", 64))
    validation_seed = int(seed) + 10_000_019
    if task == "maze":
        return MazeTrajectoryDataset(
            n_samples=maximum,
            maze_cells=int(arguments.get("maze_cells", 9)),
            image_size=image_size,
            trajectory_steps=int(arguments.get("integration_steps", 8)),
            seed=validation_seed,
        )
    if task == "tetris":
        return RotationTrajectoryDataset(
            task="tetris", n_samples=maximum, image_size=image_size, seed=validation_seed,
            shape_keys=("F", "P"), split_label="heldout_shapes_FP",
        )
    return RotationTrajectoryDataset(
        task="colored", n_samples=maximum, image_size=image_size, seed=validation_seed,
        split_label="independent_procedural_validation",
    )


def total_variation(image: torch.Tensor) -> torch.Tensor:
    horizontal = (image[..., :, 1:] - image[..., :, :-1]).abs().mean(dim=(1, 2, 3))
    vertical = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(dim=(1, 2, 3))
    return horizontal + vertical


def rotation_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    mse = ((prediction - target) ** 2).mean(dim=(1, 2, 3))
    predicted_mask = prediction.amax(dim=1) >= 0.1
    target_mask = target.amax(dim=1) >= 0.1
    intersection = (predicted_mask & target_mask).sum(dim=(1, 2)).float()
    union = (predicted_mask | target_mask).sum(dim=(1, 2)).float().clamp_min(1.0)
    predicted_area = predicted_mask.sum(dim=(1, 2)).float()
    target_area = target_mask.sum(dim=(1, 2)).float().clamp_min(1.0)
    return {
        "endpoint_mse": mse,
        "endpoint_psnr_db": -10.0 * torch.log10(mse.clamp_min(1e-10)),
        "silhouette_iou": intersection / union,
        "sharpness_ratio": total_variation(prediction) / total_variation(target).clamp_min(1e-8),
        "foreground_area_ratio": predicted_area / target_area,
    }


def add_values(storage: Dict[str, List[float]], values: Dict[str, torch.Tensor]) -> None:
    for key, tensor in values.items():
        storage.setdefault(key, []).extend(float(value) for value in tensor.detach().cpu().reshape(-1))


@torch.no_grad()
def evaluate_rotation(model, dataset, steps: int, batch_size: int, device: torch.device) -> Tuple[dict, dict]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    repeated_values: Dict[str, List[float]] = {}
    single_values: Dict[str, List[float]] = {}
    grid_times = torch.linspace(0.0, 1.0, steps + 1, device=device)
    arbitrary_time = 0.37
    for base, start, delta in loader:
        base, start, delta = base.to(device), start.to(device), delta.to(device)
        source = render_rotation_state(base, start, delta, torch.zeros_like(delta))
        condition = source
        action = rotation_action(delta)
        repeated_endpoint, repeated_frames = integrate_trajectory(
            model, source, condition, action, steps=steps, clamp=(0.0, 1.0), return_frames=True
        )
        single_frames = integrate_deformation_times(
            model, source, condition, action, grid_times, max_step=1.0 / steps,
            clamp=(0.0, 1.0), return_maps=False,
        )
        truth_frames = render_rotation_frames(base, start, delta, grid_times)
        repeated = rotation_metrics(repeated_endpoint, truth_frames[:, -1])
        single = rotation_metrics(single_frames[-1], truth_frames[:, -1])
        repeated["trajectory_mse"] = torch.stack([
            ((frame - truth_frames[:, index]) ** 2).mean(dim=(1, 2, 3))
            for index, frame in enumerate(repeated_frames)
        ]).mean(dim=0)
        single["trajectory_mse"] = torch.stack([
            ((frame - truth_frames[:, index]) ** 2).mean(dim=(1, 2, 3))
            for index, frame in enumerate(single_frames)
        ]).mean(dim=0)
        arbitrary_prediction = integrate_deformation_times(
            model, source, condition, action, [arbitrary_time], max_step=1.0 / steps,
            clamp=(0.0, 1.0),
        )[0]
        arbitrary_truth = render_rotation_state(
            base, start, delta, torch.full_like(delta, arbitrary_time)
        )
        single[f"arbitrary_t_{arbitrary_time:.2f}_mse"] = (
            (arbitrary_prediction - arbitrary_truth) ** 2
        ).mean(dim=(1, 2, 3))
        add_values(repeated_values, repeated)
        add_values(single_values, single)
    summarize = lambda values: {key: mean(items) for key, items in values.items()}
    return summarize(repeated_values), summarize(single_values)


@torch.no_grad()
def evaluate_maze(model, dataset, steps: int, batch_size: int, device: torch.device) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    values: Dict[str, List[float]] = {}
    for condition, truth_frames in loader:
        condition, truth_frames = condition.to(device), truth_frames.to(device)
        action = torch.zeros(condition.shape[0], 3, device=device)
        endpoint, predicted_frames = integrate_trajectory(
            model, truth_frames[:, 0], condition, action, steps=steps,
            clamp=(0.0, 1.0), return_frames=True,
        )
        target = truth_frames[:, -1]
        binary, truth = endpoint >= 0.5, target >= 0.5
        intersection = (binary & truth).sum(dim=(1, 2, 3)).float()
        union = (binary | truth).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
        predicted_count = binary.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
        truth_count = truth.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
        wall, goal = condition[:, :1] >= 0.5, condition[:, 2:3] >= 0.5
        metrics = {
            "endpoint_iou": intersection / union,
            "endpoint_mse": ((endpoint - target) ** 2).mean(dim=(1, 2, 3)),
            "path_precision": intersection / predicted_count,
            "path_recall": intersection / truth_count,
            "goal_reached": (binary & goal).any(dim=3).any(dim=2).any(dim=1).float(),
            "obstacle_violation_rate": (binary & wall).sum(dim=(1, 2, 3)).float() / predicted_count,
            "trajectory_mse": torch.stack([
                ((frame - truth_frames[:, index]) ** 2).mean(dim=(1, 2, 3))
                for index, frame in enumerate(predicted_frames)
            ]).mean(dim=0),
            **maze_temporal_metrics(predicted_frames, truth_frames),
        }
        add_values(values, metrics)
    return {key: mean(items) for key, items in values.items()}


def configure_matplotlib(output: Path) -> None:
    cache = output / ".mpl-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


@torch.no_grad()
def rotation_audit(model, dataset, steps: int, device: torch.device, destination: Path, examples: int) -> None:
    configure_matplotlib(destination.parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit_times = torch.tensor([0.0, 0.25, 0.5, 1.0], device=device)
    columns = 1 + 4 * len(audit_times)
    figure, axes = plt.subplots(examples, columns, figsize=(1.8 * columns, 1.9 * examples), squeeze=False)
    for row in range(examples):
        base, start, delta = (value.unsqueeze(0).to(device) for value in dataset[row])
        source = render_rotation_state(base, start, delta, torch.zeros_like(delta))
        action = rotation_action(delta)
        _, repeated_all = integrate_trajectory(
            model, source, source, action, steps=steps, clamp=(0.0, 1.0), return_frames=True
        )
        single = integrate_deformation_times(
            model, source, source, action, audit_times, max_step=1.0 / steps, clamp=(0.0, 1.0)
        )
        truth = render_rotation_frames(base, start, delta, audit_times)[0]
        source_image = source[0].cpu().permute(1, 2, 0)
        axes[row, 0].imshow(source_image.squeeze(-1), cmap="gray", vmin=0, vmax=1) if source_image.shape[-1] == 1 else axes[row, 0].imshow(source_image)
        axes[row, 0].set_title(f"source\nΔ={float(delta.item()):.1f}°")
        for time_index, time_value in enumerate(audit_times.tolist()):
            repeated_step = round(time_value * steps)
            panels = (
                ("truth", truth[time_index]),
                ("repeated", repeated_all[repeated_step][0]),
                ("single", single[time_index][0]),
                ("single error", (single[time_index][0] - truth[time_index]).abs()),
            )
            for panel_index, (label, tensor) in enumerate(panels):
                axis = axes[row, 1 + 4 * time_index + panel_index]
                image = tensor.cpu().permute(1, 2, 0)
                axis.imshow(image.squeeze(-1), cmap="gray", vmin=0, vmax=1) if image.shape[-1] == 1 else axis.imshow(image.clamp(0, 1))
                axis.set_title(f"{label}\nt={time_value:g}")
        for axis in axes[row]: axis.axis("off")
    figure.suptitle(f"{dataset.task}: repeated resampling versus single-source rendering")
    figure.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(figure)


@torch.no_grad()
def maze_timing_audit(model, dataset, steps: int, device: torch.device, destination: Path, examples: int) -> None:
    configure_matplotlib(destination.parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(examples, 5, figsize=(11, 2.2 * examples), squeeze=False)
    for row in range(examples):
        condition, truth = (value.unsqueeze(0).to(device) for value in dataset[row])
        _, predicted = integrate_trajectory(
            model, truth[:, 0], condition, torch.zeros(1, 3, device=device),
            steps=steps, clamp=(0.0, 1.0), return_frames=True,
        )
        prediction = torch.stack(predicted, dim=1)
        step_numbers = torch.arange(steps + 1, device=device).view(1, steps + 1, 1, 1, 1)
        never = torch.full_like(prediction, steps + 1, dtype=torch.long)
        true_time = torch.where(truth >= 0.5, step_numbers, never).amin(dim=1)[0, 0].float() / steps
        predicted_time = torch.where(prediction >= 0.5, step_numbers, never).amin(dim=1)[0, 0].float() / steps
        final_path = truth[0, -1, 0] >= 0.5
        timing_error = (predicted_time - true_time).abs() * final_path
        future_intensity = torch.stack([
            predicted[step][0, 0] * ((truth[0, -1, 0] >= 0.5) & (truth[0, step, 0] < 0.5))
            for step in range(1, steps)
        ]).amax(dim=0)
        condition_image = torch.stack([
            0.55 * condition[0, 0] + condition[0, 2],
            0.55 * condition[0, 0] + condition[0, 1],
            0.55 * condition[0, 0],
        ], dim=-1).clamp(0, 1)
        panels = (
            ("maze", condition_image.cpu(), None),
            ("true activation t", true_time.cpu(), "viridis"),
            ("pred activation t", predicted_time.cpu(), "viridis"),
            ("|timing error|", timing_error.cpu(), "magma"),
            ("max future intensity", future_intensity.cpu(), "magma"),
        )
        for column, (title, image, cmap) in enumerate(panels):
            axes[row, column].imshow(image, cmap=cmap, vmin=0, vmax=1)
            axes[row, column].set_title(title); axes[row, column].axis("off")
    figure.suptitle("Maze temporal-causality audit")
    figure.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(figure)


def bounded_metric(metric: str) -> bool:
    return any(token in metric for token in ("iou", "rate", "recall", "precision", "goal_reached"))


def aggregate(per_seed: List[dict], output: Path) -> None:
    groups: Dict[Tuple[str, str, str], List[float]] = {}
    for row in per_seed:
        for renderer, metrics in row["renderers"].items():
            for metric, value in metrics.items():
                groups.setdefault((row["task"], renderer, metric), []).append(float(value))
    aggregate_rows = []
    for (task, renderer, metric), values in sorted(groups.items()):
        if len(values) >= 2:
            low, high = mean_t_ci(values)
            display_low, display_high = low, high
            if bounded_metric(metric):
                display_low, display_high = max(0.0, low), min(1.0, high)
            elif metric.endswith("mse") or "error" in metric or "intensity" in metric:
                display_low = max(0.0, low)
        else:
            low = high = display_low = display_high = None
        aggregate_rows.append({
            "task": task, "renderer": renderer, "metric": metric, "n_seeds": len(values),
            "mean": mean(values), "std": stdev(values) if len(values) >= 2 else None,
            "ci95_low_raw": low, "ci95_high_raw": high,
            "ci95_low_display": display_low, "ci95_high_display": display_high, "values": values,
        })
    write_json(output / "posthoc_results.json", {
        "schema_version": 1,
        "protocol": "frozen checkpoints; no retraining or model selection",
        "per_seed": per_seed,
        "aggregates": aggregate_rows,
    })
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("task", "renderer", "metric", "n_seeds", "mean", "std", "ci95_low_display", "ci95_high_display")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in aggregate_rows)
    lines = ["# Frozen-flow post-hoc audit", "", "No checkpoint was retrained or selected using these results.", "",
             "| Task | Renderer | Metric | Mean | 95% CI |", "|---|---|---|---:|---:|"]
    for row in aggregate_rows:
        lines.append(
            f"| {row['task']} | {row['renderer']} | {row['metric']} | {row['mean']:.6f} | "
            + ("undefined (n<2) |" if row["ci95_low_display"] is None else
               f"[{row['ci95_low_display']:.6f}, {row['ci95_high_display']:.6f}] |")
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args(); device = torch.device(args.device); args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    per_seed = []
    for task in args.tasks:
        for seed in args.seeds:
            checkpoint_path = args.run_root / task / f"seed{seed}" / "best_checkpoint.pt"
            if not checkpoint_path.exists(): raise SystemExit(f"Missing frozen checkpoint: {checkpoint_path}")
            model, checkpoint = load_model(checkpoint_path, device)
            dataset = validation_dataset(task, seed, checkpoint, args.validation_samples)
            steps = int(checkpoint["training_arguments"]["integration_steps"])
            if task == "maze":
                renderers = {"additive_flow": evaluate_maze(model, dataset, steps, args.batch_size, device)}
                maze_timing_audit(model, dataset, steps, device,
                                  args.output_dir / "audits" / f"maze_seed{seed}_timing.png", args.audit_examples)
            else:
                repeated, single = evaluate_rotation(model, dataset, steps, args.batch_size, device)
                renderers = {"repeated_resampling": repeated, "single_source": single}
                rotation_audit(model, dataset, steps, device,
                               args.output_dir / "audits" / f"{task}_seed{seed}_renderer_comparison.png",
                               args.audit_examples)
            row = {"task": task, "seed": seed, "checkpoint": str(checkpoint_path.resolve()),
                   "best_epoch": checkpoint["best_epoch"], "renderers": renderers}
            per_seed.append(row)
            write_json(args.output_dir / "per_seed" / f"{task}_seed{seed}.json", row)
            aggregate(per_seed, args.output_dir)
            print(json.dumps(row, sort_keys=True), flush=True)
    aggregate(per_seed, args.output_dir)


if __name__ == "__main__":
    main()

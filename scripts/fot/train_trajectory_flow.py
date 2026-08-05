#!/usr/bin/env python3
"""Train and audit the PPO-free, trajectory-supervised spatial flow rebuild."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.fot.reproducibility import collect_run_metadata, write_json
from utils.fot.torch_utils import seed_worker, seeded_generator, set_seed
from utils.fot.trajectory_datasets import (
    MazeTrajectoryDataset,
    RotationTrajectoryDataset,
    render_rotation_frames,
    render_rotation_state,
)
from utils.fot.trajectory_flow import (
    TrajectoryFlowField,
    integrate_trajectory,
    rotation_action,
    soft_dice_loss,
    weighted_image_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("tetris", "colored", "maze"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--validation-samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--integration-steps", type=int, default=8)
    parser.add_argument("--rollout-batch", type=int, default=4)
    parser.add_argument("--rollout-every", type=int, default=2)
    parser.add_argument("--rollout-weight", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maze-cells", type=int, default=9)
    parser.add_argument("--audit-examples", type=int, default=4)
    parser.add_argument("--audit-every", type=int, default=5)
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--preliminary", action="store_true")
    return parser.parse_args()


def make_datasets(args: argparse.Namespace):
    common = {"n_samples": args.train_samples, "image_size": args.image_size, "seed": args.seed}
    valid_common = {
        "n_samples": args.validation_samples,
        "image_size": args.image_size,
        "seed": args.seed + 10_000_019,
    }
    if args.task == "maze":
        return (
            MazeTrajectoryDataset(
                **common, maze_cells=args.maze_cells, trajectory_steps=args.integration_steps
            ),
            MazeTrajectoryDataset(
                **valid_common, maze_cells=args.maze_cells, trajectory_steps=args.integration_steps
            ),
        )
    if args.task == "tetris":
        return (
            RotationTrajectoryDataset(
                task="tetris", **common, shape_keys=("J", "L", "S", "Z"), split_label="train_shapes_JLSZ"
            ),
            RotationTrajectoryDataset(
                task="tetris", **valid_common, shape_keys=("F", "P"), split_label="heldout_shapes_FP"
            ),
        )
    return RotationTrajectoryDataset(task=args.task, **common, split_label="procedural_train"), RotationTrajectoryDataset(
        task=args.task, **valid_common, split_label="independent_procedural_validation"
    )


def model_for_task(args: argparse.Namespace) -> TrajectoryFlowField:
    channels = 3 if args.task == "colored" else 1
    condition_channels = 3 if args.task == "maze" else channels
    return TrajectoryFlowField(
        state_channels=channels,
        condition_channels=condition_channels,
        action_dim=3,
        width=args.width,
        context_dim=args.context_dim,
        dynamics_mode="additive" if args.task == "maze" else "transport",
    )


def rotation_loss(
    model: TrajectoryFlowField,
    batch,
    args: argparse.Namespace,
    batch_index: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    base, start, delta = (item.to(device) for item in batch)
    count = base.shape[0]
    t0 = torch.rand(count, device=device)
    current = render_rotation_state(base, start, delta, t0)
    condition = render_rotation_state(base, start, delta, torch.zeros_like(t0))
    action = rotation_action(delta)
    predicted_field = model(current, condition, t0[:, None], action)
    # Dense tangent of the rotation group orbit in normalized image coordinates.
    # The sign maps output pixels back to their upstream source locations.
    coordinates = model._sampling_grid(current)
    x_coordinate = coordinates[..., 0]
    y_coordinate = coordinates[..., 1]
    angular_speed = torch.deg2rad(delta)[:, None, None]
    target_field = torch.stack(
        [-angular_speed * y_coordinate, angular_speed * x_coordinate], dim=1
    )
    velocity_loss = F.smooth_l1_loss(predicted_field, target_field, beta=0.1)
    total = velocity_loss
    values = {"velocity_loss": float(velocity_loss.detach().cpu()), "rollout_loss": 0.0}
    if batch_index % args.rollout_every == 0:
        limit = min(args.rollout_batch, count)
        # PyTorch MPS does not implement grid_sample backward. The exact dense
        # field target above is therefore the MPS-native training objective;
        # rollout images remain a strict no-gradient quality-control metric.
        with torch.no_grad():
            _, predicted_frames = integrate_trajectory(
                model,
                condition[:limit],
                condition[:limit],
                action[:limit],
                steps=args.integration_steps,
                clamp=(0.0, 1.0),
                return_frames=True,
            )
            times = torch.linspace(0.0, 1.0, args.integration_steps + 1, device=device)
            true_frames = render_rotation_frames(base[:limit], start[:limit], delta[:limit], times)
            rollout_loss = torch.stack(
                [weighted_image_loss(prediction, target) for prediction, target in zip(predicted_frames[1:], true_frames[:, 1:].unbind(1))]
            ).mean()
        values["rollout_loss"] = float(rollout_loss.detach().cpu())
    values["total_loss"] = float(total.detach().cpu())
    return total, values


def maze_loss(
    model: TrajectoryFlowField,
    batch,
    args: argparse.Namespace,
    batch_index: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    condition, frames = (item.to(device) for item in batch)
    count = condition.shape[0]
    indices = torch.randint(0, args.integration_steps, (count,), device=device)
    row = torch.arange(count, device=device)
    current = frames[row, indices]
    following = frames[row, indices + 1]
    dt = 1.0 / args.integration_steps
    target_velocity = (following - current) / dt
    t = indices.float()[:, None] * dt
    action = torch.zeros(count, 3, device=device)
    predicted_velocity = model.state_velocity(current, condition, t, action)
    added_path = (following - current).clamp_min(0.0)
    weight = 1.0 + 20.0 * added_path + 3.0 * following
    velocity_loss = (((predicted_velocity - target_velocity) ** 2) * weight).mean()
    total = velocity_loss
    values = {"velocity_loss": float(velocity_loss.detach().cpu()), "rollout_loss": 0.0}
    if batch_index % args.rollout_every == 0:
        limit = min(args.rollout_batch, count)
        _, predicted_frames = integrate_trajectory(
            model,
            frames[:limit, 0],
            condition[:limit],
            action[:limit],
            steps=args.integration_steps,
            clamp=(0.0, 1.0),
            return_frames=True,
        )
        rollout_terms = []
        for step, prediction in enumerate(predicted_frames[1:], start=1):
            target = frames[:limit, step]
            rollout_terms.append(
                F.binary_cross_entropy(prediction.clamp(1e-5, 1.0 - 1e-5), target) + soft_dice_loss(prediction, target)
            )
        rollout_loss = torch.stack(rollout_terms).mean()
        total = total + args.rollout_weight * rollout_loss
        values["rollout_loss"] = float(rollout_loss.detach().cpu())
    values["total_loss"] = float(total.detach().cpu())
    return total, values


@torch.no_grad()
def validate(model, loader, args, device) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    examples = 0
    for batch in loader:
        if args.task == "maze":
            condition, frames = (item.to(device) for item in batch)
            action = torch.zeros(condition.shape[0], 3, device=device)
            prediction, predicted_frames = integrate_trajectory(
                model, frames[:, 0], condition, action, steps=args.integration_steps,
                clamp=(0.0, 1.0), return_frames=True,
            )
            target = frames[:, -1]
            binary = prediction >= 0.5
            truth = target >= 0.5
            intersection = (binary & truth).sum(dim=(1, 2, 3)).float()
            union = (binary | truth).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
            wall = condition[:, :1] >= 0.5
            predicted_count = binary.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
            truth_count = truth.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
            goal = condition[:, 2:3] >= 0.5
            metrics = {
                "endpoint_mse": ((prediction - target) ** 2).mean(dim=(1, 2, 3)),
                "endpoint_iou": intersection / union,
                "path_precision": intersection / predicted_count,
                "path_recall": intersection / truth_count,
                "goal_reached": (binary & goal).any(dim=3).any(dim=2).any(dim=1).float(),
                "obstacle_violation_rate": (binary & wall).sum(dim=(1, 2, 3)).float() / predicted_count,
                "trajectory_mse": torch.stack([
                    ((pred - frames[:, step]) ** 2).mean(dim=(1, 2, 3))
                    for step, pred in enumerate(predicted_frames)
                ]).mean(dim=0),
            }
        else:
            base, start, delta = (item.to(device) for item in batch)
            source = render_rotation_state(base, start, delta, torch.zeros_like(delta))
            action = rotation_action(delta)
            prediction, predicted_frames = integrate_trajectory(
                model, source, source, action, steps=args.integration_steps,
                clamp=(0.0, 1.0), return_frames=True,
            )
            times = torch.linspace(0.0, 1.0, args.integration_steps + 1, device=device)
            truth_frames = render_rotation_frames(base, start, delta, times)
            target = truth_frames[:, -1]
            mse = ((prediction - target) ** 2).mean(dim=(1, 2, 3))
            binary = prediction.amax(dim=1) >= 0.1
            truth = target.amax(dim=1) >= 0.1
            union = (binary | truth).sum(dim=(1, 2)).float().clamp_min(1.0)
            reverse = integrate_trajectory(
                model, prediction, prediction, rotation_action(-delta), steps=args.integration_steps,
                clamp=(0.0, 1.0), return_frames=False,
            )
            metrics = {
                "endpoint_mse": mse,
                "endpoint_psnr_db": -10.0 * torch.log10(mse.clamp_min(1e-10)),
                "silhouette_iou": (binary & truth).sum(dim=(1, 2)).float() / union,
                "cycle_mse": ((reverse - source) ** 2).mean(dim=(1, 2, 3)),
                "trajectory_mse": torch.stack([
                    ((pred - truth_frames[:, step]) ** 2).mean(dim=(1, 2, 3))
                    for step, pred in enumerate(predicted_frames)
                ]).mean(dim=0),
            }
        batch_size = int(next(iter(metrics.values())).shape[0])
        examples += batch_size
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.sum().cpu())
    return {key: value / max(1, examples) for key, value in sums.items()}


@torch.no_grad()
def save_audit_grid(model, dataset, args, device, destination: Path) -> None:
    cache_dir = destination.parent / ".mpl-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    examples = min(args.audit_examples, len(dataset))
    columns = 1 + 3 * 4
    figure, axes = plt.subplots(examples, columns, figsize=(2.0 * columns, 2.0 * examples), squeeze=False)
    audit_times = (0.0, 0.25, 0.5, 1.0)
    for row_index in range(examples):
        sample = dataset[row_index]
        if args.task == "maze":
            condition, truth_frames = (item.unsqueeze(0).to(device) for item in sample)
            _, predicted_frames = integrate_trajectory(
                model, truth_frames[:, 0], condition, torch.zeros(1, 3, device=device),
                steps=args.integration_steps, clamp=(0.0, 1.0), return_frames=True,
            )
            condition_image = torch.stack([
                0.55 * condition[0, 0] + condition[0, 2],
                0.55 * condition[0, 0] + condition[0, 1],
                0.55 * condition[0, 0],
            ], dim=-1).clamp(0.0, 1.0).cpu()
            axes[row_index, 0].imshow(condition_image)
            axes[row_index, 0].set_title("maze (start green, goal red)")
            true_sequence = truth_frames[0]
        else:
            base, start, delta = (item.unsqueeze(0).to(device) for item in sample)
            source = render_rotation_state(base, start, delta, torch.zeros_like(delta))
            _, predicted_frames = integrate_trajectory(
                model, source, source, rotation_action(delta), steps=args.integration_steps,
                clamp=(0.0, 1.0), return_frames=True,
            )
            times = torch.linspace(0.0, 1.0, args.integration_steps + 1, device=device)
            true_sequence = render_rotation_frames(base, start, delta, times)[0]
            image = source[0].cpu().permute(1, 2, 0)
            axes[row_index, 0].imshow(image.squeeze(-1), cmap="gray", vmin=0, vmax=1) if image.shape[-1] == 1 else axes[row_index, 0].imshow(image)
            axes[row_index, 0].set_title(f"source\nΔ={float(delta.item()):.1f}°")
        for time_index, fraction in enumerate(audit_times):
            step = round(fraction * args.integration_steps)
            truth = true_sequence[step]
            prediction = predicted_frames[step][0]
            residual = (prediction - truth).abs()
            for offset, (label, image_tensor) in enumerate((
                ("true", truth), ("pred", prediction), ("|error|", residual)
            )):
                axis = axes[row_index, 1 + 3 * time_index + offset]
                image = image_tensor.detach().cpu().permute(1, 2, 0)
                axis.imshow(image.squeeze(-1), cmap="gray", vmin=0, vmax=1) if image.shape[-1] == 1 else axis.imshow(image.clamp(0, 1))
                axis.set_title(f"{label} t={fraction:g}")
        for axis in axes[row_index]:
            axis.axis("off")
    split_label = getattr(dataset, "split_label", "fixed_validation")
    figure.suptitle(f"{args.task} trajectory flow: {split_label}", fontsize=14)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, validation_dataset = make_datasets(args)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        worker_init_fn=seed_worker, generator=seeded_generator(args.seed), persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        worker_init_fn=seed_worker, persistent_workers=args.num_workers > 0,
    )
    model = model_for_task(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output_dir / "resolved_config.json", {"schema_version": 1, "arguments": resolved})
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    checkpoint_path = args.output_dir / "best_checkpoint.pt"
    history: List[Dict[str, float]] = []
    best = float("-inf")
    best_epoch = 0
    started = time.perf_counter()
    loss_function = maze_loss if args.task == "maze" else rotation_loss
    for epoch in range(1, args.epochs + 1):
        model.train()
        accumulators = {"total_loss": 0.0, "velocity_loss": 0.0, "rollout_loss": 0.0}
        for batch_index, batch in enumerate(train_loader):
            loss, values = loss_function(model, batch, args, batch_index, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in accumulators:
                accumulators[key] += values[key]
        should_validate = epoch == 1 or epoch % args.validation_every == 0 or epoch == args.epochs
        validation = validate(model, validation_loader, args, device) if should_validate else {}
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / max(1, len(train_loader)) for key, value in accumulators.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        selection_metric = "endpoint_iou" if args.task == "maze" else "silhouette_iou"
        if validation and validation[selection_metric] > best:
            best = validation[selection_metric]
            best_epoch = epoch
            torch.save(
                {
                    "schema_version": 1,
                    "model_name": "trajectory_spatial_flow",
                    "task": args.task,
                    "model_config": model.config(),
                    "model_state_dict": model.state_dict(),
                    "training_arguments": resolved,
                    "best_epoch": best_epoch,
                    "validation_metrics": validation,
                    "selection_metric": selection_metric,
                },
                checkpoint_path,
            )
        if should_validate and (epoch == 1 or epoch % args.audit_every == 0 or epoch == args.epochs):
            save_audit_grid(model, validation_dataset, args, device, args.output_dir / "audits" / f"epoch_{epoch:03d}.png")
        scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = validate(model, validation_loader, args, device)
    save_audit_grid(model, validation_dataset, args, device, args.output_dir / "audit_best.png")
    write_json(args.output_dir / "epoch_metrics.json", {"epochs": history})
    write_json(args.output_dir / "quality_metrics.json", {"split": "fixed_validation", "metrics": final_metrics})
    summary = {
        "experiment_name": f"neurreps_trajectory_flow_{args.task}",
        "task": args.task,
        "model": "trajectory_supervised_spatial_flow_no_ppo",
        "seed": args.seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "validation_protocol": getattr(validation_dataset, "split_label", "fixed_validation"),
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_selection_value": best,
        "metrics": {"validation": final_metrics},
        "elapsed_seconds": time.perf_counter() - started,
        "preliminary": bool(args.preliminary),
        "audit_image": str((args.output_dir / "audit_best.png").resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

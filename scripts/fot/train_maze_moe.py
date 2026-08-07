#!/usr/bin/env python3
"""Train a maze flow routed over frozen Tetris and colored spatial experts."""

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

from utils.fot.maze_moe import MazeExpertMixtureFlow, PreparedExperts, integrate_maze_moe
from utils.fot.reproducibility import collect_run_metadata, sha256_file, write_json
from utils.fot.torch_utils import seed_worker, seeded_generator, set_seed
from utils.fot.trajectory_datasets import MazeTrajectoryDataset
from utils.fot.trajectory_flow import TrajectoryFlowField, soft_dice_loss
from utils.fot.trajectory_metrics import maze_temporal_metrics


ROUTER_MODES = ("learned", "uniform", "tetris_only", "colored_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tetris-checkpoint", type=Path, required=True)
    parser.add_argument("--colored-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--maze-cells", type=int, default=9)
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--validation-samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--expert-dim", type=int, default=16)
    parser.add_argument("--router-width", type=int, default=16)
    parser.add_argument("--router-temperature", type=float, default=0.5)
    parser.add_argument("--router-mode", choices=ROUTER_MODES, default="learned")
    parser.add_argument("--router-balance-weight", type=float, default=0.01)
    parser.add_argument("--router-entropy-weight", type=float, default=0.001)
    parser.add_argument("--integration-steps", type=int, default=8)
    parser.add_argument("--rollout-batch", type=int, default=4)
    parser.add_argument("--rollout-every", type=int, default=2)
    parser.add_argument("--rollout-weight", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--audit-examples", type=int, default=4)
    parser.add_argument("--audit-every", type=int, default=5)
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--preliminary", action="store_true")
    return parser.parse_args()


def load_expert(path: Path, device: torch.device) -> Tuple[TrajectoryFlowField, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TrajectoryFlowField(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def make_datasets(args: argparse.Namespace):
    train = MazeTrajectoryDataset(
        n_samples=args.train_samples,
        maze_cells=args.maze_cells,
        image_size=args.image_size,
        trajectory_steps=args.integration_steps,
        seed=args.seed,
    )
    validation = MazeTrajectoryDataset(
        n_samples=args.validation_samples,
        maze_cells=args.maze_cells,
        image_size=args.image_size,
        trajectory_steps=args.integration_steps,
        seed=args.seed + 10_000_019,
    )
    return train, validation


def subset_experts(prepared: PreparedExperts, count: int) -> PreparedExperts:
    return prepared[0][:count], prepared[1][:count]


def maze_moe_loss(
    model: MazeExpertMixtureFlow,
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
    time_value = indices.float()[:, None] * dt
    action = torch.zeros(count, 3, device=device)
    prepared = model.prepare_experts(condition)
    predicted_velocity, routing = model(
        current,
        condition,
        time_value,
        action,
        prepared_experts=prepared,
        return_router=True,
    )
    added_path = (following - current).clamp_min(0.0)
    weight = 1.0 + 20.0 * added_path + 3.0 * following
    velocity_loss = (((predicted_velocity - target_velocity) ** 2) * weight).mean()
    importance = routing.mean(dim=(0, 2, 3))
    balance_loss = ((importance - 0.5) ** 2).sum()
    entropy_loss = -(routing.clamp_min(1e-8).log() * routing).sum(dim=1).mean()
    total = (
        velocity_loss
        + args.router_balance_weight * balance_loss
        + args.router_entropy_weight * entropy_loss
    )
    values = {
        "velocity_loss": float(velocity_loss.detach().cpu()),
        "rollout_loss": 0.0,
        "router_balance_loss": float(balance_loss.detach().cpu()),
        "router_entropy_loss": float(entropy_loss.detach().cpu()),
        "router_tetris_weight": float(importance[0].detach().cpu()),
    }
    if batch_index % args.rollout_every == 0:
        limit = min(args.rollout_batch, count)
        _, predicted_frames = integrate_maze_moe(
            model,
            frames[:limit, 0],
            condition[:limit],
            action[:limit],
            steps=args.integration_steps,
            clamp=(0.0, 1.0),
            return_frames=True,
            prepared_experts=subset_experts(prepared, limit),
        )
        rollout_terms = []
        for step, prediction in enumerate(predicted_frames[1:], start=1):
            target = frames[:limit, step]
            rollout_terms.append(
                F.binary_cross_entropy(prediction.clamp(1e-5, 1.0 - 1e-5), target)
                + soft_dice_loss(prediction, target)
            )
        rollout_loss = torch.stack(rollout_terms).mean()
        total = total + args.rollout_weight * rollout_loss
        values["rollout_loss"] = float(rollout_loss.detach().cpu())
    values["total_loss"] = float(total.detach().cpu())
    return total, values


def safe_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=(-1, -2)) / mask.sum(dim=(-1, -2)).clamp_min(1.0)


@torch.no_grad()
def validate(
    model: MazeExpertMixtureFlow,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    *,
    router_mode: str,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    examples = 0
    for condition, frames in loader:
        condition, frames = condition.to(device), frames.to(device)
        action = torch.zeros(condition.shape[0], 3, device=device)
        prepared = model.prepare_experts(condition)
        endpoint, predicted_frames = integrate_maze_moe(
            model,
            frames[:, 0],
            condition,
            action,
            steps=args.integration_steps,
            clamp=(0.0, 1.0),
            return_frames=True,
            prepared_experts=prepared,
            router_mode=router_mode,
        )
        target = frames[:, -1]
        binary, truth = endpoint >= 0.5, target >= 0.5
        intersection = (binary & truth).sum(dim=(1, 2, 3)).float()
        union = (binary | truth).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
        predicted_count = binary.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
        truth_count = truth.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
        wall, goal = condition[:, :1] >= 0.5, condition[:, 2:3] >= 0.5
        metrics: Dict[str, torch.Tensor] = {
            "endpoint_iou": intersection / union,
            "endpoint_mse": ((endpoint - target) ** 2).mean(dim=(1, 2, 3)),
            "path_precision": intersection / predicted_count,
            "path_recall": intersection / truth_count,
            "goal_reached": (binary & goal).any(dim=3).any(dim=2).any(dim=1).float(),
            "obstacle_violation_rate": (binary & wall).sum(dim=(1, 2, 3)).float() / predicted_count,
            "trajectory_mse": torch.stack([
                ((prediction - frames[:, step]) ** 2).mean(dim=(1, 2, 3))
                for step, prediction in enumerate(predicted_frames)
            ]).mean(dim=0),
            **maze_temporal_metrics(predicted_frames, frames),
        }
        routing_sequence = []
        for step, state in enumerate(predicted_frames):
            time_value = torch.full(
                (condition.shape[0], 1), step / args.integration_steps,
                device=device, dtype=condition.dtype,
            )
            _, weights = model(
                state,
                condition,
                time_value,
                action,
                prepared_experts=prepared,
                router_mode=router_mode,
                return_router=True,
            )
            routing_sequence.append(weights)
        routing = torch.stack(routing_sequence, dim=1)
        tetris_weight = routing[:, :, 0]
        final_path = truth[:, 0].float()
        wall_mask = wall[:, 0].float()
        metrics["router_tetris_weight"] = tetris_weight.mean(dim=(1, 2, 3))
        metrics["router_tetris_weight_on_path"] = safe_masked_mean(
            tetris_weight.mean(dim=1), final_path
        )
        metrics["router_tetris_weight_on_walls"] = safe_masked_mean(
            tetris_weight.mean(dim=1), wall_mask
        )
        metrics["router_entropy"] = -(
            routing.clamp_min(1e-8).log() * routing
        ).sum(dim=2).mean(dim=(1, 2, 3))
        batch_size = condition.shape[0]
        examples += batch_size
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.sum().cpu())
    return {key: value / examples for key, value in sums.items()}


def configure_matplotlib(destination: Path) -> None:
    cache = destination.parent / ".mpl-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


@torch.no_grad()
def save_audit_grid(
    model: MazeExpertMixtureFlow,
    dataset: MazeTrajectoryDataset,
    args: argparse.Namespace,
    device: torch.device,
    destination: Path,
) -> None:
    configure_matplotlib(destination)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    examples = min(args.audit_examples, len(dataset))
    figure, axes = plt.subplots(examples, 7, figsize=(15, 2.3 * examples), squeeze=False)
    for row in range(examples):
        condition, truth = (value.unsqueeze(0).to(device) for value in dataset[row])
        action = torch.zeros(1, 3, device=device)
        prepared = model.prepare_experts(condition)
        _, predicted = integrate_maze_moe(
            model,
            truth[:, 0],
            condition,
            action,
            steps=args.integration_steps,
            clamp=(0.0, 1.0),
            return_frames=True,
            prepared_experts=prepared,
        )
        prediction = torch.stack(predicted, dim=1)
        step_numbers = torch.arange(args.integration_steps + 1, device=device).view(
            1, args.integration_steps + 1, 1, 1, 1
        )
        never = torch.full_like(prediction, args.integration_steps + 1, dtype=torch.long)
        true_time = torch.where(truth >= 0.5, step_numbers, never).amin(dim=1)[0, 0].float()
        pred_time = torch.where(prediction >= 0.5, step_numbers, never).amin(dim=1)[0, 0].float()
        true_time /= args.integration_steps
        pred_time /= args.integration_steps
        final_path = truth[0, -1, 0] >= 0.5
        timing_error = (pred_time - true_time).abs() * final_path
        future_intensity = torch.stack([
            predicted[step][0, 0] * (final_path & (truth[0, step, 0] < 0.5))
            for step in range(1, args.integration_steps)
        ]).amax(dim=0)
        router_maps = []
        for step in (0, args.integration_steps):
            _, weights = model(
                predicted[step],
                condition,
                torch.tensor([[step / args.integration_steps]], device=device),
                action,
                prepared_experts=prepared,
                return_router=True,
            )
            router_maps.append(weights[0, 0])
        condition_image = torch.stack([
            0.55 * condition[0, 0] + condition[0, 2],
            0.55 * condition[0, 0] + condition[0, 1],
            0.55 * condition[0, 0],
        ], dim=-1).clamp(0, 1)
        panels = (
            ("maze", condition_image.cpu(), None),
            ("true activation t", true_time.cpu(), "viridis"),
            ("pred activation t", pred_time.cpu(), "viridis"),
            ("|timing error|", timing_error.cpu(), "magma"),
            ("future intensity", future_intensity.cpu(), "magma"),
            ("Tetris weight t=0", router_maps[0].cpu(), "coolwarm"),
            ("Tetris weight t=1", router_maps[1].cpu(), "coolwarm"),
        )
        for column, (title, image, cmap) in enumerate(panels):
            axes[row, column].imshow(image, cmap=cmap, vmin=0, vmax=1)
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
    figure.suptitle("Maze mixture of frozen Tetris and colored spatial experts")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
    for path in (args.tetris_checkpoint, args.colored_checkpoint):
        if not path.exists():
            raise SystemExit(f"Missing frozen expert checkpoint: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, validation_dataset = make_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=seeded_generator(args.seed),
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    tetris, tetris_checkpoint = load_expert(args.tetris_checkpoint, device)
    colored, colored_checkpoint = load_expert(args.colored_checkpoint, device)
    model = MazeExpertMixtureFlow(
        tetris,
        colored,
        width=args.width,
        context_dim=args.context_dim,
        expert_dim=args.expert_dim,
        router_width=args.router_width,
        router_temperature=args.router_temperature,
        router_mode=args.router_mode,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    resolved = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    expert_metadata = {
        "tetris": {
            "path": str(args.tetris_checkpoint.resolve()),
            "sha256": sha256_file(args.tetris_checkpoint),
            "best_epoch": int(tetris_checkpoint["best_epoch"]),
        },
        "colored": {
            "path": str(args.colored_checkpoint.resolve()),
            "sha256": sha256_file(args.colored_checkpoint),
            "best_epoch": int(colored_checkpoint["best_epoch"]),
        },
    }
    write_json(args.output_dir / "resolved_config.json", {
        "schema_version": 1,
        "arguments": resolved,
        "frozen_experts": expert_metadata,
    })
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    checkpoint_path = args.output_dir / "best_checkpoint.pt"
    history: List[Dict[str, float]] = []
    best, best_epoch = float("-inf"), 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        accumulators = {
            "total_loss": 0.0,
            "velocity_loss": 0.0,
            "rollout_loss": 0.0,
            "router_balance_loss": 0.0,
            "router_entropy_loss": 0.0,
            "router_tetris_weight": 0.0,
        }
        for batch_index, batch in enumerate(train_loader):
            loss, values = maze_moe_loss(model, batch, args, batch_index, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for key in accumulators:
                accumulators[key] += values[key]
        should_validate = (
            epoch == 1 or epoch % args.validation_every == 0 or epoch == args.epochs
        )
        validation = (
            validate(model, validation_loader, args, device, router_mode=args.router_mode)
            if should_validate else {}
        )
        row = {
            "epoch": epoch,
            **{
                f"train_{key}": value / max(1, len(train_loader))
                for key, value in accumulators.items()
            },
            **{f"validation_{key}": value for key, value in validation.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if validation and validation["endpoint_iou"] > best:
            best = validation["endpoint_iou"]
            best_epoch = epoch
            torch.save({
                "schema_version": 1,
                "model_name": "maze_frozen_rotation_expert_mixture",
                "task": "maze",
                "model_config": model.config(),
                "model_state_dict": model.state_dict(),
                "training_arguments": resolved,
                "frozen_experts": expert_metadata,
                "best_epoch": best_epoch,
                "validation_metrics": validation,
                "selection_metric": "endpoint_iou",
            }, checkpoint_path)
        if should_validate and (
            epoch == 1 or epoch % args.audit_every == 0 or epoch == args.epochs
        ):
            save_audit_grid(
                model,
                validation_dataset,
                args,
                device,
                args.output_dir / "audits" / f"epoch_{epoch:03d}.png",
            )
        scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = {
        mode: validate(model, validation_loader, args, device, router_mode=mode)
        for mode in ROUTER_MODES
    }
    save_audit_grid(model, validation_dataset, args, device, args.output_dir / "audit_best.png")
    write_json(args.output_dir / "epoch_metrics.json", {"epochs": history})
    write_json(args.output_dir / "quality_metrics.json", {
        "split": "fixed_validation",
        "post_training_router_interventions": True,
        "metrics": final_metrics,
    })
    summary = {
        "experiment_name": "neurreps_maze_frozen_rotation_expert_mixture",
        "task": "maze",
        "model": "maze_moe_frozen_tetris_colored_spatial_experts",
        "seed": args.seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "validation_protocol": "fixed_seed_disjoint_generated_mazes",
        "best_epoch": best_epoch,
        "selection_metric": "endpoint_iou",
        "best_selection_value": best,
        "metrics": final_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "preliminary": bool(args.preliminary),
        "frozen_experts": expert_metadata,
        "router_intervention_note": (
            "uniform/tetris_only/colored_only are post-training gate interventions on the learned-router model"
        ),
        "audit_image": str((args.output_dir / "audit_best.png").resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

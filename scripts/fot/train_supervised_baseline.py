from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from utils.fot.experiment_config import load_experiment_config, validate_experiment_config
from utils.fot.metrics import binary_classification_metrics
from utils.fot.reproducibility import collect_run_metadata, sha256_file, write_json
from utils.fot.rotation_dataset import RotationPairDataset, nested_fraction
from utils.fot.supervised_models import PairCNN, PairVisionTransformer, count_parameters
from utils.fot.torch_utils import seed_worker, seeded_generator, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a matched supervised rotation baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed for a multi-seed run.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    model = config["model"]
    common = {
        "image_size": int(config["dataset"]["image_size"]),
        "in_channels": int(model["input_channels"]),
        "num_classes": int(model["num_classes"]),
    }
    if model["type"] == "cnn":
        return PairCNN(**common, widths=tuple(model.get("widths", [32, 64, 128])))
    return PairVisionTransformer(
        **common,
        patch_size=int(model["patch_size"]),
        embed_dim=int(model["embed_dim"]),
        depth=int(model["depth"]),
        num_heads=int(model["num_heads"]),
        mlp_ratio=float(model["mlp_ratio"]),
        dropout=float(model["dropout"]),
    )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        worker_init_fn=seed_worker,
        generator=seeded_generator(seed),
        persistent_workers=bool(num_workers > 0),
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float | int], List[Dict[str, Any]]]:
    model.eval()
    labels: List[int] = []
    probabilities: List[float] = []
    predictions: List[Dict[str, Any]] = []
    for batch in loader:
        logits = model(batch["pair"].to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
        batch_labels = batch["label"].tolist()
        batch_angles = batch["angle_deg"].tolist()
        labels.extend(int(value) for value in batch_labels)
        probabilities.extend(float(value) for value in probs)
        for sample_id, base_id, angle, label, probability in zip(
            batch["sample_id"], batch["base_id"], batch_angles, batch_labels, probs
        ):
            predictions.append(
                {
                    "sample_id": str(sample_id),
                    "base_id": str(base_id),
                    "angle_deg": float(angle),
                    "label": int(label),
                    "positive_probability": float(probability),
                    "prediction": int(float(probability) >= 0.5),
                }
            )
    return binary_classification_metrics(labels, probabilities), predictions


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        images = batch["pair"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count = int(labels.numel())
        loss_sum += float(loss.detach().cpu()) * count
        sample_count += count
    return loss_sum / max(1, sample_count)


def limited(dataset: Dataset, maximum: Any) -> Dataset:
    if maximum is None or int(maximum) >= len(dataset):
        return dataset
    return Subset(dataset, list(range(int(maximum))))


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = copy.deepcopy(load_experiment_config(config_path))
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
        validate_experiment_config(config)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"]["seed"])
    set_seed(seed, deterministic=True)
    device = resolve_device(str(config["experiment"]["device"]))

    manifest_path = Path(config["dataset"]["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    config["dataset"]["manifest"] = str(manifest_path.resolve())
    config["dataset"]["manifest_sha256"] = sha256_file(manifest_path)
    config["experiment"]["resolved_device"] = str(device)
    write_json(output_dir / "resolved_config.json", config)
    write_json(
        output_dir / "run_metadata.json",
        collect_run_metadata(repo_root=REPO_ROOT, config_path=config_path),
    )

    train_dataset = nested_fraction(
        RotationPairDataset(manifest_path, "train"),
        float(config["dataset"].get("train_fraction", 1.0)),
    )
    validation_dataset = limited(
        RotationPairDataset(manifest_path, "validation"),
        config["dataset"].get("max_eval_samples"),
    )
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=seed + 1,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = None
    if config["training"].get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif config["training"].get("scheduler") != "none":
        raise ValueError(f"Unsupported scheduler: {config['training'].get('scheduler')!r}")

    history: List[Dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = -1
    checkpoint_path = output_dir / "best_checkpoint.pt"
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        validation_metrics, _ = evaluate(model, validation_loader, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True))
        if float(validation_metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(validation_metrics["accuracy"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "config": config,
                },
                checkpoint_path,
            )
        if scheduler is not None:
            scheduler.step()
    write_json(output_dir / "epoch_metrics.json", {"epochs": history})

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics: Dict[str, Any] = {}
    for split in config["evaluation"]["splits"]:
        dataset = limited(
            RotationPairDataset(manifest_path, str(split)),
            config["dataset"].get("max_eval_samples"),
        )
        loader = make_loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=seed + 2,
        )
        metrics, predictions = evaluate(model, loader, device)
        final_metrics[str(split)] = metrics
        write_json(output_dir / f"predictions_{split}.json", {"predictions": predictions})

    summary = {
        "experiment_name": config["experiment"]["name"],
        "seed": seed,
        "task": config["dataset"]["task"],
        "model": config["model"]["type"],
        "parameter_count": count_parameters(model),
        "train_samples": len(train_dataset),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "checkpoint_selection": "validation_accuracy",
        "metrics": final_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "preliminary": "smoke" in str(config["experiment"]["name"]).lower(),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Reproducible rotation-orbit flow training for colored-shape images."""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import kornia as K
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.fot.colored_shapes_ops import random_colored_rectangles
from utils.fot.models import CondEncoder, FastRotator
from utils.fot.reproducibility import collect_run_metadata, write_json
from utils.fot.supervised_models import count_parameters
from utils.fot.torch_utils import seeded_generator, set_seed


class ColoredFlowDataset(Dataset):
    def __init__(self, n: int, image_size: int, seed: int, num_shapes: int = 4):
        self.n = int(n)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.num_shapes = int(num_shapes)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = random.Random(self.seed + 1_000_003 * int(index))
        return random_colored_rectangles(self.image_size, self.image_size, num_shapes=self.num_shapes, rng=rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train colored-shape rotation-orbit flow model.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--flow-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preliminary", action="store_true")
    return parser.parse_args()


def batch_loss(model: FastRotator, encoder: CondEncoder, images: torch.Tensor, dt: float) -> torch.Tensor:
    batch = int(images.shape[0])
    start = torch.rand(batch, device=images.device) * 360.0
    delta = torch.rand(batch, device=images.device) * 360.0 - 180.0
    t = torch.rand(batch, 1, device=images.device)
    current_angle = start + t[:, 0] * delta
    x_t = K.geometry.transform.rotate(images, current_angle)
    x_next = K.geometry.transform.rotate(images, current_angle + dt * delta)
    target_velocity = (x_next - x_t) / dt
    prediction = model(x_t, t, encoder(images), delta[:, None])
    return F.mse_loss(prediction, target_velocity)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output_dir / "resolved_config.json", {"schema_version": 1,
               "experiment": {"name": "colored_rotation_orbit_flow", "seed": args.seed,
                              "device": args.device, "preliminary": bool(args.preliminary)},
               "arguments": {key: value for key, value in resolved.items() if key != "seed"}})
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    train = ColoredFlowDataset(args.train_samples, args.image_size, args.seed)
    validation = ColoredFlowDataset(args.validation_samples, args.image_size, args.seed + 10_000_019)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              generator=seeded_generator(args.seed))
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    encoder = CondEncoder(in_ch=3, emb_dim=args.embedding_dim).to(device)
    model = FastRotator(in_ch=3, out_ch=3, backbone_dim=args.embedding_dim, flow_dim=args.flow_dim).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(encoder.parameters()),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint = args.output_dir / "best_checkpoint.pt"
    best = float("inf")
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); encoder.train()
        total = 0.0
        for images in train_loader:
            loss = batch_loss(model, encoder, images.to(device), args.dt)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach().cpu())
        model.eval(); encoder.eval()
        with torch.no_grad():
            values = [float(batch_loss(model, encoder, images.to(device), args.dt).cpu()) for images in validation_loader]
        row = {"epoch": epoch, "train_velocity_mse": total / len(train_loader),
               "validation_velocity_mse": sum(values) / len(values), "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row); print(json.dumps(row, sort_keys=True))
        if row["validation_velocity_mse"] < best:
            best = row["validation_velocity_mse"]; best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "encoder_state_dict": encoder.state_dict(),
                        "emb_dim": args.embedding_dim, "flow_dim": args.flow_dim, "epoch": epoch,
                        "validation_velocity_mse": best}, checkpoint)
        scheduler.step()
    write_json(args.output_dir / "epoch_metrics.json", {"epochs": history})
    summary: Dict[str, Any] = {"experiment_name": "colored_rotation_orbit_flow", "task": "colored_rotation",
        "model": "rotation_orbit_flow", "seed": args.seed,
        "parameter_count": count_parameters(model) + count_parameters(encoder), "train_samples": len(train),
        "best_epoch": best_epoch, "metrics": {"validation": {"velocity_mse": best}},
        "elapsed_seconds": time.perf_counter() - started, "preliminary": bool(args.preliminary)}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

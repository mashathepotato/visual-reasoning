from __future__ import annotations

"""Matched DINOv3 pair baseline for the rotation and 3-D transfer tasks.

The default is a frozen ViT-S/16 backbone with a nonlinear relation head.  The
same backbone encodes each member of the pair and the head receives
[a, b, |a-b|, a*b].  ``--mode partial`` unfreezes the final transformer blocks.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.fot.dino_utils import create_dinov3, dino_embed_rgb01, dino_features_rgb01
from utils.fot.metrics import binary_classification_metrics
from utils.fot.reproducibility import collect_run_metadata, write_json
from utils.fot.rotation_dataset import RotationPairDataset
from utils.fot.supervised_models import count_parameters
from utils.fot.torch_utils import seeded_generator, set_seed


class ArrayPairDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], *, prefix: str):
        self.rows = rows
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[int(index)]
        source = torch.as_tensor(row["x0"], dtype=torch.float32)
        target = torch.as_tensor(row["x1"], dtype=torch.float32)
        source = ((source + 1.0) * 0.5).repeat(3, 1, 1).clamp(0.0, 1.0)
        target = ((target + 1.0) * 0.5).repeat(3, 1, 1).clamp(0.0, 1.0)
        label = 1 if str(row.get("label", "")).lower() == "same" else 0
        return {
            "source": source,
            "target": target,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": f"{self.prefix}-{index:05d}-{row.get('name', '')}",
            "base_id": str(row.get("name", index)),
            "angle_deg": torch.tensor(float(row.get("angle", row.get("angle_diff", 0.0)))),
        }


class EmbeddingDataset(Dataset):
    def __init__(self, payload: Dict[str, Any]):
        self.a = payload["source_embeddings"].float()
        self.b = payload["target_embeddings"].float()
        self.labels = payload["labels"].long()
        self.sample_ids = list(payload["sample_ids"])
        self.angles = payload["angles"].float()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.a[index], self.b[index], self.labels[index], self.sample_ids[index], self.angles[index]


class PairHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # DINO features are already normalized by the backbone.  Avoid an
        # additional trainable LayerNorm here: torch 2.9's MPS backward can
        # produce non-finite affine gradients for this small pair batch.
        return self.net(torch.cat((a, b, torch.abs(a - b), a * b), dim=1))


class DinoPairModel(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int, hidden: int):
        super().__init__()
        self.backbone = backbone
        self.head = PairHead(dim, hidden=hidden)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.head(dino_features_rgb01(source, self.backbone), dino_features_rgb01(target, self.backbone))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DINOv3 ViT-S/16 pair baseline.")
    parser.add_argument("--dataset", choices=("tetris", "colored", "ganis3d"), required=True)
    parser.add_argument("--mode", choices=("frozen", "partial"), default="frozen")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "models" / "cache" / "dinov3_pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--preliminary", action="store_true")
    return parser.parse_args()


def limit(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, range(int(maximum)))


def datasets_for(args: argparse.Namespace) -> Dict[str, Dataset]:
    if args.dataset in {"tetris", "colored"}:
        default = REPO_ROOT / "data" / "splits" / f"{args.dataset}_rotation_v1.json"
        manifest = (args.manifest or default).resolve()
        return {
            "train": limit(RotationPairDataset(manifest, "train"), args.max_train),
            "validation": limit(RotationPairDataset(manifest, "validation"), args.max_eval),
            "test_id": limit(RotationPairDataset(manifest, "test_id"), args.max_eval),
            "test_ood_angle": limit(RotationPairDataset(manifest, "test_ood_angle"), args.max_eval),
        }
    # The 3-D condition is deliberately transfer-only: tune on synthetic Tetris,
    # evaluate once on Ganis-Kievit.  The checked-in 3-D training array has only
    # positive labels and overlapping identities, so fitting on it is invalid.
    manifest = (args.manifest or REPO_ROOT / "data" / "splits" / "tetris_rotation_v1.json").resolve()
    test_rows = list(np.load(REPO_ROOT / "data" / "test_balanced.npy", allow_pickle=True))
    return {
        "train": limit(RotationPairDataset(manifest, "train"), args.max_train),
        "validation": limit(RotationPairDataset(manifest, "validation"), args.max_eval),
        "test_ganis3d": limit(ArrayPairDataset(test_rows, prefix="ganis3d"), args.max_eval),
    }


@torch.no_grad()
def embed_dataset(backbone: nn.Module, dataset: Dataset, device: torch.device, batch_size: int, workers: int) -> Dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    aa: List[torch.Tensor] = []
    bb: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    ids: List[str] = []
    angles: List[torch.Tensor] = []
    for batch in loader:
        aa.append(dino_embed_rgb01(batch["source"].to(device), backbone).cpu())
        bb.append(dino_embed_rgb01(batch["target"].to(device), backbone).cpu())
        labels.append(batch["label"].cpu())
        ids.extend(str(value) for value in batch["sample_id"])
        angles.append(batch["angle_deg"].cpu())
    return {
        "source_embeddings": torch.cat(aa), "target_embeddings": torch.cat(bb),
        "labels": torch.cat(labels), "sample_ids": ids, "angles": torch.cat(angles),
    }


def cache_embeddings(args: argparse.Namespace, datasets: Dict[str, Dataset], backbone: nn.Module, device: torch.device) -> Dict[str, EmbeddingDataset]:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output: Dict[str, EmbeddingDataset] = {}
    suffix = "full" if args.max_train is None and args.max_eval is None else f"{args.max_train}-{args.max_eval}"
    for split, dataset in datasets.items():
        path = args.cache_dir / f"dinov3_vits16_{args.dataset}_{split}_{suffix}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
        else:
            payload = embed_dataset(backbone, dataset, device, args.batch_size, args.num_workers)
            torch.save(payload, path)
        output[split] = EmbeddingDataset(payload)
    return output


def loader(dataset: Dataset, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
                      generator=seeded_generator(args.seed))


def evaluate_head(head: nn.Module, data: DataLoader, device: torch.device) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    head.eval()
    labels: List[int] = []
    probabilities: List[float] = []
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for a, b, y, sample_ids, angles in data:
            probs = torch.softmax(head(a.to(device), b.to(device)), dim=1)[:, 1].cpu()
            for sample_id, angle, label, probability in zip(sample_ids, angles, y, probs):
                rows.append({"sample_id": str(sample_id), "angle_deg": float(angle), "label": int(label),
                             "positive_probability": float(probability), "prediction": int(probability >= 0.5)})
            labels.extend(int(v) for v in y)
            probabilities.extend(float(v) for v in probs)
    return binary_classification_metrics(labels, probabilities), rows


@torch.no_grad()
def evaluate_model(model: DinoPairModel, data: DataLoader, device: torch.device) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    labels: List[int] = []
    probabilities: List[float] = []
    rows: List[Dict[str, Any]] = []
    for batch in data:
        probs = torch.softmax(model(batch["source"].to(device), batch["target"].to(device)), dim=1)[:, 1].cpu()
        for sample_id, angle, label, probability in zip(batch["sample_id"], batch["angle_deg"], batch["label"], probs):
            rows.append({"sample_id": str(sample_id), "angle_deg": float(angle), "label": int(label),
                         "positive_probability": float(probability), "prediction": int(probability >= 0.5)})
        labels.extend(int(v) for v in batch["label"])
        probabilities.extend(float(v) for v in probs)
    return binary_classification_metrics(labels, probabilities), rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output_dir / "resolved_config.json", {
        "schema_version": 1,
        "experiment": {"name": f"dinov3_vits16_{args.mode}_{args.dataset}", "seed": args.seed,
                       "device": args.device, "preliminary": bool(args.preliminary)},
        "arguments": {key: value for key, value in resolved_args.items() if key != "seed"},
    })
    write_json(args.output_dir / "run_metadata.json", collect_run_metadata(repo_root=REPO_ROOT))
    started = time.perf_counter()
    raw = datasets_for(args)
    backbone = create_dinov3(device=device, freeze=True)
    dim = int(getattr(backbone, "num_features", 384))
    if args.mode == "partial":
        for block in list(backbone.blocks)[-int(args.unfreeze_blocks):]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        # torch 2.9 MPS can emit non-finite LayerNorm affine gradients.  Keep
        # normalization affine parameters frozen while adapting attention/MLP
        # weights in the final blocks.
        for module in backbone.modules():
            if isinstance(module, nn.LayerNorm):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        head = PairHead(dim, hidden=args.hidden_dim).to(device)
        model = DinoPairModel(backbone, dim, args.hidden_dim).to(device)
        model.head = head
        trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{"params": head.parameters(), "lr": args.learning_rate},
             {"params": trainable_backbone, "lr": args.backbone_learning_rate}],
            weight_decay=args.weight_decay,
        )
        train_loader = loader(raw["train"], args, shuffle=True)
        validation_loader = loader(raw["validation"], args, shuffle=False)
        embedded = None
    else:
        embedded = cache_embeddings(args, raw, backbone, device)
        head = PairHead(dim, hidden=args.hidden_dim).to(device)
        model = None
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        train_loader = loader(embedded["train"], args, shuffle=True)
        validation_loader = loader(embedded["validation"], args, shuffle=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = -math.inf
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    checkpoint = args.output_dir / "best_checkpoint.pt"
    for epoch in range(1, args.epochs + 1):
        head.train()
        if model is not None:
            model.train()
            model.backbone.eval()
        total_loss = 0.0
        total_n = 0
        for batch in train_loader:
            if model is None:
                a, b, y, _, _ = batch
                logits = head(a.to(device), b.to(device))
            else:
                y = batch["label"]
                logits = model(batch["source"].to(device), batch["target"].to(device))
            loss = F.cross_entropy(logits, y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(y.numel())
            total_n += int(y.numel())
        validation, _ = (evaluate_model(model, validation_loader, device) if model is not None
                         else evaluate_head(head, validation_loader, device))
        row = {"epoch": epoch, "train_loss": total_loss / total_n, "validation": validation}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if float(validation["accuracy"]) > best:
            best = float(validation["accuracy"])
            best_epoch = epoch
            torch.save((model if model is not None else head).state_dict(), checkpoint)
        scheduler.step()
    (model if model is not None else head).load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    metrics: Dict[str, Any] = {}
    eval_datasets = raw if model is not None else embedded
    assert eval_datasets is not None
    for split, dataset in eval_datasets.items():
        data_loader = loader(dataset, args, shuffle=False)
        split_metrics, predictions = (evaluate_model(model, data_loader, device) if model is not None
                                      else evaluate_head(head, data_loader, device))
        metrics[split] = split_metrics
        write_json(args.output_dir / f"predictions_{split}.json", {"predictions": predictions})
    write_json(args.output_dir / "epoch_metrics.json", {"epochs": history})
    summary = {
        "experiment_name": f"dinov3_vits16_{args.mode}_{args.dataset}", "task": args.dataset,
        "model": f"dinov3_vits16_{args.mode}_pair_head", "seed": args.seed,
        "parameter_count": count_parameters(backbone) + count_parameters(head),
        "trainable_parameter_count": count_parameters(model if model is not None else head, trainable_only=True),
        "external_pretraining": "DINOv3 LVD-1689M", "train_samples": len(eval_datasets["train"]),
        "best_epoch": best_epoch, "best_validation_accuracy": best, "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started, "preliminary": bool(args.preliminary),
        "protocol_notes": "ganis3d is zero-shot domain transfer from the Tetris-trained head" if args.dataset == "ganis3d" else "matched fixed manifests",
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

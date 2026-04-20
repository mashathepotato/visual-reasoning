from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.fot.external_datasets import SATv2Dataset
from utils.fot.text_ops import tokenize_choices_to_buckets, tokenize_to_buckets
from utils.fot.torch_utils import get_device, set_seed
from utils.fot.vqa_heatmap_model import FoTHeatmapMCQModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FoT heatmap MCQ model on SAT-v2 (array/SAT-v2).")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--streaming", action="store_true", help="Use HF streaming (requires --max-train/--max-val).")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=2000)

    p.add_argument("--text-buckets", type=int, default=50_000)
    p.add_argument("--text-dim", type=int, default=256)
    p.add_argument("--text-max-len", type=int, default=64)

    p.add_argument("--flow-dim", type=int, default=32)
    p.add_argument("--vision-feat-dim", type=int, default=128)
    p.add_argument("--sketch-steps", type=int, default=8)
    p.add_argument("--mlp-hidden", type=int, default=256)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="models/fot_sat_v2_heatmap_mcq.pt")
    return p.parse_args()


def collate_fn(batch, *, text_buckets: int, text_max_len: int) -> Dict[str, torch.Tensor]:
    device = torch.device("cpu")
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B,3,H,W)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    questions = [b["question"] for b in batch]
    choices = [b["choices"] for b in batch]

    q = tokenize_to_buckets(questions, num_buckets=text_buckets, max_len=text_max_len, device=device)
    c_ids, c_mask = tokenize_choices_to_buckets(choices, num_buckets=text_buckets, max_len=text_max_len, device=device)

    return {
        "images": images,
        "labels": labels,
        "q_input_ids": q.input_ids,
        "q_attention_mask": q.attention_mask,
        "choice_input_ids": c_ids,
        "choice_attention_mask": c_mask,
    }


@torch.no_grad()
def evaluate(model: FoTHeatmapMCQModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        logits, _heat = model(
            images=images,
            q_input_ids=batch["q_input_ids"].to(device),
            q_attention_mask=batch["q_attention_mask"].to(device),
            choice_input_ids=batch["choice_input_ids"].to(device),
            choice_attention_mask=batch["choice_attention_mask"].to(device),
        )
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        total += int(labels.numel())
        correct += int((preds == labels).sum().item())
        loss_sum += float(loss.detach().cpu()) * int(labels.numel())

    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "n": float(total),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    train_ds = SATv2Dataset(
        split=args.train_split,
        image_size=args.image_size,
        max_samples=args.max_train,
        cache_dir=args.cache_dir,
        streaming=bool(args.streaming),
    )
    val_ds = SATv2Dataset(
        split=args.val_split,
        image_size=args.image_size,
        max_samples=args.max_val,
        cache_dir=args.cache_dir,
        streaming=bool(args.streaming),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, text_buckets=args.text_buckets, text_max_len=args.text_max_len),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, text_buckets=args.text_buckets, text_max_len=args.text_max_len),
    )

    model = FoTHeatmapMCQModel(
        num_text_buckets=args.text_buckets,
        text_dim=args.text_dim,
        flow_dim=args.flow_dim,
        vision_feat_dim=args.vision_feat_dim,
        sketch_steps=args.sketch_steps,
        mlp_hidden=args.mlp_hidden,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    for epoch in range(int(args.epochs)):
        model.train()
        loss_sum = 0.0
        n_sum = 0

        for batch in train_loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            logits, _heat = model(
                images=images,
                q_input_ids=batch["q_input_ids"].to(device),
                q_attention_mask=batch["q_attention_mask"].to(device),
                choice_input_ids=batch["choice_input_ids"].to(device),
                choice_attention_mask=batch["choice_attention_mask"].to(device),
            )
            loss = F.cross_entropy(logits, labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            loss_sum += float(loss.detach().cpu()) * int(labels.numel())
            n_sum += int(labels.numel())

        train_loss = loss_sum / max(1, n_sum)
        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1:03d} | train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} | val_acc {val['acc']:.4f}")

        if val["acc"] > best_acc:
            best_acc = float(val["acc"])
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "val": val},
                str(out),
            )
            print("Saved:", out)


if __name__ == "__main__":
    main()

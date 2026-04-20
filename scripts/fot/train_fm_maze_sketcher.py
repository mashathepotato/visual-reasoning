from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.fot.maze_ops import MazeTraceDataset
from utils.fot.models import MazeSketcher
from utils.fot.torch_utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FM sketcher for maze trace drawing.")
    p.add_argument("--out", type=str, default="models/maze_sketcher_fm.pth")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--maze-cells", type=int, default=9)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--flow-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    ds = MazeTraceDataset(n_samples=args.train_samples, maze_cells=args.maze_cells, img_size=args.img_size, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    sketcher = MazeSketcher(cond_ch=3, flow_dim=args.flow_dim).to(device)
    optim = torch.optim.AdamW(sketcher.parameters(), lr=args.lr)

    for epoch in range(int(args.epochs)):
        sketcher.train()
        epoch_loss = 0.0
        for cond, trace_t, t, delta in loader:
            cond = cond.to(device)
            trace_t = trace_t.to(device)
            t = t.to(device)
            delta = delta.to(device)

            pred = sketcher(trace_t, cond, t)
            loss = F.mse_loss(pred, delta)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach().cpu())

        avg = epoch_loss / max(1, len(loader))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:03d} | loss {avg:.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sketcher.state_dict(), str(out))
    print("Saved sketcher:", out)


if __name__ == "__main__":
    main()


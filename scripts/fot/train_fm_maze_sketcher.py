from __future__ import annotations

import sys
import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.fot.maze_ops import MazeTraceDataset
from utils.fot.models import MazeSketcher
from utils.fot.torch_utils import get_device, set_seed
from utils.fot.reproducibility import write_json
from utils.fot.supervised_models import count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FM sketcher for maze trace drawing.")
    p.add_argument("--out", type=str, default="models/maze_sketcher_fm.pth")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--maze-cells", type=int, default=9)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--validation-samples", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--flow-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-out", type=str, default="")
    p.add_argument("--history-out", type=str, default="")
    p.add_argument("--preliminary", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    ds = MazeTraceDataset(n_samples=args.train_samples, maze_cells=args.maze_cells, img_size=args.img_size, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    validation = MazeTraceDataset(n_samples=args.validation_samples, maze_cells=args.maze_cells,
                                  img_size=args.img_size, seed=args.seed + 10_000_019)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)

    sketcher = MazeSketcher(cond_ch=3, flow_dim=args.flow_dim).to(device)
    optim = torch.optim.AdamW(sketcher.parameters(), lr=args.lr)

    history = []; best_loss = float("inf"); best_epoch = 0; started = time.perf_counter()
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
        sketcher.eval(); validation_loss = 0.0
        with torch.no_grad():
            for cond, trace_t, t, delta in validation_loader:
                prediction = sketcher(trace_t.to(device), cond.to(device), t.to(device))
                validation_loss += float(F.mse_loss(prediction, delta.to(device)).cpu())
        validation_loss /= max(1, len(validation_loader))
        row = {"epoch": epoch + 1, "train_delta_mse": avg, "validation_delta_mse": validation_loss}
        history.append(row); print(json.dumps(row, sort_keys=True))
        if validation_loss < best_loss:
            best_loss = validation_loss; best_epoch = epoch + 1
            out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); torch.save(sketcher.state_dict(), str(out))

    out = Path(args.out)
    if args.history_out:
        write_json(Path(args.history_out), {"epochs": history})
    if args.summary_out:
        write_json(Path(args.summary_out), {"experiment_name": "maze_trace_flow", "task": "maze_trace",
            "model": "trace_delta_flow", "seed": args.seed, "parameter_count": count_parameters(sketcher),
            "train_samples": args.train_samples, "best_epoch": best_epoch,
            "metrics": {"validation": {"delta_mse": best_loss}},
            "elapsed_seconds": time.perf_counter() - started, "preliminary": bool(args.preliminary)})
    print("Saved sketcher:", out)


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import kornia as K
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.fot.dino_utils import create_dinov3, dino_embed_fm_gray64
from utils.fot.models import CorrectorUNet, FastRotator
from utils.fot.tetris_ops import get_tetris_tensor, to_fm_tensor
from utils.fot.torch_utils import get_device, set_seed
from utils.llm_baselines import CHIRAL_TETRIS_SHAPES


@dataclass(frozen=True)
class ShapeEntry:
    img_fm: torch.Tensor  # (1, H, W) in [-1, 1]
    emb: torch.Tensor  # (384,)


def build_shape_cache(*, device: torch.device, img_size: int) -> Dict[str, ShapeEntry]:
    dino = create_dinov3(device=device)
    cache: Dict[str, ShapeEntry] = {}

    with torch.no_grad():
        for name in CHIRAL_TETRIS_SHAPES.keys():
            big01 = get_tetris_tensor(name, 224, channels=1).to(device)  # (1, 224, 224) in [0,1]
            big_fm = to_fm_tensor(big01)  # (1, 1, 224, 224) in [-1,1]
            emb = dino_embed_fm_gray64(big_fm, dino)[0].detach().cpu()

            small01 = get_tetris_tensor(name, img_size, channels=1)  # (1, H, W) in [0,1]
            small_fm = (small01 * 2.0) - 1.0  # (1, H, W) in [-1,1]

            cache[name] = ShapeEntry(img_fm=small_fm.detach().cpu(), emb=emb)

    return cache


class FastTetrisDataset(Dataset):
    def __init__(self, *, entries: Dict[str, ShapeEntry], n_samples: int, seed: int = 0):
        self.entries = entries
        self.keys: List[str] = list(entries.keys())
        self.n = int(n_samples)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self.rng.choice(self.keys)
        e = self.entries[key]
        return e.img_fm, e.emb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FM rotator + corrector on Tetris shapes.")
    p.add_argument("--out-rotator", type=str, default="models/rotator_fm_tetris_with_corrector.pth")
    p.add_argument("--out-corrector", type=str, default="models/rotator_fm_tetris_corrector.pth")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--test-samples", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--flow-dim", type=int, default=64)
    p.add_argument("--corr-base", type=int, default=16)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--corr-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print("Device:", device)

    entries = build_shape_cache(device=device, img_size=args.img_size)
    train_loader = DataLoader(
        FastTetrisDataset(entries=entries, n_samples=args.train_samples, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        FastTetrisDataset(entries=entries, n_samples=args.test_samples, seed=args.seed + 1),
        batch_size=args.batch_size,
        shuffle=False,
    )

    rotator = FastRotator(in_ch=1, out_ch=1, backbone_dim=384, flow_dim=args.flow_dim).to(device)
    corrector = CorrectorUNet(in_ch=1, out_ch=1, base_ch=args.corr_base).to(device)
    optim = torch.optim.AdamW(list(rotator.parameters()) + list(corrector.parameters()), lr=args.lr)

    dt = float(args.dt)
    for epoch in range(int(args.epochs)):
        rotator.train()
        corrector.train()
        epoch_loss = 0.0

        for base_img, base_emb in train_loader:
            base_img = base_img.to(device)  # (B, 1, H, W) in [-1,1]
            base_emb = base_emb.to(device)  # (B, 384)
            b = int(base_img.shape[0])

            ang_start = torch.rand(b, device=device) * 360.0
            ang_delta = torch.rand(b, device=device) * 360.0 - 180.0
            t = torch.rand(b, 1, device=device)

            ang_t = ang_start + (t.squeeze(1) * ang_delta)
            x_t = K.geometry.transform.rotate(base_img, ang_t)

            ang_next = ang_t + (dt * ang_delta)
            x_next = K.geometry.transform.rotate(base_img, ang_next)

            target_v = (x_next - x_t) / dt
            pred_v = rotator(x_t, t, base_emb, ang_delta.view(b, 1))

            x_pred = x_t + pred_v * dt
            corr = corrector(x_pred)
            x_corr = (x_pred + corr).clamp(-1.0, 1.0)

            loss_v = F.mse_loss(pred_v, target_v)
            loss_corr = F.mse_loss(x_corr, x_next)
            loss = loss_v + float(args.corr_weight) * loss_corr

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            epoch_loss += float(loss.detach().cpu())

        avg = epoch_loss / max(1, len(train_loader))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            rotator.eval()
            corrector.eval()
            with torch.no_grad():
                test_loss = 0.0
                for base_img, base_emb in test_loader:
                    base_img = base_img.to(device)
                    base_emb = base_emb.to(device)
                    b = int(base_img.shape[0])
                    ang_start = torch.rand(b, device=device) * 360.0
                    ang_delta = torch.rand(b, device=device) * 360.0 - 180.0
                    t = torch.rand(b, 1, device=device)
                    ang_t = ang_start + (t.squeeze(1) * ang_delta)
                    x_t = K.geometry.transform.rotate(base_img, ang_t)
                    ang_next = ang_t + (dt * ang_delta)
                    x_next = K.geometry.transform.rotate(base_img, ang_next)
                    target_v = (x_next - x_t) / dt
                    pred_v = rotator(x_t, t, base_emb, ang_delta.view(b, 1))
                    x_pred = x_t + pred_v * dt
                    corr = corrector(x_pred)
                    x_corr = (x_pred + corr).clamp(-1.0, 1.0)
                    loss = F.mse_loss(pred_v, target_v) + float(args.corr_weight) * F.mse_loss(x_corr, x_next)
                    test_loss += float(loss.detach().cpu())
                test_loss /= max(1, len(test_loader))

            print(f"Epoch {epoch + 1:03d} | train {avg:.6f} | test {test_loss:.6f}")

    out_rot = Path(args.out_rotator)
    out_corr = Path(args.out_corrector)
    out_rot.parent.mkdir(parents=True, exist_ok=True)
    out_corr.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rotator.state_dict(), str(out_rot))
    torch.save(corrector.state_dict(), str(out_corr))
    print("Saved rotator:", out_rot)
    print("Saved corrector:", out_corr)


if __name__ == "__main__":
    main()

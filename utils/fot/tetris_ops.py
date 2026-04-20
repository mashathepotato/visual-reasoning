from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from utils.llm_baselines import CHIRAL_TETRIS_SHAPES, draw_tetris_shape_np

from .rotation_ops import rotate_tensor


_TETRIS_CACHE: Dict[Tuple[str, int, int], torch.Tensor] = {}


def get_tetris_tensor(name: str, size: int, *, channels: int = 3) -> torch.Tensor:
    key = (name, int(size), int(channels))
    if key in _TETRIS_CACHE:
        return _TETRIS_CACHE[key]
    img = draw_tetris_shape_np(name, size)  # (H, W) u8
    t = torch.tensor(img).float().unsqueeze(0) / 255.0  # (1, H, W)
    if channels == 3:
        t = t.repeat(3, 1, 1)  # (3, H, W)
    _TETRIS_CACHE[key] = t
    return t


@torch.no_grad()
def infer_tetris_shape_key(target_obs: torch.Tensor) -> Optional[str]:
    """Infer which canonical shape produced a target image (for embedding consistency)."""
    if target_obs.dim() == 4:
        t = target_obs[0, 0].detach().cpu()
    elif target_obs.dim() == 3:
        t = target_obs[0].detach().cpu()
    else:
        return None

    h = int(t.shape[-2])
    keys = list(CHIRAL_TETRIS_SHAPES.keys())

    for k in keys:
        canon = get_tetris_tensor(k, h, channels=1)[0]
        if torch.allclose(t, canon, atol=1e-6):
            return k

    best_key = None
    best_err = float("inf")
    for k in keys:
        canon = get_tetris_tensor(k, h, channels=1)[0]
        err = float(torch.mean((t - canon) ** 2))
        if err < best_err:
            best_err = err
            best_key = k
    return best_key


def to_fm_tensor(img: torch.Tensor) -> torch.Tensor:
    """(C,H,W) or (B,C,H,W) in [0,1] -> (B,1,H,W) in [-1,1]."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.shape[1] == 3:
        img = img[:, :1]
    return (img * 2.0) - 1.0


def to_obs_tensor(img_fm: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) in [-1,1] -> (B,3,H,W) in [0,1]."""
    if img_fm.dim() == 3:
        img_fm = img_fm.unsqueeze(0)
    img_fm = torch.nan_to_num(img_fm, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    img = (img_fm + 1.0) / 2.0
    return img.repeat(1, 3, 1, 1)


def make_tetris_pair_sampler(
    *,
    image_shape: Tuple[int, int, int],
    mirror_prob: float = 0.5,
    angle_step: float = 5.0,
    shape_keys=None,
) -> Callable[[torch.device], Tuple[torch.Tensor, torch.Tensor, bool]]:
    c, h, w = image_shape
    keys = list(CHIRAL_TETRIS_SHAPES.keys()) if shape_keys is None else list(shape_keys)

    def sampler(device: torch.device):
        key = random.choice(keys)
        target = get_tetris_tensor(key, h, channels=c)
        is_mirrored = random.random() < mirror_prob
        source = torch.flip(target, dims=[2]) if is_mirrored else target.clone()
        angle = float(random.choice(np.arange(0.0, 360.0, angle_step, dtype=np.float32)))
        source = rotate_tensor(source, angle)
        return source.to(device), target.to(device), bool(is_mirrored)

    return sampler


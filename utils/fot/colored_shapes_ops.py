from __future__ import annotations

import random
from typing import Callable, Tuple

import numpy as np
import torch

from .rotation_ops import rotate_tensor


def random_colored_rectangles(h: int, w: int, *, num_shapes: int, rng: random.Random) -> torch.Tensor:
    """Generate a simple RGB canvas of axis-aligned rectangles in [0,1]."""
    img = torch.zeros((3, h, w), dtype=torch.float32)
    for _ in range(int(num_shapes)):
        color = torch.tensor([rng.random(), rng.random(), rng.random()], dtype=torch.float32).view(3, 1, 1)
        y0 = rng.randint(0, max(0, h - 8))
        x0 = rng.randint(0, max(0, w - 8))
        y1 = min(h, y0 + rng.randint(6, max(7, h // 2)))
        x1 = min(w, x0 + rng.randint(6, max(7, w // 2)))
        img[:, y0:y1, x0:x1] = color
    return img


def make_colored_shapes_pair_sampler(
    *,
    image_shape: Tuple[int, int, int],
    mirror_prob: float = 0.5,
    angle_step: float = 5.0,
    num_shapes: int = 4,
    seed: int = 0,
) -> Callable[[torch.device], Tuple[torch.Tensor, torch.Tensor, bool]]:
    c, h, w = image_shape
    if c != 3:
        raise ValueError(f"Colored shapes expect 3 channels, got {image_shape}")
    rng = random.Random(seed)

    def sampler(device: torch.device):
        target = random_colored_rectangles(h, w, num_shapes=num_shapes, rng=rng)
        is_mirrored = rng.random() < mirror_prob
        source = torch.flip(target, dims=[2]) if is_mirrored else target.clone()
        angle = float(rng.choice(np.arange(0.0, 360.0, angle_step, dtype=np.float32)))
        source = rotate_tensor(source, angle, pad_value=0.0)
        return source.to(device), target.to(device), bool(is_mirrored)

    return sampler


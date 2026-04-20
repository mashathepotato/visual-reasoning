from __future__ import annotations

from typing import List, Optional

import torch
from torch.utils.data import Dataset


class ToyMCQDataset(Dataset):
    """Small deterministic MCQ dataset for smoke-testing VQA training loops.

    Produces random images in [0,1] with fixed-format 4-choice questions.
    """

    def __init__(
        self,
        *,
        n_samples: int,
        image_size: int,
        seed: int = 0,
        num_choices: int = 4,
    ):
        self.n = int(n_samples)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.num_choices = int(num_choices)
        if self.num_choices <= 1:
            raise ValueError("num_choices must be >= 2")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed + int(idx))
        image = torch.rand((3, self.image_size, self.image_size), generator=g, dtype=torch.float32)
        label = int(torch.randint(low=0, high=self.num_choices, size=(1,), generator=g).item())

        question = f"Toy question {int(idx)}: pick the correct option."
        choices: List[str] = [f"Option {chr(ord('A') + i)}" for i in range(self.num_choices)]
        return {"image": image, "question": question, "choices": choices, "label": label}


class ToyMCQWithMaskDataset(Dataset):
    """Toy dataset with an additional supervision mask for smoke-testing mask losses."""

    def __init__(
        self,
        *,
        n_samples: int,
        image_size: int,
        seed: int = 0,
        num_choices: int = 4,
    ):
        self.base = ToyMCQDataset(n_samples=n_samples, image_size=image_size, seed=seed, num_choices=num_choices)
        self.image_size = int(image_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        row = dict(self.base[int(idx)])
        g = torch.Generator().manual_seed(self.seed + 10_000 + int(idx))

        # Random rectangle mask (1,H,W) in {0,1}.
        h = self.image_size
        w = self.image_size
        y0 = int(torch.randint(low=0, high=max(1, h - 8), size=(1,), generator=g).item())
        x0 = int(torch.randint(low=0, high=max(1, w - 8), size=(1,), generator=g).item())
        rh = int(torch.randint(low=4, high=min(32, h - y0) + 1, size=(1,), generator=g).item())
        rw = int(torch.randint(low=4, high=min(32, w - x0) + 1, size=(1,), generator=g).item())
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        mask[:, y0 : y0 + rh, x0 : x0 + rw] = 1.0

        row["target_mask"] = mask
        return row


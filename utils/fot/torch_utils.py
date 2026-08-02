from __future__ import annotations

import random
from typing import Optional
import os

import numpy as np
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 0, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch.

    Deterministic algorithms are requested by default. PyTorch warns rather than
    failing when an operation has no deterministic implementation so that smoke
    tests remain portable across CPU, CUDA, and MPS.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch DataLoader worker from its assigned initial seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seeded_generator(seed: int) -> torch.Generator:
    """Return a CPU generator for deterministic DataLoader shuffling."""
    return torch.Generator().manual_seed(int(seed))


def as_float_tensor(x: torch.Tensor, *, device: Optional[torch.device] = None) -> torch.Tensor:
    t = x.float()
    if device is not None:
        t = t.to(device)
    return t

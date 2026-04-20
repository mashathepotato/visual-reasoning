from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def load_3d_blocks_pairs(repo_root: Path) -> List[Dict[str, Any]]:
    path = repo_root / "data" / "test_balanced.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Create it by running notebooks/pipeline_3d.ipynb or benchmarks/dinov3_3d_baseline.ipynb."
        )
    raw = np.load(str(path), allow_pickle=True)
    return list(raw)


def blocks_pair_to_tensors(pair: Dict[str, Any], *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, str]:
    x0 = torch.tensor(pair["x0"]).float().to(device)  # (1,64,64) in [-1,1]
    x1 = torch.tensor(pair["x1"]).float().to(device)
    label = str(pair.get("label", ""))
    return x0, x1, label


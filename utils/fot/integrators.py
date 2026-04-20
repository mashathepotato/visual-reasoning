from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def apply_heun_steps(
    *,
    model: nn.Module,
    x0: torch.Tensor,
    cond_emb: torch.Tensor,
    target_angle_deg: torch.Tensor,
    steps: int = 10,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    corrector: Optional[nn.Module] = None,
    corrector_weight: float = 0.0,
    corrector_each_step: bool = False,
) -> torch.Tensor:
    """Heun (RK2) integration for the FM rotator-style velocity models."""
    model.eval()
    dt = 1.0 / float(steps)
    curr = x0.clone()

    if curr.dim() == 3:
        curr = curr.unsqueeze(0)
    b = int(curr.shape[0])

    if target_angle_deg.dim() == 1:
        target_angle_deg = target_angle_deg.view(b, 1)

    for i in range(int(steps)):
        t1 = torch.full((b, 1), float(i) / float(steps), device=curr.device, dtype=curr.dtype)
        v1 = model(curr, t1, cond_emb, target_angle_deg)
        mid = curr + (v1 * dt)
        t2 = torch.full((b, 1), float(i + 1) / float(steps), device=curr.device, dtype=curr.dtype)
        v2 = model(mid, t2, cond_emb, target_angle_deg)
        curr = curr + 0.5 * (v1 + v2) * dt

        if corrector_each_step and corrector is not None and float(corrector_weight) != 0.0:
            corr = corrector(curr)
            curr = curr + float(corrector_weight) * corr
            curr = torch.nan_to_num(curr, nan=0.0, posinf=clamp_max, neginf=clamp_min).clamp(clamp_min, clamp_max)

    if (not corrector_each_step) and corrector is not None and float(corrector_weight) != 0.0:
        corr = corrector(curr)
        curr = curr + float(corrector_weight) * corr

    curr = torch.nan_to_num(curr, nan=0.0, posinf=clamp_max, neginf=clamp_min).clamp(clamp_min, clamp_max)
    return curr


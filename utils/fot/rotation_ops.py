from __future__ import annotations

import math
from typing import Union

import kornia as K
import torch
import torch.nn.functional as F


def wrap_angle_deg(angle: float) -> float:
    """Map an angle to [-180, 180) to stay close to the FM training distribution."""
    return ((float(angle) + 180.0) % 360.0) - 180.0


def build_state(source: torch.Tensor, target: torch.Tensor, angle_deg: Union[None, float, torch.Tensor] = None) -> torch.Tensor:
    """Concatenate (source, target, angle_channel) into a state tensor.

    - source, target: (B, C, H, W) or (C, H, W)
    - angle channel is normalized to [0, 1] and broadcast to HxW.
    """
    if source.dim() == 3:
        source = source.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    if target.shape[0] == 1 and source.shape[0] > 1:
        target = target.expand(source.shape[0], -1, -1, -1)

    state = torch.cat([source, target], dim=1)
    if angle_deg is None:
        return state

    if not torch.is_tensor(angle_deg):
        angle = torch.tensor([float(angle_deg)], device=state.device, dtype=state.dtype)
    else:
        angle = angle_deg.to(device=state.device, dtype=state.dtype)

    if angle.dim() == 0:
        angle = angle.unsqueeze(0)
    if angle.numel() == 1 and state.shape[0] > 1:
        angle = angle.repeat(state.shape[0])

    angle_norm = (angle % 360.0) / 360.0
    angle_ch = angle_norm.view(-1, 1, 1, 1).expand(-1, 1, state.shape[2], state.shape[3])
    return torch.cat([state, angle_ch], dim=1)


def rotate_tensor(img: torch.Tensor, angle_deg, *, pad_to_diag: bool = True, pad_value: float = 0.0) -> torch.Tensor:
    """Rotate image(s) by angle(s) in degrees using kornia.

    If pad_to_diag=True, pads to the diagonal size before rotating, then center-crops.
    """
    if img.dim() == 3:
        img_b = img.unsqueeze(0)
    else:
        img_b = img

    if not torch.is_tensor(angle_deg):
        angle = torch.tensor([float(angle_deg)], device=img_b.device, dtype=img_b.dtype)
    else:
        angle = angle_deg.to(device=img_b.device, dtype=img_b.dtype)

    if angle.dim() == 0:
        angle = angle.unsqueeze(0)
    if angle.numel() == 1 and img_b.shape[0] > 1:
        angle = angle.repeat(img_b.shape[0])

    b, c, h, w = img_b.shape
    if pad_to_diag:
        diag = int(math.ceil(math.sqrt(h * h + w * w)))
        pad_h = max(0, diag - h)
        pad_w = max(0, diag - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h > 0 or pad_w > 0:
            img_p = F.pad(img_b, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=float(pad_value))
        else:
            img_p = img_b
    else:
        img_p = img_b

    rotated = K.geometry.transform.rotate(img_p, angle)

    if pad_to_diag:
        y0 = (rotated.shape[2] - h) // 2
        x0 = (rotated.shape[3] - w) // 2
        rotated = rotated[:, :, y0 : y0 + h, x0 : x0 + w]

    return rotated if img.dim() == 4 else rotated.squeeze(0)


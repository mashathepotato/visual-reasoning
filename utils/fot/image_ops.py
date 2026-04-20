from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def to_rgb_tensor01(img) -> torch.Tensor:
    """Convert a PIL Image / numpy array / torch tensor to (3,H,W) float32 in [0,1]."""
    if isinstance(img, torch.Tensor):
        t = img
        if t.dtype != torch.float32:
            t = t.float()
        if t.dim() == 3 and t.shape[0] in (1, 3):
            # (C,H,W)
            pass
        elif t.dim() == 3 and t.shape[-1] == 3:
            # (H,W,3)
            t = t.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported tensor image shape: {tuple(t.shape)}")
        if t.max() > 1.5:
            t = t / 255.0
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        return t.clamp(0.0, 1.0)

    try:
        import PIL.Image

        if isinstance(img, PIL.Image.Image):
            arr = np.array(img.convert("RGB"))
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            return t.clamp(0.0, 1.0)
    except Exception:
        pass

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 array, got {arr.shape}")
        t = torch.from_numpy(arr).permute(2, 0, 1).float()
        if t.max() > 1.5:
            t = t / 255.0
        return t.clamp(0.0, 1.0)

    raise TypeError(f"Unsupported image type: {type(img)}")


def resize_image_tensor(img: torch.Tensor, *, size: int) -> torch.Tensor:
    """Resize (3,H,W) -> (3,size,size) with bilinear; preserves [0,1]."""
    if img.dim() != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected (3,H,W), got {tuple(img.shape)}")
    x = img.unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x[0].clamp(0.0, 1.0)


def bbox_xywh_to_mask(
    *,
    bbox_xywh: Sequence[float],
    original_size_wh: Sequence[int],
    out_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a (1,out_size,out_size) mask from an [x,y,w,h] bbox in original image coordinates."""
    if len(bbox_xywh) != 4:
        raise ValueError(f"Expected 4 bbox values, got {bbox_xywh}")
    if len(original_size_wh) != 2:
        raise ValueError(f"Expected original size [W,H], got {original_size_wh}")
    ow, oh = int(original_size_wh[0]), int(original_size_wh[1])
    if ow <= 0 or oh <= 0:
        raise ValueError(f"Invalid original_size_wh={original_size_wh}")

    x, y, w, h = [float(v) for v in bbox_xywh]
    x0 = max(0.0, x)
    y0 = max(0.0, y)
    x1 = min(float(ow), x0 + max(0.0, w))
    y1 = min(float(oh), y0 + max(0.0, h))

    # Map to out_size coordinates
    sx = out_size / float(ow)
    sy = out_size / float(oh)
    ix0 = int(round(x0 * sx))
    iy0 = int(round(y0 * sy))
    ix1 = int(round(x1 * sx))
    iy1 = int(round(y1 * sy))

    ix0 = max(0, min(out_size - 1, ix0))
    iy0 = max(0, min(out_size - 1, iy0))
    ix1 = max(ix0 + 1, min(out_size, ix1))
    iy1 = max(iy0 + 1, min(out_size, iy1))

    m = torch.zeros((1, out_size, out_size), dtype=torch.float32, device=device)
    m[:, iy0:iy1, ix0:ix1] = 1.0
    return m


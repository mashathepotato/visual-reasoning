from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def create_dinov3(*, device: torch.device) -> nn.Module:
    model = timm.create_model("vit_small_patch16_dinov3", pretrained=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def dino_embed_fm_gray64(img_fm: torch.Tensor, dino_model: nn.Module) -> torch.Tensor:
    """DINOv3 CLS embedding for 64x64 grayscale inputs in [-1, 1] (FM tensor).

    Matches the fm_tetris normalization which broadcasts mean/std across channels.
    """
    if img_fm.dim() == 3:
        img_fm = img_fm.unsqueeze(0)
    # (B, 1, 64, 64) in [-1, 1] -> (B, 3, 224, 224) in [0, 1]
    img = (img_fm * 0.5) + 0.5
    img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
    img = img.repeat(1, 3, 1, 1)
    # Broadcast normalize (single-channel stats), as used in fm_tetris.ipynb
    img = (img - 0.485) / 0.229
    feats = dino_model.forward_features(img)
    return feats[:, 0, :]


@torch.no_grad()
def dino_embed_rgb01(img_rgb01: torch.Tensor, dino_model: nn.Module) -> torch.Tensor:
    """DINOv3 CLS embedding for RGB inputs in [0, 1]."""
    if img_rgb01.dim() == 3:
        img_rgb01 = img_rgb01.unsqueeze(0)
    img = F.interpolate(img_rgb01, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    img = (img - mean) / std
    feats = dino_model.forward_features(img)
    return feats[:, 0, :]


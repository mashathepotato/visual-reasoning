from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallVisionBackbone(nn.Module):
    """Lightweight CNN that returns a spatial feature map."""

    def __init__(self, *, in_ch: int = 3, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HashedTextEncoder(nn.Module):
    def __init__(self, *, num_buckets: int = 50_000, emb_dim: int = 256):
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.emb_dim = int(emb_dim)
        self.emb = nn.Embedding(self.num_buckets, self.emb_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: (B,L), attention_mask: (B,L)
        x = self.emb(input_ids)  # (B,L,D)
        m = attention_mask.unsqueeze(-1)  # (B,L,1)
        x = x * m
        denom = torch.clamp(m.sum(dim=1), min=1.0)
        return x.sum(dim=1) / denom  # (B,D)


class HeatmapSketcher(nn.Module):
    """FoT sketcher: iteratively draws a heatmap (attention mask) conditioned on image+text."""

    def __init__(self, *, img_ch: int = 3, flow_dim: int = 32, text_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, flow_dim * 4),
            nn.GELU(),
            nn.Linear(flow_dim * 4, flow_dim * 4),
        )
        self.img_encoder = nn.Sequential(
            nn.Conv2d(img_ch, flow_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(flow_dim, flow_dim * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.img_proj = nn.Linear(flow_dim * 2, flow_dim * 4)
        self.txt_proj = nn.Linear(text_dim, flow_dim * 4)

        from .models import DoubleConv  # avoid circular import

        self.inc = DoubleConv(1 + img_ch, flow_dim)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(flow_dim, flow_dim * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(flow_dim * 2, flow_dim * 4))

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(flow_dim * 6, flow_dim * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = DoubleConv(flow_dim * 3, flow_dim)

        self.outc = nn.Conv2d(flow_dim, 1, kernel_size=1)

    def forward(self, heatmap_t: torch.Tensor, img: torch.Tensor, t: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_emb = self.time_mlp(t)
        img_c = self.img_encoder(img).squeeze(-1).squeeze(-1)
        img_emb = self.img_proj(img_c)
        txt = self.txt_proj(txt_emb)
        global_cond = (t_emb + img_emb + txt).unsqueeze(-1).unsqueeze(-1)

        x = torch.cat([heatmap_t, img], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = x3 + global_cond
        x = self.conv1(torch.cat([self.up1(x3), x2], dim=1))
        x = self.conv2(torch.cat([self.up2(x), x1], dim=1))
        return self.outc(x)


@torch.no_grad()
def integrate_heatmap(
    *,
    sketcher: HeatmapSketcher,
    img: torch.Tensor,
    txt_emb: torch.Tensor,
    steps: int,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    b, _, h, w = img.shape
    heat = torch.zeros((b, 1, h, w), device=img.device, dtype=img.dtype)
    for i in range(int(steps)):
        t = torch.full((b, 1), float(i) / float(max(1, steps)), device=img.device, dtype=img.dtype)
        delta = sketcher(heat, img, t, txt_emb)
        heat = (heat + delta).clamp(clamp_min, clamp_max)
    return heat


class FoTHeatmapMCQModel(nn.Module):
    """End-to-end MCQ baseline: FoT heatmap sketcher + attended visual features + choice scoring."""

    def __init__(
        self,
        *,
        num_text_buckets: int = 50_000,
        text_dim: int = 256,
        flow_dim: int = 32,
        vision_feat_dim: int = 128,
        sketch_steps: int = 8,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.sketch_steps = int(sketch_steps)

        self.text = HashedTextEncoder(num_buckets=num_text_buckets, emb_dim=text_dim)
        self.vision = SmallVisionBackbone(in_ch=3, feat_dim=vision_feat_dim)
        self.sketcher = HeatmapSketcher(img_ch=3, flow_dim=flow_dim, text_dim=text_dim)

        joint_dim = vision_feat_dim + text_dim + text_dim
        self.scorer = nn.Sequential(
            nn.Linear(joint_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        *,
        images: torch.Tensor,  # (B,3,H,W) in [0,1]
        q_input_ids: torch.Tensor,
        q_attention_mask: torch.Tensor,
        choice_input_ids: torch.Tensor,  # (B,N,L)
        choice_attention_mask: torch.Tensor,  # (B,N,L)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = images.shape
        q_emb = self.text(q_input_ids, q_attention_mask)  # (B,D)

        # Sketch heatmap iteratively
        heat = torch.zeros((b, 1, h, w), device=images.device, dtype=images.dtype)
        for i in range(self.sketch_steps):
            t = torch.full((b, 1), float(i) / float(max(1, self.sketch_steps)), device=images.device, dtype=images.dtype)
            delta = self.sketcher(heat, images, t, q_emb)
            heat = (heat + delta).clamp(0.0, 1.0)

        # Attend over a lower-res visual feature map
        feat_map = self.vision(images)  # (B,C,h',w')
        heat_small = F.interpolate(heat, size=feat_map.shape[-2:], mode="bilinear", align_corners=False)
        weights = torch.clamp(heat_small, 0.0, 1.0)
        denom = torch.clamp(weights.sum(dim=(2, 3)), min=1e-6)  # (B,1)
        vis = (feat_map * weights).sum(dim=(2, 3)) / denom  # (B,C)

        # Encode choices and score
        bn, n, l = choice_input_ids.shape
        flat_ids = choice_input_ids.view(bn * n, l)
        flat_mask = choice_attention_mask.view(bn * n, l)
        c_emb = self.text(flat_ids, flat_mask).view(bn, n, -1)  # (B,N,D)

        vis_rep = vis.unsqueeze(1).expand(-1, n, -1)
        q_rep = q_emb.unsqueeze(1).expand(-1, n, -1)
        joint = torch.cat([vis_rep, q_rep, c_emb], dim=-1)
        logits = self.scorer(joint).squeeze(-1)  # (B,N)
        return logits, heat


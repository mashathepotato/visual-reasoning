from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastRotator(nn.Module):
    """U-Net velocity model conditioned on (t, cond_emb, target_angle).

    This matches the notebook definitions so that existing checkpoints under `models/`
    can be loaded without key-mismatch.
    """

    def __init__(
        self,
        *,
        in_ch: int = 1,
        out_ch: int = 1,
        backbone_dim: int = 384,
        flow_dim: int = 64,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, flow_dim * 4),
            nn.GELU(),
            nn.Linear(flow_dim * 4, flow_dim * 4),
        )
        self.angle_mlp = nn.Sequential(
            nn.Linear(1, flow_dim * 4),
            nn.GELU(),
            nn.Linear(flow_dim * 4, flow_dim * 4),
        )
        self.cond_proj = nn.Linear(backbone_dim, flow_dim * 4)

        self.inc = DoubleConv(in_ch, flow_dim)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(flow_dim, flow_dim * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(flow_dim * 2, flow_dim * 4))

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(flow_dim * 6, flow_dim * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = DoubleConv(flow_dim * 3, flow_dim)

        self.outc = nn.Conv2d(flow_dim, out_ch, kernel_size=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
        target_angle_deg: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        c_emb = self.cond_proj(cond_emb)
        a_emb = self.angle_mlp(target_angle_deg / 360.0)
        global_cond = (t_emb + c_emb + a_emb).unsqueeze(-1).unsqueeze(-1)

        x1 = self.inc(x_t)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = x3 + global_cond

        x = self.conv1(torch.cat([self.up1(x3), x2], dim=1))
        x = self.conv2(torch.cat([self.up2(x), x1], dim=1))
        return self.outc(x)


class CorrectorUNet(nn.Module):
    def __init__(self, *, in_ch: int = 1, out_ch: int = 1, base_ch: int = 16):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 2, base_ch * 4))
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(base_ch * 6, base_ch * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = DoubleConv(base_ch * 3, base_ch)
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.conv1(torch.cat([self.up1(x3), x2], dim=1))
        x = self.conv2(torch.cat([self.up2(x), x1], dim=1))
        return self.outc(x)


class CondEncoder(nn.Module):
    """Small CNN encoder used in the colored-rectangles FM rotator notebook."""

    def __init__(self, *, in_ch: int = 3, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1).squeeze(-1)
        return self.proj(h)


class MazeSketcher(nn.Module):
    """Time-conditioned sketcher for maze traces.

    Input: current trace mask + conditioning (walls/start/goal), output: delta mask.
    """

    def __init__(self, *, cond_ch: int = 3, flow_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, flow_dim * 4),
            nn.GELU(),
            nn.Linear(flow_dim * 4, flow_dim * 4),
        )
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_ch, flow_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(flow_dim, flow_dim * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cond_proj = nn.Linear(flow_dim * 2, flow_dim * 4)

        self.inc = DoubleConv(1 + cond_ch, flow_dim)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(flow_dim, flow_dim * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(flow_dim * 2, flow_dim * 4))

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(flow_dim * 6, flow_dim * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = DoubleConv(flow_dim * 3, flow_dim)

        self.outc = nn.Conv2d(flow_dim, 1, kernel_size=1)

    def forward(self, trace_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_emb = self.time_mlp(t)
        c = self.cond_encoder(cond).squeeze(-1).squeeze(-1)
        c_emb = self.cond_proj(c)
        global_cond = (t_emb + c_emb).unsqueeze(-1).unsqueeze(-1)

        x = torch.cat([trace_t, cond], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = x3 + global_cond

        x = self.conv1(torch.cat([self.up1(x3), x2], dim=1))
        x = self.conv2(torch.cat([self.up2(x), x1], dim=1))
        return self.outc(x)


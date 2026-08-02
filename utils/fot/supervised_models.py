"""Compact supervised baselines for pair-image classification.

Both models expect an NCHW tensor containing two images concatenated along the
channel dimension.  For RGB pairs this gives the default six input channels.
The models are intentionally self-contained and do not require pretrained
weights, timm, or network access.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Return the number of scalar parameters in ``model``.

    Args:
        model: Module whose parameters should be counted.
        trainable_only: If true, exclude parameters with ``requires_grad=False``.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _validate_pair_input(
    x: torch.Tensor,
    *,
    in_channels: int,
    image_size: int,
    model_name: str,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{model_name} expected a torch.Tensor, got {type(x).__name__}.")
    if x.ndim != 4:
        raise ValueError(f"{model_name} expected an NCHW tensor, got shape {tuple(x.shape)}.")
    if x.shape[1] != in_channels:
        raise ValueError(
            f"{model_name} expected {in_channels} channels, got {x.shape[1]}. "
            "Concatenate the two images along the channel dimension."
        )
    if tuple(x.shape[-2:]) != (image_size, image_size):
        raise ValueError(
            f"{model_name} expected {image_size}x{image_size} inputs, "
            f"got {x.shape[-2]}x{x.shape[-1]}."
        )
    if not x.is_floating_point():
        raise TypeError(f"{model_name} expected a floating-point input tensor, got {x.dtype}.")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PairCNN(nn.Module):
    """Compact CNN for binary pair-image classification.

    Args:
        image_size: Required square input resolution.
        in_channels: Channels after concatenating both images (six for RGB).
        num_classes: Number of output logits; use two for binary cross-entropy
            via ``torch.nn.functional.cross_entropy``.
        widths: Output channels for successive convolutional stages.
    """

    def __init__(
        self,
        *,
        image_size: int = 64,
        in_channels: int = 6,
        num_classes: int = 2,
        widths: Sequence[int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("widths must contain positive integers.")

        self.image_size = int(image_size)
        self.in_channels = int(in_channels)

        stages = []
        current_channels = self.in_channels
        for width in widths:
            width = int(width)
            stages.append(_ConvBlock(current_channels, width, stride=2))
            current_channels = width

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(current_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_pair_input(
            x,
            in_channels=self.in_channels,
            image_size=self.image_size,
            model_name=type(self).__name__,
        )
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class PairVisionTransformer(nn.Module):
    """Small from-scratch Vision Transformer for pair-image classification.

    Args:
        image_size: Required square input resolution.
        patch_size: Width and height of each non-overlapping image patch.
        in_channels: Channels after concatenating both images (six for RGB).
        num_classes: Number of output logits.
        embed_dim: Transformer token dimension.
        depth: Number of Transformer encoder layers.
        num_heads: Number of attention heads per encoder layer.
        mlp_ratio: Hidden-to-embedding dimension ratio in feed-forward blocks.
        dropout: Dropout probability. The default is zero for a deterministic
            baseline architecture.
    """

    def __init__(
        self,
        *,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 6,
        num_classes: int = 2,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size <= 0 or patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive.")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")
        if embed_dim <= 0 or depth <= 0 or num_heads <= 0:
            raise ValueError("embed_dim, depth, and num_heads must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)

        grid_size = self.image_size // self.patch_size
        num_patches = grid_size * grid_size
        hidden_dim = int(embed_dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError("embed_dim * mlp_ratio must produce a positive hidden dimension.")

        self.patch_embed = nn.Conv2d(
            self.in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_pair_input(
            x,
            in_channels=self.in_channels,
            image_size=self.image_size,
            model_name=type(self).__name__,
        )
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.encoder(tokens)
        return self.head(self.norm(tokens[:, 0]))

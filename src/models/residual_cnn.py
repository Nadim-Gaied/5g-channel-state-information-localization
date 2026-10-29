"""Residual CNN sized to match the deeper CSI Vision Transformer."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


def downsample(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class CSIResidualCNN(nn.Module):
    """Three-stage residual CNN for coordinate regression."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualBlock(32),
            downsample(32, 64),
            ResidualBlock(64),
            downsample(64, 96),
            ResidualBlock(96),
            nn.AdaptiveAvgPool2d((1, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4, 304),
            nn.LayerNorm(304),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(304, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.backbone(x))

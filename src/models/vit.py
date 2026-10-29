"""Small Vision Transformer adapted to antenna-by-subcarrier CSI matrices."""

from __future__ import annotations

import torch
from torch import nn


class CSIVisionTransformer(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int] = (16, 32),
        patch_size: tuple[int, int] = (4, 4),
        embedding_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        height, width = input_size
        patch_h, patch_w = patch_size
        if height % patch_h or width % patch_w:
            raise ValueError("Input dimensions must be divisible by patch dimensions")

        self.patch_embedding = nn.Conv2d(
            2,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (height // patch_h) * (width // patch_w)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embedding_dim)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 2))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.position_embedding
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])

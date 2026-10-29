"""Map-conditioned localization model for a variable set of BTS observations."""

from __future__ import annotations

import torch
from torch import nn


class SharedCSIEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, embedding_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, embedding_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        batch, stations = csi.shape[:2]
        encoded = self.network(csi.flatten(0, 1)).flatten(1)
        return encoded.view(batch, stations, -1)


class MetricMapEncoder(nn.Module):
    """Encode occupancy/height plus explicit normalized coordinate channels."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, embedding_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embedding_dim, embedding_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, raster: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = raster.shape
        y = torch.linspace(-1.0, 1.0, height, device=raster.device, dtype=raster.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=raster.device, dtype=raster.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((grid_x, grid_y)).expand(batch, -1, -1, -1)
        feature_map = self.network(torch.cat((raster, coordinates), dim=1))
        return feature_map.flatten(2).transpose(1, 2)


class MultiMapLocalizer(nn.Module):
    """Fuse CSI/BTS set tokens with a metric map using cross-attention."""

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        set_depth: int = 2,
        fusion: str = "add",
    ) -> None:
        super().__init__()
        if fusion not in {"add", "concat_mlp", "gated"}:
            raise ValueError(f"Unknown CSI/BTS fusion: {fusion!r}")
        self.fusion = fusion
        self.csi_encoder = SharedCSIEncoder(embedding_dim)
        self.bs_position_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.radio_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.station_set_encoder = nn.TransformerEncoder(layer, num_layers=set_depth)
        self.map_encoder = MetricMapEncoder(embedding_dim)
        self.map_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 2),
            nn.Tanh(),
        )
        nn.init.trunc_normal_(self.radio_token, std=0.02)
        # Register learned fusion after the historical modules: seeded shared
        # initialization and the complete add state_dict remain unchanged.
        if fusion != "add":
            # Keep the caller's CPU RNG stream aligned with add: otherwise
            # extra initialization changes subsequent shuffling/BTS sampling.
            with torch.random.fork_rng(devices=[]):
                self.fusion_network = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )

    def forward(
        self,
        csi: torch.Tensor,
        bs_positions: torch.Tensor,
        bs_mask: torch.Tensor,
        map_raster: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.all(bs_mask.any(dim=1)):
            raise ValueError("Every sample must contain at least one active BTS")
        csi_tokens = self.csi_encoder(csi)
        position_tokens = self.bs_position_encoder(bs_positions)
        if self.fusion == "add":
            station_tokens = csi_tokens + position_tokens
        else:
            joint = torch.cat((csi_tokens, position_tokens), dim=-1)
            learned = self.fusion_network(joint)
            if self.fusion == "concat_mlp":
                station_tokens = learned
            else:
                gate = torch.sigmoid(learned)
                station_tokens = gate * csi_tokens + (1.0 - gate) * position_tokens
        radio_token = self.radio_token.expand(len(csi), -1, -1)
        tokens = torch.cat((radio_token, station_tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(len(csi), 1, dtype=torch.bool, device=csi.device),
                ~bs_mask,
            ),
            dim=1,
        )
        encoded = self.station_set_encoder(tokens, src_key_padding_mask=padding_mask)
        radio_context = encoded[:, :1]
        map_tokens = self.map_encoder(map_raster)
        map_context, _ = self.map_attention(radio_context, map_tokens, map_tokens)
        fused = torch.cat((radio_context[:, 0], map_context[:, 0]), dim=1)
        return self.head(fused)

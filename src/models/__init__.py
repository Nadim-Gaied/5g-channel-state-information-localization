"""Model factory for CSI localization experiments."""

from __future__ import annotations

from .cnn import CSIConvNet
from .residual_cnn import CSIResidualCNN
from .vit import CSIVisionTransformer
from .multimap import MultiMapLocalizer


def build_model(
    name: str,
    input_size: tuple[int, int] = (16, 32),
    patch_size: tuple[int, int] | None = None,
):
    """Build a model for a given antenna-by-subcarrier input shape."""
    if name == "cnn":
        return CSIConvNet()
    if name == "vit":
        return CSIVisionTransformer(
            input_size=input_size,
            patch_size=patch_size or (4, 4),
        )
    if name == "cnn_residual":
        return CSIResidualCNN()
    if name == "vit_deep":
        return CSIVisionTransformer(
            input_size=input_size,
            patch_size=patch_size or (4, 4),
            embedding_dim=96,
            depth=4,
            num_heads=6,
        )
    raise ValueError(f"Unknown model {name!r}")


__all__ = [
    "CSIConvNet",
    "CSIResidualCNN",
    "CSIVisionTransformer",
    "MultiMapLocalizer",
    "build_model",
]

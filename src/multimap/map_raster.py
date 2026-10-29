"""Rasterize DeepMIMO v4 scene geometry into metric bird's-eye-view maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class MapBounds:
    """Axis-aligned map bounds in the ray-tracer coordinate frame."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width_m(self) -> float:
        return self.xmax - self.xmin

    @property
    def height_m(self) -> float:
        return self.ymax - self.ymin

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.xmin, self.ymin, self.xmax, self.ymax], dtype=np.float32
        )


def centered_bounds(
    points_xy: np.ndarray,
    raster_size: tuple[int, int],
    resolution_m: float,
) -> MapBounds:
    """Create fixed-size metric bounds centered on a collection of points."""
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or not len(points_xy):
        raise ValueError("points_xy must have shape [N, 2] and cannot be empty")
    height, width = raster_size
    center = (points_xy.min(axis=0) + points_xy.max(axis=0)) / 2.0
    half_width = width * resolution_m / 2.0
    half_height = height * resolution_m / 2.0
    required_span = points_xy.max(axis=0) - points_xy.min(axis=0)
    available_span = np.asarray([2.0 * half_width, 2.0 * half_height])
    if np.any(required_span > available_span):
        raise ValueError(
            f"Points span {required_span.tolist()} m but the raster covers "
            f"only {available_span.tolist()} m"
        )
    return MapBounds(
        xmin=float(center[0] - half_width),
        ymin=float(center[1] - half_height),
        xmax=float(center[0] + half_width),
        ymax=float(center[1] + half_height),
    )


def normalize_xy(points_xy: np.ndarray, bounds: MapBounds) -> np.ndarray:
    """Convert world coordinates to the common [-1, 1] map frame."""
    points = np.asarray(points_xy, dtype=np.float32)
    lower = np.asarray([bounds.xmin, bounds.ymin], dtype=np.float32)
    span = np.asarray([bounds.width_m, bounds.height_m], dtype=np.float32)
    return 2.0 * (points - lower) / span - 1.0


def _pixel_polygon(
    vertices_xy: np.ndarray, bounds: MapBounds, raster_size: tuple[int, int]
) -> list[tuple[float, float]]:
    height, width = raster_size
    x = (vertices_xy[:, 0] - bounds.xmin) / bounds.width_m * (width - 1)
    y = (bounds.ymax - vertices_xy[:, 1]) / bounds.height_m * (height - 1)
    return list(zip(x.tolist(), y.tolist(), strict=True))


def rasterize_scene(
    scene,
    bounds: MapBounds,
    raster_size: tuple[int, int] = (256, 256),
    height_scale_m: float = 50.0,
) -> np.ndarray:
    """Return `[occupancy, normalized_height]` for building objects."""
    if height_scale_m <= 0:
        raise ValueError("height_scale_m must be positive")
    height, width = raster_size
    occupancy = Image.new("F", (width, height), 0.0)
    height_map = np.zeros((height, width), dtype=np.float32)

    for element in scene.objects:
        if str(element.label).lower() != "buildings":
            continue
        # Some Wireless InSite exports classify large, flat auxiliary surfaces
        # as buildings. They are not obstacles and otherwise dominate the
        # occupancy raster while contributing zero height.
        if float(element.height) <= 0.5:
            continue
        element_mask = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(element_mask)
        for face in element.faces:
            vertices = np.asarray(face.vertices, dtype=np.float32)
            if len(vertices) < 3:
                continue
            polygon = _pixel_polygon(vertices[:, :2], bounds, raster_size)
            draw.polygon(polygon, fill=1)
        mask = np.asarray(element_mask, dtype=bool)
        if not mask.any():
            continue
        ImageDraw.Draw(occupancy).bitmap((0, 0), element_mask, fill=1.0)
        normalized_height = min(float(element.height) / height_scale_m, 1.0)
        height_map[mask] = np.maximum(height_map[mask], normalized_height)

    return np.stack(
        (np.asarray(occupancy, dtype=np.float32), height_map), axis=0
    )

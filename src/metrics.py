"""Localization metrics expressed in metres."""

from __future__ import annotations

import numpy as np


def localization_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    errors = np.linalg.norm(predictions - targets, axis=1)
    return {
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
    }

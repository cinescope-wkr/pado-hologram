"""Backward-compatible re-exports for target representations."""

from .core.targets import IntensityTarget, MultiPlaneIntensityTarget, normalize_mean_intensity

__all__ = [
    "IntensityTarget",
    "MultiPlaneIntensityTarget",
    "normalize_mean_intensity",
]

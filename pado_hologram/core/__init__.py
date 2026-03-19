"""Core holography abstractions for PADO Hologram."""

from .losses import (
    amplitude_mse,
    as_intensity_tensor,
    intensity_mse,
    multi_plane_intensity_mse,
    multi_plane_reconstruction_metrics,
    reconstruction_metrics,
    tensor_reconstruction_metrics,
)
from .pipelines import (
    HologramForwardResult,
    HologramPipeline,
    MultiPlaneHologramForwardResult,
    MultiPlaneHologramPipeline,
)
from .specs import PropagationSpec, SourceSpec
from .targets import IntensityTarget, MultiPlaneIntensityTarget, normalize_mean_intensity

__all__ = [
    "PropagationSpec",
    "SourceSpec",
    "IntensityTarget",
    "MultiPlaneIntensityTarget",
    "normalize_mean_intensity",
    "as_intensity_tensor",
    "intensity_mse",
    "amplitude_mse",
    "reconstruction_metrics",
    "tensor_reconstruction_metrics",
    "multi_plane_intensity_mse",
    "multi_plane_reconstruction_metrics",
    "HologramForwardResult",
    "HologramPipeline",
    "MultiPlaneHologramForwardResult",
    "MultiPlaneHologramPipeline",
]

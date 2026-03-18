"""PADO Hologram.

An emerging holography-oriented layer built on top of the PADO optics core.
"""

from .algorithms import (
    DPACResult,
    DoublePhaseAmplitudeCoder,
    GerchbergSaxtonPhaseOptimizer,
    GerchbergSaxtonResult,
)
from .config import PropagationSpec, SourceSpec
from .experiment import ExperimentSummary, run_experiment
from .losses import (
    amplitude_mse,
    intensity_mse,
    multi_plane_intensity_mse,
    multi_plane_reconstruction_metrics,
    reconstruction_metrics,
)
from .pipeline import (
    HologramForwardResult,
    HologramPipeline,
    MultiPlaneHologramForwardResult,
    MultiPlaneHologramPipeline,
)
from .slm import PhaseEncodingConfig, PhaseEncodingResult, PhaseOnlyLCOSSLM
from .targets import IntensityTarget, MultiPlaneIntensityTarget, normalize_mean_intensity

PROJECT_NAME = "PADO Hologram"
PACKAGE_NAME = "pado_hologram"
DESCRIPTION = "An open-source computer-generated holography framework built on top of PADO."
MAINTAINER = "Jinwoo Lee"
MAINTAINER_EMAIL = "cinescope@kaist.ac.kr"
__version__ = "0.2.0"

__all__ = [
    "PROJECT_NAME",
    "PACKAGE_NAME",
    "DESCRIPTION",
    "MAINTAINER",
    "MAINTAINER_EMAIL",
    "__version__",
    "DPACResult",
    "DoublePhaseAmplitudeCoder",
    "GerchbergSaxtonPhaseOptimizer",
    "GerchbergSaxtonResult",
    "PropagationSpec",
    "SourceSpec",
    "amplitude_mse",
    "intensity_mse",
    "multi_plane_intensity_mse",
    "multi_plane_reconstruction_metrics",
    "reconstruction_metrics",
    "HologramForwardResult",
    "HologramPipeline",
    "MultiPlaneHologramForwardResult",
    "MultiPlaneHologramPipeline",
    "PhaseEncodingConfig",
    "PhaseEncodingResult",
    "PhaseOnlyLCOSSLM",
    "IntensityTarget",
    "MultiPlaneIntensityTarget",
    "normalize_mean_intensity",
    "ExperimentSummary",
    "run_experiment",
]

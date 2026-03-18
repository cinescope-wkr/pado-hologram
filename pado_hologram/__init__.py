"""PADO Hologram.

An emerging holography-oriented layer built on top of the PADO optics core.
"""

from .algorithms import (
    DPACResult,
    DoublePhaseAmplitudeCoder,
    GerchbergSaxtonPhaseOptimizer,
    GerchbergSaxtonResult,
)
from .backends import (
    DEFAULT_WARP_CACHE_DIR,
    KernelBackendSelection,
    SUPPORTED_KERNEL_BACKENDS,
    checkerboard_phase_select,
    resolve_kernel_backend,
    warp_checkerboard_mask,
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
    "DEFAULT_WARP_CACHE_DIR",
    "DoublePhaseAmplitudeCoder",
    "GerchbergSaxtonPhaseOptimizer",
    "GerchbergSaxtonResult",
    "KernelBackendSelection",
    "PropagationSpec",
    "SourceSpec",
    "SUPPORTED_KERNEL_BACKENDS",
    "amplitude_mse",
    "checkerboard_phase_select",
    "intensity_mse",
    "multi_plane_intensity_mse",
    "multi_plane_reconstruction_metrics",
    "reconstruction_metrics",
    "resolve_kernel_backend",
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
    "warp_checkerboard_mask",
    "ExperimentSummary",
    "run_experiment",
]

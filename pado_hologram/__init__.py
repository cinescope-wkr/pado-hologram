"""PADO Hologram."""

from .backends import (
    DEFAULT_WARP_CACHE_DIR,
    KernelBackendSelection,
    SUPPORTED_KERNEL_BACKENDS,
    checkerboard_phase_select,
    resolve_kernel_backend,
    warp_checkerboard_mask,
)
from .core import (
    HologramForwardResult,
    HologramPipeline,
    IntensityTarget,
    MultiPlaneHologramForwardResult,
    MultiPlaneHologramPipeline,
    MultiPlaneIntensityTarget,
    PropagationSpec,
    SourceSpec,
    amplitude_mse,
    intensity_mse,
    multi_plane_intensity_mse,
    multi_plane_reconstruction_metrics,
    normalize_mean_intensity,
    reconstruction_metrics,
    tensor_reconstruction_metrics,
)
from .devices import CameraObservationSpec, PhaseEncodingConfig, PhaseEncodingResult, PhaseOnlyLCOSSLM
from .experiments import ExperimentSummary, run_experiment
from .phase_only import (
    DPACResult,
    DoublePhaseAmplitudeCoder,
    GerchbergSaxtonPhaseOptimizer,
    GerchbergSaxtonResult,
)
from .primitive_based import (
    PrimitiveFieldRenderResult,
    available_primitive_backends,
    available_primitive_renderers,
    build_primitive_scene_from_config,
    render_gaussian_scene,
    render_gaussian_scene_gws_exact,
    render_gaussian_scene_gws_exact_awb,
    render_gaussian_scene_gws_rpws_exact,
    render_gaussian_scene_splat,
    render_gaussian_scene_wave,
    render_gaussian_scene_wave_awb,
)
from .representations import (
    GaussianPrimitive2D,
    GaussianPrimitive3D,
    GaussianWavePrimitive2D,
    PointPrimitive2D,
    PrimitiveScene2D,
)

PROJECT_NAME = "PADO Hologram"
PACKAGE_NAME = "pado_hologram"
DESCRIPTION = (
    "An open-source computer-generated holography framework built on top of PADO, "
    "a PyTorch differentiable optics library."
)
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
    "GaussianPrimitive2D",
    "GaussianPrimitive3D",
    "GaussianWavePrimitive2D",
    "KernelBackendSelection",
    "PointPrimitive2D",
    "PropagationSpec",
    "PrimitiveFieldRenderResult",
    "PrimitiveScene2D",
    "SourceSpec",
    "SUPPORTED_KERNEL_BACKENDS",
    "CameraObservationSpec",
    "amplitude_mse",
    "available_primitive_backends",
    "available_primitive_renderers",
    "build_primitive_scene_from_config",
    "checkerboard_phase_select",
    "intensity_mse",
    "multi_plane_intensity_mse",
    "multi_plane_reconstruction_metrics",
    "reconstruction_metrics",
    "tensor_reconstruction_metrics",
    "render_gaussian_scene",
    "render_gaussian_scene_gws_exact",
    "render_gaussian_scene_gws_exact_awb",
    "render_gaussian_scene_gws_rpws_exact",
    "render_gaussian_scene_splat",
    "render_gaussian_scene_wave",
    "render_gaussian_scene_wave_awb",
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

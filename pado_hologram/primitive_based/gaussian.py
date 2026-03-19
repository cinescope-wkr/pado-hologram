from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping

import torch

from ..core.specs import PropagationSpec, SourceSpec
from ..representations import (
    GaussianPrimitive2D,
    GaussianPrimitive3D,
    GaussianWavePrimitive2D,
    PointPrimitive2D,
    PrimitiveScene2D,
)
from .backends import KernelBackendSelection, available_primitive_backends, resolve_primitive_backend
from .exact import (
    apply_phase_compensation,
    build_angular_emission_profile,
    exact_projected_gaussian_wavefronts_batched,
    make_frequency_grid,
    project_gaussians3d_to_hologram_space,
    stack_gaussian3d_parameters,
    sample_structured_random_phase_kernels,
)
from .warp import build_projected_gaussian_warp_params, render_gaussian_scene_warp, render_projected_gaussian_wavefronts_warp


@dataclass(frozen=True)
class PrimitiveFieldRenderResult:
    """Store the rendered complex field and derived quantities for a primitive scene."""

    field: torch.Tensor
    intensity: torch.Tensor
    scene: PrimitiveScene2D
    renderer: str
    backend: str
    backend_reason: str
    frame_fields: tuple[torch.Tensor, ...] | None = None

    @property
    def amplitude(self) -> torch.Tensor:
        return self.field.abs()

    @property
    def phase(self) -> torch.Tensor:
        return torch.angle(self.field)

    @property
    def dim(self) -> tuple[int, int, int, int]:
        return tuple(int(v) for v in self.field.shape)

    @property
    def num_frames(self) -> int:
        return len(self.frame_fields) if self.frame_fields is not None else 1


def available_primitive_renderers() -> tuple[str, ...]:
    return (
        "gaussian_naive",
        "gaussian_splat",
        "gaussian_wave",
        "gaussian_wave_awb",
        "gaussian_gws_exact",
        "gaussian_gws_exact_awb",
        "gaussian_gws_rpws_exact",
    )


def _coerce_scene_mapping(cfg: Mapping[str, object] | None) -> Mapping[str, object]:
    cfg = cfg or {}
    path = cfg.get("path")
    if path is None:
        return cfg
    with open(str(path), "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"primitive scene file must contain a mapping, got {type(loaded)}")
    merged = dict(loaded)
    for key, value in cfg.items():
        if key != "path":
            merged[key] = value
    return merged


def _require_plane_shape(dim: tuple[int, int, int, int] | None) -> tuple[int, int]:
    if dim is None:
        raise ValueError("primitive scene config requires dim when using presets or normalized coordinates")
    return int(dim[2]), int(dim[3])


def _require_pitch(pitch: float | None) -> float:
    if pitch is None or pitch <= 0:
        raise ValueError("primitive scene config requires a positive pitch for physical 3D presets")
    return float(pitch)


def _resolve_coord(value: float, size: int, coordinate_system: str) -> float:
    if coordinate_system == "pixel":
        return float(value)
    if coordinate_system == "normalized":
        return float(value) * float(size - 1)
    raise ValueError(f"unsupported coordinate_system {coordinate_system!r}")


def _resolve_center_yx(
    value: object,
    rows: int,
    cols: int,
    coordinate_system: str,
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"center specification must have length 2, got {value}")
    return (
        _resolve_coord(float(value[0]), rows, coordinate_system),
        _resolve_coord(float(value[1]), cols, coordinate_system),
    )


def _make_ring_scene(
    cfg: Mapping[str, object],
    *,
    dim: tuple[int, int, int, int] | None,
) -> PrimitiveScene2D:
    rows, cols = _require_plane_shape(dim)
    coordinate_system = str(cfg.get("coordinate_system", "normalized"))
    center_yx = _resolve_center_yx(cfg.get("center_yx", (0.5, 0.5)), rows, cols, coordinate_system)
    radius = float(cfg.get("radius", 0.25))
    radius_px = radius * min(rows - 1, cols - 1) if coordinate_system == "normalized" else radius
    count = int(cfg.get("count", 8))
    sigma_yx = tuple(float(v) for v in cfg.get("sigma_yx", (1.75, 1.75)))
    amplitude = float(cfg.get("amplitude", 1.0))
    phase_start = float(cfg.get("phase_start", 0.0))
    phase_step = float(cfg.get("phase_step", 0.0))

    gaussians = []
    for idx in range(count):
        theta = 2.0 * math.pi * (idx / max(count, 1))
        gaussians.append(
            GaussianPrimitive2D(
                center_yx=(
                    center_yx[0] + radius_px * math.sin(theta),
                    center_yx[1] + radius_px * math.cos(theta),
                ),
                sigma_yx=sigma_yx,
                amplitude=amplitude,
                phase=phase_start + phase_step * idx,
                rotation=theta,
            )
        )
    return PrimitiveScene2D(
        gaussians=tuple(gaussians),
        name=str(cfg.get("name", "gaussian_ring")),
    )


def _make_grid_scene(
    cfg: Mapping[str, object],
    *,
    dim: tuple[int, int, int, int] | None,
) -> PrimitiveScene2D:
    rows, cols = _require_plane_shape(dim)
    coordinate_system = str(cfg.get("coordinate_system", "normalized"))
    grid_shape = tuple(int(v) for v in cfg.get("grid_shape", (3, 3)))
    if len(grid_shape) != 2:
        raise ValueError(f"grid_shape must have length 2, got {grid_shape}")
    padding = tuple(float(v) for v in cfg.get("padding", (0.15, 0.15)))
    sigma_yx = tuple(float(v) for v in cfg.get("sigma_yx", (1.5, 1.5)))
    amplitude = float(cfg.get("amplitude", 1.0))
    phase = float(cfg.get("phase", 0.0))

    y0 = _resolve_coord(padding[0], rows, coordinate_system)
    x0 = _resolve_coord(padding[1], cols, coordinate_system)
    if coordinate_system == "normalized":
        y1 = _resolve_coord(1.0 - padding[0], rows, coordinate_system)
        x1 = _resolve_coord(1.0 - padding[1], cols, coordinate_system)
    else:
        y1 = rows - 1 - padding[0]
        x1 = cols - 1 - padding[1]

    ys = torch.linspace(float(y0), float(y1), grid_shape[0]).tolist()
    xs = torch.linspace(float(x0), float(x1), grid_shape[1]).tolist()
    gaussians = tuple(
        GaussianPrimitive2D(center_yx=(y, x), sigma_yx=sigma_yx, amplitude=amplitude, phase=phase)
        for y in ys
        for x in xs
    )
    return PrimitiveScene2D(
        gaussians=gaussians,
        name=str(cfg.get("name", "gaussian_grid")),
    )


def _make_depth_ring_scene(
    cfg: Mapping[str, object],
    *,
    dim: tuple[int, int, int, int] | None,
) -> PrimitiveScene2D:
    rows, cols = _require_plane_shape(dim)
    coordinate_system = str(cfg.get("coordinate_system", "normalized"))
    center_yx = _resolve_center_yx(cfg.get("center_yx", (0.5, 0.5)), rows, cols, coordinate_system)
    radius = float(cfg.get("radius", 0.2))
    radius_px = radius * min(rows - 1, cols - 1) if coordinate_system == "normalized" else radius
    count = int(cfg.get("count", 6))
    sigma_yx = tuple(float(v) for v in cfg.get("sigma_yx", (1.5, 1.5)))
    amplitude = float(cfg.get("amplitude", 1.0))
    opacity = float(cfg.get("opacity", 0.65))
    phase_start = float(cfg.get("phase_start", 0.0))
    phase_step = float(cfg.get("phase_step", 0.0))
    depth_start = float(cfg.get("depth_start", 1.0e-3))
    depth_step = float(cfg.get("depth_step", 2.5e-4))

    wave_gaussians = []
    for idx in range(count):
        theta = 2.0 * math.pi * (idx / max(count, 1))
        wave_gaussians.append(
            GaussianWavePrimitive2D(
                center_yx=(
                    center_yx[0] + radius_px * math.sin(theta),
                    center_yx[1] + radius_px * math.cos(theta),
                ),
                sigma_yx=sigma_yx,
                depth=depth_start + depth_step * idx,
                amplitude=amplitude,
                opacity=opacity,
                phase=phase_start + phase_step * idx,
                rotation=theta,
            )
        )
    return PrimitiveScene2D(
        wave_gaussians=tuple(wave_gaussians),
        name=str(cfg.get("name", "gaussian_depth_ring")),
    )


def _make_gaussian3d_depth_ring_scene(
    cfg: Mapping[str, object],
    *,
    pitch: float | None,
) -> PrimitiveScene2D:
    pitch_value = _require_pitch(pitch)
    count = int(cfg.get("count", 8))
    radius_pixels = float(cfg.get("radius_pixels", 5.0))
    radius_m = radius_pixels * pitch_value
    scale_pixels = tuple(float(v) for v in cfg.get("scale_pixels", (1.6, 1.6, 1.0e-4)))
    if len(scale_pixels) != 3:
        raise ValueError(f"scale_pixels must have length 3, got {scale_pixels}")
    scale_xyz = (
        scale_pixels[0] * pitch_value,
        scale_pixels[1] * pitch_value,
        max(scale_pixels[2] * pitch_value, torch.finfo(torch.float32).eps),
    )
    amplitude = float(cfg.get("amplitude", 1.0))
    opacity = float(cfg.get("opacity", 0.7))
    phase_start = float(cfg.get("phase_start", 0.0))
    phase_step = float(cfg.get("phase_step", 0.0))
    depth_start = float(cfg.get("depth_start", 1.0e-3))
    depth_step = float(cfg.get("depth_step", 2.5e-4))
    projection_cfg = cfg.get("projection") or {}
    projection_focal_px = tuple(float(v) for v in projection_cfg.get("focal_lengths_px", (160.0, 160.0)))
    projection_principal_raw = projection_cfg.get("principal_point_px")
    projection_principal_px = (
        tuple(float(v) for v in projection_principal_raw) if projection_principal_raw is not None else None
    )
    projection_K_px = (
        (
            (projection_focal_px[0], 0.0, projection_principal_px[0]),
            (0.0, projection_focal_px[1], projection_principal_px[1]),
            (0.0, 0.0, 1.0),
        )
        if projection_principal_px is not None
        else None
    )

    gaussians_3d = []
    for idx in range(count):
        theta = 2.0 * math.pi * (idx / max(count, 1))
        gaussians_3d.append(
            GaussianPrimitive3D(
                mean_xyz=(
                    radius_m * math.cos(theta),
                    radius_m * math.sin(theta),
                    depth_start + depth_step * idx,
                ),
                quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                scale_xyz=scale_xyz,
                opacity=opacity,
                amplitude=amplitude,
                phase=phase_start + phase_step * idx,
            )
        )
    return PrimitiveScene2D(
        gaussians_3d=tuple(gaussians_3d),
        name=str(cfg.get("name", "gaussian3d_depth_ring")),
        projection_focal_px=projection_focal_px,
        projection_principal_px=projection_principal_px,
        projection_K_px=projection_K_px,
        phase_matching=bool(projection_cfg.get("phase_matching", True)),
    )


def build_primitive_scene_from_config(
    cfg: Mapping[str, object] | None,
    *,
    dim: tuple[int, int, int, int] | None = None,
    pitch: float | None = None,
) -> PrimitiveScene2D:
    cfg = _coerce_scene_mapping(cfg)
    preset = cfg.get("preset")
    if preset is not None:
        preset_name = str(preset)
        if preset_name == "ring":
            return _make_ring_scene(cfg, dim=dim)
        if preset_name == "grid":
            return _make_grid_scene(cfg, dim=dim)
        if preset_name == "depth_ring":
            return _make_depth_ring_scene(cfg, dim=dim)
        if preset_name == "gaussian3d_depth_ring":
            return _make_gaussian3d_depth_ring_scene(cfg, pitch=pitch)
        raise ValueError(f"unsupported primitive preset {preset_name!r}")

    gaussian_cfgs = cfg.get("gaussians", ())
    gaussian3d_cfgs = cfg.get("gaussians_3d", ())
    wave_gaussian_cfgs = cfg.get("wave_gaussians", ())
    point_cfgs = cfg.get("points", ())
    name = cfg.get("name")
    coordinate_system = str(cfg.get("coordinate_system", "pixel"))
    projection_cfg = cfg.get("projection") or {}
    projection_focal_px = projection_cfg.get("focal_lengths_px")
    if projection_focal_px is not None:
        projection_focal_px = tuple(float(v) for v in projection_focal_px)
    projection_principal_px = projection_cfg.get("principal_point_px")
    if projection_principal_px is not None:
        projection_principal_px = tuple(float(v) for v in projection_principal_px)
    projection_K_px = projection_cfg.get("K_px")
    if projection_K_px is not None:
        projection_K_px = tuple(tuple(float(v) for v in row) for row in projection_K_px)
    elif projection_focal_px is not None:
        if projection_principal_px is not None:
            projection_K_px = (
                (projection_focal_px[0], 0.0, projection_principal_px[0]),
                (0.0, projection_focal_px[1], projection_principal_px[1]),
                (0.0, 0.0, 1.0),
            )
    projection_view_matrix = projection_cfg.get("view_matrix")
    if projection_view_matrix is not None:
        projection_view_matrix = tuple(tuple(float(v) for v in row) for row in projection_view_matrix)
    rows, cols = (None, None)
    if coordinate_system != "pixel":
        rows, cols = _require_plane_shape(dim)

    gaussians = tuple(
        GaussianPrimitive2D(
            center_yx=(
                _resolve_center_yx(gaussian_cfg["center_yx"], rows, cols, coordinate_system)
                if coordinate_system != "pixel"
                else tuple(float(v) for v in gaussian_cfg["center_yx"])
            ),
            sigma_yx=tuple(float(v) for v in gaussian_cfg["sigma_yx"]),
            amplitude=float(gaussian_cfg.get("amplitude", 1.0)),
            phase=float(gaussian_cfg.get("phase", 0.0)),
            rotation=float(gaussian_cfg.get("rotation", 0.0)),
        )
        for gaussian_cfg in gaussian_cfgs
    )
    gaussians_3d = tuple(
        GaussianPrimitive3D(
            mean_xyz=tuple(float(v) for v in gaussian_cfg["mean_xyz"]),
            quat_wxyz=tuple(float(v) for v in gaussian_cfg.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0))),
            scale_xyz=tuple(float(v) for v in gaussian_cfg["scale_xyz"]),
            opacity=float(gaussian_cfg.get("opacity", 1.0)),
            amplitude=float(gaussian_cfg.get("amplitude", 1.0)),
            phase=float(gaussian_cfg.get("phase", 0.0)),
        )
        for gaussian_cfg in gaussian3d_cfgs
    )
    wave_gaussians = tuple(
        GaussianWavePrimitive2D(
            center_yx=(
                _resolve_center_yx(wave_gaussian_cfg["center_yx"], rows, cols, coordinate_system)
                if coordinate_system != "pixel"
                else tuple(float(v) for v in wave_gaussian_cfg["center_yx"])
            ),
            sigma_yx=tuple(float(v) for v in wave_gaussian_cfg["sigma_yx"]),
            depth=float(wave_gaussian_cfg["depth"]),
            amplitude=float(wave_gaussian_cfg.get("amplitude", 1.0)),
            opacity=float(wave_gaussian_cfg.get("opacity", 1.0)),
            phase=float(wave_gaussian_cfg.get("phase", 0.0)),
            rotation=float(wave_gaussian_cfg.get("rotation", 0.0)),
        )
        for wave_gaussian_cfg in wave_gaussian_cfgs
    )
    points = tuple(
        PointPrimitive2D(
            yx=(
                _resolve_center_yx(point_cfg["yx"], rows, cols, coordinate_system)
                if coordinate_system != "pixel"
                else tuple(float(v) for v in point_cfg["yx"])
            ),
            amplitude=float(point_cfg.get("amplitude", 1.0)),
            phase=float(point_cfg.get("phase", 0.0)),
        )
        for point_cfg in point_cfgs
    )

    scene = PrimitiveScene2D(
        gaussians=gaussians,
        gaussians_3d=gaussians_3d,
        wave_gaussians=wave_gaussians,
        points=points,
        name=str(name) if name is not None else None,
        projection_focal_px=projection_focal_px,
        projection_principal_px=projection_principal_px,
        projection_K_px=projection_K_px,
        projection_view_matrix=projection_view_matrix,
        phase_matching=bool(projection_cfg.get("phase_matching", True)),
    )
    if scene.is_empty():
        raise ValueError(
            "primitive scene config must define at least one gaussian, gaussian_3d, wave_gaussian, or point primitive"
        )
    return scene


def _make_pixel_grid(
    rows: int,
    cols: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(rows, device=device, dtype=dtype)
    x = torch.arange(cols, device=device, dtype=dtype)
    return torch.meshgrid(y, x, indexing="ij")


def _render_single_gaussian(
    primitive: GaussianPrimitive2D,
    yy: torch.Tensor,
    xx: torch.Tensor,
) -> torch.Tensor:
    center_y = yy.new_tensor(float(primitive.center_yx[0]))
    center_x = xx.new_tensor(float(primitive.center_yx[1]))
    sigma_y = yy.new_tensor(float(primitive.sigma_yx[0]))
    sigma_x = xx.new_tensor(float(primitive.sigma_yx[1]))
    rotation = yy.new_tensor(float(primitive.rotation))
    amplitude = yy.new_tensor(float(primitive.amplitude))
    phase = yy.new_tensor(float(primitive.phase))

    dy = yy - center_y
    dx = xx - center_x
    cos_t = torch.cos(rotation)
    sin_t = torch.sin(rotation)
    y_rot = cos_t * dy + sin_t * dx
    x_rot = -sin_t * dy + cos_t * dx
    envelope = torch.exp(-0.5 * ((y_rot / sigma_y) ** 2 + (x_rot / sigma_x) ** 2))
    return amplitude * envelope * torch.exp(1j * phase)


def _render_single_point(
    primitive: PointPrimitive2D,
    rows: int,
    cols: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    field = torch.zeros((rows, cols), dtype=torch.cfloat, device=device)
    y_idx = int(round(float(primitive.yx[0])))
    x_idx = int(round(float(primitive.yx[1])))
    if 0 <= y_idx < rows and 0 <= x_idx < cols:
        amplitude = torch.tensor(float(primitive.amplitude), dtype=torch.float32, device=device)
        phase = torch.tensor(float(primitive.phase), dtype=torch.float32, device=device)
        field[y_idx, x_idx] = amplitude * torch.exp(1j * phase)
    return field


def _render_single_wave_gaussian(
    primitive: GaussianWavePrimitive2D,
    yy: torch.Tensor,
    xx: torch.Tensor,
) -> torch.Tensor:
    center_y = yy.new_tensor(float(primitive.center_yx[0]))
    center_x = xx.new_tensor(float(primitive.center_yx[1]))
    sigma_y = yy.new_tensor(float(primitive.sigma_yx[0]))
    sigma_x = xx.new_tensor(float(primitive.sigma_yx[1]))
    rotation = yy.new_tensor(float(primitive.rotation))
    amplitude = yy.new_tensor(float(primitive.amplitude))
    phase = yy.new_tensor(float(primitive.phase))

    dy = yy - center_y
    dx = xx - center_x
    cos_t = torch.cos(rotation)
    sin_t = torch.sin(rotation)
    y_rot = cos_t * dy + sin_t * dx
    x_rot = -sin_t * dy + cos_t * dx
    envelope = torch.exp(-0.5 * ((y_rot / sigma_y) ** 2 + (x_rot / sigma_x) ** 2))
    return amplitude * envelope * torch.exp(1j * phase)


def _make_phase_noise_generator(
    device: torch.device,
    random_seed: int | None,
) -> torch.Generator | None:
    if random_seed is None:
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(random_seed))
    return generator


def _apply_random_phase(
    field: torch.Tensor,
    *,
    std: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if std <= 0:
        return field
    noise = torch.randn(
        field.shape,
        generator=generator,
        device=field.device,
        dtype=field.real.dtype,
    ) * float(std)
    return field * torch.exp(1j * noise)


def _expand_plane_field(
    plane_field: torch.Tensor,
    dim: tuple[int, int, int, int],
    *,
    scene: PrimitiveScene2D,
    renderer: str,
    backend: str,
    backend_reason: str,
    intensity_override: torch.Tensor | None = None,
    frame_fields: tuple[torch.Tensor, ...] | None = None,
) -> PrimitiveFieldRenderResult:
    batch, channels, rows, cols = (int(v) for v in dim)
    field = plane_field.view(1, 1, rows, cols).expand(batch, channels, rows, cols).clone()
    intensity_plane = field.abs().square() if intensity_override is None else intensity_override.view(1, 1, rows, cols)
    intensity = intensity_plane.expand(batch, channels, rows, cols).clone()
    expanded_frames = None
    if frame_fields is not None:
        expanded_frames = tuple(
            frame_field.view(1, 1, rows, cols).expand(batch, channels, rows, cols).clone()
            for frame_field in frame_fields
        )
    return PrimitiveFieldRenderResult(
        field=field,
        intensity=intensity,
        scene=scene,
        renderer=renderer,
        backend=backend,
        backend_reason=backend_reason,
        frame_fields=expanded_frames,
    )


def _project_scene_gaussians_3d(
    scene: PrimitiveScene2D,
    primitives: tuple[GaussianPrimitive3D, ...],
    *,
    source_spec: SourceSpec,
    rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[GaussianPrimitive3D, ...]:
    return project_gaussians3d_to_hologram_space(
        primitives,
        pitch=source_spec.pitch,
        rows=rows,
        cols=cols,
        device=device,
        dtype=dtype,
        focal_px=scene.projection_focal_px,
        principal_px=scene.projection_principal_px,
        K_px=scene.projection_K_px,
        view_matrix=scene.projection_view_matrix,
    )


def _render_exact_projected_wavefronts(
    projected_primitives: tuple[GaussianPrimitive3D, ...],
    *,
    rows: int,
    cols: int,
    source_spec: SourceSpec,
    phase_matching: bool,
    return_at_object_depth: bool,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    warp_cache_dir: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, KernelBackendSelection]:
    if backend == "auto":
        selection = KernelBackendSelection(
            "auto",
            "torch",
            "PyTorch remains the default backend for exact analytic primitives unless Warp is requested explicitly.",
        )
    else:
        selection = resolve_primitive_backend(
            requested=backend,
            device=device,
            warp_cache_dir=warp_cache_dir,
        )
    if selection.resolved == "warp":
        if dtype != torch.float32:
            selection = KernelBackendSelection(
                selection.requested,
                "torch",
                "Warp exact primitive kernels currently operate on float32 tensors, so PyTorch handled this exact path.",
            )
        else:
            fx, fy, fz, band_mask = make_frequency_grid(
                rows,
                cols,
                pitch=source_spec.pitch,
                wavelength=source_spec.wvl,
                device=device,
                dtype=dtype,
            )
            means, quats, scales, _, amplitudes, phases = stack_gaussian3d_parameters(
                projected_primitives,
                device=device,
                dtype=dtype,
            )
            projected_params = build_projected_gaussian_warp_params(
                means,
                quats,
                scales,
                amplitudes,
                phases,
            )
            try:
                wavefronts, phase_compensations = render_projected_gaussian_wavefronts_warp(
                    projected_params,
                    fx,
                    fy,
                    fz,
                    band_mask,
                    wavelength=source_spec.wvl,
                    pitch=source_spec.pitch,
                    phase_matching=phase_matching,
                    return_at_object_depth=return_at_object_depth,
                    device=device,
                    warp_cache_dir=warp_cache_dir,
                )
            except Exception as exc:  # pragma: no cover - exercised through optional backend path
                raise RuntimeError("Warp exact primitive backend failed during execution.") from exc
            return wavefronts.to(dtype=torch.complex64 if dtype == torch.float32 else torch.complex128), phase_compensations, selection

    wavefronts, phase_compensations = exact_projected_gaussian_wavefronts_batched(
        projected_primitives,
        rows=rows,
        cols=cols,
        pitch=source_spec.pitch,
        wavelength=source_spec.wvl,
        device=device,
        dtype=dtype,
        phase_matching=phase_matching,
        return_at_object_depth=return_at_object_depth,
    )
    return wavefronts, phase_compensations, selection


def render_gaussian_scene_naive(
    scene: PrimitiveScene2D,
    dim: tuple[int, int, int, int],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PrimitiveFieldRenderResult:
    """Render a primitive scene into a complex field with a simple loop baseline."""

    batch, channels, rows, cols = (int(v) for v in dim)
    if channels != 1:
        raise ValueError(f"primitive-based baseline currently expects a single channel, got dim={dim}")
    if scene.gaussians_3d:
        raise ValueError("gaussian_naive does not support gaussians_3d; use renderer='gaussian_gws_exact'")
    if scene.wave_gaussians:
        raise ValueError("gaussian_naive does not support wave_gaussians; use renderer='gaussian_wave'")

    target_device = torch.device(device)
    yy, xx = _make_pixel_grid(rows, cols, device=target_device, dtype=dtype)
    plane_field = torch.zeros((rows, cols), dtype=torch.cfloat, device=target_device)

    for primitive in scene.gaussians:
        plane_field = plane_field + _render_single_gaussian(primitive, yy, xx)
    for primitive in scene.points:
        plane_field = plane_field + _render_single_point(primitive, rows, cols, device=target_device)

    return _expand_plane_field(
        plane_field,
        dim,
        scene=scene,
        renderer="gaussian_naive",
        backend="torch",
        backend_reason="Naive primitive renderer uses the PyTorch baseline path.",
    )


def render_gaussian_scene_splat(
    scene: PrimitiveScene2D,
    dim: tuple[int, int, int, int],
    *,
    backend: str = "auto",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    warp_cache_dir: str | None = None,
) -> PrimitiveFieldRenderResult:
    """Render a primitive scene with a vectorized Gaussian splat path."""

    batch, channels, rows, cols = (int(v) for v in dim)
    if channels != 1:
        raise ValueError(f"primitive-based baseline currently expects a single channel, got dim={dim}")
    if scene.gaussians_3d:
        raise ValueError("gaussian_splat does not support gaussians_3d; use renderer='gaussian_gws_exact'")
    if scene.wave_gaussians:
        raise ValueError("gaussian_splat does not support wave_gaussians; use renderer='gaussian_wave'")

    target_device = torch.device(device)
    plane_field = torch.zeros((rows, cols), dtype=torch.cfloat, device=target_device)

    gaussian_params = scene.gaussian_parameters().to(device=target_device, dtype=dtype)
    selection = resolve_primitive_backend(
        requested=backend,
        device=str(target_device),
        warp_cache_dir=warp_cache_dir,
    )

    if gaussian_params.numel() > 0 and selection.resolved == "warp":
        plane_field = plane_field + render_gaussian_scene_warp(
            gaussian_params,
            rows,
            cols,
            device=target_device,
            warp_cache_dir=warp_cache_dir,
        ).to(torch.cfloat)
    elif gaussian_params.numel() > 0:
        yy, xx = _make_pixel_grid(rows, cols, device=target_device, dtype=dtype)
        center_y = gaussian_params[:, 0].view(-1, 1, 1)
        center_x = gaussian_params[:, 1].view(-1, 1, 1)
        sigma_y = gaussian_params[:, 2].view(-1, 1, 1)
        sigma_x = gaussian_params[:, 3].view(-1, 1, 1)
        amplitude = gaussian_params[:, 4].view(-1, 1, 1)
        phase = gaussian_params[:, 5].view(-1, 1, 1)
        rotation = gaussian_params[:, 6].view(-1, 1, 1)

        dy = yy.unsqueeze(0) - center_y
        dx = xx.unsqueeze(0) - center_x
        cos_t = torch.cos(rotation)
        sin_t = torch.sin(rotation)
        y_rot = cos_t * dy + sin_t * dx
        x_rot = -sin_t * dy + cos_t * dx
        envelope = torch.exp(-0.5 * ((y_rot / sigma_y) ** 2 + (x_rot / sigma_x) ** 2))
        contributions = amplitude * envelope * torch.exp(1j * phase)
        plane_field = plane_field + contributions.sum(dim=0).to(torch.cfloat)

    for primitive in scene.points:
        plane_field = plane_field + _render_single_point(primitive, rows, cols, device=target_device)

    return _expand_plane_field(
        plane_field,
        dim,
        scene=scene,
        renderer="gaussian_splat",
        backend=selection.resolved,
        backend_reason=selection.reason,
    )


def render_gaussian_scene(
    scene: PrimitiveScene2D,
    dim: tuple[int, int, int, int],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PrimitiveFieldRenderResult:
    """Backward-compatible alias for the naive Gaussian renderer."""

    return render_gaussian_scene_naive(scene, dim, device=device, dtype=dtype)


def render_gaussian_scene_wave(
    scene: PrimitiveScene2D,
    dim: tuple[int, int, int, int],
    *,
    source_spec: SourceSpec,
    propagation_spec: PropagationSpec,
    random_phase_std: float = 0.0,
    random_seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PrimitiveFieldRenderResult:
    """Render a scene by propagating depth-aware Gaussian wave primitives to the hologram plane.

    This baseline follows the principle used by wave-based primitive CGH methods:
    define a complex envelope on each primitive's local object plane, group
    primitives by depth, and use linear propagation to bring each depth layer to
    the shared hologram plane before coherent summation.
    """

    batch, channels, rows, cols = (int(v) for v in dim)
    if channels != 1:
        raise ValueError(f"primitive-based baseline currently expects a single channel, got dim={dim}")
    if scene.gaussians_3d:
        raise ValueError("gaussian_wave does not support gaussians_3d; use renderer='gaussian_gws_exact'")
    if tuple(source_spec.dim) != tuple(dim):
        raise ValueError(f"source_spec.dim must match dim {dim}, got {source_spec.dim}")

    target_device = torch.device(device)
    yy, xx = _make_pixel_grid(rows, cols, device=target_device, dtype=dtype)
    plane_field = torch.zeros((rows, cols), dtype=torch.cfloat, device=target_device)
    generator = _make_phase_noise_generator(target_device, random_seed)

    for primitive in scene.gaussians:
        plane_field = plane_field + _render_single_gaussian(primitive, yy, xx)
    for primitive in scene.points:
        plane_field = plane_field + _render_single_point(primitive, rows, cols, device=target_device)

    if scene.wave_gaussians:
        depth_fields: dict[float, torch.Tensor] = {}
        for primitive in scene.wave_gaussians:
            depth = float(primitive.depth)
            if depth not in depth_fields:
                depth_fields[depth] = torch.zeros((rows, cols), dtype=torch.cfloat, device=target_device)
            local_field = _render_single_wave_gaussian(primitive, yy, xx)
            local_field = _apply_random_phase(local_field, std=random_phase_std, generator=generator)
            depth_fields[depth] = depth_fields[depth] + local_field

        wave_source = SourceSpec(
            dim=source_spec.dim,
            pitch=source_spec.pitch,
            wvl=source_spec.wvl,
            device=str(target_device),
        )
        for depth, local_field in sorted(depth_fields.items(), key=lambda item: item[0]):
            layer_light = wave_source.make_light()
            layer_field = local_field.view(1, 1, rows, cols).to(layer_light.field.device, dtype=layer_light.field.dtype)
            layer_light.set_field(layer_field)
            propagated = propagation_spec.forward(layer_light, distance=depth)
            plane_field = plane_field + propagated.field.to(target_device)[0, 0]

    return _expand_plane_field(
        plane_field,
        dim,
        scene=scene,
        renderer="gaussian_wave",
        backend="torch",
        backend_reason=(
            "Wave primitive renderer groups primitives by depth and reuses "
            "PADO propagation for coherent layer-wise transport."
        ),
    )


def render_gaussian_scene_wave_awb(
    scene: PrimitiveScene2D,
    dim: tuple[int, int, int, int],
    *,
    source_spec: SourceSpec,
    propagation_spec: PropagationSpec,
    sort_order: str = "back2front",
    alpha_binary_threshold: float = 0.0,
    random_phase_std: float = 0.0,
    random_seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PrimitiveFieldRenderResult:
    """Render depth-aware Gaussian primitives with hsplat-inspired alpha wave blending.

    This keeps the current PADO-first architecture while borrowing the core AWB
    idea from hsplat: order primitives by depth, compute a local complex object
    field for each primitive, modulate it by a running transmittance mask, and
    coherently propagate each masked contribution to the hologram plane.

    Note:
        This is still a spatial-domain baseline. It is not yet the paper-faithful
        closed-form Fourier-domain GWS/RPWS implementation, and it does not yet
        implement the structured-random-phase formulation from RPWS.
    """

    batch, channels, rows, cols = (int(v) for v in dim)
    if channels != 1:
        raise ValueError(f"primitive-based baseline currently expects a single channel, got dim={dim}")
    if scene.gaussians_3d:
        raise ValueError("gaussian_wave_awb does not support gaussians_3d; use renderer='gaussian_gws_exact_awb'")
    if tuple(source_spec.dim) != tuple(dim):
        raise ValueError(f"source_spec.dim must match dim {dim}, got {source_spec.dim}")
    if alpha_binary_threshold < 0:
        raise ValueError(f"alpha_binary_threshold must be non-negative, got {alpha_binary_threshold}")

    target_device = torch.device(device)
    yy, xx = _make_pixel_grid(rows, cols, device=target_device, dtype=dtype)
    plane_field = torch.zeros((rows, cols), dtype=torch.cfloat, device=target_device)
    transmittance = torch.ones((rows, cols), dtype=torch.float32, device=target_device)
    generator = _make_phase_noise_generator(target_device, random_seed)

    for primitive in scene.gaussians:
        plane_field = plane_field + _render_single_gaussian(primitive, yy, xx)
    for primitive in scene.points:
        plane_field = plane_field + _render_single_point(primitive, rows, cols, device=target_device)

    wave_source = SourceSpec(
        dim=source_spec.dim,
        pitch=source_spec.pitch,
        wvl=source_spec.wvl,
        device=str(target_device),
    )
    for primitive in scene.ordered_wave_gaussians(sort_order):
        local_field = _render_single_wave_gaussian(primitive, yy, xx)
        local_field = _apply_random_phase(local_field, std=random_phase_std, generator=generator)
        alpha = float(primitive.opacity) * local_field.abs().float()
        if alpha_binary_threshold > 0:
            alpha = (alpha > alpha_binary_threshold).float()
        else:
            alpha = alpha.clamp(0.0, 1.0)

        masked_field = transmittance.to(torch.cfloat) * float(primitive.opacity) * local_field
        layer_light = wave_source.make_light()
        layer_field = masked_field.view(1, 1, rows, cols).to(layer_light.field.device, dtype=layer_light.field.dtype)
        layer_light.set_field(layer_field)
        propagated = propagation_spec.forward(layer_light, distance=float(primitive.depth))
        plane_field = plane_field + propagated.field.to(target_device)[0, 0]
        transmittance = transmittance * (1.0 - alpha)

    return _expand_plane_field(
        plane_field,
        dim,
        scene=scene,
        renderer="gaussian_wave_awb",
        backend="torch",
        backend_reason=(
            "Depth-aware alpha wave blending orders wave primitives, updates a "
            "transmittance mask, and propagates masked complex contributions with PADO."
        ),
    )


def render_gaussian_scene_gws_exact(
    scene: PrimitiveScene2D,
    dim: tuple[int, int, int, int],
    *,
    source_spec: SourceSpec,
    backend: str = "auto",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    warp_cache_dir: str | None = None,
) -> PrimitiveFieldRenderResult:
    """Render a 3D Gaussian scene with a closed-form Fourier-domain GWS slice.

    This path follows the paper/hsplat principle more closely than the spatial
    baseline: 3D Gaussian primitives are first projected into hologram-space 2D
    Gaussians using the perspective Jacobian, and each projected primitive then
    contributes a closed-form Gaussian spectrum that is propagated to the SLM.
    """

    batch, channels, rows, cols = (int(v) for v in dim)
    if channels != 1:
        raise ValueError(f"primitive-based baseline currently expects a single channel, got dim={dim}")
    if tuple(source_spec.dim) != tuple(dim):
        raise ValueError(f"source_spec.dim must match dim {dim}, got {source_spec.dim}")
    if not scene.gaussians_3d:
        raise ValueError("gaussian_gws_exact requires scene.gaussians_3d")

    target_device = torch.device(device)
    projected_primitives = _project_scene_gaussians_3d(
        scene,
        scene.gaussians_3d,
        source_spec=source_spec,
        rows=rows,
        cols=cols,
        device=target_device,
        dtype=dtype,
    )
    wavefronts, _, selection = _render_exact_projected_wavefronts(
        projected_primitives,
        rows=rows,
        cols=cols,
        source_spec=source_spec,
        phase_matching=scene.phase_matching,
        return_at_object_depth=False,
        backend=backend,
        device=target_device,
        dtype=dtype,
        warp_cache_dir=warp_cache_dir,
    )
    if wavefronts.numel() == 0:
        plane_field = torch.zeros(
            (rows, cols),
            dtype=torch.complex64 if dtype == torch.float32 else torch.complex128,
            device=target_device,
        )
    else:
        opacities = torch.tensor(
            [primitive.opacity for primitive in projected_primitives],
            device=target_device,
            dtype=wavefronts.real.dtype,
        ).view(-1, 1, 1)
        plane_field = (wavefronts * opacities.to(wavefronts.dtype)).sum(dim=0)
    return _expand_plane_field(
        plane_field,
        dim,
        scene=scene,
        renderer="gaussian_gws_exact",
        backend=selection.resolved,
        backend_reason=(
            f"{selection.reason} Projected 3D Gaussians are converted into a "
            "batched closed-form Fourier-domain exact path before inverse transforming to the SLM plane."
        ),
    )


def render_gaussian_scene_gws_exact_awb(
    scene: PrimitiveScene2D,
    dim: tuple[int, int, int, int],
    *,
    source_spec: SourceSpec,
    sort_order: str = "front2back",
    alpha_binary_threshold: float = 0.0,
    backend: str = "auto",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    warp_cache_dir: str | None = None,
) -> PrimitiveFieldRenderResult:
    """Render a projected-Gaussian scene with front-to-back exact-style AWB."""

    batch, channels, rows, cols = (int(v) for v in dim)
    if channels != 1:
        raise ValueError(f"primitive-based baseline currently expects a single channel, got dim={dim}")
    if tuple(source_spec.dim) != tuple(dim):
        raise ValueError(f"source_spec.dim must match dim {dim}, got {source_spec.dim}")
    if not scene.gaussians_3d:
        raise ValueError("gaussian_gws_exact_awb requires scene.gaussians_3d")
    if alpha_binary_threshold < 0:
        raise ValueError(f"alpha_binary_threshold must be non-negative, got {alpha_binary_threshold}")

    target_device = torch.device(device)
    _, _, _, band_mask = make_frequency_grid(
        rows,
        cols,
        pitch=source_spec.pitch,
        wavelength=source_spec.wvl,
        device=target_device,
        dtype=dtype,
    )
    ordered_primitives = scene.ordered_gaussians_3d(sort_order)
    projected_primitives = _project_scene_gaussians_3d(
        scene,
        ordered_primitives,
        source_spec=source_spec,
        rows=rows,
        cols=cols,
        device=target_device,
        dtype=dtype,
    )
    object_wavefronts, phase_compensations, selection = _render_exact_projected_wavefronts(
        projected_primitives,
        rows=rows,
        cols=cols,
        source_spec=source_spec,
        phase_matching=scene.phase_matching,
        return_at_object_depth=True,
        backend=backend,
        device=target_device,
        dtype=dtype,
        warp_cache_dir=warp_cache_dir,
    )
    assert phase_compensations is not None
    plane_field = torch.zeros(
        (rows, cols),
        dtype=torch.complex64 if dtype == torch.float32 else torch.complex128,
        device=target_device,
    )
    transmittance = torch.ones((rows, cols), dtype=dtype, device=target_device)

    for index, primitive in enumerate(ordered_primitives):
        object_wavefront = object_wavefronts[index]
        phase_compensation = phase_compensations[index]
        alpha = object_wavefront.abs().to(dtype) * float(primitive.opacity)
        if alpha_binary_threshold > 0:
            alpha = (alpha > alpha_binary_threshold).to(dtype)
        else:
            alpha = alpha.clamp(0.0, 1.0)

        masked_field = transmittance.to(object_wavefront.dtype) * float(primitive.opacity) * object_wavefront
        plane_field = plane_field + apply_phase_compensation(
            masked_field,
            phase_compensation=phase_compensation,
            band_mask=band_mask,
        ).to(plane_field.dtype)
        transmittance = transmittance * (1.0 - alpha)

    return _expand_plane_field(
        plane_field,
        dim,
        scene=scene,
        renderer="gaussian_gws_exact_awb",
        backend=selection.resolved,
        backend_reason=(
            f"{selection.reason} Projected 3D Gaussians reuse exact object-plane "
            "wavefronts and phase compensation for alpha wave blending at the SLM."
        ),
    )


def render_gaussian_scene_gws_rpws_exact(
    scene: PrimitiveScene2D,
    dim: tuple[int, int, int, int],
    *,
    source_spec: SourceSpec,
    sort_order: str = "back2front",
    alpha_binary_threshold: float = 0.0,
    num_frames: int = 8,
    angular_profile: str = "uniform",
    angular_radius_fraction: float = 1.0,
    angular_sigma_fraction: float = 0.35,
    random_phase_range: float = math.pi,
    random_seed: int | None = None,
    backend: str = "auto",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    warp_cache_dir: str | None = None,
) -> PrimitiveFieldRenderResult:
    """Render a paper-aligned RPWS baseline with structured random phase.

    Each 3D Gaussian is first converted into the exact GWS object-plane wavefront.
    For every time-multiplexed frame, we sample a random Fourier-domain kernel
    ``Y(k) exp(i q_t(k))``, convert it into a spatial modulation kernel, multiply
    the object-plane wavefront by that kernel, and then apply exact-style alpha
    wave blending plus phase-compensated propagation back to the SLM plane.
    """

    batch, channels, rows, cols = (int(v) for v in dim)
    if channels != 1:
        raise ValueError(f"primitive-based baseline currently expects a single channel, got dim={dim}")
    if tuple(source_spec.dim) != tuple(dim):
        raise ValueError(f"source_spec.dim must match dim {dim}, got {source_spec.dim}")
    if not scene.gaussians_3d:
        raise ValueError("gaussian_gws_rpws_exact requires scene.gaussians_3d")
    if alpha_binary_threshold < 0:
        raise ValueError(f"alpha_binary_threshold must be non-negative, got {alpha_binary_threshold}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    target_device = torch.device(device)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    fx, fy, _, band_mask = make_frequency_grid(
        rows,
        cols,
        pitch=source_spec.pitch,
        wavelength=source_spec.wvl,
        device=target_device,
        dtype=dtype,
    )
    angular_emission = build_angular_emission_profile(
        fx,
        fy,
        wavelength=source_spec.wvl,
        profile=angular_profile,
        radius_fraction=angular_radius_fraction,
        gaussian_sigma_fraction=angular_sigma_fraction,
        band_mask=band_mask,
    )
    ordered_primitives = scene.ordered_gaussians_3d(sort_order)
    projected_primitives = _project_scene_gaussians_3d(
        scene,
        ordered_primitives,
        source_spec=source_spec,
        rows=rows,
        cols=cols,
        device=target_device,
        dtype=dtype,
    )
    object_wavefronts, phase_compensations, selection = _render_exact_projected_wavefronts(
        projected_primitives,
        rows=rows,
        cols=cols,
        source_spec=source_spec,
        phase_matching=scene.phase_matching,
        return_at_object_depth=True,
        backend=backend,
        device=target_device,
        dtype=dtype,
        warp_cache_dir=warp_cache_dir,
    )
    assert phase_compensations is not None
    generator = _make_phase_noise_generator(target_device, random_seed)
    frame_fields: list[torch.Tensor] = []
    frame_intensities: list[torch.Tensor] = []

    for _frame_idx in range(int(num_frames)):
        plane_field = torch.zeros((rows, cols), dtype=complex_dtype, device=target_device)
        transmittance = torch.ones((rows, cols), dtype=dtype, device=target_device)
        _, modulation_kernels = sample_structured_random_phase_kernels(
            angular_emission,
            num_samples=len(ordered_primitives),
            generator=generator,
            phase_range=random_phase_range,
        )
        modulation_kernels = modulation_kernels.to(dtype=object_wavefronts.dtype)
        modulated_object_wavefronts = object_wavefronts * modulation_kernels

        for index, primitive in enumerate(ordered_primitives):
            modulated_object_wavefront = modulated_object_wavefronts[index]
            phase_compensation = phase_compensations[index]
            alpha = modulated_object_wavefront.abs().to(dtype) * float(primitive.opacity)
            if alpha_binary_threshold > 0:
                alpha = (alpha > alpha_binary_threshold).to(dtype)
            else:
                alpha = alpha.clamp(0.0, 1.0)

            masked_field = transmittance.to(modulated_object_wavefront.dtype) * float(primitive.opacity) * modulated_object_wavefront
            plane_field = plane_field + apply_phase_compensation(
                masked_field,
                phase_compensation=phase_compensation,
                band_mask=band_mask,
            ).to(plane_field.dtype)
            transmittance = transmittance * (1.0 - alpha)

        frame_fields.append(plane_field)
        frame_intensities.append(plane_field.abs().square())

    mean_plane_field = torch.stack(frame_fields, dim=0).mean(dim=0)
    mean_intensity = torch.stack(frame_intensities, dim=0).mean(dim=0)
    return _expand_plane_field(
        mean_plane_field,
        dim,
        scene=scene,
        renderer="gaussian_gws_rpws_exact",
        backend=selection.resolved,
        backend_reason=(
            f"{selection.reason} Exact GWS object-plane wavefronts are modulated by "
            "structured random phase kernels Y(k) exp(i q_t(k)) and averaged over multiplexed frames."
        ),
        intensity_override=mean_intensity,
        frame_fields=tuple(frame_fields),
    )


def render_primitive_scene(
    scene: PrimitiveScene2D,
    dim: tuple[int, int, int, int],
    *,
    renderer: str = "gaussian_naive",
    backend: str = "auto",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    warp_cache_dir: str | None = None,
    source_spec: SourceSpec | None = None,
    propagation_spec: PropagationSpec | None = None,
    sort_order: str = "back2front",
    alpha_binary_threshold: float = 0.0,
    random_phase_std: float = 0.0,
    num_frames: int = 1,
    angular_profile: str = "uniform",
    angular_radius_fraction: float = 1.0,
    angular_sigma_fraction: float = 0.35,
    random_phase_range: float = math.pi,
    random_seed: int | None = None,
) -> PrimitiveFieldRenderResult:
    if renderer == "gaussian_naive":
        return render_gaussian_scene_naive(scene, dim, device=device, dtype=dtype)
    if renderer == "gaussian_splat":
        return render_gaussian_scene_splat(
            scene,
            dim,
            backend=backend,
            device=device,
            dtype=dtype,
            warp_cache_dir=warp_cache_dir,
        )
    if renderer == "gaussian_wave":
        if source_spec is None or propagation_spec is None:
            raise ValueError("gaussian_wave requires source_spec and propagation_spec")
        return render_gaussian_scene_wave(
            scene,
            dim,
            source_spec=source_spec,
            propagation_spec=propagation_spec,
            random_phase_std=random_phase_std,
            random_seed=random_seed,
            device=device,
            dtype=dtype,
        )
    if renderer == "gaussian_wave_awb":
        if source_spec is None or propagation_spec is None:
            raise ValueError("gaussian_wave_awb requires source_spec and propagation_spec")
        return render_gaussian_scene_wave_awb(
            scene,
            dim,
            source_spec=source_spec,
            propagation_spec=propagation_spec,
            sort_order=sort_order,
            alpha_binary_threshold=alpha_binary_threshold,
            random_phase_std=random_phase_std,
            random_seed=random_seed,
            device=device,
            dtype=dtype,
        )
    if renderer == "gaussian_gws_exact":
        if source_spec is None:
            raise ValueError("gaussian_gws_exact requires source_spec")
        return render_gaussian_scene_gws_exact(
            scene,
            dim,
            source_spec=source_spec,
            backend=backend,
            device=device,
            dtype=dtype,
            warp_cache_dir=warp_cache_dir,
        )
    if renderer == "gaussian_gws_exact_awb":
        if source_spec is None:
            raise ValueError("gaussian_gws_exact_awb requires source_spec")
        return render_gaussian_scene_gws_exact_awb(
            scene,
            dim,
            source_spec=source_spec,
            sort_order=sort_order,
            alpha_binary_threshold=alpha_binary_threshold,
            backend=backend,
            device=device,
            dtype=dtype,
            warp_cache_dir=warp_cache_dir,
        )
    if renderer == "gaussian_gws_rpws_exact":
        if source_spec is None:
            raise ValueError("gaussian_gws_rpws_exact requires source_spec")
        return render_gaussian_scene_gws_rpws_exact(
            scene,
            dim,
            source_spec=source_spec,
            sort_order=sort_order,
            alpha_binary_threshold=alpha_binary_threshold,
            num_frames=num_frames,
            angular_profile=angular_profile,
            angular_radius_fraction=angular_radius_fraction,
            angular_sigma_fraction=angular_sigma_fraction,
            random_phase_range=random_phase_range,
            random_seed=random_seed,
            backend=backend,
            device=device,
            dtype=dtype,
            warp_cache_dir=warp_cache_dir,
        )
    supported = ", ".join(available_primitive_renderers())
    raise ValueError(f"unsupported primitive renderer {renderer!r}; available: {supported}")


__all__ = [
    "PrimitiveFieldRenderResult",
    "available_primitive_backends",
    "available_primitive_renderers",
    "build_primitive_scene_from_config",
    "render_gaussian_scene",
    "render_gaussian_scene_naive",
    "render_gaussian_scene_splat",
    "render_gaussian_scene_wave",
    "render_gaussian_scene_wave_awb",
    "render_gaussian_scene_gws_exact",
    "render_gaussian_scene_gws_exact_awb",
    "render_gaussian_scene_gws_rpws_exact",
    "render_primitive_scene",
]

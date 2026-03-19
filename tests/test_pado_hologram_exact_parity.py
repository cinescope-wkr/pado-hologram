from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pado_hologram import SourceSpec
from pado_hologram.primitive_based import (
    build_primitive_scene_from_config,
    render_gaussian_scene_gws_exact,
    render_gaussian_scene_gws_exact_awb,
)
from pado_hologram.primitive_based.exact import (
    ProjectedGaussianPrimitive2D,
    build_angular_emission_profile,
    exact_gaussian_wavefront,
    exact_projected_gaussian_wavefronts_batched,
    make_frequency_grid,
    project_gaussian3d_to_hologram_space,
    project_gaussian3d_to_parallel,
    sample_structured_random_phase_kernel,
    sample_structured_random_phase_kernels,
)
from pado_hologram.representations import GaussianPrimitive3D


def _reference_project_covariance(
    primitive: GaussianPrimitive3D,
    *,
    pitch: float,
    rows: int,
    cols: int,
    focal_px: tuple[float, float],
    principal_px: tuple[float, float],
) -> ProjectedGaussianPrimitive2D:
    dtype = torch.float64
    fx_m = float(focal_px[0]) * pitch
    fy_m = float(focal_px[1]) * pitch
    cx_offset = (float(principal_px[0]) - cols / 2.0) * pitch
    cy_offset = (float(principal_px[1]) - rows / 2.0) * pitch

    x, y, z = primitive.mean_xyz
    sx, sy, sz = primitive.scale_xyz
    covariance_xyz = torch.diag(torch.tensor([sx**2, sy**2, sz**2], dtype=dtype))
    jacobian = torch.tensor(
        [
            [fx_m / z, 0.0, -fx_m * x / (z**2)],
            [0.0, fy_m / z, -fy_m * y / (z**2)],
        ],
        dtype=dtype,
    )
    covariance_xy = jacobian @ covariance_xyz @ jacobian.transpose(0, 1)
    projected_x = fx_m * x / z + cx_offset
    projected_y = fy_m * y / z + cy_offset
    covariance_yx = covariance_xy[[1, 0]][:, [1, 0]]
    return ProjectedGaussianPrimitive2D(
        mean_yx=(float(projected_y), float(projected_x)),
        cov_yx=(
            (float(covariance_yx[0, 0]), float(covariance_yx[0, 1])),
            (float(covariance_yx[1, 0]), float(covariance_yx[1, 1])),
        ),
        depth=float(z),
        amplitude=float(primitive.amplitude),
        phase=float(primitive.phase),
    )


def _reference_exact_wavefront_identity(
    primitive: GaussianPrimitive3D,
    *,
    rows: int,
    cols: int,
    pitch: float,
    wavelength: float,
    phase_matching: bool = True,
) -> torch.Tensor:
    dtype = torch.float64
    device = torch.device("cpu")
    dfx = 1.0 / pitch / cols
    dfy = 1.0 / pitch / rows
    fx_axis = torch.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols, dtype=dtype, device=device) * dfx
    fy_axis = torch.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows, dtype=dtype, device=device) * dfy
    fx, fy = torch.meshgrid(fx_axis, fy_axis, indexing="xy")
    fz = torch.sqrt(torch.clamp((1.0 / wavelength) ** 2 - fx.square() - fy.square(), min=0.0))
    fz = torch.clamp(fz, min=torch.finfo(dtype).eps)

    x, y, z = primitive.mean_xyz
    sx, sy, _ = primitive.scale_xyz
    gaussian_ref = (
        2.0
        * torch.pi
        * (sx * sy)
        * torch.exp(-((2.0 * torch.pi) ** 2) * ((sx * fx) ** 2 + (sy * fy) ** 2) / 2.0)
    )
    spectrum = gaussian_ref * torch.exp(-1j * 2.0 * torch.pi * (fx * x + fy * y))
    if phase_matching:
        spectrum = spectrum * torch.exp(1j * torch.tensor(2.0 * torch.pi / wavelength * z, dtype=dtype, device=device))
    spectrum = spectrum * torch.exp(-1j * 2.0 * torch.pi * fz * z)
    spectrum = spectrum * primitive.amplitude * torch.exp(1j * torch.tensor(primitive.phase, dtype=dtype))
    wavefront = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1), norm="backward"),
        dim=(-2, -1),
    )
    return wavefront * (1.0 / pitch**2)


def test_projection_to_parallel_matches_reference_covariance() -> None:
    primitive = GaussianPrimitive3D(
        mean_xyz=(4.0e-5, -2.0e-5, 1.5e-3),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        scale_xyz=(9.0e-6, 5.0e-6, 1.0e-9),
        opacity=0.6,
        amplitude=0.8,
        phase=0.3,
    )
    projected = project_gaussian3d_to_parallel(
        primitive,
        pitch=6.4e-6,
        rows=32,
        cols=48,
        device=torch.device("cpu"),
        dtype=torch.float64,
        focal_px=(120.0, 132.0),
        principal_px=(26.0, 15.0),
    )
    reference = _reference_project_covariance(
        primitive,
        pitch=6.4e-6,
        rows=32,
        cols=48,
        focal_px=(120.0, 132.0),
        principal_px=(26.0, 15.0),
    )

    assert projected.mean_yx == reference.mean_yx
    assert torch.allclose(
        projected.covariance_matrix(device=torch.device("cpu"), dtype=torch.float64),
        reference.covariance_matrix(device=torch.device("cpu"), dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_projection_to_hologram_space_respects_view_matrix_and_intrinsics() -> None:
    primitive = GaussianPrimitive3D(
        mean_xyz=(0.0, 0.0, 2.0e-3),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        scale_xyz=(8.0e-6, 6.0e-6, 1.0e-9),
        opacity=0.7,
    )
    projected = project_gaussian3d_to_hologram_space(
        primitive,
        pitch=6.4e-6,
        rows=32,
        cols=32,
        device=torch.device("cpu"),
        dtype=torch.float64,
        K_px=((150.0, 0.0, 18.0), (0.0, 145.0, 13.0), (0.0, 0.0, 1.0)),
        view_matrix=((1.0, 0.0, 0.0, 1.0e-5), (0.0, 1.0, 0.0, -2.0e-5), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )

    expected_x = 150.0 * 6.4e-6 * (1.0e-5 / 2.0e-3) + (18.0 - 16.0) * 6.4e-6
    expected_y = 145.0 * 6.4e-6 * (-2.0e-5 / 2.0e-3) + (13.0 - 16.0) * 6.4e-6

    assert abs(projected.mean_xyz[0] - expected_x) < 1e-12
    assert abs(projected.mean_xyz[1] - expected_y) < 1e-12
    assert abs(projected.mean_xyz[2] - 2.0e-3) < 1e-15


def test_exact_wavefront_matches_reference_for_identity_gaussian() -> None:
    primitive = GaussianPrimitive3D(
        mean_xyz=(1.2e-5, -0.8e-5, 1.1e-3),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        scale_xyz=(7.0e-6, 9.0e-6, 0.0),
        opacity=1.0,
        amplitude=0.9,
        phase=0.25,
    )

    ours, _ = exact_gaussian_wavefront(
        primitive,
        rows=16,
        cols=20,
        pitch=6.4e-6,
        wavelength=532e-9,
        device=torch.device("cpu"),
        dtype=torch.float64,
        phase_matching=True,
        return_at_object_depth=False,
    )
    reference = _reference_exact_wavefront_identity(
        primitive,
        rows=16,
        cols=20,
        pitch=6.4e-6,
        wavelength=532e-9,
        phase_matching=True,
    )

    assert torch.allclose(ours, reference.to(ours.dtype), atol=1e-8, rtol=1e-8)


def test_exact_awb_matches_exact_for_single_opaque_primitive() -> None:
    scene = build_primitive_scene_from_config(
        {
            "gaussians_3d": [
                {
                    "mean_xyz": [0.0, 0.0, 1.2e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [8.0e-6, 8.0e-6, 1.0e-9],
                    "opacity": 1.0,
                    "amplitude": 1.0,
                }
            ],
            "projection": {
                "K_px": [[96.0, 0.0, 8.0], [0.0, 96.0, 8.0], [0.0, 0.0, 1.0]],
                "phase_matching": True,
            },
        }
    )
    source = SourceSpec(dim=(1, 1, 16, 16), pitch=6.4e-6, wvl=532e-9)

    exact = render_gaussian_scene_gws_exact(scene, source.dim, source_spec=source)
    awb = render_gaussian_scene_gws_exact_awb(scene, source.dim, source_spec=source, sort_order="front2back")

    assert torch.allclose(exact.field, awb.field, atol=2e-4, rtol=2e-4)


def test_batched_exact_wavefront_matches_single_projected_exact() -> None:
    primitive = GaussianPrimitive3D(
        mean_xyz=(1.5e-5, -1.0e-5, 1.35e-3),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        scale_xyz=(8.0e-6, 6.5e-6, 1.0e-9),
        opacity=0.7,
        amplitude=0.85,
        phase=0.15,
    )
    projected = project_gaussian3d_to_hologram_space(
        primitive,
        pitch=6.4e-6,
        rows=18,
        cols=20,
        device=torch.device("cpu"),
        dtype=torch.float64,
        focal_px=(110.0, 115.0),
    )

    single, _ = exact_gaussian_wavefront(
        projected,
        rows=18,
        cols=20,
        pitch=6.4e-6,
        wavelength=532e-9,
        device=torch.device("cpu"),
        dtype=torch.float64,
        phase_matching=True,
        return_at_object_depth=False,
    )
    batched, _ = exact_projected_gaussian_wavefronts_batched(
        (projected,),
        rows=18,
        cols=20,
        pitch=6.4e-6,
        wavelength=532e-9,
        device=torch.device("cpu"),
        dtype=torch.float64,
        phase_matching=True,
        return_at_object_depth=False,
    )

    assert batched.shape == (1, 18, 20)
    assert torch.allclose(single, batched[0], atol=1e-8, rtol=1e-8)


def test_batched_structured_random_phase_matches_single_sampling_with_same_seed() -> None:
    fx, fy, _, band_mask = make_frequency_grid(
        12,
        14,
        pitch=6.4e-6,
        wavelength=532e-9,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    angular_profile = build_angular_emission_profile(
        fx,
        fy,
        wavelength=532e-9,
        profile="gaussian",
        band_mask=band_mask,
    )

    generator_single = torch.Generator(device="cpu")
    generator_single.manual_seed(9)
    _, kernel_a = sample_structured_random_phase_kernel(
        angular_profile,
        generator=generator_single,
        phase_range=torch.pi,
    )
    _, kernel_b = sample_structured_random_phase_kernel(
        angular_profile,
        generator=generator_single,
        phase_range=torch.pi,
    )

    generator_batch = torch.Generator(device="cpu")
    generator_batch.manual_seed(9)
    _, kernels = sample_structured_random_phase_kernels(
        angular_profile,
        num_samples=2,
        generator=generator_batch,
        phase_range=torch.pi,
    )

    assert torch.allclose(kernel_a, kernels[0])
    assert torch.allclose(kernel_b, kernels[1])


def test_structured_random_phase_kernel_has_unit_rms_magnitude() -> None:
    fx, fy, _, band_mask = make_frequency_grid(
        16,
        16,
        pitch=6.4e-6,
        wavelength=532e-9,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    angular_profile = build_angular_emission_profile(
        fx,
        fy,
        wavelength=532e-9,
        profile="uniform",
        band_mask=band_mask,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)
    _, kernel = sample_structured_random_phase_kernel(
        angular_profile,
        generator=generator,
        phase_range=torch.pi,
    )

    rms = torch.sqrt(torch.mean(kernel.abs().square()))
    assert abs(float(rms) - 1.0) < 1e-10


def test_circular_pupil_profile_restricts_frequency_support() -> None:
    fx, fy, _, band_mask = make_frequency_grid(
        32,
        32,
        pitch=6.4e-6,
        wavelength=532e-9,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    uniform = build_angular_emission_profile(
        fx,
        fy,
        wavelength=532e-9,
        profile="uniform",
        band_mask=band_mask,
    )
    pupil = build_angular_emission_profile(
        fx,
        fy,
        wavelength=532e-9,
        profile="circular_pupil",
        radius_fraction=0.35,
        band_mask=band_mask,
    )

    assert float(pupil.sum()) < float(uniform.sum())

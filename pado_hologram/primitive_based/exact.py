from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from ..representations.primitives.gaussian3d import GaussianPrimitive3D


@dataclass(frozen=True)
class ProjectedGaussianPrimitive2D:
    """Perspective-projected 2D Gaussian covariance for parity/debug checks."""

    mean_yx: tuple[float, float]
    cov_yx: tuple[tuple[float, float], tuple[float, float]]
    depth: float
    amplitude: float
    phase: float

    def covariance_matrix(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.cov_yx, device=device, dtype=dtype)


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def centered_fft2_ortho(field: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(field, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


def centered_ifft2_backward(spectrum: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1), norm="backward"),
        dim=(-2, -1),
    )


def coordinate_rotation_matrix(axis: str, angle: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(angle):
        angle = torch.tensor(float(angle), dtype=torch.float32)
    angle = angle.to(dtype=torch.get_default_dtype())
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)

    if axis == "x":
        rows = (
            torch.stack([one, zero, zero]),
            torch.stack([zero, cos, -sin]),
            torch.stack([zero, sin, cos]),
        )
    elif axis == "y":
        rows = (
            torch.stack([cos, zero, sin]),
            torch.stack([zero, one, zero]),
            torch.stack([-sin, zero, cos]),
        )
    elif axis == "z":
        rows = (
            torch.stack([cos, -sin, zero]),
            torch.stack([sin, cos, zero]),
            torch.stack([zero, zero, one]),
        )
    else:
        raise ValueError(f"unsupported rotation axis {axis!r}")
    return torch.stack(rows, dim=0)


def _normalize_quaternion(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return quat_wxyz / torch.linalg.norm(quat_wxyz)


def quaternion_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quaternion(quat_wxyz)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]),
        ],
        dim=0,
    )


def matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    r = rotation
    trace = torch.trace(r)
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = torch.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = torch.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    quat = torch.stack([w, x, y, z], dim=0)
    return _normalize_quaternion(quat)


def quaternion_to_euler_angles_zyx(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = _normalize_quaternion(quat_wxyz)
    w, x, y, z = q
    eps = q.new_tensor(3e-4)
    theta_z = torch.atan2(-2 * (x * y - w * z), 1 - 2 * (y**2 + z**2))
    theta_y = torch.asin(torch.clamp(2 * (w * y + x * z), -1.0 + eps, 1.0 - eps))
    theta_x = torch.atan2(-2 * (y * z - w * x), 1 - 2 * (x**2 + y**2))
    ninety = q.new_tensor(math.pi / 2)
    one_eighty = q.new_tensor(math.pi)
    if theta_x > ninety:
        theta_x = theta_x - one_eighty
        theta_y = -theta_y
        theta_z = -theta_z
    elif theta_x < -ninety:
        theta_x = theta_x + one_eighty
        theta_y = -theta_y
        theta_z = -theta_z
    return theta_x, theta_y, theta_z


def make_frequency_grid(
    rows: int,
    cols: int,
    *,
    pitch: float,
    wavelength: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dfx = 1.0 / float(pitch) / float(cols)
    dfy = 1.0 / float(pitch) / float(rows)
    fx_axis = torch.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols, device=device, dtype=dtype) * dfx
    fy_axis = torch.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows, device=device, dtype=dtype) * dfy
    fx, fy = torch.meshgrid(fx_axis, fy_axis, indexing="xy")
    radial_sq = fx.square() + fy.square()
    band_mask = radial_sq <= (1.0 / float(wavelength)) ** 2
    eps = torch.finfo(dtype).eps
    fz = torch.sqrt(torch.clamp((1.0 / float(wavelength)) ** 2 - radial_sq, min=0.0))
    fz = torch.clamp(fz, min=eps)
    return fx, fy, fz, band_mask


def rotate_frequency_grid(
    rotation: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor | None = None,
    *,
    wavelength: float | None = None,
) -> tuple[torch.Tensor, ...]:
    if fz is None:
        flx = rotation[0, 0] * fx + rotation[0, 1] * fy
        fly = rotation[1, 0] * fx + rotation[1, 1] * fy
        return flx, fly
    flx = rotation[0, 0] * fx + rotation[0, 1] * fy + rotation[0, 2] * fz
    fly = rotation[1, 0] * fx + rotation[1, 1] * fy + rotation[1, 2] * fz
    eps = torch.finfo(fx.dtype).eps
    flz = torch.sqrt(torch.clamp((1.0 / float(wavelength)) ** 2 - flx.square() - fly.square(), min=0.0))
    flz = torch.clamp(flz, min=eps)
    return flx, fly, flz


def resolve_intrinsics_matrix_px(
    *,
    rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype,
    K_px: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None,
    focal_px: tuple[float, float] | None = None,
    principal_px: tuple[float, float] | None = None,
) -> torch.Tensor:
    if K_px is not None:
        return torch.tensor(K_px, device=device, dtype=dtype)
    fx = float(focal_px[0]) if focal_px is not None else float(cols) / 2.0
    fy = float(focal_px[1]) if focal_px is not None else float(rows) / 2.0
    cx = float(principal_px[0]) if principal_px is not None else float(cols) / 2.0
    cy = float(principal_px[1]) if principal_px is not None else float(rows) / 2.0
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype)


def resolve_view_matrix(
    *,
    device: torch.device,
    dtype: torch.dtype,
    view_matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] | None = None,
) -> torch.Tensor:
    if view_matrix is None:
        return torch.eye(4, device=device, dtype=dtype)
    return torch.tensor(view_matrix, device=device, dtype=dtype)


def apply_view_transform_to_primitive(
    primitive: GaussianPrimitive3D,
    *,
    view_matrix: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> GaussianPrimitive3D:
    mean = torch.tensor(primitive.mean_xyz, device=device, dtype=dtype)
    rotation = quaternion_to_matrix(torch.tensor(primitive.quat_wxyz, device=device, dtype=dtype))
    view_rotation = view_matrix[:3, :3]
    view_translation = view_matrix[:3, 3]
    mean_camera = view_rotation @ mean + view_translation
    rotation_camera = view_rotation @ rotation
    quat_camera = matrix_to_quaternion(rotation_camera)
    return GaussianPrimitive3D(
        mean_xyz=tuple(float(v) for v in mean_camera),
        quat_wxyz=tuple(float(v) for v in quat_camera),
        scale_xyz=primitive.scale_xyz,
        opacity=primitive.opacity,
        amplitude=primitive.amplitude,
        phase=primitive.phase,
    )


def project_gaussian3d_to_parallel(
    primitive: GaussianPrimitive3D,
    *,
    pitch: float,
    rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype,
    focal_px: tuple[float, float] | None = None,
    principal_px: tuple[float, float] | None = None,
    K_px: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None,
    view_matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] | None = None,
) -> ProjectedGaussianPrimitive2D:
    camera_primitive = apply_view_transform_to_primitive(
        primitive,
        view_matrix=resolve_view_matrix(device=device, dtype=dtype, view_matrix=view_matrix),
        device=device,
        dtype=dtype,
    )
    K = resolve_intrinsics_matrix_px(
        rows=rows,
        cols=cols,
        device=device,
        dtype=dtype,
        K_px=K_px,
        focal_px=focal_px,
        principal_px=principal_px,
    )
    fx_m = K[0, 0] * float(pitch)
    fy_m = K[1, 1] * float(pitch)
    cx_offset_m = (K[0, 2] - float(cols) / 2.0) * float(pitch)
    cy_offset_m = (K[1, 2] - float(rows) / 2.0) * float(pitch)

    mean = torch.tensor(camera_primitive.mean_xyz, device=device, dtype=dtype)
    x, y, z = mean
    if float(z) <= 0:
        raise ValueError(f"primitive projects behind the hologram/camera plane: z={float(z)}")
    quat = torch.tensor(camera_primitive.quat_wxyz, device=device, dtype=dtype)
    scales = torch.tensor(camera_primitive.scale_xyz, device=device, dtype=dtype)
    rotation = quaternion_to_matrix(quat)
    covariance_xyz = rotation @ torch.diag(scales.square()) @ rotation.transpose(0, 1)
    zero = torch.zeros((), device=device, dtype=dtype)
    jacobian = torch.stack(
        [
            torch.stack([fx_m / z, zero, -fx_m * x / z.square()]),
            torch.stack([zero, fy_m / z, -fy_m * y / z.square()]),
        ],
        dim=0,
    )
    covariance_xy = jacobian @ covariance_xyz @ jacobian.transpose(0, 1)
    covariance_xy = 0.5 * (covariance_xy + covariance_xy.transpose(0, 1))
    covariance_scale = torch.clamp(covariance_xy.abs().max(), min=torch.finfo(dtype).tiny)
    covariance_xy = covariance_xy + torch.eye(2, device=device, dtype=dtype) * (torch.finfo(dtype).eps * covariance_scale)
    projected_x = fx_m * x / z + cx_offset_m
    projected_y = fy_m * y / z + cy_offset_m
    covariance_yx = covariance_xy[[1, 0]][:, [1, 0]]
    return ProjectedGaussianPrimitive2D(
        mean_yx=(float(projected_y), float(projected_x)),
        cov_yx=(
            (float(covariance_yx[0, 0]), float(covariance_yx[0, 1])),
            (float(covariance_yx[1, 0]), float(covariance_yx[1, 1])),
        ),
        depth=float(z),
        amplitude=float(camera_primitive.amplitude),
        phase=float(camera_primitive.phase),
    )


def project_gaussian3d_to_hologram_space(
    primitive: GaussianPrimitive3D,
    *,
    pitch: float,
    rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype,
    focal_px: tuple[float, float] | None = None,
    principal_px: tuple[float, float] | None = None,
    K_px: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None,
    view_matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] | None = None,
) -> GaussianPrimitive3D:
    projected = project_gaussian3d_to_parallel(
        primitive,
        pitch=pitch,
        rows=rows,
        cols=cols,
        device=device,
        dtype=dtype,
        focal_px=focal_px,
        principal_px=principal_px,
        K_px=K_px,
        view_matrix=view_matrix,
    )
    covariance = projected.covariance_matrix(device=device, dtype=dtype)[[1, 0]][:, [1, 0]]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigen_scale = torch.clamp(eigenvalues.abs().max(), min=torch.finfo(dtype).tiny)
    eigenvalues = torch.clamp(eigenvalues, min=torch.finfo(dtype).eps * eigen_scale)
    rotation3 = torch.eye(3, device=device, dtype=dtype)
    rotation3[:2, :2] = eigenvectors
    if torch.det(rotation3) < 0:
        rotation3[:, 0] = -rotation3[:, 0]
    quat = matrix_to_quaternion(rotation3)
    scales = torch.zeros(3, device=device, dtype=dtype)
    scales[:2] = torch.sqrt(eigenvalues)
    return GaussianPrimitive3D(
        mean_xyz=(projected.mean_yx[1], projected.mean_yx[0], projected.depth),
        quat_wxyz=tuple(float(v) for v in quat),
        scale_xyz=tuple(float(v) for v in scales),
        opacity=primitive.opacity,
        amplitude=primitive.amplitude,
        phase=primitive.phase,
    )


def project_gaussians3d_to_hologram_space(
    primitives: Sequence[GaussianPrimitive3D],
    *,
    pitch: float,
    rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype,
    focal_px: tuple[float, float] | None = None,
    principal_px: tuple[float, float] | None = None,
    K_px: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None,
    view_matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] | None = None,
) -> tuple[GaussianPrimitive3D, ...]:
    return tuple(
        project_gaussian3d_to_hologram_space(
            primitive,
            pitch=pitch,
            rows=rows,
            cols=cols,
            device=device,
            dtype=dtype,
            focal_px=focal_px,
            principal_px=principal_px,
            K_px=K_px,
            view_matrix=view_matrix,
        )
        for primitive in primitives
    )


def stack_gaussian3d_parameters(
    primitives: Sequence[GaussianPrimitive3D],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not primitives:
        empty = torch.empty((0,), device=device, dtype=dtype)
        return (
            empty.view(0, 3),
            empty.view(0, 4),
            empty.view(0, 3),
            empty,
            empty,
            empty,
        )
    means = torch.tensor([primitive.mean_xyz for primitive in primitives], device=device, dtype=dtype)
    quats = torch.tensor([primitive.quat_wxyz for primitive in primitives], device=device, dtype=dtype)
    scales = torch.tensor([primitive.scale_xyz for primitive in primitives], device=device, dtype=dtype)
    opacities = torch.tensor([primitive.opacity for primitive in primitives], device=device, dtype=dtype)
    amplitudes = torch.tensor([primitive.amplitude for primitive in primitives], device=device, dtype=dtype)
    phases = torch.tensor([primitive.phase for primitive in primitives], device=device, dtype=dtype)
    return means, quats, scales, opacities, amplitudes, phases


def quaternion_to_theta_z_batched(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat = quat_wxyz / torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(-2 * (x * y - w * z), 1 - 2 * (y.square() + z.square()))


def gaussian_angular_spectrum_reference(fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    return 2.0 * math.pi * torch.exp(-((2.0 * math.pi) ** 2) * (fx.square() + fy.square()) / 2.0)


def exact_gaussian_spectrum(
    primitive: GaussianPrimitive3D,
    *,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    wavelength: float,
    phase_matching: bool = True,
    return_at_object_depth: bool = False,
    band_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    device = fx.device
    dtype = fx.dtype
    coord = torch.tensor(primitive.mean_xyz, device=device, dtype=dtype)
    quat = torch.tensor(primitive.quat_wxyz, device=device, dtype=dtype)
    scale = torch.tensor(primitive.scale_xyz, device=device, dtype=dtype)

    theta_x, theta_y, theta_z = quaternion_to_euler_angles_zyx(quat)
    rx = coordinate_rotation_matrix("x", -theta_x).to(device=device, dtype=dtype)
    ry = coordinate_rotation_matrix("y", -theta_y).to(device=device, dtype=dtype)
    rz = coordinate_rotation_matrix("z", -theta_z).to(device=device, dtype=dtype)
    sinv = torch.diag(1.0 / torch.clamp(scale[:2], min=torch.finfo(dtype).eps))
    rotation = ry @ rx
    affine = sinv @ rz[:2, :2]

    flx, fly, flz = rotate_frequency_grid(rotation, fx, fy, fz, wavelength=wavelength)
    uc = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    du = (rotation @ uc) / float(wavelength)
    fx_local = flx - du[0]
    fy_local = fly - du[1]
    fx_ref, fy_ref = rotate_frequency_grid(torch.linalg.inv(affine.transpose(0, 1)), fx_local, fy_local)

    g0 = gaussian_angular_spectrum_reference(fx_ref, fy_ref)
    gl = g0 / torch.det(affine)
    if phase_matching and not return_at_object_depth:
        gl = gl * torch.exp(1j * 2.0 * math.pi / float(wavelength) * coord[2])

    spectrum = gl * flz / fz * torch.exp(-1j * 2.0 * math.pi * (fx * coord[0] + fy * coord[1]))
    if not return_at_object_depth:
        spectrum = spectrum * torch.exp(-1j * 2.0 * math.pi * fz * coord[2])
    spectrum = spectrum * coord.new_tensor(float(primitive.amplitude)) * torch.exp(
        1j * coord.new_tensor(float(primitive.phase))
    )
    if band_mask is not None:
        spectrum = spectrum * band_mask.to(dtype=spectrum.real.dtype)
    return spectrum


def exact_projected_gaussian_spectra_batched(
    primitives: Sequence[GaussianPrimitive3D],
    *,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    wavelength: float,
    phase_matching: bool = True,
    return_at_object_depth: bool = False,
    band_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not primitives:
        return torch.empty((0, *fx.shape), device=fx.device, dtype=_complex_dtype(fx.dtype))

    device = fx.device
    dtype = fx.dtype
    means, quats, scales, _, amplitudes, phases = stack_gaussian3d_parameters(primitives, device=device, dtype=dtype)
    theta_z = quaternion_to_theta_z_batched(quats)
    cos_t = torch.cos(theta_z).view(-1, 1, 1)
    sin_t = torch.sin(theta_z).view(-1, 1, 1)
    sx = torch.clamp(scales[:, 0], min=torch.finfo(dtype).eps).view(-1, 1, 1)
    sy = torch.clamp(scales[:, 1], min=torch.finfo(dtype).eps).view(-1, 1, 1)

    inv_affine_t = torch.zeros((len(primitives), 2, 2), device=device, dtype=dtype)
    inv_affine_t[:, 0, 0] = cos_t.view(-1) * sx.view(-1)
    inv_affine_t[:, 0, 1] = sin_t.view(-1) * sx.view(-1)
    inv_affine_t[:, 1, 0] = -sin_t.view(-1) * sy.view(-1)
    inv_affine_t[:, 1, 1] = cos_t.view(-1) * sy.view(-1)
    det_affine = 1.0 / (sx.view(-1) * sy.view(-1))

    fx_expanded = fx.unsqueeze(0).expand(len(primitives), -1, -1)
    fy_expanded = fy.unsqueeze(0).expand(len(primitives), -1, -1)
    fx_ref = inv_affine_t[:, 0, 0].view(-1, 1, 1) * fx_expanded + inv_affine_t[:, 0, 1].view(-1, 1, 1) * fy_expanded
    fy_ref = inv_affine_t[:, 1, 0].view(-1, 1, 1) * fx_expanded + inv_affine_t[:, 1, 1].view(-1, 1, 1) * fy_expanded

    g0 = gaussian_angular_spectrum_reference(fx_ref, fy_ref)
    gl = g0 / det_affine.view(-1, 1, 1)
    z = means[:, 2].view(-1, 1, 1)
    if phase_matching and not return_at_object_depth:
        gl = gl * torch.exp(1j * (2.0 * math.pi / float(wavelength)) * z)

    x = means[:, 0].view(-1, 1, 1)
    y = means[:, 1].view(-1, 1, 1)
    spectrum = gl * torch.exp(-1j * 2.0 * math.pi * (fx_expanded * x + fy_expanded * y))
    if not return_at_object_depth:
        spectrum = spectrum * torch.exp(-1j * 2.0 * math.pi * fz.unsqueeze(0) * z)
    spectrum = spectrum * amplitudes.view(-1, 1, 1) * torch.exp(1j * phases.view(-1, 1, 1))
    if band_mask is not None:
        spectrum = spectrum * band_mask.unsqueeze(0).to(dtype=dtype)
    return spectrum.to(dtype=_complex_dtype(dtype))


def exact_projected_gaussian_wavefronts_batched(
    primitives: Sequence[GaussianPrimitive3D],
    *,
    rows: int,
    cols: int,
    pitch: float,
    wavelength: float,
    device: torch.device,
    dtype: torch.dtype,
    phase_matching: bool = True,
    return_at_object_depth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    fx, fy, fz, band_mask = make_frequency_grid(
        rows,
        cols,
        pitch=pitch,
        wavelength=wavelength,
        device=device,
        dtype=dtype,
    )
    spectra = exact_projected_gaussian_spectra_batched(
        primitives,
        fx=fx,
        fy=fy,
        fz=fz,
        wavelength=wavelength,
        phase_matching=phase_matching,
        return_at_object_depth=return_at_object_depth,
        band_mask=band_mask,
    )
    wavefronts = centered_ifft2_backward(spectra) * (1.0 / float(pitch) ** 2)
    if return_at_object_depth:
        if not primitives:
            phase_compensation = torch.empty((0, rows, cols), device=device, dtype=dtype)
        else:
            means, _, _, _, _, _ = stack_gaussian3d_parameters(primitives, device=device, dtype=dtype)
            z = means[:, 2].view(-1, 1, 1)
            phase_compensation = 2.0 * math.pi / float(wavelength) * z - 2.0 * math.pi * fz.unsqueeze(0) * z
        return wavefronts.to(dtype=_complex_dtype(dtype)), phase_compensation
    return wavefronts.to(dtype=_complex_dtype(dtype)), None


def exact_gaussian_wavefront(
    primitive: GaussianPrimitive3D,
    *,
    rows: int,
    cols: int,
    pitch: float,
    wavelength: float,
    device: torch.device,
    dtype: torch.dtype,
    phase_matching: bool = True,
    return_at_object_depth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    fx, fy, fz, band_mask = make_frequency_grid(
        rows,
        cols,
        pitch=pitch,
        wavelength=wavelength,
        device=device,
        dtype=dtype,
    )
    spectrum = exact_gaussian_spectrum(
        primitive,
        fx=fx,
        fy=fy,
        fz=fz,
        wavelength=wavelength,
        phase_matching=phase_matching,
        return_at_object_depth=return_at_object_depth,
        band_mask=band_mask,
    )
    wavefront = centered_ifft2_backward(spectrum) * (1.0 / float(pitch) ** 2)
    if return_at_object_depth:
        phase_compensation = 2.0 * math.pi / float(wavelength) * float(primitive.mean_xyz[2]) - 2.0 * math.pi * fz * float(
            primitive.mean_xyz[2]
        )
        return wavefront.to(_complex_dtype(dtype)), phase_compensation
    return wavefront.to(_complex_dtype(dtype)), None


def apply_phase_compensation(
    object_wavefront: torch.Tensor,
    *,
    phase_compensation: torch.Tensor,
    band_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    u_in = torch.fft.ifftshift(object_wavefront, dim=(-2, -1))
    spectrum = torch.fft.fftshift(torch.fft.fftn(u_in, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    if band_mask is not None:
        spectrum = spectrum * band_mask.to(dtype=spectrum.real.dtype)
    spectrum = spectrum * torch.exp(1j * phase_compensation)
    u_out = torch.fft.ifftn(torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(u_out, dim=(-2, -1))


def build_angular_emission_profile(
    fx: torch.Tensor,
    fy: torch.Tensor,
    *,
    wavelength: float,
    profile: str = "uniform",
    radius_fraction: float = 1.0,
    gaussian_sigma_fraction: float = 0.35,
    band_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the Fourier-amplitude profile Y(k) used by structured random phase.

    The RPWS paper discusses several choices for the angular emission profile.
    We keep the surface small but explicit:
    - ``uniform``: diffuse emission over the full band-limited pupil
    - ``circular_pupil``: binary low-pass pupil used for depth-of-field control
    - ``gaussian``: smooth radial emission profile
    """

    dtype = fx.dtype
    radial = torch.sqrt(fx.square() + fy.square())
    if band_mask is not None and torch.any(band_mask):
        max_frequency = float(radial[band_mask].max())
    else:
        max_frequency = float(radial.max())
    if profile == "uniform":
        amplitude = torch.ones_like(fx, dtype=dtype)
    elif profile == "circular_pupil":
        radius = max(0.0, min(float(radius_fraction), 1.0)) * max_frequency
        amplitude = (radial <= radius).to(dtype=dtype)
    elif profile == "gaussian":
        sigma = max(float(gaussian_sigma_fraction), torch.finfo(dtype).eps) * max_frequency
        amplitude = torch.exp(-0.5 * (radial / sigma).square())
    else:
        raise ValueError(f"unsupported angular emission profile {profile!r}")

    if band_mask is not None:
        amplitude = amplitude * band_mask.to(dtype=dtype)
    return amplitude


def sample_structured_random_phase_kernel(
    angular_emission_profile: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    phase_range: float = math.pi,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a structured random-phase kernel in Fourier and spatial domains.

    RPWS modulates each primitive by convolving its angular spectrum with
    ``Y(k) exp(i q_t(k))``. In the spatial domain this becomes a multiplication
    by ``F^{-1}{Y(k) exp(i q_t(k))}``, which we return as ``kernel``.

    We normalize the sampled spatial kernel to unit RMS magnitude so that
    time-multiplexed averages preserve the primitive amplitude scale more
    predictably under the discrete FFT conventions used here.
    """

    if phase_range < 0:
        raise ValueError(f"phase_range must be non-negative, got {phase_range}")

    real_dtype = angular_emission_profile.dtype
    ones = torch.ones_like(angular_emission_profile, dtype=real_dtype)
    if phase_range == 0:
        sampled_phase = torch.zeros_like(angular_emission_profile, dtype=real_dtype)
    else:
        sampled_phase = (
            2.0
            * torch.rand(
                angular_emission_profile.shape,
                generator=generator,
                device=angular_emission_profile.device,
                dtype=real_dtype,
            )
            - 1.0
        ) * float(phase_range)
    spectrum = angular_emission_profile.to(dtype=_complex_dtype(real_dtype)) * torch.polar(ones, sampled_phase)
    kernel = centered_ifft2_backward(spectrum)
    rms = torch.sqrt(torch.mean(kernel.abs().square()).clamp_min(torch.finfo(real_dtype).eps))
    kernel = kernel / rms
    return spectrum, kernel.to(dtype=_complex_dtype(real_dtype))


def sample_structured_random_phase_kernels(
    angular_emission_profile: torch.Tensor,
    *,
    num_samples: int,
    generator: torch.Generator | None = None,
    phase_range: float = math.pi,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    if num_samples == 0:
        empty_complex = torch.empty((0, *angular_emission_profile.shape), device=angular_emission_profile.device, dtype=_complex_dtype(angular_emission_profile.dtype))
        return empty_complex, empty_complex
    if phase_range < 0:
        raise ValueError(f"phase_range must be non-negative, got {phase_range}")

    real_dtype = angular_emission_profile.dtype
    if phase_range == 0:
        sampled_phase = torch.zeros(
            (num_samples, *angular_emission_profile.shape),
            device=angular_emission_profile.device,
            dtype=real_dtype,
        )
    else:
        sampled_phase = (
            2.0
            * torch.rand(
                (num_samples, *angular_emission_profile.shape),
                generator=generator,
                device=angular_emission_profile.device,
                dtype=real_dtype,
            )
            - 1.0
        ) * float(phase_range)
    amplitude = angular_emission_profile.unsqueeze(0).to(dtype=_complex_dtype(real_dtype))
    spectrum = amplitude * torch.polar(torch.ones_like(amplitude.real), sampled_phase)
    kernel = centered_ifft2_backward(spectrum)
    rms = torch.sqrt(torch.mean(kernel.abs().square(), dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(real_dtype).eps))
    kernel = kernel / rms
    return spectrum, kernel.to(dtype=_complex_dtype(real_dtype))


__all__ = [
    "ProjectedGaussianPrimitive2D",
    "apply_phase_compensation",
    "build_angular_emission_profile",
    "coordinate_rotation_matrix",
    "exact_gaussian_spectrum",
    "exact_gaussian_wavefront",
    "exact_projected_gaussian_spectra_batched",
    "exact_projected_gaussian_wavefronts_batched",
    "gaussian_angular_spectrum_reference",
    "make_frequency_grid",
    "matrix_to_quaternion",
    "project_gaussians3d_to_hologram_space",
    "project_gaussian3d_to_hologram_space",
    "project_gaussian3d_to_parallel",
    "quaternion_to_euler_angles_zyx",
    "quaternion_to_theta_z_batched",
    "quaternion_to_matrix",
    "resolve_intrinsics_matrix_px",
    "resolve_view_matrix",
    "rotate_frequency_grid",
    "sample_structured_random_phase_kernel",
    "sample_structured_random_phase_kernels",
    "stack_gaussian3d_parameters",
]

from __future__ import annotations

import os

import torch

from ..backends import DEFAULT_WARP_CACHE_DIR
from .exact import centered_ifft2_backward, quaternion_to_theta_z_batched

_WARP_IMPORT_ERROR = None

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - import guard only
    wp = None
    _WARP_IMPORT_ERROR = exc


if wp is not None:
    _WARP_INITIALIZED = False

    @wp.kernel
    def _gaussian_splat_kernel(
        center_y: wp.array(dtype=wp.float32),
        center_x: wp.array(dtype=wp.float32),
        sigma_y: wp.array(dtype=wp.float32),
        sigma_x: wp.array(dtype=wp.float32),
        amplitude: wp.array(dtype=wp.float32),
        phase: wp.array(dtype=wp.float32),
        rotation: wp.array(dtype=wp.float32),
        real_out: wp.array2d(dtype=wp.float32),
        imag_out: wp.array2d(dtype=wp.float32),
    ):
        pixel_y, pixel_x = wp.tid()
        real_accum = wp.float32(0.0)
        imag_accum = wp.float32(0.0)

        y = wp.float32(pixel_y)
        x = wp.float32(pixel_x)

        for i in range(center_y.shape[0]):
            dy = y - center_y[i]
            dx = x - center_x[i]
            cos_t = wp.cos(rotation[i])
            sin_t = wp.sin(rotation[i])
            y_rot = cos_t * dy + sin_t * dx
            x_rot = -sin_t * dy + cos_t * dx
            gaussian = wp.exp(
                wp.float32(-0.5)
                * ((y_rot / sigma_y[i]) * (y_rot / sigma_y[i]) + (x_rot / sigma_x[i]) * (x_rot / sigma_x[i]))
            )
            contrib = amplitude[i] * gaussian
            real_accum = real_accum + contrib * wp.cos(phase[i])
            imag_accum = imag_accum + contrib * wp.sin(phase[i])

        real_out[pixel_y, pixel_x] = real_accum
        imag_out[pixel_y, pixel_x] = imag_accum

    @wp.kernel
    def _exact_projected_gaussian_spectrum_kernel(
        mean_x: wp.array(dtype=wp.float32),
        mean_y: wp.array(dtype=wp.float32),
        mean_z: wp.array(dtype=wp.float32),
        scale_x: wp.array(dtype=wp.float32),
        scale_y: wp.array(dtype=wp.float32),
        theta_z: wp.array(dtype=wp.float32),
        amplitude: wp.array(dtype=wp.float32),
        phase: wp.array(dtype=wp.float32),
        fx: wp.array2d(dtype=wp.float32),
        fy: wp.array2d(dtype=wp.float32),
        fz: wp.array2d(dtype=wp.float32),
        band_mask: wp.array2d(dtype=wp.int32),
        wavelength: float,
        phase_matching: int,
        return_at_object_depth: int,
        real_out: wp.array(dtype=wp.float32, ndim=3),
        imag_out: wp.array(dtype=wp.float32, ndim=3),
    ):
        primitive_idx, pixel_y, pixel_x = wp.tid()

        if band_mask[pixel_y, pixel_x] == 0:
            real_out[primitive_idx, pixel_y, pixel_x] = wp.float32(0.0)
            imag_out[primitive_idx, pixel_y, pixel_x] = wp.float32(0.0)
            return

        two_pi = wp.float32(6.283185307179586)
        fx_value = fx[pixel_y, pixel_x]
        fy_value = fy[pixel_y, pixel_x]
        fz_value = fz[pixel_y, pixel_x]

        cos_t = wp.cos(theta_z[primitive_idx])
        sin_t = wp.sin(theta_z[primitive_idx])
        sx = scale_x[primitive_idx]
        sy = scale_y[primitive_idx]

        fx_ref = sx * (cos_t * fx_value + sin_t * fy_value)
        fy_ref = sy * (-sin_t * fx_value + cos_t * fy_value)
        gaussian_ref = two_pi * wp.exp(
            -wp.float32(0.5)
            * (two_pi * two_pi)
            * (fx_ref * fx_ref + fy_ref * fy_ref)
        )
        local_amplitude = gaussian_ref * sx * sy * amplitude[primitive_idx]

        total_phase = phase[primitive_idx] - two_pi * (
            fx_value * mean_x[primitive_idx] + fy_value * mean_y[primitive_idx]
        )
        if phase_matching != 0 and return_at_object_depth == 0:
            total_phase = total_phase + two_pi / wavelength * mean_z[primitive_idx]
        if return_at_object_depth == 0:
            total_phase = total_phase - two_pi * fz_value * mean_z[primitive_idx]

        real_out[primitive_idx, pixel_y, pixel_x] = local_amplitude * wp.cos(total_phase)
        imag_out[primitive_idx, pixel_y, pixel_x] = local_amplitude * wp.sin(total_phase)


def primitive_warp_available() -> bool:
    return wp is not None


def primitive_warp_unavailable_reason() -> str:
    if _WARP_IMPORT_ERROR is None:
        return "Warp is not available."
    return f"Warp import failed: {_WARP_IMPORT_ERROR}"


def _ensure_warp_initialized(warp_cache_dir: str | None = None) -> None:
    global _WARP_INITIALIZED
    if wp is None:
        raise RuntimeError(primitive_warp_unavailable_reason())
    if _WARP_INITIALIZED:
        return
    wp.config.kernel_cache_dir = warp_cache_dir or os.environ.get("WARP_CACHE_DIR") or DEFAULT_WARP_CACHE_DIR
    wp.init()
    _WARP_INITIALIZED = True


def _warp_device_from_torch(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return f"cuda:{index}"
    return "cpu"


def render_gaussian_scene_warp(
    gaussian_params: torch.Tensor,
    rows: int,
    cols: int,
    *,
    device: str | torch.device = "cpu",
    warp_cache_dir: str | None = None,
) -> torch.Tensor:
    _ensure_warp_initialized(warp_cache_dir=warp_cache_dir)

    target_device = torch.device(device)
    params = gaussian_params.to(device=target_device, dtype=torch.float32).contiguous()
    if params.ndim != 2 or params.shape[1] != 7:
        raise ValueError(f"gaussian_params must have shape [N, 7], got {tuple(params.shape)}")

    real_out = torch.zeros((rows, cols), dtype=torch.float32, device=target_device)
    imag_out = torch.zeros((rows, cols), dtype=torch.float32, device=target_device)
    launch_kwargs = {
        "kernel": _gaussian_splat_kernel,
        "dim": (rows, cols),
        "inputs": [
            params[:, 0],
            params[:, 1],
            params[:, 2],
            params[:, 3],
            params[:, 4],
            params[:, 5],
            params[:, 6],
        ],
        "outputs": [real_out, imag_out],
        "device": _warp_device_from_torch(target_device),
    }
    if target_device.type == "cuda":
        launch_kwargs["stream"] = wp.stream_from_torch(
            torch.cuda.current_stream(device=target_device)
        )

    wp.launch(**launch_kwargs)
    return torch.complex(real_out, imag_out)


def render_projected_gaussian_wavefronts_warp(
    projected_params: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    band_mask: torch.Tensor,
    *,
    wavelength: float,
    pitch: float,
    phase_matching: bool,
    return_at_object_depth: bool,
    device: str | torch.device = "cpu",
    warp_cache_dir: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    _ensure_warp_initialized(warp_cache_dir=warp_cache_dir)

    target_device = torch.device(device)
    params = projected_params.to(device=target_device, dtype=torch.float32).contiguous()
    if params.ndim != 2 or params.shape[1] != 8:
        raise ValueError(f"projected_params must have shape [N, 8], got {tuple(params.shape)}")

    num_primitives = int(params.shape[0])
    rows, cols = int(fx.shape[0]), int(fx.shape[1])
    real_out = torch.zeros((num_primitives, rows, cols), dtype=torch.float32, device=target_device)
    imag_out = torch.zeros((num_primitives, rows, cols), dtype=torch.float32, device=target_device)
    band_mask_i32 = band_mask.to(device=target_device, dtype=torch.int32).contiguous()
    launch_kwargs = {
        "kernel": _exact_projected_gaussian_spectrum_kernel,
        "dim": (num_primitives, rows, cols),
        "inputs": [
            params[:, 0],
            params[:, 1],
            params[:, 2],
            params[:, 3],
            params[:, 4],
            params[:, 5],
            params[:, 6],
            params[:, 7],
            fx.to(device=target_device, dtype=torch.float32).contiguous(),
            fy.to(device=target_device, dtype=torch.float32).contiguous(),
            fz.to(device=target_device, dtype=torch.float32).contiguous(),
            band_mask_i32,
            float(wavelength),
            int(phase_matching),
            int(return_at_object_depth),
        ],
        "outputs": [real_out, imag_out],
        "device": _warp_device_from_torch(target_device),
    }
    if target_device.type == "cuda":
        launch_kwargs["stream"] = wp.stream_from_torch(torch.cuda.current_stream(device=target_device))

    wp.launch(**launch_kwargs)
    spectra = torch.complex(real_out, imag_out)
    wavefronts = centered_ifft2_backward(spectra) * (1.0 / float(pitch) ** 2)
    if return_at_object_depth:
        z = params[:, 2].view(-1, 1, 1).to(dtype=fx.dtype)
        phase_compensation = 2.0 * torch.pi / float(wavelength) * z - 2.0 * torch.pi * fz.to(dtype=fx.dtype).unsqueeze(0) * z
        return wavefronts, phase_compensation
    return wavefronts, None


def build_projected_gaussian_warp_params(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    amplitudes: torch.Tensor,
    phases: torch.Tensor,
) -> torch.Tensor:
    theta_z = quaternion_to_theta_z_batched(quats)
    return torch.stack(
        [
            means[:, 0],
            means[:, 1],
            means[:, 2],
            scales[:, 0],
            scales[:, 1],
            theta_z,
            amplitudes,
            phases,
        ],
        dim=-1,
    )


__all__ = [
    "build_projected_gaussian_warp_params",
    "primitive_warp_available",
    "primitive_warp_unavailable_reason",
    "render_projected_gaussian_wavefronts_warp",
    "render_gaussian_scene_warp",
]

from __future__ import annotations

from typing import Dict, Sequence

import torch

from pado.light import Light
from pado.math import calculate_psnr, calculate_ssim

from .._tensor import coerce_4d_real
from .targets import IntensityTarget, MultiPlaneIntensityTarget, normalize_mean_intensity


def as_intensity_tensor(prediction: Light | torch.Tensor) -> torch.Tensor:
    if isinstance(prediction, Light):
        return prediction.get_intensity().real
    return coerce_4d_real(prediction, name="prediction")


def intensity_mse(
    prediction: Light | torch.Tensor,
    target: IntensityTarget,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    pred = as_intensity_tensor(prediction).float()
    tgt = target.tensor(normalize=normalize).to(pred.device).float()
    if normalize:
        pred = normalize_mean_intensity(pred)
    return torch.mean((pred - tgt) ** 2)


def amplitude_mse(
    prediction: Light | torch.Tensor,
    target: IntensityTarget,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    pred = torch.sqrt(as_intensity_tensor(prediction).float().clamp_min(0.0))
    tgt = target.amplitude(normalize=normalize).to(pred.device).float()
    if normalize:
        pred = torch.sqrt(normalize_mean_intensity(pred.square()).clamp_min(0.0))
    return torch.mean((pred - tgt) ** 2)


def reconstruction_metrics(
    prediction: Light | torch.Tensor,
    target: IntensityTarget,
    *,
    normalize: bool = True,
    data_range: float = 1.0,
) -> Dict[str, torch.Tensor]:
    pred = as_intensity_tensor(prediction).float()
    tgt = target.tensor(normalize=normalize).to(pred.device).float()
    if normalize:
        pred = normalize_mean_intensity(pred)

    mse = torch.mean((pred - tgt) ** 2)
    psnr = calculate_psnr(pred, tgt, data_range=data_range)

    spatial_extent = min(pred.shape[-2], pred.shape[-1])
    if spatial_extent < 3:
        ssim = torch.tensor(float("nan"), device=pred.device)
    else:
        window_size = min(21, spatial_extent if spatial_extent % 2 == 1 else spatial_extent - 1)
        window_size = max(window_size, 3)
        ssim = calculate_ssim(pred, tgt, window_size=window_size, data_range=data_range)

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
    }


def tensor_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    normalize: bool = True,
    data_range: float = 1.0,
) -> Dict[str, torch.Tensor]:
    pred = coerce_4d_real(prediction, name="prediction").float()
    tgt = coerce_4d_real(target, name="target", dim=tuple(pred.shape)).to(pred.device).float()
    if normalize:
        pred = normalize_mean_intensity(pred)
        tgt = normalize_mean_intensity(tgt)

    mse = torch.mean((pred - tgt) ** 2)
    psnr = calculate_psnr(pred, tgt, data_range=data_range)

    spatial_extent = min(pred.shape[-2], pred.shape[-1])
    if spatial_extent < 3:
        ssim = torch.tensor(float("nan"), device=pred.device)
    else:
        window_size = min(21, spatial_extent if spatial_extent % 2 == 1 else spatial_extent - 1)
        window_size = max(window_size, 3)
        ssim = calculate_ssim(pred, tgt, window_size=window_size, data_range=data_range)

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
    }


def multi_plane_intensity_mse(
    predictions: Sequence[Light | torch.Tensor],
    target: MultiPlaneIntensityTarget,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    if len(predictions) != len(target):
        raise ValueError(f"predictions must match target length {len(target)}, got {len(predictions)}")

    losses = [
        intensity_mse(prediction, plane_target, normalize=normalize)
        for prediction, plane_target in zip(predictions, target)
    ]
    return torch.stack(losses).mean()


def multi_plane_reconstruction_metrics(
    predictions: Sequence[Light | torch.Tensor],
    target: MultiPlaneIntensityTarget,
    *,
    normalize: bool = True,
    data_range: float = 1.0,
) -> Dict[str, torch.Tensor | tuple[Dict[str, torch.Tensor], ...]]:
    if len(predictions) != len(target):
        raise ValueError(f"predictions must match target length {len(target)}, got {len(predictions)}")

    per_plane = tuple(
        reconstruction_metrics(prediction, plane_target, normalize=normalize, data_range=data_range)
        for prediction, plane_target in zip(predictions, target)
    )
    mse = torch.stack([torch.as_tensor(plane["mse"]) for plane in per_plane]).mean()
    psnr = torch.stack([torch.as_tensor(plane["psnr"]) for plane in per_plane]).mean()
    ssim = torch.stack([torch.as_tensor(plane["ssim"]) for plane in per_plane]).mean()
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "per_plane": per_plane,
    }


__all__ = [
    "amplitude_mse",
    "as_intensity_tensor",
    "intensity_mse",
    "multi_plane_intensity_mse",
    "multi_plane_reconstruction_metrics",
    "reconstruction_metrics",
    "tensor_reconstruction_metrics",
]

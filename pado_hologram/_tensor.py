from __future__ import annotations

from typing import Tuple

import torch

from pado.light import Light


def coerce_4d_real(
    tensor: torch.Tensor,
    *,
    name: str,
    dim: Tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if tensor.is_complex():
        raise TypeError(f"{name} must be a real tensor, got complex dtype {tensor.dtype}")

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape (R,C), (B,R,C), or (B,Ch,R,C); got {tuple(tensor.shape)}"
        )

    if dim is not None and tuple(tensor.shape) != tuple(dim):
        raise ValueError(f"{name} must have shape {dim}, got {tuple(tensor.shape)}")

    return tensor


def coerce_4d_complex(
    tensor: torch.Tensor,
    *,
    name: str,
    dim: Tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if not tensor.is_complex():
        raise TypeError(f"{name} must be a complex tensor, got dtype {tensor.dtype}")

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape (R,C), (B,R,C), or (B,Ch,R,C); got {tuple(tensor.shape)}"
        )

    if dim is not None and tuple(tensor.shape) != tuple(dim):
        raise ValueError(f"{name} must have shape {dim}, got {tuple(tensor.shape)}")

    return tensor


def validate_light_against_source(light: Light, dim: Tuple[int, int, int, int], pitch: float, wvl: float) -> None:
    if tuple(light.dim) != tuple(dim):
        raise ValueError(f"light.dim must match source dim {dim}, got {light.dim}")
    if abs(float(light.pitch) - float(pitch)) > 1e-12:
        raise ValueError(f"light.pitch must match source pitch {pitch}, got {light.pitch}")

    if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
        if len(light.wvl) != 1 or abs(float(light.wvl[0]) - float(wvl)) > 1e-12:
            raise ValueError(f"light.wvl must match scalar source wavelength {wvl}, got {light.wvl}")
        return

    if abs(float(light.wvl) - float(wvl)) > 1e-12:
        raise ValueError(f"light.wvl must match source wavelength {wvl}, got {light.wvl}")

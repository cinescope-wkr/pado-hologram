from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from typing import Optional, Sequence

import torch

SUPPORTED_KERNEL_BACKENDS = ("auto", "torch", "warp")
DEFAULT_WARP_CACHE_DIR = os.path.join(tempfile.gettempdir(), "pado-hologram-warp-cache")

_WARP_MODULE = None
_WARP_INIT_ERROR: Exception | None = None
_CHECKERBOARD_MASK_KERNEL = None
_CHECKERBOARD_PHASE_KERNEL = None


@dataclass(frozen=True)
class KernelBackendSelection:
    requested: str
    resolved: str
    reason: str


def _torch_device_name(device: str | torch.device) -> str:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if torch_device.index is None:
            return "cuda"
        return f"cuda:{torch_device.index}"
    return torch_device.type


def _torch_checkerboard_mask(
    dim: Sequence[int],
    *,
    device: str | torch.device,
) -> torch.Tensor:
    rows = torch.arange(int(dim[2]), device=device).view(1, 1, -1, 1)
    cols = torch.arange(int(dim[3]), device=device).view(1, 1, 1, -1)
    return ((rows + cols) % 2 == 0).expand(tuple(int(v) for v in dim))


def _ensure_warp(
    *,
    warp_cache_dir: Optional[str] = None,
):
    global _WARP_MODULE, _WARP_INIT_ERROR, _CHECKERBOARD_MASK_KERNEL, _CHECKERBOARD_PHASE_KERNEL

    if _WARP_MODULE is not None:
        return _WARP_MODULE
    if _WARP_INIT_ERROR is not None:
        raise RuntimeError("Warp initialization previously failed") from _WARP_INIT_ERROR

    try:
        import warp as wp

        wp.config.kernel_cache_dir = warp_cache_dir or wp.config.kernel_cache_dir or DEFAULT_WARP_CACHE_DIR
        wp.init()

        @wp.kernel
        def checkerboard_mask_kernel(out: wp.array(dtype=wp.int32, ndim=4)):
            b, c, y, x = wp.tid()
            out[b, c, y, x] = 1 if ((y + x) & 1) == 0 else 0

        @wp.kernel
        def checkerboard_phase_kernel(
            phase_a: wp.array(dtype=wp.float32, ndim=4),
            phase_b: wp.array(dtype=wp.float32, ndim=4),
            out: wp.array(dtype=wp.float32, ndim=4),
        ):
            b, c, y, x = wp.tid()
            if ((y + x) & 1) == 0:
                out[b, c, y, x] = phase_a[b, c, y, x]
            else:
                out[b, c, y, x] = phase_b[b, c, y, x]

        _WARP_MODULE = wp
        _CHECKERBOARD_MASK_KERNEL = checkerboard_mask_kernel
        _CHECKERBOARD_PHASE_KERNEL = checkerboard_phase_kernel
        return _WARP_MODULE
    except Exception as exc:  # pragma: no cover - exercised through fallback logic
        _WARP_INIT_ERROR = exc
        raise RuntimeError("Warp is installed but could not be initialized") from exc


def _warp_supports_device(
    wp,
    *,
    device: str | torch.device,
) -> bool:
    device_name = _torch_device_name(device)
    try:
        wp.get_device(device_name)
    except Exception:
        return False
    return True


def resolve_kernel_backend(
    requested: str = "auto",
    *,
    device: str | torch.device = "cpu",
    warp_cache_dir: Optional[str] = None,
) -> KernelBackendSelection:
    if requested not in SUPPORTED_KERNEL_BACKENDS:
        raise ValueError(
            f"requested backend must be one of {SUPPORTED_KERNEL_BACKENDS}, got {requested}"
        )
    if requested == "torch":
        return KernelBackendSelection("torch", "torch", "PyTorch backend selected explicitly.")

    try:
        wp = _ensure_warp(warp_cache_dir=warp_cache_dir)
    except RuntimeError as exc:
        if requested == "warp":
            raise RuntimeError("Warp backend was requested but is not available.") from exc
        return KernelBackendSelection(
            requested,
            "torch",
            "Warp could not be initialized, so PyTorch remains the active backend.",
        )

    if not _warp_supports_device(wp, device=device):
        device_name = _torch_device_name(device)
        if requested == "warp":
            raise RuntimeError(f"Warp backend was requested but does not support device {device_name}.")
        return KernelBackendSelection(
            requested,
            "torch",
            f"Warp is available but does not support device {device_name}, so PyTorch remains active.",
        )

    return KernelBackendSelection(
        requested,
        "warp",
        "Warp is available and is being used for the current custom holography kernel path.",
    )


def warp_checkerboard_mask(
    dim: Sequence[int],
    *,
    backend: str = "auto",
    device: str | torch.device = "cpu",
    warp_cache_dir: Optional[str] = None,
) -> torch.Tensor:
    selection = resolve_kernel_backend(
        backend,
        device=device,
        warp_cache_dir=warp_cache_dir,
    )
    if selection.resolved == "torch":
        return _torch_checkerboard_mask(dim, device=device)

    wp = _ensure_warp(warp_cache_dir=warp_cache_dir)
    device_name = _torch_device_name(device)
    mask = torch.empty(tuple(int(v) for v in dim), dtype=torch.int32, device=device)
    wp.launch(
        _CHECKERBOARD_MASK_KERNEL,
        dim=mask.shape,
        inputs=[wp.from_torch(mask, dtype=wp.int32)],
        device=device_name,
    )
    wp.synchronize()
    return mask.bool()


def checkerboard_phase_select(
    phase_a: torch.Tensor,
    phase_b: torch.Tensor,
    *,
    backend: str = "auto",
    device: str | torch.device | None = None,
    warp_cache_dir: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, KernelBackendSelection]:
    if phase_a.shape != phase_b.shape:
        raise ValueError(f"phase_a and phase_b must have the same shape, got {phase_a.shape} and {phase_b.shape}")
    if phase_a.device != phase_b.device:
        raise ValueError(f"phase_a and phase_b must be on the same device, got {phase_a.device} and {phase_b.device}")

    target_device = phase_a.device if device is None else torch.device(device)
    selection = resolve_kernel_backend(
        backend,
        device=target_device,
        warp_cache_dir=warp_cache_dir,
    )

    mask = warp_checkerboard_mask(
        phase_a.shape,
        backend=selection.resolved,
        device=target_device,
        warp_cache_dir=warp_cache_dir,
    )

    if selection.resolved == "torch":
        return torch.where(mask, phase_a, phase_b), mask, selection

    if phase_a.dtype != torch.float32 or phase_b.dtype != torch.float32:
        fallback = KernelBackendSelection(
            selection.requested,
            "torch",
            "Warp custom kernels currently operate on float32 phase tensors, so PyTorch handled this phase selection.",
        )
        return torch.where(mask, phase_a, phase_b), mask, fallback

    wp = _ensure_warp(warp_cache_dir=warp_cache_dir)
    device_name = _torch_device_name(target_device)
    selected = torch.empty_like(phase_a)
    wp.launch(
        _CHECKERBOARD_PHASE_KERNEL,
        dim=tuple(int(v) for v in phase_a.shape),
        inputs=[
            wp.from_torch(phase_a.contiguous(), dtype=wp.float32),
            wp.from_torch(phase_b.contiguous(), dtype=wp.float32),
            wp.from_torch(selected, dtype=wp.float32),
        ],
        device=device_name,
    )
    wp.synchronize()
    return selected, mask, selection

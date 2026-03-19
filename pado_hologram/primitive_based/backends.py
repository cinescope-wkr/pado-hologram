from __future__ import annotations

from ..backends import (
    KernelBackendSelection,
    SUPPORTED_KERNEL_BACKENDS,
    resolve_kernel_backend,
)


def available_primitive_backends() -> tuple[str, ...]:
    return SUPPORTED_KERNEL_BACKENDS


def resolve_primitive_backend(
    requested: str = "auto",
    *,
    device: str = "cpu",
    warp_cache_dir: str | None = None,
) -> KernelBackendSelection:
    return resolve_kernel_backend(
        requested=requested,
        device=device,
        warp_cache_dir=warp_cache_dir,
    )


__all__ = [
    "KernelBackendSelection",
    "available_primitive_backends",
    "resolve_primitive_backend",
]

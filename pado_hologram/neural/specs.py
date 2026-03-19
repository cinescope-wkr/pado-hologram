from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaptureSessionSpec:
    """Describe a capture-oriented neural holography session."""

    wavelengths: tuple[float, ...]
    device: str = "cpu"
    slm_name: str | None = None
    camera_name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if len(self.wavelengths) == 0:
            raise ValueError("wavelengths must contain at least one wavelength")
        if any(wvl <= 0 for wvl in self.wavelengths):
            raise ValueError(f"wavelengths must be positive, got {self.wavelengths}")


__all__ = ["CaptureSessionSpec"]

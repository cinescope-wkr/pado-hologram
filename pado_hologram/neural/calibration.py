from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CalibrationRecord:
    """Store lightweight calibration metadata for neural holography experiments."""

    wavelength: float
    phase_rmse: float | None = None
    amplitude_rmse: float | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.wavelength <= 0:
            raise ValueError(f"wavelength must be positive, got {self.wavelength}")
        if self.phase_rmse is not None and self.phase_rmse < 0:
            raise ValueError(f"phase_rmse must be non-negative, got {self.phase_rmse}")
        if self.amplitude_rmse is not None and self.amplitude_rmse < 0:
            raise ValueError(f"amplitude_rmse must be non-negative, got {self.amplitude_rmse}")


__all__ = ["CalibrationRecord"]

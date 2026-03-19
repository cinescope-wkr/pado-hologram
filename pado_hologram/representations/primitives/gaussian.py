from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaussianPrimitive2D:
    """Parameterize a 2D anisotropic Gaussian primitive on an image plane."""

    center_yx: tuple[float, float]
    sigma_yx: tuple[float, float]
    amplitude: float = 1.0
    phase: float = 0.0
    rotation: float = 0.0

    def __post_init__(self) -> None:
        if len(self.center_yx) != 2:
            raise ValueError(f"center_yx must have length 2, got {self.center_yx}")
        if len(self.sigma_yx) != 2:
            raise ValueError(f"sigma_yx must have length 2, got {self.sigma_yx}")
        if self.sigma_yx[0] <= 0 or self.sigma_yx[1] <= 0:
            raise ValueError(f"sigma values must be positive, got {self.sigma_yx}")
        if self.amplitude < 0:
            raise ValueError(f"amplitude must be non-negative, got {self.amplitude}")

    def as_parameter_row(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.center_yx[0],
                self.center_yx[1],
                self.sigma_yx[0],
                self.sigma_yx[1],
                self.amplitude,
                self.phase,
                self.rotation,
            ],
            dtype=torch.float32,
        )


__all__ = ["GaussianPrimitive2D"]

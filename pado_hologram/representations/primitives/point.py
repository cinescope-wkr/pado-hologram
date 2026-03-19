from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PointPrimitive2D:
    """Parameterize a phase-carrying point primitive on an image plane."""

    yx: tuple[float, float]
    amplitude: float = 1.0
    phase: float = 0.0

    def __post_init__(self) -> None:
        if len(self.yx) != 2:
            raise ValueError(f"yx must have length 2, got {self.yx}")
        if self.amplitude < 0:
            raise ValueError(f"amplitude must be non-negative, got {self.amplitude}")

    def as_parameter_row(self) -> torch.Tensor:
        return torch.tensor(
            [self.yx[0], self.yx[1], self.amplitude, self.phase],
            dtype=torch.float32,
        )


__all__ = ["PointPrimitive2D"]

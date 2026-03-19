from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaussianPrimitive3D:
    """Parameterize a 3D Gaussian disk before hologram-space projection."""

    mean_xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    scale_xyz: tuple[float, float, float]
    opacity: float = 1.0
    amplitude: float = 1.0
    phase: float = 0.0

    def __post_init__(self) -> None:
        if len(self.mean_xyz) != 3:
            raise ValueError(f"mean_xyz must have length 3, got {self.mean_xyz}")
        if len(self.quat_wxyz) != 4:
            raise ValueError(f"quat_wxyz must have length 4, got {self.quat_wxyz}")
        if len(self.scale_xyz) != 3:
            raise ValueError(f"scale_xyz must have length 3, got {self.scale_xyz}")
        if self.scale_xyz[0] <= 0 or self.scale_xyz[1] <= 0:
            raise ValueError(f"in-plane scales must be positive, got {self.scale_xyz}")
        if self.mean_xyz[2] <= 0:
            raise ValueError(f"mean_xyz z must be positive, got {self.mean_xyz}")
        if self.opacity < 0 or self.opacity > 1:
            raise ValueError(f"opacity must be in [0, 1], got {self.opacity}")
        if self.amplitude < 0:
            raise ValueError(f"amplitude must be non-negative, got {self.amplitude}")
        quat_norm = sum(float(v) ** 2 for v in self.quat_wxyz) ** 0.5
        if quat_norm <= 0:
            raise ValueError(f"quat_wxyz must have non-zero norm, got {self.quat_wxyz}")

    def as_parameter_row(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.mean_xyz[0],
                self.mean_xyz[1],
                self.mean_xyz[2],
                self.quat_wxyz[0],
                self.quat_wxyz[1],
                self.quat_wxyz[2],
                self.quat_wxyz[3],
                self.scale_xyz[0],
                self.scale_xyz[1],
                self.scale_xyz[2],
                self.opacity,
                self.amplitude,
                self.phase,
            ],
            dtype=torch.float32,
        )


__all__ = ["GaussianPrimitive3D"]

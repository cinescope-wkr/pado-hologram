from __future__ import annotations

from dataclasses import dataclass

import torch

from .gaussian import GaussianPrimitive2D
from .gaussian3d import GaussianPrimitive3D
from .point import PointPrimitive2D
from .wave_gaussian import GaussianWavePrimitive2D


@dataclass(frozen=True)
class PrimitiveScene2D:
    """A lightweight container for primitive-based holography scenes."""

    gaussians: tuple[GaussianPrimitive2D, ...] = ()
    gaussians_3d: tuple[GaussianPrimitive3D, ...] = ()
    wave_gaussians: tuple[GaussianWavePrimitive2D, ...] = ()
    points: tuple[PointPrimitive2D, ...] = ()
    name: str | None = None
    projection_focal_px: tuple[float, float] | None = None
    projection_principal_px: tuple[float, float] | None = None
    projection_K_px: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None
    projection_view_matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] | None = None
    phase_matching: bool = True

    @property
    def num_primitives(self) -> int:
        return len(self.gaussians) + len(self.gaussians_3d) + len(self.wave_gaussians) + len(self.points)

    def is_empty(self) -> bool:
        return self.num_primitives == 0

    def gaussian_parameters(self) -> torch.Tensor:
        if not self.gaussians:
            return torch.empty((0, 7), dtype=torch.float32)
        return torch.stack([primitive.as_parameter_row() for primitive in self.gaussians], dim=0)

    def gaussian3d_parameters(self) -> torch.Tensor:
        if not self.gaussians_3d:
            return torch.empty((0, 13), dtype=torch.float32)
        return torch.stack([primitive.as_parameter_row() for primitive in self.gaussians_3d], dim=0)

    def wave_gaussian_parameters(self) -> torch.Tensor:
        if not self.wave_gaussians:
            return torch.empty((0, 9), dtype=torch.float32)
        return torch.stack([primitive.as_parameter_row() for primitive in self.wave_gaussians], dim=0)

    def point_parameters(self) -> torch.Tensor:
        if not self.points:
            return torch.empty((0, 4), dtype=torch.float32)
        return torch.stack([primitive.as_parameter_row() for primitive in self.points], dim=0)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        coords: list[tuple[float, float]] = [primitive.center_yx for primitive in self.gaussians]
        coords.extend((primitive.mean_xyz[1], primitive.mean_xyz[0]) for primitive in self.gaussians_3d)
        coords.extend(primitive.center_yx for primitive in self.wave_gaussians)
        coords.extend(primitive.yx for primitive in self.points)
        if not coords:
            return None
        ys = [coord[0] for coord in coords]
        xs = [coord[1] for coord in coords]
        return (min(ys), min(xs)), (max(ys), max(xs))

    def depth_bounds(self) -> tuple[float, float] | None:
        depths = [primitive.depth for primitive in self.wave_gaussians]
        depths.extend(primitive.mean_xyz[2] for primitive in self.gaussians_3d)
        if not depths:
            return None
        return min(depths), max(depths)

    def ordered_wave_gaussians(self, order: str = "front2back") -> tuple[GaussianWavePrimitive2D, ...]:
        if order == "front2back":
            key = lambda primitive: float(primitive.depth)
        elif order == "back2front":
            key = lambda primitive: -float(primitive.depth)
        elif order == "opacity":
            key = lambda primitive: -float(primitive.opacity)
        else:
            raise ValueError(f"unsupported wave primitive ordering {order!r}")
        return tuple(sorted(self.wave_gaussians, key=key))

    def ordered_gaussians_3d(self, order: str = "front2back") -> tuple[GaussianPrimitive3D, ...]:
        if order == "front2back":
            key = lambda primitive: float(primitive.mean_xyz[2])
        elif order == "back2front":
            key = lambda primitive: -float(primitive.mean_xyz[2])
        elif order == "opacity":
            key = lambda primitive: -float(primitive.opacity)
        else:
            raise ValueError(f"unsupported 3D primitive ordering {order!r}")
        return tuple(sorted(self.gaussians_3d, key=key))


__all__ = ["PrimitiveScene2D"]

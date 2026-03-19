from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .._tensor import coerce_4d_real
from ..core.targets import normalize_mean_intensity


@dataclass(frozen=True)
class CameraObservationSpec:
    """Describe an optional observation transform applied after propagation."""

    enabled: bool = True
    name: str | None = None
    downsample: int = 1
    crop_shape: tuple[int, int] | None = None
    exposure: float = 1.0
    normalize_mean: bool = False

    def __post_init__(self) -> None:
        if self.downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {self.downsample}")
        if self.crop_shape is not None:
            if len(self.crop_shape) != 2:
                raise ValueError(f"crop_shape must have length 2, got {self.crop_shape}")
            if any(int(v) <= 0 for v in self.crop_shape):
                raise ValueError(f"crop_shape entries must be positive, got {self.crop_shape}")
        if self.exposure <= 0:
            raise ValueError(f"exposure must be positive, got {self.exposure}")

    def observe_intensity(self, intensity: torch.Tensor) -> torch.Tensor:
        observed = coerce_4d_real(intensity, name="intensity").float()

        if self.crop_shape is not None:
            observed = self._center_crop(observed, self.crop_shape)

        if self.downsample > 1:
            rows, cols = observed.shape[-2:]
            target_rows = max(1, rows // self.downsample)
            target_cols = max(1, cols // self.downsample)
            observed = F.interpolate(
                observed,
                size=(target_rows, target_cols),
                mode="area",
            )

        observed = observed * float(self.exposure)
        observed = observed.clamp_min(0.0)

        if self.normalize_mean:
            observed = normalize_mean_intensity(observed)

        return observed

    @staticmethod
    def _center_crop(intensity: torch.Tensor, crop_shape: tuple[int, int]) -> torch.Tensor:
        rows, cols = intensity.shape[-2:]
        crop_rows, crop_cols = (int(crop_shape[0]), int(crop_shape[1]))
        if crop_rows > rows or crop_cols > cols:
            raise ValueError(
                f"crop_shape {crop_shape} must fit inside intensity shape {(rows, cols)}"
            )
        row0 = (rows - crop_rows) // 2
        col0 = (cols - crop_cols) // 2
        return intensity[..., row0 : row0 + crop_rows, col0 : col0 + crop_cols]


__all__ = ["CameraObservationSpec"]

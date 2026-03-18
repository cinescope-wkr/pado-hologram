from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch

from ._tensor import coerce_4d_real


def normalize_mean_intensity(intensity: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if intensity.is_complex():
        raise TypeError("intensity must be real-valued")
    denom = intensity.mean(dim=(-1, -2), keepdim=True).clamp_min(eps)
    return intensity / denom


@dataclass(frozen=True)
class IntensityTarget:
    """Represent an intensity target for hologram reconstruction."""

    intensity: torch.Tensor
    normalize_mean: bool = True
    eps: float = 1e-8

    def __post_init__(self) -> None:
        intensity = coerce_4d_real(self.intensity, name="intensity")
        if torch.any(intensity < 0):
            raise ValueError("intensity target must be non-negative")
        object.__setattr__(self, "intensity", intensity)

    @property
    def dim(self) -> tuple[int, int, int, int]:
        return tuple(self.intensity.shape)

    def tensor(self, *, normalize: Optional[bool] = None) -> torch.Tensor:
        intensity = self.intensity
        do_normalize = self.normalize_mean if normalize is None else normalize
        if do_normalize:
            intensity = normalize_mean_intensity(intensity, eps=self.eps)
        return intensity

    def amplitude(self, *, normalize: Optional[bool] = None) -> torch.Tensor:
        return torch.sqrt(self.tensor(normalize=normalize).clamp_min(0.0))

    @classmethod
    def from_amplitude(
        cls,
        amplitude: torch.Tensor,
        *,
        normalize_mean: bool = True,
        eps: float = 1e-8,
    ) -> "IntensityTarget":
        amplitude = coerce_4d_real(amplitude, name="amplitude")
        if torch.any(amplitude < 0):
            raise ValueError("amplitude target must be non-negative")
        return cls(intensity=amplitude.square(), normalize_mean=normalize_mean, eps=eps)


@dataclass(frozen=True)
class MultiPlaneIntensityTarget:
    """Represent a sequence of plane-specific intensity targets."""

    targets: Tuple[IntensityTarget, ...]
    names: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if len(self.targets) == 0:
            raise ValueError("targets must contain at least one IntensityTarget")
        if self.names is not None and len(self.names) != len(self.targets):
            raise ValueError("names must match the number of targets")

        dims = {target.dim for target in self.targets}
        if len(dims) != 1:
            raise ValueError(f"all targets must share the same dim, got {sorted(dims)}")

    @property
    def dim(self) -> tuple[int, int, int, int]:
        return self.targets[0].dim

    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self) -> Iterable[IntensityTarget]:
        return iter(self.targets)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .._tensor import coerce_4d_real


@dataclass(frozen=True)
class NeuralHolographyBatch:
    """Bundle target, measurement, and optional phase labels for learned CGH."""

    target_intensity: torch.Tensor
    measured_intensity: torch.Tensor | None = None
    phase_target: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target = coerce_4d_real(self.target_intensity, name="target_intensity")
        object.__setattr__(self, "target_intensity", target)

        measured = self.measured_intensity
        if measured is not None:
            measured = coerce_4d_real(measured, name="measured_intensity", dim=tuple(target.shape))
            object.__setattr__(self, "measured_intensity", measured)

        phase = self.phase_target
        if phase is not None:
            phase = coerce_4d_real(phase, name="phase_target", dim=tuple(target.shape))
            object.__setattr__(self, "phase_target", phase)

        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def dim(self) -> tuple[int, int, int, int]:
        return tuple(self.target_intensity.shape)

    @property
    def reference_intensity(self) -> torch.Tensor:
        if self.measured_intensity is not None:
            return self.measured_intensity
        return self.target_intensity


__all__ = ["NeuralHolographyBatch"]

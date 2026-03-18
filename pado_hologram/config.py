from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from pado.light import Light
from pado.propagator import Propagator

from ._tensor import coerce_4d_real


@dataclass(frozen=True)
class SourceSpec:
    """Describe the SLM/source plane field used by a holography experiment."""

    dim: Tuple[int, int, int, int]
    pitch: float
    wvl: float
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.dim, tuple) or len(self.dim) != 4:
            raise ValueError(f"dim must be a 4-element tuple, got {self.dim}")
        if any(d <= 0 for d in self.dim):
            raise ValueError(f"all dimensions must be positive, got {self.dim}")
        if self.dim[1] != 1:
            raise ValueError(
                f"SourceSpec currently expects a single-channel SLM/source plane, got dim={self.dim}"
            )
        if self.pitch <= 0:
            raise ValueError(f"pitch must be positive, got {self.pitch}")
        if self.wvl <= 0:
            raise ValueError(f"wvl must be positive, got {self.wvl}")

    def make_light(
        self,
        *,
        amplitude: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
    ) -> Light:
        light = Light(self.dim, self.pitch, self.wvl, device=self.device)
        if amplitude is not None:
            amplitude = coerce_4d_real(amplitude, name="amplitude", dim=self.dim).to(self.device)
            light.set_amplitude(amplitude)
        if phase is not None:
            phase = coerce_4d_real(phase, name="phase", dim=self.dim).to(self.device)
            light.set_phase(phase)
        return light


@dataclass(frozen=True)
class PropagationSpec:
    """Describe how to propagate a source or SLM-plane light field."""

    distance: float
    mode: str = "ASM"
    polar: str = "non"
    offset: Tuple[float, float] = (0.0, 0.0)
    linear: bool = True
    band_limit: bool = True
    b: float = 1.0
    target_plane: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    sampling_ratio: int = 1
    vectorized: bool = False
    steps: int = 100

    def __post_init__(self) -> None:
        supported_modes = {"Fraunhofer", "Fresnel", "FFT", "ASM", "RS"}
        if self.mode not in supported_modes:
            raise ValueError(f"mode must be one of {sorted(supported_modes)}, got {self.mode}")
        if self.polar not in {"non", "polar"}:
            raise ValueError(f"polar must be 'non' or 'polar', got {self.polar}")
        if len(self.offset) != 2:
            raise ValueError(f"offset must have length 2, got {self.offset}")
        if self.sampling_ratio < 1:
            raise ValueError(f"sampling_ratio must be >= 1, got {self.sampling_ratio}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")

    def build(self) -> Propagator:
        return Propagator(mode=self.mode, polar=self.polar)

    def forward(self, light: Light, *, distance: Optional[float] = None) -> Light:
        z = self.distance if distance is None else distance
        return self.build().forward(
            light,
            z=z,
            offset=self.offset,
            linear=self.linear,
            band_limit=self.band_limit,
            b=self.b,
            target_plane=self.target_plane,
            sampling_ratio=self.sampling_ratio,
            vectorized=self.vectorized,
            steps=self.steps,
        )

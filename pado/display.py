from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .light import Light
from .math import wrap_phase


def _as_4d_phase(phase: torch.Tensor, dim: Tuple[int, int, int, int]) -> torch.Tensor:
    if phase.ndim == 2:
        phase = phase.unsqueeze(0).unsqueeze(0)
    elif phase.ndim == 3:
        # Assume (B,R,C) -> (B,1,R,C)
        phase = phase.unsqueeze(1)
    elif phase.ndim != 4:
        raise ValueError(f"phase must have shape (R,C), (B,R,C), or (B,1,R,C); got {tuple(phase.shape)}")

    if phase.shape[0] != dim[0] or phase.shape[1] != 1 or phase.shape[2] != dim[2] or phase.shape[3] != dim[3]:
        raise ValueError(f"phase must have shape (B,1,R,C)={dim}; got {tuple(phase.shape)}")
    return phase


def _ste_quantize_unit_interval(x: torch.Tensor, levels: int) -> torch.Tensor:
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")
    x = torch.clamp(x, 0.0, 1.0)
    q = torch.round(x * (levels - 1)) / (levels - 1)
    # Straight-through estimator: forward uses q, backward treats as identity.
    return x + (q - x).detach()


@dataclass(frozen=True)
class LCOSLUT:
    """A minimal LUT model for LCOS phase-only SLM response.

    The LUT is sampled uniformly over normalized gray values g in [0, 1].
    `phase_lut[i]` gives the realized phase in radians at g = i/(L-1).
    """

    phase_lut: torch.Tensor  # [L], radians
    amplitude_lut: Optional[torch.Tensor] = None  # [L], optional
    wvl_ref: Optional[float] = None  # meters, optional reference wavelength for phase scaling

    def __post_init__(self) -> None:
        if not isinstance(self.phase_lut, torch.Tensor):
            raise TypeError(f"phase_lut must be a torch.Tensor, got {type(self.phase_lut)}")
        if self.phase_lut.ndim != 1:
            raise ValueError(f"phase_lut must be 1D [L], got shape {tuple(self.phase_lut.shape)}")
        if self.phase_lut.numel() < 2:
            raise ValueError("phase_lut must have at least 2 samples")
        if self.amplitude_lut is not None:
            if not isinstance(self.amplitude_lut, torch.Tensor):
                raise TypeError(f"amplitude_lut must be a torch.Tensor, got {type(self.amplitude_lut)}")
            if self.amplitude_lut.shape != self.phase_lut.shape:
                raise ValueError(
                    f"amplitude_lut must match phase_lut shape {tuple(self.phase_lut.shape)}, got {tuple(self.amplitude_lut.shape)}"
                )

    @property
    def L(self) -> int:
        return int(self.phase_lut.numel())

    def is_monotonic(self) -> bool:
        d = torch.diff(self.phase_lut)
        return bool(torch.all(d >= 0) or torch.all(d <= 0))

    def uses_positive_phase_range(self, *, wvl: Optional[float] = None) -> bool:
        """Return True when the LUT is expressed on a non-negative phase domain.

        This covers the common LCOS convention where measured phase response is
        represented over ``[0, 2π]`` (or a truncated non-negative subset).
        """
        lut = self._phase_lut_for_wvl(wvl)
        return bool(torch.min(lut) >= 0)

    def wrap_phase_for_lut(self, phase: torch.Tensor, *, wvl: Optional[float] = None) -> torch.Tensor:
        """Wrap target phase into the phase convention used by this LUT."""
        return wrap_phase(phase, stay_positive=self.uses_positive_phase_range(wvl=wvl))

    def _phase_lut_for_wvl(self, wvl: Optional[float]) -> torch.Tensor:
        if wvl is None or self.wvl_ref is None:
            return self.phase_lut
        # Approximate wavelength scaling: phase ∝ 1/λ for a fixed physical retardation.
        scale = float(self.wvl_ref / wvl)
        return self.phase_lut * scale

    def gray_to_phase(self, gray: torch.Tensor, *, wvl: Optional[float] = None) -> torch.Tensor:
        """Linear interpolation from gray in [0,1] to phase in radians."""
        lut = self._phase_lut_for_wvl(wvl).to(gray.device)
        g = torch.clamp(gray, 0.0, 1.0)
        idx_f = g * (self.L - 1)
        idx0 = torch.floor(idx_f).to(torch.long)
        idx1 = torch.clamp(idx0 + 1, max=self.L - 1)
        w = (idx_f - idx0.to(idx_f.dtype))
        p0 = lut[idx0]
        p1 = lut[idx1]
        return p0 * (1.0 - w) + p1 * w

    def phase_to_gray(self, phase: torch.Tensor, *, wvl: Optional[float] = None) -> torch.Tensor:
        """Invert phase->gray assuming a monotonic phase_lut.

        If you have a non-monotonic measured LUT, provide a preprocessed monotonic LUT
        or implement a custom inverse outside of this minimal API.
        """
        lut = self._phase_lut_for_wvl(wvl).to(phase.device)
        if not self.is_monotonic():
            raise ValueError("phase_lut is not monotonic; cannot invert phase->gray reliably")

        increasing = bool(torch.all(torch.diff(lut) >= 0))
        if not increasing:
            lut = torch.flip(lut, dims=(0,))

        # Search for insertion indices so lut[i0] <= phase < lut[i1]
        p = phase
        idx1 = torch.searchsorted(lut, p)
        idx1 = torch.clamp(idx1, 0, self.L - 1)
        idx0 = torch.clamp(idx1 - 1, 0, self.L - 1)

        p0 = lut[idx0]
        p1 = lut[idx1]
        denom = (p1 - p0)
        # Avoid divide-by-zero on flat regions.
        w = torch.where(denom.abs() > 0, (p - p0) / denom, torch.zeros_like(p))
        g = (idx0.to(p.dtype) + w) / (self.L - 1)

        if not increasing:
            g = 1.0 - g
        return torch.clamp(g, 0.0, 1.0)


def lcos_encode_phase(
    phase_target: torch.Tensor,
    lut: LCOSLUT,
    *,
    wvl: Optional[float] = None,
    bits: Optional[int] = 8,
    ste: bool = True,
    wrap: bool = True,
) -> Dict[str, torch.Tensor]:
    """Encode a desired phase into an LCOS-like realized phase via LUT + quantization.

    Returns a dict with:
    - gray: normalized gray in [0,1]
    - phase_realized: realized phase in radians
    - amplitude_realized: ones (or amplitude LUT if provided)

    When ``wrap=True``, the target phase is wrapped into the phase convention
    used by ``lut``. For example, LUTs expressed over ``[0, 2π]`` will receive
    positive wrapped phase, while LUTs expressed over ``[-π, π]`` will receive
    bipolar wrapped phase.
    """
    if wrap:
        phase_target = lut.wrap_phase_for_lut(phase_target, wvl=wvl)

    gray = lut.phase_to_gray(phase_target, wvl=wvl)
    if bits is not None:
        levels = 2 ** int(bits)
        if ste:
            gray = _ste_quantize_unit_interval(gray, levels)
        else:
            gray = torch.round(torch.clamp(gray, 0.0, 1.0) * (levels - 1)) / (levels - 1)

    phase_realized = lut.gray_to_phase(gray, wvl=wvl)

    if lut.amplitude_lut is not None:
        amp = lut.amplitude_lut.to(gray.device)
        # reuse gray_to_phase interpolation for amplitude by temporarily swapping LUT
        tmp = LCOSLUT(phase_lut=amp, amplitude_lut=None, wvl_ref=None)
        amplitude_realized = tmp.gray_to_phase(gray)
    else:
        amplitude_realized = torch.ones_like(phase_realized)

    return {
        "gray": gray,
        "phase_realized": phase_realized,
        "amplitude_realized": amplitude_realized,
    }


def phase_only_field(phase_realized: torch.Tensor, amplitude: Optional[torch.Tensor] = None) -> torch.Tensor:
    if amplitude is None:
        amplitude = torch.ones_like(phase_realized)
    return amplitude * torch.exp(1j * phase_realized)


def slm_light_from_phase(
    *,
    dim: Tuple[int, int, int, int],
    pitch: float,
    wvl: float,
    phase_target: torch.Tensor,
    lut: LCOSLUT,
    device: str = "cpu",
    bits: Optional[int] = 8,
    ste: bool = True,
    wrap: bool = True,
) -> Light:
    """Create a Light at the SLM plane from an optimized phase (phase-first workflow)."""
    phase_target = phase_target.to(device)
    phase_target = _as_4d_phase(phase_target, dim)

    enc = lcos_encode_phase(phase_target, lut, wvl=wvl, bits=bits, ste=ste, wrap=wrap)
    field_change = phase_only_field(enc["phase_realized"], amplitude=enc["amplitude_realized"])

    light = Light(dim, pitch, wvl, device=device)
    light.set_field(field_change.to(torch.cfloat))
    return light

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from pado.display import LCOSLUT, lcos_encode_phase, phase_only_field, slm_light_from_phase
from pado.light import Light

from .._tensor import coerce_4d_real, validate_light_against_source
from ..core.specs import SourceSpec


@dataclass(frozen=True)
class PhaseEncodingConfig:
    """Control LCOS/SLM phase encoding behavior."""

    bits: Optional[int] = 8
    ste: bool = True
    wrap: bool = True

    def __post_init__(self) -> None:
        if self.bits is not None and self.bits < 1:
            raise ValueError(f"bits must be >= 1 or None, got {self.bits}")


@dataclass(frozen=True)
class PhaseEncodingResult:
    """Store the realized display-domain representation of a target phase."""

    gray: torch.Tensor
    phase_realized: torch.Tensor
    amplitude_realized: torch.Tensor
    field: torch.Tensor


class PhaseOnlyLCOSSLM:
    """Phase-only LCOS/SLM bridge for PADO Hologram workflows."""

    def __init__(
        self,
        source: SourceSpec,
        lut: LCOSLUT,
        encoding: Optional[PhaseEncodingConfig] = None,
    ) -> None:
        self.source = source
        self.lut = lut
        self.encoding = encoding or PhaseEncodingConfig()

    def encode_phase(self, phase_target: torch.Tensor) -> PhaseEncodingResult:
        phase_target = coerce_4d_real(
            phase_target,
            name="phase_target",
            dim=self.source.dim,
        ).to(self.source.device)

        encoded = lcos_encode_phase(
            phase_target,
            self.lut,
            wvl=self.source.wvl,
            bits=self.encoding.bits,
            ste=self.encoding.ste,
            wrap=self.encoding.wrap,
        )
        field = phase_only_field(
            encoded["phase_realized"],
            amplitude=encoded["amplitude_realized"],
        ).to(torch.cfloat)
        return PhaseEncodingResult(
            gray=encoded["gray"],
            phase_realized=encoded["phase_realized"],
            amplitude_realized=encoded["amplitude_realized"],
            field=field,
        )

    def light_from_phase(self, phase_target: torch.Tensor) -> Light:
        phase_target = coerce_4d_real(
            phase_target,
            name="phase_target",
            dim=self.source.dim,
        ).to(self.source.device)
        return slm_light_from_phase(
            dim=self.source.dim,
            pitch=self.source.pitch,
            wvl=self.source.wvl,
            phase_target=phase_target,
            lut=self.lut,
            device=self.source.device,
            bits=self.encoding.bits,
            ste=self.encoding.ste,
            wrap=self.encoding.wrap,
        )

    def apply_encoding(self, source_light: Light, encoding: PhaseEncodingResult) -> Light:
        validate_light_against_source(
            source_light,
            self.source.dim,
            self.source.pitch,
            self.source.wvl,
        )
        modulated = source_light.clone()
        field = encoding.field.to(modulated.field.device, dtype=modulated.field.dtype)
        modulated.set_field(modulated.field * field)
        return modulated

    def modulate(self, source_light: Light, phase_target: torch.Tensor) -> Light:
        return self.apply_encoding(source_light, self.encode_phase(phase_target))


__all__ = [
    "PhaseEncodingConfig",
    "PhaseEncodingResult",
    "PhaseOnlyLCOSSLM",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

from pado.light import Light

from ._tensor import coerce_4d_real, validate_light_against_source
from .config import PropagationSpec, SourceSpec
from .losses import multi_plane_reconstruction_metrics, reconstruction_metrics
from .slm import PhaseEncodingResult, PhaseOnlyLCOSSLM
from .targets import IntensityTarget, MultiPlaneIntensityTarget


@dataclass(frozen=True)
class HologramForwardResult:
    source_light: Light
    slm_light: Light
    propagated_light: Light
    metrics: Optional[Dict[str, torch.Tensor]] = None
    encoding: Optional[PhaseEncodingResult] = None


@dataclass(frozen=True)
class MultiPlaneHologramForwardResult:
    source_light: Light
    slm_light: Light
    propagated_lights: Tuple[Light, ...]
    metrics: Optional[Dict[str, torch.Tensor | tuple[Dict[str, torch.Tensor], ...]]] = None
    encoding: Optional[PhaseEncodingResult] = None


class HologramPipeline:
    """Compose source, SLM/device model, propagation, and evaluation."""

    def __init__(
        self,
        source: SourceSpec,
        propagation: PropagationSpec,
        *,
        slm: Optional[PhaseOnlyLCOSSLM] = None,
    ) -> None:
        self.source = source
        self.propagation = propagation
        self.slm = slm

    def make_source_light(
        self,
        *,
        amplitude: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
    ) -> Light:
        return self.source.make_light(amplitude=amplitude, phase=phase)

    def forward_source(
        self,
        source_light: Light,
        *,
        target: Optional[IntensityTarget] = None,
        normalize_metrics: bool = True,
    ) -> HologramForwardResult:
        validate_light_against_source(
            source_light,
            self.source.dim,
            self.source.pitch,
            self.source.wvl,
        )
        source_copy = source_light.clone()
        propagated = self.propagation.forward(source_light.clone())
        metrics = None
        if target is not None:
            metrics = reconstruction_metrics(propagated, target, normalize=normalize_metrics)
        return HologramForwardResult(
            source_light=source_copy,
            slm_light=source_copy.clone(),
            propagated_light=propagated,
            metrics=metrics,
            encoding=None,
        )

    def forward_phase(
        self,
        phase_target: torch.Tensor,
        *,
        source_light: Optional[Light] = None,
        target: Optional[IntensityTarget] = None,
        normalize_metrics: bool = True,
    ) -> HologramForwardResult:
        phase_target = coerce_4d_real(
            phase_target,
            name="phase_target",
            dim=self.source.dim,
        ).to(self.source.device)
        base_source = source_light.clone() if source_light is not None else self.make_source_light()
        validate_light_against_source(
            base_source,
            self.source.dim,
            self.source.pitch,
            self.source.wvl,
        )

        encoding = None
        if self.slm is None:
            slm_light = base_source.clone()
            slm_light.set_phase(phase_target.to(slm_light.field.device))
        else:
            encoding = self.slm.encode_phase(phase_target)
            slm_light = self.slm.apply_encoding(base_source, encoding)

        propagated = self.propagation.forward(slm_light.clone())
        metrics = None
        if target is not None:
            metrics = reconstruction_metrics(propagated, target, normalize=normalize_metrics)

        return HologramForwardResult(
            source_light=base_source,
            slm_light=slm_light,
            propagated_light=propagated,
            metrics=metrics,
            encoding=encoding,
        )


class MultiPlaneHologramPipeline:
    """Compose one source/SLM plane with multiple observation planes."""

    def __init__(
        self,
        source: SourceSpec,
        propagations: Sequence[PropagationSpec],
        *,
        slm: Optional[PhaseOnlyLCOSSLM] = None,
    ) -> None:
        if len(propagations) == 0:
            raise ValueError("propagations must contain at least one PropagationSpec")
        self.source = source
        self.propagations = tuple(propagations)
        self.slm = slm

    def _build_slm_light(
        self,
        phase_target: torch.Tensor,
        source_light: Optional[Light],
    ) -> tuple[Light, Light, Optional[PhaseEncodingResult]]:
        phase_target = coerce_4d_real(
            phase_target,
            name="phase_target",
            dim=self.source.dim,
        ).to(self.source.device)
        base_source = source_light.clone() if source_light is not None else self.source.make_light()
        validate_light_against_source(
            base_source,
            self.source.dim,
            self.source.pitch,
            self.source.wvl,
        )

        encoding = None
        if self.slm is None:
            slm_light = base_source.clone()
            slm_light.set_phase(phase_target.to(slm_light.field.device))
        else:
            encoding = self.slm.encode_phase(phase_target)
            slm_light = self.slm.apply_encoding(base_source, encoding)

        return base_source, slm_light, encoding

    def forward_phase(
        self,
        phase_target: torch.Tensor,
        *,
        source_light: Optional[Light] = None,
        target: Optional[MultiPlaneIntensityTarget] = None,
        normalize_metrics: bool = True,
    ) -> MultiPlaneHologramForwardResult:
        if target is not None and len(target) != len(self.propagations):
            raise ValueError(
                f"target must match the number of propagations {len(self.propagations)}, got {len(target)}"
            )

        base_source, slm_light, encoding = self._build_slm_light(phase_target, source_light)
        propagated_lights = tuple(
            propagation.forward(slm_light.clone())
            for propagation in self.propagations
        )

        metrics = None
        if target is not None:
            metrics = multi_plane_reconstruction_metrics(
                propagated_lights,
                target,
                normalize=normalize_metrics,
            )

        return MultiPlaneHologramForwardResult(
            source_light=base_source,
            slm_light=slm_light,
            propagated_lights=propagated_lights,
            metrics=metrics,
            encoding=encoding,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from pado.light import Light
from pado.math import wrap_phase

from ._tensor import coerce_4d_complex, coerce_4d_real
from .backends import checkerboard_phase_select
from .config import PropagationSpec, SourceSpec
from .losses import intensity_mse
from .targets import IntensityTarget


@dataclass(frozen=True)
class GerchbergSaxtonResult:
    phase: torch.Tensor
    history: Tuple[float, ...]
    slm_light: Light
    propagated_light: Light


@dataclass(frozen=True)
class DPACResult:
    phase_a: torch.Tensor
    phase_b: torch.Tensor
    checkerboard_phase: torch.Tensor
    checkerboard_mask: torch.Tensor
    reconstructed_field: torch.Tensor
    normalized_amplitude: torch.Tensor
    kernel_backend: str
    backend_reason: str


class DoublePhaseAmplitudeCoder:
    """Encode a complex target field with double-phase amplitude coding."""

    def __init__(
        self,
        source: SourceSpec,
        *,
        stay_positive: bool = False,
        backend: str = "auto",
        warp_cache_dir: Optional[str] = None,
    ) -> None:
        self.source = source
        self.stay_positive = stay_positive
        self.backend = backend
        self.warp_cache_dir = warp_cache_dir

    def encode_field(
        self,
        target_field: torch.Tensor,
        *,
        normalize_amplitude: bool = True,
    ) -> DPACResult:
        target_field = coerce_4d_complex(
            target_field,
            name="target_field",
            dim=self.source.dim,
        ).to(self.source.device)

        amplitude = target_field.abs()
        max_amplitude = amplitude.max()
        if normalize_amplitude:
            scale = max(max_amplitude.item(), 1.0)
            amplitude = amplitude / scale
        elif torch.any(amplitude > 1.0):
            raise ValueError("target_field amplitude must be <= 1 when normalize_amplitude=False")

        phase = torch.angle(target_field)
        alpha = torch.arccos(amplitude.clamp(0.0, 1.0))
        phase_a = wrap_phase(phase + alpha, stay_positive=self.stay_positive)
        phase_b = wrap_phase(phase - alpha, stay_positive=self.stay_positive)
        checkerboard_phase, mask, backend_selection = checkerboard_phase_select(
            phase_a,
            phase_b,
            backend=self.backend,
            device=self.source.device,
            warp_cache_dir=self.warp_cache_dir,
        )
        reconstructed_field = 0.5 * (torch.exp(1j * phase_a) + torch.exp(1j * phase_b))

        return DPACResult(
            phase_a=phase_a,
            phase_b=phase_b,
            checkerboard_phase=checkerboard_phase,
            checkerboard_mask=mask,
            reconstructed_field=reconstructed_field.to(torch.cfloat),
            normalized_amplitude=amplitude,
            kernel_backend=backend_selection.resolved,
            backend_reason=backend_selection.reason,
        )

    def encode_target(
        self,
        target: IntensityTarget,
        *,
        phase_target: Optional[torch.Tensor] = None,
        normalize_target: bool = True,
        normalize_amplitude: bool = True,
    ) -> DPACResult:
        amplitude = target.amplitude(normalize=normalize_target).to(self.source.device)
        if phase_target is None:
            phase_target = torch.zeros(self.source.dim, device=self.source.device)
        else:
            phase_target = coerce_4d_real(
                phase_target,
                name="phase_target",
                dim=self.source.dim,
            ).to(self.source.device)
        target_field = amplitude * torch.exp(1j * phase_target)
        return self.encode_field(target_field, normalize_amplitude=normalize_amplitude)


class GerchbergSaxtonPhaseOptimizer:
    """A compact Gerchberg-Saxton optimizer for phase-only hologram generation."""

    def __init__(self, source: SourceSpec, propagation: PropagationSpec) -> None:
        self.source = source
        self.propagation = propagation

    def optimize(
        self,
        target: IntensityTarget,
        *,
        iterations: int = 20,
        initial_phase: Optional[torch.Tensor] = None,
        source_amplitude: Optional[torch.Tensor] = None,
        normalize_target: bool = True,
    ) -> GerchbergSaxtonResult:
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        if tuple(target.dim) != tuple(self.source.dim):
            raise ValueError(f"target dim must match source dim {self.source.dim}, got {target.dim}")

        if initial_phase is None:
            phase = torch.zeros(self.source.dim, device=self.source.device)
        else:
            phase = coerce_4d_real(
                initial_phase,
                name="initial_phase",
                dim=self.source.dim,
            ).to(self.source.device)

        source_light = self.source.make_light(amplitude=source_amplitude)
        target_amplitude = target.amplitude(normalize=normalize_target).to(self.source.device)
        history: List[float] = []

        for _ in range(iterations):
            slm_light = source_light.clone()
            slm_light.set_phase(phase)

            propagated = self.propagation.forward(slm_light.clone())
            loss = intensity_mse(propagated, target, normalize=normalize_target)
            history.append(float(loss.detach().cpu()))

            constrained = propagated.clone()
            constrained.set_amplitude(target_amplitude.to(constrained.field.device))
            backpropagated = self.propagation.forward(
                constrained,
                distance=-self.propagation.distance,
            )
            phase = backpropagated.get_phase().detach()

        final_slm = source_light.clone()
        final_slm.set_phase(phase)
        final_propagated = self.propagation.forward(final_slm.clone())
        final_loss = intensity_mse(final_propagated, target, normalize=normalize_target)
        history.append(float(final_loss.detach().cpu()))

        return GerchbergSaxtonResult(
            phase=phase,
            history=tuple(history),
            slm_light=final_slm,
            propagated_light=final_propagated,
        )

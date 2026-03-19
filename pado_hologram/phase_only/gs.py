from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from pado.light import Light

from .._tensor import coerce_4d_real
from ..core.losses import intensity_mse
from ..core.specs import PropagationSpec, SourceSpec
from ..core.targets import IntensityTarget


@dataclass(frozen=True)
class GerchbergSaxtonResult:
    phase: torch.Tensor
    history: Tuple[float, ...]
    slm_light: Light
    propagated_light: Light


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


__all__ = [
    "GerchbergSaxtonPhaseOptimizer",
    "GerchbergSaxtonResult",
]

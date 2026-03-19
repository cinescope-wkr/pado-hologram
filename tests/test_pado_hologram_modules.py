from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pado.display import LCOSLUT
from pado_hologram import (
    CameraObservationSpec,
    GerchbergSaxtonPhaseOptimizer,
    HologramPipeline,
    IntensityTarget,
    PhaseEncodingConfig,
    PhaseOnlyLCOSSLM,
    PropagationSpec,
    SourceSpec,
)


def test_source_spec_make_light_applies_amplitude_and_phase() -> None:
    source = SourceSpec(dim=(1, 1, 4, 4), pitch=6.4e-6, wvl=532e-9)
    amplitude = torch.full((4, 4), 2.0)
    phase = torch.full((4, 4), 0.5)

    light = source.make_light(amplitude=amplitude, phase=phase)

    assert torch.allclose(light.get_amplitude(), torch.full((1, 1, 4, 4), 2.0))
    assert torch.allclose(light.get_phase(), torch.full((1, 1, 4, 4), 0.5))


def test_phase_only_lcos_slm_modulates_source_with_realized_phase() -> None:
    source = SourceSpec(dim=(1, 1, 4, 4), pitch=6.4e-6, wvl=532e-9)
    lut = LCOSLUT(phase_lut=torch.linspace(0.0, torch.pi, 33))
    slm = PhaseOnlyLCOSSLM(source, lut, PhaseEncodingConfig(bits=None))

    source_light = source.make_light(amplitude=torch.full((4, 4), 2.0))
    phase_target = torch.full((4, 4), torch.pi / 2)
    modulated = slm.modulate(source_light, phase_target)

    assert torch.allclose(modulated.get_amplitude(), torch.full((1, 1, 4, 4), 2.0), atol=1e-5)
    assert torch.allclose(modulated.get_phase(), torch.full((1, 1, 4, 4), torch.pi / 2), atol=1e-5)


def test_hologram_pipeline_forward_phase_reports_zero_error_for_flat_target() -> None:
    source = SourceSpec(dim=(1, 1, 8, 8), pitch=6.4e-6, wvl=532e-9)
    propagation = PropagationSpec(distance=0.0, mode="ASM")
    pipeline = HologramPipeline(source, propagation)
    target = IntensityTarget(torch.ones(8, 8))

    result = pipeline.forward_phase(torch.zeros(8, 8), target=target)

    assert result.metrics is not None
    assert torch.allclose(result.propagated_light.get_intensity(), torch.ones(1, 1, 8, 8))
    assert float(result.metrics["mse"]) == 0.0


def test_gerchberg_saxton_optimizer_handles_flat_identity_case() -> None:
    source = SourceSpec(dim=(1, 1, 8, 8), pitch=6.4e-6, wvl=532e-9)
    propagation = PropagationSpec(distance=0.0, mode="ASM")
    target = IntensityTarget(torch.ones(8, 8))

    result = GerchbergSaxtonPhaseOptimizer(source, propagation).optimize(target, iterations=2)

    assert len(result.history) == 3
    assert torch.allclose(result.propagated_light.get_intensity(), torch.ones(1, 1, 8, 8))
    assert result.history[-1] == 0.0


def test_camera_observation_spec_supports_crop_downsample_and_exposure() -> None:
    camera = CameraObservationSpec(
        name="ideal_binned2",
        downsample=2,
        crop_shape=(4, 4),
        exposure=2.0,
    )
    intensity = torch.arange(36, dtype=torch.float32).view(1, 1, 6, 6)

    observed = camera.observe_intensity(intensity)

    expected = torch.tensor([[[[21.0, 25.0], [45.0, 49.0]]]])
    assert observed.shape == (1, 1, 2, 2)
    assert torch.allclose(observed, expected)

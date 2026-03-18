from pathlib import Path
import sys

import torch
from hydra import compose, initialize_config_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pado_hologram import (
    DoublePhaseAmplitudeCoder,
    IntensityTarget,
    MultiPlaneHologramPipeline,
    MultiPlaneIntensityTarget,
    PropagationSpec,
    SourceSpec,
    run_experiment,
)


def test_dpac_reconstructs_target_field_by_phase_pair_average() -> None:
    source = SourceSpec(dim=(1, 1, 4, 4), pitch=6.4e-6, wvl=532e-9)
    coder = DoublePhaseAmplitudeCoder(source)
    amplitude = torch.full((4, 4), 0.5)
    phase = torch.full((4, 4), 0.25)
    target_field = amplitude.view(1, 1, 4, 4) * torch.exp(1j * phase.view(1, 1, 4, 4))

    result = coder.encode_field(target_field)

    assert torch.allclose(result.reconstructed_field, target_field.to(torch.cfloat), atol=1e-5)


def test_multiplane_pipeline_aggregates_identity_metrics() -> None:
    source = SourceSpec(dim=(1, 1, 8, 8), pitch=6.4e-6, wvl=532e-9)
    propagations = (
        PropagationSpec(distance=0.0, mode="ASM"),
        PropagationSpec(distance=0.0, mode="ASM"),
    )
    target = MultiPlaneIntensityTarget(
        targets=(
            IntensityTarget(torch.ones(8, 8)),
            IntensityTarget(torch.ones(8, 8)),
        ),
        names=("near", "far"),
    )

    result = MultiPlaneHologramPipeline(source, propagations).forward_phase(torch.zeros(8, 8), target=target)

    assert result.metrics is not None
    assert len(result.propagated_lights) == 2
    assert float(result.metrics["mse"]) == 0.0
    assert len(result.metrics["per_plane"]) == 2


def test_hydra_config_composes_and_runs_default_experiment() -> None:
    with initialize_config_module(version_base=None, config_module="pado_hologram.conf"):
        cfg = compose(config_name="config")
    summary = run_experiment(cfg)

    assert summary.method == "gs"
    assert summary.metrics["final_mse"] == 0.0

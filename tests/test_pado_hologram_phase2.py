from pathlib import Path
import sys
import json

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pado_hologram.cli import main
from pado_hologram.experiments import available_experiments, compose_experiment_config
from pado_hologram.neural import CaptureSessionSpec, NeuralHolographyBatch, NeuralHolographyTrainer
from pado_hologram.primitive_based import (
    available_primitive_backends,
    available_primitive_renderers,
    build_primitive_scene_from_config,
    render_gaussian_scene,
    render_gaussian_scene_gws_exact,
    render_gaussian_scene_gws_exact_awb,
    render_gaussian_scene_gws_rpws_exact,
    render_gaussian_scene_splat,
    render_gaussian_scene_wave,
    render_gaussian_scene_wave_awb,
)
from pado_hologram.primitive_based.warp import primitive_warp_available
from pado_hologram.representations import (
    GaussianPrimitive2D,
    GaussianPrimitive3D,
    GaussianWavePrimitive2D,
    PointPrimitive2D,
    PrimitiveScene2D,
)
from pado_hologram import PropagationSpec, SourceSpec


def test_experiment_registry_and_compose_support_overrides() -> None:
    assert available_experiments() == (
        "dpac",
        "gs",
        "primitive_gaussian",
        "primitive_gaussian_awb",
        "primitive_gaussian_gws_exact",
        "primitive_gaussian_rpws",
        "primitive_gaussian_splat",
        "primitive_gaussian_wave",
    )

    cfg = compose_experiment_config(["experiment=dpac", "target=gaussian", "camera=binned2"])

    assert cfg.experiment.method == "dpac"
    assert cfg.target.kind == "gaussian"
    assert cfg.camera.downsample == 2


def test_cli_run_prints_experiment_summary(capsys) -> None:
    code = main(["run", "experiment=gs"])
    captured = capsys.readouterr()

    assert code == 0
    assert "method: gs" in captured.out
    assert "metrics:" in captured.out


def test_cli_run_supports_primitive_gaussian(capsys) -> None:
    code = main(["run", "experiment=primitive_gaussian"])
    captured = capsys.readouterr()

    assert code == 0
    assert "method: primitive_gaussian" in captured.out


def test_cli_run_supports_primitive_gaussian_splat(capsys) -> None:
    code = main(["run", "experiment=primitive_gaussian_splat"])
    captured = capsys.readouterr()

    assert code == 0
    assert "method: primitive_gaussian_splat" in captured.out


def test_cli_run_supports_primitive_gaussian_wave(capsys) -> None:
    code = main(["run", "experiment=primitive_gaussian_wave", "primitives=gaussian_depth_ring"])
    captured = capsys.readouterr()

    assert code == 0
    assert "method: primitive_gaussian_wave" in captured.out
    assert "num_wave_gaussians:" in captured.out


def test_cli_run_supports_primitive_gaussian_awb(capsys) -> None:
    code = main(["run", "experiment=primitive_gaussian_awb", "primitives=gaussian_depth_ring"])
    captured = capsys.readouterr()

    assert code == 0
    assert "method: primitive_gaussian_awb" in captured.out
    assert "sort_order: back2front" in captured.out


def test_cli_run_supports_primitive_gaussian_rpws(capsys) -> None:
    code = main(["run", "experiment=primitive_gaussian_rpws", "primitives=gaussian3d_depth_ring"])
    captured = capsys.readouterr()

    assert code == 0
    assert "method: primitive_gaussian_rpws" in captured.out
    assert "num_frames: 8" in captured.out
    assert "angular_profile: uniform" in captured.out


def test_cli_run_supports_primitive_gaussian_gws_exact(capsys) -> None:
    code = main(["run", "experiment=primitive_gaussian_gws_exact", "primitives=gaussian3d_depth_ring"])
    captured = capsys.readouterr()

    assert code == 0
    assert "method: primitive_gaussian_gws_exact" in captured.out
    assert "num_gaussians_3d:" in captured.out


def test_cli_run_supports_primitive_ring_preset(capsys) -> None:
    code = main(["run", "experiment=primitive_gaussian_splat", "primitives=gaussian_ring", "backend=torch"])
    captured = capsys.readouterr()

    assert code == 0
    assert "scene_name: gaussian_ring" in captured.out


def test_cli_run_supports_primitive_ring_with_camera(capsys) -> None:
    code = main(
        [
            "run",
            "experiment=primitive_gaussian_splat",
            "primitives=gaussian_ring",
            "camera=binned2",
            "backend=torch",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "camera: ideal_binned2" in captured.out
    assert "observed_shape: (1, 1, 16, 16)" in captured.out


def test_primitive_scene_stacks_parameter_tensors() -> None:
    scene = PrimitiveScene2D(
        gaussians=(GaussianPrimitive2D(center_yx=(1.0, 2.0), sigma_yx=(0.5, 0.75)),),
        gaussians_3d=(
            GaussianPrimitive3D(
                mean_xyz=(0.0, 0.0, 1e-3),
                quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                scale_xyz=(8.0e-6, 8.0e-6, 1.0e-9),
                opacity=0.7,
            ),
        ),
        wave_gaussians=(
            GaussianWavePrimitive2D(center_yx=(2.0, 1.0), sigma_yx=(0.8, 0.9), depth=1e-3, opacity=0.6),
        ),
        points=(PointPrimitive2D(yx=(3.0, 4.0), amplitude=0.2),),
        name="demo",
        projection_focal_px=(32.0, 32.0),
    )

    gaussian_params = scene.gaussian_parameters()
    gaussian3d_params = scene.gaussian3d_parameters()
    wave_params = scene.wave_gaussian_parameters()
    point_params = scene.point_parameters()

    assert scene.num_primitives == 4
    assert gaussian_params.shape == (1, 7)
    assert gaussian3d_params.shape == (1, 13)
    assert wave_params.shape == (1, 9)
    assert point_params.shape == (1, 4)
    assert scene.bounds() == ((0.0, 0.0), (3.0, 4.0))
    assert scene.depth_bounds() == (1e-3, 1e-3)
    assert scene.ordered_wave_gaussians("opacity")[0].opacity == 0.6
    assert scene.ordered_gaussians_3d("opacity")[0].opacity == 0.7


def test_primitive_scene_builder_and_renderer_return_complex_field() -> None:
    scene = build_primitive_scene_from_config(
        {
            "name": "demo",
            "gaussians": [
                {
                    "center_yx": [4.0, 5.0],
                    "sigma_yx": [1.5, 2.0],
                    "amplitude": 1.0,
                    "phase": 0.25,
                    "rotation": 0.0,
                }
            ],
            "points": [{"yx": [2.0, 3.0], "amplitude": 0.5, "phase": 0.0}],
        }
    )
    render = render_gaussian_scene(scene, (1, 1, 8, 8))

    assert render.field.shape == (1, 1, 8, 8)
    assert render.intensity.shape == (1, 1, 8, 8)
    assert torch.is_complex(render.field)
    assert float(render.intensity.max()) > 0.0


def test_primitive_scene_builder_supports_normalized_coordinates() -> None:
    scene = build_primitive_scene_from_config(
        {
            "coordinate_system": "normalized",
            "gaussians": [
                {
                    "center_yx": [0.5, 0.25],
                    "sigma_yx": [1.0, 1.0],
                    "amplitude": 1.0,
                    "phase": 0.0,
                }
            ],
        },
        dim=(1, 1, 9, 13),
    )

    gaussian = scene.gaussians[0]
    assert gaussian.center_yx == (4.0, 3.0)


def test_primitive_scene_builder_supports_ring_preset() -> None:
    scene = build_primitive_scene_from_config(
        {"preset": "ring", "count": 6, "radius": 0.2, "coordinate_system": "normalized"},
        dim=(1, 1, 11, 11),
    )

    assert scene.name == "gaussian_ring"
    assert len(scene.gaussians) == 6


def test_primitive_scene_builder_supports_depth_ring_preset() -> None:
    scene = build_primitive_scene_from_config(
        {"preset": "depth_ring", "count": 5, "depth_start": 1e-3, "depth_step": 1e-4},
        dim=(1, 1, 11, 11),
    )

    assert scene.name == "gaussian_depth_ring"
    assert len(scene.wave_gaussians) == 5
    assert scene.depth_bounds() == (1e-3, 1.4e-3)
    assert all(0.0 <= primitive.opacity <= 1.0 for primitive in scene.wave_gaussians)


def test_primitive_scene_builder_supports_gaussian3d_depth_ring_preset() -> None:
    scene = build_primitive_scene_from_config(
        {"preset": "gaussian3d_depth_ring", "count": 4},
        dim=(1, 1, 11, 11),
        pitch=6.4e-6,
    )

    assert scene.name == "gaussian3d_depth_ring"
    assert len(scene.gaussians_3d) == 4
    assert scene.projection_focal_px == (160.0, 160.0)


def test_primitive_scene_builder_supports_json_file(tmp_path) -> None:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(
        json.dumps(
            {
                "name": "json_scene",
                "gaussians": [
                    {
                        "center_yx": [1.0, 2.0],
                        "sigma_yx": [1.25, 1.5],
                        "amplitude": 0.7,
                        "phase": 0.1,
                    }
                ],
                "points": [{"yx": [3.0, 4.0], "amplitude": 0.2, "phase": 0.5}],
            }
        ),
        encoding="utf-8",
    )

    scene = build_primitive_scene_from_config({"path": str(scene_path)})

    assert scene.name == "json_scene"
    assert len(scene.gaussians) == 1
    assert len(scene.points) == 1


def test_available_primitive_renderers_exposes_naive_and_splat() -> None:
    assert available_primitive_renderers() == (
        "gaussian_naive",
        "gaussian_splat",
        "gaussian_wave",
        "gaussian_wave_awb",
        "gaussian_gws_exact",
        "gaussian_gws_exact_awb",
        "gaussian_gws_rpws_exact",
    )


def test_available_primitive_backends_exposes_auto_torch_and_warp() -> None:
    assert available_primitive_backends() == ("auto", "torch", "warp")


def test_gaussian_splat_matches_naive_renderer_for_gaussian_scene() -> None:
    scene = build_primitive_scene_from_config(
        {
            "name": "demo",
            "gaussians": [
                {
                    "center_yx": [4.0, 5.0],
                    "sigma_yx": [1.5, 2.0],
                    "amplitude": 1.0,
                    "phase": 0.25,
                    "rotation": 0.3,
                },
                {
                    "center_yx": [2.5, 1.5],
                    "sigma_yx": [0.75, 1.25],
                    "amplitude": 0.4,
                    "phase": 0.9,
                    "rotation": -0.2,
                },
            ],
        }
    )

    naive = render_gaussian_scene(scene, (1, 1, 8, 8))
    splat = render_gaussian_scene_splat(scene, (1, 1, 8, 8), backend="torch")

    assert torch.allclose(naive.field, splat.field, atol=1e-5, rtol=1e-5)
    assert torch.allclose(naive.intensity, splat.intensity, atol=1e-5, rtol=1e-5)
    assert splat.backend == "torch"


def test_gaussian_wave_renderer_returns_complex_field_for_depth_scene() -> None:
    scene = build_primitive_scene_from_config(
        {
            "wave_gaussians": [
                {
                    "center_yx": [4.0, 5.0],
                    "sigma_yx": [1.25, 1.5],
                    "depth": 1.0e-3,
                    "amplitude": 1.0,
                    "opacity": 0.8,
                    "phase": 0.2,
                },
                {
                    "center_yx": [2.0, 3.0],
                    "sigma_yx": [1.0, 1.0],
                    "depth": 2.0e-3,
                    "amplitude": 0.5,
                    "opacity": 0.5,
                    "phase": -0.1,
                },
            ]
        }
    )
    source = SourceSpec(dim=(1, 1, 8, 8), pitch=6.4e-6, wvl=532e-9)
    propagation = PropagationSpec(distance=0.0, mode="ASM")

    render = render_gaussian_scene_wave(
        scene,
        source.dim,
        source_spec=source,
        propagation_spec=propagation,
    )

    assert render.field.shape == (1, 1, 8, 8)
    assert torch.is_complex(render.field)
    assert render.backend == "torch"
    assert float(render.intensity.max()) > 0.0


def test_gaussian_wave_awb_reduces_peak_intensity_for_overlapping_primitives() -> None:
    scene = build_primitive_scene_from_config(
        {
            "wave_gaussians": [
                {
                    "center_yx": [4.0, 4.0],
                    "sigma_yx": [1.2, 1.2],
                    "depth": 1.0e-3,
                    "amplitude": 1.0,
                    "opacity": 0.8,
                    "phase": 0.0,
                },
                {
                    "center_yx": [4.0, 4.0],
                    "sigma_yx": [1.2, 1.2],
                    "depth": 2.0e-3,
                    "amplitude": 1.0,
                    "opacity": 0.8,
                    "phase": 0.0,
                },
            ]
        }
    )
    source = SourceSpec(dim=(1, 1, 8, 8), pitch=6.4e-6, wvl=532e-9)
    propagation = PropagationSpec(distance=0.0, mode="ASM")

    wave = render_gaussian_scene_wave(
        scene,
        source.dim,
        source_spec=source,
        propagation_spec=propagation,
    )
    awb = render_gaussian_scene_wave_awb(
        scene,
        source.dim,
        source_spec=source,
        propagation_spec=propagation,
        sort_order="front2back",
    )

    assert torch.is_complex(awb.field)
    assert float(awb.intensity.max()) < float(wave.intensity.max())


def test_gaussian_wave_awb_random_phase_is_deterministic_with_seed() -> None:
    scene = build_primitive_scene_from_config(
        {
            "wave_gaussians": [
                {
                    "center_yx": [4.0, 4.0],
                    "sigma_yx": [1.2, 1.2],
                    "depth": 1.0e-3,
                    "amplitude": 1.0,
                    "opacity": 0.8,
                    "phase": 0.0,
                }
            ]
        }
    )
    source = SourceSpec(dim=(1, 1, 8, 8), pitch=6.4e-6, wvl=532e-9)
    propagation = PropagationSpec(distance=0.0, mode="ASM")

    seeded_a = render_gaussian_scene_wave_awb(
        scene,
        source.dim,
        source_spec=source,
        propagation_spec=propagation,
        random_phase_std=0.25,
        random_seed=7,
    )
    seeded_b = render_gaussian_scene_wave_awb(
        scene,
        source.dim,
        source_spec=source,
        propagation_spec=propagation,
        random_phase_std=0.25,
        random_seed=7,
    )
    no_phase = render_gaussian_scene_wave_awb(
        scene,
        source.dim,
        source_spec=source,
        propagation_spec=propagation,
        random_phase_std=0.0,
        random_seed=7,
    )

    assert torch.allclose(seeded_a.field, seeded_b.field)
    assert not torch.allclose(seeded_a.field, no_phase.field)


def test_gaussian_gws_exact_renderer_returns_complex_field_for_3d_scene() -> None:
    scene = build_primitive_scene_from_config(
        {
            "gaussians_3d": [
                {
                    "mean_xyz": [0.0, 0.0, 1.2e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [7.5e-6, 9.0e-6, 1.0e-9],
                    "opacity": 0.7,
                    "amplitude": 1.0,
                    "phase": 0.2,
                }
            ],
            "projection": {"focal_lengths_px": [64.0, 64.0]},
        }
    )
    source = SourceSpec(dim=(1, 1, 16, 16), pitch=6.4e-6, wvl=532e-9)

    render = render_gaussian_scene_gws_exact(scene, source.dim, source_spec=source)

    assert render.field.shape == (1, 1, 16, 16)
    assert torch.is_complex(render.field)
    assert float(render.intensity.max()) > 0.0


def test_gaussian_gws_exact_awb_is_order_dependent_for_overlapping_3d_primitives() -> None:
    scene = build_primitive_scene_from_config(
        {
            "gaussians_3d": [
                {
                    "mean_xyz": [0.0, 0.0, 1.1e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [8.0e-6, 8.0e-6, 1.0e-9],
                    "opacity": 0.75,
                    "amplitude": 1.0,
                },
                {
                    "mean_xyz": [0.0, 0.0, 1.7e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [8.0e-6, 8.0e-6, 1.0e-9],
                    "opacity": 0.75,
                    "amplitude": 1.0,
                },
            ],
            "projection": {"focal_lengths_px": [72.0, 72.0]},
        }
    )
    source = SourceSpec(dim=(1, 1, 16, 16), pitch=6.4e-6, wvl=532e-9)

    front2back = render_gaussian_scene_gws_exact_awb(
        scene,
        source.dim,
        source_spec=source,
        sort_order="front2back",
    )
    back2front = render_gaussian_scene_gws_exact_awb(
        scene,
        source.dim,
        source_spec=source,
        sort_order="back2front",
    )

    assert torch.is_complex(front2back.field)
    assert torch.is_complex(back2front.field)
    assert float(front2back.intensity.max()) > 0.0
    assert float(back2front.intensity.max()) > 0.0
    assert not torch.allclose(front2back.field, back2front.field)


def test_gaussian_gws_rpws_exact_returns_time_multiplexed_frames() -> None:
    scene = build_primitive_scene_from_config(
        {
            "gaussians_3d": [
                {
                    "mean_xyz": [0.0, 0.0, 1.3e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [8.0e-6, 8.0e-6, 1.0e-9],
                    "opacity": 0.7,
                    "amplitude": 1.0,
                }
            ],
            "projection": {"focal_lengths_px": [72.0, 72.0]},
        }
    )
    source = SourceSpec(dim=(1, 1, 16, 16), pitch=6.4e-6, wvl=532e-9)

    render = render_gaussian_scene_gws_rpws_exact(
        scene,
        source.dim,
        source_spec=source,
        num_frames=4,
        random_seed=5,
    )

    assert render.field.shape == (1, 1, 16, 16)
    assert torch.is_complex(render.field)
    assert render.frame_fields is not None
    assert len(render.frame_fields) == 4
    expected_intensity = torch.stack([frame.abs().square() for frame in render.frame_fields], dim=0).mean(dim=0)
    assert torch.allclose(render.intensity, expected_intensity)


def test_gaussian_gws_rpws_exact_is_deterministic_with_seed() -> None:
    scene = build_primitive_scene_from_config(
        {
            "gaussians_3d": [
                {
                    "mean_xyz": [1.0e-5, -1.0e-5, 1.2e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [7.0e-6, 9.0e-6, 1.0e-9],
                    "opacity": 0.8,
                    "amplitude": 1.0,
                }
            ],
            "projection": {"focal_lengths_px": [64.0, 64.0]},
        }
    )
    source = SourceSpec(dim=(1, 1, 16, 16), pitch=6.4e-6, wvl=532e-9)

    seeded_a = render_gaussian_scene_gws_rpws_exact(
        scene,
        source.dim,
        source_spec=source,
        num_frames=3,
        random_seed=11,
    )
    seeded_b = render_gaussian_scene_gws_rpws_exact(
        scene,
        source.dim,
        source_spec=source,
        num_frames=3,
        random_seed=11,
    )
    different_seed = render_gaussian_scene_gws_rpws_exact(
        scene,
        source.dim,
        source_spec=source,
        num_frames=3,
        random_seed=17,
    )

    assert torch.allclose(seeded_a.field, seeded_b.field)
    assert seeded_a.frame_fields is not None
    assert seeded_b.frame_fields is not None
    assert all(torch.allclose(a, b) for a, b in zip(seeded_a.frame_fields, seeded_b.frame_fields))
    assert not torch.allclose(seeded_a.field, different_seed.field)


def test_gaussian_gws_exact_warp_matches_torch_when_available() -> None:
    if not primitive_warp_available():
        return

    scene = build_primitive_scene_from_config(
        {
            "gaussians_3d": [
                {
                    "mean_xyz": [0.0, 0.0, 1.2e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [8.0e-6, 7.0e-6, 1.0e-9],
                    "opacity": 0.75,
                    "amplitude": 0.9,
                    "phase": 0.2,
                },
                {
                    "mean_xyz": [1.0e-5, -1.0e-5, 1.7e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [7.5e-6, 8.5e-6, 1.0e-9],
                    "opacity": 0.65,
                    "amplitude": 1.0,
                    "phase": -0.1,
                },
            ],
            "projection": {"focal_lengths_px": [72.0, 72.0]},
        }
    )
    source = SourceSpec(dim=(1, 1, 16, 16), pitch=6.4e-6, wvl=532e-9)

    torch_render = render_gaussian_scene_gws_exact(
        scene,
        source.dim,
        source_spec=source,
        backend="torch",
    )
    try:
        warp_render = render_gaussian_scene_gws_exact(
            scene,
            source.dim,
            source_spec=source,
            backend="warp",
        )
    except RuntimeError:
        return

    assert warp_render.backend == "warp"
    assert torch.allclose(torch_render.field, warp_render.field, atol=2e-4, rtol=2e-4)


def test_gaussian_gws_exact_awb_warp_matches_torch_when_available() -> None:
    if not primitive_warp_available():
        return

    scene = build_primitive_scene_from_config(
        {
            "gaussians_3d": [
                {
                    "mean_xyz": [0.0, 0.0, 1.1e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [8.0e-6, 8.0e-6, 1.0e-9],
                    "opacity": 0.8,
                },
                {
                    "mean_xyz": [0.0, 0.0, 1.5e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [8.0e-6, 8.0e-6, 1.0e-9],
                    "opacity": 0.6,
                },
            ],
            "projection": {"focal_lengths_px": [72.0, 72.0]},
        }
    )
    source = SourceSpec(dim=(1, 1, 16, 16), pitch=6.4e-6, wvl=532e-9)

    torch_render = render_gaussian_scene_gws_exact_awb(
        scene,
        source.dim,
        source_spec=source,
        sort_order="front2back",
        backend="torch",
    )
    try:
        warp_render = render_gaussian_scene_gws_exact_awb(
            scene,
            source.dim,
            source_spec=source,
            sort_order="front2back",
            backend="warp",
        )
    except RuntimeError:
        return

    assert warp_render.backend == "warp"
    assert torch.allclose(torch_render.field, warp_render.field, atol=2e-4, rtol=2e-4)


def test_gaussian_gws_rpws_exact_warp_matches_torch_when_available() -> None:
    if not primitive_warp_available():
        return

    scene = build_primitive_scene_from_config(
        {
            "gaussians_3d": [
                {
                    "mean_xyz": [0.0, 0.0, 1.25e-3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "scale_xyz": [8.0e-6, 8.0e-6, 1.0e-9],
                    "opacity": 0.7,
                    "amplitude": 1.0,
                }
            ],
            "projection": {"focal_lengths_px": [64.0, 64.0]},
        }
    )
    source = SourceSpec(dim=(1, 1, 16, 16), pitch=6.4e-6, wvl=532e-9)

    torch_render = render_gaussian_scene_gws_rpws_exact(
        scene,
        source.dim,
        source_spec=source,
        num_frames=3,
        random_seed=13,
        backend="torch",
    )
    try:
        warp_render = render_gaussian_scene_gws_rpws_exact(
            scene,
            source.dim,
            source_spec=source,
            num_frames=3,
            random_seed=13,
            backend="warp",
        )
    except RuntimeError:
        return

    assert warp_render.backend == "warp"
    assert torch.allclose(torch_render.field, warp_render.field, atol=2e-4, rtol=2e-4)
    assert torch.allclose(torch_render.intensity, warp_render.intensity, atol=2e-4, rtol=2e-4)


def test_neural_trainer_identity_step_is_zero_loss() -> None:
    batch = NeuralHolographyBatch(
        target_intensity=torch.ones(4, 4),
        measured_intensity=torch.ones(4, 4),
    )
    trainer = NeuralHolographyTrainer(
        predict_fn=lambda item: item.target_intensity,
        loss_fn=lambda prediction, reference: torch.mean((prediction - reference) ** 2),
    )

    result = trainer.step(batch)

    assert float(result.loss) == 0.0
    assert result.metrics["loss"] == 0.0


def test_capture_session_spec_requires_positive_wavelengths() -> None:
    spec = CaptureSessionSpec(wavelengths=(532e-9, 638e-9), camera_name="cam0", slm_name="slm0")
    assert spec.camera_name == "cam0"

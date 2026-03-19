from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
from omegaconf import DictConfig, OmegaConf

from pado.display import LCOSLUT

from ..core.losses import tensor_reconstruction_metrics
from ..core.pipelines import HologramPipeline
from ..core.specs import PropagationSpec, SourceSpec
from ..core.targets import IntensityTarget
from ..devices import CameraObservationSpec
from ..devices.slm import PhaseOnlyLCOSSLM
from ..phase_only import DoublePhaseAmplitudeCoder, GerchbergSaxtonPhaseOptimizer
from ..primitive_based import build_primitive_scene_from_config, render_primitive_scene
from .registry import register_experiment, resolve_experiment_runner


@dataclass(frozen=True)
class ExperimentSummary:
    method: str
    metrics: Dict[str, float]
    extras: Dict[str, Any]


@dataclass(frozen=True)
class ExperimentContext:
    method: str
    mapping: Mapping[str, Any]
    source: SourceSpec
    propagation: PropagationSpec
    target: IntensityTarget
    experiment_cfg: Mapping[str, Any]
    backend_cfg: Mapping[str, Any]
    camera: CameraObservationSpec | None


def _cfg_to_mapping(cfg: DictConfig | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return cfg


def build_source_from_config(cfg: Mapping[str, Any]) -> SourceSpec:
    return SourceSpec(
        dim=(int(cfg["batch"]), int(cfg["channels"]), int(cfg["height"]), int(cfg["width"])),
        pitch=float(cfg["pitch"]),
        wvl=float(cfg["wvl"]),
        device=str(cfg.get("device", "cpu")),
    )


def build_propagation_from_config(cfg: Mapping[str, Any]) -> PropagationSpec:
    return PropagationSpec(
        distance=float(cfg["distance"]),
        mode=str(cfg.get("mode", "ASM")),
        polar=str(cfg.get("polar", "non")),
        offset=tuple(cfg.get("offset", (0.0, 0.0))),
        linear=bool(cfg.get("linear", True)),
        band_limit=bool(cfg.get("band_limit", True)),
        b=float(cfg.get("b", 1.0)),
        sampling_ratio=int(cfg.get("sampling_ratio", 1)),
        vectorized=bool(cfg.get("vectorized", False)),
        steps=int(cfg.get("steps", 100)),
    )


def build_target_from_config(cfg: Mapping[str, Any], source: SourceSpec) -> IntensityTarget:
    kind = str(cfg.get("kind", "flat"))
    fill_value = float(cfg.get("value", 1.0))
    batch, channels, rows, cols = source.dim

    if kind == "flat":
        intensity = torch.full(source.dim, fill_value, device=source.device)
    elif kind == "gaussian":
        sigma = float(cfg.get("sigma", min(rows, cols) / 6.0))
        y = torch.arange(rows, device=source.device, dtype=torch.float32) - (rows - 1) / 2
        x = torch.arange(cols, device=source.device, dtype=torch.float32) - (cols - 1) / 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        image = torch.exp(-(xx.square() + yy.square()) / (2 * sigma**2))
        intensity = image.view(1, 1, rows, cols).expand(batch, channels, rows, cols) * fill_value
    else:
        raise ValueError(f"unsupported target kind {kind}")

    return IntensityTarget(intensity, normalize_mean=bool(cfg.get("normalize_mean", True)))


def build_lut_from_config(cfg: Mapping[str, Any]) -> LCOSLUT:
    levels = int(cfg.get("levels", 256))
    phase_min = float(cfg.get("phase_min", 0.0))
    phase_max = float(cfg.get("phase_max", 2 * torch.pi))
    phase_lut = torch.linspace(phase_min, phase_max, levels)
    return LCOSLUT(phase_lut=phase_lut, wvl_ref=cfg.get("wvl_ref"))


def build_camera_from_config(cfg: Mapping[str, Any] | None) -> CameraObservationSpec | None:
    mapping = cfg or {}
    if not bool(mapping.get("enabled", False)):
        return None

    crop_shape = mapping.get("crop_shape")
    if crop_shape is not None:
        crop_shape = (int(crop_shape[0]), int(crop_shape[1]))

    return CameraObservationSpec(
        enabled=True,
        name=str(mapping.get("name")) if mapping.get("name") is not None else None,
        downsample=int(mapping.get("downsample", 1)),
        crop_shape=crop_shape,
        exposure=float(mapping.get("exposure", 1.0)),
        normalize_mean=bool(mapping.get("normalize_mean", False)),
    )


def build_experiment_context(cfg: DictConfig | Mapping[str, Any]) -> ExperimentContext:
    mapping = _cfg_to_mapping(cfg)
    source = build_source_from_config(mapping["source"])
    propagation = build_propagation_from_config(mapping["propagation"])
    target = build_target_from_config(mapping["target"], source)
    experiment_cfg = mapping["experiment"]
    backend_cfg = mapping.get("backend", {"name": "auto", "warp_cache_dir": None})
    camera = build_camera_from_config(mapping.get("camera"))
    method = str(experiment_cfg["method"])
    return ExperimentContext(
        method=method,
        mapping=mapping,
        source=source,
        propagation=propagation,
        target=target,
        experiment_cfg=experiment_cfg,
        backend_cfg=backend_cfg,
        camera=camera,
    )


def _run_gs(context: ExperimentContext) -> ExperimentSummary:
    optimizer = GerchbergSaxtonPhaseOptimizer(context.source, context.propagation)
    result = optimizer.optimize(
        context.target,
        iterations=int(context.experiment_cfg.get("iterations", 5)),
    )
    metrics = {
        "final_mse": result.history[-1],
    }
    extras = {
        "history_length": len(result.history),
    }
    return ExperimentSummary(method=context.method, metrics=metrics, extras=extras)


def _run_dpac(context: ExperimentContext) -> ExperimentSummary:
    slm_cfg = context.mapping["slm"]
    lut = build_lut_from_config(slm_cfg)
    slm = PhaseOnlyLCOSSLM(context.source, lut)
    coder = DoublePhaseAmplitudeCoder(
        context.source,
        backend=str(context.backend_cfg.get("name", "auto")),
        warp_cache_dir=context.backend_cfg.get("warp_cache_dir"),
    )
    dpac = coder.encode_target(context.target, normalize_amplitude=True)
    pipeline = HologramPipeline(context.source, context.propagation, slm=slm)
    forward = pipeline.forward_phase(dpac.checkerboard_phase, target=context.target)
    assert forward.metrics is not None
    metrics = {
        "mse": float(forward.metrics["mse"]),
        "psnr": float(forward.metrics["psnr"]),
    }
    extras = {
        "phase_shape": tuple(dpac.checkerboard_phase.shape),
        "kernel_backend": dpac.kernel_backend,
        "backend_reason": dpac.backend_reason,
    }
    return ExperimentSummary(method=context.method, metrics=metrics, extras=extras)


def _run_primitive_gaussian(context: ExperimentContext) -> ExperimentSummary:
    scene_cfg = context.mapping.get("primitives", {})
    scene = build_primitive_scene_from_config(
        scene_cfg if isinstance(scene_cfg, Mapping) else {},
        dim=context.source.dim,
        pitch=context.source.pitch,
    )
    renderer_name = str(context.experiment_cfg.get("renderer", "gaussian_naive"))
    render = render_primitive_scene(
        scene,
        context.source.dim,
        renderer=renderer_name,
        backend=str(context.backend_cfg.get("name", "auto")),
        device=context.source.device,
        warp_cache_dir=context.backend_cfg.get("warp_cache_dir"),
        source_spec=context.source,
        propagation_spec=context.propagation,
        sort_order=str(context.experiment_cfg.get("sort_order", "back2front")),
        alpha_binary_threshold=float(context.experiment_cfg.get("alpha_binary_threshold", 0.0)),
        random_phase_std=float(context.experiment_cfg.get("random_phase_std", 0.0)),
        num_frames=int(context.experiment_cfg.get("num_frames", 1)),
        angular_profile=str(context.experiment_cfg.get("angular_profile", "uniform")),
        angular_radius_fraction=float(context.experiment_cfg.get("angular_radius_fraction", 1.0)),
        angular_sigma_fraction=float(context.experiment_cfg.get("angular_sigma_fraction", 0.35)),
        random_phase_range=float(context.experiment_cfg.get("random_phase_range", torch.pi)),
        random_seed=(
            int(context.experiment_cfg["random_seed"])
            if context.experiment_cfg.get("random_seed") is not None
            else None
        ),
    )

    pipeline = HologramPipeline(context.source, context.propagation)
    if render.frame_fields is not None:
        propagated_frames = []
        for frame_field in render.frame_fields:
            frame_light = context.source.make_light()
            frame_light.set_field(frame_field.to(frame_light.field.device, dtype=frame_light.field.dtype))
            forward = pipeline.forward_source(frame_light, target=None, normalize_metrics=False)
            propagated_frames.append(forward.propagated_light.get_intensity().real)
        propagated_intensity = torch.stack(propagated_frames, dim=0).mean(dim=0)
        if context.camera is None:
            observed_metrics = tensor_reconstruction_metrics(
                propagated_intensity,
                render.intensity.to(propagated_intensity.device),
                normalize=False,
            )
            metrics = {
                "mse": float(observed_metrics["mse"]),
                "psnr": float(observed_metrics["psnr"]),
            }
            observed_shape = tuple(render.intensity.shape)
        else:
            observed_prediction = context.camera.observe_intensity(propagated_intensity)
            observed_target = context.camera.observe_intensity(render.intensity.to(propagated_intensity.device))
            observed_metrics = tensor_reconstruction_metrics(
                observed_prediction,
                observed_target,
                normalize=False,
            )
            metrics = {
                "mse": float(observed_metrics["mse"]),
                "psnr": float(observed_metrics["psnr"]),
            }
            observed_shape = tuple(int(v) for v in observed_prediction.shape)
    else:
        source_light = context.source.make_light()
        source_light.set_field(render.field.to(source_light.field.device, dtype=source_light.field.dtype))
        if context.camera is None:
            target = IntensityTarget(render.intensity, normalize_mean=False)
            forward = pipeline.forward_source(
                source_light,
                target=target,
                normalize_metrics=False,
            )
            assert forward.metrics is not None
            metrics = {
                "mse": float(forward.metrics["mse"]),
                "psnr": float(forward.metrics["psnr"]),
            }
            observed_shape = tuple(render.intensity.shape)
        else:
            forward = pipeline.forward_source(source_light, target=None, normalize_metrics=False)
            propagated_intensity = forward.propagated_light.get_intensity().real
            observed_prediction = context.camera.observe_intensity(propagated_intensity)
            observed_target = context.camera.observe_intensity(render.intensity.to(propagated_intensity.device))
            observed_metrics = tensor_reconstruction_metrics(
                observed_prediction,
                observed_target,
                normalize=False,
            )
            metrics = {
                "mse": float(observed_metrics["mse"]),
                "psnr": float(observed_metrics["psnr"]),
            }
            observed_shape = tuple(int(v) for v in observed_prediction.shape)

    bounds = scene.bounds()
    extras = {
        "renderer": renderer_name,
        "backend": render.backend,
        "backend_reason": render.backend_reason,
        "scene_name": scene.name,
        "num_primitives": scene.num_primitives,
        "num_gaussians": len(scene.gaussians),
        "num_gaussians_3d": len(scene.gaussians_3d),
        "num_wave_gaussians": len(scene.wave_gaussians),
        "num_points": len(scene.points),
        "projection_focal_px": scene.projection_focal_px,
        "field_shape": tuple(render.field.shape),
        "camera": context.camera.name if context.camera is not None else None,
        "camera_downsample": context.camera.downsample if context.camera is not None else 1,
        "camera_crop_shape": context.camera.crop_shape if context.camera is not None else None,
        "observed_shape": observed_shape,
        "num_frames": render.num_frames,
        "sort_order": str(context.experiment_cfg.get("sort_order", "back2front")),
        "alpha_binary_threshold": float(context.experiment_cfg.get("alpha_binary_threshold", 0.0)),
        "angular_profile": str(context.experiment_cfg.get("angular_profile", "uniform")),
        "angular_radius_fraction": float(context.experiment_cfg.get("angular_radius_fraction", 1.0)),
        "angular_sigma_fraction": float(context.experiment_cfg.get("angular_sigma_fraction", 0.35)),
        "random_phase_range": float(context.experiment_cfg.get("random_phase_range", torch.pi)),
        "random_phase_std": float(context.experiment_cfg.get("random_phase_std", 0.0)),
        "random_seed": (
            int(context.experiment_cfg["random_seed"])
            if context.experiment_cfg.get("random_seed") is not None
            else None
        ),
        "bounds": bounds,
        "depth_bounds": scene.depth_bounds(),
    }
    return ExperimentSummary(method=context.method, metrics=metrics, extras=extras)


def run_experiment(cfg: DictConfig | Mapping[str, Any]) -> ExperimentSummary:
    context = build_experiment_context(cfg)
    return resolve_experiment_runner(context.method)(context)


register_experiment("gs", _run_gs)
register_experiment("dpac", _run_dpac)
register_experiment("primitive_gaussian", _run_primitive_gaussian)
register_experiment("primitive_gaussian_splat", _run_primitive_gaussian)
register_experiment("primitive_gaussian_wave", _run_primitive_gaussian)
register_experiment("primitive_gaussian_awb", _run_primitive_gaussian)
register_experiment("primitive_gaussian_rpws", _run_primitive_gaussian)
register_experiment("primitive_gaussian_gws_exact", _run_primitive_gaussian)


__all__ = [
    "ExperimentSummary",
    "ExperimentContext",
    "build_camera_from_config",
    "build_experiment_context",
    "build_lut_from_config",
    "build_propagation_from_config",
    "build_source_from_config",
    "build_target_from_config",
    "run_experiment",
]

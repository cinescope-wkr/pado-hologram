from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
from omegaconf import DictConfig, OmegaConf

from pado.display import LCOSLUT

from .algorithms import DoublePhaseAmplitudeCoder, GerchbergSaxtonPhaseOptimizer
from .config import PropagationSpec, SourceSpec
from .pipeline import HologramPipeline
from .slm import PhaseOnlyLCOSSLM
from .targets import IntensityTarget


@dataclass(frozen=True)
class ExperimentSummary:
    method: str
    metrics: Dict[str, float]
    extras: Dict[str, Any]


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


def run_experiment(cfg: DictConfig | Mapping[str, Any]) -> ExperimentSummary:
    mapping = _cfg_to_mapping(cfg)
    source = build_source_from_config(mapping["source"])
    propagation = build_propagation_from_config(mapping["propagation"])
    target = build_target_from_config(mapping["target"], source)
    experiment_cfg = mapping["experiment"]
    backend_cfg = mapping.get("backend", {"name": "auto", "warp_cache_dir": None})
    method = str(experiment_cfg["method"])

    if method == "gs":
        optimizer = GerchbergSaxtonPhaseOptimizer(source, propagation)
        result = optimizer.optimize(target, iterations=int(experiment_cfg.get("iterations", 5)))
        metrics = {
            "final_mse": result.history[-1],
        }
        extras = {
            "history_length": len(result.history),
        }
        return ExperimentSummary(method=method, metrics=metrics, extras=extras)

    if method == "dpac":
        slm_cfg = mapping["slm"]
        lut = build_lut_from_config(slm_cfg)
        slm = PhaseOnlyLCOSSLM(source, lut)
        coder = DoublePhaseAmplitudeCoder(
            source,
            backend=str(backend_cfg.get("name", "auto")),
            warp_cache_dir=backend_cfg.get("warp_cache_dir"),
        )
        dpac = coder.encode_target(target, normalize_amplitude=True)
        pipeline = HologramPipeline(source, propagation, slm=slm)
        forward = pipeline.forward_phase(dpac.checkerboard_phase, target=target)
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
        return ExperimentSummary(method=method, metrics=metrics, extras=extras)

    raise ValueError(f"unsupported experiment method {method}")

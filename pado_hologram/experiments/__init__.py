"""Experiment orchestration helpers for PADO Hologram."""

from .compose import compose_experiment_config, render_config_yaml
from .registry import available_experiments, register_experiment, resolve_experiment_runner
from .runner import (
    ExperimentContext,
    ExperimentSummary,
    build_camera_from_config,
    build_experiment_context,
    build_lut_from_config,
    build_propagation_from_config,
    build_source_from_config,
    build_target_from_config,
    run_experiment,
)

__all__ = [
    "ExperimentContext",
    "ExperimentSummary",
    "available_experiments",
    "build_camera_from_config",
    "build_experiment_context",
    "build_lut_from_config",
    "build_propagation_from_config",
    "build_source_from_config",
    "build_target_from_config",
    "compose_experiment_config",
    "register_experiment",
    "render_config_yaml",
    "resolve_experiment_runner",
    "run_experiment",
]

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ExperimentRunner = Callable[[Any], Any]

_EXPERIMENT_RUNNERS: dict[str, ExperimentRunner] = {}


def register_experiment(name: str, runner: ExperimentRunner) -> None:
    if not name:
        raise ValueError("experiment name must be non-empty")
    _EXPERIMENT_RUNNERS[name] = runner


def available_experiments() -> tuple[str, ...]:
    return tuple(sorted(_EXPERIMENT_RUNNERS))


def resolve_experiment_runner(name: str) -> ExperimentRunner:
    try:
        return _EXPERIMENT_RUNNERS[name]
    except KeyError as exc:
        supported = ", ".join(available_experiments()) or "<none>"
        raise ValueError(f"unsupported experiment method {name!r}; available: {supported}") from exc


__all__ = [
    "ExperimentRunner",
    "available_experiments",
    "register_experiment",
    "resolve_experiment_runner",
]

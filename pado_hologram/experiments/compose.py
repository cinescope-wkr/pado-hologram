from __future__ import annotations

from collections.abc import Sequence

from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf


def compose_experiment_config(
    overrides: Sequence[str] | None = None,
    *,
    config_name: str = "config",
) -> DictConfig:
    with initialize_config_module(version_base=None, config_module="pado_hologram.conf"):
        return compose(config_name=config_name, overrides=list(overrides or ()))


def render_config_yaml(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)


__all__ = [
    "compose_experiment_config",
    "render_config_yaml",
]

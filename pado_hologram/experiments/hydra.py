from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from .runner import run_experiment


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    summary = run_experiment(cfg)
    payload = {
        "method": summary.method,
        "metrics": summary.metrics,
        "extras": summary.extras,
    }
    print(OmegaConf.to_yaml(payload, resolve=True))


if __name__ == "__main__":
    main()

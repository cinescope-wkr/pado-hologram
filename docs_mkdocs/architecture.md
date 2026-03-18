# Architecture

`PADO Hologram` is layered on top of the original `pado` optics core.

Role split:

- `pado`: differentiable optics core
- `pado_hologram`: higher-level holography workflows

Current `pado_hologram` modules:

- `config` for source and propagation specifications
- `slm` for phase-only LCOS/SLM encoding
- `targets` for reconstruction targets
- `losses` for reconstruction losses and metrics
- `pipeline` for single-plane and multi-plane orchestration
- `algorithms` for Gerchberg-Saxton and DPAC
- `experiment` and `hydra_app` for reproducible runs

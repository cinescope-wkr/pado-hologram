# Experiments and Hydra

This page describes the current experiment layer.

## Why Hydra Is Used Here

The lower-level optics core should remain easy to import and use directly.
However, holography experiments quickly accumulate configuration choices:

- source geometry
- wavelength and pixel pitch
- propagation distance and mode
- target type
- algorithm choice
- display model

Hydra is useful at this upper layer because it makes those choices explicit and
reproducible.

## Current Entry Point

The current repository exposes:

```bash
python -m pado_hologram.hydra_app experiment=gs
python -m pado_hologram.hydra_app experiment=dpac target=gaussian
python -m pado_hologram.hydra_app experiment=dpac target=gaussian backend=warp
```

These runs are intentionally small, but they establish an experiment-oriented
entry point that can grow into a more complete research workflow layer.

## Warp in This Layer

Warp belongs here for the same reason Hydra does: it is part of the higher-level
research workflow layer, not part of the minimal optics core.

Today, the Warp backend is optional and starts with custom holography kernels in
the DPAC path. That gives the repository a maintainable place to grow future
GPU kernels without pretending that the whole propagation stack has been
rewritten.

## Design Principle

Hydra belongs in `pado_hologram`, not in `pado`.

That distinction keeps:

- the optics core lighter
- experiment orchestration explicit
- reproducibility tooling in the layer where it matters most

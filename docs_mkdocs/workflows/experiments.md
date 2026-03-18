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
```

These runs are intentionally small, but they establish an experiment-oriented
entry point that can grow into a more complete research workflow layer.

## Design Principle

Hydra belongs in `pado_hologram`, not in `pado`.

That distinction keeps:

- the optics core lighter
- experiment orchestration explicit
- reproducibility tooling in the layer where it matters most

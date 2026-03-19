# Experiments, CLI, and Hydra

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

## Current Entry Points

The preferred repository-facing entry point today is the package CLI:

```bash
pado-hologram doctor --run-smoke
pado-hologram run experiment=gs
pado-hologram run experiment=dpac target=gaussian
pado-hologram run experiment=primitive_gaussian_gws_exact primitives=gaussian3d_depth_ring
pado-hologram run experiment=primitive_gaussian_rpws primitives=gaussian3d_depth_ring
```

The Hydra-native path is still available when you want raw Hydra composition:

```bash
python -m pado_hologram.hydra_app experiment=gs
python -m pado_hologram.hydra_app experiment=dpac target=gaussian
```

## Current Experiment Families

The current config tree exposes these main families:

- `gs`: compact Gerchberg-Saxton baseline
- `dpac`: double-phase amplitude coding
- `primitive_gaussian` / `primitive_gaussian_splat`: primitive-scene Gaussian baselines
- `primitive_gaussian_wave` / `primitive_gaussian_awb`: depth-aware and alpha-wave-blending style paths
- `primitive_gaussian_gws_exact`: exact GWS-style primitive path
- `primitive_gaussian_rpws`: exact GWS-backed RPWS baseline with structured random phase and time-multiplexed intensity averaging

## Useful Overrides

Some of the most useful current overrides are:

```bash
pado-hologram run experiment=dpac target=gaussian backend=warp
pado-hologram run experiment=primitive_gaussian_gws_exact primitives=gaussian3d_depth_ring backend=warp
pado-hologram run experiment=primitive_gaussian_splat primitives=gaussian_ring camera=binned2 backend=torch
```

These runs are still intentionally small, but they establish a reproducible
experiment layer that is richer than the original DPAC-only starting point.

## Warp in This Layer

Warp belongs here for the same reason Hydra does: it is part of the higher-level
research workflow layer, not part of the minimal optics core.

Today, the Warp backend is optional and currently appears in:

- the DPAC checkerboard kernel path
- primitive-scene Gaussian splat backends
- exact primitive-based GWS/RPWS renderer paths

That gives the repository a maintainable place to grow future GPU kernels
without pretending that the whole propagation stack has been rewritten.

## Design Principle

Hydra belongs in `pado_hologram`, not in `pado`.

That distinction keeps:

- the optics core lighter
- experiment orchestration explicit
- reproducibility tooling in the layer where it matters most

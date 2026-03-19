# Module Reference Map

This page is not a full API reference. Instead, it is a guide to where the most
important repository concepts currently live.

## Core `pado` Modules

- `pado.light`: complex light-field representation
- `pado.optical_element`: optical elements and modulation
- `pado.propagator`: free-space propagation methods
- `pado.material`: material definitions
- `pado.math`: numerical helpers

## Compatibility Bridge

- `pado.display`: LUT-based LCOS/SLM encoding bridge kept in the `pado` namespace for compatibility while higher-level SLM workflows move toward `pado_hologram`

## `pado_hologram` Modules

- `pado_hologram.config`: source and propagation specifications
- `pado_hologram.backends`: optional custom-kernel backend helpers, including [`NVIDIA Warp`](https://github.com/NVIDIA/warp)
- `pado_hologram.targets`: target intensity representations
- `pado_hologram.losses`: reconstruction losses and metrics
- `pado_hologram.slm`: framework-level LCOS/SLM helpers
- `pado_hologram.pipeline`: single-plane and multi-plane workflow composition
- `pado_hologram.algorithms`: GS and DPAC helpers
- `pado_hologram.primitive_based`: gsplat-free primitive-scene Gaussian baselines, depth-aware and alpha-wave-blending variants, exact GWS/RPWS paths, backend selection, and preset/JSON scene ingestion
- `pado_hologram.devices`: SLM support and optional camera/observation transforms
- `pado_hologram.experiments`: experiment execution utilities and config composition
- `pado_hologram.experiment`: backward-compatible re-export for older experiment imports
- `pado_hologram.hydra_app`: Hydra CLI entry point

## New Internal Layout

The package is also beginning to migrate toward a more scalable internal split:

- `pado_hologram.core`: specs, targets, losses, and pipelines
- `pado_hologram.devices`: SLM-facing abstractions plus optional camera observation helpers
- `pado_hologram.phase_only`: phase-only CGH algorithms such as GS and DPAC
- `pado_hologram.primitive_based`: primitive-based renderer baselines, preset/JSON scene ingestion, the first vectorized splat path, a depth-aware `gaussian_wave` baseline, an hsplat-inspired `gaussian_wave_awb` path for alpha wave blending, and an exact GWS-backed RPWS baseline with structured random phase and time-multiplexed intensity averaging, all without an external `gsplat` dependency
- `pado_hologram.experiments`: experiment runners and Hydra-facing wrappers
- `pado_hologram.representations`: primitive-based scene representations
- `pado_hologram.neural`: neural holography, calibration, and capture-oriented scaffolds

These newer paths are meant to make future additions such as primitive-based CGH
or neural holography easier to place without overloading the top-level package.

## Where to Look Next

If you need exhaustive API details, use the Sphinx documentation that remains in
`docs/source`.

This MkDocs page is meant to answer the practical question:
"which module should I open first?"

# Module Reference Map

This page is not a full API reference. Instead, it is a guide to where the most
important repository concepts currently live.

## Core `pado` Modules

- `pado.light`: complex light-field representation
- `pado.optical_element`: optical elements and modulation
- `pado.propagator`: free-space propagation methods
- `pado.material`: material definitions
- `pado.math`: numerical helpers
- `pado.display`: LUT-based LCOS/SLM encoding bridge

## `pado_hologram` Modules

- `pado_hologram.config`: source and propagation specifications
- `pado_hologram.targets`: target intensity representations
- `pado_hologram.losses`: reconstruction losses and metrics
- `pado_hologram.slm`: framework-level LCOS/SLM helpers
- `pado_hologram.pipeline`: single-plane and multi-plane workflow composition
- `pado_hologram.algorithms`: GS and DPAC helpers
- `pado_hologram.experiment`: experiment execution utilities
- `pado_hologram.hydra_app`: Hydra CLI entry point

## Where to Look Next

If you need exhaustive API details, use the Sphinx documentation that remains in
`docs/source`.

This MkDocs page is meant to answer the practical question:
"which module should I open first?"

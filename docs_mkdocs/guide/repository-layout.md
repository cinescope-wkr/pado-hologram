# Repository Layout

This page describes how the repository is organized today.

## Top-Level Structure

- `pado/`: the original differentiable optics core
- `pado_hologram/`: the current higher-level holography framework layer
- `docs_mkdocs/`: this MkDocs-based repository guide
- `docs/source/`: the Sphinx-based API and reference documentation
- `tests/`: regression tests for the maintained repository state
- `example/`: notebooks carried over from the broader optics and CGH examples

## Core Layer: `pado`

The `pado` package remains the optics engine. It currently provides the main
building blocks for:

- complex light-field representation
- optical elements
- free-space propagation
- materials and numerical utilities
- a device-oriented `display.py` compatibility bridge

This layer is intentionally compact and reusable.

## Holography Layer: `pado_hologram`

The `pado_hologram` package is where the repository’s newer identity lives. It
currently includes:

- source and propagation specifications
- LCOS/SLM encoding support
- target definitions
- reconstruction losses and metrics
- single-plane and multi-plane pipelines
- compact GS and DPAC helpers
- CLI- and Hydra-driven experiment entry points

Internally, the current layout separates responsibilities more clearly:

- `pado_hologram.core`: shared source, target, loss, and pipeline abstractions
- `pado_hologram.devices`: SLM abstractions plus optional camera/observation transforms for primitive and capture-aware workflows
- `pado_hologram.phase_only`: compact GS and DPAC implementations
- `pado_hologram.primitive_based`: gsplat-free primitive-scene Gaussian baselines, depth-aware `gaussian_wave`, alpha-wave-blending variants, exact GWS/RPWS paths, backend selection, and preset/JSON scene ingestion
- `pado_hologram.experiments`: experiment runners, config composition, and Hydra-facing entry points
- `pado_hologram.representations`: primitive-oriented scene and parameter containers
- `pado_hologram.neural`: learning- and capture-facing scaffolds

At the top level, the package also now exposes a small CLI through
`pado_hologram.cli` and `python -m pado_hologram`.

The older top-level imports such as `pado_hologram.config` and
`pado_hologram.pipeline` are still kept as compatibility shims.

## Documentation Split

The repository currently maintains two documentation styles:

- MkDocs: high-level architecture, workflows, and contributor-facing context
- Sphinx: fuller API reference and source-linked technical pages

This split is intentional: the MkDocs site is meant to be readable and
self-contained, while the Sphinx site remains useful for exhaustive API lookup.

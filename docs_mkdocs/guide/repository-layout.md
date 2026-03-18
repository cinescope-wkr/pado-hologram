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
- a device-oriented `display.py` bridge

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
- Hydra-driven experiment entry points

## Documentation Split

The repository currently maintains two documentation styles:

- MkDocs: high-level architecture, workflows, and contributor-facing context
- Sphinx: fuller API reference and source-linked technical pages

This split is intentional: the MkDocs site is meant to be readable and
self-contained, while the Sphinx site remains useful for exhaustive API lookup.

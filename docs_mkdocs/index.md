# PADO Hologram

`PADO Hologram` is an open-source computer-generated holography framework built
on top of [`PADO`](https://github.com/shwbaek/pado), the differentiable optics
core originally developed by the
[`POSTECH Computer Graphics Lab`](https://sites.google.com/view/shbaek/home).

This repository should be understood as the maintained, CGH-focused evolution of
that core: a leaner and more native stack for holography workflows, device-aware
display modeling, and reproducible experimentation.

The name `PADO` also comes from the Korean word
[`파도`](https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%8F%84), meaning `wave`. It
points both to the optical waves we simulate and optimize, and to the broader
community we want to build around them.

## What This Documentation Covers

This MkDocs site is the high-level, self-contained guide to the repository. It
is designed to answer questions such as:

- What is `PADO Hologram` trying to be?
- How is it different from upstream `PADO`?
- Which modules exist today, and what do they do?
- How do I run a first holography experiment?
- Where do `display.py`, `pado_hologram`, Hydra, and the core optics API fit together?

For detailed API reference, the repository also keeps a Sphinx documentation
tree. This MkDocs site focuses instead on architecture, workflows, repository
direction, and contributor-facing context.

## Current Scope

Today, `PADO Hologram` already includes:

- phase-only hologram generation with compact Gerchberg-Saxton helpers
- double-phase amplitude coding (DPAC)
- device-aware LCOS/SLM encoding through `pado.display`
- single-plane and multi-plane holography pipelines
- Hydra-based experiment entry points
- a documented path for extending the framework on top of the smaller `pado` optics core

## Suggested Reading Order

If you are new to the repository, the best path is:

1. Read [Vision](community/vision.md) to understand the long-term direction.
2. Follow [Installation](guide/installation.md) and [Quickstart](guide/quickstart.md).
3. Read [Architecture Overview](concepts/architecture.md) and [Core vs Hologram](concepts/core-vs-hologram.md).
4. Continue to the workflow guides under [Workflows](workflows/phase-only-cgh.md).
5. Use [Module Reference](reference/modules.md) as a roadmap to the current codebase.

## Who This Is For

This repository is intended as a shared home for people working on holography
and computational imaging from different backgrounds, including:

- computer science
- electrical engineering
- optics and photonics
- physics
- perception science
- adjacent areas that care about display, imaging, and wave-based computation

The aim is to move beyond fragmented one-off codebases and toward a maintained,
reusable, and well-documented stack for differentiable holography.

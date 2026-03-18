---
hide:
  - toc
---

<section class="pado-hero" markdown="1">

<div class="pado-kicker">PADO Hologram</div>

# Differentiable Holography, Documented as a Framework

`PADO Hologram` is an open-source computer-generated holography framework built on top of
[`PADO`](https://github.com/shwbaek/pado), the differentiable optics core originally developed by the
[`POSTECH Computer Graphics Lab`](https://sites.google.com/view/shbaek/home).

This site is the high-level guide to the repository: what it is trying to become, how it is organized,
which workflows already exist, and where contributors can help grow it next.

[Get Started](guide/installation.md){ .md-button .md-button--primary }
[Read the Vision](community/vision.md){ .md-button }

</section>

`PADO` also draws from the Korean word [`파도`](https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%8F%84), meaning
`wave`. The repository uses that name both literally, for the optical waves it simulates, and socially,
for the momentum of a research community building reusable holography tools together.

## What You Will Find Here

This MkDocs site is meant to be self-contained and practical. It focuses on:

<div class="grid cards" markdown="1">

- **Repository direction**

  ---

  Why `PADO Hologram` exists, how it relates to upstream `PADO`, and what kind of long-term research stack it aims to become.

- **Architecture and boundaries**

  ---

  Where the optics core ends, where `pado_hologram` begins, and why that split matters for maintainability.

- **Workflow guides**

  ---

  Phase-only CGH, display-aware encoding, multi-plane composition, and Hydra-based experiment entry points.

- **Reference maps**

  ---

  A human-readable map of the main modules before you dive into the code or the deeper Sphinx API reference.

</div>

## Current Capabilities

<div class="grid cards" markdown="1">

- **Phase-only baselines**

  ---

  Compact Gerchberg-Saxton helpers and pipeline composition for first-pass hologram generation.

- **DPAC support**

  ---

  Double-phase amplitude coding as part of the growing higher-level CGH algorithm layer.

- **Device-aware encoding**

  ---

  `pado.display` bridges ideal phase outputs to LUT-based LCOS/SLM behavior and quantized realization.

- **Experiment orchestration**

  ---

  Hydra-backed runs for reproducibility without pushing configuration complexity into the lower-level optics core.

</div>

## Recommended Reading Path

1. Start with [Vision](community/vision.md) for the repository’s long-term direction.
2. Move to [Installation](guide/installation.md) and [Quickstart](guide/quickstart.md) for a first working run.
3. Read [Architecture Overview](concepts/architecture.md) and [Core vs Hologram](concepts/core-vs-hologram.md).
4. Continue with [Phase-Only CGH](workflows/phase-only-cgh.md), [Device Modeling](workflows/device-modeling.md), and [Experiments and Hydra](workflows/experiments.md).
5. Use [Module Reference Map](reference/modules.md) when you want to jump into the codebase.

## Audience

`PADO Hologram` is intended as a shared home for people working on holography and computational imaging from backgrounds such as computer science, electrical engineering, optics, physics, perception science, and neighboring areas that care about wave-based computation and display.

<div class="pado-mini-note" markdown="1">
The goal is to move beyond fragmented one-off codebases and toward a maintained, reusable, and well-documented stack for [`differentiable holography`](https://github.com/cinescope-wkr/awesome-neural-holography).
</div>

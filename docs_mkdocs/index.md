---
hide:
  - toc
---

<section class="pado-hero" markdown="1">

<div class="pado-kicker">PADO Hologram</div>

# Differentiable Holography, Documented as a Framework

<p class="pado-hero-lead">
An open-source computer-generated holography framework built on top of
<a href="https://github.com/shwbaek/pado">PADO</a>, a PyTorch differentiable optics library.
</p>

This site is the high-level guide to the repository: what it is trying to become, how it is organized,
which workflows already exist, and where contributors can help grow it next. The underlying PADO optics
core was originally developed by the <a href="https://sites.google.com/view/shbaek/home">POSTECH Computer Graphics Lab</a>.

[Get Started](guide/installation.md){ .md-button .md-button--primary }
[Read the Vision](community/vision.md){ .md-button }
[Find by Task](#find-what-you-need){ .md-button }

</section>

`PADO` can also be read as [`PADO (파도)`](https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%8F%84), the Korean word
for `wave`. The repository uses that name both literally, for the optical waves it simulates, and socially,
for the momentum of a research community building reusable holography tools together.

## What You Will Find Here

This MkDocs site is meant to be self-contained and practical. It focuses on architecture, workflows,
reference maps, and contributor-facing context before you dive into the lower-level API details.

<div class="grid cards" markdown="1">

- [**Repository direction**](community/vision.md)

  ---

  Why `PADO Hologram` exists, how it relates to upstream `PADO`, and what kind of long-term research stack it aims to become.

- [**Architecture and boundaries**](concepts/architecture.md)

  ---

  Where the optics core ends, where `pado_hologram` begins, and why that split matters for maintainability.

- [**Workflow guides**](workflows/phase-only-cgh.md)

  ---

  Phase-only CGH, display-aware encoding, multi-plane composition, and Hydra-based experiment entry points.

- [**Reference maps**](reference/modules.md)

  ---

  A human-readable map of the main modules before you dive into the code or the deeper Sphinx API reference.

</div>

## Find What You Need

If you are here for something specific, start from one of these pages:

<div class="grid cards" markdown="1">

- [**I want to install and run something quickly**](guide/quickstart.md)

  ---

  Start with installation, imports, and the smallest CLI-backed or Hydra-compatible experiments.

- [**I want to understand the repository structure**](guide/repository-layout.md)

  ---

  See how `pado`, `pado_hologram`, MkDocs, Sphinx, tests, and notebooks fit together.

- [**I want to understand the architecture choices**](concepts/core-vs-hologram.md)

  ---

  Learn why the optics core and the holography layer are intentionally kept separate.

- [**I want the experiment entry points**](workflows/experiments.md)

  ---

  Find CLI commands, Hydra compatibility, backend choices, and the current workflow scope.

- [**I want device-aware SLM details**](workflows/device-modeling.md)

  ---

  Go straight to LCOS/LUT-based phase encoding and the current display-model bridge.

- [**I want to help build this project**](community/contributing.md)

  ---

  Jump to contribution priorities, scope, and the broader project direction.

</div>

## Current Capabilities

<div class="grid cards" markdown="1">

- [**Phase-only baselines**](workflows/phase-only-cgh.md)

  ---

  Compact Gerchberg-Saxton helpers and pipeline composition for first-pass hologram generation.

- [**DPAC support**](workflows/phase-only-cgh.md)

  ---

  Double-phase amplitude coding as part of the growing higher-level CGH algorithm layer.

- [**Device-aware encoding**](workflows/device-modeling.md)

  ---

  `pado.display` remains a compatibility bridge for LUT-based LCOS/SLM behavior while higher-level wrappers live in `pado_hologram`.

- [**Experiment orchestration**](workflows/experiments.md)

  ---

  CLI-backed and Hydra-compatible runs for reproducibility without pushing configuration complexity into the lower-level optics core.

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
The goal is to move beyond fragmented one-off codebases and toward a maintained, reusable, and well-documented stack for [`differentiable holography`](https://github.com/cinescope-wkr/awesome-differentiable-holography).
</div>

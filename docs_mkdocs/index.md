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

<div class="pado-hero-actions">
  <a class="md-button md-button--primary" href="guide/installation/">Get Started</a>
  <a class="md-button" href="community/vision/">Read the Vision</a>
  <a class="md-button" href="#find-what-you-need">Find by Task</a>
</div>

</section>

`PADO` can also be read as [`PADO (파도)`](https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%8F%84), the Korean word
for `wave`. The repository uses that name both literally, for the optical waves it simulates, and socially,
for the momentum of a research community building reusable holography tools together.

## What You Will Find Here

This MkDocs site is meant to be self-contained and practical. It focuses on architecture, workflows,
reference maps, and contributor-facing context before you dive into the lower-level API details.

<div class="grid cards pado-card-grid">
<ul>
  <li>
    <p class="pado-card-title"><a href="community/vision/"><strong>Repository direction</strong></a></p>
    <p>Why <code>PADO Hologram</code> exists, how it relates to upstream <code>PADO</code>, and what kind of long-term research stack it aims to become.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="concepts/architecture/"><strong>Architecture and boundaries</strong></a></p>
    <p>Where the optics core ends, where <code>pado_hologram</code> begins, and why that split matters for maintainability.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="workflows/phase-only-cgh/"><strong>Workflow guides</strong></a></p>
    <p>Phase-only CGH, display-aware encoding, multi-plane composition, and Hydra-based experiment entry points.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="reference/modules/"><strong>Reference maps</strong></a></p>
    <p>A human-readable map of the main modules before you dive into the code or the deeper Sphinx API reference.</p>
  </li>
</ul>
</div>

## Find What You Need

If you are here for something specific, start from one of these pages:

<div class="grid cards pado-card-grid">
<ul>
  <li>
    <p class="pado-card-title"><a href="guide/quickstart/"><strong>I want to install and run something quickly</strong></a></p>
    <p>Start with installation, imports, and the smallest CLI-backed or Hydra-compatible experiments.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="guide/repository-layout/"><strong>I want to understand the repository structure</strong></a></p>
    <p>See how <code>pado</code>, <code>pado_hologram</code>, MkDocs, Sphinx, tests, and notebooks fit together.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="concepts/core-vs-hologram/"><strong>I want to understand the architecture choices</strong></a></p>
    <p>Learn why the optics core and the holography layer are intentionally kept separate.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="workflows/experiments/"><strong>I want the experiment entry points</strong></a></p>
    <p>Find CLI commands, Hydra compatibility, backend choices, and the current workflow scope.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="workflows/device-modeling/"><strong>I want device-aware SLM details</strong></a></p>
    <p>Go straight to LCOS/LUT-based phase encoding and the current display-model bridge.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="community/contributing/"><strong>I want to help build this project</strong></a></p>
    <p>Jump to contribution priorities, scope, and the broader project direction.</p>
  </li>
</ul>
</div>

## Current Capabilities

<div class="grid cards pado-card-grid">
<ul>
  <li>
    <p class="pado-card-title"><a href="workflows/phase-only-cgh/"><strong>Phase-only baselines</strong></a></p>
    <p>Compact Gerchberg-Saxton helpers and pipeline composition for first-pass hologram generation.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="workflows/phase-only-cgh/"><strong>DPAC support</strong></a></p>
    <p>Double-phase amplitude coding as part of the growing higher-level CGH algorithm layer.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="workflows/device-modeling/"><strong>Device-aware encoding</strong></a></p>
    <p><code>pado.display</code> remains a compatibility bridge for LUT-based LCOS/SLM behavior while higher-level wrappers live in <code>pado_hologram</code>.</p>
  </li>
  <li>
    <p class="pado-card-title"><a href="workflows/experiments/"><strong>Experiment orchestration</strong></a></p>
    <p>CLI-backed and Hydra-compatible runs for reproducibility without pushing configuration complexity into the lower-level optics core.</p>
  </li>
</ul>
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

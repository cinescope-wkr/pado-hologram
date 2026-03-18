# Phase-Only CGH Workflow

This page explains the current phase-only workflow at a conceptual level.

## What the Workflow Tries to Solve

Phase-only holography usually starts with a desired intensity pattern at one or
more observation planes. The system then searches for a phase pattern at the
SLM plane whose propagated field produces that target as closely as possible.

## Current Building Blocks

Today, the repository exposes the following pieces for this workflow:

- `IntensityTarget` for specifying the reconstruction objective
- `GerchbergSaxtonPhaseOptimizer` for a compact iterative phase-only solver
- `HologramPipeline` for end-to-end source → SLM → propagation → evaluation

## Conceptual Loop

The current Gerchberg-Saxton path can be read as:

1. start from an initial phase
2. propagate to the target plane
3. replace the amplitude with the target amplitude
4. backpropagate to the source plane
5. keep the phase and repeat

This is not yet the entire story of modern inverse design, but it is a solid
and understandable baseline.

## Why This Matters

This workflow is one of the core reasons `pado_hologram` exists. It is where the
repository starts moving beyond a pure optics library and toward a reusable CGH
framework.

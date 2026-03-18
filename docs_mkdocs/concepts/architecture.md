# Architecture Overview

`PADO Hologram` is structured as a layered repository rather than a monolithic
framework.

!!! abstract

    Think of the repository as two cooperating layers: a smaller wave-optics engine and a higher-level CGH framework built on top of it.

## The Main Split

- `pado`: the optics core
- `pado_hologram`: the higher-level holography layer

This split matters because holography workflows often need more than wave
propagation alone. They need targets, optimization loops, device models,
experiment orchestration, and reproducibility tooling.

## Why Not Put Everything Into `pado`?

If the entire CGH stack were pushed into the core package, the result would be a
larger and less stable optics library. The current structure tries to preserve
the strengths of the original core while giving holography-specific logic room
to grow.

That means:

- the optics layer stays smaller and easier to reason about
- holography features can expand without forcing every optics user to adopt them
- contributors can work on upper-layer workflows without destabilizing the lower layer

## Relation to Earlier Research Stacks

The repository direction is informed by the kinds of needs that older projects
such as [`holotorch`](https://github.com/facebookresearch/holotorch) made clear:

- higher-level holography workflows matter
- experiment composition matters
- device-aware abstractions matter

The goal here is not to clone those stacks, but to rebuild the useful ideas in a
smaller, clearer, and more maintainable form on top of `PADO`.

## Optional [`NVIDIA Warp`](https://github.com/NVIDIA/warp) Layer

`PADO Hologram` also now includes an optional [`NVIDIA Warp`](https://github.com/NVIDIA/warp) integration.

The intended role of Warp here is specific:

- not to replace every propagation path
- not to force the optics core away from PyTorch
- to provide a maintainable home for future custom holography kernels

The first integration point is the DPAC checkerboard kernel path. That makes the
current Warp layer meaningful without overstating what has already been moved.

## Current End-to-End Path

At a high level, a typical workflow looks like this:

1. define a source plane through `SourceSpec`
2. define propagation through `PropagationSpec`
3. define a reconstruction target
4. optimize or encode a phase pattern
5. apply optional LCOS/SLM modeling
6. propagate to one or more observation planes
7. evaluate intensity or amplitude metrics

This is the conceptual backbone of the current `pado_hologram` package.

# Core vs Hologram

This page clarifies the difference between `pado` and `pado_hologram`.

## `pado`: the Optics Core

The `pado` package is primarily about differentiable wave optics. It is the
layer you use when you want direct access to:

- complex light fields
- propagation models
- optical elements
- materials and low-level utilities

It is a useful package even outside of CGH-specific workflows.

## `pado_hologram`: the Framework Layer

The `pado_hologram` package exists because holography is not only about forward
optics simulation. In practice, CGH work also requires:

- targets
- loss functions
- phase optimization routines
- display/SLM-aware encoding
- multi-plane orchestration
- reproducible experiment configs

Those concerns belong to a different layer than the raw optics core.

## `display.py` as the Bridge

The file `pado.display` is an important transitional layer in the current
repository. It helps connect:

- ideal phase targets from optimization
- LUT-based display encoding
- realized phase and amplitude at the SLM plane

In other words, it begins to move the repository from idealized optics-only
simulation toward device-aware holography workflows.

## Practical Rule of Thumb

Use:

- `pado` when you want direct optics primitives
- `pado_hologram` when you want CGH-oriented workflows and abstractions

This distinction is central to the current repository identity.

# Contributing to PADO Hologram

Contributions are welcome.

`PADO Hologram` is being built as a maintained, modular holography stack on top of
the original `pado` optics core, and outside perspectives are genuinely useful.
If you have worked with related research frameworks such as
[`holotorch`](https://github.com/facebookresearch/holotorch), your experience with
CGH workflows, SLM/device abstractions, experiment orchestration, or lab-facing
tooling is especially welcome.

## Good areas to contribute

- hologram-generation algorithms such as DPAC, GS variants, and multi-plane methods
- SLM and display models, measured LUT support, and hardware-aware abstractions
- reproducible experiment tooling, including Hydra configs and evaluation workflows
- tests, examples, and documentation that make the stack easier to maintain

## Ground rules

- keep the `pado` core small and broadly useful
- put higher-level CGH workflow logic in `pado_hologram`
- prefer clear APIs and tests over one-off notebook logic
- do not copy non-MIT code into this repository

## How to help

- open an issue describing a bug, idea, or missing abstraction
- open a pull request with focused changes and matching tests/docs
- improve examples or docs if you find a confusing part of the current design

Thanks for helping build a cleaner long-term home for differentiable holography work.

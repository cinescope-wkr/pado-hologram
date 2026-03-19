<div align="center">
  <img src="docs/images/banner_1.0.0.png" width="100%">
</div>

<h1 align="center">PADO Hologram</h1>
<h3 align="center">An open-source computer-generated holography framework built on top of PADO, a PyTorch differentiable optics library.</h3>

<p align="center">
  <a href="https://cinescope-wkr.github.io/pado-hologram/">
    <img src="https://img.shields.io/badge/Documentation-online-blue" alt="Documentation">
  </a>
  <a href="https://github.com/cinescope-wkr/pado-hologram/actions/workflows/deploy-docs.yml">
    <img src="https://github.com/cinescope-wkr/pado-hologram/actions/workflows/deploy-docs.yml/badge.svg" alt="Docs Deploy">
  </a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#nvidia-warp-backend">NVIDIA Warp Backend</a> •
  <a href="#pado-hologram">PADO Hologram</a> •
  <a href="#pado-core-api">PADO Core API</a> •
  <a href="#examples">Examples</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#repository-updates">Repository Updates</a> •
  <a href="#fork-notice">Fork Notice</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

`PADO Hologram` is the CGH-focused evolution of the [`PADO`](https://github.com/shwbaek/pado)
differentiable optics core. Rebuilt as a lean, native stack, it picks up where
earlier frameworks such as [`Holotorch`](https://github.com/facebookresearch/holotorch)
left off, with a stronger emphasis on long-term maintainability, clarity, and performance.

The underlying optics engine still lives in the `pado` package for compatibility,
but this repository should be understood as `PADO Hologram` in its current
maintained form.

This repository state is maintained by Jinwoo Lee (`cinescope@kaist.ac.kr`) and
is being shaped into a clearer, longer-lived home for
[`neural holography`](https://github.com/cinescope-wkr/awesome-differentiable-holography).

The name `PADO` can also be read as [`PADO (파도)`](https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%8F%84), the Korean word for `wave`.
It reflects both the physical waves we manipulate and the collective momentum of
the researchers who work on them.

We want this repository to be a unified home for
[`neural holography`](https://github.com/cinescope-wkr/awesome-differentiable-holography):
a place where physicists, computer scientists, electrical engineers, optical engineers,
perception researchers, and curious builders can move beyond fragmented one-off
efforts and build together.

Let’s surf this [`PADO (파도)`](https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%8F%84) together.

---

## Getting Started

For this maintained repository state, the recommended path is to work from source:

```bash
git clone https://github.com/cinescope-wkr/pado-hologram.git
cd pado-hologram
pip install -e .
```

Optional [`NVIDIA Warp`](https://github.com/NVIDIA/warp) support for custom holography kernels:

```bash
pip install -e ".[warp]"
```

This repository keeps the core optics import path as `pado` and also exposes the
new holography-layer scaffold as `pado_hologram`:

```python
import pado
import pado_hologram
```

The first device-aware holography helper is still available through
`pado.display` as a compatibility bridge:

```python
from pado.display import LCOSLUT, lcos_encode_phase, slm_light_from_phase
```

The higher-level holography layer now also provides configuration and orchestration
entry points:

```python
from pado_hologram import SourceSpec, PropagationSpec, HologramPipeline
```

For reproducible runs, the package also exposes a small package banner and a CLI
entry point:

```bash
python -m pado_hologram
pado-hologram doctor --run-smoke
pado-hologram run experiment=gs
pado-hologram run experiment=dpac target=gaussian
pado-hologram run experiment=primitive_gaussian_gws_exact primitives=gaussian3d_depth_ring
```

More detailed CLI and Hydra examples live in the documentation quickstart and
experiment workflow pages.

## Documentation

The documentation is now organized around the `PADO Hologram` repository identity
while still exposing the `pado` core API:

- landing page and architecture direction
- installation and package layout
- `PADO` core API reference
- notebook examples
- repository-maintained updates

The main documentation site lives at:
<https://cinescope-wkr.github.io/pado-hologram/>

## Architecture

The intended role split is:

- `PADO`: differentiable optics core for light fields, optical elements, propagation, materials, and numerical helpers
- `PADO Hologram`: CGH workflows, SLM/display abstractions, optimization utilities, and future experiment orchestration

The current starting point for that upper layer already exists in this repository:

- `pado.display` remains available as a compatibility bridge for LCOS/SLM-oriented LUT encoding and phase-to-field conversion
- `pado_hologram` now exposes concrete modules for configs, device encoding, targets, losses, multi-plane orchestration, DPAC/GS algorithms, and Hydra-based experiment entry points
- `pado_hologram.backends` now exposes an optional [`NVIDIA Warp`](https://github.com/NVIDIA/warp) path for custom holography kernels without forcing a full propagation rewrite
- the documentation now treats this direction as the main repository identity instead of a side note

## NVIDIA Warp Backend

`PADO Hologram` now includes an optional [`NVIDIA Warp`](https://github.com/NVIDIA/warp)
integration for custom holography kernels.

The significance of this integration is deliberate:

- it is **not** presented as a full replacement for `pado` propagation
- it is **not** a claim that the entire optics stack has moved away from PyTorch
- it **is** the start of a more maintainable custom-kernel layer for future holography research

In other words, Warp matters here less as a blanket “make everything faster”
story and more as a safer foundation for the kinds of bespoke kernels that CGH
research often accumulates over time.

Current scope:

- optional backend selection through `backend=torch|warp|auto`
- custom-kernel paths in DPAC, primitive-scene splat renderers, and exact primitive-based GWS/RPWS renderers
- safe Warp cache handling when `WARP_CACHE_DIR` is not already set
- PyTorch propagation remains the default core path

## PADO Hologram

`PADO Hologram` is where we want the repository to grow:

- computer-generated holography workflows
- SLM-aware phase optimization
- device-oriented display encoding
- future setup orchestration and hardware-aware layers

This lets us keep `pado` as a compact optics core while clearly separating the
larger holography stack from the lower-level simulation primitives.

It also reflects a broader goal: not just a code release, but a place where the
field can gather around common abstractions, reusable tools, and a healthier
open research culture.

The first robust module set is already in place:

| Module | Role |
| --- | --- |
| `pado_hologram.config` | Source and propagation specifications |
| `pado_hologram.backends` | Optional custom-kernel backends such as [`NVIDIA Warp`](https://github.com/NVIDIA/warp) |
| `pado_hologram.slm` | Phase-only LCOS/SLM encoding |
| `pado_hologram.targets` | Reconstruction targets |
| `pado_hologram.losses` | Intensity/amplitude losses and metrics |
| `pado_hologram.pipeline` | Source -> SLM -> propagation -> evaluation orchestration |
| `pado_hologram.algorithms` | Compact hologram-generation algorithms such as Gerchberg-Saxton and DPAC |
| `pado_hologram.primitive_based` | gsplat-free primitive-scene Gaussian renderers, including a vectorized splat path with optional Warp backend selection and preset/JSON scene ingestion |
| `pado_hologram.devices` | SLM support plus optional camera/observation transforms for primitive and future capture-aware experiments |
| `pado_hologram.primitive_based` (`gaussian_wave`) | Depth-aware baseline that forms local Gaussian envelopes on per-depth object planes, groups them by depth, and propagates them coherently to the hologram plane with PADO propagation |
| `pado_hologram.primitive_based` (`gaussian_wave_awb`) | hsplat-inspired path with opacity, depth ordering, and alpha wave blending style transmittance accumulation |
| `pado_hologram.primitive_based` (exact RPWS) | Exact GWS-backed RPWS baseline with structured random phase and time-multiplexed intensity averaging |
| `pado_hologram.experiments`, `pado_hologram.hydra_app` | CLI- and Hydra-friendly experiment execution |
| `pado_hologram.representations` | Future primitive-based CGH scene representations |
| `pado_hologram.neural` | Future capture, calibration, and learned holography workflows |

## PADO Core API

The `pado` package remains the core simulation API and currently provides:

- `pado.light`
- `pado.optical_element`
- `pado.propagator`
- `pado.material`
- `pado.math`

The API reference in the documentation should be read as the core foundation that
`PADO Hologram` is built on top of.

## Compatibility Bridge

The module `pado.display` remains available in the `pado` namespace as a
compatibility bridge for device-aware LCOS/SLM phase encoding. It is documented
as a transitional layer rather than as part of the long-term `PADO` core
identity.

For new holography workflows, prefer the higher-level `pado_hologram` device
and SLM abstractions built on top of that bridge.

## Examples

This repository already contains strong CGH-oriented examples inside the original
PADO notebook set. The most relevant starting points for the hologram direction are:

- `example/2_Computer_Generated_Holography/2.1_DPAC.ipynb`
- `example/2_Computer_Generated_Holography/2.2_multi_depth_cgh.ipynb`
- `example/2_Computer_Generated_Holography/2.3_cgh_optimization_gs_sgd_adam.ipynb`
- `example/2_Computer_Generated_Holography/2.5_cgh_optimization_with_phase_only_slm.ipynb`
- `example/2_Computer_Generated_Holography/2.6_multi_depth_hologram_generation_using_adam_with_phase_only_slm.ipynb`

The broader optics examples remain valuable as the core layer beneath `PADO Hologram`.

The new `pado_hologram` package is intended to become the code-level counterpart
to these notebooks: a reusable orchestration layer rather than notebook-only logic.

## Contributing

Contributors are welcome.

If you have experience with related holography frameworks such as
[`Holotorch`](https://github.com/facebookresearch/holotorch), your perspective is
especially valuable here. The goal is not to clone that project, but to learn from
what it enabled and rebuild the most useful workflow ideas in a smaller,
maintainable, PADO-native form.

More broadly, this project is intended as a meeting point for the holography and
computational imaging community across disciplines. Whether your background is
computer science, electrical engineering, optics, physics, psychology, perception science, or something adjacent,
you are welcome here if you care about this field and want to help it grow.

The long-term vision is to move past fragmented efforts and build something that
helps people collaborate, learn from one another, and inspire one another over time.

Good areas to help with:

- new hologram-generation algorithms and multi-plane methods
- SLM and display models, measured LUT support, and hardware-facing abstractions
- Hydra configs, experiments, tests, and documentation

See the contributor guide:
<https://cinescope-wkr.github.io/pado-hologram/community/contributing/>

## Repository Updates

> [!NOTE]
> The original PADO framework is developed and maintained by the [POSTECH Computer Graphics Lab](https://sites.google.com/view/shbaek/home).
> This forked repository state is maintained by Jinwoo Lee, and the items below
> should be described as fork-specific or repository-maintained contributions.

**Our contributions in this repository update**:

- Added `pado.display` for LCOS/SLM-oriented phase encoding workflows with `LCOSLUT`, `lcos_encode_phase`, and `slm_light_from_phase`.
- Added robust `pado_hologram` modules for configuration, SLM/device handling, targets, losses, single-plane and multi-plane pipelines, DPAC, and Gerchberg-Saxton optimization.
- Added primitive-based Gaussian renderer baselines, exact GWS-style paths, and an exact GWS-backed RPWS baseline with structured random phase and time-multiplexed intensity averaging.
- Added an optional [`NVIDIA Warp`](https://github.com/NVIDIA/warp) backend layer for custom holography kernels across DPAC and primitive-based renderer paths rather than a full propagation rewrite.
- Added a CLI- and Hydra-friendly experiment layer and packaged config tree for reproducible holography runs.
- Stabilized core tensor-shape handling across `Light`, `OpticalElement`, `SLM`, and polarization-aware paths.
- Expanded regression and parity tests to cover the display bridge, primitive-based exact paths, and recently fixed stability issues.
- Reframed the README and Sphinx documentation around the `PADO Hologram` repository identity.

**Stability fixes in this update**:

<details>
<summary>Show stability fixes</summary>

- Fixed `pad()` dimension bookkeeping so metadata stays aligned with the underlying tensor shape.
- Fixed complex-field resize and magnification paths used by `Light` and `OpticalElement`.
- Fixed LCOS phase wrapping so LUTs using `[0, 2π]` and `[-π, π]` conventions both encode target phase sanely.
- Fixed `Light.load_image(..., random_phase=True, batch_idx=...)` for single-batch random phase injection.
- Fixed `PolarizedLight.clone()`, `PolarizedLight.crop()`, and `PolarizedLight.magnify()`.
- Fixed `SLM.set_lens()` to work with the current wavelength-managed setter path.
- Fixed multi-channel `calculate_ssim()` support.
- Fixed `PolarizedSLM` so polarization-specific amplitude and phase state are tracked consistently.

</details>

## Fork Notice

> [!NOTE]
> This repository is maintained as `PADO Hologram`, a forked holography-oriented
> extension of the original PADO project.
> The original PADO framework is developed by the [POSTECH Computer Graphics Lab](https://sites.google.com/view/shbaek/home).
>
> **Fork maintainer**: Jinwoo Lee  
> **Maintainer contact**: cinescope@kaist.ac.kr
>
> **Naming note**
> - Project and repository identity: `PADO Hologram`
> - Core optics package kept for compatibility: `pado`
> - Higher-level holography namespace reserved in this repository: `pado_hologram`

## License

This repository remains under the MIT License. The maintained fork state is
currently coordinated by Jinwoo Lee (`cinescope@kaist.ac.kr`), while the legal
license notice remains the one preserved in [LICENSE](./LICENSE).

## Citation

If you use this repository, cite the original PADO work as the core optics foundation.
If your work specifically depends on the maintained holography layer introduced in
this repository state, cite `PADO Hologram` as well.

```bib
@misc{Pado,
   Author = {Seung-Hwan Baek, Dong-Ha Shin, Yujin Jeon, Seung-Woo Yoon, Eunsue Choi, Gawoon Ban, Hyunmo Kang},
   Year = {2025},
   Note = {https://github.com/shwbaek/pado},
   Title = {Pado: Pytorch Automatic Differentiable Optics}
}
```

```bib
@software{lee2026padohologram,
   author = {Jinwoo Lee},
   title = {PADO Hologram: An open-source computer-generated holography framework built on top of PADO, a PyTorch differentiable optics library.},
   year = {2026},
   note = {https://github.com/cinescope-wkr/pado-hologram},
   abstract = {Repository-maintained holography-oriented layer built on top of PADO, including DPAC, multi-plane workflows, and Hydra-based experiments}
}
```

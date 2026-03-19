# Installation

This page explains how to install and verify `PADO Hologram` from source.

!!! note

    The repository is still primarily documented as a source install, but it is now packaged cleanly enough to expose a small `pado-hologram` CLI after installation.

## Recommended Setup

The repository is currently intended to be used from source:

```bash
git clone https://github.com/cinescope-wkr/pado-hologram.git
cd pado-hologram
pip install -e .
```

This installs:

- the original optics core under the import path `pado`
- the higher-level holography package under the import path `pado_hologram`

After this editable install, the `pado-hologram` console command should also be
available. If you are working before installation or in an environment where
the console script is not on `PATH`, `python -m pado_hologram` remains the most
portable entry point.

## Optional [`NVIDIA Warp`](https://github.com/NVIDIA/warp) Support

If you want to enable the experimental Warp-backed custom-kernel path:

```bash
pip install -e ".[warp]"
```

This optional dependency is currently intended for `pado_hologram` custom
holography kernels. It is not presented as a full replacement for the PyTorch
propagation stack in `pado`.

## Documentation Dependencies

If you want to build the local documentation site:

```bash
pip install -r requirements/requirements-docs.txt
```

Then:

```bash
mkdocs serve
```

or:

```bash
mkdocs build --clean --strict
```

## Optional Verification

Minimal import check:

```python
import pado
import pado_hologram
```

Minimal package and experiment check:

```bash
python -m pado_hologram
pado-hologram doctor --run-smoke
pado-hologram run experiment=gs
pado-hologram run experiment=dpac target=gaussian
pado-hologram run experiment=primitive_gaussian_gws_exact primitives=gaussian3d_depth_ring
pado-hologram run experiment=primitive_gaussian_rpws primitives=gaussian3d_depth_ring
```

!!! tip

    `pip` does not provide a stable, cross-backend post-install hook for showing a friendly banner. The recommended package-level welcome path is therefore `python -m pado_hologram` or `pado-hologram`, which prints a lightweight ASCII banner and environment summary.

For a fuller command list, including primitive-scene presets, RPWS, camera
overrides, and the Hydra-native compatibility path, continue to
[Quickstart](quickstart.md) and [Experiments](../workflows/experiments.md).

## Package Layout

After installation, the repository should be understood as having two layers:

- `pado`: the differentiable optics core
- `pado_hologram`: the higher-level CGH framework layer

Within `pado_hologram`, the currently documented upper-layer surface includes:

- compact phase-only baselines such as GS and DPAC
- primitive-scene Gaussian baselines and splat renderers
- exact primitive-based GWS and RPWS experiment paths
- optional Warp-backed custom-kernel paths for DPAC and primitive renderers

The `pado` package path is intentionally retained for compatibility with the
original upstream structure.

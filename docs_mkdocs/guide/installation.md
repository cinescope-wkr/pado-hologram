# Installation

This page explains how to install and verify `PADO Hologram` from source.

!!! note

    The repository is currently designed to be used from source rather than as a pre-packaged release.

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

## Optional [`NVIDIA Warp`](https://github.com/NVIDIA/warp) Support

If you want to enable the experimental Warp-backed custom-kernel path:

```bash
pip install -r requirements-extra.txt
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

Minimal experiment check:

```bash
python -m pado_hologram.hydra_app experiment=gs
python -m pado_hologram.hydra_app experiment=dpac target=gaussian
```

!!! tip

    If you only want to verify that the framework imports and the smallest holography paths run, these two Hydra commands are enough for a quick sanity check.

## Package Layout

After installation, the repository should be understood as having two layers:

- `pado`: the differentiable optics core
- `pado_hologram`: the higher-level CGH framework layer

The `pado` package path is intentionally retained for compatibility with the
original upstream structure.

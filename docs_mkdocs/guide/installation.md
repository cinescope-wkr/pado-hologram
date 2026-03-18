# Installation

This page explains how to install and verify `PADO Hologram` from source.

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

## Package Layout

After installation, the repository should be understood as having two layers:

- `pado`: the differentiable optics core
- `pado_hologram`: the higher-level CGH framework layer

The `pado` package path is intentionally retained for compatibility with the
original upstream structure.

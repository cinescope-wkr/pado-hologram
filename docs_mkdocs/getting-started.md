# Getting Started

The recommended path is to work from source:

```bash
git clone https://github.com/cinescope-wkr/pado-hologram.git
cd pado-hologram
pip install -e .
```

Core imports:

```python
import pado
import pado_hologram
```

Hydra experiment examples:

```bash
python -m pado_hologram.hydra_app experiment=gs
python -m pado_hologram.hydra_app experiment=dpac target=gaussian
```

For the fuller API and notebook-oriented docs, see the Sphinx documentation under `docs/source`.

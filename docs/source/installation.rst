Installation
============

This documentation describes the maintained ``PADO Hologram`` repository state.
The recommended installation path is to work from source so that both the core
``pado`` package and the higher-level ``pado_hologram`` scaffold are available.

Recommended Development Install
-------------------------------

.. code-block:: bash

   git clone https://github.com/cinescope-wkr/pado-hologram.git
   cd pado-hologram
   pip install -e .

Package Layout
--------------

- ``pado``: the core differentiable optics package
- ``pado_hologram``: the higher-level holography layer
- ``pado.display``: the current device-aware bridge for LCOS/SLM encoding workflows
- ``pado_hologram.pipeline`` and related modules: orchestration built on top of the core package
- ``pado_hologram.conf``: Hydra config package for reproducible experiments

Minimal Import Check
--------------------

.. code-block:: python

   import pado
   import pado_hologram

Core Dependencies
-----------------

- Python 3.9 or higher
- PyTorch 1.10.0 or higher
- NumPy 1.16.0 or higher
- Matplotlib 3.3.0 or higher
- SciPy 1.0.0 or higher

Next Steps
----------

- Read :doc:`pado_hologram` for the repository architecture direction
- Explore :doc:`api/index` for the current ``pado`` core API
- Use :doc:`examples/index` to start from CGH-oriented notebooks

Compatibility Note
------------------

The original upstream package and citation identity remain ``PADO``.
This repository is documented as ``PADO Hologram`` to reflect its maintained,
holography-oriented direction.

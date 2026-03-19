PADO Hologram Repository Updates
================================

This page documents the additions and maintenance updates that are explicitly
introduced in this repository state.

.. note::

   The original PADO framework is developed by the `POSTECH Computer Graphics Lab <https://sites.google.com/view/shbaek/home>`_.
   This repository state is a forked, repository-maintained, holography-oriented
   version of PADO.
   Fork maintainer: Jinwoo Lee (``cinescope@kaist.ac.kr``).
   The items below are repository-maintained contributions layered on top of the
   core framework and should be described as such.

Our Contributions
-----------------

- Added ``pado.display`` for LCOS/SLM-oriented phase encoding workflows.
- Added robust ``pado_hologram`` modules for configuration, SLM/device modeling, targets, losses, single-plane and multi-plane pipelines, DPAC, and Gerchberg-Saxton optimization.
- Added primitive-based Gaussian renderer baselines, exact GWS-style paths, and an exact GWS-backed RPWS baseline with structured random phase and time-multiplexed intensity averaging.
- Added an optional `NVIDIA Warp <https://github.com/NVIDIA/warp>`_ backend layer for custom holography kernels across DPAC and primitive-based renderer paths rather than presenting Warp as a full propagation rewrite.
- Added a package CLI, a Hydra-compatible experiment layer, and a packaged config tree for reproducible holography runs.
- Added regression and parity tests that cover the display bridge, primitive-based exact paths, and recent stability fixes.
- Updated the README and Sphinx documentation so these additions are visible from the main navigation.

Stability Fixes
---------------

- Fixed dimension bookkeeping in ``Light.pad()`` and ``OpticalElement.pad()``.
- Fixed complex-field resize paths used by ``Light.magnify()`` and ``OpticalElement.resize()``.
- Fixed LCOS phase wrapping so LUTs using ``[0, 2π]`` and ``[-π, π]`` conventions both encode target phase sanely.
- Fixed batch-specific random phase injection in ``Light.load_image()``.
- Fixed ``PolarizedLight.clone()``, ``PolarizedLight.crop()``, and ``PolarizedLight.magnify()``.
- Fixed ``SLM.set_lens()`` so it works with the managed wavelength setter path.
- Fixed multi-channel ``calculate_ssim()`` support.
- Fixed ``PolarizedSLM`` state handling so polarization-specific amplitude and phase data remain consistent.

Documentation and API Coverage
------------------------------

- Added a dedicated API page for ``pado.display``.
- Added this update log to the main documentation navigation.
- Kept the public package surface aligned with the top-level ``pado`` import path.

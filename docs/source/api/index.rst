PADO Core API Reference
=======================

This section documents the current ``pado`` core API that underpins the
``PADO Hologram`` repository direction.

Core Components
---------------

.. toctree::
   :maxdepth: 2
   :caption: PADO Core Modules

   light
   optical_element
   propagator
   material
   math
   display

PADO Hologram Layer
-------------------

.. toctree::
   :maxdepth: 2
   :caption: PADO Hologram Modules

   pado_hologram

Module Overview
---------------

The ``pado`` package currently consists of the following main modules:

1. **Light Module** (`pado.light`)
   
   Handles light wavefronts and their properties:
   - Wavefront generation and manipulation
   - Polarization handling
   - Intensity and phase calculations

2. **Optical Elements** (`pado.optical_element`)
   
   Defines various optical components:
   - Lenses and mirrors
   - Diffractive elements
   - Apertures and stops
   - Custom optical elements

3. **Propagator** (`pado.propagator`)
   
   Manages light propagation:
   - Angular spectrum method
   - Fresnel propagation
   - Rayleigh-Sommerfeld diffraction
   - Custom propagation methods

4. **Materials** (`pado.material`)
   
   Defines optical properties of materials:
   - Refractive indices
   - Dispersion relations
   - Absorption coefficients

5. **Math Utilities** (`pado.math`)
   
   Provides mathematical tools:
   - Fourier transforms
   - Special functions
   - Numerical methods
   - Optimization utilities

6. **Display and Encoding** (`pado.display`)
   
   Provides the first device-aware bridge used by the holography layer:
   - LCOS lookup-table models
   - Quantized phase encoding
   - SLM-plane light generation from optimized phase

7. **PADO Hologram Layer** (`pado_hologram`)

   Provides the higher-level holography orchestration stack:
   - source and propagation specifications
   - phase-only LCOS/SLM device models
   - reconstruction targets and multi-plane evaluation
   - experiment pipelines, compact optimization algorithms, and Hydra-driven runs

PADO Hologram
=============

``PADO Hologram`` is the repository identity for this maintained fork state:
a holography-oriented layer built on top of the original ``pado`` optics core.

This direction is informed by the kinds of higher-level CGH needs that older
frameworks such as `holotorch <https://github.com/facebookresearch/holotorch>`_ tried to solve, but it is being rebuilt here as
a smaller and more maintainable PADO-native stack.

.. note::

   The original PADO framework is developed by the `POSTECH Computer Graphics Lab <https://sites.google.com/view/shbaek/home>`_.
   This repository state is a forked, repository-maintained version maintained by
   Jinwoo Lee (``cinescope@kaist.ac.kr``).
   ``PADO Hologram`` refers to repository-maintained work layered on top of the
   original PADO optics core.

Naming and Package Layout
-------------------------

- Repository identity: ``PADO Hologram``
- Core optics package: ``pado``
- Higher-level holography namespace: ``pado_hologram``
- ``PADO`` comes from the Korean word `파도 <https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%8F%84>`_, meaning ``wave``

The package path is intentionally not renamed away from ``pado`` because the
existing optics API should remain stable while the upper holography layer grows.

Role Split
----------

- ``PADO`` remains the compact differentiable optics core:
  light fields, propagation, optical elements, materials, and a small compatibility bridge for device-aware encoding.
- ``PADO Hologram`` is intended to host higher-level holography workflows:
  CGH optimization, SLM-aware pipelines, setup orchestration, and future display/capture interfaces.

Why Build This Layer
--------------------

- The optics core stays small, readable, and easier to maintain.
- Holography-specific concerns can grow without forcing every optics user to adopt a large framework.
- This fork can provide a maintained path for CGH research workflows that would otherwise depend on older, lightly maintained stacks.

Community Vision
----------------

``PADO Hologram`` is not only a software architecture direction. It is also meant
to be a community direction.

The name helps express that vision. ``PADO`` means `파도 <https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%8F%84>`_ in Korean, or
``wave``. That fits the physics, but it also gives us a useful metaphor:
this is a wave we want to ride together.

The goal is to help create a shared place where people from computer science,
electrical engineering, optics, physics, psychology, perception science, and
other neighboring backgrounds can work on holography and computational imaging
together.

The hope is to move beyond fragmented, one-off attempts and toward a more
maintained, reusable, and inspiring ecosystem where people can contribute, learn,
and motivate one another across disciplinary boundaries.

Current Starting Point
----------------------

- ``pado.display`` remains available as a compatibility bridge from ideal phase targets to LCOS/SLM-oriented encoding.
- ``pado_hologram`` is now an importable package with concrete modules for source/propagation specs, device encoding, targets, losses, pipelines, phase-only optimization, primitive-based rendering, experiment orchestration, and learning-facing scaffolds.
- ``pado_hologram.backends`` now includes an optional `NVIDIA Warp <https://github.com/NVIDIA/warp>`_ path for custom holography kernels across DPAC and primitive-based renderer paths.
- Documentation now treats this as an explicit architectural direction rather than an implicit fork-only idea.

Current Modules
---------------

The initial ``pado_hologram`` module set is:

- ``pado_hologram.config`` for source and propagation specifications
- ``pado_hologram.backends`` for optional custom-kernel backends such as `NVIDIA Warp <https://github.com/NVIDIA/warp>`_
- ``pado_hologram.slm`` for phase-only LCOS/SLM encoding helpers
- ``pado_hologram.targets`` for reconstruction targets
- ``pado_hologram.losses`` for intensity and amplitude losses plus reconstruction metrics
- ``pado_hologram.pipeline`` for composed holography workflows
- ``pado_hologram.algorithms`` for compact hologram-generation algorithms including DPAC and Gerchberg-Saxton
- ``pado_hologram.devices`` for SLM wrappers and optional camera/observation transforms
- ``pado_hologram.primitive_based`` for Gaussian baselines, exact GWS/RPWS paths, and optional backend selection
- ``pado_hologram.experiments`` and ``pado_hologram.hydra_app`` for reproducible experiment entry points
- ``pado_hologram.representations`` for primitive-scene data containers
- ``pado_hologram.neural`` for capture-, calibration-, and learning-facing scaffolds

Near-Term Scope
---------------

The next important building blocks for ``PADO Hologram`` are:

- stronger benchmark and parity harnesses across methods
- richer phase-optimization interfaces beyond compact GS/DPAC baselines
- higher-level supervision abstractions for depth and multi-plane workflows
- capture/calibration layers kept separate from the optics core
- learned forward-model interfaces for future neural holography workflows

Optional `NVIDIA Warp <https://github.com/NVIDIA/warp>`_ Layer
--------------------------------------------------------------

The repository now includes an optional `NVIDIA Warp <https://github.com/NVIDIA/warp>`_
integration. Its intended significance is narrow and practical:

- not to replace the full ``pado`` propagation stack
- not to over-claim wholesale acceleration
- to create a maintainable place for future custom holography kernels

Current integration points include the DPAC checkerboard kernel path,
primitive-scene splat backends, and primitive-based exact renderer paths. That
gives the project a real Warp-backed starting point while keeping the core
optics engine PyTorch-first.

Contributor Welcome
-------------------

Contributors are welcome, especially people interested in helping build a cleaner
long-term alternative to larger CGH research stacks.

If you have experience with `holotorch <https://github.com/facebookresearch/holotorch>`_,
that perspective is valuable here. The aim is not to clone it, but to learn from
what it made possible and rebuild the most useful ideas in a smaller, PADO-native,
more maintainable form.

That invitation is intentionally broad. People from computer science, electrical engineering, optics, physics,
psychology, perception science, and adjacent fields are all welcome if they care
about advancing holography and computational imaging together.

Boundary
--------

The goal is not to turn ``pado`` itself into a monolithic end-to-end research framework.
Instead, the goal is to keep the core reliable and build the larger holography stack as
an explicit upper layer.

.. image:: ../images/banner_1.0.0.png
   :width: 100%
   :class: banner-image

PADO Hologram
=============

``PADO Hologram`` is the CGH-focused evolution of the original
`PADO <https://github.com/shwbaek/pado>`_ differentiable optics core.
Rebuilt as a lean, native stack, it picks up where earlier frameworks such as
`holotorch <https://github.com/facebookresearch/holotorch>`_ left off, with a stronger
emphasis on long-term maintainability, clarity, and performance.

``PADO`` can also be read as `PADO (파도) <https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%8F%84>`_, the Korean
word for ``wave``. The name points both to the physical waves we manipulate and
to the collective momentum of the researchers who work on them.

This repository is intended as a shared home for the broader holography and
computational imaging community: a place where people from physics, computer science, electrical engineering,
optics, psychology, perception research, and neighboring areas can move beyond
fragmented one-off efforts and build together. In that spirit,
``PADO Hologram`` is an invitation to surf this `PADO (파도) <https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%8F%84>`_ together.

.. note::

   This documentation covers a forked, repository-maintained, holography-oriented
   repository built on top of the original PADO framework.
   The original framework is developed by the `POSTECH Computer Graphics Lab <https://sites.google.com/view/shbaek/home>`_.
   Fork maintainer: Jinwoo Lee (``cinescope@kaist.ac.kr``).

The repository identity is ``PADO Hologram``.
The core optics package path remains ``pado`` for compatibility.
The higher-level holography namespace reserved in this repository is ``pado_hologram``.

.. grid:: 1 1 2 5
   :gutter: 3
   :class-container: grid-container

   .. grid-item::
      :class: grid-item-card

      .. card::
         :link: pado_hologram
         :link-type: doc
         :class-card: custom-card

         PADO Hologram
         ^^^^^^^^^^^^^

         Architecture, scope, and repository direction for the holography layer.

   .. grid-item::
      :class: grid-item-card

      .. card::
         :link: installation
         :link-type: doc
         :class-card: custom-card

         Installation
         ^^^^^^^^^^^^

         Set up the maintained repository state and understand the package layout.

   .. grid-item::
      :class: grid-item-card

      .. card::
         :link: api/index
         :link-type: doc
         :class-card: custom-card

         PADO Core API
         ^^^^^^^^^^^^^

         Reference documentation for the ``pado`` optics core and the current compatibility bridge.

   .. grid-item::
      :class: grid-item-card

      .. card::
         :link: examples/index
         :link-type: doc
         :class-card: custom-card

         Examples
         ^^^^^^^^

         Holography-first notebooks plus the broader optics examples already in the repository.

   .. grid-item::
      :class: grid-item-card

      .. card::
         :link: updates
         :link-type: doc
         :class-card: custom-card

         Updates
         ^^^^^^^

         Repository-maintained additions and stabilization patches.

   .. grid-item::
      :class: grid-item-card

      .. card::
         :link: contributing
         :link-type: doc
         :class-card: custom-card

         Contributing
         ^^^^^^^^^^^^

         Welcome for contributors interested in building the holography layer.

   .. grid-item::
      :class: grid-item-card

      .. card::
         :link: license
         :link-type: doc
         :class-card: custom-card

         License
         ^^^^^^^

         Information about PADO's license and usage terms.

.. image:: ../images/footer_1.0.0.svg
   :width: 100%
   :class: footer-image

.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   pado_hologram
   updates
   contributing
   api/index
   examples/index
   license
   citation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* `Maintained PADO Hologram repository <https://github.com/cinescope-wkr/pado-hologram>`_
* `Original PADO GitHub repo <https://github.com/shwbaek/pado>`_

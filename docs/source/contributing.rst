Contributing
============

Contributors are welcome.

``PADO Hologram`` is being built as a maintained holography stack on top of the
original ``pado`` optics core, and contributions across algorithms, abstractions,
tests, and documentation are valuable.

This is also intended to be a community project in the broader sense: a place
where people from computer science, electrical engineering, optics, physics, psychology, perception science, and
related backgrounds can contribute to a shared effort instead of staying trapped
inside fragmented one-off codebases and disconnected attempts.

If you have experience with related research codebases such as
`holotorch <https://github.com/facebookresearch/holotorch>`_, your perspective is
especially welcome. That project helped demonstrate the value of higher-level CGH
workflow layers; here, the goal is to rebuild that kind of capability as a smaller,
cleaner, and more maintainable PADO-native stack.

We want this repository to be a place where people not only contribute code, but
also inspire one another, exchange ideas across disciplines, and help shape a
more coherent long-term ecosystem for holography and computational imaging.

Good areas to contribute
------------------------

- hologram-generation algorithms such as DPAC, Gerchberg-Saxton variants, and multi-plane methods
- SLM and display models, measured LUT handling, and hardware-aware abstractions
- Hydra-based experiment configuration and reproducibility tooling
- tests, examples, and documentation that keep the project maintainable

Working principles
------------------

- keep the ``pado`` core compact and reusable
- place higher-level CGH workflow code in ``pado_hologram``
- prefer documented, testable APIs over notebook-only logic
- avoid importing non-MIT code into this repository

The canonical contributor guide for the maintained repository state lives in the
MkDocs documentation:
`https://cinescope-wkr.github.io/pado-hologram/community/contributing/ <https://cinescope-wkr.github.io/pado-hologram/community/contributing/>`_.

The repository-root ``CONTRIBUTING.md`` is kept only as a short GitHub-visible
shim pointing to that guide.

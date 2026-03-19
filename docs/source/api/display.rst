Display and Encoding Bridge
===========================

.. currentmodule:: pado.display

This module remains available under ``pado`` for compatibility, but in the
current repository architecture it is documented as a bridge into
``pado_hologram`` rather than as part of the long-term optics-core identity.

LCOS Models
-----------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :recursive:

   LCOSLUT

Encoding Helpers
----------------

The LCOS helpers automatically wrap target phase into the phase convention used
by the supplied LUT. This keeps common measured LUTs expressed over either
``[0, 2π]`` or ``[-π, π]`` on a sane default path.

.. autosummary::
   :toctree: _autosummary
   :template: function.rst
   :recursive:

   lcos_encode_phase
   phase_only_field
   slm_light_from_phase

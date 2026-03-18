# Device Modeling and Display Encoding

This page explains where device-aware modeling currently lives.

## Why Device Modeling Matters

An ideal phase pattern is not the same thing as a realizable display pattern.
Real SLMs and LCOS devices introduce effects such as:

- quantization
- wavelength-dependent phase response
- measured LUT behavior
- non-ideal amplitude response

Ignoring that gap makes optimization results harder to trust in practical use.

## Current Repository Path

The first device-aware layer currently appears in `pado.display`, especially
through:

- `LCOSLUT`
- `lcos_encode_phase`
- `slm_light_from_phase`

On top of that, `pado_hologram.slm` provides framework-level wrappers such as
`PhaseOnlyLCOSSLM`.

## Current Scope

The current implementation is intentionally modest. It focuses on:

- LUT-based phase encoding
- quantized phase realization
- optional amplitude LUT modulation
- integration into a pipeline-friendly API

It does not yet model the full complexity of real hardware systems, but it gives
the repository a clean place to keep growing.

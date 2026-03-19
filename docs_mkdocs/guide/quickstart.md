# Quickstart

This page gives a compact, self-contained path from installation to a first CGH
run.

!!! note

    The examples below are intentionally small. Their job is to show how the repository is layered, not to present the final word on hologram quality or benchmarking.

## 1. Import the Main Layers

```python
import pado
import pado_hologram
```

At a high level:

- `pado` contains the differentiable optics core
- `pado_hologram` contains holography-oriented abstractions and workflows

## 2. Run a Built-In Experiment

The easiest entry point today is the package CLI:

```bash
pado-hologram run experiment=gs
```

This exercises:

- source specification
- propagation configuration
- target creation
- a compact phase optimization loop

You can also run the current DPAC path:

```bash
pado-hologram run experiment=dpac target=gaussian
```

The primitive-based exact path is also available:

```bash
pado-hologram run experiment=primitive_gaussian_gws_exact primitives=gaussian3d_depth_ring
```

If you have installed the optional Warp dependency, you can request the
Warp-backed path explicitly for either DPAC or the exact primitive renderer:

```bash
pado-hologram run experiment=dpac target=gaussian backend=warp
pado-hologram run experiment=primitive_gaussian_gws_exact primitives=gaussian3d_depth_ring backend=warp
```

The Hydra-native compatibility path still exists when you want the lower-level
Hydra entrypoint directly:

```bash
python -m pado_hologram.hydra_app experiment=gs
```

## 3. Use the Core API Directly

```python
from pado_hologram import SourceSpec, PropagationSpec, HologramPipeline

source = SourceSpec(dim=(1, 1, 128, 128), pitch=8e-6, wvl=532e-9)
propagation = PropagationSpec(distance=0.2, mode="ASM")
pipeline = HologramPipeline(source, propagation)

light = pipeline.make_source_light()
result = pipeline.forward_source(light)
```

This is the simplest way to see the current layering:

- `SourceSpec` describes the source/SLM plane
- `PropagationSpec` describes the observation-plane propagation
- `HologramPipeline` composes the end-to-end forward pass

## 4. Use Device-Aware Phase Encoding

The first LCOS/SLM-oriented helper remains available through `pado.display` as a
compatibility bridge:

```python
from pado.display import LCOSLUT, lcos_encode_phase, slm_light_from_phase
```

This layer is important because it bridges ideal phase optimization to realized
display-domain behavior such as quantization and LUT-based phase response while
the higher-level workflow surface moves toward `pado_hologram.devices` and
`pado_hologram.slm`.

## 5. Next Steps

- Read [Repository Layout](repository-layout.md)
- Read [Architecture Overview](../concepts/architecture.md)
- Read [Experiments and Hydra](../workflows/experiments.md)
- Read [Phase-Only CGH Workflow](../workflows/phase-only-cgh.md)

"""Generate the PADO Hologram hero banner from an actual PADO experiment.

The banner is intentionally built from a real holography workflow:

1. Rasterize the text target ``PADO`` into an amplitude target.
2. Optimize a phase-only hologram with Gerchberg-Saxton.
3. Apply a minimal LCOS LUT model.
4. Propagate the encoded field to multiple planes.
5. Compose the resulting reconstruction, phase fringes, and defocus glow into
   a single abstract hero image.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pado-banner")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter, zoom

from pado.display import LCOSLUT
from pado_hologram import (
    GerchbergSaxtonPhaseOptimizer,
    HologramPipeline,
    IntensityTarget,
    PhaseOnlyLCOSSLM,
    PropagationSpec,
    SourceSpec,
)

OUT_PATH = ROOT / "docs" / "images" / "banner_1.0.0.png"
SEED = 7


def _ensure_matplotlib_cache() -> None:
    cache_dir = Path("/tmp/matplotlib-pado-banner")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _normalize(image: np.ndarray, *, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    lo = np.percentile(image, low)
    hi = np.percentile(image, high)
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _rasterize_text(text: str, height: int, width: int) -> np.ndarray:
    prop = FontProperties(family="DejaVu Sans", weight="bold")
    text_path = TextPath((0.0, 0.0), text, size=1.0, prop=prop)
    bbox = text_path.get_extents()

    scale_x = width * 0.70 / bbox.width
    scale_y = height * 0.34 / bbox.height
    scale = min(scale_x, scale_y)

    tx = (width - bbox.width * scale) / 2.0 - bbox.x0 * scale
    ty = (height - bbox.height * scale) / 2.0 - bbox.y0 * scale
    transform = Affine2D().scale(scale, scale).translate(tx, ty)
    transformed = transform.transform_path(text_path)

    yy, xx = np.mgrid[0:height, 0:width]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    mask = transformed.contains_points(points).reshape(height, width).astype(np.float32)
    return mask


def _build_target_amplitude(height: int, width: int) -> np.ndarray:
    text_mask = _rasterize_text("PADO", height, width)
    soft_text = gaussian_filter(text_mask, sigma=2.0)
    halo = gaussian_filter(text_mask, sigma=14.0)
    broad_halo = gaussian_filter(text_mask, sigma=28.0)

    yy, xx = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    radius = np.sqrt((xx * 1.15) ** 2 + (yy * 0.9) ** 2)
    angle = np.arctan2(yy, xx)

    orbital = 0.5 + 0.5 * np.cos(26.0 * radius - 3.4 * angle)
    carrier = 0.5 + 0.5 * np.sin(42.0 * (0.85 * xx - 0.18 * yy) + 0.8 * np.sin(4.0 * yy))
    plume = np.exp(-((xx + 0.18) ** 2 / 0.35 + (yy - 0.02) ** 2 / 0.10))
    veil = np.exp(-((xx - 0.24) ** 2 / 0.42 + (yy + 0.08) ** 2 / 0.16))

    amplitude = (
        0.05
        + 0.84 * soft_text
        + 0.38 * halo * orbital
        + 0.16 * broad_halo * carrier
        + 0.12 * plume
        + 0.08 * veil
    )
    amplitude = gaussian_filter(amplitude, sigma=1.2)
    amplitude = _normalize(amplitude, low=0.2, high=99.7)
    amplitude = np.clip(0.12 + 0.88 * amplitude, 0.0, 1.0)
    return amplitude.astype(np.float32)


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).view(1, 1, image.shape[0], image.shape[1]).to(torch.float32)


def _make_lut() -> LCOSLUT:
    levels = 256
    phase_lut = torch.linspace(0.0, float(2.0 * torch.pi * 0.97), levels)
    phase_axis = torch.linspace(0.0, 1.0, levels)
    amplitude_lut = 0.93 + 0.05 * torch.cos(phase_axis * torch.pi - 0.35)
    amplitude_lut = amplitude_lut.clamp(0.86, 1.0)
    return LCOSLUT(
        phase_lut=phase_lut,
        amplitude_lut=amplitude_lut,
        wvl_ref=532e-9,
    )


def _compose_banner(
    target_amp: np.ndarray,
    phase: np.ndarray,
    gray: np.ndarray,
    focus: np.ndarray,
    near: np.ndarray,
    far: np.ndarray,
) -> np.ndarray:
    focus_n = _normalize(focus, low=2.0, high=99.8)
    near_n = _normalize(near, low=1.0, high=99.6)
    far_n = _normalize(far, low=1.0, high=99.6)
    phase_n = np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi)
    gray_n = np.clip(gray, 0.0, 1.0)

    phase_rgb = colormaps["twilight_shifted"](phase_n)[..., :3].astype(np.float32)
    halo = gaussian_filter(target_amp, sigma=10.0)
    letter_core = gaussian_filter(target_amp, sigma=1.6)
    focus_glow = gaussian_filter(focus_n, sigma=3.4)
    near_glow = gaussian_filter(near_n, sigma=5.0)
    far_glow = gaussian_filter(far_n, sigma=6.5)

    yy, xx = np.mgrid[-1.0:1.0:complex(target_amp.shape[0]), -1.0:1.0:complex(target_amp.shape[1])]
    vignette = np.clip(1.0 - 0.38 * (xx**2 + (yy * 1.08) ** 2), 0.55, 1.0).astype(np.float32)
    floor = np.exp(-((yy - 0.76) ** 2) / 0.020).astype(np.float32)
    gray_strands = gaussian_filter(gray_n, sigma=1.2)

    canvas = np.zeros((*target_amp.shape, 3), dtype=np.float32)
    canvas += np.array([0.025, 0.055, 0.085], dtype=np.float32)
    canvas += 0.16 * vignette[..., None] * np.array([0.00, 0.10, 0.14], dtype=np.float32)
    canvas += 0.20 * (phase_rgb ** 1.18)
    canvas += 0.14 * gray_strands[..., None] * np.array([0.10, 0.42, 0.48], dtype=np.float32)
    canvas += 0.40 * near_glow[..., None] * np.array([0.16, 0.86, 0.93], dtype=np.float32)
    canvas += 0.28 * far_glow[..., None] * np.array([0.10, 0.48, 0.90], dtype=np.float32)
    canvas += 0.62 * focus_glow[..., None] * np.array([1.00, 0.66, 0.26], dtype=np.float32)
    canvas += 0.36 * halo[..., None] * np.array([0.22, 0.92, 0.96], dtype=np.float32)
    canvas += 0.54 * letter_core[..., None] * np.array([1.00, 0.96, 0.88], dtype=np.float32)
    canvas += 0.10 * floor[..., None] * np.array([0.90, 0.48, 0.18], dtype=np.float32)

    rng = np.random.default_rng(SEED)
    speckle = gaussian_filter(rng.random(target_amp.shape).astype(np.float32), sigma=0.45)
    stars = (speckle > 0.74).astype(np.float32) * gaussian_filter(halo, sigma=18.0)
    canvas += 0.12 * stars[..., None] * np.array([0.95, 0.98, 1.00], dtype=np.float32)

    canvas *= vignette[..., None]
    canvas = np.clip(canvas, 0.0, 1.0)
    canvas = np.power(canvas, 0.92, dtype=np.float32)
    return canvas


def main() -> None:
    _ensure_matplotlib_cache()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    height, width = 280, 800
    target_amp = _build_target_amplitude(height, width)

    source = SourceSpec(dim=(1, 1, height, width), pitch=8e-6, wvl=532e-9)
    focus_spec = PropagationSpec(distance=0.24, mode="ASM", band_limit=True)
    near_spec = PropagationSpec(distance=0.225, mode="ASM", band_limit=True)
    far_spec = PropagationSpec(distance=0.258, mode="ASM", band_limit=True)

    target = IntensityTarget.from_amplitude(_to_tensor(target_amp), normalize_mean=True)
    optimizer = GerchbergSaxtonPhaseOptimizer(source, focus_spec)
    result = optimizer.optimize(target, iterations=18)

    slm = PhaseOnlyLCOSSLM(source, _make_lut())
    pipeline = HologramPipeline(source, focus_spec, slm=slm)
    forward = pipeline.forward_phase(result.phase, target=target)
    if forward.encoding is None:
        raise RuntimeError("Expected a realized encoding for banner generation")

    slm_light = forward.slm_light.clone()
    focus = forward.propagated_light.get_intensity()[0, 0].detach().cpu().numpy()
    near = near_spec.forward(slm_light.clone()).get_intensity()[0, 0].detach().cpu().numpy()
    far = far_spec.forward(slm_light.clone()).get_intensity()[0, 0].detach().cpu().numpy()
    phase = forward.encoding.phase_realized[0, 0].detach().cpu().numpy()
    gray = forward.encoding.gray[0, 0].detach().cpu().numpy()

    banner = np.flipud(_compose_banner(target_amp, phase, gray, focus, near, far))
    banner = zoom(banner, (2.0, 2.0, 1.0), order=3)
    banner = np.clip(banner, 0.0, 1.0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    plt.imsave(OUT_PATH, banner)
    print(f"saved {OUT_PATH}")
    print(f"focus-plane mse: {float(forward.metrics['mse']) if forward.metrics else 'n/a'}")


if __name__ == "__main__":
    main()

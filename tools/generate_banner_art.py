"""Generate the PADO Hologram hero banner from a physically grounded workflow.

The banner is built in two layers:

1. Physics layer:
   - rasterize ``PADO`` into a target amplitude map
   - build a more realistic illumination profile
   - optimize a phase-only hologram
   - apply a simple LCOS LUT model
   - reconstruct separate R/G/B wavelengths at multiple planes
2. Presentation layer:
   - tone map the RGB reconstructions
   - derive mild bloom/defocus from physical planes
   - compose a polished hero banner without repainting the physics from scratch
"""

from __future__ import annotations

import argparse
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
BASE_HEIGHT = 320
BASE_WIDTH = 920
PITCH = 8e-6
WAVELENGTHS = {
    "r": 638e-9,
    "g": 532e-9,
    "b": 450e-9,
}
CHANNEL_BALANCE = {
    "r": 1.06,
    "g": 0.94,
    "b": 1.18,
}
PLANE_DISTANCES = {
    "focus": 0.240,
    "near": 0.225,
    "far": 0.258,
}
PRESET_OUTPUTS = {
    "default": (1840, 640),
    "4k": (3840, 1336),
}
QUALITY_PRESETS = {
    "draft": {"render_scale": 0.85, "gs_iterations": 14, "upsample_order": 3},
    "default": {"render_scale": 1.0, "gs_iterations": 18, "upsample_order": 3},
    "high": {"render_scale": 1.5, "gs_iterations": 24, "upsample_order": 5},
    "max": {"render_scale": 2.0, "gs_iterations": 30, "upsample_order": 5},
}


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


def _normalize_rgb(image: np.ndarray, *, low: float = 0.8, high: float = 99.4) -> np.ndarray:
    channels = [_normalize(image[..., idx], low=low, high=high) for idx in range(image.shape[-1])]
    return np.stack(channels, axis=-1).astype(np.float32)


def _unsharp_mask(image: np.ndarray, *, sigma: float, amount: float) -> np.ndarray:
    blurred = gaussian_filter(image, sigma=(sigma, sigma, 0.0) if image.ndim == 3 else sigma)
    sharpened = image + amount * (image - blurred)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


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


def build_target_amplitude(height: int, width: int) -> np.ndarray:
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


def build_source_amplitude(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    radial = np.sqrt((xx * 1.08) ** 2 + (yy * 0.92) ** 2)
    gaussian_core = np.exp(-(radial**2) / 0.92)
    wide_fill = np.exp(-((xx + 0.08) ** 2 / 1.75 + (yy + 0.02) ** 2 / 0.95))
    aperture = np.clip(1.0 - np.maximum(radial - 0.84, 0.0) / 0.24, 0.0, 1.0)
    illumination_tilt = 0.94 + 0.08 * (0.65 * xx - 0.35 * yy)
    ripple = 1.0 + 0.012 * np.cos(8.0 * xx + 5.0 * yy)

    source = (0.78 * gaussian_core + 0.22 * wide_fill) * aperture
    source *= illumination_tilt
    source *= ripple
    source = gaussian_filter(source.astype(np.float32), sigma=0.9)
    source = _normalize(source, low=0.4, high=99.8)
    source = np.clip(0.18 + 0.82 * source, 0.0, 1.0)
    return source.astype(np.float32)


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).view(1, 1, image.shape[0], image.shape[1]).to(torch.float32)


def make_lut_for_wavelength(wvl: float) -> LCOSLUT:
    levels = 256
    phase_lut = torch.linspace(0.0, float(2.0 * torch.pi * 0.97), levels)
    phase_axis = torch.linspace(0.0, 1.0, levels)
    chroma = float((wvl - WAVELENGTHS["g"]) / WAVELENGTHS["g"])
    amplitude_lut = 0.93 + (0.05 + 0.01 * chroma) * torch.cos(phase_axis * torch.pi - 0.35)
    amplitude_lut = amplitude_lut.clamp(0.84, 1.0)
    return LCOSLUT(
        phase_lut=phase_lut,
        amplitude_lut=amplitude_lut,
        wvl_ref=WAVELENGTHS["g"],
    )


def render_single_wavelength(
    *,
    wvl: float,
    pitch: float,
    target_amp: np.ndarray,
    source_amp: np.ndarray,
    distances: dict[str, float],
    iterations: int,
) -> dict[str, np.ndarray | float]:
    height, width = target_amp.shape
    source = SourceSpec(dim=(1, 1, height, width), pitch=pitch, wvl=wvl)
    focus_spec = PropagationSpec(distance=distances["focus"], mode="ASM", band_limit=True)
    near_spec = PropagationSpec(distance=distances["near"], mode="ASM", band_limit=True)
    far_spec = PropagationSpec(distance=distances["far"], mode="ASM", band_limit=True)

    target = IntensityTarget.from_amplitude(_to_tensor(target_amp), normalize_mean=True)
    optimizer = GerchbergSaxtonPhaseOptimizer(source, focus_spec)
    result = optimizer.optimize(
        target,
        iterations=iterations,
        source_amplitude=_to_tensor(source_amp),
    )

    slm = PhaseOnlyLCOSSLM(source, make_lut_for_wavelength(wvl))
    pipeline = HologramPipeline(source, focus_spec, slm=slm)
    source_light = source.make_light(amplitude=_to_tensor(source_amp))
    forward = pipeline.forward_phase(result.phase, source_light=source_light, target=target)
    if forward.encoding is None:
        raise RuntimeError("Expected a realized encoding for banner generation")

    slm_light = forward.slm_light.clone()
    focus = forward.propagated_light.get_intensity()[0, 0].detach().cpu().numpy()
    near = near_spec.forward(slm_light.clone()).get_intensity()[0, 0].detach().cpu().numpy()
    far = far_spec.forward(slm_light.clone()).get_intensity()[0, 0].detach().cpu().numpy()
    phase = forward.encoding.phase_realized[0, 0].detach().cpu().numpy()
    gray = forward.encoding.gray[0, 0].detach().cpu().numpy()
    source_realized = forward.source_light.get_intensity()[0, 0].detach().cpu().numpy()

    return {
        "focus": focus.astype(np.float32),
        "near": near.astype(np.float32),
        "far": far.astype(np.float32),
        "phase": phase.astype(np.float32),
        "gray": gray.astype(np.float32),
        "source": source_realized.astype(np.float32),
        "mse": float(forward.metrics["mse"]) if forward.metrics else float("nan"),
    }


def render_rgb_reconstruction(
    *,
    target_amp: np.ndarray,
    source_amp: np.ndarray,
    pitch: float,
    distances: dict[str, float],
    iterations: int,
) -> dict[str, np.ndarray | dict[str, float]]:
    per_channel: dict[str, dict[str, np.ndarray | float]] = {}
    for channel, wvl in WAVELENGTHS.items():
        per_channel[channel] = render_single_wavelength(
            wvl=wvl,
            pitch=pitch,
            target_amp=target_amp,
            source_amp=source_amp,
            distances=distances,
            iterations=iterations,
        )

    def stack_plane(key: str, *, low: float, high: float) -> np.ndarray:
        stacked = np.stack(
            [
                CHANNEL_BALANCE[channel] * _normalize(per_channel[channel][key], low=low, high=high)
                for channel in ("r", "g", "b")
            ],
            axis=-1,
        ).astype(np.float32)
        return np.clip(stacked, 0.0, 1.0)

    phase = np.mean(np.stack([per_channel[channel]["phase"] for channel in ("r", "g", "b")], axis=0), axis=0)
    gray = np.mean(np.stack([per_channel[channel]["gray"] for channel in ("r", "g", "b")], axis=0), axis=0)

    return {
        "focus_rgb": stack_plane("focus", low=1.0, high=99.8),
        "near_rgb": stack_plane("near", low=0.8, high=99.6),
        "far_rgb": stack_plane("far", low=0.8, high=99.6),
        "source_rgb": stack_plane("source", low=0.0, high=100.0),
        "phase_mean": phase.astype(np.float32),
        "gray_mean": gray.astype(np.float32),
        "mse": {channel: float(per_channel[channel]["mse"]) for channel in ("r", "g", "b")},
    }


def compose_banner_from_rgb(
    *,
    target_amp: np.ndarray,
    rendered: dict[str, np.ndarray | dict[str, float]],
) -> np.ndarray:
    focus_rgb = _normalize_rgb(rendered["focus_rgb"], low=0.6, high=99.6)
    near_rgb = _normalize_rgb(rendered["near_rgb"], low=0.6, high=99.2)
    far_rgb = _normalize_rgb(rendered["far_rgb"], low=0.6, high=99.2)
    source_rgb = _normalize_rgb(rendered["source_rgb"], low=0.0, high=100.0)
    phase_mean = np.mod(rendered["phase_mean"], 2.0 * np.pi) / (2.0 * np.pi)
    gray_mean = np.clip(rendered["gray_mean"], 0.0, 1.0)
    phase_rgb = colormaps["twilight_shifted"](phase_mean)[..., :3].astype(np.float32)

    halo = gaussian_filter(target_amp, sigma=11.0)
    letter_core = gaussian_filter(target_amp, sigma=1.4)
    letter_glow = gaussian_filter(target_amp, sigma=4.5)
    focus_glow = gaussian_filter(focus_rgb, sigma=(3.0, 3.0, 0.0))
    near_glow = gaussian_filter(near_rgb, sigma=(5.4, 5.4, 0.0))
    far_glow = gaussian_filter(far_rgb, sigma=(7.0, 7.0, 0.0))
    phase_strands = gaussian_filter(phase_mean.astype(np.float32), sigma=1.1)
    gray_sheen = gaussian_filter(gray_mean.astype(np.float32), sigma=1.5)
    focus_sharp = _unsharp_mask(focus_rgb, sigma=1.2, amount=1.35)
    near_sharp = _unsharp_mask(near_rgb, sigma=1.6, amount=0.55)
    phase_sharp = _unsharp_mask(phase_rgb, sigma=1.1, amount=0.70)

    yy, xx = np.mgrid[-1.0:1.0:complex(target_amp.shape[0]), -1.0:1.0:complex(target_amp.shape[1])]
    vignette = np.clip(1.0 - 0.34 * (xx**2 + (yy * 1.08) ** 2), 0.58, 1.0).astype(np.float32)
    floor = np.exp(-((yy - 0.76) ** 2) / 0.022).astype(np.float32)
    top_glow = np.exp(-((yy + 0.82) ** 2) / 0.10).astype(np.float32)

    canvas = np.zeros((*target_amp.shape, 3), dtype=np.float32)
    canvas += np.array([0.020, 0.032, 0.060], dtype=np.float32)
    canvas += 0.18 * vignette[..., None] * np.array([0.00, 0.10, 0.16], dtype=np.float32)
    canvas += 0.10 * source_rgb
    canvas += 0.18 * (phase_rgb ** 1.10)
    canvas += 0.12 * phase_sharp * halo[..., None]
    canvas += 0.34 * far_glow * np.array([0.60, 0.82, 1.00], dtype=np.float32)
    canvas += 0.44 * near_glow * np.array([0.82, 0.92, 1.00], dtype=np.float32)
    canvas += 0.56 * near_sharp
    canvas += 0.78 * focus_sharp
    canvas += 0.30 * focus_glow
    canvas += 0.08 * phase_strands[..., None] * np.array([0.72, 0.80, 0.96], dtype=np.float32)
    canvas += 0.05 * gray_sheen[..., None] * np.array([0.96, 0.98, 1.00], dtype=np.float32)
    canvas += 0.20 * halo[..., None] * np.array([0.28, 0.62, 0.88], dtype=np.float32)
    canvas += 0.28 * letter_glow[..., None] * np.array([0.94, 0.96, 0.98], dtype=np.float32)
    canvas += 0.46 * letter_core[..., None] * np.array([1.00, 0.98, 0.94], dtype=np.float32)
    canvas += 0.10 * floor[..., None] * np.array([0.80, 0.50, 0.24], dtype=np.float32)
    canvas += 0.05 * top_glow[..., None] * np.array([0.18, 0.24, 0.36], dtype=np.float32)

    rng = np.random.default_rng(SEED)
    sparkle_field = gaussian_filter(rng.random(target_amp.shape).astype(np.float32), sigma=0.55)
    sparkles = (sparkle_field > 0.80).astype(np.float32) * gaussian_filter(halo, sigma=19.0)
    canvas += 0.05 * sparkles[..., None] * np.array([0.94, 0.97, 1.00], dtype=np.float32)

    canvas *= vignette[..., None]
    canvas = np.clip(canvas, 0.0, 1.0)
    canvas = np.power(canvas, 0.91, dtype=np.float32)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_PRESETS),
        default="default",
        help="Internal render quality preset. Controls render scale, GS iterations, and upscale quality.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_OUTPUTS),
        default="default",
        help="Output resolution preset.",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="Explicit output width in pixels. Overrides the preset width.",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=None,
        help="Explicit output height in pixels. Overrides the preset height.",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=None,
        help="Scale factor applied to the internal physics render resolution. Overrides the quality preset.",
    )
    parser.add_argument(
        "--gs-iterations",
        type=int,
        default=None,
        help="Gerchberg-Saxton iteration count. Overrides the quality preset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_PATH,
        help="Output image path.",
    )
    return parser.parse_args()


def resolve_runtime_config(args: argparse.Namespace) -> tuple[int, int, int, int, float, int, int]:
    quality = QUALITY_PRESETS[args.quality]
    render_scale = quality["render_scale"] if args.render_scale is None else args.render_scale
    gs_iterations = quality["gs_iterations"] if args.gs_iterations is None else args.gs_iterations
    upsample_order = quality["upsample_order"]

    if render_scale <= 0:
        raise ValueError(f"render_scale must be positive, got {render_scale}")
    if gs_iterations < 1:
        raise ValueError(f"gs_iterations must be >= 1, got {gs_iterations}")

    render_height = max(1, int(round(BASE_HEIGHT * render_scale)))
    render_width = max(1, int(round(BASE_WIDTH * render_scale)))

    preset_width, preset_height = PRESET_OUTPUTS[args.preset]
    output_width = args.output_width if args.output_width is not None else preset_width
    output_height = args.output_height if args.output_height is not None else preset_height

    if output_width <= 0 or output_height <= 0:
        raise ValueError(
            f"output dimensions must be positive, got width={output_width}, height={output_height}"
        )

    return render_height, render_width, output_width, output_height, render_scale, gs_iterations, upsample_order


def main() -> None:
    args = parse_args()
    _ensure_matplotlib_cache()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    (
        render_height,
        render_width,
        output_width,
        output_height,
        render_scale,
        gs_iterations,
        upsample_order,
    ) = resolve_runtime_config(args)

    target_amp = build_target_amplitude(render_height, render_width)
    source_amp = build_source_amplitude(render_height, render_width)
    rendered = render_rgb_reconstruction(
        target_amp=target_amp,
        source_amp=source_amp,
        pitch=PITCH,
        distances=PLANE_DISTANCES,
        iterations=gs_iterations,
    )
    banner = np.flipud(compose_banner_from_rgb(target_amp=target_amp, rendered=rendered))
    banner = zoom(
        banner,
        (
            output_height / banner.shape[0],
            output_width / banner.shape[1],
            1.0,
        ),
        order=upsample_order,
    )
    banner = np.clip(banner, 0.0, 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    plt.imsave(args.output, banner)
    mse = rendered["mse"]
    print(
        f"saved {args.output} "
        f"(quality={args.quality}, render_scale={render_scale:.2f}, "
        f"gs_iterations={gs_iterations}, render={render_width}x{render_height}, "
        f"output={output_width}x{output_height})"
    )
    print(
        "focus-plane mse (r/g/b): "
        f"{mse['r']:.6f} / {mse['g']:.6f} / {mse['b']:.6f}"
    )


if __name__ == "__main__":
    main()

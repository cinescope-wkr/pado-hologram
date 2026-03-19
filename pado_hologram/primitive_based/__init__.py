"""Primitive-based CGH algorithms and renderers."""

from .gaussian import (
    PrimitiveFieldRenderResult,
    available_primitive_backends,
    available_primitive_renderers,
    build_primitive_scene_from_config,
    render_gaussian_scene,
    render_gaussian_scene_gws_exact,
    render_gaussian_scene_gws_exact_awb,
    render_gaussian_scene_gws_rpws_exact,
    render_gaussian_scene_naive,
    render_gaussian_scene_splat,
    render_gaussian_scene_wave,
    render_gaussian_scene_wave_awb,
    render_primitive_scene,
)

__all__ = [
    "PrimitiveFieldRenderResult",
    "available_primitive_backends",
    "available_primitive_renderers",
    "build_primitive_scene_from_config",
    "render_gaussian_scene",
    "render_gaussian_scene_gws_exact",
    "render_gaussian_scene_gws_exact_awb",
    "render_gaussian_scene_gws_rpws_exact",
    "render_gaussian_scene_naive",
    "render_gaussian_scene_splat",
    "render_gaussian_scene_wave",
    "render_gaussian_scene_wave_awb",
    "render_primitive_scene",
]

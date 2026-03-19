"""Backward-compatible re-exports for SLM helpers."""

from .devices.slm import PhaseEncodingConfig, PhaseEncodingResult, PhaseOnlyLCOSSLM

__all__ = [
    "PhaseEncodingConfig",
    "PhaseEncodingResult",
    "PhaseOnlyLCOSSLM",
]

"""Learning- and capture-facing scaffolds for future neural holography work."""

from .calibration import CalibrationRecord
from .datasets import NeuralHolographyBatch
from .specs import CaptureSessionSpec
from .training import NeuralHolographyStepResult, NeuralHolographyTrainer

__all__ = [
    "CalibrationRecord",
    "CaptureSessionSpec",
    "NeuralHolographyBatch",
    "NeuralHolographyStepResult",
    "NeuralHolographyTrainer",
]

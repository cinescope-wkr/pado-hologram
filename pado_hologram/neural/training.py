from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from .._tensor import coerce_4d_real
from .datasets import NeuralHolographyBatch


@dataclass(frozen=True)
class NeuralHolographyStepResult:
    """Capture the result of one model/evaluation step."""

    prediction: torch.Tensor
    reference: torch.Tensor
    loss: torch.Tensor
    metrics: dict[str, float]


class NeuralHolographyTrainer:
    """A small orchestration hook for learned or capture-in-the-loop pipelines."""

    def __init__(
        self,
        predict_fn: Callable[[NeuralHolographyBatch], torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        capture_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.predict_fn = predict_fn
        self.loss_fn = loss_fn
        self.capture_fn = capture_fn

    def step(self, batch: NeuralHolographyBatch) -> NeuralHolographyStepResult:
        prediction = coerce_4d_real(
            self.predict_fn(batch),
            name="prediction",
            dim=batch.dim,
        ).float()

        if self.capture_fn is None:
            reference = batch.reference_intensity.to(prediction.device).float()
            compared = prediction
        else:
            compared = coerce_4d_real(
                self.capture_fn(prediction),
                name="captured_prediction",
                dim=batch.dim,
            ).to(prediction.device).float()
            reference = batch.reference_intensity.to(prediction.device).float()

        loss = self.loss_fn(compared, reference)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "prediction_mean": float(prediction.mean().detach().cpu()),
            "reference_mean": float(reference.mean().detach().cpu()),
        }
        if self.capture_fn is not None:
            metrics["captured_mean"] = float(compared.mean().detach().cpu())

        return NeuralHolographyStepResult(
            prediction=prediction,
            reference=reference,
            loss=loss,
            metrics=metrics,
        )


__all__ = [
    "NeuralHolographyStepResult",
    "NeuralHolographyTrainer",
]

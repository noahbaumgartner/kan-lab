from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class BaseKANModel(ABC):
    model: torch.nn.Module | None = None
    device: str = "cpu"

    @abstractmethod
    def build(self, device: str = "cpu") -> None:
        """Construct the underlying model from config."""

    def regularization_loss(self) -> float:
        return 0.0

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self) -> torch.nn.Module:
        return self.model

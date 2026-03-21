from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class BaseKANModel(ABC):
    @abstractmethod
    def build(self, device: str = "cpu") -> None:
        """Construct the underlying model from config."""

    @abstractmethod
    def fit(
        self,
        dataset: dict,
        steps: int,
        lr: float,
        optimizer: str,
        loss_fn: callable | None,
        batch_size: int,
        lamb: float,
        **kwargs,
    ) -> dict:
        """Train the model. Returns dict with 'train_loss' and 'test_loss' lists."""

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

    @abstractmethod
    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """Return the underlying nn.Module."""

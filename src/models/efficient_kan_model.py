import torch

from .base import BaseKANModel
from modules.efficientkan.src.efficient_kan import KAN


class EfficientKANModel(BaseKANModel):
    def __init__(self, layers_hidden, grid=5, k=3, **kwargs):
        self.layers_hidden = layers_hidden
        self.grid = grid
        self.k = k

    def build(self, device="cpu"):
        self.model = KAN(
            layers_hidden=list(self.layers_hidden),
            grid=self.grid,
            spline_order=self.k,
        ).to(device)
        self.device = device

    def regularization_loss(self):
        return self.model.regularization_loss()

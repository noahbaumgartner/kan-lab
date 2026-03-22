import torch

from .base import BaseKANModel
from modules.efficientkan.src.efficient_kan import KAN


class EfficientKANModel(BaseKANModel):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, **kwargs):
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order

    def build(self, device="cpu"):
        self.model = KAN(
            layers_hidden=list(self.layers_hidden),
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        ).to(device)
        self.device = device

    def regularization_loss(self):
        return self.model.regularization_loss()

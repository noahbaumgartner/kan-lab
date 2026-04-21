from .base import BaseKANModel
from src.modules.efficientkan import KAN


class EfficientKANModel(BaseKANModel):
    def __init__(self, layers_hidden, grid_size=5, k=3, **kwargs):
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.k = k

    def build(self, device="cpu"):
        self.model = KAN(
            layers_hidden=list(self.layers_hidden),
            grid_size=self.grid_size,
            spline_order=self.k,
            grid_range=[-3, 3],
        ).to(device)
        self.device = device

    def regularization_loss(self):
        return self.model.regularization_loss()

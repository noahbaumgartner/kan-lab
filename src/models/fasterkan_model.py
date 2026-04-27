from .base import BaseKANModel
from src.modules.fasterkan import FasterKAN


class FasterKANModel(BaseKANModel):
    def __init__(
        self, layers_hidden, num_grids=8, exponent=2, inv_denominator=0.5, **kwargs
    ):
        self.layers_hidden = layers_hidden
        self.grid_min = -3
        self.grid_max = 3
        self.num_grids = num_grids
        self.exponent = exponent
        self.inv_denominator = inv_denominator

    def build(self, device="cpu"):
        self.model = FasterKAN(
            layers_hidden=list(self.layers_hidden),
            grid_min=self.grid_min,
            grid_max=self.grid_max,
            num_grids=self.num_grids,
            exponent=self.exponent,
            inv_denominator=self.inv_denominator,
        ).to(device)
        self.device = device

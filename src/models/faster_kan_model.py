import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules" / "fasterkan"))

from .base import BaseKANModel
from modules.fasterkan.fasterkan import FasterKAN


class FasterKANModel(BaseKANModel):
    def __init__(self, layers_hidden, num_grids=8, **kwargs):
        self.layers_hidden = layers_hidden
        self.grid_min = -1.2
        self.grid_max = 1.2
        self.num_grids = num_grids

    def build(self, device="cpu", grid_range=None):
        if grid_range is not None:
            self.grid_min = grid_range[0]
            self.grid_max = grid_range[1]
        self.model = FasterKAN(
            layers_hidden=list(self.layers_hidden),
            grid_min=self.grid_min,
            grid_max=self.grid_max,
            num_grids=self.num_grids,
        ).to(device)
        self.device = device

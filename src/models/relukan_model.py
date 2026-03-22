import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules" / "relukan"))

from .base import BaseKANModel
from modules.relukan.torch_relu_kan import ReLUKAN


class ReLUKANModel(BaseKANModel):
    def __init__(self, layers_hidden, grid=5, k=3, **kwargs):
        self.layers_hidden = layers_hidden
        self.grid = grid
        self.k = k

    def build(self, device="cpu"):
        self.model = ReLUKAN(
            width=list(self.layers_hidden),
            grid=self.grid,
            k=self.k,
        ).to(device)
        self.device = device

        # ReLUKANLayer outputs (batch, out, 1) but expects (batch, in) input.
        def _forward(self_model, x):
            x = x.unsqueeze(-1)
            for rk_layer in self_model.rk_layers:
                x = rk_layer(x)
            return x.squeeze(-1)

        self.model.forward = types.MethodType(_forward, self.model)

import torch.nn as nn

from .base import BaseKANModel
from modules.fastkan.fastkan import FastKAN, FastKANLayer


class FastKANModel(BaseKANModel):
    def __init__(self, layers_hidden, num_grids=8, **kwargs):
        self.layers_hidden = layers_hidden
        self.grid_min = -2.0
        self.grid_max = 2.0
        self.num_grids = num_grids

    def build(self, device="cpu"):
        layers_hidden = list(self.layers_hidden)
        needs_custom = any(d == 1 for d in layers_hidden[:-1])
        if needs_custom:
            model = FastKAN(layers_hidden=[2, 1], num_grids=self.num_grids)
            model.layers = nn.ModuleList([
                FastKANLayer(
                    in_dim, out_dim,
                    grid_min=self.grid_min,
                    grid_max=self.grid_max,
                    num_grids=self.num_grids,
                    use_layernorm=in_dim > 1,
                )
                for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
            ])
            self.model = model.to(device)
        else:
            self.model = FastKAN(
                layers_hidden=layers_hidden,
                grid_min=self.grid_min,
                grid_max=self.grid_max,
                num_grids=self.num_grids,
            ).to(device)
        self.device = device

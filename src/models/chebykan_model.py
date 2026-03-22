import torch.nn as nn

from .base import BaseKANModel
from modules.chebykan.ChebyKANLayer import ChebyKANLayer


class ChebyKANNetwork(nn.Module):
    def __init__(self, layers_hidden, degree=4):
        super().__init__()
        self.layers = nn.ModuleList([
            ChebyKANLayer(layers_hidden[i], layers_hidden[i + 1], degree)
            for i in range(len(layers_hidden) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChebyKANModel(BaseKANModel):
    def __init__(self, layers_hidden, degree=4, **kwargs):
        self.layers_hidden = layers_hidden
        self.degree = degree

    def build(self, device="cpu"):
        self.model = ChebyKANNetwork(
            layers_hidden=list(self.layers_hidden),
            degree=self.degree,
        ).to(device)
        self.device = device

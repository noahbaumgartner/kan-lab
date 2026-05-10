import torch
from torch import nn

from .base import BaseKANModel
from src.modules.convkan import KAN_Convolutional_Layer
from src.modules.convkan.kanlinear import KANLinear


class KKAN_Small(nn.Module):
    def __init__(self, grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=1,
            out_channels=5,
            kernel_size=(3, 3),
            grid_size=grid_size,
            padding=(0, 0),
        )

        self.conv2 = KAN_Convolutional_Layer(
            in_channels=5,
            out_channels=5,
            kernel_size=(3, 3),
            grid_size=grid_size,
            padding=(0, 0),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.kan1 = KANLinear(
            125,
            10,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
        )
        self.name = f"KKAN (Small) (gs = {grid_size})"

    def forward(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.kan1(x)

        return x


class KKANModel(BaseKANModel):
    def __init__(self, grid_size=5, **kwargs):
        self.grid_size = grid_size

    def build(self, device="cpu"):
        self.model = KKAN_Small(grid_size=self.grid_size).to(device)
        self.device = device

    def predict(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)

import torch

from .base import BaseKANModel
from src.modules.convkan import KANC_MLP


class KANCMLPModel(BaseKANModel):
    def __init__(
        self,
        num_classes,
        img_size=28,
        in_chans=1,
        conv1_out=5,
        conv2_out=10,
        kernel_size=3,
        hidden_dim=100,
        grid_size=5,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.img_size = img_size
        self.in_chans = in_chans
        self.conv1_out = conv1_out
        self.conv2_out = conv2_out
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

    def build(self, device="cpu"):
        self.model = KANC_MLP(
            in_chans=self.in_chans,
            img_size=self.img_size,
            num_classes=self.num_classes,
            conv1_out=self.conv1_out,
            conv2_out=self.conv2_out,
            kernel_size=self.kernel_size,
            hidden_dim=self.hidden_dim,
            grid_size=self.grid_size,
        ).to(device)
        self.device = device

    def predict(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, self.in_chans, self.img_size, self.img_size)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)

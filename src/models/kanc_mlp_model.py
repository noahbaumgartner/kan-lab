import torch

from .base import BaseKANModel
from src.modules.convkan import KANC_MLP


class KANCMLPModel(BaseKANModel):
    def __init__(self, num_classes, img_size=28, in_chans=1, grid_size=5, **kwargs):
        self.num_classes = num_classes
        self.img_size = img_size
        self.in_chans = in_chans
        self.grid_size = grid_size

    def build(self, device="cpu"):
        self.model = KANC_MLP(
            grid_size=self.grid_size,
            img_size=self.img_size,
            in_chans=self.in_chans,
            num_classes=self.num_classes,
        ).to(device)
        self.device = device

    def predict(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, self.in_chans, self.img_size, self.img_size)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)

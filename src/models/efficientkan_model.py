import torch.nn as nn

from .base import BaseKANModel
from src.modules.efficientkan import KAN


class _FlattenWrapper(nn.Module):
    """Flatten everything past the batch dim so a KAN MLP can consume images.

    Optionally average-pools first so a 1424x176 weak-lensing map doesn't
    explode the input-layer width.
    """

    def __init__(self, kan: nn.Module, pool_stride: int = 1):
        super().__init__()
        self.pool = (
            nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)
            if pool_stride > 1
            else nn.Identity()
        )
        self.flatten = nn.Flatten()
        self.kan = kan

    def forward(self, x):
        if x.dim() == 4:  # (B, C, H, W) -> pool then flatten
            x = self.pool(x)
        x = self.flatten(x)
        return self.kan(x)

    # KAN delegates ----------------------------------------------------------
    def regularization_loss(self, *args, **kwargs):
        return self.kan.regularization_loss(*args, **kwargs)


class EfficientKANModel(BaseKANModel):
    def __init__(self, layers_hidden, grid_size=5, k=3, pool_stride=1, **kwargs):
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.k = k
        # For 2D image inputs (e.g. weak_lensing) you'll usually want a
        # large pool_stride so the KAN's first layer width matches
        # layers_hidden[0]. Defaults to 1 (no-op) for tabular/1D inputs.
        self.pool_stride = pool_stride

    def build(self, device="cpu"):
        kan = KAN(
            layers_hidden=list(self.layers_hidden),
            grid_size=self.grid_size,
            spline_order=self.k,
            grid_range=[-3, 3],
        )
        self.model = _FlattenWrapper(kan, pool_stride=self.pool_stride).to(device)
        self.device = device

    def regularization_loss(self):
        return self.model.regularization_loss()

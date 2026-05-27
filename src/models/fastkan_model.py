import torch.nn as nn

from .base import BaseKANModel
from src.modules.fastkan import FastKAN, FastKANLayer


class _FlattenWrapper(nn.Module):
    """Avg-pool (optional) + flatten so an MLP-style KAN can consume images.

    Mirrors the wrapper in ``EfficientKANModel``: for 2D image inputs
    (e.g. weak-lensing maps) we avg-pool first so the KAN's first layer
    width matches ``layers_hidden[0]``.  For tabular / already-flat inputs
    the pool is a no-op and ``forward`` just passes through.
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


class FastKANModel(BaseKANModel):
    def __init__(self, layers_hidden, num_grids=8, pool_stride=1, **kwargs):
        self.layers_hidden = layers_hidden
        self.grid_min = -2.0
        self.grid_max = 2.0
        self.num_grids = num_grids
        self.pool_stride = pool_stride

    def build(self, device="cpu"):
        layers_hidden = list(self.layers_hidden)
        needs_custom = any(d == 1 for d in layers_hidden[:-1])
        if needs_custom:
            kan = FastKAN(layers_hidden=[2, 1], num_grids=self.num_grids)
            kan.layers = nn.ModuleList(
                [
                    FastKANLayer(
                        in_dim,
                        out_dim,
                        grid_min=self.grid_min,
                        grid_max=self.grid_max,
                        num_grids=self.num_grids,
                        use_layernorm=in_dim > 1,
                    )
                    for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
                ]
            )
        else:
            kan = FastKAN(
                layers_hidden=layers_hidden,
                grid_min=self.grid_min,
                grid_max=self.grid_max,
                num_grids=self.num_grids,
            )
        self.model = _FlattenWrapper(kan, pool_stride=self.pool_stride).to(device)
        self.device = device

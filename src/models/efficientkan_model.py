from .base import BaseKANModel
from src.modules.efficientkan import KAN
from src.modules.reduction import ReductionWrapper


class EfficientKANModel(BaseKANModel):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        k=3,
        reduction="none",
        pool_stride=1,
        scattering_j=3,
        scattering_l=8,
        scattering_order=2,
        img_height=0,
        img_width=0,
        in_chans=1,
        **kwargs,
    ):
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.k = k
        # Image -> vector reduction applied to 2D inputs (e.g. weak_lensing)
        # before flattening. Defaults to a no-op passthrough for tabular/1D.
        self.reduction = dict(
            method=reduction,
            in_chans=in_chans,
            img_height=img_height,
            img_width=img_width,
            pool_stride=pool_stride,
            scattering_j=scattering_j,
            scattering_l=scattering_l,
            scattering_order=scattering_order,
        )

    def build(self, device="cpu"):
        kan = KAN(
            layers_hidden=list(self.layers_hidden),
            grid_size=self.grid_size,
            spline_order=self.k,
            grid_range=[-3, 3],
        )
        self.model = ReductionWrapper(kan, **self.reduction).to(device)
        self.device = device

    def regularization_loss(self):
        return self.model.regularization_loss()
